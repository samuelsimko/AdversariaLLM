"""Quick benchmark: sequential vs. batched soft-prompt optimization.

Picks N prompts from adv_behaviors, runs the existing
adversariallm.attacks.soft_prompt.run_soft_opt sequentially (the current
behaviour) and a hand-batched version that optimises N soft prompts in one
forward/backward. Reports wall time, peak GPU memory, and final loss per
prompt for both. No mutation of repo code; this is a probe before deciding
whether to integrate.

Usage:
    PYTHONPATH=. /workspace/AdversariaLLM/venv/bin/python \\
      scripts/benchmark_soft_prompt_batching.py \\
      --model Qwen/Qwen3-8B --adapter <optional adapter path> \\
      --n 8 --num-steps 100 --num-tokens 5 --lr 0.01
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from adversariallm.attacks.soft_prompt import SoftPromptConfig, run_soft_opt
from adversariallm.dataset import AdvBehaviorsConfig, AdvBehaviorsDataset


# ---------------------------------------------------------------------- helpers

def pick_prompts(n: int) -> list[tuple[list[dict], str]]:
    """Return [(messages_without_assistant, target_str), ...] of length n."""
    cfg = AdvBehaviorsConfig(seed=0, idx=list(range(n)), shuffle=False)
    ds = AdvBehaviorsDataset(cfg)
    out = []
    for conv in ds:
        if conv[-1]["role"] != "assistant":
            raise RuntimeError("expected assistant target turn")
        msgs = copy.deepcopy(conv[:-1])
        target = conv[-1]["content"]
        out.append((msgs, target))
    return out


def reset_mem(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_mem_gb(device) -> float:
    return torch.cuda.max_memory_allocated(device) / (1024 ** 3)


# ----------------------------------------------------------------- batched impl

def _ensure_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return copy.deepcopy(messages)


def run_soft_opt_batched(
    model,
    tokenizer,
    items: list[tuple[list[dict], str]],
    num_steps: int,
    num_tokens: int,
    lr: float,
    seed: int = 0,
    early_stop_loss: float = 0.0,
    verbose: bool = False,
):
    """Optimize len(items) soft prompts in parallel.

    Layout per row (left-padded to common length): [PAD..., before, optim, after, target].
    Each row has its own [num_tokens, hidden] optim slot. They are concatenated
    into a single [B, T, H] embedding tensor for one forward/backward per step.
    """
    device = model.device
    model.enable_input_require_grads()
    torch.manual_seed(seed)

    emb = model.get_input_embeddings()
    dtype = emb.weight.dtype
    H = model.config.hidden_size
    B = len(items)

    # Tokenize before / after / target per row.
    before_ids_list, after_ids_list, target_ids_list = [], [], []
    for msgs, target in items:
        m = _ensure_messages(msgs)
        if not any("{optim_str}" in d["content"] for d in m):
            m[-1]["content"] = m[-1]["content"] + "{optim_str}"
        template = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "", 1)
        if "{optim_str}" not in template:
            raise ValueError("chat template stripped {optim_str}")
        before_str, after_str = template.split("{optim_str}", 1)
        before_ids = tokenizer(before_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        after_ids = tokenizer(after_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if target_ids.numel() == 0:
            raise ValueError("empty target")
        before_ids_list.append(before_ids.to(device))
        after_ids_list.append(after_ids.to(device))
        target_ids_list.append(target_ids.to(device))

    before_lens = torch.tensor([t.numel() for t in before_ids_list], device=device)
    after_lens = torch.tensor([t.numel() for t in after_ids_list], device=device)
    target_lens = torch.tensor([t.numel() for t in target_ids_list], device=device)
    seq_lens = before_lens + num_tokens + after_lens + target_lens
    T = int(seq_lens.max().item())

    # Initialize optim embeddings: random Gaussian, one set per row.
    optim_embeds = torch.randn(B, num_tokens, H, device=device, dtype=dtype, requires_grad=True)

    # Precompute (and detach) the fixed embedding pieces per row.
    before_embs = [emb(t).detach().to(dtype) for t in before_ids_list]
    after_embs = [emb(t).detach().to(dtype) for t in after_ids_list]
    target_embs = [emb(t).detach().to(dtype) for t in target_ids_list]

    optim = torch.optim.Adam([optim_embeds], lr=lr)
    losses_per_step = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_emb = emb(torch.tensor([pad_id], device=device)).detach().to(dtype).squeeze(0)  # [H]

    final_losses = [None] * B
    done_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for step in range(num_steps):
        optim.zero_grad(set_to_none=True)

        # Build [B, T, H] inputs_embeds with LEFT padding so target sits at right edge.
        rows_emb = []
        attn_rows = []
        target_pos_rows = []  # tensor of shape [target_lens[i]] giving positions in the row
        for i in range(B):
            before_e = before_embs[i]
            after_e = after_embs[i]
            target_e = target_embs[i]
            optim_e = optim_embeds[i]  # [num_tokens, H], differentiable
            content = torch.cat([before_e, optim_e, after_e, target_e], dim=0)  # [L, H]
            L = content.shape[0]
            pad_n = T - L
            pad_block = pad_emb.unsqueeze(0).expand(pad_n, H) if pad_n > 0 else None
            if pad_block is not None:
                row = torch.cat([pad_block, content], dim=0)  # [T, H]
            else:
                row = content
            rows_emb.append(row.unsqueeze(0))
            mask = torch.zeros(T, dtype=torch.long, device=device)
            mask[pad_n:] = 1
            attn_rows.append(mask.unsqueeze(0))
            # Target positions in [0, T): last target_lens[i] positions.
            tlen = int(target_lens[i].item())
            target_pos_rows.append(torch.arange(T - tlen, T, device=device))

        inputs_embeds = torch.cat(rows_emb, dim=0)  # [B, T, H]
        attention_mask = torch.cat(attn_rows, dim=0)  # [B, T]

        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits  # [B, T, V]

        # Per-row CE: predict target_ids[i] from logits at positions target_pos_rows[i] - 1.
        per_row_losses = []
        for i in range(B):
            pos = target_pos_rows[i]
            shift_logits = logits[i, pos - 1, :]
            ce = F.cross_entropy(shift_logits, target_ids_list[i])
            per_row_losses.append(ce)
        per_row_t = torch.stack(per_row_losses)  # [B]
        # Only count rows that haven't early-stopped yet.
        active = (~done_mask).float()
        loss = (per_row_t * active).sum() / max(active.sum().item(), 1.0)

        for i in range(B):
            li = per_row_losses[i].detach().item()
            if not bool(done_mask[i].item()):
                final_losses[i] = li
                if early_stop_loss and li < early_stop_loss:
                    done_mask[i] = True
        losses_per_step.append([per_row_losses[i].detach().item() for i in range(B)])
        if verbose:
            print(f"  step={step:3d}  losses=" + " ".join(f"{l:.3f}" for l in losses_per_step[-1]))

        if bool(done_mask.all().item()):
            break

        loss.backward()
        optim.step()

    return {
        "final_losses": final_losses,
        "num_steps_run": step + 1,
        "losses_per_step": losses_per_step,
    }


# --------------------------------------------------------------------- runner

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--num-tokens", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out", default="runs/soft_prompt_batching_bench.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    if args.adapter:
        print(f"[load] adapter {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    items = pick_prompts(args.n)
    print(f"[data] {len(items)} prompts")
    for i, (m, t) in enumerate(items):
        print(f"  [{i}] target={t[:60]!r}  prompt={m[-1]['content'][:60]!r}")

    # ---- sequential ----
    print(f"\n[seq] running {args.n} prompts × {args.num_steps} steps")
    reset_mem(device)
    seq_t0 = time.time()
    seq_losses = []
    for i, (msgs, target) in enumerate(items):
        cfg = SoftPromptConfig(
            seed=0, num_steps=args.num_steps, num_tokens=args.num_tokens,
            lr=args.lr, rand_init=True, early_stop_loss=0.0,  # disable early-stop for clean timing
        )
        res = run_soft_opt(model, tok, msgs, target, cfg, device=device)
        seq_losses.append(res.losses[-1] if res.losses else None)
    seq_dt = time.time() - seq_t0
    seq_peak = peak_mem_gb(device)
    print(f"[seq] total {seq_dt:.1f}s  per-prompt {seq_dt/args.n:.2f}s  peak {seq_peak:.2f} GB")

    # ---- batched ----
    print(f"\n[batch] running {args.n} prompts in parallel × {args.num_steps} steps")
    reset_mem(device)
    bat_t0 = time.time()
    bat = run_soft_opt_batched(
        model, tok, items,
        num_steps=args.num_steps, num_tokens=args.num_tokens,
        lr=args.lr, seed=0, early_stop_loss=0.0,
    )
    bat_dt = time.time() - bat_t0
    bat_peak = peak_mem_gb(device)
    print(f"[batch] total {bat_dt:.1f}s  per-prompt {bat_dt/args.n:.2f}s  peak {bat_peak:.2f} GB")

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n": args.n,
        "num_steps": args.num_steps,
        "num_tokens": args.num_tokens,
        "lr": args.lr,
        "sequential": {"total_s": seq_dt, "per_prompt_s": seq_dt / args.n,
                       "peak_gb": seq_peak, "final_losses": seq_losses},
        "batched": {"total_s": bat_dt, "per_prompt_s": bat_dt / args.n,
                    "peak_gb": bat_peak, "final_losses": bat["final_losses"],
                    "steps_run": bat["num_steps_run"]},
        "speedup": seq_dt / max(bat_dt, 1e-6),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[summary] speedup = {summary['speedup']:.2f}x")
    print(f"[summary] memory   seq={seq_peak:.1f} GB  batched={bat_peak:.1f} GB")
    print(f"[summary] losses   seq    last = {seq_losses}")
    print(f"[summary] losses   batch  last = {bat['final_losses']}")
    print(f"[summary] wrote {args.out}")


if __name__ == "__main__":
    main()
