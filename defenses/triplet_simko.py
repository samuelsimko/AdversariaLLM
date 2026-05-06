#!/usr/bin/env python3
"""Triplet defense (Simko) with optional JEPA / PRA bolt-on.

Implements the two-hinge representation-space triplet defense:

    L_safe   = ReLU( d(h_safe,   h_safe_base)   - d(h_safe,   unsafe_centroid) + m_b ).mean()
    L_unsafe = ReLU( d(h_unsafe, unsafe_centroid) - d(h_unsafe, h_unsafe_base)   + m_h ).mean()
    L_kl     = KL( p_def_benign || p_base_benign )
    Loss     = alpha * L_safe + beta * L_unsafe + gamma * L_kl

with hybrid distance d(x,y) = ||x-y||_2 + 10 * ReLU(1 - cos(x_norm, y_norm)).

Optional PRA: when --w_jepa > 0, also compute a JEPA-style alignment loss
between the defended encoder's representation of the (vanilla, adversarial)
sibling pair from a wildjailbreak-format pair file. The predictor projects
the adversarial rep into the vanilla manifold; loss is MSE.

Output contract (matches experiments/run_experiment.py expectations):
    <output_dir>/
        lora_adapter/      (saved LoRA)
        jepa_predictor.pt  (only when w_jepa > 0)
        manifest.json
        hparams.json
        READY              (touched on success)
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Defaults — match user's reference unless otherwise needed
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_MAX_STEPS = 1500

DEFAULT_ALPHA = 0.5    # safe triplet weight
DEFAULT_BETA = 0.4     # unsafe triplet weight
DEFAULT_GAMMA = 0.9    # KL retain weight

DEFAULT_MB = 500.0     # safe-side margin
DEFAULT_MH = 1500.0    # unsafe-side margin

DEFAULT_REP_LAYERS = list(range(20, 31))  # 11 layers, matches reference

DEFAULT_W_JEPA = 0.0
DEFAULT_ALIGN_LAYER = 25
DEFAULT_PREDICTOR_LAYERS = 2
DEFAULT_PREDICTOR_BOTTLENECK_DIM = 512
DEFAULT_PREDICTOR_LR_MULTIPLIER = 5.0


# ============================================================
# Tokenization
# ============================================================

def tokenize_chat(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
):
    if len(prompts) != len(responses):
        raise ValueError("prompts/responses length mismatch")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=True, add_generation_prompt=False,
            max_length=max_length, truncation=True, return_tensors="pt",
        )
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True,
            max_length=max_length, truncation=True, return_tensors="pt",
        )
        labels = full.clone()
        labels[0, :prompt_only.shape[-1]] = -100

        seq_len = full.shape[-1]
        pad = max_length - seq_len
        input_ids = F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(0)
        labels = F.pad(labels, (0, pad), value=-100).squeeze(0)
        attn = F.pad(torch.ones_like(full), (0, pad), value=0).squeeze(0)
        out.append({"input_ids": input_ids, "attention_mask": attn, "labels": labels})
    return out


def tokenize_prompt_only(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
):
    """For JEPA pair encoding: just the user prompt, generation prompt at end."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True,
            max_length=max_length, truncation=True, return_tensors="pt",
        )
        seq_len = ids.shape[-1]
        pad = max_length - seq_len
        input_ids = F.pad(ids, (0, pad), value=tokenizer.pad_token_id).squeeze(0)
        attn = F.pad(torch.ones_like(ids), (0, pad), value=0).squeeze(0)
        out.append({"input_ids": input_ids, "attention_mask": attn})
    return out


class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


# ============================================================
# Data loading
# ============================================================

def load_ultrachat(num_samples: int, seed: int = 42):
    ds = (load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
          .shuffle(seed=seed).select(range(num_samples)))
    prompts, responses = [], []
    for item in ds:
        msgs = item["messages"]
        if not msgs:
            continue
        u = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompts.append(u); responses.append(a)
    return prompts, responses


def _read_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192); f.seek(0)
        first = next((c for c in head if not c.isspace()), None)
        if first == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
        return [json.loads(s) for s in f if s.strip() and not s.lstrip().startswith(("#", "//"))]


def load_harmful(path: str, limit: Optional[int] = None):
    rec = _read_json_or_jsonl(path)
    if limit is not None:
        rec = rec[:limit]
    prompts = [r.get("prompt", "") for r in rec]
    outputs = [r.get("output", "") for r in rec]
    return prompts, outputs


def load_wj_pairs(path: str, limit: Optional[int] = None):
    """WildJailbreak pair format. Returns (vanillas, adversarials) lists."""
    rec = _read_json_or_jsonl(path)
    if limit is not None:
        rec = rec[:limit]
    vanillas = [r.get("vanilla", "") for r in rec]
    adversarials = [r.get("adversarial", "") for r in rec]
    return vanillas, adversarials


# ============================================================
# Loss helpers
# ============================================================

@contextmanager
def adapters_disabled(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def get_hidden_states(model, batch, layers):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                output_hidden_states=True, use_cache=False)
    # hidden_states[0] is embeddings → +1
    return torch.stack([out.hidden_states[l + 1] for l in layers])


def masked_mean(h, mask):
    """h: [L, B, T, D], mask: [B, T] → [L, D]"""
    mask = mask.unsqueeze(0).unsqueeze(-1).float()
    num = (h * mask).sum(dim=(1, 2))
    den = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return num / den


def d_mix(x, y):
    """Hybrid distance: Euclidean + 10 * clipped cosine distance.
    Per the reference: large margins (500/1500) only make sense with this."""
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    return (torch.norm(x - y, dim=-1)
            + 10.0 * (1.0 - F.cosine_similarity(xn, yn, dim=-1).relu()))


# ============================================================
# JEPA predictor (PRA)
# ============================================================

class Predictor(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, bottleneck_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("predictor_layers must be >= 2")
        layers = [nn.Linear(dim, bottleneck_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers += [nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(bottleneck_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Pair-batch loader
# ============================================================

class TripleBatchLoader:
    """Pairs (benign, harmful, jepa_pair) into one step. JEPA pair optional."""
    def __init__(self, benign_loader, harmful_loader, jepa_loader=None):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self.jepa_loader = jepa_loader
        if jepa_loader is None:
            self._len = min(len(benign_loader), len(harmful_loader))
        else:
            self._len = min(len(benign_loader), len(harmful_loader), len(jepa_loader))

    def __iter__(self):
        if self.jepa_loader is None:
            for b, h in zip(self.benign_loader, self.harmful_loader):
                yield (b, h, None)
        else:
            for b, h, j in zip(self.benign_loader, self.harmful_loader, self.jepa_loader):
                yield (b, h, j)

    def __len__(self):
        return self._len


# ============================================================
# Trainer
# ============================================================

class TripletTrainer(Trainer):
    def __init__(self, *, benign_ds, harmful_ds, jepa_ds, predictor,
                 rep_layers, align_layer, alpha, beta, gamma, mb, mh,
                 w_jepa, predictor_lr_multiplier, **kwargs):
        super().__init__(**kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.jepa_ds = jepa_ds
        self.predictor = predictor
        self.rep_layers = rep_layers
        self.align_layer = align_layer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mb = mb
        self.mh = mh
        self.w_jepa = w_jepa
        self.predictor_lr_multiplier = predictor_lr_multiplier

    def get_train_dataloader(self):
        bs = self.args.per_device_train_batch_size
        bl = DataLoader(self.benign_ds, batch_size=bs, shuffle=True)
        hl = DataLoader(self.harmful_ds, batch_size=bs, shuffle=True)
        jl = None
        if self.jepa_ds is not None:
            jl = DataLoader(self.jepa_ds, batch_size=bs, shuffle=True)
        return TripleBatchLoader(bl, hl, jl)

    def create_optimizer(self):
        # Standard optimizer over LoRA params + predictor (with LR multiplier).
        if self.optimizer is not None:
            return self.optimizer
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [{"params": lora_params, "lr": self.args.learning_rate}]
        if self.predictor is not None:
            param_groups.append({
                "params": list(self.predictor.parameters()),
                "lr": self.args.learning_rate * self.predictor_lr_multiplier,
            })
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.args.learning_rate)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign, harmful, jepa = inputs
        device = model.device
        benign  = {k: v.to(device) for k, v in benign.items()}
        harmful = {k: v.to(device) for k, v in harmful.items()}

        # ---- Hidden states with adapters
        h_safe   = get_hidden_states(model, benign,  self.rep_layers)
        h_unsafe = get_hidden_states(model, harmful, self.rep_layers)
        # ---- Hidden states without adapters (base)
        with adapters_disabled(model), torch.no_grad():
            h_safe_base   = get_hidden_states(model, benign,  self.rep_layers)
            h_unsafe_base = get_hidden_states(model, harmful, self.rep_layers)

        mask_safe   = benign["labels"]  != -100
        mask_unsafe = harmful["labels"] != -100

        # Harmful centroid (broadcast back to [L, B, T, D])
        unsafe_centroid = masked_mean(h_unsafe, mask_unsafe).unsqueeze(1).unsqueeze(2)

        # ---- Safe triplet: keep benign close to base benign, far from harmful centroid
        d_pos_safe = d_mix(h_safe, h_safe_base)
        d_neg_safe = d_mix(h_safe, unsafe_centroid)
        l_safe = (F.relu(d_pos_safe - d_neg_safe + self.mb) * mask_safe.unsqueeze(0)).mean()

        # ---- Unsafe triplet: keep harmful close to harmful centroid, far from base harmful
        d_pos_unsafe = d_mix(h_unsafe, unsafe_centroid)
        d_neg_unsafe = d_mix(h_unsafe, h_unsafe_base)
        l_unsafe = (F.relu(d_pos_unsafe - d_neg_unsafe + self.mh) * mask_unsafe.unsqueeze(0)).mean()

        # ---- KL retain on benign (defended logits → base logits)
        out_safe = model(**benign, use_cache=False)
        with adapters_disabled(model), torch.no_grad():
            base_logits = model(**benign, use_cache=False).logits
        kl = F.kl_div(F.log_softmax(out_safe.logits, dim=-1),
                      F.softmax(base_logits, dim=-1),
                      reduction="batchmean")

        loss = self.alpha * l_safe + self.beta * l_unsafe + self.gamma * kl
        log = {
            "loss/safe_triplet": l_safe.detach().item(),
            "loss/unsafe_triplet": l_unsafe.detach().item(),
            "loss/kl_retain": kl.detach().item(),
        }

        # ---- Optional JEPA / PRA term
        if self.w_jepa > 0 and jepa is not None and self.predictor is not None:
            van = {k: v.to(device) for k, v in jepa["vanilla"].items()}
            adv = {k: v.to(device) for k, v in jepa["adversarial"].items()}
            van_out = model(input_ids=van["input_ids"], attention_mask=van["attention_mask"],
                            output_hidden_states=True, use_cache=False)
            adv_out = model(input_ids=adv["input_ids"], attention_mask=adv["attention_mask"],
                            output_hidden_states=True, use_cache=False)
            van_h = van_out.hidden_states[self.align_layer + 1]   # [B, T, D]
            adv_h = adv_out.hidden_states[self.align_layer + 1]   # [B, T, D]
            van_pooled = (van_h * van["attention_mask"].unsqueeze(-1).float()).sum(1) / van["attention_mask"].sum(1, keepdim=True).clamp_min(1).float()
            adv_pooled = (adv_h * adv["attention_mask"].unsqueeze(-1).float()).sum(1) / adv["attention_mask"].sum(1, keepdim=True).clamp_min(1).float()
            # Stop-gradient on target (vanilla rep), like standard JEPA.
            jepa_pred = self.predictor(adv_pooled.float())
            jepa_loss = F.mse_loss(jepa_pred, van_pooled.detach().float())
            loss = loss + self.w_jepa * jepa_loss
            log["loss/jepa"] = jepa_loss.detach().item()

        log["loss/total"] = loss.detach().item()
        self.log(log)
        return (loss, None) if return_outputs else loss


# ============================================================
# Pair dataset for JEPA
# ============================================================

class JEPADataset(Dataset):
    def __init__(self, vanillas, adversarials, tokenizer, max_length):
        self.van = tokenize_prompt_only(vanillas, tokenizer, max_length=max_length)
        self.adv = tokenize_prompt_only(adversarials, tokenizer, max_length=max_length)
        if len(self.van) != len(self.adv):
            raise ValueError("vanilla/adversarial length mismatch")
    def __len__(self): return len(self.van)
    def __getitem__(self, idx):
        return {"vanilla": self.van[idx], "adversarial": self.adv[idx]}


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output_dir", type=str, required=True)

    # Required harmful-CE corpus (interface match with jepa_ce.py: --cb_path)
    p.add_argument("--cb_path", type=str, required=True,
                   help="JSON list or JSONL with {prompt, output} harmful pairs.")
    p.add_argument("--limit_cb", type=int, default=5000)

    # Optional JEPA pair corpus
    p.add_argument("--pair_path", type=str, default=None,
                   help="WildJailbreak pair file with {vanilla, adversarial}. Required if w_jepa>0.")
    p.add_argument("--pair_format", type=str, default="wildjailbreak",
                   help="Only 'wildjailbreak' supported here.")
    p.add_argument("--pair_sample_size", type=int, default=8000)

    # Benign LM data (UltraChat)
    p.add_argument("--ultrachat_samples", type=int, default=5000)
    p.add_argument("--extra_benign_path", type=str, default=None,
                   help="Optional extra benign JSONL with {prompt, response}.")
    p.add_argument("--extra_benign_samples", type=int, default=5000)

    # Training
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--num_max_steps", type=int, default=DEFAULT_MAX_STEPS)

    # Loss weights
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="safe triplet weight")
    p.add_argument("--beta",  type=float, default=DEFAULT_BETA,  help="unsafe triplet weight")
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="KL retain weight")
    p.add_argument("--mb",    type=float, default=DEFAULT_MB)
    p.add_argument("--mh",    type=float, default=DEFAULT_MH)
    p.add_argument("--rep_layers", type=int, nargs="*", default=DEFAULT_REP_LAYERS)

    # JEPA / PRA
    p.add_argument("--w_jepa", type=float, default=DEFAULT_W_JEPA)
    p.add_argument("--align_layer", type=int, default=DEFAULT_ALIGN_LAYER)
    p.add_argument("--predictor_layers", type=int, default=DEFAULT_PREDICTOR_LAYERS)
    p.add_argument("--predictor_bottleneck_dim", type=int, default=DEFAULT_PREDICTOR_BOTTLENECK_DIM)
    p.add_argument("--predictor_lr_multiplier", type=float, default=DEFAULT_PREDICTOR_LR_MULTIPLIER)

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # Trainer logistics
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to", type=str, default="none")

    args = p.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Data: benign (UltraChat + optional extra benign)
    benign_prompts, benign_responses = load_ultrachat(args.ultrachat_samples)
    if args.extra_benign_path and os.path.exists(args.extra_benign_path):
        eb = _read_json_or_jsonl(args.extra_benign_path)[: args.extra_benign_samples]
        for r in eb:
            benign_prompts.append(r.get("prompt", ""))
            benign_responses.append(r.get("response", "") or r.get("output", ""))

    # ----- Data: harmful
    harm_prompts, harm_outputs = load_harmful(args.cb_path, args.limit_cb)

    benign_data  = tokenize_chat(benign_prompts,  benign_responses, tokenizer, args.max_length)
    harmful_data = tokenize_chat(harm_prompts,    harm_outputs,     tokenizer, args.max_length)

    jepa_data = None
    if args.w_jepa > 0:
        if not args.pair_path or not os.path.exists(args.pair_path):
            raise ValueError(f"w_jepa>0 requires --pair_path; got {args.pair_path}")
        van, adv = load_wj_pairs(args.pair_path, args.pair_sample_size)
        jepa_data = JEPADataset(van, adv, tokenizer, max_length=args.max_length)
        print(f"[triplet_simko] JEPA pairs: {len(jepa_data)}")

    # ----- Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # ----- Predictor (only when PRA on). Keep weights in float32 — MSE
    # alignment is sensitive to small differences and bf16 underflows.
    # Inputs are cast to float at the call site (see TripletTrainer.compute_loss).
    predictor = None
    if args.w_jepa > 0:
        hidden_dim = model.config.hidden_size if hasattr(model, "config") else model.get_input_embeddings().weight.shape[-1]
        predictor = Predictor(
            dim=hidden_dim,
            num_layers=args.predictor_layers,
            bottleneck_dim=args.predictor_bottleneck_dim,
        ).to(model.device)  # float32 by default

    # ----- Trainer
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.num_max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=1.0,
        report_to=args.report_to,
        run_name=f"triplet_simko_{Path(out_dir).name}",
    )

    trainer = TripletTrainer(
        model=model, args=targs, tokenizer=tokenizer,
        benign_ds=ListDataset(benign_data),
        harmful_ds=ListDataset(harmful_data),
        jepa_ds=jepa_data,
        predictor=predictor,
        rep_layers=args.rep_layers,
        align_layer=args.align_layer,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        mb=args.mb, mh=args.mh,
        w_jepa=args.w_jepa,
        predictor_lr_multiplier=args.predictor_lr_multiplier,
    )

    trainer.train()

    # ----- Save outputs
    model.save_pretrained(out_dir / "lora_adapter")
    if predictor is not None:
        torch.save(predictor.state_dict(), out_dir / "jepa_predictor.pt")

    with open(out_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    manifest = {
        "schema_version": "1.0",
        "defense_name": "triplet_simko",
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "training_completed": True,
        "rep_layers": args.rep_layers,
        "align_layer": args.align_layer,
        "w_jepa": args.w_jepa,
        "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma,
        "mb": args.mb, "mh": args.mh,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    (out_dir / "READY").touch()


if __name__ == "__main__":
    main()
