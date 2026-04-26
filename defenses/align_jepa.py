import os
import json
import csv
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
import matplotlib.pyplot as plt


# ============================================================
# LLM-JEPA-style alignment for jailbreak prompts -> behavior text
#
# Key idea:
#   Pred(Enc(j)) is implemented with the SAME LM by appending one or
#   more special [PRED] tokens to the jailbreak prompt and taking the
#   final hidden state of the last [PRED] token, following the LLM-JEPA
#   tied-weights predictor idea.
#
#   Enc(b) is the last-token hidden state of the ground-truth behavior
#   text, not the full harmful response. This makes the objective focus
#   on intent rather than one specific realization of the harmful output.
# ============================================================


# -----------------------------
# 1. Defaults
# -----------------------------
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_ALIGN_LIMIT = 5000
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_ALIGN = 1.0
DEFAULT_W_BENIGN_KL = 0.0
DEFAULT_NUM_PRED_TOKENS = 1
DEFAULT_ALIGN_LAYER = -1  # final hidden layer by default
DEFAULT_PRED_TOKEN = "[PRED]"


# -----------------------------
# 2. JSON helpers
# -----------------------------
def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192)
        f.seek(0)
        first_non_ws = next((ch for ch in head if not ch.isspace()), None)

        def parse_jsonl(file_obj):
            out = []
            for line in file_obj:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                try:
                    out.append(json.loads(s))
                except json.JSONDecodeError:
                    continue
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
                return parse_jsonl(f)
        return parse_jsonl(f)


# -----------------------------
# 3. Data loading
# -----------------------------
def load_ultrachat(num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    ds = (
        load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        .shuffle(seed=seed)
        .select(range(num_samples))
    )
    prompts, responses = [], []
    for item in ds:
        messages = item["messages"]
        if not messages:
            continue
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if user and assistant:
            prompts.append(user)
            responses.append(assistant)
    return prompts, responses


def load_reverse_alignment_pairs(path: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
    """
    Creates (jailbreak_prompt, behavior_text) pairs.

    Expected reverse-model-style data:
      - record.prompt or true_prompt = original harmful request / behavior text
      - generated_prompts = jailbreak prompts that try to request the same thing

    We use the ground-truth behavior text as the JEPA target view.
    """
    records = _load_json_or_jsonl(path)
    if limit is not None:
        records = records[:limit]

    prompts: List[str] = []
    behaviors: List[str] = []
    source_labels: List[str] = []

    for rec in records:
        base = rec.get("record", rec)
        behavior_text = rec.get("true_prompt") or base.get("prompt") or ""
        generated_prompts = rec.get("generated_prompts") or []

        if not behavior_text:
            continue

        for gp in generated_prompts:
            if isinstance(gp, str) and gp.strip():
                prompts.append(gp.strip())
                behaviors.append(behavior_text.strip())
                source_labels.append("generated_prompt")

    return prompts, behaviors, source_labels


# -----------------------------
# 4. Tokenization helpers
# -----------------------------
def _extract_input_ids(template_output: Any) -> torch.Tensor:
    if isinstance(template_output, torch.Tensor):
        return template_output
    if hasattr(template_output, "input_ids"):
        return template_output.input_ids
    if isinstance(template_output, dict) and "input_ids" in template_output:
        return template_output["input_ids"]
    raise TypeError(f"Unsupported chat template output type: {type(template_output)!r}")


def tokenize_chat_generic(
    prompts: List[str],
    responses: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
) -> List[Dict[str, torch.Tensor]]:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        full_tokens = _extract_input_ids(full_tokens)

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_only = _extract_input_ids(prompt_only)

        labels = full_tokens.clone()
        prompt_len = min(prompt_only.shape[-1], full_tokens.shape[-1])
        labels[0, :prompt_len] = -100

        L = min(full_tokens.shape[-1], max_length)
        full_tokens = full_tokens[:, :L]
        labels = labels[:, :L]
        pad_len = max_length - L

        input_ids = F.pad(full_tokens, (0, pad_len), value=tokenizer.pad_token_id).squeeze(0)
        labels = F.pad(labels, (0, pad_len), value=-100).squeeze(0)
        attn = F.pad(torch.ones_like(full_tokens), (0, pad_len), value=0).squeeze(0)

        out.append(
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "labels": labels,
            }
        )
    return out


def build_pred_suffix(tokenizer: PreTrainedTokenizer, num_pred_tokens: int) -> torch.Tensor:
    pred_id = tokenizer.convert_tokens_to_ids(DEFAULT_PRED_TOKEN)
    if pred_id is None or pred_id == tokenizer.unk_token_id:
        raise ValueError(f"Predictor token {DEFAULT_PRED_TOKEN!r} was not added correctly.")
    return torch.tensor([pred_id] * num_pred_tokens, dtype=torch.long).unsqueeze(0)


def tokenize_alignment_pairs(
    jailbreak_prompts: List[str],
    behaviors: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    num_pred_tokens: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    For each pair (j, b):
      - prompt branch: user jailbreak prompt with appended [PRED] tokens
      - behavior branch: raw behavior text as a single-view input

    Pred(Enc(j)) := last hidden state of final [PRED] token.
    Enc(b)       := last hidden state of final token of behavior text.
    """
    if len(jailbreak_prompts) != len(behaviors):
        raise ValueError("jailbreak_prompts and behaviors must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pred_suffix = build_pred_suffix(tokenizer, num_pred_tokens)
    out: List[Dict[str, torch.Tensor]] = []

    for prompt, behavior in zip(jailbreak_prompts, behaviors):
        # Prompt-side input: normal user prompt with generation prompt, then append [PRED] tokens.
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_tokens = _extract_input_ids(prompt_tokens)
        prompt_with_pred = torch.cat([prompt_tokens, pred_suffix], dim=-1)
        prompt_with_pred = prompt_with_pred[:, :max_length]
        pred_last_idx = prompt_with_pred.shape[-1] - 1

        prompt_pad = max_length - prompt_with_pred.shape[-1]
        prompt_ids = F.pad(prompt_with_pred, (0, prompt_pad), value=tokenizer.pad_token_id).squeeze(0)
        prompt_mask = F.pad(torch.ones_like(prompt_with_pred), (0, prompt_pad), value=0).squeeze(0)

        # Behavior-side input: plain text behavior target.
        behavior_tok = tokenizer(
            behavior,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        behavior_ids_2d = behavior_tok["input_ids"][:, :max_length]
        behavior_last_idx = behavior_ids_2d.shape[-1] - 1
        behavior_pad = max_length - behavior_ids_2d.shape[-1]
        behavior_ids = F.pad(behavior_ids_2d, (0, behavior_pad), value=tokenizer.pad_token_id).squeeze(0)
        behavior_mask = F.pad(torch.ones_like(behavior_ids_2d), (0, behavior_pad), value=0).squeeze(0)

        out.append(
            {
                "prompt_input_ids": prompt_ids,
                "prompt_attention_mask": prompt_mask,
                "pred_last_idx": torch.tensor(pred_last_idx, dtype=torch.long),
                "behavior_input_ids": behavior_ids,
                "behavior_attention_mask": behavior_mask,
                "behavior_last_idx": torch.tensor(behavior_last_idx, dtype=torch.long),
            }
        )
    return out


class TensorDictDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# -----------------------------
# 5. Core losses / utils
# -----------------------------
def ce_loss_from_outputs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def topk_kl_from_logits(adapted_logits: torch.Tensor, base_logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    logp_adapted = F.log_softmax(adapted_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    topk_idx = torch.topk(logp_base, k=min(k, logp_base.size(-1)), dim=-1).indices
    adapted_selected = logp_adapted.gather(-1, topk_idx)
    base_selected = logp_base.gather(-1, topk_idx)
    base_probs = base_selected.exp()
    return (base_probs * (base_selected - adapted_selected)).sum(dim=-1).mean()


# -----------------------------
# 6. Logging callback
# -----------------------------
class LoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / "metrics.csv"
        self.all_logs: List[Dict[str, Any]] = []
        self.fieldnames = ["step"]

    def _write_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in self.all_logs:
                writer.writerow({k: row.get(k, "") for k in self.fieldnames})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step}
        row.update(logs)
        for k in row.keys():
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        self.all_logs.append(row)
        self._write_csv()

    def on_train_end(self, args, state, control, **kwargs):
        if not self.all_logs:
            return
        import pandas as pd
        df = pd.DataFrame(self.all_logs)
        plt.figure(figsize=(14, 8))
        for col in df.columns:
            if col != "step" and df[col].dtype != object:
                plt.plot(df["step"], df[col], label=col)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics.png")
        print(f"Saved metrics to {self.output_dir}/metrics.csv and metrics.png")


# -----------------------------
# 7. Dual loader + trainer
# -----------------------------
class DualBatchLoader:
    def __init__(self, benign_loader: DataLoader, align_loader: DataLoader):
        self.benign_loader = benign_loader
        self.align_loader = align_loader
        self._len = min(len(benign_loader), len(align_loader))

    def __iter__(self):
        return zip(self.benign_loader, self.align_loader)

    def __len__(self) -> int:
        return self._len


class AlignmentJEPATrainer(Trainer):
    def __init__(
        self,
        benign_ds: Dataset,
        align_ds: Dataset,
        align_layer: int,
        w_benign: float,
        w_align: float,
        w_benign_kl: float,
        align_metric: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.align_ds = align_ds
        self.align_layer = align_layer
        self.w_benign = w_benign
        self.w_align = w_align
        self.w_benign_kl = w_benign_kl
        self.align_metric = align_metric

    def get_train_dataloader(self):
        args = self.args
        benign_loader = DataLoader(
            self.benign_ds,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )
        align_loader = DataLoader(
            self.align_ds,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
        )
        return DualBatchLoader(benign_loader, align_loader)

    def _extract_last_token_reps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states[self.align_layer]
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_indices, :]

    def _alignment_loss(self, model, align_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = model.device
        prompt_ids = align_batch["prompt_input_ids"].to(device)
        prompt_mask = align_batch["prompt_attention_mask"].to(device)
        pred_last_idx = align_batch["pred_last_idx"].to(device)

        behavior_ids = align_batch["behavior_input_ids"].to(device)
        behavior_mask = align_batch["behavior_attention_mask"].to(device)
        behavior_last_idx = align_batch["behavior_last_idx"].to(device)

        pred_rep = self._extract_last_token_reps(model, prompt_ids, prompt_mask, pred_last_idx)
        behavior_rep = self._extract_last_token_reps(model, behavior_ids, behavior_mask, behavior_last_idx)

        if self.align_metric == "cosine":
            pred_rep = F.normalize(pred_rep, dim=-1)
            behavior_rep = F.normalize(behavior_rep, dim=-1)
            return 1.0 - F.cosine_similarity(pred_rep, behavior_rep, dim=-1).mean()
        if self.align_metric == "l2":
            return F.mse_loss(pred_rep, behavior_rep)
        raise ValueError(f"Unknown align_metric: {self.align_metric}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign_batch, align_batch = inputs
        device = model.device

        benign_inputs = {k: v.to(device) for k, v in benign_batch.items()}
        benign_out = model(**benign_inputs, use_cache=False)
        benign_ce = ce_loss_from_outputs(benign_out.logits, benign_batch["labels"].to(device))

        benign_kl = torch.zeros((), device=device)
        if self.w_benign_kl > 0.0:
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    with torch.no_grad():
                        base_out = model(**benign_inputs, use_cache=False)
            else:
                with torch.no_grad():
                    base_out = model(**benign_inputs, use_cache=False)
            benign_kl = topk_kl_from_logits(benign_out.logits, base_out.logits)

        align_loss = self._alignment_loss(model, align_batch)

        total_loss = (
            self.w_benign * benign_ce
            + self.w_align * align_loss
            + self.w_benign_kl * benign_kl
        )

        self.log(
            {
                "loss/benign_ce": benign_ce.item(),
                "loss/align_jepa": align_loss.item(),
                "loss/benign_kl": benign_kl.item(),
                "loss/total": total_loss.item(),
            }
        )
        return (total_loss, None) if return_outputs else total_loss


# -----------------------------
# 8. Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LLM-JEPA-style alignment model for jailbreak prompts -> behavior text.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--reverse_path", type=str, required=True, help="Path to reverse-model JSONL/JSON file.")
    parser.add_argument("--output_dir", type=str, default="./align_jepa_qwen")

    parser.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    parser.add_argument("--align_limit", type=int, default=DEFAULT_ALIGN_LIMIT)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)

    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_align", type=float, default=DEFAULT_W_ALIGN)
    parser.add_argument("--w_benign_kl", type=float, default=DEFAULT_W_BENIGN_KL)
    parser.add_argument("--num_pred_tokens", type=int, default=DEFAULT_NUM_PRED_TOKENS)
    parser.add_argument("--align_layer", type=int, default=DEFAULT_ALIGN_LAYER)
    parser.add_argument("--align_metric", type=str, default="cosine", choices=["cosine", "l2"])

    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special predictor token like LLM-JEPA.
    added = tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_PRED_TOKEN]})

    safe_prompts, safe_responses = load_ultrachat(args.ultrachat_samples)
    benign_data = tokenize_chat_generic(
        safe_prompts,
        safe_responses,
        tokenizer,
        max_length=args.max_length,
    )

    align_prompts, align_behaviors, align_sources = load_reverse_alignment_pairs(
        args.reverse_path,
        limit=args.align_limit,
    )
    if len(align_prompts) == 0:
        raise ValueError("No alignment pairs loaded from reverse_path.")

    align_data = tokenize_alignment_pairs(
        align_prompts,
        align_behaviors,
        tokenizer,
        max_length=args.max_length,
        num_pred_tokens=args.num_pred_tokens,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if added > 0 and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        # Only grow — never shrink. Many base models (e.g. Qwen3) pad the embedding
        # beyond len(tokenizer); resizing down would bake a mismatched embed_tokens/
        # lm_head into the saved adapter and break PEFT loading against the base model.
        model.resize_token_embeddings(len(tokenizer))

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(args.device)
    model.train()

    run_name = (
        f"align_jepa_tied"
        f"_wb{args.w_benign}"
        f"_wa{args.w_align}"
        f"_wkl{args.w_benign_kl}"
        f"_pred{args.num_pred_tokens}"
        f"_layer{args.align_layer}"
        f"_{args.align_metric}"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.num_max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=1.0,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = AlignmentJEPATrainer(
        model=model,
        benign_ds=TensorDictDataset(benign_data),
        align_ds=TensorDictDataset(align_data),
        align_layer=args.align_layer,
        w_benign=args.w_benign,
        w_align=args.w_align,
        w_benign_kl=args.w_benign_kl,
        align_metric=args.align_metric,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[LoggingCallback(args.output_dir)],
    )

    trainer.train()
    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))

    manifest = {
        "schema_version": "1.0",
        "defense_name": "align_jepa_tied",
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "training_completed": True,
        "weights": {
            "w_benign": args.w_benign,
            "w_align": args.w_align,
            "w_benign_kl": args.w_benign_kl,
        },
        "data": {
            "ultrachat_samples": args.ultrachat_samples,
            "reverse_path": args.reverse_path,
            "align_limit": args.align_limit,
            "num_alignment_pairs": len(align_data),
            "source_counts": {
                "generated_prompt": int(sum(x == "generated_prompt" for x in align_sources)),
            },
        },
        "config": {
            "num_pred_tokens": args.num_pred_tokens,
            "align_layer": args.align_layer,
            "align_metric": args.align_metric,
            "pred_token": DEFAULT_PRED_TOKEN,
            "target_view": "ground_truth_behavior_text_last_token",
        },
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    (output_dir / "READY").touch()
    print(f"Done. Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
