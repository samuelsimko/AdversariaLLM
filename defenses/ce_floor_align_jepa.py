import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_CB_LIMIT = 5000
DEFAULT_ALIGN_LIMIT = 5000
DEFAULT_HARM_CE_MIN = 4.0
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_HARM = 1.0
DEFAULT_W_KL = 0.1
DEFAULT_W_ALIGN = 0.5
DEFAULT_NUM_PRED_TOKENS = 1
DEFAULT_ALIGN_LAYER = -1
DEFAULT_PRED_TOKEN = "[PRED]"
DEFAULT_ALIGN_MODE = "original_plus_augmented"

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05


def _chat_template_input_ids(chat_tokens):
    if isinstance(chat_tokens, torch.Tensor):
        return chat_tokens
    if hasattr(chat_tokens, "input_ids"):
        return chat_tokens.input_ids
    if isinstance(chat_tokens, dict) and "input_ids" in chat_tokens:
        return chat_tokens["input_ids"]
    raise TypeError(f"Unsupported chat template output type: {type(chat_tokens)!r}")


def _extract_input_ids(template_output: Any) -> torch.Tensor:
    if isinstance(template_output, torch.Tensor):
        return template_output
    if hasattr(template_output, "input_ids"):
        return template_output.input_ids
    if isinstance(template_output, dict) and "input_ids" in template_output:
        return template_output["input_ids"]
    raise TypeError(f"Unsupported chat template output type: {type(template_output)!r}")


def tokenize_chat_generic(prompts, responses, tokenizer, max_length=256):
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have matching lengths")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        full = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        full = _chat_template_input_ids(full)

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_generation_prompt=True,
        )
        prompt_only = _chat_template_input_ids(prompt_only)

        labels = full.clone()
        prompt_len = min(prompt_only.shape[-1], full.shape[-1])
        labels[0, :prompt_len] = -100

        full = full[:, :max_length]
        labels = labels[:, :max_length]
        pad = max_length - full.shape[-1]

        out.append(
            {
                "input_ids": F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(0),
                "labels": F.pad(labels, (0, pad), value=-100).squeeze(0),
                "attention_mask": F.pad(torch.ones_like(full), (0, pad), value=0).squeeze(0),
            }
        )
    return out


class TensorDictDataset(Dataset):
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]


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
                out.append(json.loads(s))
            return out

        if first_non_ws == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
        return parse_jsonl(f)


def load_ultrachat(num_samples: int, seed: int = 42) -> Tuple[List[str], List[str]]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(num_samples))
    prompts, responses = [], []
    for item in ds:
        messages = item["messages"]
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        if user and assistant:
            prompts.append(user)
            responses.append(assistant)
    return prompts, responses


def load_cb(path: str, limit: int) -> Tuple[List[str], List[str]]:
    records = _load_json_or_jsonl(path)[:limit]
    prompts = [record["prompt"] for record in records]
    harmful_outputs = [record.get("output", "") for record in records]
    return prompts, harmful_outputs


def load_reverse_alignment_pairs(
    path: str,
    limit: Optional[int],
    align_mode: str,
) -> Tuple[List[str], List[str], List[str]]:
    records = _load_json_or_jsonl(path)
    if limit is not None:
        records = records[:limit]

    prompts: List[str] = []
    behaviors: List[str] = []
    source_labels: List[str] = []

    for rec in records:
        base = rec.get("record", rec)
        behavior_text = rec.get("true_prompt") or base.get("prompt") or rec.get("prompt") or ""
        original_prompt = rec.get("true_prompt") or base.get("prompt") or rec.get("prompt") or ""
        generated_prompts = rec.get("generated_prompts") or []

        if not behavior_text:
            continue

        if original_prompt.strip():
            prompts.append(original_prompt.strip())
            behaviors.append(behavior_text.strip())
            source_labels.append("original_prompt")

        if align_mode == "original_plus_augmented":
            for gp in generated_prompts:
                if isinstance(gp, str) and gp.strip():
                    prompts.append(gp.strip())
                    behaviors.append(behavior_text.strip())
                    source_labels.append("generated_prompt")

    return prompts, behaviors, source_labels


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
    if len(jailbreak_prompts) != len(behaviors):
        raise ValueError("jailbreak_prompts and behaviors must match length")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pred_suffix = build_pred_suffix(tokenizer, num_pred_tokens)
    out: List[Dict[str, torch.Tensor]] = []

    for prompt, behavior in zip(jailbreak_prompts, behaviors):
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_tokens = _extract_input_ids(prompt_tokens)
        prompt_with_pred = torch.cat([prompt_tokens, pred_suffix], dim=-1)[:, :max_length]
        pred_last_idx = prompt_with_pred.shape[-1] - 1

        prompt_pad = max_length - prompt_with_pred.shape[-1]
        prompt_ids = F.pad(prompt_with_pred, (0, prompt_pad), value=tokenizer.pad_token_id).squeeze(0)
        prompt_mask = F.pad(torch.ones_like(prompt_with_pred), (0, prompt_pad), value=0).squeeze(0)

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


def infer_lora_target_modules(model) -> List[str]:
    common_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    seen = set()
    for name, _ in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in common_suffixes:
            seen.add(suffix)
    preferred_order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    resolved = [suffix for suffix in preferred_order if suffix in seen]
    if not resolved:
        raise ValueError("Could not infer LoRA target modules from model.")
    return resolved


@contextmanager
def no_adapter(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def token_average_ce_per_example(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels.ne(-100)
    denom = valid.sum(dim=-1).clamp(min=1)
    return flat_loss.sum(dim=-1) / denom


def topk_kl_from_logits(adapted_logits: torch.Tensor, base_logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    logp_adapted = F.log_softmax(adapted_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)
    topk_idx = torch.topk(logp_base, k=min(k, logp_base.size(-1)), dim=-1).indices
    adapted_selected = logp_adapted.gather(-1, topk_idx)
    base_selected = logp_base.gather(-1, topk_idx)
    base_probs = base_selected.exp()
    return (base_probs * (base_selected - adapted_selected)).sum(dim=-1).mean()


class LoggingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.csv_path = Path(output_dir) / "metrics.csv"
        self.rows: List[Dict[str, Any]] = []
        self.fieldnames = ["step"]

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row = {"step": state.global_step}
        row.update(logs)
        for key in row:
            if key not in self.fieldnames:
                self.fieldnames.append(key)
        self.rows.append(row)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            for existing in self.rows:
                writer.writerow({k: existing.get(k, "") for k in self.fieldnames})


class TripleBatchLoader:
    def __init__(self, benign_loader: DataLoader, harmful_loader: DataLoader, align_loader: DataLoader):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self.align_loader = align_loader
        self._len = min(len(benign_loader), len(harmful_loader), len(align_loader))

    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader, self.align_loader)

    def __len__(self) -> int:
        return self._len


class CEFloorAlignJEPATrainer(Trainer):
    def __init__(
        self,
        benign_ds: Dataset,
        harmful_ds: Dataset,
        align_ds: Dataset,
        harm_ce_min: float,
        w_benign: float,
        w_harm: float,
        w_kl: float,
        w_align: float,
        align_layer: int,
        align_metric: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.align_ds = align_ds
        self.harm_ce_min = harm_ce_min
        self.w_benign = w_benign
        self.w_harm = w_harm
        self.w_kl = w_kl
        self.w_align = w_align
        self.align_layer = align_layer
        self.align_metric = align_metric

    def get_train_dataloader(self):
        batch_size = self.args.per_device_train_batch_size
        benign_loader = DataLoader(self.benign_ds, batch_size=batch_size, shuffle=True)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=batch_size, shuffle=True)
        align_loader = DataLoader(self.align_ds, batch_size=batch_size, shuffle=True)
        return TripleBatchLoader(benign_loader, harmful_loader, align_loader)

    def _extract_last_token_reps(self, model, input_ids, attention_mask, last_indices):
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
        benign_batch, harmful_batch, align_batch = inputs
        device = model.device

        benign_inputs = {k: v.to(device) for k, v in benign_batch.items()}
        harmful_inputs = {k: v.to(device) for k, v in harmful_batch.items()}

        benign_out = model(**benign_inputs, use_cache=False)
        harmful_out = model(**harmful_inputs, use_cache=False)

        benign_ce_def = token_average_ce_per_example(benign_out.logits, benign_inputs["labels"])
        harmful_ce_def = token_average_ce_per_example(harmful_out.logits, harmful_inputs["labels"])

        with no_adapter(model):
            with torch.no_grad():
                benign_base_out = model(**benign_inputs, use_cache=False)

        benign_ce_base = token_average_ce_per_example(benign_base_out.logits, benign_inputs["labels"])
        benign_anchor = (benign_ce_def - benign_ce_base).pow(2).mean()
        harm_floor = F.relu(self.harm_ce_min - harmful_ce_def).pow(2).mean()

        benign_kl = torch.zeros((), device=device)
        if self.w_kl > 0.0:
            benign_kl = topk_kl_from_logits(benign_out.logits, benign_base_out.logits)

        align_loss = torch.zeros((), device=device)
        if self.w_align > 0.0:
            align_loss = self._alignment_loss(model, align_batch)

        total_loss = (
            self.w_benign * benign_anchor
            + self.w_harm * harm_floor
            + self.w_kl * benign_kl
            + self.w_align * align_loss
        )

        self.log(
            {
                "loss/benign_anchor": benign_anchor.item(),
                "loss/harm_floor": harm_floor.item(),
                "loss/benign_kl": benign_kl.item(),
                "loss/align_jepa": align_loss.item(),
                "loss/total": total_loss.item(),
                "metrics/benign_ce_def": benign_ce_def.mean().item(),
                "metrics/benign_ce_base": benign_ce_base.mean().item(),
                "metrics/harm_ce_def": harmful_ce_def.mean().item(),
                "metrics/harm_ce_gap_to_floor": (harmful_ce_def.mean() - self.harm_ce_min).item(),
            }
        )
        return (total_loss, None) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description="Train a CE-floor plus tied-weight JEPA safety model.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", type=str, required=True)
    parser.add_argument("--reverse_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ce_floor_align_jepa")
    parser.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    parser.add_argument("--limit_cb", type=int, default=DEFAULT_CB_LIMIT)
    parser.add_argument("--align_limit", type=int, default=DEFAULT_ALIGN_LIMIT)
    parser.add_argument("--align_mode", type=str, default=DEFAULT_ALIGN_MODE, choices=["original_only", "original_plus_augmented"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--harm_ce_min", type=float, default=DEFAULT_HARM_CE_MIN)
    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_harm", type=float, default=DEFAULT_W_HARM)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)
    parser.add_argument("--w_align", type=float, default=DEFAULT_W_ALIGN)
    parser.add_argument("--num_pred_tokens", type=int, default=DEFAULT_NUM_PRED_TOKENS)
    parser.add_argument("--align_layer", type=int, default=DEFAULT_ALIGN_LAYER)
    parser.add_argument("--align_metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_PRED_TOKEN]})

    benign_prompts, benign_responses = load_ultrachat(args.ultrachat_samples)
    harmful_prompts, harmful_responses = load_cb(args.cb_path, args.limit_cb)
    align_prompts, align_behaviors, align_sources = load_reverse_alignment_pairs(
        args.reverse_path,
        limit=args.align_limit,
        align_mode=args.align_mode,
    )
    if not align_prompts:
        raise ValueError("No JEPA alignment pairs loaded from reverse_path.")

    benign_ds = TensorDictDataset(
        tokenize_chat_generic(benign_prompts, benign_responses, tokenizer, max_length=args.max_length)
    )
    harmful_ds = TensorDictDataset(
        tokenize_chat_generic(harmful_prompts, harmful_responses, tokenizer, max_length=args.max_length)
    )
    align_ds = TensorDictDataset(
        tokenize_alignment_pairs(
            align_prompts,
            align_behaviors,
            tokenizer,
            max_length=args.max_length,
            num_pred_tokens=args.num_pred_tokens,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if added > 0 and len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    if args.target_modules:
        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    else:
        target_modules = infer_lora_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.to(args.device)
    model.train()

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
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = CEFloorAlignJEPATrainer(
        model=model,
        benign_ds=benign_ds,
        harmful_ds=harmful_ds,
        align_ds=align_ds,
        harm_ce_min=args.harm_ce_min,
        w_benign=args.w_benign,
        w_harm=args.w_harm,
        w_kl=args.w_kl,
        w_align=args.w_align,
        align_layer=args.align_layer,
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
        "defense_name": "ce_floor_align_jepa",
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "training_completed": True,
        "weights": {
            "w_benign": args.w_benign,
            "w_harm": args.w_harm,
            "w_kl": args.w_kl,
            "w_align": args.w_align,
        },
        "data": {
            "cb_path": args.cb_path,
            "ultrachat_samples": args.ultrachat_samples,
            "limit_cb": args.limit_cb,
            "reverse_path": args.reverse_path,
            "align_limit": args.align_limit,
            "num_alignment_pairs": len(align_ds),
            "source_counts": {
                "original_prompt": int(sum(x == "original_prompt" for x in align_sources)),
                "generated_prompt": int(sum(x == "generated_prompt" for x in align_sources)),
            },
        },
        "config": {
            "harm_ce_min": args.harm_ce_min,
            "align_mode": args.align_mode,
            "num_pred_tokens": args.num_pred_tokens,
            "align_layer": args.align_layer,
            "align_metric": args.align_metric,
            "pred_token": DEFAULT_PRED_TOKEN,
            "target_modules": target_modules,
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
