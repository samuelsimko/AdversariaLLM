import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_CB_LIMIT = 5000
DEFAULT_HARM_CE_MIN = 4.0
DEFAULT_W_BENIGN = 1.0
DEFAULT_W_KL = 0.1
DEFAULT_W_CE = 1.0
DEFAULT_W_Y_TO_S = 0.1
DEFAULT_W_SEP = 0.1
DEFAULT_W_ORTH = 1e-4
DEFAULT_PROJECTION_DIM = 16
DEFAULT_MARGIN = 0.5
DEFAULT_REFUSAL_TEXT = "I'm sorry, I can't help with that."
DEFAULT_REP_LAYERS = list(range(20, 36))
DEFAULT_REPORT_TO = "wandb"

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
        text = f.read()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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


def resolve_rep_layers(requested: List[int], hidden_state_count: int) -> List[int]:
    if hidden_state_count <= 1:
        return [0]
    max_index = hidden_state_count - 1
    resolved: List[int] = []
    for raw in requested:
        if raw < 0:
            idx = hidden_state_count + raw
        else:
            idx = raw
        idx = min(max(idx, 1), max_index)
        if idx not in resolved:
            resolved.append(idx)
    return resolved or [max_index]


def pooled_assistant_rep_from_outputs(outputs, labels: torch.Tensor, rep_layers: List[int]) -> torch.Tensor:
    assistant_mask = labels.ne(-100)
    hidden_states = outputs.hidden_states
    selected_layers = resolve_rep_layers(rep_layers, len(hidden_states))
    pooled_layers = []
    fallback_mask = labels.ne(-100)
    for layer_idx in selected_layers:
        hidden = hidden_states[layer_idx]
        mask = assistant_mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        if bool(mask.sum().item()) is False:
            mask = fallback_mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_layers.append((hidden * mask).sum(dim=1) / denom)
    return torch.stack(pooled_layers, dim=0).mean(dim=0)


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


class DualBatchLoader:
    def __init__(self, benign_loader: DataLoader, harmful_loader: DataLoader):
        self.benign_loader = benign_loader
        self.harmful_loader = harmful_loader
        self._len = min(len(benign_loader), len(harmful_loader))

    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader)

    def __len__(self) -> int:
        return self._len


class CEFloorRefusalAttractorTrainer(Trainer):
    def __init__(
        self,
        benign_ds: Dataset,
        harmful_ds: Dataset,
        refusal_batch: Dict[str, torch.Tensor],
        rep_layers: List[int],
        harm_ce_min: float,
        w_benign: float,
        w_kl: float,
        w_ce: float,
        w_y_to_s: float,
        w_sep: float,
        w_orth: float,
        margin: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign_ds = benign_ds
        self.harmful_ds = harmful_ds
        self.refusal_batch = refusal_batch
        self.rep_layers = rep_layers
        self.harm_ce_min = harm_ce_min
        self.w_benign = w_benign
        self.w_kl = w_kl
        self.w_ce = w_ce
        self.w_y_to_s = w_y_to_s
        self.w_sep = w_sep
        self.w_orth = w_orth
        self.margin = margin

    def get_train_dataloader(self):
        batch_size = self.args.per_device_train_batch_size
        benign_loader = DataLoader(self.benign_ds, batch_size=batch_size, shuffle=True)
        harmful_loader = DataLoader(self.harmful_ds, batch_size=batch_size, shuffle=True)
        return DualBatchLoader(benign_loader, harmful_loader)

    def _refusal_rep(self, model, device: torch.device) -> torch.Tensor:
        refusal_inputs = {k: v.unsqueeze(0).to(device) for k, v in self.refusal_batch.items()}
        refusal_out = model(**refusal_inputs, output_hidden_states=True, use_cache=False)
        return pooled_assistant_rep_from_outputs(refusal_out, refusal_inputs["labels"], self.rep_layers)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign_batch, harmful_batch = inputs
        device = model.device

        benign_inputs = {k: v.to(device) for k, v in benign_batch.items()}
        harmful_inputs = {k: v.to(device) for k, v in harmful_batch.items()}

        benign_out = model(**benign_inputs, output_hidden_states=True, use_cache=False)
        harmful_out = model(**harmful_inputs, output_hidden_states=True, use_cache=False)

        benign_ce_def = token_average_ce_per_example(benign_out.logits, benign_inputs["labels"])
        harmful_ce_def = token_average_ce_per_example(harmful_out.logits, harmful_inputs["labels"])

        with no_adapter(model):
            with torch.no_grad():
                benign_base_out = model(**benign_inputs, use_cache=False)

        benign_ce_base = token_average_ce_per_example(benign_base_out.logits, benign_inputs["labels"])
        loss_benign = (benign_ce_def - benign_ce_base).pow(2).mean()

        loss_kl = torch.zeros((), device=device)
        if self.w_kl > 0.0:
            loss_kl = topk_kl_from_logits(benign_out.logits, benign_base_out.logits)

        ce_harm = harmful_ce_def.mean()
        loss_ce_floor = F.relu(self.harm_ce_min - harmful_ce_def).pow(2).mean()

        h_harm = pooled_assistant_rep_from_outputs(harmful_out, harmful_inputs["labels"], self.rep_layers)
        h_benign = pooled_assistant_rep_from_outputs(benign_out, benign_inputs["labels"], self.rep_layers)
        h_stop = self._refusal_rep(model, device)

        projector = model.refusal_projector
        z_harm = F.normalize(projector(h_harm), dim=-1)
        z_benign = F.normalize(projector(h_benign), dim=-1)
        z_stop = F.normalize(projector(h_stop), dim=-1)
        if z_stop.ndim == 1:
            z_stop = z_stop.unsqueeze(0)
        z_stop_harm = z_stop.expand(z_harm.size(0), -1)
        z_stop_benign = z_stop.expand(z_benign.size(0), -1)

        dist_harm_stop = 1.0 - (z_harm * z_stop_harm).sum(dim=-1)
        dist_benign_stop = 1.0 - (z_benign * z_stop_benign).sum(dim=-1)
        loss_y_to_s = dist_harm_stop.mean()
        loss_sep = F.relu(self.margin - dist_benign_stop).pow(2).mean()

        W = projector.weight
        gram = W @ W.T
        I = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        loss_orth = (gram - I).pow(2).mean()

        total_loss = (
            self.w_benign * loss_benign
            + self.w_kl * loss_kl
            + self.w_ce * loss_ce_floor
            + self.w_y_to_s * loss_y_to_s
            + self.w_sep * loss_sep
            + self.w_orth * loss_orth
        )

        self.log(
            {
                "loss/total": total_loss.item(),
                "loss/benign": loss_benign.item(),
                "loss/kl": loss_kl.item(),
                "loss/ce_floor": loss_ce_floor.item(),
                "loss/y_to_s": loss_y_to_s.item(),
                "loss/sep": loss_sep.item(),
                "loss/orth": loss_orth.item(),
                "stats/dist_harm_stop": dist_harm_stop.mean().item(),
                "stats/dist_benign_stop": dist_benign_stop.mean().item(),
                "stats/ce_harm": ce_harm.item(),
                "stats/ce_benign_def": benign_ce_def.mean().item(),
                "stats/ce_benign_base": benign_ce_base.mean().item(),
            }
        )
        return (total_loss, None) if return_outputs else total_loss


def main():
    parser = argparse.ArgumentParser(description="CE-floor defense with refusal-attractor projected collapse.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ce_floor_refusal_attractor")
    parser.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    parser.add_argument("--limit_cb", type=int, default=DEFAULT_CB_LIMIT)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--harm_ce_min", type=float, default=DEFAULT_HARM_CE_MIN)
    parser.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)
    parser.add_argument("--w_ce", type=float, default=DEFAULT_W_CE)
    parser.add_argument("--w_y_to_s", type=float, default=DEFAULT_W_Y_TO_S)
    parser.add_argument("--w_sep", type=float, default=DEFAULT_W_SEP)
    parser.add_argument("--w_orth", type=float, default=DEFAULT_W_ORTH)
    parser.add_argument("--projection_dim", type=int, default=DEFAULT_PROJECTION_DIM)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--refusal_text", type=str, default=DEFAULT_REFUSAL_TEXT)
    parser.add_argument("--rep_layers", type=int, nargs="*", default=DEFAULT_REP_LAYERS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--report_to", type=str, default=DEFAULT_REPORT_TO)
    parser.add_argument("--run_name", type=str, default=None)
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

    benign_prompts, benign_responses = load_ultrachat(args.ultrachat_samples)
    harmful_prompts, harmful_responses = load_cb(args.cb_path, args.limit_cb)
    refusal_batch = tokenize_chat_generic(
        ["Can you help me with something unsafe?"],
        [args.refusal_text],
        tokenizer,
        max_length=args.max_length,
    )[0]

    benign_ds = TensorDictDataset(
        tokenize_chat_generic(benign_prompts, benign_responses, tokenizer, max_length=args.max_length)
    )
    harmful_ds = TensorDictDataset(
        tokenize_chat_generic(harmful_prompts, harmful_responses, tokenizer, max_length=args.max_length)
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
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
    model.add_module("refusal_projector", nn.Linear(model.config.hidden_size, args.projection_dim, bias=False))
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
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
    )

    trainer = CEFloorRefusalAttractorTrainer(
        model=model,
        benign_ds=benign_ds,
        harmful_ds=harmful_ds,
        refusal_batch=refusal_batch,
        rep_layers=args.rep_layers,
        harm_ce_min=args.harm_ce_min,
        w_benign=args.w_benign,
        w_kl=args.w_kl,
        w_ce=args.w_ce,
        w_y_to_s=args.w_y_to_s,
        w_sep=args.w_sep,
        w_orth=args.w_orth,
        margin=args.margin,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[LoggingCallback(args.output_dir)],
    )

    trainer.train()

    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    torch.save(model.refusal_projector.state_dict(), output_dir / "projector.pt")

    manifest = {
        "schema_version": "1.0",
        "defense_name": "ce_floor_refusal_attractor",
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "projection_dim": args.projection_dim,
        "margin": args.margin,
        "refusal_text": args.refusal_text,
        "weights": {
            "w_benign": args.w_benign,
            "w_kl": args.w_kl,
            "w_ce": args.w_ce,
            "w_y_to_s": args.w_y_to_s,
            "w_sep": args.w_sep,
            "w_orth": args.w_orth,
        },
        "rep_layers": args.rep_layers,
        "data": {
            "cb_path": args.cb_path,
            "ultrachat_samples": args.ultrachat_samples,
            "limit_cb": args.limit_cb,
        },
        "training_completed": True,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(output_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    (output_dir / "READY").touch()
    print(f"Done. Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
