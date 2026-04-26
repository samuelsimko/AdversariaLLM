import argparse
import json
import os
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_CB_LIMIT = 5000
DEFAULT_REP_LAYERS = list(range(20, 30))
DEFAULT_W_HARM_JEPA = 2.0
DEFAULT_W_KL = 1.0
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_PREDICTOR_TYPE = "identity"


def _chat_template_input_ids(chat_tokens):
    if isinstance(chat_tokens, torch.Tensor):
        return chat_tokens
    if hasattr(chat_tokens, "input_ids"):
        return chat_tokens.input_ids
    if isinstance(chat_tokens, dict) and "input_ids" in chat_tokens:
        return chat_tokens["input_ids"]
    raise TypeError(f"Unsupported chat template output type: {type(chat_tokens)!r}")


def tokenize_chat_generic(prompts, responses, tokenizer, max_length=256):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = []
    for prompt, response in zip(prompts, responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        full = tokenizer.apply_chat_template(
            messages,
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

        pad = max_length - full.shape[-1]
        out.append(
            {
                "input_ids": F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(),
                "labels": F.pad(labels, (0, pad), value=-100).squeeze(),
                "attention_mask": F.pad(torch.ones_like(full), (0, pad), value=0).squeeze(),
            }
        )
    return out


class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_ultrachat(n):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(n))
    prompts, responses = [], []
    for row in ds:
        messages = row["messages"]
        user = next((message["content"] for message in messages if message["role"] == "user"), "")
        assistant = next((message["content"] for message in messages if message["role"] == "assistant"), "")
        prompts.append(user)
        responses.append(assistant)
    return prompts, responses


def _load_json_or_jsonl(path):
    with open(path) as f:
        text = f.read()
        try:
            return json.loads(text)
        except Exception:
            return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_cb(path, limit):
    records = _load_json_or_jsonl(path)[:limit]
    prompts = [record["prompt"] for record in records]
    harmful = [record.get("output", "") for record in records]
    return prompts, harmful


def parse_int_list(raw: Optional[str], default: List[int]) -> List[int]:
    if raw is None:
        return list(default)
    if isinstance(raw, list):
        return [int(x) for x in raw]
    values = [chunk.strip() for chunk in str(raw).split(",")]
    return [int(x) for x in values if x]


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


def topk_kl(adapted_logits, base_logits, k=50):
    logp_adapted = F.log_softmax(adapted_logits, dim=-1)
    logp_base = F.log_softmax(base_logits, dim=-1)

    topk = torch.topk(logp_base, k, dim=-1).indices
    adapted_selected = logp_adapted.gather(-1, topk)
    base_selected = logp_base.gather(-1, topk)
    base_probs = base_selected.exp()

    return (base_probs * (base_selected - adapted_selected)).sum(-1).mean()


def pooled_hidden_rep(hidden_states, batch, layers):
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    reps = []
    for layer in layers:
        layer_hidden = hidden_states[layer]
        target_mask = (labels != -100).unsqueeze(-1)
        if target_mask.any():
            mask = target_mask
        else:
            mask = attention_mask.unsqueeze(-1).bool()

        masked_hidden = layer_hidden * mask.to(layer_hidden.dtype)
        denom = mask.sum(dim=1).clamp(min=1).to(layer_hidden.dtype)
        reps.append(masked_hidden.sum(dim=1) / denom)
    return torch.stack(reps, dim=1)


def apply_predictor(reps, predictor_type):
    if predictor_type == "identity":
        return reps
    raise ValueError(f"Unsupported predictor_type: {predictor_type}")


def harmful_jepa_loss(model, batch_a, batch_b, layers, predictor_type):
    inputs_a = {k: v.to(model.device) for k, v in batch_a.items()}
    inputs_b = {k: v.to(model.device) for k, v in batch_b.items()}

    out_a = model(**inputs_a, output_hidden_states=True, use_cache=False)
    out_b = model(**inputs_b, output_hidden_states=True, use_cache=False)

    rep_a = pooled_hidden_rep(out_a.hidden_states, inputs_a, layers)
    rep_b = pooled_hidden_rep(out_b.hidden_states, inputs_b, layers)
    pred_a = apply_predictor(rep_a, predictor_type)
    return F.mse_loss(pred_a, rep_b)


class TripleBatchLoader:
    def __init__(self, benign_loader, harmful_loader_a, harmful_loader_b):
        self.benign_loader = benign_loader
        self.harmful_loader_a = harmful_loader_a
        self.harmful_loader_b = harmful_loader_b
        self._len = min(len(benign_loader), len(harmful_loader_a), len(harmful_loader_b))

    def __iter__(self):
        return zip(self.benign_loader, self.harmful_loader_a, self.harmful_loader_b)

    def __len__(self):
        return self._len


class TrainerHarmfulJEPA(Trainer):
    def __init__(
        self,
        benign,
        harmful,
        rep_layers,
        w_harm_jepa=2.0,
        w_kl=1.0,
        predictor_type="identity",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.benign = benign
        self.harmful = harmful
        self.rep_layers = rep_layers
        self.w_harm_jepa = w_harm_jepa
        self.w_kl = w_kl
        self.predictor_type = predictor_type

    def get_train_dataloader(self):
        batch_size = self.args.per_device_train_batch_size
        benign_loader = DataLoader(self.benign, batch_size=batch_size, shuffle=True)
        harmful_loader_a = DataLoader(self.harmful, batch_size=batch_size, shuffle=True)
        harmful_loader_b = DataLoader(self.harmful, batch_size=batch_size, shuffle=True)
        return TripleBatchLoader(benign_loader, harmful_loader_a, harmful_loader_b)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign_batch, harmful_batch_a, harmful_batch_b = inputs

        benign_inputs = {k: v.to(model.device) for k, v in benign_batch.items()}
        benign_out = model(**benign_inputs, use_cache=False)

        with no_adapter(model):
            with torch.no_grad():
                base_out = model(**benign_inputs, use_cache=False)

        kl = topk_kl(benign_out.logits, base_out.logits)
        harm_jepa = harmful_jepa_loss(
            model,
            harmful_batch_a,
            harmful_batch_b,
            self.rep_layers,
            self.predictor_type,
        )
        total = self.w_kl * kl + self.w_harm_jepa * harm_jepa

        self.log(
            {
                "loss": total.item(),
                "harm_jepa": harm_jepa.item(),
                "kl": kl.item(),
            }
        )
        return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", required=True)
    parser.add_argument("--output_dir", default="./harmful_jepa")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    parser.add_argument("--limit_cb", type=int, default=DEFAULT_CB_LIMIT)
    parser.add_argument("--rep_layers", default=None)
    parser.add_argument("--w_harm_jepa", type=float, default=DEFAULT_W_HARM_JEPA)
    parser.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)
    parser.add_argument("--predictor_type", default=DEFAULT_PREDICTOR_TYPE)
    parser.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--target_modules", default=None)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    rep_layers = parse_int_list(args.rep_layers, DEFAULT_REP_LAYERS)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benign_prompts, benign_responses = load_ultrachat(args.ultrachat_samples)
    harmful_prompts, harmful_responses = load_cb(args.cb_path, args.limit_cb)

    benign = ChatDataset(tokenize_chat_generic(benign_prompts, benign_responses, tokenizer, max_length=args.max_length))
    harmful = ChatDataset(tokenize_chat_generic(harmful_prompts, harmful_responses, tokenizer, max_length=args.max_length))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.target_modules:
        target_modules = [chunk.strip() for chunk in args.target_modules.split(",") if chunk.strip()]
    else:
        target_modules = infer_lora_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
    )

    trainer = TrainerHarmfulJEPA(
        model=model,
        benign=benign,
        harmful=harmful,
        rep_layers=rep_layers,
        w_harm_jepa=args.w_harm_jepa,
        w_kl=args.w_kl,
        predictor_type=args.predictor_type,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=args.batch_size,
            max_steps=args.num_max_steps,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            report_to=[],
            save_strategy="steps",
            save_steps=max(1, args.num_max_steps),
            remove_unused_columns=False,
        ),
    )

    trainer.train()
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(output_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "cb_path": args.cb_path,
                "output_dir": output_dir,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "num_max_steps": args.num_max_steps,
                "learning_rate": args.learning_rate,
                "ultrachat_samples": args.ultrachat_samples,
                "limit_cb": args.limit_cb,
                "rep_layers": rep_layers,
                "w_harm_jepa": args.w_harm_jepa,
                "w_kl": args.w_kl,
                "predictor_type": args.predictor_type,
                "target_modules": target_modules,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
