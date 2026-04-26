# defenses/velocity_collapse_cb.py

import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from typing import List, Optional
import argparse
from contextlib import contextmanager

# =============================
# Defaults
# =============================
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_MAX_STEPS = 1500
DEFAULT_LR = 2e-4
DEFAULT_ULTRACHAT_SAMPLES = 5000
DEFAULT_CB_LIMIT = 5000
DEFAULT_REP_LAYERS = list(range(20, 30))
DEFAULT_W_VEL_HARM = 2.0
DEFAULT_W_VEL_BENIGN = 1.0
DEFAULT_W_KL = 1.0
DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.0

DEFAULT_REFUSAL_RESPONSE = "I'm sorry, I cannot do that."

# =============================
# Tokenization
# =============================
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
    for p, r in zip(prompts, responses):
        msgs = [{"role": "user", "content": p},
                {"role": "assistant", "content": r}]
        full = tokenizer.apply_chat_template(
            msgs, tokenize=True, return_tensors="pt",
            truncation=True, max_length=max_length
        )
        full = _chat_template_input_ids(full)

        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, return_tensors="pt",
            truncation=True, max_length=max_length,
            add_generation_prompt=True
        )
        prompt_only = _chat_template_input_ids(prompt_only)

        labels = full.clone()
        prompt_len = min(prompt_only.shape[-1], full.shape[-1])
        labels[0, :prompt_len] = -100

        pad = max_length - full.shape[-1]
        out.append({
            "input_ids": F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(),
            "labels": F.pad(labels, (0, pad), value=-100).squeeze(),
            "attention_mask": F.pad(torch.ones_like(full), (0, pad), value=0).squeeze()
        })
    return out


class ChatDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

# =============================
# Data loading
# =============================
def load_ultrachat(n):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").select(range(n))
    p, r = [], []
    for x in ds:
        msgs = x["messages"]
        u = next((m["content"] for m in msgs if m["role"]=="user"), "")
        a = next((m["content"] for m in msgs if m["role"]=="assistant"), "")
        p.append(u); r.append(a)
    return p, r


def _load_json_or_jsonl(path):
    with open(path) as f:
        txt = f.read()
        try: return json.loads(txt)
        except: return [json.loads(l) for l in txt.splitlines() if l.strip()]


def load_cb(path, limit):
    recs = _load_json_or_jsonl(path)[:limit]
    p = [r["prompt"] for r in recs]
    h = [r.get("output","") for r in recs]
    ref = [r.get("llama3_output", DEFAULT_REFUSAL_RESPONSE) for r in recs]
    return p, h, ref


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


# =============================
# Utils
# =============================
@contextmanager
def no_adapter(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter(): yield
    else: yield


def ce_loss(logits, labels):
    return F.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)),
        labels[..., 1:].reshape(-1),
        ignore_index=-100
    )


# =============================
# 🔥 Velocity losses
# =============================
def velocity_loss(model, batch, layers, collapse=True):
    ids = batch["input_ids"].to(model.device)
    mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)

    out = model(ids, attention_mask=mask, output_hidden_states=True)
    hs = out.hidden_states

    with no_adapter(model):
        with torch.no_grad():
            base = model(ids, attention_mask=mask, output_hidden_states=True)
            hs_base = base.hidden_states

    losses = []
    for l in layers:
        h = hs[l]
        v = h[:,1:] - h[:,:-1]
        m = (labels[:,1:] != -100).unsqueeze(-1)

        if collapse:
            losses.append((v.pow(2)*m).mean())
        else:
            v_base = hs_base[l][:,1:] - hs_base[l][:,:-1]
            losses.append(((v-v_base).pow(2)*m).mean())

    return sum(losses)/len(losses)


# =============================
# 🔥 KL constraint
# =============================
def topk_kl(a, b, k=50):
    logp_a = F.log_softmax(a, dim=-1)
    logp_b = F.log_softmax(b, dim=-1)

    topk = torch.topk(logp_b, k, dim=-1).indices
    la = logp_a.gather(-1, topk)
    lb = logp_b.gather(-1, topk)
    pb = lb.exp()

    return (pb*(lb-la)).sum(-1).mean()


# =============================
# Trainer
# =============================
class TrainerVC(Trainer):
    def __init__(self, benign, harmful, refusal,
                 w_vel_harm=2.0,
                 w_vel_benign=1.0,
                 w_kl=1.0,
                 rep_layers=[15,20,25],
                 *args, **kw):
        super().__init__(*args, **kw)
        self.benign = benign
        self.harmful = harmful
        self.refusal = refusal
        self.layers = rep_layers
        self.w_vel_harm = w_vel_harm
        self.w_vel_benign = w_vel_benign
        self.w_kl = w_kl

    def get_train_dataloader(self):
        return zip(
            DataLoader(self.benign, batch_size=self.args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.refusal, batch_size=self.args.per_device_train_batch_size, shuffle=True),
            DataLoader(self.harmful, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        b, r, h = inputs

        # forward
        ob = model(**{k:v.to(model.device) for k,v in b.items()}, output_hidden_states=True)
        orf = model(**{k:v.to(model.device) for k,v in r.items()})
        oh = model(**{k:v.to(model.device) for k,v in h.items()})

        # CE
        loss = ce_loss(ob.logits, b["labels"].to(model.device))
        loss += ce_loss(orf.logits, r["labels"].to(model.device))

        # 🔴 velocity collapse harmful
        vel_h = velocity_loss(model, h, self.layers, True)

        # 🟢 preserve benign velocity
        vel_b = velocity_loss(model, b, self.layers, False)

        # 🟢 KL preserve benign logits
        with no_adapter(model):
            with torch.no_grad():
                base = model(**{k:v.to(model.device) for k,v in b.items()})
        kl = topk_kl(ob.logits, base.logits)

        total = loss + self.w_vel_harm*vel_h + self.w_vel_benign*vel_b + self.w_kl*kl

        self.log({
            "loss": total.item(),
            "vel_h": vel_h.item(),
            "vel_b": vel_b.item(),
            "kl": kl.item()
        })

        return total


# =============================
# Main
# =============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cb_path", required=True)
    p.add_argument("--output_dir", default="./vc")
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num_max_steps", type=int, default=DEFAULT_NUM_MAX_STEPS)
    p.add_argument("--learning_rate", type=float, default=DEFAULT_LR)
    p.add_argument("--ultrachat_samples", type=int, default=DEFAULT_ULTRACHAT_SAMPLES)
    p.add_argument("--limit_cb", type=int, default=DEFAULT_CB_LIMIT)
    p.add_argument("--rep_layers", default=None)
    p.add_argument("--w_vel_harm", type=float, default=DEFAULT_W_VEL_HARM)
    p.add_argument("--w_vel_benign", type=float, default=DEFAULT_W_VEL_BENIGN)
    p.add_argument("--w_kl", type=float, default=DEFAULT_W_KL)
    p.add_argument("--lora_r", type=int, default=DEFAULT_LORA_R)
    p.add_argument("--lora_alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--target_modules", default=None)
    args = p.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    adapter_dir = os.path.join(output_dir, "lora_adapter")
    rep_layers = parse_int_list(args.rep_layers, DEFAULT_REP_LAYERS)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # data
    bp, br = load_ultrachat(args.ultrachat_samples)
    cp, ch, cr = load_cb(args.cb_path, args.limit_cb)

    benign = ChatDataset(tokenize_chat_generic(bp, br, tok, max_length=args.max_length))
    harmful = ChatDataset(tokenize_chat_generic(cp, ch, tok, max_length=args.max_length))
    refusal = ChatDataset(tokenize_chat_generic(cp, cr, tok, max_length=args.max_length))

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

    trainer = TrainerVC(
        model=model,
        benign=benign,
        harmful=harmful,
        refusal=refusal,
        rep_layers=rep_layers,
        w_vel_harm=args.w_vel_harm,
        w_vel_benign=args.w_vel_benign,
        w_kl=args.w_kl,
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
        )
    )

    trainer.train()
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
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
                "w_vel_harm": args.w_vel_harm,
                "w_vel_benign": args.w_vel_benign,
                "w_kl": args.w_kl,
                "target_modules": target_modules,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
