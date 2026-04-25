import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from typing import List
import argparse

# =========================
# Config
# =========================
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LEN = 512

TEMPLATE = """You are given a model response. Infer a plausible user prompt that could have led to this response.
Return only the prompt, wrapped exactly in <PROMPT> and </PROMPT> tags.

Response:
{response}

User prompt:
<PROMPT>{prompt}</PROMPT>
"""

# =========================
# Utils
# =========================
def _load_json_or_jsonl(path):
    with open(path, "r") as f:
        txt = f.read()
        try:
            return json.loads(txt)
        except:
            return [json.loads(l) for l in txt.splitlines() if l.strip()]

# =========================
# Dataset builders
# =========================
def build_cb_reverse(path, limit=None):
    data = _load_json_or_jsonl(path)
    if limit:
        data = data[:limit]

    examples = []
    for r in data:
        y = r.get("output", "")
        x = r.get("prompt", "")
        if y and x:
            examples.append((y, x))
    return examples

def build_ultrachat_reverse(n=20000):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle().select(range(n))

    examples = []
    for item in ds:
        msgs = item["messages"]
        user = next((m["content"] for m in msgs if m["role"]=="user"), "")
        asst = next((m["content"] for m in msgs if m["role"]=="assistant"), "")
        if user and asst:
            examples.append((asst, user))
    return examples

# =========================
# Tokenization
# =========================
def tokenize(tokenizer, pairs: List, max_len=512):
    out = []
    for y, x in pairs:
        text = TEMPLATE.format(response=y, prompt=x)

        toks = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )

        labels = toks["input_ids"].clone()

        out.append({
            "input_ids": toks["input_ids"].squeeze(),
            "attention_mask": toks["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        })
    return out

class SimpleDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--cb_path", required=True)
    parser.add_argument("--output_dir", default="./reverse_model")
    parser.add_argument("--cb_limit", type=int, default=5000)
    parser.add_argument("--ultra_samples", type=int, default=20000)
    parser.add_argument("--holdout", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    print("Loading datasets...")
    cb_data = build_cb_reverse(args.cb_path, args.cb_limit)

    # holdout split
    random.shuffle(cb_data)
    holdout = cb_data[:args.holdout]
    train_cb = cb_data[args.holdout:]

    ultra = build_ultrachat_reverse(args.ultra_samples)

    train_data = train_cb + ultra
    random.shuffle(train_data)

    print(f"Train size: {len(train_data)}")
    print(f"Holdout size: {len(holdout)}")

    # Save holdout
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "holdout.json"), "w") as f:
        json.dump(holdout, f, indent=2)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    train_tok = tokenize(tokenizer, train_data, max_len=args.max_len)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # LoRA
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora)

    # Trainer
    trainer = Trainer(
        model=model,
        train_dataset=SimpleDataset(train_tok),
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            bf16=True,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            report_to="none"
        )
    )

    print("Training...")
    trainer.train()

    model.save_pretrained(os.path.join(args.output_dir, "lora"))

    print("Done.")

if __name__ == "__main__":
    main()
