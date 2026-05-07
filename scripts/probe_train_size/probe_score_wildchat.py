"""Score WildChat first-user prompts with the trained advmix probe per cell.

Loads the saved joblib (scaler + svm) from probe_gate_advmix and applies it
to 300 WildChat conversations (first user message), through the cell's adapter.
Outputs <cell>_wildchat.csv with per-prompt probe scores.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ADVMIX = REPO / "runs/experiments/headline_backup_rerun_probing/probe_gate_advmix"

CELL_BASE_MODEL = {
    "l_cb_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "l_cb_no_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "l_ce_in_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "l_ce_in_no_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "l_triplet_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "l_triplet_no_pra": "meta-llama/Meta-Llama-3-8B-Instruct",
    "q_cb_pra": "Qwen/Qwen3-8B",
    "q_cb_no_pra": "Qwen/Qwen3-8B",
    "q_ce_in_pra": "Qwen/Qwen3-8B",
    "q_ce_in_no_pra": "Qwen/Qwen3-8B",
    "q_triplet_pra": "Qwen/Qwen3-8B",
    "q_triplet_no_pra": "Qwen/Qwen3-8B",
}


def load_cell(cell, device="cuda"):
    base = CELL_BASE_MODEL[cell]
    adapter = REPO / "runs/experiments/headline_backup_rerun" / cell / "lora_adapter"
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    model = PeftModel.from_pretrained(model, str(adapter))
    return tok, model


def sample_wildchat(n: int, seed: int = 42) -> list[str]:
    """Pull n non-toxic WildChat first-user prompts."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    ds = load_dataset("allenai/WildChat-1M", split="train")
    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    out = []
    for i in indices:
        row = ds[i]
        # Filter: non-toxic, English, has user content
        if row.get("toxic", False):
            continue
        conv = row.get("conversation") or []
        if not conv:
            continue
        # First user turn
        for m in conv:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and 10 < len(content) < 4000:
                    out.append(content)
                    break
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def extract_reps(model, tok, prompts, layer=25):
    out = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    for i, p in enumerate(tqdm(prompts, unit="prompt")):
        inputs = tok(p, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        h = model(**inputs, output_hidden_states=True).hidden_states[layer][0, -1, :]
        out[i] = h.detach().cpu().float().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"\n=== {args.cell} on WildChat ===", flush=True)
    prompts = sample_wildchat(args.n, seed=args.seed)
    print(f"[load] sampled {len(prompts)} WildChat first-user prompts", flush=True)

    tok, model = load_cell(args.cell, device=args.device)
    reps = extract_reps(model, tok, prompts)

    obj = joblib.load(ADVMIX / f"{args.cell}.joblib")
    scaler, svm = obj["scaler"], obj["svm"]
    scores = svm.predict_proba(scaler.transform(reps))[:, 1]

    df = pd.DataFrame({"source": "wildchat", "prompt": prompts, "probe_score": scores})
    out = ADVMIX / f"{args.cell}_wildchat.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}  median={float(np.median(scores)):.4f}  p95={float(np.quantile(scores,0.95)):.4f}  max={float(scores.max()):.4f}", flush=True)


if __name__ == "__main__":
    main()
