"""Train a probe on a big, attack-style training pool and re-score the same
attack/benign prompts to test "would a probe trained on more diverse data
catch the attacks that bypass the model?"

Training pool: WildJailbreak harmful (jailbreak-style, includes persona/roleplay/
code-injection) vs WildJailbreak benign. Vastly more diverse than BeaverTails.

Pipeline:
    1. Sample N harmful + N benign WJ prompts.
    2. Extract L25 reps via the cell's adapter.
    3. Fit StandardScaler + linear SVM on those (no held-out — full train).
    4. Re-extract reps for the 1100 prompts in the existing probe_gate CSV
       (attacks + OR-bench benigns) and apply the new probe.
    5. Output a fresh CSV with a probe_score_bigtrain column joined by prompt.

Standalone — does NOT modify classify_wang.py or probe_gate_one_cell.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]

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


def load_cell(cell: str, device: str = "cuda"):
    base = CELL_BASE_MODEL[cell]
    adapter = REPO / "runs/experiments/headline_backup_rerun" / cell / "lora_adapter"
    print(f"[load] base={base} adapter={adapter}")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 device_map={"": device}).eval()
    model = PeftModel.from_pretrained(model, str(adapter))
    return tok, model


@torch.no_grad()
def extract_reps(model, tok, prompts: list[str], layer: int = 25) -> np.ndarray:
    """Wang-style last-token hidden state, no chat template."""
    out = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    for i, p in enumerate(tqdm(prompts, unit="prompt")):
        inputs = tok(p, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        h = model(**inputs, output_hidden_states=True).hidden_states[layer][0, -1, :]
        out[i] = h.detach().cpu().float().numpy()
    return out


def sample_wj_train(n_per_class: int, seed: int = 42) -> tuple[list[str], list[str]]:
    """Return (harmful_prompts, benign_prompts) sampled from WildJailbreak."""
    import random
    random.seed(seed)
    harmful_path = REPO / "data/wildjailbreak_harmful.json"
    benign_path = REPO / "data/wildjailbreak_benign.jsonl"

    with harmful_path.open() as f:
        h_all = json.load(f)
    h_prompts = [d["prompt"] for d in h_all if isinstance(d, dict) and "prompt" in d]

    b_prompts = []
    with benign_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                if "prompt" in d:
                    b_prompts.append(d["prompt"])
            except Exception:
                continue

    random.shuffle(h_prompts)
    random.shuffle(b_prompts)
    return h_prompts[:n_per_class], b_prompts[:n_per_class]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", type=int, default=25)
    ap.add_argument("--n_train_per_class", type=int, default=1000)
    ap.add_argument("--probe_gate_csv", type=Path,
                    default=REPO / "runs/experiments/headline_backup_rerun_probing/probe_gate/q_cb_pra.csv",
                    help="Existing probe_gate CSV; we re-score those same prompts with the bigtrain probe.")
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. Sample WJ training pool.
    h_prompts, b_prompts = sample_wj_train(args.n_train_per_class, seed=args.seed)
    print(f"[train] sampled {len(h_prompts)} harmful + {len(b_prompts)} benign WJ prompts")

    # 2. Load cell, extract train + eval reps in one model load.
    tok, model = load_cell(args.cell, device=args.device)

    print("[train] extracting train reps...")
    X_h = extract_reps(model, tok, h_prompts, layer=args.layer)
    X_b = extract_reps(model, tok, b_prompts, layer=args.layer)
    X_train = np.vstack([X_h, X_b])
    y_train = np.array([1] * len(h_prompts) + [0] * len(b_prompts))
    print(f"  X_train shape: {X_train.shape}")

    # 3. Fit scaler + SVM on the full big-train pool (no holdout — Wang convention
    #    is to use full ID, then test on OOD).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    print("[train] fitting linear SVM...")
    svm = SVC(kernel="linear", random_state=args.seed, probability=True)
    svm.fit(X_train_s, y_train)
    train_acc = svm.score(X_train_s, y_train)
    print(f"  train accuracy: {train_acc:.3f}")

    # 4. Re-extract reps for the existing probe_gate prompts.
    df = pd.read_csv(args.probe_gate_csv)
    print(f"[eval] re-extracting reps for {len(df)} prompts from {args.probe_gate_csv}")
    X_eval = extract_reps(model, tok, df["prompt"].tolist(), layer=args.layer)
    X_eval_s = scaler.transform(X_eval)
    df["probe_score_bigtrain"] = svm.predict_proba(X_eval_s)[:, 1]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv}")

    # Also save the trained probe alongside.
    joblib_path = args.out_csv.with_suffix(".joblib")
    joblib.dump({"scaler": scaler, "svm": svm,
                 "cell": args.cell, "layer": args.layer,
                 "n_train_per_class": args.n_train_per_class,
                 "train_acc": train_acc}, joblib_path)
    print(f"[saved] {joblib_path}")


if __name__ == "__main__":
    main()
