#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from defenses.jepa_ce import PerPositionPredictor, _load_json_or_jsonl


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_model_and_predictor(run_dir: Path, device: str, dtype: torch.dtype):
    manifest = load_manifest(run_dir)
    base_model = manifest["base_model"]
    adapter_dir = run_dir / manifest.get("adapter_path", "lora_adapter")
    predictor_path = run_dir / manifest.get("predictor_path", "jepa_predictor.pt")
    config = manifest.get("config", {})

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else None,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    if device != "cuda":
        model.to(device)
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "d_model", None)
    if hidden_size is None:
        raise ValueError("Could not infer model hidden size.")
    predictor = PerPositionPredictor(
        int(hidden_size),
        predictor_type=config.get("predictor_type", "mlp"),
        num_layers=int(config.get("predictor_layers", 2)),
        dropout=float(config.get("predictor_dropout", 0.0)),
        bottleneck_dim=int(config.get("predictor_bottleneck_dim", 0)),
    ).to(device=model.device, dtype=dtype)
    state = torch.load(predictor_path, map_location=model.device)
    predictor.load_state_dict(state)
    predictor.eval()

    align_layer = int(config.get("align_layer", 25))
    return manifest, tokenizer, model, predictor, align_layer


def chat_prompt_ids(tokenizer, text: str, max_length: int) -> Dict[str, torch.Tensor]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_generation_prompt=True,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    attention_mask = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": attention_mask}


def batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    predictor,
    align_layer: int,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    enc_reps: List[torch.Tensor] = []
    pred_reps: List[torch.Tensor] = []
    device = model.device
    pred_param = next(predictor.parameters(), None)
    pred_dtype = pred_param.dtype if pred_param is not None else next(model.parameters()).dtype

    for batch_texts in batched(texts, batch_size):
        tokenized = [chat_prompt_ids(tokenizer, text, max_length) for text in batch_texts]
        max_len = max(item["input_ids"].shape[-1] for item in tokenized)
        input_ids = []
        attention_masks = []
        for item in tokenized:
            ids = item["input_ids"]
            mask = item["attention_mask"]
            pad = max_len - ids.shape[-1]
            input_ids.append(F.pad(ids, (0, pad), value=tokenizer.pad_token_id))
            attention_masks.append(F.pad(mask, (0, pad), value=0))
        input_ids_t = torch.cat(input_ids, dim=0).to(device)
        attention_mask_t = torch.cat(attention_masks, dim=0).to(device)

        out = model(
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            output_hidden_states=True,
            use_cache=False,
        )
        layer = align_layer if align_layer >= 0 else len(out.hidden_states) + align_layer
        if layer <= 0 or layer >= len(out.hidden_states):
            raise ValueError(f"align_layer={align_layer} is invalid for {len(out.hidden_states)} hidden states.")
        hidden = out.hidden_states[layer]
        last_idx = attention_mask_t.sum(dim=-1) - 1
        rows = torch.arange(hidden.size(0), device=device)
        s = hidden[rows, last_idx, :]
        s_pred = predictor(s.to(dtype=pred_dtype))
        enc_reps.append(s.detach().float().cpu())
        pred_reps.append(s_pred.detach().float().cpu())

    return torch.cat(enc_reps, dim=0), torch.cat(pred_reps, dim=0)


def load_cb_by_category(path: str) -> Dict[str, List[str]]:
    by_category: Dict[str, List[str]] = defaultdict(list)
    for rec in _load_json_or_jsonl(path):
        prompt = rec.get("prompt")
        category = rec.get("category") or rec.get("meta", {}).get("category") or "harmful"
        if prompt:
            by_category[str(category)].append(prompt)
    return dict(by_category)


def load_ultrachat_prompts(num_samples: int, seed: int) -> List[str]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=seed).select(range(num_samples))
    prompts: List[str] = []
    for item in ds:
        user = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        if user:
            prompts.append(user)
    return prompts


def load_reverse_jailbreak_prompts(path: str, limit: int) -> List[str]:
    prompts: List[str] = []
    for rec in _load_json_or_jsonl(path):
        for prompt in rec.get("generated_prompts") or []:
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
                if len(prompts) >= limit:
                    return prompts
    return prompts


def load_wildjailbreak_ood(
    path: str | None,
    dataset: str | None,
    config: str,
    split: str,
    limit: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not path and not dataset:
        return [], []

    if dataset:
        ds = load_dataset(dataset, config, split=split, delimiter="\t", keep_default_na=False).shuffle(seed=seed)
        records = [dict(item) for item in ds]
    else:
        suffix = Path(path).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            records = _load_json_or_jsonl(path)
        else:
            ds = load_dataset("csv", data_files=path, split="train", delimiter="\t", keep_default_na=False).shuffle(seed=seed)
            records = [dict(item) for item in ds]

    benign: List[str] = []
    harmful: List[str] = []
    for rec in records:
        prompt = rec.get("adversarial") or rec.get("vanilla") or rec.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        data_type = str(rec.get("data_type", "")).lower()
        label = rec.get("label")
        if "benign" in data_type or label == 0 or label == "0":
            benign.append(prompt.strip())
        elif "harmful" in data_type or label == 1 or label == "1":
            harmful.append(prompt.strip())
        if limit and len(benign) + len(harmful) >= limit:
            break
    return benign, harmful


def make_centroids(
    category_texts: Dict[str, List[str]],
    benign_texts: List[str],
    tokenizer,
    model,
    predictor,
    align_layer: int,
    max_length: int,
    batch_size: int,
    per_category: int,
) -> Tuple[List[str], torch.Tensor]:
    labels = ["benign"]
    centroid_rows = []
    benign_reps, _ = encode_texts(
        benign_texts[:per_category], tokenizer, model, predictor, align_layer, max_length, batch_size
    )
    centroid_rows.append(benign_reps.mean(dim=0))

    for category in sorted(category_texts):
        texts = category_texts[category][:per_category]
        if not texts:
            continue
        reps, _ = encode_texts(texts, tokenizer, model, predictor, align_layer, max_length, batch_size)
        labels.append(category)
        centroid_rows.append(reps.mean(dim=0))

    return labels, torch.stack(centroid_rows, dim=0)


def score_reps(enc_reps: torch.Tensor, pred_reps: torch.Tensor, centroids: torch.Tensor) -> Dict[str, torch.Tensor]:
    c = F.normalize(centroids.float(), dim=-1)
    enc_sims = F.normalize(enc_reps.float(), dim=-1) @ c.T
    pred_sims = F.normalize(pred_reps.float(), dim=-1) @ c.T
    enc_harm = enc_sims[:, 1:].max(dim=-1).values
    pred_harm = pred_sims[:, 1:].max(dim=-1).values
    return {
        "enc": enc_harm,
        "pred": pred_harm,
        "max": torch.maximum(enc_harm, pred_harm),
        "enc_category": enc_sims[:, 1:].argmax(dim=-1) + 1,
        "pred_category": pred_sims[:, 1:].argmax(dim=-1) + 1,
    }


def auroc(labels: List[int], scores: List[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:end])
        idx = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize_metric(name: str, benign_scores: torch.Tensor, harmful_scores: torch.Tensor) -> Dict[str, float]:
    labels = [0] * len(benign_scores) + [1] * len(harmful_scores)
    scores = [float(x) for x in benign_scores.tolist()] + [float(x) for x in harmful_scores.tolist()]
    threshold = float(torch.quantile(benign_scores, 0.99).item()) if len(benign_scores) else float("nan")
    tpr = float((harmful_scores > threshold).float().mean().item()) if len(harmful_scores) else float("nan")
    return {
        f"{name}_auroc": auroc(labels, scores),
        f"{name}_threshold_fpr_1pct": threshold,
        f"{name}_tpr_at_fpr_1pct": tpr,
        f"{name}_benign_mean": float(benign_scores.mean().item()),
        f"{name}_harmful_mean": float(harmful_scores.mean().item()),
    }


def threshold_at_fpr(benign_scores: torch.Tensor, target_fpr: float) -> float:
    if len(benign_scores) == 0:
        return float("nan")
    q = max(0.0, min(1.0, 1.0 - target_fpr))
    return float(torch.quantile(benign_scores, q).item())


def optimal_threshold_youden(benign_scores: torch.Tensor, harmful_scores: torch.Tensor) -> float:
    if len(benign_scores) == 0 or len(harmful_scores) == 0:
        return float("nan")
    candidates = torch.unique(torch.cat([benign_scores, harmful_scores])).sort().values
    best_threshold = float(candidates[0].item())
    best_score = -float("inf")
    for threshold in candidates:
        fpr = (benign_scores > threshold).float().mean().item()
        tpr = (harmful_scores > threshold).float().mean().item()
        score = tpr - fpr
        if score > best_score:
            best_score = score
            best_threshold = float(threshold.item())
    return best_threshold


def threshold_report(
    name: str,
    cal_benign: torch.Tensor,
    cal_harmful: torch.Tensor,
    test_benign: torch.Tensor,
    test_harmful: torch.Tensor,
    target_fpr: float,
) -> Dict[str, float]:
    labels = [0] * len(test_benign) + [1] * len(test_harmful)
    scores = [float(x) for x in test_benign.tolist()] + [float(x) for x in test_harmful.tolist()]
    out = {
        f"{name}_test_auroc": auroc(labels, scores),
        f"{name}_test_benign_mean": float(test_benign.mean().item()),
        f"{name}_test_harmful_mean": float(test_harmful.mean().item()),
    }
    for threshold_name, threshold in (
        ("cal_fpr", threshold_at_fpr(cal_benign, target_fpr)),
        ("cal_youden", optimal_threshold_youden(cal_benign, cal_harmful)),
    ):
        threshold_t = torch.tensor(threshold)
        out.update(
            {
                f"{name}_{threshold_name}_threshold": threshold,
                f"{name}_{threshold_name}_cal_fpr": float((cal_benign > threshold_t).float().mean().item()),
                f"{name}_{threshold_name}_cal_tpr": float((cal_harmful > threshold_t).float().mean().item()),
                f"{name}_{threshold_name}_test_fpr": float((test_benign > threshold_t).float().mean().item()),
                f"{name}_{threshold_name}_test_tpr": float((test_harmful > threshold_t).float().mean().item()),
            }
        )
    return out


def apply_calibrated_thresholds(
    name: str,
    scores: torch.Tensor,
    labels: List[int],
    cal_benign: torch.Tensor,
    cal_harmful: torch.Tensor,
    target_fpr: float,
) -> Dict[str, float]:
    if not labels or len(set(labels)) < 2:
        return {}
    benign_scores = scores[torch.tensor([label == 0 for label in labels], dtype=torch.bool)]
    harmful_scores = scores[torch.tensor([label == 1 for label in labels], dtype=torch.bool)]
    out = {
        f"{name}_auroc": auroc(labels, [float(x) for x in scores.tolist()]),
        f"{name}_benign_mean": float(benign_scores.mean().item()),
        f"{name}_harmful_mean": float(harmful_scores.mean().item()),
    }
    for threshold_name, threshold in (
        ("cal_fpr", threshold_at_fpr(cal_benign, target_fpr)),
        ("cal_youden", optimal_threshold_youden(cal_benign, cal_harmful)),
    ):
        threshold_t = torch.tensor(threshold)
        out.update(
            {
                f"{name}_{threshold_name}_threshold": threshold,
                f"{name}_{threshold_name}_fpr": float((benign_scores > threshold_t).float().mean().item()),
                f"{name}_{threshold_name}_tpr": float((harmful_scores > threshold_t).float().mean().item()),
            }
        )
    return out


def split_half(items: List[str]) -> Tuple[List[str], List[str]]:
    mid = len(items) // 2
    return items[:mid], items[mid:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate JEPA predictor/encoder centroid guardrail.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--cb_path", type=str, default="data/circuit_breakers_train.json")
    parser.add_argument("--reverse_path", type=str, default="reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl")
    parser.add_argument("--wildjailbreak_path", type=str, default=None)
    parser.add_argument("--wildjailbreak_dataset", type=str, default=None)
    parser.add_argument("--wildjailbreak_config", type=str, default="eval")
    parser.add_argument("--wildjailbreak_split", type=str, default="train")
    parser.add_argument("--wildjailbreak_limit", type=int, default=5000)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--centroid_per_category", type=int, default=32)
    parser.add_argument("--eval_per_category", type=int, default=32)
    parser.add_argument("--benign_centroid_samples", type=int, default=256)
    parser.add_argument("--benign_eval_samples", type=int, default=256)
    parser.add_argument("--jailbreak_eval_samples", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    run_dir = Path(args.run_dir)
    manifest, tokenizer, model, predictor, align_layer = load_model_and_predictor(run_dir, args.device, dtype)
    print(f"[jepa-guardrail] loaded {run_dir}; align_layer={align_layer}")

    cb_by_category = load_cb_by_category(args.cb_path)
    benign_needed = args.benign_centroid_samples + args.benign_eval_samples
    benign_prompts = load_ultrachat_prompts(benign_needed, args.seed)
    benign_centroid = benign_prompts[: args.benign_centroid_samples]
    benign_eval = benign_prompts[args.benign_centroid_samples : benign_needed]

    labels, centroids = make_centroids(
        cb_by_category,
        benign_centroid,
        tokenizer,
        model,
        predictor,
        align_layer,
        args.max_length,
        args.batch_size,
        args.centroid_per_category,
    )
    print(f"[jepa-guardrail] centroids={len(labels)} labels; first labels={labels[:5]}")

    harmful_eval: List[str] = []
    for category in sorted(cb_by_category):
        start = args.centroid_per_category
        harmful_eval.extend(cb_by_category[category][start : start + args.eval_per_category])
    jailbreak_eval = load_reverse_jailbreak_prompts(args.reverse_path, args.jailbreak_eval_samples)
    harmful_eval.extend(jailbreak_eval)
    benign_cal, benign_test = split_half(benign_eval)
    harmful_cal, harmful_test = split_half(harmful_eval)
    print(
        f"[jepa-guardrail] eval benign={len(benign_eval)} harmful_clean_plus_jailbreak={len(harmful_eval)} "
        f"(jailbreak={len(jailbreak_eval)})"
    )
    print(
        f"[jepa-guardrail] threshold calibration split: benign cal/test={len(benign_cal)}/{len(benign_test)} "
        f"harmful cal/test={len(harmful_cal)}/{len(harmful_test)} target_fpr={args.target_fpr}"
    )
    wild_benign, wild_harmful = load_wildjailbreak_ood(
        args.wildjailbreak_path,
        args.wildjailbreak_dataset,
        args.wildjailbreak_config,
        args.wildjailbreak_split,
        args.wildjailbreak_limit,
        args.seed,
    )
    if wild_benign or wild_harmful:
        print(f"[jepa-guardrail] WildJailbreak OOD benign={len(wild_benign)} harmful={len(wild_harmful)}")

    benign_cal_enc, benign_cal_pred = encode_texts(
        benign_cal, tokenizer, model, predictor, align_layer, args.max_length, args.batch_size
    )
    benign_test_enc, benign_test_pred = encode_texts(
        benign_test, tokenizer, model, predictor, align_layer, args.max_length, args.batch_size
    )
    harmful_cal_enc, harmful_cal_pred = encode_texts(
        harmful_cal, tokenizer, model, predictor, align_layer, args.max_length, args.batch_size
    )
    harmful_test_enc, harmful_test_pred = encode_texts(
        harmful_test, tokenizer, model, predictor, align_layer, args.max_length, args.batch_size
    )
    benign_cal_scores = score_reps(benign_cal_enc, benign_cal_pred, centroids)
    benign_test_scores = score_reps(benign_test_enc, benign_test_pred, centroids)
    harmful_cal_scores = score_reps(harmful_cal_enc, harmful_cal_pred, centroids)
    harmful_test_scores = score_reps(harmful_test_enc, harmful_test_pred, centroids)

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "base_model": manifest["base_model"],
        "align_layer": align_layer,
        "centroid_labels": labels,
        "counts": {
            "benign_eval": len(benign_eval),
            "harmful_eval": len(harmful_eval),
            "jailbreak_eval": len(jailbreak_eval),
            "benign_cal": len(benign_cal),
            "benign_test": len(benign_test),
            "harmful_cal": len(harmful_cal),
            "harmful_test": len(harmful_test),
        },
    }
    for key in ("enc", "pred", "max"):
        summary.update(
            summarize_metric(
                f"{key}_all",
                torch.cat([benign_cal_scores[key], benign_test_scores[key]]),
                torch.cat([harmful_cal_scores[key], harmful_test_scores[key]]),
            )
        )
        summary.update(
            threshold_report(
                key,
                benign_cal_scores[key],
                harmful_cal_scores[key],
                benign_test_scores[key],
                harmful_test_scores[key],
                args.target_fpr,
            )
        )

    if wild_benign and wild_harmful:
        wild_texts = wild_benign + wild_harmful
        wild_labels = [0] * len(wild_benign) + [1] * len(wild_harmful)
        wild_enc, wild_pred = encode_texts(
            wild_texts, tokenizer, model, predictor, align_layer, args.max_length, args.batch_size
        )
        wild_scores = score_reps(wild_enc, wild_pred, centroids)
        summary["wildjailbreak_counts"] = {"benign": len(wild_benign), "harmful": len(wild_harmful)}
        for key in ("enc", "pred", "max"):
            summary.update(
                apply_calibrated_thresholds(
                    f"wild_{key}",
                    wild_scores[key],
                    wild_labels,
                    benign_cal_scores[key],
                    harmful_cal_scores[key],
                    args.target_fpr,
                )
            )

    for key in ("enc", "pred", "max"):
        print(
            f"{key:>4} test_AUROC={summary[f'{key}_test_auroc']:.4f} "
            f"calFPR-thresh={summary[f'{key}_cal_fpr_threshold']:.4f} "
            f"test FPR/TPR={summary[f'{key}_cal_fpr_test_fpr']:.4f}/{summary[f'{key}_cal_fpr_test_tpr']:.4f} "
            f"youden-thresh={summary[f'{key}_cal_youden_threshold']:.4f} "
            f"test FPR/TPR={summary[f'{key}_cal_youden_test_fpr']:.4f}/{summary[f'{key}_cal_youden_test_tpr']:.4f}"
        )
    if wild_benign and wild_harmful:
        for key in ("enc", "pred", "max"):
            prefix = f"wild_{key}"
            print(
                f"{prefix:>8} AUROC={summary[f'{prefix}_auroc']:.4f} "
                f"calFPR-thresh FPR/TPR={summary[f'{prefix}_cal_fpr_fpr']:.4f}/{summary[f'{prefix}_cal_fpr_tpr']:.4f} "
                f"youden FPR/TPR={summary[f'{prefix}_cal_youden_fpr']:.4f}/{summary[f'{prefix}_cal_youden_tpr']:.4f}"
            )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"[jepa-guardrail] wrote {out_path}")


if __name__ == "__main__":
    main()
