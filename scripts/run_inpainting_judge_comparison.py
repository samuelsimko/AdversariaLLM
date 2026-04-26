#!/usr/bin/env python3
"""Run an inpainting attack and compare judge outputs.

This script stays aligned with the repo's normal execution path by calling
`run_attacks.py`, which in turn logs runs and invokes the configured judges.

It then scans the newly created `run.json` files and writes:

- a per-completion CSV with prompts, completions, and judge outputs
- a summary CSV with context-difference stats for local judges
- a summary JSON with a compact machine-readable report

Default target:
- attack: `inpainting`
- dataset: `jbb_behaviors`
- model: `meta-llama/Meta-Llama-3-8B-Instruct`
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLASSIFIERS = [
    "strong_reject",
    "harmbench",
    "local:strongreject",
    "local:harmbench",
    "local:gpt_oss",
]
DEFAULT_TOPK_APPLY_TO = [
    "harmbench",
    "local:strongreject",
    "local:harmbench",
    "local:gpt_oss",
]

PRIMARY_METRIC_CANDIDATES = (
    "score",
    "p_harmful",
    "score_attack_context",
    "score_original_request_context",
    "score_with_jailbreak",
    "score_without_jailbreak",
)


def metric_key_variants(base: str, canonical_suffix: str, legacy_suffix: str) -> list[str]:
    return [f"{base}.{canonical_suffix}", f"{base}.{legacy_suffix}"]


def first_present_key(row: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        if key in row:
            return key
    return None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model key from conf/models/models.yaml",
    )
    parser.add_argument("--dataset", default="jbb_behaviors")
    parser.add_argument("--attack", default="inpainting")
    parser.add_argument(
        "--idx",
        default="range(0,10)",
        help='Dataset slice expression, e.g. "range(0,10)" or "[0,1,2]"',
    )
    parser.add_argument(
        "--num-samples-per-behavior",
        type=int,
        default=64,
        help="Number of inpainting variants to keep per behavior.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=DEFAULT_CLASSIFIERS,
        help="Judge specs. Plain names use judgezoo. local:* uses judges.py.",
    )
    parser.add_argument(
        "--judge-top-k",
        type=int,
        default=1,
        help="If set, run non-prescore judges only on the top K completions ranked by the prescore judge.",
    )
    parser.add_argument(
        "--prescore-classifier",
        default="strong_reject",
        help="Judge used to prescore and rank completions before expensive judges run.",
    )
    parser.add_argument(
        "--prescore-score-field",
        default="p_harmful",
        help="Metric field to rank by from the prescore judge.",
    )
    parser.add_argument(
        "--judge-top-k-apply-to",
        nargs="+",
        default=DEFAULT_TOPK_APPLY_TO,
        help="Judges that should use the top-K filter. By default this excludes the prescore judge itself.",
    )
    parser.add_argument(
        "--hydra-launcher",
        default="basic",
        help="Hydra launcher override. Use 'basic' for local runs or a repo launcher like 'a100h100.yaml' for Slurm.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=repo_root / "scripts" / "artifacts",
    )
    parser.add_argument(
        "--prefix",
        default="inpainting_judge_compare",
        help="Prefix for output artifact filenames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated command and exit without running it.",
    )
    return parser.parse_args()


def last_user_message(conversation: list[dict[str, Any]] | None) -> str:
    if not conversation:
        return ""
    for message in reversed(conversation):
        if message.get("role") == "user":
            return message.get("content", "")
    return conversation[-1].get("content", "")


def build_run_attacks_command(args: argparse.Namespace) -> list[str]:
    dataset_idx_key = f"datasets.{args.dataset}.idx={args.idx}"
    classifier_override = "classifiers=" + json.dumps(args.classifiers)
    command = [
        args.python,
        "run_attacks.py",
        "--multirun",
        f"hydra/launcher={args.hydra_launcher}",
        f"model={args.model}",
        f"dataset={args.dataset}",
        f"attack={args.attack}",
        dataset_idx_key,
        f"attacks.{args.attack}.num_samples_per_behavior={args.num_samples_per_behavior}",
        classifier_override,
    ]
    if args.judge_top_k is not None:
        command.extend(
            [
                "judge_selection.enabled=true",
                f"judge_selection.prescore_classifier={args.prescore_classifier}",
                f"judge_selection.score_field={args.prescore_score_field}",
                f"judge_selection.top_k={args.judge_top_k}",
                "judge_selection.apply_to=" + json.dumps(args.judge_top_k_apply_to),
            ]
        )
    return command


def find_new_run_files(outputs_dir: Path, started_at: float) -> list[Path]:
    run_files = []
    if not outputs_dir.exists():
        return run_files
    for path in outputs_dir.glob("**/run.json"):
        try:
            if path.stat().st_mtime >= started_at - 1.0:
                run_files.append(path.resolve())
        except FileNotFoundError:
            continue
    return sorted(run_files)


def pick_primary_metric(score_dict: dict[str, Any]) -> tuple[str | None, Any]:
    for key in PRIMARY_METRIC_CANDIDATES:
        if key in score_dict:
            return key, score_dict[key]
    for key, value in score_dict.items():
        if isinstance(value, list) and value:
            return key, value
    return None, None


def flatten_runs(run_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_file in run_files:
        payload = json.loads(run_file.read_text())
        run_config = payload.get("config", {})
        for subrun_index, subrun in enumerate(payload.get("runs", [])):
            harmful_prompt = last_user_message(subrun.get("original_prompt"))
            for step in subrun.get("steps", []):
                attack_prompt = last_user_message(step.get("model_input"))
                prompt_changed = harmful_prompt != attack_prompt
                completions = step.get("model_completions", [])
                for completion_index, completion in enumerate(completions):
                    row = {
                        "run_file": str(run_file),
                        "model": run_config.get("model"),
                        "dataset": run_config.get("dataset"),
                        "attack": run_config.get("attack"),
                        "dataset_idx": run_config.get("dataset_params", {}).get("idx"),
                        "subrun_index": subrun_index,
                        "step": step.get("step"),
                        "completion_index": completion_index,
                        "harmful_prompt": harmful_prompt,
                        "attack_prompt": attack_prompt,
                        "prompt_changed": prompt_changed,
                        "completion": completion,
                    }

                    for score_name, score_dict in step.get("scores", {}).items():
                        for metric_name, metric_values in score_dict.items():
                            if isinstance(metric_values, list) and completion_index < len(metric_values):
                                row[f"{score_name}.{metric_name}"] = metric_values[completion_index]
                        primary_key, primary_value = pick_primary_metric(score_dict)
                        if (
                            primary_key is not None
                            and isinstance(primary_value, list)
                            and completion_index < len(primary_value)
                        ):
                            row[f"{score_name}.__primary_metric"] = primary_key
                            row[f"{score_name}.__primary_value"] = primary_value[completion_index]
                    rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_rows": len(rows),
        "n_prompt_changed": sum(bool(row.get("prompt_changed")) for row in rows),
        "local_context_comparison": {},
        "backend_comparison": {},
    }

    local_judges = sorted(
        {
            key.rsplit(".", 1)[0]
            for row in rows
            for key in row
            if (key.endswith(".score_attack_context") or key.endswith(".score_with_jailbreak")) and key.startswith("local:")
        }
    )
    for judge_name in local_judges:
        paired = []
        for row in rows:
            with_key = first_present_key(
                row,
                metric_key_variants(judge_name, "score_attack_context", "score_with_jailbreak"),
            )
            without_key = first_present_key(
                row,
                metric_key_variants(judge_name, "score_original_request_context", "score_without_jailbreak"),
            )
            if with_key and without_key:
                paired.append((row[without_key], row[with_key]))
        if not paired:
            continue
        equal = sum(1 for a, b in paired if a == b)
        abs_diffs = [abs(float(b) - float(a)) for a, b in paired]
        summary["local_context_comparison"][judge_name] = {
            "count": len(paired),
            "equal_count": equal,
            "equal_rate": equal / len(paired),
            "mean_without_jailbreak": sum(float(a) for a, _ in paired) / len(paired),
            "mean_with_jailbreak": sum(float(b) for _, b in paired) / len(paired),
            "mean_abs_diff": sum(abs_diffs) / len(abs_diffs),
        }

    backend_pairs = [
        ("strong_reject", "local:strongreject"),
        ("harmbench", "local:harmbench"),
    ]
    for baseline_name, local_name in backend_pairs:
        baseline_key = f"{baseline_name}.__primary_value"
        paired = []
        for row in rows:
            local_without_key = first_present_key(
                row,
                metric_key_variants(local_name, "score_original_request_context", "score_without_jailbreak"),
            )
            local_with_key = first_present_key(
                row,
                metric_key_variants(local_name, "score_attack_context", "score_with_jailbreak"),
            )
            if baseline_key in row and local_without_key and local_with_key:
                paired.append((row[baseline_key], row[local_without_key], row[local_with_key]))
        if not paired:
            continue
        summary["backend_comparison"][baseline_name] = {
            "count": len(paired),
            "mean_abs_diff_vs_local_without": sum(abs(float(base) - float(local_wo)) for base, local_wo, _ in paired)
            / len(paired),
            "mean_abs_diff_vs_local_with": sum(abs(float(base) - float(local_w)) for base, _, local_w in paired)
            / len(paired),
        }
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    fieldnames = [
        "judge",
        "count",
        "equal_count",
        "equal_rate",
        "mean_without_jailbreak",
        "mean_with_jailbreak",
        "mean_abs_diff",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for judge_name, stats in summary.get("local_context_comparison", {}).items():
            writer.writerow({"judge": judge_name, **stats})


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    command = build_run_attacks_command(args)
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    completed = subprocess.run(command, cwd=repo_root, env=env)
    if completed.returncode != 0:
        return completed.returncode

    run_files = find_new_run_files(args.outputs_dir.resolve(), started_at)
    if not run_files:
        print("No new run.json files found after execution.", file=sys.stderr)
        return 1

    rows = flatten_runs(run_files)
    summary = summarize_rows(rows)
    summary["run_files"] = [str(path) for path in run_files]
    summary["classifiers"] = args.classifiers
    summary["judge_selection"] = {
        "enabled": args.judge_top_k is not None,
        "prescore_classifier": args.prescore_classifier,
        "score_field": args.prescore_score_field,
        "top_k": args.judge_top_k,
        "apply_to": args.judge_top_k_apply_to,
    }
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{args.prefix}_{timestamp}"
    rows_csv = artifacts_dir / f"{stem}_rows.csv"
    summary_csv = artifacts_dir / f"{stem}_summary.csv"
    summary_json = artifacts_dir / f"{stem}_summary.json"

    write_rows_csv(rows_csv, rows)
    write_summary_csv(summary_csv, summary)
    summary_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote rows CSV: {rows_csv}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary JSON: {summary_json}")
    print(f"New run files: {len(run_files)}")
    print(f"Prompt changed rows: {summary['n_prompt_changed']} / {summary['n_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
