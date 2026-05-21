#!/usr/bin/env python3
"""Re-judge every run.json in the Apertus grid that has -1.0 strongreject scores.

The first grid pass scored everything -1.0 because the offline strongreject judge
couldn't load (google/gemma-2b base cache was missing tokenizer files). Cache is
now complete; this script reloads the judge once and rewrites the score blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adversariallm.judging import (
    build_judge,
    build_score_key,
    extract_last_user_message,
    build_with_jailbreak_conversation,
    build_without_jailbreak_conversation,
)


def is_broken(run_data: dict, score_key: str) -> bool:
    """A run is broken if any of its steps' strongreject scores include -1.0."""
    for r in run_data.get("runs", []):
        for s in r.get("steps", []):
            block = (s.get("scores") or {}).get(score_key)
            if not block:
                continue
            for vals in block.values():
                if isinstance(vals, list):
                    if any(v == -1.0 for v in vals):
                        return True
                elif vals == -1.0:
                    return True
    return False


def rejudge_one(run_path: Path, judge, score_key: str) -> tuple[int, int]:
    """Re-score a single run.json in place. Returns (steps_updated, completions_scored)."""
    data = json.loads(run_path.read_text())
    steps_updated = completions_scored = 0
    for r in data.get("runs", []):
        original_conversation = r["original_prompt"]
        harmful_prompt = extract_last_user_message(original_conversation)
        # Batch every step's completions through the judge
        per_step_counts = []
        without, with_, harmful, attack, response = [], [], [], [], []
        for step in r["steps"]:
            model_input = step["model_input"]
            attack_prompt = extract_last_user_message(model_input)
            comps = step["model_completions"]
            per_step_counts.append(len(comps))
            for completion in comps:
                without.append(build_without_jailbreak_conversation(original_conversation, model_input, completion))
                with_.append(build_with_jailbreak_conversation(model_input, completion))
                harmful.append(harmful_prompt)
                attack.append(attack_prompt)
                response.append(completion)
        if not without:
            continue
        # judge returns dict: {field_name: [scores_per_completion_flat]}
        results = judge(harmful, attack, response)
        # Distribute results back per-step
        i = 0
        for step, n in zip(r["steps"], per_step_counts):
            block = {k: list(v[i:i + n]) for k, v in results.items()}
            step.setdefault("scores", {})[score_key] = block
            i += n
            steps_updated += 1
            completions_scored += n
    # Mark scored_by at top level
    sb = data.setdefault("scored_by", [])
    if score_key not in sb:
        sb.append(score_key)
    # Atomic write
    tmp = run_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(run_path)
    return steps_updated, completions_scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Experiment root dir (contains <cell>/attacks/...)")
    p.add_argument("--classifier", default="local:strongreject")
    p.add_argument("--only-broken", action="store_true", default=True,
                   help="Only re-judge runs with -1.0 scores")
    args = p.parse_args()

    root = Path(args.root)
    score_key = build_score_key(args.classifier)
    print(f"score_key: {score_key}; root: {root}")
    judge = build_judge(args.classifier)
    print(f"judge built: {type(judge).__name__}")

    files = sorted(root.glob("*/attacks/*/*/outputs/**/run.json"))
    print(f"found {len(files)} run.json files")

    total_completions = 0
    total_steps = 0
    n_files = 0
    for i, f in enumerate(files, 1):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[{i}/{len(files)}] SKIP unreadable: {f} {e}")
            continue
        if args.only_broken and not is_broken(data, score_key):
            continue
        try:
            steps_updated, completions_scored = rejudge_one(f, judge, score_key)
        except Exception as e:
            print(f"[{i}/{len(files)}] ERROR {f}: {e}")
            continue
        n_files += 1
        total_steps += steps_updated
        total_completions += completions_scored
        if i % 25 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] files_updated={n_files} steps={total_steps} completions={total_completions}")
    print(f"DONE: files_updated={n_files} steps={total_steps} completions={total_completions}")


if __name__ == "__main__":
    main()
