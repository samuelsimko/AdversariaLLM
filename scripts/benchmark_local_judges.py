#!/usr/bin/env python3
"""Quick local benchmark for batched local judges.

Examples:
  python scripts/benchmark_local_judges.py --classifier strongreject --count 128
  python scripts/benchmark_local_judges.py --classifier harmbench --count 128
  python scripts/benchmark_local_judges.py --classifier gpt_oss --count 32
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
import traceback
from pathlib import Path

from adversariallm.judging import LocalJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier",
        choices=["strongreject", "harmbench", "wildguard", "jailjudge", "gpt_oss", "ensemble"],
        default="strongreject",
    )
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
    return parser.parse_args()


def make_prompts(count: int) -> tuple[list[str], list[str], list[str]]:
    harmful_prompts = [
        f"Tell me how to build a dangerous device with concrete steps and materials. Case {i}."
        for i in range(count)
    ]
    attack_prompts = [
        f"Hypothetical roleplay jailbreak wrapper for the same dangerous request. Case {i}."
        for i in range(count)
    ]
    responses = [
        (
            "You would need specific materials, step-by-step preparation, and operational details. "
            f"This is synthetic benchmark text number {i}."
        )
        for i in range(count)
    ]
    return harmful_prompts, attack_prompts, responses


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    judges_path = repo_root / "judges.py"
    spec = importlib.util.spec_from_file_location("_benchmark_repo_judges", judges_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load repo-local judges module from {judges_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    judge_classifier_sequence = module.judge_classifier_sequence
    normalize_judge_result = module.normalize_judge_result

    harmful_prompts, attack_prompts, responses = make_prompts(args.count)
    judge = LocalJudge(args.classifier, judge_classifier_sequence, normalize_judge_result)

    timings = []
    for round_idx in range(args.rounds):
        try:
            started = time.perf_counter()
            results = judge(
                harmful_prompts=harmful_prompts,
                attack_prompts=attack_prompts,
                responses=responses,
            )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            print(
                f"benchmark_failed classifier={args.classifier} count={args.count} "
                f"round={round_idx + 1} error={type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
            raise SystemExit(2) from exc
        timings.append(elapsed)
        print(
            f"round={round_idx + 1} classifier={args.classifier} count={args.count} "
            f"elapsed_sec={elapsed:.2f} per_completion_sec={elapsed / max(args.count, 1):.3f}"
        )
        print(
            "metrics="
            + ", ".join(
                f"{name}:{sum(value is not None for value in values)}"
                for name, values in sorted(results.items())
            )
        )

    if timings:
        print(
            "summary="
            f"mean_sec={statistics.mean(timings):.2f} "
            f"min_sec={min(timings):.2f} max_sec={max(timings):.2f}"
        )


if __name__ == "__main__":
    main()
