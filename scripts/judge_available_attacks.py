#!/usr/bin/env python3
"""Judge all available attack run files under a directory.

Examples:
    python scripts/judge_available_attacks.py runs/experiments/velocity_inpainting_dual_family \
        --classifiers local:harmbench local:strongreject

    python scripts/judge_available_attacks.py \
        runs/experiments/velocity_inpainting_dual_family/velocity_llama3/attacks \
        --classifiers strong_reject local:wildguard --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import filelock

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adversariallm.io_utils import CompactJSONEncoder  # noqa: E402
from adversariallm.judging import (  # noqa: E402
    build_judge,
    build_judgezoo_dual_results,
    build_score_key,
    build_with_jailbreak_conversation,
    build_without_jailbreak_conversation,
    extract_last_user_message,
    parse_judge_spec,
)


@dataclass
class CompletionRef:
    path: Path
    subrun_idx: int
    step_idx: int
    completion_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="Experiment directory, attack directory, outputs directory, or a single run.json file.",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        required=True,
        help=(
            "Judge specs to apply. Examples: strong_reject local:harmbench "
            "local:strongreject local:wildguard local:jailjudge local:gpt_oss"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute scores even if a classifier is already present in the run file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of run.json files to process.",
    )
    parser.add_argument(
        "--completion-batch-size",
        type=int,
        default=128,
        help="Number of completions to judge together before stitching scores back into payloads.",
    )
    return parser.parse_args()


def find_run_files(target: Path) -> list[Path]:
    target = target.expanduser().resolve()
    if target.is_file():
        if target.name != "run.json":
            raise ValueError(f"Expected a run.json file, got {target}")
        return [target]
    if not target.exists():
        raise FileNotFoundError(target)
    return sorted(p.resolve() for p in target.glob("**/run.json"))


def ensure_scores_dict(step: dict[str, Any]) -> dict[str, Any]:
    scores = step.get("scores")
    if not isinstance(scores, dict):
        scores = {}
        step["scores"] = scores
    return scores


def classifier_present(subrun: dict[str, Any], score_key: str) -> bool:
    steps = subrun.get("steps", [])
    if not steps:
        return False
    return any(score_key in step.get("scores", {}) for step in steps)


def load_payloads(run_files: list[Path]) -> dict[Path, dict[str, Any]]:
    payloads: dict[Path, dict[str, Any]] = {}
    for path in run_files:
        with filelock.FileLock(str(path) + ".lock"):
            payloads[path] = json.loads(path.read_text())
    return payloads


def collect_pending_completions(
    payloads: dict[Path, dict[str, Any]],
    classifier: str,
    overwrite: bool,
) -> tuple[list[CompletionRef], list[list[dict[str, Any]]], list[list[dict[str, Any]]], list[str], list[str], list[str]]:
    score_key = build_score_key(classifier)
    refs: list[CompletionRef] = []
    without_jailbreak_prompts: list[list[dict[str, Any]]] = []
    with_jailbreak_prompts: list[list[dict[str, Any]]] = []
    harmful_prompts: list[str] = []
    attack_prompts: list[str] = []
    responses: list[str] = []

    for path, payload in payloads.items():
        for subrun_idx, subrun in enumerate(payload.get("runs", [])):
            if classifier_present(subrun, score_key) and not overwrite:
                continue
            original_conversation = subrun["original_prompt"]
            harmful_prompt = extract_last_user_message(original_conversation)
            for step_idx, step in enumerate(subrun["steps"]):
                model_input = step["model_input"]
                attack_prompt = extract_last_user_message(model_input)
                for completion_idx, completion in enumerate(step["model_completions"]):
                    refs.append(
                        CompletionRef(
                            path=path,
                            subrun_idx=subrun_idx,
                            step_idx=step_idx,
                            completion_idx=completion_idx,
                        )
                    )
                    without_jailbreak_prompts.append(
                        build_without_jailbreak_conversation(original_conversation, model_input, completion)
                    )
                    with_jailbreak_prompts.append(build_with_jailbreak_conversation(model_input, completion))
                    harmful_prompts.append(harmful_prompt)
                    attack_prompts.append(attack_prompt)
                    responses.append(completion)

    return (
        refs,
        without_jailbreak_prompts,
        with_jailbreak_prompts,
        harmful_prompts,
        attack_prompts,
        responses,
    )


def run_classifier_batch(
    classifier: str,
    judge: Any,
    without_jailbreak_prompts: list[list[dict[str, Any]]],
    with_jailbreak_prompts: list[list[dict[str, Any]]],
    harmful_prompts: list[str],
    attack_prompts: list[str],
    responses: list[str],
) -> dict[str, list[Any]]:
    backend, _ = parse_judge_spec(classifier)
    if backend == "judgezoo":
        without_results = judge(without_jailbreak_prompts)
        with_results = judge(with_jailbreak_prompts)
        return build_judgezoo_dual_results(without_results, with_results)
    return judge(harmful_prompts, attack_prompts, responses)


def stitch_results(
    payloads: dict[Path, dict[str, Any]],
    refs: list[CompletionRef],
    score_key: str,
    results: dict[str, list[Any]],
) -> int:
    if not refs:
        return 0

    metric_names = list(results.keys())
    written = 0
    for metric in metric_names:
        if len(results[metric]) != len(refs):
            raise ValueError(
                f"Classifier '{score_key}' returned {len(results[metric])} values for '{metric}', expected {len(refs)}"
            )

    for idx, ref in enumerate(refs):
        payload = payloads[ref.path]
        step = payload["runs"][ref.subrun_idx]["steps"][ref.step_idx]
        completions = step["model_completions"]
        score_block = ensure_scores_dict(step).setdefault(
            score_key,
            {metric: [None] * len(completions) for metric in metric_names},
        )
        for metric in metric_names:
            if metric not in score_block or len(score_block[metric]) != len(completions):
                score_block[metric] = [None] * len(completions)
            score_block[metric][ref.completion_idx] = results[metric][idx]
        if metric_names and results[metric_names[0]][idx] is not None:
            written += 1

    return written


def write_modified_payloads(payloads: dict[Path, dict[str, Any]], original_texts: dict[Path, str]) -> int:
    touched_files = 0
    for path, payload in payloads.items():
        new_text = json.dumps(payload, indent=2, cls=CompactJSONEncoder)
        if new_text == original_texts[path]:
            continue
        with filelock.FileLock(str(path) + ".lock"):
            path.write_text(new_text)
        touched_files += 1
    return touched_files


def main() -> None:
    args = parse_args()
    run_files = find_run_files(args.target)
    if args.limit is not None:
        run_files = run_files[: args.limit]

    if not run_files:
        print("No run.json files found.")
        return

    payloads = load_payloads(run_files)
    original_texts = {path: json.dumps(payload, indent=2, cls=CompactJSONEncoder) for path, payload in payloads.items()}
    judges = {classifier: build_judge(classifier) for classifier in args.classifiers}
    totals = {build_score_key(classifier): 0 for classifier in args.classifiers}

    print(f"Target: {args.target.resolve()}")
    print(f"Run files: {len(run_files)}")
    print(f"Classifiers: {', '.join(args.classifiers)}")
    print(f"Completion batch size: {args.completion_batch_size}")

    for classifier in args.classifiers:
        score_key = build_score_key(classifier)
        (
            refs,
            without_jailbreak_prompts,
            with_jailbreak_prompts,
            harmful_prompts,
            attack_prompts,
            responses,
        ) = collect_pending_completions(payloads, classifier, args.overwrite)

        print(f"{score_key}: pending completions={len(refs)}")
        if not refs:
            continue

        judge = judges[classifier]
        batch_size = max(1, args.completion_batch_size)
        for start in range(0, len(refs), batch_size):
            end = min(start + batch_size, len(refs))
            print(f"  scoring completions {start + 1}-{end}")
            batch_results = run_classifier_batch(
                classifier=classifier,
                judge=judge,
                without_jailbreak_prompts=without_jailbreak_prompts[start:end],
                with_jailbreak_prompts=with_jailbreak_prompts[start:end],
                harmful_prompts=harmful_prompts[start:end],
                attack_prompts=attack_prompts[start:end],
                responses=responses[start:end],
            )
            totals[score_key] += stitch_results(
                payloads=payloads,
                refs=refs[start:end],
                score_key=score_key,
                results=batch_results,
            )

    touched_files = write_modified_payloads(payloads, original_texts)

    print("Summary:")
    print(f"  run_files_processed: {len(run_files)}")
    print(f"  run_files_with_new_scores: {touched_files}")
    for score_key, count in totals.items():
        print(f"  {score_key}: {count}")


if __name__ == "__main__":
    main()
