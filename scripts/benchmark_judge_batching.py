#!/usr/bin/env python3
"""Benchmark judge batch sizes and throughput on real completions.

This script samples prompt/response pairs from `data/circuit_breakers_train.json`
and benchmarks the local judge models with different quantization settings.

For each judge + quantization pair it:
- loads the model once
- finds the largest batch size that fits with exponential search + binary search
- measures throughput across a sweep of successful batch sizes
- writes a JSON report with recommended batch sizes

Example:
  python scripts/benchmark_judge_batching.py \
      --judges harmbench wildguard jailjudge gpt_oss \
      --quantizations bf16 int8 int4 \
      --num-samples 64 \
      --max-batch-size 128
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from fastchat.model import get_conversation_template
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adversariallm.io_utils import free_vram
from adversariallm.io_utils.model_loading import build_causallm_load_kwargs

_JUDGES_SPEC = importlib.util.spec_from_file_location("repo_local_judges", REPO_ROOT / "judges.py")
if _JUDGES_SPEC is None or _JUDGES_SPEC.loader is None:
    raise ImportError(f"Could not load repo judges module from {REPO_ROOT / 'judges.py'}")
_JUDGES_MODULE = importlib.util.module_from_spec(_JUDGES_SPEC)
_JUDGES_SPEC.loader.exec_module(_JUDGES_MODULE)

GPT_OSS_FINAL_SYSTEM_PROMPT = _JUDGES_MODULE.GPT_OSS_FINAL_SYSTEM_PROMPT
GPT_OSS_MODEL_ID = _JUDGES_MODULE.GPT_OSS_MODEL_ID
HB_MODEL_ID = _JUDGES_MODULE.HB_MODEL_ID
JAILJUDGE_MODEL = _JUDGES_MODULE.JAILJUDGE_MODEL
LLAMA2_CLS_PROMPT = _JUDGES_MODULE.LLAMA2_CLS_PROMPT
WILDGUARD_INSTRUCTION = _JUDGES_MODULE.WILDGUARD_INSTRUCTION
WILDGUARD_MODEL_ID = _JUDGES_MODULE.WILDGUARD_MODEL_ID
extract_oss_json = _JUDGES_MODULE.extract_oss_json
get_judge_system_prompt = _JUDGES_MODULE.get_judge_system_prompt


DEFAULT_JUDGES = ["harmbench", "wildguard", "jailjudge", "gpt_oss"]
DEFAULT_QUANTIZATIONS = ["bf16", "int8", "int4"]
SUPPORTED_QUANTIZATIONS = {
    "harmbench": {"bf16", "int8", "int4"},
    "wildguard": {"bf16", "int8", "int4"},
    "jailjudge": {"bf16", "int8", "int4"},
    "gpt_oss": {"bf16", "int8", "int4"},
}


@dataclass
class Sample:
    category: str
    prompt: str
    response: str


@dataclass
class BatchMeasurement:
    batch_size: int
    examples_per_second: float
    seconds_total: float
    num_examples: int
    peak_vram_gb: float | None


@dataclass
class BenchmarkResult:
    judge: str
    quantization: str
    model_id: str
    num_samples: int
    max_fit_batch_size: int | None
    recommended_stable_batch_size: int | None
    best_throughput_batch_size: int | None
    best_examples_per_second: float | None
    measurements: list[BatchMeasurement]
    load_error: str | None = None
    benchmark_error: str | None = None


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=repo_root / "data" / "circuit_breakers_train.json",
    )
    parser.add_argument(
        "--response-field",
        default="output",
        help="Field from the dataset to benchmark as the model response.",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=DEFAULT_JUDGES,
        choices=sorted(SUPPORTED_QUANTIZATIONS),
    )
    parser.add_argument(
        "--quantizations",
        nargs="+",
        default=DEFAULT_QUANTIZATIONS,
        choices=["bf16", "int8", "int4"],
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sampled rows before taking the first N.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=repo_root / "scripts" / "artifacts",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit path for the report JSON.",
    )
    return parser.parse_args()


def load_samples(dataset_path: Path, response_field: str, num_samples: int, seed: int, shuffle: bool) -> list[Sample]:
    payload = json.loads(dataset_path.read_text())
    rows: list[Sample] = []
    for item in payload:
        prompt = str(item.get("prompt", "")).strip()
        response = str(item.get(response_field, "")).strip()
        if not prompt or not response:
            continue
        rows.append(
            Sample(
                category=str(item.get("category", "")).strip(),
                prompt=prompt,
                response=response,
            )
        )
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if not rows:
        raise ValueError(f"No usable rows found in {dataset_path} for response field '{response_field}'")
    return rows[:num_samples]


def repeat_to_length(items: list[Any], target_len: int) -> list[Any]:
    if not items:
        raise ValueError("Cannot repeat an empty list")
    return [items[i % len(items)] for i in range(target_len)]


def next_power_of_two(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def floor_power_of_two(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def model_id_for_judge(judge: str) -> str:
    if judge == "harmbench":
        return HB_MODEL_ID
    if judge == "wildguard":
        return WILDGUARD_MODEL_ID
    if judge == "jailjudge":
        return JAILJUDGE_MODEL
    if judge == "gpt_oss":
        return GPT_OSS_MODEL_ID
    raise ValueError(f"Unsupported judge: {judge}")

def load_model_and_tokenizer_for_benchmark(judge: str, quantization: str):
    model_id = model_id_for_judge(judge)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        truncation_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs = build_causallm_load_kwargs(
        model_id,
        dtype=quantization,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).eval()
    return tokenizer, model


def build_batch_prompts(judge: str, batch: list[Sample], tokenizer) -> list[str]:
    if judge == "harmbench":
        return [
            LLAMA2_CLS_PROMPT["prompt"].format(
                behavior=sample.prompt,
                generation=sample.response,
            )
            for sample in batch
        ]

    if judge == "wildguard":
        return [
            WILDGUARD_INSTRUCTION.format(
                prompt=sample.prompt,
                response=sample.response,
            )
            for sample in batch
        ]

    if judge == "jailjudge":
        system_prompt = get_judge_system_prompt()
        full_prompts = []
        for sample in batch:
            conv = get_conversation_template("llama-2")
            conv.set_system_message(system_prompt)
            conv.append_message(
                conv.roles[0],
                f"[PROMPT]:{sample.prompt}\n[TARGET MODEL'S RESPONSE]:{sample.response}",
            )
            conv.append_message(conv.roles[1], None)
            full_prompts.append(conv.get_prompt())
        return full_prompts

    if judge == "gpt_oss":
        return [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": GPT_OSS_FINAL_SYSTEM_PROMPT + "\n\nReasoning: high"},
                    {
                        "role": "user",
                        "content": f"HUMAN PROMPT:\n{sample.prompt}\n\nAI ASSISTANT RESPONSE:\n{sample.response}\n",
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sample in batch
        ]

    raise ValueError(f"Unsupported judge: {judge}")


def generation_params_for_judge(judge: str) -> dict[str, Any]:
    params = {
        "do_sample": False,
        "temperature": 0.0,
    }
    if judge == "harmbench":
        params["max_new_tokens"] = 10
    elif judge == "wildguard":
        params["max_new_tokens"] = 32
    elif judge == "jailjudge":
        params["max_new_tokens"] = 1000
    elif judge == "gpt_oss":
        params["max_new_tokens"] = 1512
    else:
        raise ValueError(f"Unsupported judge: {judge}")
    return params


def decode_generation_outputs(tokenizer, outputs, prompt_width: int) -> list[str]:
    return tokenizer.batch_decode(outputs[:, prompt_width:], skip_special_tokens=True)


def parse_outputs(judge: str, decoded: list[str]) -> list[Any]:
    if judge == "harmbench":
        return ["yes" if "yes" in item.lower() else "no" for item in decoded]
    if judge == "wildguard":
        return [
            {
                "harmful_request": "harmful request: yes" in item.lower(),
                "response_refusal": "response refusal: yes" in item.lower(),
                "harmful_response": "harmful response: yes" in item.lower(),
            }
            for item in decoded
        ]
    if judge == "jailjudge":
        return decoded
    if judge == "gpt_oss":
        return [extract_oss_json(item) for item in decoded]
    raise ValueError(f"Unsupported judge: {judge}")


def run_one_batch(judge: str, tokenizer, model, batch: list[Sample]) -> list[Any]:
    prompts = build_batch_prompts(judge, batch, tokenizer)
    add_special_tokens = judge not in {"wildguard", "gpt_oss"}
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
        add_special_tokens=add_special_tokens,
    ).to(model.device)
    params = generation_params_for_judge(judge)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **params,
        )
    decoded = decode_generation_outputs(tokenizer, outputs, inputs["input_ids"].shape[1])
    parsed = parse_outputs(judge, decoded)
    del inputs, outputs
    free_vram()
    return parsed


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda error" in text


def try_batch(judge: str, tokenizer, model, base_samples: list[Sample], batch_size: int) -> bool:
    batch = repeat_to_length(base_samples, batch_size)
    try:
        run_one_batch(judge, tokenizer, model, batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except Exception as exc:  # pragma: no cover - exercised only with real GPU runs
        free_vram()
        if is_oom_error(exc):
            return False
        raise


def find_max_batch_size(
    judge: str,
    tokenizer,
    model,
    base_samples: list[Sample],
    max_batch_size: int,
) -> int | None:
    if not try_batch(judge, tokenizer, model, base_samples, 1):
        return None

    low = 1
    high = 1
    while high < max_batch_size:
        candidate = min(high * 2, max_batch_size)
        if try_batch(judge, tokenizer, model, base_samples, candidate):
            low = candidate
            if candidate == max_batch_size:
                return candidate
            high = candidate
        else:
            high = candidate
            break

    if low == max_batch_size:
        return low

    left = low + 1
    right = high - 1 if high > low else low
    best = low
    while left <= right:
        mid = (left + right) // 2
        if try_batch(judge, tokenizer, model, base_samples, mid):
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def benchmark_batch_size(
    judge: str,
    tokenizer,
    model,
    samples: list[Sample],
    batch_size: int,
    warmup_iters: int,
    benchmark_iters: int,
) -> BatchMeasurement:
    chunks = [
        samples[i:i + batch_size]
        for i in range(0, len(samples), batch_size)
    ]
    for _ in range(warmup_iters):
        for chunk in chunks:
            run_one_batch(judge, tokenizer, model, chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    total_examples = 0
    for _ in range(benchmark_iters):
        for chunk in chunks:
            run_one_batch(judge, tokenizer, model, chunk)
            total_examples += len(chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
        peak_vram_gb = None
    elapsed = time.perf_counter() - started
    return BatchMeasurement(
        batch_size=batch_size,
        examples_per_second=total_examples / elapsed,
        seconds_total=elapsed,
        num_examples=total_examples,
        peak_vram_gb=peak_vram_gb,
    )


def candidate_batch_sizes(max_fit_batch_size: int) -> list[int]:
    sizes = []
    size = 1
    while size < max_fit_batch_size:
        sizes.append(size)
        size *= 2
    if max_fit_batch_size not in sizes:
        sizes.append(max_fit_batch_size)
    return sizes


def benchmark_one_configuration(
    judge: str,
    quantization: str,
    samples: list[Sample],
    max_batch_size: int,
    warmup_iters: int,
    benchmark_iters: int,
) -> BenchmarkResult:
    model_id = model_id_for_judge(judge)
    if quantization not in SUPPORTED_QUANTIZATIONS[judge]:
        return BenchmarkResult(
            judge=judge,
            quantization=quantization,
            model_id=model_id,
            num_samples=len(samples),
            max_fit_batch_size=None,
            recommended_stable_batch_size=None,
            best_throughput_batch_size=None,
            best_examples_per_second=None,
            measurements=[],
            load_error=f"Unsupported quantization '{quantization}' for judge '{judge}'",
        )

    tokenizer = None
    model = None
    try:
        tokenizer, model = load_model_and_tokenizer_for_benchmark(judge, quantization)
    except Exception as exc:  # pragma: no cover - depends on local environment
        free_vram()
        return BenchmarkResult(
            judge=judge,
            quantization=quantization,
            model_id=model_id,
            num_samples=len(samples),
            max_fit_batch_size=None,
            recommended_stable_batch_size=None,
            best_throughput_batch_size=None,
            best_examples_per_second=None,
            measurements=[],
            load_error=str(exc),
        )

    try:
        max_fit = find_max_batch_size(judge, tokenizer, model, samples, max_batch_size)
        if max_fit is None:
            return BenchmarkResult(
                judge=judge,
                quantization=quantization,
                model_id=model_id,
                num_samples=len(samples),
                max_fit_batch_size=None,
                recommended_stable_batch_size=None,
                best_throughput_batch_size=None,
                best_examples_per_second=None,
                measurements=[],
                benchmark_error="Batch size 1 did not fit.",
            )

        measurements = [
            benchmark_batch_size(
                judge=judge,
                tokenizer=tokenizer,
                model=model,
                samples=samples,
                batch_size=batch_size,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            for batch_size in candidate_batch_sizes(max_fit)
        ]
        best = max(measurements, key=lambda item: item.examples_per_second)
        return BenchmarkResult(
            judge=judge,
            quantization=quantization,
            model_id=model_id,
            num_samples=len(samples),
            max_fit_batch_size=max_fit,
            recommended_stable_batch_size=floor_power_of_two(max_fit),
            best_throughput_batch_size=best.batch_size,
            best_examples_per_second=best.examples_per_second,
            measurements=measurements,
        )
    except Exception as exc:  # pragma: no cover - depends on local environment
        return BenchmarkResult(
            judge=judge,
            quantization=quantization,
            model_id=model_id,
            num_samples=len(samples),
            max_fit_batch_size=None,
            recommended_stable_batch_size=None,
            best_throughput_batch_size=None,
            best_examples_per_second=None,
            measurements=[],
            benchmark_error=str(exc),
        )
    finally:
        del model, tokenizer
        free_vram()


def print_summary(results: list[BenchmarkResult]) -> None:
    print()
    print("Judge benchmark summary")
    print("=" * 80)
    for result in results:
        header = f"{result.judge} [{result.quantization}]"
        print(header)
        print("-" * len(header))
        if result.load_error:
            print(f"load_error: {result.load_error}")
            continue
        if result.benchmark_error:
            print(f"benchmark_error: {result.benchmark_error}")
            continue
        print(f"model_id: {result.model_id}")
        print(f"num_samples: {result.num_samples}")
        print(f"max_fit_batch_size: {result.max_fit_batch_size}")
        print(f"recommended_stable_batch_size: {result.recommended_stable_batch_size}")
        print(f"best_throughput_batch_size: {result.best_throughput_batch_size}")
        print(f"best_examples_per_second: {result.best_examples_per_second:.3f}")
        for measurement in result.measurements:
            peak_vram = (
                f"{measurement.peak_vram_gb:.2f} GB"
                if measurement.peak_vram_gb is not None
                else "n/a"
            )
            print(
                "  "
                f"b={measurement.batch_size:<4d} "
                f"eps={measurement.examples_per_second:>8.3f} "
                f"peak_vram={peak_vram}"
            )
        print()


def serialize_results(results: list[BenchmarkResult], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_path": str(args.dataset_path),
            "response_field": args.response_field,
            "judges": args.judges,
            "quantizations": args.quantizations,
            "num_samples": args.num_samples,
            "max_batch_size": args.max_batch_size,
            "warmup_iters": args.warmup_iters,
            "benchmark_iters": args.benchmark_iters,
            "seed": args.seed,
            "shuffle": args.shuffle,
        },
        "results": [
            {
                **{
                    key: value
                    for key, value in asdict(result).items()
                    if key != "measurements"
                },
                "measurements": [asdict(measurement) for measurement in result.measurements],
            }
            for result in results
        ],
    }


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_json is not None:
        return args.output_json
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return args.artifacts_dir / f"judge_batch_benchmark_{stamp}.json"


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(
        dataset_path=args.dataset_path,
        response_field=args.response_field,
        num_samples=args.num_samples,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    results = []
    for judge in args.judges:
        for quantization in args.quantizations:
            print(f"Benchmarking {judge} [{quantization}]...")
            result = benchmark_one_configuration(
                judge=judge,
                quantization=quantization,
                samples=samples,
                max_batch_size=args.max_batch_size,
                warmup_iters=args.warmup_iters,
                benchmark_iters=args.benchmark_iters,
            )
            results.append(result)

    print_summary(results)
    payload = serialize_results(results, args)
    output_path = resolve_output_path(args)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
