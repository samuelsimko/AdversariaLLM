#!/usr/bin/env python3
"""Generate many candidate prompts from a trained reverse model.

This script loads a reverse-model LoRA adapter and, for each response in a source
dataset, samples candidate user prompts that could have produced that response.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TEMPLATE = """You are given a model response. Infer a plausible user prompt that could have led to this response.
Return only the prompt, wrapped exactly in <PROMPT> and </PROMPT> tags.

Response:
{response}

User prompt:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-dir", required=True)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cb-path")
    source.add_argument("--input-file")

    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="output")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--generation-mode",
        choices=("single", "temperature_sweep"),
        default="single",
        help="`single` does one fast batched pass. `temperature_sweep` does one full pass per temperature.",
    )
    parser.add_argument("--num-generations", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--random-temperature-count", type=int, default=None)
    parser.add_argument("--temperature-min", type=float, default=0.0)
    parser.add_argument("--temperature-max", type=float, default=3.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--initial-batch-size", type=int, default=64)
    parser.add_argument("--use-cache", action="store_true", default=None)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--backup-every", type=int, default=10)
    return parser.parse_args()


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}: {type(payload)!r}")


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.cb_path or args.input_file)
    records = _load_json_or_jsonl(path)
    if args.limit is not None:
        records = records[: args.limit]
    return records


def clean_generation(text: str) -> str:
    text = text.strip()
    if "<PROMPT>" in text and "</PROMPT>" in text:
        return text.split("<PROMPT>", 1)[1].split("</PROMPT>", 1)[0].strip()
    if "User prompt:" in text:
        text = text.split("User prompt:", 1)[-1].strip()
    return text.replace("<PROMPT>", "").replace("</PROMPT>", "").strip()


def default_use_cache(base_model: str) -> bool:
    lowered = base_model.lower()
    if "qwen/qwen3.5" in lowered or "qwen3_5" in lowered or "qwen3.5" in lowered:
        return False
    return True


def build_temperature_schedule(args: argparse.Namespace) -> list[float]:
    if args.random_temperature_count is not None:
        rng = random.Random(args.seed)
        return [
            round(rng.uniform(args.temperature_min, args.temperature_max), 6)
            for _ in range(args.random_temperature_count)
        ]
    if args.generation_mode == "single":
        return [args.temperature]
    return args.temperatures if args.temperatures is not None else [args.temperature]


def write_snapshot(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    backup_path: Path | None = None,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    if backup_path is not None:
        backup_tmp = backup_path.with_suffix(backup_path.suffix + ".tmp")
        with backup_tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        backup_tmp.replace(backup_path)


@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    prompt_texts: list[str],
    *,
    batch_size: int,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    use_cache: bool,
) -> list[list[str]]:
    outputs_by_prompt: list[list[str]] = []

    for start in range(0, len(prompt_texts), batch_size):
        end = min(start + batch_size, len(prompt_texts))
        chunk = prompt_texts[start:end]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )

        decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        cleaned = [clean_generation(text) for text in decoded]
        for idx in range(len(chunk)):
            begin = idx * num_generations
            outputs_by_prompt.append(cleaned[begin : begin + num_generations])

    return outputs_by_prompt


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    records = load_records(args)
    if not records:
        raise ValueError("No records loaded.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir).eval()

    prompt_texts = [
        TEMPLATE.format(response=str(record.get(args.response_field, "")))
        for record in records
    ]

    temperatures = build_temperature_schedule(args)
    per_temperature_generations = args.num_generations
    if args.random_temperature_count is not None:
        per_temperature_generations = 1

    use_cache = args.use_cache if args.use_cache is not None else default_use_cache(args.base_model)
    if not use_cache:
        print("Using non-cache generation path for compatibility with the selected reverse model.")

    effective_prompt_batch_size = max(1, args.initial_batch_size // max(1, per_temperature_generations))
    print(
        f"Prompt batch size: {effective_prompt_batch_size} "
        f"(requested budget={args.initial_batch_size}, num_generations={per_temperature_generations})"
    )

    rows = [
        {
            "index": index,
            "record": record,
            "response": record.get(args.response_field, ""),
            "true_prompt": record.get(args.prompt_field, ""),
            "generated_prompts": [],
            "generated_prompts_by_temperature": {},
        }
        for index, record in enumerate(records)
    ]
    events_path = args.output_path.with_suffix(args.output_path.suffix + ".events.jsonl")
    backup_dir = args.output_path.parent / f"{args.output_path.stem}_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for temp_index, temperature in enumerate(temperatures, start=1):
        raw_generations = batched_generate(
            model,
            tokenizer,
            prompt_texts,
            batch_size=effective_prompt_batch_size,
            num_generations=per_temperature_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_cache=use_cache,
        )
        temperature_key = str(temperature)
        with events_path.open("a", encoding="utf-8") as event_file:
            for index, outputs in enumerate(raw_generations):
                rows[index]["generated_prompts_by_temperature"][temperature_key] = outputs
                rows[index]["generated_prompts"].extend(outputs)
                event_file.write(
                    json.dumps(
                        {
                            "index": index,
                            "temperature": temperature,
                            "generated_prompts": outputs,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        backup_path = None
        if args.backup_every > 0 and temp_index % args.backup_every == 0:
            backup_path = backup_dir / f"{args.output_path.stem}.temp_{temp_index:03d}.jsonl"
        write_snapshot(rows, args.output_path, backup_path=backup_path)
        print(f"Completed temperature {temp_index}/{len(temperatures)} = {temperature}")

    duration = time.time() - start

    print(f"Generated reverse prompts for {len(records)} behaviors in {duration:.2f}s")
    print(f"Avg behaviors/sec: {len(records) / duration:.2f}")
    print(f"Temperatures: {', '.join(str(t) for t in temperatures)}")
    print(f"Output: {args.output_path}")
    print(f"Events log: {events_path}")
    print(f"Backup dir: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
