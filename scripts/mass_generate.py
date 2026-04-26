#!/usr/bin/env python3
"""Fast batched generation over repo datasets, Circuit Breakers JSON/JSONL, or custom inputs.

Examples:
    /workspace/AdversariaLLM/venv/bin/python scripts/mass_generate.py \
        --model-id Qwen/Qwen3-8B \
        --cb-path data/circuit_breakers_train.json \
        --limit 128 \
        --output-path /tmp/qwen3_cb_generations.jsonl

    /workspace/AdversariaLLM/venv/bin/python scripts/mass_generate.py \
        --model-id meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset-name adv_behaviors \
        --idx "list(range(50))" \
        --output-path /tmp/adv_behaviors.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from adversariallm.dataset import PromptDataset
from adversariallm.io_utils.model_loading import load_model_and_tokenizer
from adversariallm.lm_utils import generate_ragged_batched, prepare_conversation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--peft-path", default=None)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-name", choices=sorted(PromptDataset._registry.keys()))
    source.add_argument("--cb-path")
    source.add_argument("--input-file")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--idx", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--messages-path", default=None)
    parser.add_argument("--targets-path", default=None)

    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--target-field", default="output")
    parser.add_argument("--response-field", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--generation-mode",
        choices=("single", "temperature_sweep"),
        default="single",
        help="`single` does one fast batched pass. `temperature_sweep` does one full pass per temperature.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=None)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--initial-batch-size", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")

    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--print-every", type=int, default=50)
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


def maybe_parse_idx(raw: str | None) -> Any:
    if raw is None:
        return None
    if raw.startswith("list(range("):
        return raw
    try:
        return int(raw)
    except ValueError:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return parsed


def load_conversations(args: argparse.Namespace) -> list[list[dict[str, str]]]:
    if args.dataset_name:
        dataset_cfg: dict[str, Any] = {
            "name": args.dataset_name,
            "seed": args.seed,
            "shuffle": args.shuffle,
            "idx": maybe_parse_idx(args.idx),
        }
        if args.messages_path is not None:
            dataset_cfg["messages_path"] = args.messages_path
        if args.targets_path is not None:
            dataset_cfg["targets_path"] = args.targets_path
        dataset = PromptDataset.from_name(args.dataset_name)(OmegaConf.create(dataset_cfg))
        conversations = [dataset[i] for i in range(len(dataset))]
    elif args.cb_path:
        records = _load_json_or_jsonl(Path(args.cb_path))
        if args.limit is not None:
            records = records[: args.limit]
        response_field = args.response_field or args.target_field
        conversations = [
            [
                {"role": "user", "content": str(record[args.prompt_field])},
                {"role": "assistant", "content": str(record.get(response_field, ""))},
            ]
            for record in records
        ]
    else:
        records = _load_json_or_jsonl(Path(args.input_file))
        if args.limit is not None:
            records = records[: args.limit]
        conversations = []
        for record in records:
            if "messages" in record and isinstance(record["messages"], list):
                messages = record["messages"]
            else:
                response_field = args.response_field or args.target_field
                messages = [
                    {"role": "user", "content": str(record[args.prompt_field])},
                    {"role": "assistant", "content": str(record.get(response_field, ""))},
                ]
            conversations.append(messages)

    if args.limit is not None and args.dataset_name:
        conversations = conversations[: args.limit]
    return conversations


def build_model_params(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer_id = args.tokenizer_id or args.model_id
    return {
        "id": args.model_id,
        "tokenizer_id": tokenizer_id,
        "short_name": args.model_id.split("/")[-1],
        "developer_name": args.model_id.split("/")[0],
        "compile": False,
        "dtype": args.dtype,
        "chat_template": args.chat_template,
        "trust_remote_code": args.trust_remote_code,
        "peft_path": args.peft_path,
    }


def main() -> int:
    args = parse_args()
    conversations = load_conversations(args)
    if not conversations:
        raise ValueError("No input conversations loaded.")

    model, tokenizer = load_model_and_tokenizer(build_model_params(args))
    prompt_token_tensors = []
    for conversation in conversations:
        token_tensors = prepare_conversation(tokenizer, conversation)
        flat_tokens = [t for turn_tokens in token_tensors for t in turn_tokens]
        prompt_tokens = flat_tokens if len(flat_tokens) == 1 else flat_tokens[:-1]
        if len(prompt_tokens) == 1:
            prompt_token_tensors.append(prompt_tokens[0])
        else:
            prompt_token_tensors.append(torch.cat(prompt_tokens))

    if args.generation_mode == "single":
        temperatures = [args.temperature]
    else:
        temperatures = args.temperatures if args.temperatures is not None else [args.temperature]

    generations_by_temperature: dict[str, list[list[str]]] = {}
    start = time.time()
    for temperature in temperatures:
        generations_by_temperature[str(temperature)] = generate_ragged_batched(
            model,
            tokenizer,
            token_list=prompt_token_tensors,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            initial_batch_size=args.initial_batch_size or len(prompt_token_tensors),
            use_cache=not args.no_cache,
            verbose=True,
        )
    duration = time.time() - start

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        for i, conversation in enumerate(conversations):
            temp_outputs = {
                temp: outputs[i]
                for temp, outputs in generations_by_temperature.items()
            }
            default_outputs = temp_outputs[str(temperatures[0])]
            record = {
                "index": i,
                "messages": conversation,
                "prompt": conversation[0]["content"] if conversation else "",
                "target": conversation[-1]["content"] if conversation else "",
                "generated": default_outputs,
                "generated_by_temperature": temp_outputs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.print_every > 0 and (i + 1) % args.print_every == 0:
                print(f"Wrote {i + 1}/{len(conversations)} generations")

    print(f"Generated {len(conversations)} prompts in {duration:.2f}s")
    print(f"Avg prompts/sec: {len(conversations) / duration:.2f}")
    print(f"Temperatures: {', '.join(str(t) for t in temperatures)}")
    print(f"Output: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
