#!/usr/bin/env python3
"""Generate helpful responses to OR-Bench training prompts using Qwen3-32B (8-bit).

Reads a prompts-only JSONL (e.g. data/or_bench_train_1000.jsonl) with at least
`{prompt}` per line, generates a response per prompt with Qwen3-32B in
`enable_thinking=False` mode (we want answers, not reasoning traces), filters
out responses that look like refusals via a simple regex heuristic, and writes
`{prompt, response, category}` JSONL.

By design: GPU-only — must run on a compute node with ≥40GB free GPU memory
(8-bit quantization fits Qwen3-32B in ~32GB).

Refusal detection: cheap regex on the first 100 chars. False positives are
acceptable (we just lose a few training examples). The full overrefusal judge
(Qwen3-8B-based) would be more accurate but ~5x the compute; not worth it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# --- Refusal detector ----------------------------------------------------
# Lowercased regex, applied to first ~120 chars of the response.
_REFUSAL_PATTERNS = [
    r"^i\s+(?:cannot|can'?t|won'?t|am unable|am not able|will not|do not)\b",
    r"^i'?m\s+(?:sorry|unable|not able)\b",
    r"^i\s+apologi[sz]e\b",
    r"^as an ai\b",
    r"^unfortunately,?\s+(?:i\s+)?(?:cannot|can'?t)\b",
    r"^i must (?:decline|refuse)\b",
    r"^sorry,?\s+(?:but\s+)?(?:i\s+)?(?:cannot|can'?t)\b",
    r"refusing to (?:help|answer|provide|assist)",
    r"i (?:cannot|can'?t) (?:provide|help with|fulfill|assist with|answer)",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def looks_like_refusal(text: str) -> bool:
    if not text:
        return True
    head = text.strip()[:200].lower()
    return bool(_REFUSAL_RE.search(head))


# --- Main ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSONL with {prompt, category} per line")
    p.add_argument("--output", required=True, help="JSONL with {prompt, response, category}")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process first N prompts (for testing)")
    p.add_argument("--max_new_tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--system_prompt", default=(
        "You are a helpful and informative AI assistant. Answer the user's question "
        "thoroughly and accurately. Many questions touch on sensitive but legitimate "
        "topics (research, fiction, policy analysis, education); provide useful "
        "factual responses without unnecessary refusals or moralizing."))
    p.add_argument("--no_quantize", action="store_true",
                   help="Skip 8-bit quantization (only set if you have an 80GB GPU).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[:args.limit]
    print(f"loaded {len(rows)} prompts from {in_path}", flush=True)

    # --- Load model ---
    print(f"loading {args.model} (8-bit={'no' if args.no_quantize else 'yes'})...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched .generate

    if args.no_quantize:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    else:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_cfg, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    # --- Generate ---
    n_kept = 0
    n_refused = 0
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            # Build chat-format prompts using Qwen's template, enable_thinking=False.
            chat_strs = []
            for r in batch:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": r["prompt"]},
                ]
                chat = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                chat_strs.append(chat)

            enc = tokenizer(chat_strs, return_tensors="pt", padding=True, truncation=True,
                            max_length=2048).to(model.device)
            t1 = time.time()
            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Decode only the new tokens
            new_ids = out_ids[:, enc.input_ids.shape[1]:]
            responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for r, resp in zip(batch, responses):
                resp = resp.strip()
                n_total += 1
                if looks_like_refusal(resp):
                    n_refused += 1
                    continue
                fout.write(json.dumps({
                    "prompt": r["prompt"],
                    "response": resp,
                    "category": r.get("category", ""),
                }, ensure_ascii=False) + "\n")
                fout.flush()
                n_kept += 1
            elapsed = time.time() - t1
            if (i // args.batch_size) % 5 == 0 or i + args.batch_size >= len(rows):
                print(f"  [{i + len(batch)}/{len(rows)}] kept={n_kept} refused={n_refused} "
                      f"batch_time={elapsed:.1f}s", flush=True)

    print(f"\nDONE total={n_total}  kept={n_kept}  refused={n_refused}  "
          f"refusal_rate={n_refused / max(n_total, 1):.1%}  out={out_path}", flush=True)


if __name__ == "__main__":
    main()
