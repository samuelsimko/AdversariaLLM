#!/usr/bin/env python3
"""Build a non-overlapping OR-Bench training pool.

Output: data/or_bench_train_1000.jsonl with {prompt, category} records, sampled
from `bench-llm/or-bench` config `or-bench-80k`, excluding any prompt that
appears in `or-bench-hard-1k` (which we use for eval).

Responses are NOT included in this file — pair with a teacher-model generator
(see scripts/generate_or_bench_responses.py) before plugging into a defense's
`extra_benign_path`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def _norm(s: str) -> str:
    """Whitespace-collapsed lowercased key for dedup."""
    return " ".join(s.split()).strip().lower()


def _hash(s: str) -> str:
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/or_bench_train_1000.jsonl")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also_exclude_toxic", action="store_true",
                        help="Also exclude prompts that appear in or-bench-toxic.")
    args = parser.parse_args()

    print(f"loading or-bench-hard-1k (eval split — to exclude)...")
    hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
    hard_hashes = {_hash(r["prompt"]) for r in hard}
    print(f"  hard-1k: {len(hard)} rows, {len(hard_hashes)} unique hashes")

    exclude = set(hard_hashes)
    if args.also_exclude_toxic:
        print("loading or-bench-toxic...")
        tox = load_dataset("bench-llm/or-bench", "or-bench-toxic")["train"]
        exclude |= {_hash(r["prompt"]) for r in tox}
        print(f"  + toxic adds; total excluded hashes: {len(exclude)}")

    print(f"loading or-bench-80k (training pool)...")
    big = load_dataset("bench-llm/or-bench", "or-bench-80k")["train"]
    print(f"  80k: {len(big)} rows")

    # Stream through 80k once: keep first occurrence of each normalized prompt
    # that doesn't collide with the eval set.
    seen_in_pool = set()
    eligible: list[dict] = []
    excluded_overlap = 0
    excluded_dup = 0
    for r in big:
        p = r["prompt"]
        if not isinstance(p, str) or not p.strip():
            continue
        h = _hash(p)
        if h in exclude:
            excluded_overlap += 1
            continue
        if h in seen_in_pool:
            excluded_dup += 1
            continue
        seen_in_pool.add(h)
        eligible.append({"prompt": p, "category": r.get("category", "")})

    print(f"  eligible (no eval overlap, deduped): {len(eligible)}  "
          f"(dropped {excluded_overlap} eval-overlap, {excluded_dup} dup)")

    if len(eligible) < args.n:
        raise SystemExit(
            f"Only {len(eligible)} eligible rows but {args.n} requested. "
            "Lower --n or relax filters."
        )

    rng = random.Random(args.seed)
    sample = rng.sample(eligible, args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(sample)} rows -> {out}")
    # Category breakdown
    cats: dict[str, int] = {}
    for r in sample:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("category breakdown:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {c:<30s}  {n}")


if __name__ == "__main__":
    main()
