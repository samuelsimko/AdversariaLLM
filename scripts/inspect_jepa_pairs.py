#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from defenses.jepa_ce import load_jepa_pairs


def _load_raw_records(path: Optional[str], dataset: Optional[str], split: str, limit: int) -> List[Dict[str, Any]]:
    if path:
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data[:limit]
        except json.JSONDecodeError:
            pass
        return [json.loads(line) for line in text.splitlines() if line.strip()][:limit]

    if not dataset:
        return []

    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    return [dict(item) for item in ds.select(range(min(limit, len(ds))))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect paired-view records before JEPA training.")
    parser.add_argument("--pair_path", type=str, default=None)
    parser.add_argument("--pair_dataset", type=str, default=None)
    parser.add_argument("--pair_split", type=str, default="train")
    parser.add_argument("--pair_format", type=str, default="auto", choices=["auto", "wildjailbreak", "reverse"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    raw = _load_raw_records(args.pair_path, args.pair_dataset, args.pair_split, args.limit)
    print(f"raw_records={len(raw)}")
    if raw:
        print(f"first_raw_keys={sorted(raw[0].keys())}")

    adv, clean, responses, intents, sources = load_jepa_pairs(
        pair_path=args.pair_path,
        pair_dataset=args.pair_dataset,
        pair_split=args.pair_split,
        pair_format=args.pair_format,
        limit=args.limit,
    )
    print(f"loaded_pairs={len(adv)}")
    print(f"intents={intents}")
    print(f"sources={sources}")
    for idx, (a, c, r, intent) in enumerate(zip(adv, clean, responses, intents)):
        print(f"\n[{idx}] intent: {intent}")
        print(f"[{idx}] adv_prompt: {a[:240]!r}")
        print(f"[{idx}] clean_prompt: {c[:240]!r}")
        print(f"[{idx}] response: {r[:240]!r}")


if __name__ == "__main__":
    main()
