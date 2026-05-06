"""Pull multilingual + JEPA-attack-family probe data into WPF CSV layout.

Writes:
  Why-Probe-Fails/data/multi/malicious/<lang>.csv         per-language jailbreak prompts
  Why-Probe-Fails/data/jepa/malicious/<family>.csv         per-family adversarial prompts
  Why-Probe-Fails/data/jepa/benign/<family>.csv            per-family over-refusal prompts

Sources:
  Multi:  wow2000/multilingual_jailbreak_challenges (gated; HF_TOKEN required)
  JEPA:   memo-ozdincer/jepadata
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from datasets import load_dataset


WPF_DATA = Path("/root/AdversariaLLM/Why-Probe-Fails/data")
LANGS = ["en", "zh-cn", "es", "ja", "ar", "fy", "cs", "id", "pt"]
JEPA_FAMILIES = ["encoding", "prefilling", "persona", "inpainting"]
ROWS_PER_SLICE = 200


def _write_csv(path: Path, prompts: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt"])
        for p in prompts:
            if p and p.strip():
                w.writerow([p.strip()])
    print(f"  wrote {path}: {sum(1 for _ in open(path)) - 1} rows")


def build_multi():
    print("[multi] loading wow2000/multilingual_jailbreak_challenges ...")
    ds = load_dataset("wow2000/multilingual_jailbreak_challenges", split="train").to_pandas()
    print(f"  rows: {len(ds)}; cols: {list(ds.columns)[:10]}...")
    out_dir = WPF_DATA / "multi" / "malicious"
    for lang in LANGS:
        col = f"{lang}-user_instruction"
        if col not in ds.columns:
            print(f"  skip {lang} (no column)")
            continue
        prompts = [s for s in ds[col].astype(str).tolist() if s and s.strip() and s.strip().lower() not in ("nan", "none")]
        prompts = prompts[:ROWS_PER_SLICE]
        _write_csv(out_dir / f"{lang}.csv", prompts)


def build_jepa():
    print("\n[jepa] loading memo-ozdincer/jepadata ...")
    ds = load_dataset("memo-ozdincer/jepadata", split="train")
    print(f"  rows: {len(ds)}")
    out_mal = WPF_DATA / "jepa" / "malicious"
    out_ben = WPF_DATA / "jepa" / "benign"

    def _last_user_message(view) -> str:
        if not isinstance(view, dict):
            return ""
        msgs = view.get("messages", [])
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m.get("content", "") or ""
        return ""

    for family in JEPA_FAMILIES:
        rows_mal, rows_ben = [], []
        for ex in ds:
            if ex.get("attack_type") != family:
                continue
            prompt = _last_user_message(ex.get("attack_view"))
            if not prompt:
                continue
            if ex.get("side") == "harmful":
                if len(rows_mal) < ROWS_PER_SLICE:
                    rows_mal.append(prompt)
            elif ex.get("side") == "benign":
                if len(rows_ben) < ROWS_PER_SLICE:
                    rows_ben.append(prompt)
            if len(rows_mal) >= ROWS_PER_SLICE and len(rows_ben) >= ROWS_PER_SLICE:
                break
        _write_csv(out_mal / f"{family}.csv", rows_mal)
        _write_csv(out_ben / f"{family}.csv", rows_ben)


if __name__ == "__main__":
    build_multi()
    build_jepa()
    print("\nDone.")
