#!/usr/bin/env python3
"""Print a status table for an in-flight cluster experiment.

Looks at:
  - local READY sentinels under runs/experiments/<exp>/<cell>/...
  - HF dataset repo (if HF_REPO + HF_TOKEN are set): adapters/, attack_results/,
    eval_results/ folder listings.

Usage:
    python cluster_scripts/sync_status.py
    python cluster_scripts/sync_status.py --exp headline_rerun_full
    python cluster_scripts/sync_status.py --hf-only
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def cells_from_yaml(path: Path) -> list[str]:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("pipelines", {}).keys())


def local_status(exp: str, cells: list[str]) -> dict[str, dict]:
    base = REPO_ROOT / "runs" / "experiments" / exp
    out: dict[str, dict] = {}
    for cell in cells:
        cell_root = base / cell
        train_ready = (cell_root / "READY").exists()
        attack_dir = cell_root / "attacks" / cell
        attacks_done = []
        if attack_dir.is_dir():
            for atk in sorted(p for p in attack_dir.iterdir() if p.is_dir()):
                if (atk / "READY").exists():
                    attacks_done.append(atk.name)
        eval_dir = cell_root / "benign_eval" / cell
        evals_done = []
        if eval_dir.is_dir():
            for ev in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
                if (ev / "READY").exists():
                    evals_done.append(ev.name)
        out[cell] = {
            "train": train_ready,
            "attacks": attacks_done,
            "evals": evals_done,
        }
    return out


def hf_status(repo_id: str) -> dict[str, dict]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed; skipping HF status", file=sys.stderr)
        return {}
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as exc:
        print(f"HF list_repo_files failed: {exc}", file=sys.stderr)
        return {}

    cells: dict[str, dict] = collections.defaultdict(lambda: {
        "adapter": False, "attacks": set(), "evals": set(),
    })
    for f in files:
        parts = f.split("/")
        if len(parts) < 2:
            continue
        kind, cell = parts[0], parts[1]
        if kind == "adapters":
            cells[cell]["adapter"] = True
        elif kind == "attack_results" and len(parts) >= 3:
            cells[cell]["attacks"].add(parts[2])
        elif kind == "eval_results" and len(parts) >= 3:
            cells[cell]["evals"].add(parts[2])
    return {k: {"adapter": v["adapter"],
                "attacks": sorted(v["attacks"]),
                "evals": sorted(v["evals"])}
            for k, v in cells.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="headline_rerun_full")
    ap.add_argument("--config", default="experiments/configs/headline_rerun_full.yaml")
    ap.add_argument("--hf-only", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    args = ap.parse_args()

    cells = cells_from_yaml(REPO_ROOT / args.config)
    print(f"Experiment: {args.exp}  ({len(cells)} cells)")
    print("=" * 80)

    if not args.hf_only:
        local = local_status(args.exp, cells)
        print("\nLOCAL (runs/experiments/...):")
        print(f"  {'cell':<22} train  #attacks  #evals  attacks")
        for cell in cells:
            s = local[cell]
            mark = "✓" if s["train"] else "·"
            atk_str = ",".join(s["attacks"]) if s["attacks"] else "-"
            print(f"  {cell:<22}   {mark}      {len(s['attacks']):>2}      {len(s['evals']):>2}     {atk_str}")

    if not args.local_only:
        repo_id = os.environ.get("HF_REPO", "").strip().lstrip("/")
        if not repo_id:
            print("\nHF: HF_REPO env var unset; skipping")
        else:
            hf = hf_status(repo_id)
            print(f"\nHF ({repo_id}):")
            print(f"  {'cell':<22} adapter  #attacks  #evals")
            for cell in cells:
                s = hf.get(cell, {"adapter": False, "attacks": [], "evals": []})
                mark = "✓" if s["adapter"] else "·"
                print(f"  {cell:<22}    {mark}        {len(s['attacks']):>2}      {len(s['evals']):>2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
