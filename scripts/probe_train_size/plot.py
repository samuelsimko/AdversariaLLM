"""Plot probe accuracy as a function of probe-train-set size.

Reads per-cell sweep CSVs and draws four panels (l_cb, l_triplet, q_cb,
q_triplet), each with PRA vs no-PRA vs base-model lines, faceted by probe
type and OOD slice family.

Slice families:
  - id_val
  - rs3_para     : ood_paraphrased_advbench, ood_heldout_paraphrased_harmbench
  - rs3_heldout  : ood_heldout_malicious_harmbench, ood_heldout_cleaned_harmbench
  - jepa_attacks : ood_jepa_*  (avg over 4 attacks)
  - multilingual : ood_multi_* (avg over 9 langs)

Usage:
    python scripts/probe_train_size/plot.py \\
        --sweep_root runs/experiments/headline_backup_rerun_probing/sweep \\
        --out_dir assets/figures/probe_train_size \\
        --metric acc
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Cell name → (model, defense, variant)
CELL_RE = re.compile(r"^(?P<model>[lq])_(?P<defense>cb|triplet)_(?P<variant>pra|no_pra|base)$")
BASE_NAMES = {"l_base", "q_base"}


def parse_cell(name: str) -> Dict[str, str]:
    if name in BASE_NAMES:
        return {"model": name[0], "defense": "base", "variant": "base"}
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized cell name: {name}")
    return m.groupdict()


SLICE_FAMILIES: Dict[str, List[str]] = {
    "id_val": ["id_val"],
    "rs3_para": ["ood_paraphrased_advbench", "ood_heldout_paraphrased_harmbench"],
    "rs3_heldout": [
        "ood_heldout_malicious_harmbench",
        "ood_heldout_cleaned_harmbench",
    ],
}

JEPA_PREFIX = "ood_jepa_"
MULTI_PREFIX = "ood_multi_"


def family_for_slice(s: str) -> str | None:
    for fam, slices in SLICE_FAMILIES.items():
        if s in slices:
            return fam
    if s.startswith(JEPA_PREFIX):
        return "jepa_attacks"
    if s.startswith(MULTI_PREFIX):
        return "multilingual"
    return None


def load_all(sweep_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for csv in sorted(sweep_root.glob("*.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No sweep CSVs under {sweep_root}")
    df = pd.concat(rows, ignore_index=True)
    df["family"] = df["slice"].map(family_for_slice)
    df = df.dropna(subset=["family"])
    parsed = df["cell"].apply(parse_cell).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df


VARIANT_STYLE = {
    "pra": {"color": "#1f77b4", "marker": "o", "label": "PRA"},
    "no_pra": {"color": "#d62728", "marker": "s", "label": "no-PRA"},
    "base": {"color": "#7f7f7f", "marker": "^", "label": "base", "linestyle": "--"},
}


def plot_panel(df: pd.DataFrame, model: str, defense: str, metric: str, out_dir: Path) -> None:
    """One figure per (model, defense). Subplots = (probe × family)."""
    sub = df[(df["model"] == model) & ((df["defense"] == defense) | (df["defense"] == "base"))]
    if sub.empty:
        return

    families = ["id_val", "rs3_para", "rs3_heldout", "jepa_attacks", "multilingual"]
    families = [f for f in families if f in sub["family"].unique()]
    probes = sorted(sub["probe"].unique())

    n_rows = len(probes)
    n_cols = len(families)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.2 * n_cols, 2.8 * n_rows), sharey=True, squeeze=False
    )

    for r, probe in enumerate(probes):
        for c, fam in enumerate(families):
            ax = axes[r][c]
            cell = sub[(sub["probe"] == probe) & (sub["family"] == fam)]
            for variant, style in VARIANT_STYLE.items():
                cv = cell[cell["variant"] == variant]
                if cv.empty:
                    continue
                # Aggregate by (n_train, seed) → mean over slices in family,
                # then mean ± std over seeds.
                per_seed = (
                    cv.groupby(["n_train", "seed"])[metric].mean().reset_index()
                )
                agg = (
                    per_seed.groupby("n_train")[metric]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
                ax.errorbar(
                    agg["n_train"],
                    agg["mean"],
                    yerr=agg["sem"].fillna(0),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style.get("linestyle", "-"),
                    label=style["label"],
                    capsize=2,
                    markersize=4,
                    linewidth=1.2,
                )
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(fam, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("probe train size")
            if c == 0:
                ax.set_ylabel(f"{probe}\n{metric}")
            ax.set_ylim(0, 1.02)
    # Single legend at the top
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{model.upper()} · {defense}  —  probe {metric} vs train size", y=1.06)
    fig.tight_layout()
    out = out_dir / f"probe_train_size_{model}_{defense}_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument(
        "--sweep_root",
        type=Path,
        default=Path("runs/experiments/headline_backup_rerun_probing/sweep"),
    )
    ap.add_argument("--out_dir", type=Path, default=Path("assets/figures/probe_train_size"))
    ap.add_argument("--metric", default="acc", choices=["acc", "auc", "f1"])
    args = ap.parse_args()

    df = load_all(args.sweep_root)
    print(f"[loaded] {len(df)} rows from {args.sweep_root}")
    print(f"[families] {sorted(df['family'].unique())}")
    print(f"[cells] {sorted(df['cell'].unique())}")
    print(f"[probes] {sorted(df['probe'].unique())}")

    # 4 panels
    for model in ["l", "q"]:
        for defense in ["cb", "triplet"]:
            plot_panel(df, model, defense, args.metric, args.out_dir)

    # Also save a tidy long-format CSV for downstream use
    df.to_csv(args.out_dir / "all_results.csv", index=False)
    print(f"[saved] {args.out_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
