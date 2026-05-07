"""Plot Wang-style probe accuracy at layer 25, comparing raw reps vs.
reps projected through the trained JEPA predictor (jepa_predictor.pt).

Reads sweep CSVs named ``<cell>_raw.csv`` and ``<cell>_pred.csv`` produced
by sweep_l25_all_cells.sh. Renders one figure per (model, defense) showing
each probe's accuracy as a function of probe-train-set size, with separate
lines for {pra, no_pra, base} × {raw, pred}.

Usage:
    python scripts/probe_train_size/plot_l25.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELL_RE = re.compile(r"^(?P<model>[lq])_(?P<defense>cb|triplet)_(?P<variant>pra|no_pra)$")
BASE_NAMES = {"l_base", "q_base"}


def parse_cell(name: str) -> Dict[str, str]:
    if name in BASE_NAMES:
        return {"model": name[0], "defense": "base", "variant": "base"}
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized cell name: {name}")
    return m.groupdict()


# Wang setup slices only
KEEP_FAMILIES = {
    "id_val": ["id_val"],
    "rs3_para": ["ood_paraphrased_advbench", "ood_heldout_paraphrased_harmbench"],
    "rs3_heldout": [
        "ood_heldout_malicious_harmbench",
        "ood_heldout_cleaned_harmbench",
    ],
}


def family_for_slice(s: str) -> str | None:
    for fam, slices in KEEP_FAMILIES.items():
        if s in slices:
            return fam
    return None


def load_all(sweep_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for csv in sorted(sweep_root.glob("*.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        # Filename = <cell>_<tag>.csv. Tag = raw|pred.
        stem = csv.stem
        if stem.endswith("_pred"):
            cell, tag = stem[: -len("_pred")], "pred"
        elif stem.endswith("_raw"):
            cell, tag = stem[: -len("_raw")], "raw"
        else:
            cell, tag = stem, "raw"
        df["cell"] = cell
        df["tag"] = tag
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No sweep CSVs under {sweep_root}")
    df = pd.concat(rows, ignore_index=True)
    df["family"] = df["slice"].map(family_for_slice)
    df = df.dropna(subset=["family"])
    parsed = df["cell"].apply(parse_cell).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    return df


# variant × tag → style. Solid = raw, dashed = projected.
LINE_STYLE = {
    ("pra", "raw"): {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "PRA · raw"},
    ("pra", "pred"): {"color": "#1f77b4", "marker": "o", "ls": "--", "label": "PRA · proj"},
    ("no_pra", "raw"): {"color": "#d62728", "marker": "s", "ls": "-", "label": "no-PRA · raw"},
    ("no_pra", "pred"): {"color": "#d62728", "marker": "s", "ls": "--", "label": "no-PRA · proj"},
    ("base", "raw"): {"color": "#7f7f7f", "marker": "^", "ls": "-", "label": "base"},
}


def plot_panel(df: pd.DataFrame, model: str, defense: str, metric: str, out_dir: Path) -> None:
    sub = df[(df["model"] == model) & ((df["defense"] == defense) | (df["defense"] == "base"))]
    if sub.empty:
        return

    families = ["id_val", "rs3_para", "rs3_heldout"]
    families = [f for f in families if f in sub["family"].unique()]
    probes = sorted(sub["probe"].unique())

    n_rows = len(probes)
    n_cols = len(families)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.4 * n_cols, 2.8 * n_rows), sharey=True, squeeze=False
    )

    seen_styles: List[str] = []
    for r, probe in enumerate(probes):
        for c, fam in enumerate(families):
            ax = axes[r][c]
            cell = sub[(sub["probe"] == probe) & (sub["family"] == fam)]
            for (variant, tag), style in LINE_STYLE.items():
                cv = cell[(cell["variant"] == variant) & (cell["tag"] == tag)]
                if cv.empty:
                    continue
                per_seed = cv.groupby(["n_train", "seed"])[metric].mean().reset_index()
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
                    linestyle=style["ls"],
                    label=style["label"] if style["label"] not in seen_styles else None,
                    capsize=2,
                    markersize=4,
                    linewidth=1.2,
                )
                seen_styles.append(style["label"])
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(fam, fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel("probe train size")
            if c == 0:
                ax.set_ylabel(f"{probe}\n{metric}")
            ax.set_ylim(0, 1.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=5, fontsize=8,
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.suptitle(
        f"{model.upper()} · {defense}  —  Wang setup, layer 25, raw vs predictor-projected",
        y=1.08,
    )
    fig.tight_layout()
    out = out_dir / f"probe_l25_wang_{model}_{defense}_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument(
        "--sweep_root",
        type=Path,
        default=Path("runs/experiments/headline_backup_rerun_probing/sweep_l25"),
    )
    ap.add_argument(
        "--out_dir", type=Path, default=Path("assets/figures/probe_train_size_l25_wang")
    )
    ap.add_argument("--metric", default="acc", choices=["acc", "auc", "f1"])
    args = ap.parse_args()

    df = load_all(args.sweep_root)
    print(f"[loaded] {len(df)} rows from {args.sweep_root}")
    print(f"[cells]  {sorted(df['cell'].unique())}")
    print(f"[tags]   {sorted(df['tag'].unique())}")
    print(f"[probes] {sorted(df['probe'].unique())}")

    for model in ["l", "q"]:
        for defense in ["cb", "triplet"]:
            plot_panel(df, model, defense, args.metric, args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "all_results_l25.csv", index=False)
    print(f"[saved] {args.out_dir / 'all_results_l25.csv'}")


if __name__ == "__main__":
    main()
