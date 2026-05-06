"""Build the Wang-faithful results pivot table.

Reads classify_wang/<cell>_L<layer>_<tag>.csv (tag = raw|pred), pivots into
rows keyed by (model, defense, variant, layer, tag), and emits:
  - WANG_FAITHFUL_RESULTS.md  — main table + per-language and per-attack
                                 sub-tables + raw-vs-pred deltas.
  - wide_table_wang.csv       — pivoted table.
  - all_results_wang.csv      — long-form including tag column.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


CELL_RE = re.compile(r"^(?P<model>[lq])_(?P<defense>cb|triplet)_(?P<variant>pra|no_pra)$")
BASE_NAMES = {"l_base", "q_base"}

MULTI_LANGS = ("ar", "cs", "en", "es", "fy", "id", "ja", "pt", "zh-cn")
JEPA_ATTACKS = ("encoding", "inpainting", "persona", "prefilling")
WANG_MAL_DATASETS = ("advbench", "harmbench", "jailbreakbench", "maliciousinstruct")


def parse_cell(name: str) -> dict:
    if name in BASE_NAMES:
        return {"model": name[0], "defense": "base", "variant": "base"}
    m = CELL_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized cell name: {name}")
    return m.groupdict()


def parse_filename(path: Path) -> tuple[str, str]:
    """Filename = <cell>_L<layer>_<tag>.csv. Returns (cell, tag)."""
    stem = path.stem  # e.g. l_cb_pra_L25_raw
    m = re.match(r"^(?P<cell>.+)_L-?\d+_(?P<tag>raw|pred)$", stem)
    if not m:
        # legacy: <cell>_L<layer>.csv (no tag)
        m = re.match(r"^(?P<cell>.+)_L-?\d+$", stem)
        if not m:
            raise ValueError(f"unparseable filename: {path}")
        return m.group("cell"), "raw"
    return m.group("cell"), m.group("tag")


def load_all(in_dir: Path) -> pd.DataFrame:
    rows = []
    for csv in sorted(in_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        cell, tag = parse_filename(csv)
        df["tag"] = tag
        # cell + layer come from inside the file already
        rows.append(df)
    if not rows:
        raise SystemExit(f"no CSVs under {in_dir}")
    return pd.concat(rows, ignore_index=True)


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = (model, defense, variant, layer, tag). Cols = id_val + each OOD stem."""
    parsed = df["cell"].apply(parse_cell).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    id_acc = (
        df[df["split"] == "id_val"]
        .groupby(["model", "defense", "variant", "layer", "tag"])["acc"]
        .first()
        .rename("id_val")
    )
    ood = (
        df[df["split"] == "ood"]
        .pivot_table(
            index=["model", "defense", "variant", "layer", "tag"],
            columns="stem",
            values="acc",
        )
    )
    wide = pd.concat([id_acc, ood], axis=1).round(3)

    # Sort columns: id_val, Wang malicious, paraphrased, cleaned, multilingual,
    # JEPA-attack malicious, JEPA-attack benign, RS1 benign.
    cols_order = ["id_val"]
    for prefix in (
        "malicious_",       # all malicious_* (Wang mal + multi langs + JEPA attacks)
        "paraphrased_",
        "cleaned_",
        "benign_",
    ):
        cols_order += sorted(c for c in wide.columns if c.startswith(prefix))
    wide = wide[[c for c in cols_order if c in wide.columns]]
    return wide


def families(wide: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(wide.columns)
    return {
        "wang_mal": [f"malicious_{d}" for d in WANG_MAL_DATASETS if f"malicious_{d}" in cols],
        "wang_para": [c for c in cols if c.startswith("paraphrased_")],
        "wang_cleaned": [c for c in cols if c.startswith("cleaned_")],
        "multi": [f"malicious_{l}" for l in MULTI_LANGS if f"malicious_{l}" in cols],
        "jepa_mal": [f"malicious_{a}" for a in JEPA_ATTACKS if f"malicious_{a}" in cols],
        "jepa_ben": [f"benign_{a}" for a in JEPA_ATTACKS if f"benign_{a}" in cols],
        "rs1_ben": [c for c in cols if c.startswith("benign_") and c.split("_",1)[1] not in JEPA_ATTACKS],
    }


def md_section_per_lang(wide: pd.DataFrame, layer: int) -> str:
    """Per-language accuracy table for a given layer (raw + pred side by side)."""
    lines = []
    cols = [f"malicious_{l}" for l in MULTI_LANGS if f"malicious_{l}" in wide.columns]
    if not cols:
        return ""
    lines.append(f"### Per-language (multilingual malicious), layer {layer}")
    lines.append("")
    sub = wide.xs(layer, level="layer")[cols].round(3)
    sub.columns = [c.removeprefix("malicious_") for c in sub.columns]
    lines.append(sub.to_markdown())
    lines.append("")
    return "\n".join(lines)


def md_section_per_attack(wide: pd.DataFrame, layer: int) -> str:
    """Per-attack-style accuracy: malicious & benign side by side."""
    lines = []
    mal_cols = [f"malicious_{a}" for a in JEPA_ATTACKS if f"malicious_{a}" in wide.columns]
    ben_cols = [f"benign_{a}" for a in JEPA_ATTACKS if f"benign_{a}" in wide.columns]
    if not mal_cols and not ben_cols:
        return ""
    lines.append(f"### Per-attack-style (JEPA), layer {layer}")
    lines.append("")
    if mal_cols:
        lines.append(f"**Malicious-attack accuracy (label=1, higher is better):**")
        lines.append("")
        sub = wide.xs(layer, level="layer")[mal_cols].round(3)
        sub.columns = [c.removeprefix("malicious_") for c in sub.columns]
        lines.append(sub.to_markdown())
        lines.append("")
    if ben_cols:
        lines.append(f"**Benign-attack accuracy (label=0, higher = fewer false positives):**")
        lines.append("")
        sub = wide.xs(layer, level="layer")[ben_cols].round(3)
        sub.columns = [c.removeprefix("benign_") for c in sub.columns]
        lines.append(sub.to_markdown())
        lines.append("")
    return "\n".join(lines)


def md_section_per_family_delta(wide: pd.DataFrame, layer: int, fams: dict[str, list[str]]) -> str:
    """Δ(PRA − no-PRA) per family per (model, defense, tag)."""
    fam_means = pd.DataFrame({k: wide[v].mean(axis=1) for k, v in fams.items() if v})
    fam_means = pd.concat([wide[["id_val"]], fam_means], axis=1).round(3)
    sub = fam_means.xs(layer, level="layer")
    sub = sub[sub.index.get_level_values("variant") != "base"]
    no_pra = sub.xs("no_pra", level="variant")
    pra = sub.xs("pra", level="variant")
    delta = (pra - no_pra).round(3)
    lines = [f"### Δ(PRA − no-PRA) per family, layer {layer}", ""]
    lines.append(delta.to_markdown())
    lines.append("")
    return "\n".join(lines)


def md_section_pred_minus_raw(wide: pd.DataFrame, fams: dict[str, list[str]]) -> str:
    """For PRA cells: how much does the predictor projection change accuracy?"""
    if "pred" not in wide.index.get_level_values("tag"):
        return ""
    lines = ["## Δ(predictor-projected − raw), per layer × family", ""]
    fam_means = pd.DataFrame({k: wide[v].mean(axis=1) for k, v in fams.items() if v})
    fam_means = pd.concat([wide[["id_val"]], fam_means], axis=1).round(3)
    # Only PRA cells have a predictor that was actually trained.
    pra_only = fam_means[fam_means.index.get_level_values("variant") == "pra"]
    raw = pra_only.xs("raw", level="tag")
    pred = pra_only.xs("pred", level="tag")
    common = raw.index.intersection(pred.index)
    delta = (pred.loc[common] - raw.loc[common]).round(3)
    lines.append("Positive = predictor projection improves probe accuracy on this family.")
    lines.append("")
    lines.append(delta.to_markdown())
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument(
        "--in_dir",
        type=Path,
        default=Path("runs/experiments/headline_backup_rerun_probing/classify_wang"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("assets/figures/probe_wang_faithful"),
    )
    args = ap.parse_args()

    df_long = load_all(args.in_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(args.out_dir / "all_results_wang.csv", index=False)

    wide = build_wide(df_long)
    wide.to_csv(args.out_dir / "wide_table_wang.csv")

    fams = families(wide)

    md = []
    md.append("# Wang-faithful probe accuracy (linear SVM)")
    md.append("")
    md.append(
        "Direct port of Wang et al.'s `classify.py` "
        "(github.com/WangCheng0116/Why-Probe-Fails). "
        "**ID** = `malicious_beaver` + `benign_alpaca` + `benign_dolly` "
        "(stratified 80/20, seed=42, `StandardScaler`, "
        "`SVC(kernel='linear', probability=True)`). "
        "**OOD** = every other npy in the directory, transformed with the same scaler."
    )
    md.append("")
    md.append(
        "Two probes per cell × layer:"
    )
    md.append("- `tag=raw`: SVM directly on the model's last-token hidden state.")
    md.append(
        "- `tag=pred`: same SVM, but every state (train + OOD) is first projected "
        "through the cell's trained `jepa_predictor.pt`. Only PRA cells produce "
        "a meaningfully-trained predictor; no-PRA cells project through what is "
        "effectively a near-random initialization, included as a control."
    )
    md.append("")

    # Family-mean tables per layer
    fam_cols = ["id_val"] + list(fams.keys())
    fam_means = pd.DataFrame({k: wide[v].mean(axis=1) for k, v in fams.items() if v})
    fam_means = pd.concat([wide[["id_val"]], fam_means], axis=1).round(3)
    for layer in [-1, 25]:
        md.append(f"## Layer {layer} — family means")
        md.append("")
        if layer not in wide.index.get_level_values("layer"):
            continue
        md.append(fam_means.xs(layer, level="layer").to_markdown())
        md.append("")

    md.append("## Per-language and per-attack breakdowns")
    md.append("")
    for layer in [-1, 25]:
        if layer not in wide.index.get_level_values("layer"):
            continue
        md.append(md_section_per_lang(wide, layer))
        md.append(md_section_per_attack(wide, layer))

    md.append("## Δ (PRA − no-PRA), family-mean")
    md.append("")
    for layer in [-1, 25]:
        if layer not in wide.index.get_level_values("layer"):
            continue
        md.append(md_section_per_family_delta(wide, layer, fams))

    md.append(md_section_pred_minus_raw(wide, fams))

    # Combined-pair F1/FP metrics (linear vs MLP probe on balanced mal+benign halves)
    pair_csv = args.in_dir.parent.parent / "probe_wang_faithful" / "combined_pair_metrics.csv"
    # Try assets-side first (where combined_metrics_wang.py writes)
    asset_pair_csv = args.out_dir / "combined_pair_metrics.csv"
    if asset_pair_csv.exists():
        pair_csv = asset_pair_csv
    if pair_csv.exists():
        cm = pd.read_csv(pair_csv)
        md.append("## Combined-pair metrics (linear vs MLP probe on balanced mal+benign halves)")
        md.append("")
        md.append(
            "Per attack family the SVM is evaluated on a balanced test set "
            "(`malicious_<attack>` + `benign_<attack>`). Reports recall, "
            "false-positive rate, and F1. Linear = `SVC(kernel='linear')`. "
            "MLP = `MLPClassifier((512,128))` early-stopped."
        )
        md.append("")
        for layer in [-1, 25]:
            sub = cm[cm["layer"] == layer]
            if sub.empty:
                continue
            md.append(f"### Layer {layer} — F1 per attack")
            md.append("")
            piv = sub.pivot_table(index=["cell", "probe"], columns="attack", values="f1").round(3)
            md.append(piv.to_markdown())
            md.append("")
            md.append(f"### Layer {layer} — FP rate per attack (lower = less over-refusal)")
            md.append("")
            piv = sub.pivot_table(index=["cell", "probe"], columns="attack", values="fp_rate").round(3)
            md.append(piv.to_markdown())
            md.append("")
            md.append(f"### Layer {layer} — Mean F1 + Mean FP, all 4 JEPA attacks")
            md.append("")
            agg = sub.groupby(["cell", "probe"]).agg(
                mean_f1=("f1", "mean"),
                mean_fp=("fp_rate", "mean"),
                mean_recall=("mal_recall", "mean"),
            ).round(3)
            md.append(agg.to_markdown())
            md.append("")

    md.append("## Full pivot")
    md.append("")
    md.append("```")
    md.append("see wide_table_wang.csv")
    md.append("```")

    out_md = args.out_dir / "WANG_FAITHFUL_RESULTS.md"
    out_md.write_text("\n".join(md))
    print(f"[saved] {out_md}")
    print(f"[saved] {args.out_dir / 'wide_table_wang.csv'}")
    print(f"[saved] {args.out_dir / 'all_results_wang.csv'}")


if __name__ == "__main__":
    main()
