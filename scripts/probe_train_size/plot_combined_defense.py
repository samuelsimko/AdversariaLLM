"""Plot probe+model combined-defense ASR for all 8 (model × harm-reg × pra) cells.

Combined defense: count an attack as a success only if BOTH the model failed
(SR judge > 0.5) AND the probe did not flag it (probe_score <= T). This is the
"defense in depth" view: the system fails only when both layers fail.

Figure: 1 row × 2 columns (one panel per base model), x = FPR target on benigns,
y = combined ASR. Two lines per panel: CB and CE-floor. Solid = no_pra,
dashed = pra. Markers at FPR ∈ {1, 5, 10, 20}%.

Also prints a wide table comparing vanilla vs combined ASR per cell.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "runs/experiments/headline_backup_rerun_probing/probe_gate_full"
OUT = REPO / "assets/figures/probe_gate_combined"

CELLS = ["q_cb_no_pra", "q_cb_pra", "q_ce_in_no_pra", "q_ce_in_pra",
         "l_cb_no_pra", "l_cb_pra", "l_ce_in_no_pra", "l_ce_in_pra"]


def load_fpr_asr(cell: str) -> pd.DataFrame | None:
    p = ROOT / f"{cell}_fpr_asr.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def cell_meta(cell: str) -> dict:
    parts = cell.split("_")
    model = "Llama-3-8B" if parts[0] == "l" else "Qwen3-8B"
    if "cb" in cell:
        reg = "circuit_breaker"
    elif "ce_in" in cell:
        reg = "ce_floor"
    else:
        reg = "triplet"
    pra = "pra" if cell.endswith("_pra") and "no_pra" not in cell else "no_pra"
    return {"model": model, "reg": reg, "pra": pra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all cells ----
    rows = []
    for cell in CELLS:
        df = load_fpr_asr(cell)
        if df is None:
            print(f"[skip] {cell} (missing fpr_asr.csv)")
            continue
        meta = cell_meta(cell)
        for _, r in df.iterrows():
            rows.append({**meta, "cell": cell,
                         "fpr_target": r.fpr_target,
                         "fpr_actual": r.actual_fpr,
                         "or_bench_rejection": r.or_bench_rejection,
                         "gsm8k_rejection": r.gsm8k_rejection,
                         "vanilla_asr": r.vanilla_asr,
                         "gated_asr": r.gated_asr})
    big = pd.DataFrame(rows)
    print(f"[load] {big.cell.nunique()} cells")

    # ---- Wide table ----
    print("\n=== Combined (probe + model) ASR at various FPR targets ===\n")
    piv = big.pivot_table(
        index=["model", "reg", "pra"],
        columns="fpr_target",
        values="gated_asr",
    ).round(4) * 100
    piv.columns = [f"FPR={int(c*100)}%" for c in piv.columns]
    print(piv.to_string())

    # Side-by-side comparison: at FPR=10%, vanilla vs gated
    print("\n=== Vanilla vs combined at FPR=10% ===\n")
    f10 = big[big.fpr_target == 0.1].copy()
    f10["combined_asr"] = f10.gated_asr * 100
    f10["vanilla_asr_pct"] = f10.vanilla_asr * 100
    f10["delta"] = f10["combined_asr"] - f10["vanilla_asr_pct"]
    print(f10[["model", "reg", "pra", "vanilla_asr_pct", "combined_asr", "delta"]]
          .round(2).to_string(index=False))

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    cmap = {"circuit_breaker": "C0", "ce_floor": "C1"}
    for ax, model in zip(axes, ["Llama-3-8B", "Qwen3-8B"]):
        sub = big[big.model == model].copy()
        for reg in ["circuit_breaker", "ce_floor"]:
            for pra, ls in [("no_pra", "-"), ("pra", "--")]:
                d = sub[(sub.reg == reg) & (sub.pra == pra)].sort_values("fpr_target")
                if d.empty:
                    continue
                ax.plot(d.fpr_actual * 100, d.gated_asr * 100, marker="o",
                        linestyle=ls, color=cmap[reg],
                        label=f"{reg.replace('_', '-')} {pra}",
                        linewidth=2)
        ax.set_title(model)
        ax.set_xlabel("FPR on benigns (or_bench + gsm8k) [%]")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Combined ASR (probe + model fail) [%]")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Stacked-defense ASR: probe-gating + model refusal\n"
                 "(success = SR judge > 0.5 AND probe_score ≤ T)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out_dir / "combined_asr_by_cell.png", dpi=130)
    print(f"\n[saved] {args.out_dir / 'combined_asr_by_cell.png'}")

    # Bar chart at FPR=10%
    fig2, ax = plt.subplots(figsize=(11, 5))
    f10 = big[big.fpr_target == 0.1].copy()
    f10["label"] = f10.model.str[0].str.lower() + "_" + f10.reg.str[:2] + "_" + f10.pra
    f10 = f10.sort_values(["model", "reg", "pra"])
    width = 0.35
    x = np.arange(len(f10) // 2)
    no_pra = f10[f10.pra == "no_pra"].reset_index(drop=True)
    pra = f10[f10.pra == "pra"].reset_index(drop=True)
    for i, row in no_pra.iterrows():
        ax.bar(i - width/2, row.vanilla_asr * 100, width, color="lightgray",
               edgecolor="k", linewidth=0.5,
               label="no_pra vanilla" if i == 0 else None)
        ax.bar(i - width/2, row.gated_asr * 100, width, color="C0",
               label="no_pra combined" if i == 0 else None, edgecolor="k", linewidth=0.5)
    for i, row in pra.iterrows():
        ax.bar(i + width/2, row.vanilla_asr * 100, width, color="lightgray",
               edgecolor="k", linewidth=0.5)
        ax.bar(i + width/2, row.gated_asr * 100, width, color="C3",
               label="pra combined" if i == 0 else None, edgecolor="k", linewidth=0.5,
               hatch="//")
    labels = [f"{r.model.split('-')[0]}\n{r.reg.replace('_','-')}"
              for _, r in no_pra.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ASR @ FPR=10% on benigns [%]")
    ax.set_title("Probe+Model combined defense at FPR=10%\n(gray bar = vanilla, colored = combined)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "bars_combined_at_fpr10.png", dpi=130)
    print(f"[saved] {args.out_dir / 'bars_combined_at_fpr10.png'}")


if __name__ == "__main__":
    main()
