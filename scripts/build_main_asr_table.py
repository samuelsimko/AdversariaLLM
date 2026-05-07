"""Generate the paper main-table LaTeX from /workspace/headline-rerun ASR data.

Reads /workspace/headline-rerun/asr_summary.csv (produced by
analyze_headline_rerun_asr.py). For each (model, regularizer) pair,
emits a no-alignment / PRA pair of rows with deltas color-coded:
  green ↓ when PRA reduces ASR (defense improvement),
  red   ↑ when PRA increases ASR (regression).

SoftPrompt is reported on the final n=200 batch (soft_prompt_101_200).
Avg is computed only over attacks present for *both* the no-PRA and PRA
rows; if any required attack is missing for either, prints '---'.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# Map our cell prefix -> (display section header, sort order)
SECTIONS = [
    ("cb",      "\\textbf{Circuit Breakers}"),
    ("ce_in",   "\\textbf{CE-floor (target included)}"),
    ("triplet", "\\textbf{Triplet}"),
]

ATTACKS = [
    ("bon_50",              "BoN"),
    ("direct_300",          "Direct"),
    ("inpainting_50",       "Inpaint"),
    ("prefilling_300",      "Prefill"),
    # SoftPrompt N=200 = soft_prompt_100 (idx 1-100) + soft_prompt_101_200 (idx 101-200)
    # combined into a single ASR over 200 behaviors. Handled specially in get_asr.
    ("soft_prompt_n200",    "SoftPrompt"),
]


def fmt_delta(no_pra: float, pra: float) -> str:
    """Returns the LaTeX delta marker. 'no_pra' / 'pra' are percentages (0..100)."""
    diff = pra - no_pra
    if abs(diff) < 0.05:
        return ""  # treat as no change
    if diff < 0:  # PRA reduced ASR -> good
        return f"\\,\\dgood{{{abs(diff):.1f}}}"
    else:  # PRA increased ASR -> bad
        return f"\\,\\dbad{{{abs(diff):.1f}}}"


def fmt_cell(asr: float | None, no_pra_asr: float | None = None) -> str:
    if asr is None:
        return "---"
    pct = asr * 100.0
    if no_pra_asr is None:
        return f"{pct:.1f}"
    return f"{pct:.1f}{fmt_delta(no_pra_asr * 100.0, pct)}"


def get_asr(df: pd.DataFrame, cell: str, attack: str) -> float | None:
    if attack == "soft_prompt_n200":
        # Combine the two batches into a single ASR over n=200 behaviors.
        # Requires BOTH halves to be present and reasonably complete (>=50 each).
        a = df[(df["cell"] == cell) & (df["attack"] == "soft_prompt_100")]
        b = df[(df["cell"] == cell) & (df["attack"] == "soft_prompt_101_200")]
        if a.empty or b.empty:
            return None
        na, sa = int(a["n"].iloc[0]), int(a.get("n_succ", a["asr"] * a["n"]).iloc[0])
        nb, sb = int(b["n"].iloc[0]), int(b.get("n_succ", b["asr"] * b["n"]).iloc[0])
        if na < 50 or nb < 50:
            return None  # partial run, suppress
        return (sa + sb) / (na + nb)
    sub = df[(df["cell"] == cell) & (df["attack"] == attack)]
    if sub.empty:
        return None
    v = sub["asr"].iloc[0]
    n = sub["n"].iloc[0]
    if pd.isna(v) or n == 0:
        return None
    # Suppress single-attack ASR if n is too small (run incomplete).
    expected = {"bon_50": 50, "direct_300": 300, "inpainting_50": 50, "prefilling_300": 300}
    if attack in expected and n < expected[attack] // 2:
        return None
    return float(v)


def avg_or_dash(values: list[float | None]) -> float | None:
    """Average of a list of ASRs; returns None if any is None (need full row)."""
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)


def render_section(df: pd.DataFrame, model_letter: str, model_name: str) -> str:
    lines: list[str] = []
    lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{\\textbf{{{model_name}}}}}}} \\\\")
    lines.append("Base                                            & ---  & --- & --- & --- & ---           & ---  \\\\")
    lines.append("Triplet \\citep{simko2025improving}              & ---  & --- & --- & --- & ---           & ---  \\\\")
    if model_letter == "l":
        lines.append("SafeSwitch \\\\")
    lines.append("\\midrule")

    for j, (reg, header) in enumerate(SECTIONS):
        cell_no = f"{model_letter}_{reg}_no_pra"
        cell_pra = f"{model_letter}_{reg}_pra"

        no_vals = [get_asr(df, cell_no, atk) for atk, _ in ATTACKS]
        pra_vals = [get_asr(df, cell_pra, atk) for atk, _ in ATTACKS]

        no_avg = avg_or_dash(no_vals)
        pra_avg = avg_or_dash(pra_vals)

        # No-PRA row
        no_cells = [fmt_cell(v) for v in no_vals]
        no_avg_cell = fmt_cell(no_avg)

        # PRA row (with deltas referencing no-PRA)
        pra_cells = [fmt_cell(p, n) for p, n in zip(pra_vals, no_vals)]
        pra_avg_cell = fmt_cell(pra_avg, no_avg)

        lines.append(f"\\multicolumn{{7}}{{l}}{{{header}}} \\\\")
        lines.append(
            "\\quad $\\triangleright$ No alignment             & "
            + " & ".join(no_cells)
            + f" & {no_avg_cell}  \\\\"
        )
        lines.append(
            "\\quad $\\triangleright$ PRA ($K{=}256$)          & "
            + " & ".join(pra_cells)
            + f" & {pra_avg_cell}  \\\\"
        )
        if j < len(SECTIONS) - 1:
            lines.append("\\midrule")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--in_csv", default="/workspace/headline-rerun/asr_summary.csv", type=Path)
    ap.add_argument("--out_tex", default="/workspace/headline-rerun/main_asr_table.tex", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    preamble = (
        r"\definecolor{deltagood}{HTML}{2E7D32}" + "\n" +
        r"\definecolor{deltabad}{HTML}{C62828}" + "\n" +
        r"\newcommand{\dgood}[1]{{\scriptsize\textcolor{deltagood}{$\downarrow$#1}}}" + "\n" +
        r"\newcommand{\dbad}[1]{{\scriptsize\textcolor{deltabad}{$\uparrow$#1}}}" + "\n\n"
    )

    table_open = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{Attack success rate (\%) on \textsc{Llama-3-8B} and \textsc{Qwen3-8B} across "
        r"attack categories, evaluated under StrongREJECT validated dual-context. SoftPrompt is "
        r"the combined $n{=}200$ ASR over both batches (idx 1-200). Cells with ``---'' have "
        r"incomplete or missing attack runs. "
        r"$\downarrow$ green = PRA reduces ASR, $\uparrow$ red = PRA increases ASR.}" + "\n"
        r"\small" + "\n"
        r"\setlength{\tabcolsep}{4pt}" + "\n"
        r"\begin{tabular}{l c c c c c c}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Defense} & \textsc{BoN} & \textsc{Direct} & \textsc{Inpaint} & "
        r"\textsc{Prefill} & \textsc{SoftPrompt} & \textbf{avg} \\" + "\n"
        r"\midrule" + "\n"
    )

    table_close = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\label{tab:pra_attack_asr}" + "\n"
        r"\end{table}" + "\n"
    )

    body = []
    body.append(render_section(df, "l", "Llama-3-8B"))
    body.append(r"\midrule")
    body.append(render_section(df, "q", "Qwen3-8B"))

    full = preamble + table_open + "\n".join(body) + "\n" + table_close
    args.out_tex.write_text(full)
    print(full)
    print(f"\n[saved] {args.out_tex}")


if __name__ == "__main__":
    main()
