"""V2 probe analysis (unified-anchor protocol).

Train Wang's SVM ONCE per (variant, state) on:
  positive = 80% English AdvBench malicious
  negative = 80% English Alpaca benign

Then evaluate every OOD slice with the SAME anchor:
  positive = OOD malicious  (paraphrased AdvBench, paraphrased HarmBench,
              multi_<lang> malicious, jepa_<family> malicious, harmbench cleaned, ...)
  negative = held-out 20% English Alpaca benign

This eliminates the type confound from the wow2000 multilingual benigns (which were
model-safety responses, not user queries), and gives us a single common reference
distribution against which to measure probe transfer along every OOD axis.

Outputs (under runs/why_probe_fails_headline_n25/v2/):
  - SUMMARY_v2.csv        long-format: variant, state, slice, n, auc
  - paraphrase_table.csv  Para. HB pivot (encoder x state)
  - attacks_table.csv     JepaData attack families pivot (encoder x family x state)
  - multi_table.csv       Multilingual pivot (encoder x lang x state)
  - scaling.csv           Scaling sweep at multiple train sizes (Para. HB only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DEFAULT_ROOT = Path("runs/why_probe_fails_headline_n25")
ROOT = DEFAULT_ROOT  # overridden via --root
OUT = ROOT / "v2"
SEED = 42

VARIANTS_ORDER = [
    "base_llama", "base_qwen",
    "l_cb_no_pra", "l_cb_pra",
    "l_triplet_no_pra", "l_triplet_pra",
    "l_ce_in_no_pra", "l_ce_in_pra",
    "q_cb_no_pra", "q_cb_pra", "q_cb_pra_withresp",
    "q_triplet_no_pra", "q_triplet_pra", "q_triplet_pra_withresp",
    "q_ce_in_no_pra", "q_ce_in_pra",
]

LANGS = ["en", "ar", "cs", "es", "fy", "id", "ja", "pt", "zh-cn"]
JEPA_FAMILIES = ["direct", "prefilling", "encoding", "bon"]  # families with >100 rows


# OOD slices to evaluate. Each entry: (slice_name, npy_filename_in_states_dir)
PARAPHRASE_SLICES = [
    ("paraphrased_advbench", "paraphrased_advbench.npy"),
    ("malicious_harmbench", "malicious_harmbench.npy"),
    ("cleaned_harmbench", "cleaned_harmbench.npy"),
    ("paraphrased_harmbench", "paraphrased_harmbench.npy"),
]


def load_npy(p: Path) -> Optional[np.ndarray]:
    if not p.exists():
        return None
    return np.load(p)


def fit_wang(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> Tuple[StandardScaler, SVC]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    svm = SVC(kernel="linear", random_state=seed, probability=True)
    svm.fit(Xs, y)
    return scaler, svm


def predict_proba(scaler: StandardScaler, svm: SVC, X: np.ndarray) -> np.ndarray:
    return svm.predict_proba(scaler.transform(X))[:, 1]


def evaluate_variant(states_dir: Path) -> List[Dict]:
    """Train Wang's SVM on the variant's adv+alpaca states, eval all OOD slices.
    Returns a list of {variant, slice, n_pos, n_neg, auc} dicts (variant filled by caller)."""
    if not states_dir.exists():
        return []
    adv_mal = load_npy(states_dir / "malicious_advbench.npy")
    alp_ben = load_npy(states_dir / "benign_alpaca.npy")
    if adv_mal is None or alp_ben is None:
        return []

    rng = np.random.default_rng(SEED)
    pos_perm = rng.permutation(adv_mal.shape[0])
    neg_perm = rng.permutation(alp_ben.shape[0])
    n_pos_tr = int(0.8 * adv_mal.shape[0])
    n_neg_tr = int(0.8 * alp_ben.shape[0])
    Xpos_tr = adv_mal[pos_perm[:n_pos_tr]]
    Xneg_tr = alp_ben[neg_perm[:n_neg_tr]]
    Xpos_val = adv_mal[pos_perm[n_pos_tr:]]
    Xneg_val = alp_ben[neg_perm[n_neg_tr:]]

    X_tr = np.vstack([Xpos_tr, Xneg_tr])
    y_tr = np.concatenate([np.ones(Xpos_tr.shape[0]), np.zeros(Xneg_tr.shape[0])])
    scaler, svm = fit_wang(X_tr, y_tr)

    rows: List[Dict] = []

    def add(slice_name: str, X_pos: np.ndarray, X_neg: np.ndarray):
        n = min(X_pos.shape[0], X_neg.shape[0])
        if n == 0:
            return
        rng2 = np.random.default_rng(SEED + abs(hash(slice_name)) % 10**8)
        idx_p = rng2.permutation(X_pos.shape[0])[:n]
        idx_n = rng2.permutation(X_neg.shape[0])[:n]
        X_eval = np.vstack([X_pos[idx_p], X_neg[idx_n]])
        y_eval = np.concatenate([np.ones(n), np.zeros(n)])
        scores = predict_proba(scaler, svm, X_eval)
        rows.append({
            "slice": slice_name,
            "n_pos": int(n), "n_neg": int(n),
            "auc": float(roc_auc_score(y_eval, scores)),
        })

    # ID val
    add("id_val", Xpos_val, Xneg_val)

    # Paraphrasing slices (ALL pos types use Xneg_val as the negative anchor)
    for slice_name, fname in PARAPHRASE_SLICES:
        Xp = load_npy(states_dir / fname)
        if Xp is not None:
            add(slice_name, Xp, Xneg_val)

    # Multilingual: positive = multi_<lang> malicious; negative = Xneg_val
    for lang in LANGS:
        Xp = load_npy(states_dir / f"malicious_multi_{lang}.npy")
        if Xp is not None:
            add(f"multi_{lang}", Xp, Xneg_val)

    # JepaData attack families
    for fam in JEPA_FAMILIES:
        Xp = load_npy(states_dir / f"malicious_jepa_{fam}.npy")
        if Xp is not None:
            add(f"jepa_{fam}", Xp, Xneg_val)

    return rows


def aggregate_all() -> pd.DataFrame:
    out_rows: List[Dict] = []
    for state_dirname, state in [("states", "raw"), ("states_predictor", "predictor")]:
        state_root = ROOT / state_dirname
        if not state_root.exists():
            continue
        for variant in VARIANTS_ORDER:
            sd = state_root / variant
            rows = evaluate_variant(sd)
            for r in rows:
                r["variant"] = variant
                r["state"] = state
                out_rows.append(r)
    return pd.DataFrame(out_rows)


# Pretty pivots ----------------------------------------------------------------

def pivot_paraphrase(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["slice"].isin([s for s, _ in PARAPHRASE_SLICES] + ["id_val"])]
    p = sub.pivot_table(values="auc", index="variant", columns=["slice", "state"]).round(3)
    return p.reindex([v for v in VARIANTS_ORDER if v in p.index])


def pivot_attacks(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["slice"].str.startswith("jepa_")]
    p = sub.pivot_table(values="auc", index="variant", columns=["slice", "state"]).round(3)
    return p.reindex([v for v in VARIANTS_ORDER if v in p.index])


def pivot_multi(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["slice"].str.startswith("multi_")].copy()
    sub["lang"] = sub["slice"].str.removeprefix("multi_")
    p = sub.pivot_table(values="auc", index="variant", columns=["lang", "state"]).round(3)
    return p.reindex([v for v in VARIANTS_ORDER if v in p.index])


# Scaling sweep ----------------------------------------------------------------

SCALING_SIZES = [50, 100, 200, 416]
SCALING_VARIANTS = [
    ("base_llama",  "raw"),
    ("l_cb_no_pra", "raw"),
    ("l_cb_pra",    "raw"),
    ("l_cb_pra",    "predictor"),
    ("q_cb_pra",    "raw"),
    ("q_cb_pra",    "predictor"),
]


def scaling_sweep() -> pd.DataFrame:
    rows: List[Dict] = []
    for variant, state in SCALING_VARIANTS:
        states_dir = ROOT / ("states_predictor" if state == "predictor" else "states") / variant
        if not states_dir.exists():
            continue
        adv = load_npy(states_dir / "malicious_advbench.npy")
        alp = load_npy(states_dir / "benign_alpaca.npy")
        hb_par = load_npy(states_dir / "paraphrased_harmbench.npy")
        if any(x is None for x in [adv, alp, hb_par]):
            continue
        rng = np.random.default_rng(SEED)
        pos_perm = rng.permutation(adv.shape[0])
        neg_perm = rng.permutation(alp.shape[0])
        # Hold out 20% alpaca for the eval anchor
        n_neg_tr_max = int(0.8 * alp.shape[0])
        Xneg_anchor = alp[neg_perm[n_neg_tr_max:]]
        for n in SCALING_SIZES:
            if n > min(adv.shape[0], alp.shape[0]):
                continue
            Xpos_tr = adv[pos_perm[:n]]
            Xneg_tr = alp[neg_perm[:n]]
            X_tr = np.vstack([Xpos_tr, Xneg_tr])
            y_tr = np.concatenate([np.ones(n), np.zeros(n)])
            scaler, svm = fit_wang(X_tr, y_tr)
            n_eval = min(hb_par.shape[0], Xneg_anchor.shape[0])
            X_eval = np.vstack([hb_par[:n_eval], Xneg_anchor[:n_eval]])
            y_eval = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])
            scores = predict_proba(scaler, svm, X_eval)
            rows.append({
                "variant": variant, "state": state, "train_pos_per_class": n,
                "n_eval": 2 * n_eval,
                "auc": float(roc_auc_score(y_eval, scores)),
            })
    return pd.DataFrame(rows)


# Main -------------------------------------------------------------------------

def main() -> None:
    global ROOT, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variants", type=str, default=None,
                    help="Comma-separated subset of variant names to evaluate (default: VARIANTS_ORDER).")
    args = ap.parse_args()
    ROOT = args.root
    if args.out is None:
        args.out = ROOT / "v2"
    args.out.mkdir(parents=True, exist_ok=True)
    if args.variants:
        global VARIANTS_ORDER
        VARIANTS_ORDER = args.variants.split(",")

    print("[1/4] Aggregating unified-anchor probes across all OOD axes...")
    df = aggregate_all()
    df.to_csv(args.out / "SUMMARY_v2.csv", index=False)
    print(f"  saved {args.out / 'SUMMARY_v2.csv'} ({len(df)} rows)")

    if df.empty:
        print("  no rows — nothing more to do.")
        return

    print("\n[2/4] Paraphrase + ID slices (encoder x slice x state):")
    pp = pivot_paraphrase(df)
    pp.to_csv(args.out / "paraphrase_table.csv")
    print(pp.to_string())

    print("\n[3/4] JepaData attack families (encoder x family x state):")
    pa = pivot_attacks(df)
    if not pa.empty:
        pa.to_csv(args.out / "attacks_table.csv")
        print(pa.to_string())
    else:
        print("  (no jepa_* slices yet — JepaData probe still extracting)")

    print("\n[4/4] Multilingual (encoder x lang x state):")
    pm = pivot_multi(df)
    pm.to_csv(args.out / "multi_table.csv")
    # Add a non-EN summary
    sub = df[df["slice"].str.startswith("multi_")].copy()
    sub["lang"] = sub["slice"].str.removeprefix("multi_")
    nonen = sub[sub["lang"] != "en"]
    summary = nonen.groupby(["variant", "state"])["auc"].agg(
        nonEN_mean="mean", nonEN_worst="min"
    ).round(3).reset_index()
    summary_pivot = summary.pivot_table(values=["nonEN_mean", "nonEN_worst"],
                                        index="variant", columns="state").round(3)
    summary_pivot = summary_pivot.reindex([v for v in VARIANTS_ORDER if v in summary_pivot.index])
    summary_pivot.to_csv(args.out / "multi_summary.csv")
    print(summary_pivot.to_string())

    print("\n[bonus] Scaling sweep on Para. HB:")
    sc = scaling_sweep()
    sc.to_csv(args.out / "scaling.csv", index=False)
    if not sc.empty:
        sp = sc.pivot_table(values="auc", index="train_pos_per_class",
                            columns=["variant", "state"]).round(3)
        print(sp.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
