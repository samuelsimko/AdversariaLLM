"""Wang-faithful pipeline addendum: combined-pair metrics (F1, FP rate).

Single-slice accuracy is misleading on label-imbalanced OOD slices (e.g. a
benign-only `benign_inpainting` slice has no positives, so precision/recall
degenerate). For paired families like the JEPA attacks (each has a
`malicious_<attack>` and `benign_<attack>` npy), we can compute proper F1
and FP rate by combining the two halves into a balanced binary test set.

Per cell × layer × probe(=linear|mlp) we report, for each paired family:
  - mal_recall = TP / (TP + FN)
  - fp_rate    = FP / (FP + TN)
  - f1
  - balanced_acc

ID training matches Wang's protocol: stratified 80/20, seed=42, StandardScaler.
The linear probe is `SVC(kernel='linear')`. The MLP probe is sklearn's
`MLPClassifier` with hidden=(512,128), early-stopping. Both are deliberately
simple — the goal is to read off rep-space geometry, not to maximize the
classifier itself.

Usage:
    python scripts/probe_train_size/combined_metrics_wang.py \\
        --states_root runs/experiments/headline_backup_rerun_probing/states_wang \\
        --out_csv     assets/figures/probe_wang_faithful/combined_pair_metrics.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


JEPA_ATTACKS = ("encoding", "inpainting", "persona", "prefilling")
DEFAULT_CELLS = (
    "l_cb_pra", "l_cb_no_pra", "l_triplet_pra", "l_triplet_no_pra",
    "q_cb_pra", "q_cb_no_pra", "q_triplet_pra", "q_triplet_no_pra",
    "l_base", "q_base",
)
DEFAULT_LAYERS = (-1, 25)


def load(d: Path, stem: str) -> np.ndarray:
    p = d / f"{stem}.npy"
    return np.load(p) if p.exists() else None


def fit_probe(probe_kind: str, X_tr: np.ndarray, y_tr: np.ndarray, seed: int):
    if probe_kind == "linear":
        m = SVC(kernel="linear", random_state=seed)
    elif probe_kind == "mlp":
        m = MLPClassifier(
            hidden_layer_sizes=(512, 128),
            max_iter=200,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
    else:
        raise ValueError(f"unknown probe: {probe_kind}")
    return m.fit(X_tr, y_tr)


def metrics_pair(model, scaler: StandardScaler, X_mal: np.ndarray, X_ben: np.ndarray) -> dict:
    Xm = scaler.transform(X_mal)
    Xb = scaler.transform(X_ben)
    yhat_m = model.predict(Xm)
    yhat_b = model.predict(Xb)
    mal_recall = float((yhat_m == 1).mean())
    fp_rate = float((yhat_b == 1).mean())
    X = np.vstack([Xm, Xb])
    y = np.concatenate([np.ones(Xm.shape[0]), np.zeros(Xb.shape[0])])
    yhat = np.concatenate([yhat_m, yhat_b])
    return dict(
        mal_recall=mal_recall,
        fp_rate=fp_rate,
        f1=float(f1_score(y, yhat)),
        balanced_acc=float(accuracy_score(y, yhat)),
        n_mal=int(Xm.shape[0]),
        n_ben=int(Xb.shape[0]),
    )


def run_cell_layer(
    states_dir: Path,
    cell: str,
    layer: int,
    id_mal: list[str],
    id_ben: list[str],
    probes: Iterable[str],
    seed: int,
    test_size: float,
) -> list[dict]:
    Xs, ys = [], []
    for s in id_mal:
        a = load(states_dir, s)
        if a is None:
            return []
        Xs.append(a); ys.append(np.full(a.shape[0], 1))
    for s in id_ben:
        a = load(states_dir, s)
        if a is None:
            return []
        Xs.append(a); ys.append(np.full(a.shape[0], 0))
    X_id = np.vstack(Xs); y_id = np.concatenate(ys)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_id, y_id, test_size=test_size, stratify=y_id, random_state=seed
    )
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_val_s = sc.transform(X_val)

    rows: list[dict] = []
    for probe in probes:
        model = fit_probe(probe, X_tr_s, y_tr, seed)
        id_val_acc = float(accuracy_score(y_val, model.predict(X_val_s)))

        for attack in JEPA_ATTACKS:
            X_mal = load(states_dir, f"malicious_{attack}")
            X_ben = load(states_dir, f"benign_{attack}")
            if X_mal is None or X_ben is None:
                continue
            m = metrics_pair(model, sc, X_mal, X_ben)
            rows.append({
                "cell": cell, "layer": layer, "probe": probe,
                "family": "jepa", "attack": attack,
                "id_val_acc": id_val_acc, **m,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument(
        "--states_root", type=Path,
        default=Path("runs/experiments/headline_backup_rerun_probing/states_wang"),
    )
    ap.add_argument(
        "--out_csv", type=Path,
        default=Path("assets/figures/probe_wang_faithful/combined_pair_metrics.csv"),
    )
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    ap.add_argument("--id_malicious", nargs="+", default=["malicious_beaver"])
    ap.add_argument(
        "--id_benign", nargs="+",
        default=["benign_alpaca", "benign_dolly"],
    )
    ap.add_argument("--probes", nargs="+", default=["linear", "mlp"], choices=["linear", "mlp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)
    args = ap.parse_args()

    rows: list[dict] = []
    for cell in args.cells:
        for layer in args.layers:
            d = args.states_root / f"L{layer}" / cell
            if not d.exists():
                print(f"[skip] {cell} L{layer} (no states dir)")
                continue
            rs = run_cell_layer(
                d, cell, layer,
                args.id_malicious, args.id_benign,
                args.probes, args.seed, args.test_size,
            )
            if not rs:
                print(f"[skip] {cell} L{layer} (missing ID files)")
                continue
            rows.extend(rs)
            print(f"[done] {cell} L{layer} ({len(rs)} rows)")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
