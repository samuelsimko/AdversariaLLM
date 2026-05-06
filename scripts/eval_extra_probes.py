"""Unified-anchor SVM eval over multi + jepa slices.

Reuses the AdvBench-en malicious states (ID positives) and the Alpaca-en
benign states (ID negatives) that scripts/run_probing_headline_backup.sh
already extracted. Trains one SVM per cell, evaluates on each multi_<lang>
and jepa_<family> .npy slice. AUC against the same held-out Alpaca-en
benigns each time, so deltas across slices are clean attribution to the
slice itself.

Outputs:
    <out_dir>/multi.csv     columns: lang, n, auc
    <out_dir>/jepa.csv      columns: family, view, n, auc
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load(p: Path) -> np.ndarray | None:
    return np.load(p) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--jepa_fams", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ID anchor: AdvBench-en mal vs Alpaca-en benign
    pos = _load(args.states_dir / "malicious_advbench.npy")
    neg = _load(args.states_dir / "benign_alpaca.npy")
    if pos is None or neg is None:
        raise SystemExit(f"missing anchor states under {args.states_dir}")

    # 80/20 split
    def _split(x):
        idx = rng.permutation(len(x))
        cut = int(len(x) * (1 - args.val_frac))
        return x[idx[:cut]], x[idx[cut:]]
    pos_tr, pos_va = _split(pos)
    neg_tr, neg_va = _split(neg)

    X_tr = np.concatenate([pos_tr, neg_tr], 0)
    y_tr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
    X_va_neg = neg_va  # held-out benigns reused for every slice's AUC

    scaler = StandardScaler().fit(X_tr)
    clf = SVC(kernel="rbf", probability=False, gamma="scale")
    clf.fit(scaler.transform(X_tr), y_tr)

    def _auc(slice_pos: np.ndarray) -> float:
        Xs = np.concatenate([slice_pos, X_va_neg], 0)
        ys = np.concatenate([np.ones(len(slice_pos)), np.zeros(len(X_va_neg))])
        scores = clf.decision_function(scaler.transform(Xs))
        return float(roc_auc_score(ys, scores))

    # Multi
    multi_csv = args.out_dir / "multi.csv"
    with multi_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["lang", "n", "auc"])
        for lang in args.langs:
            arr = _load(args.states_dir / f"malicious_{lang}.npy")
            if arr is None:
                w.writerow([lang, 0, ""]); continue
            w.writerow([lang, len(arr), f"{_auc(arr):.4f}"])
    print(f"wrote {multi_csv}")

    # JEPA families × {malicious, benign}
    jepa_csv = args.out_dir / "jepa.csv"
    with jepa_csv.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "view", "n", "auc"])
        for fam in args.jepa_fams:
            for view in ("malicious", "benign"):
                arr = _load(args.states_dir / f"{view}_{fam}.npy")
                if arr is None:
                    w.writerow([fam, view, 0, ""]); continue
                w.writerow([fam, view, len(arr), f"{_auc(arr):.4f}"])
    print(f"wrote {jepa_csv}")


if __name__ == "__main__":
    main()
