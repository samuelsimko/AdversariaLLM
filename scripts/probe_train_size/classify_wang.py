"""Wang-faithful probe classifier.

Direct transcription of Wang et al.'s classify.py logic
(github.com/WangCheng0116/Why-Probe-Fails). For each (cell, layer) we:

  1. Load chosen ID malicious + ID benign npys.
  2. Stratified 80/20 split (seed=42), StandardScaler, linear SVC.
  3. Eval on the 20% ID val.
  4. Eval on every other npy in the directory as OOD, transformed with the
     same scaler.
  5. Write one results.csv per cell, with extra cell/layer/dataset metadata
     columns so downstream pivots stay easy.

Discovery convention follows extract_states.py output: filenames look like
``<view>_<dataset>.npy`` where ``view`` is in {benign, malicious, cleaned,
paraphrased}. Wang's classify.py keys ``label = 1 if name.startswith('malicious_')
else 0``; we replicate that, treating ``cleaned_*`` and ``paraphrased_*`` as
OOD malicious (label=1) following Wang's RS3 protocol.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


HARMFUL_PREFIXES = ("malicious_", "cleaned_", "paraphrased_")
BENIGN_PREFIXES = ("benign_",)


def label_for(name: str) -> int | None:
    if name.startswith(HARMFUL_PREFIXES):
        return 1
    if name.startswith(BENIGN_PREFIXES):
        return 0
    return None


def load_npy(p: Path) -> np.ndarray:
    arr = np.load(p)
    if arr.ndim != 2:
        raise ValueError(f"{p} not 2-D, got {arr.shape}")
    return arr


def list_files(states_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in states_dir.glob("*.npy")}


def build_predictor(dim: int, num_layers: int, bottleneck_dim: int, dropout: float):
    """Mirror PerPositionPredictor(predictor_type='mlp') from defenses/jepa_ce.py."""
    import torch.nn as nn

    if num_layers < 2:
        raise ValueError("predictor num_layers must be >= 2 for type=mlp")
    layers: list = [nn.Linear(dim, bottleneck_dim), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(max(0, num_layers - 2)):
        layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(bottleneck_dim, dim))
    return nn.Sequential(*layers)


def load_predictor(
    predictor_path: Path,
    dim: int,
    num_layers: int,
    bottleneck_dim: int,
    dropout: float,
    device: torch.device,
):
    sd = torch.load(predictor_path, map_location="cpu", weights_only=True)
    stripped = {k.removeprefix("net."): v for k, v in sd.items()}
    net = build_predictor(dim, num_layers, bottleneck_dim, dropout)
    net.load_state_dict(stripped)
    net.eval().to(device)
    return net


def project(arr: np.ndarray, net, device: torch.device, batch: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, arr.shape[0], batch):
            t = torch.from_numpy(arr[i : i + batch]).to(device).float()
            p = next(net.parameters(), None)
            if p is not None:
                t = t.to(dtype=p.dtype)
            out.append(net(t).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics(y_true, y_pred, y_prob):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": (
            roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--states_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument(
        "--id_malicious",
        nargs="+",
        required=True,
        help="Names of malicious-side ID datasets (npy stems without view "
             "prefix), e.g. 'beaver maliciousinstruct'.",
    )
    ap.add_argument(
        "--id_benign",
        nargs="+",
        required=True,
        help="Names of benign-side ID datasets, e.g. 'alpaca dolly'.",
    )
    ap.add_argument(
        "--id_malicious_view",
        default="malicious",
        help="View prefix for ID malicious npys (default 'malicious').",
    )
    ap.add_argument(
        "--id_benign_view",
        default="benign",
        help="View prefix for ID benign npys (default 'benign').",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument(
        "--predictor_path",
        type=Path,
        default=None,
        help="If set, project every loaded state through this trained "
             "predictor (jepa_predictor.pt) before fitting/evaluating the SVM.",
    )
    ap.add_argument("--predictor_layers", type=int, default=2)
    ap.add_argument("--predictor_bottleneck_dim", type=int, default=512)
    ap.add_argument("--predictor_dropout", type=float, default=0.0)
    args = ap.parse_args()

    files = list_files(args.states_dir)
    if not files:
        raise SystemExit(f"no .npy files under {args.states_dir}")

    # Build ID stems
    id_stems = []
    id_stems += [f"{args.id_malicious_view}_{n}" for n in args.id_malicious]
    id_stems += [f"{args.id_benign_view}_{n}" for n in args.id_benign]
    missing = [s for s in id_stems if s not in files]
    if missing:
        raise SystemExit(
            f"ID datasets missing in {args.states_dir}: {missing}\navailable: {sorted(files)}"
        )

    # Optional predictor projection — load once, project every npy on read.
    predictor = None
    proj_device = None
    if args.predictor_path is not None:
        proj_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Read dim from the first ID file
        first = load_npy(files[id_stems[0]])
        predictor = load_predictor(
            args.predictor_path,
            dim=first.shape[1],
            num_layers=args.predictor_layers,
            bottleneck_dim=args.predictor_bottleneck_dim,
            dropout=args.predictor_dropout,
            device=proj_device,
        )
        print(f"[predictor] projecting through {args.predictor_path}")

    def _load(s: str) -> np.ndarray:
        arr = load_npy(files[s])
        if predictor is not None:
            arr = project(arr, predictor, proj_device)
        return arr

    # Load ID
    Xs, ys = [], []
    for s in id_stems:
        arr = _load(s)
        lab = label_for(s)
        if lab is None:
            raise SystemExit(f"can't infer label from stem {s!r}")
        Xs.append(arr)
        ys.append(np.full(arr.shape[0], lab))
    X_id = np.vstack(Xs)
    y_id = np.concatenate(ys)

    # Wang's stratified 80/20 with seed=42
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_id, y_id, test_size=args.test_size, stratify=y_id, random_state=args.seed
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    svm = SVC(kernel="linear", random_state=args.seed, probability=True)
    svm.fit(X_tr_s, y_tr)

    rows: list[dict] = []
    extra = {
        "cell": args.cell,
        "layer": args.layer,
        "n_id_train": int(X_tr.shape[0]),
        "n_id_val": int(X_val.shape[0]),
    }

    # ID val
    y_pred = svm.predict(X_val_s)
    y_prob = svm.predict_proba(X_val_s)[:, 1]
    rows.append({**extra, "split": "id_val", "stem": ",".join(id_stems), **metrics(y_val, y_pred, y_prob)})

    # OOD: every other npy in the dir
    ood_stems = [s for s in files if s not in id_stems]
    for stem in sorted(ood_stems):
        lab = label_for(stem)
        if lab is None:
            continue  # skip unknown-view files
        X_ood = _load(stem)
        y_ood = np.full(X_ood.shape[0], lab)
        X_ood_s = scaler.transform(X_ood)  # SAME scaler — Wang's protocol
        y_pred = svm.predict(X_ood_s)
        y_prob = svm.predict_proba(X_ood_s)[:, 1]
        rows.append(
            {**extra, "split": "ood", "stem": stem, "n": int(X_ood.shape[0]),
             **metrics(y_ood, y_pred, y_prob)}
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[saved] {args.out_csv} ({len(rows)} rows: 1 id_val + {len(rows)-1} ood)")


if __name__ == "__main__":
    main()
