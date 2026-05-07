"""Sweep probe-train-set size for one cell × all OOD slices.

Reuses Why-Probe-Fails (WPF) jepa modules (data, model, train, eval, baselines).
For a given states_dir (one defended/base model's pre-extracted states), trains
{svm_raw, mlp_no_jepa, jepa} on a subsampled training pool of size n_train and
evaluates on every held-out slice we care about: RS3 paraphrased, RS3 harmbench
held-out (×3 views), JEPA attacks (×4 styles), MULTI languages (×9).

Outputs one long-format CSV: cell, n_train, seed, probe, slice, n, acc, prec,
rec, f1, auc.

Usage:
    python scripts/probe_train_size/sweep.py \\
        --cell l_cb_pra \\
        --states_dir runs/experiments/headline_backup_rerun_probing/states/l_cb_pra \\
        --out_csv runs/experiments/headline_backup_rerun_probing/sweep/l_cb_pra.csv \\
        --train_sizes 25 50 100 250 500 \\
        --seeds 42 123 777
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

WPF_ROOT = Path(os.environ.get("WPF_ROOT", "/workspace/AdversariaLLM/Why-Probe-Fails"))
sys.path.insert(0, str(WPF_ROOT))

from jepa.baselines import fit_linear_svm_baseline
from jepa.data import (
    BENIGN_LABEL,
    HARMFUL_LABEL,
    PairedDataset,
    PairSplit,
    discover_npy,
    load_npy,
    load_paired_datasets,
    load_unpaired_arrays,
    make_pair_splits,
)
from jepa.eval import EvalSlice, ProbeAdapter, evaluate_probe
from jepa.model import JEPAProbe
from jepa.train import (
    TrainConfig,
    encode_array,
    predict_proba_classifier,
    train_probe,
)


# ---------- Cell config ----------

ID_PAIR_DATASET = "advbench"
HELD_OUT_PAIR_DATASET = "harmbench"
ID_BENIGN = "alpaca"
HELD_OUT_BENIGN = "dolly"

PAIRED_VIEWS = ("malicious", "cleaned", "paraphrased")
ADV_VIEW = "malicious"
CLEAN_VIEW = "cleaned"
PARA_VIEW = "paraphrased"

JEPA_ATTACKS = ("encoding", "inpainting", "persona", "prefilling")
MULTI_LANGS = ("ar", "cs", "en", "es", "fy", "id", "ja", "pt", "zh-cn")


# ---------- Loading ----------


def load_cell_states(states_dir: Path) -> Dict[str, np.ndarray]:
    """All <view>_<dataset>.npy in this directory, keyed by stem."""
    return {k: load_npy(v) for k, v in discover_npy(str(states_dir)).items()}


def build_predictor_module(
    dim: int,
    num_layers: int = 2,
    bottleneck_dim: int = 512,
    dropout: float = 0.0,
) -> torch.nn.Module:
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
) -> torch.nn.Module:
    sd = torch.load(predictor_path, map_location="cpu", weights_only=True)
    # State dict keys are saved with a "net." prefix (PerPositionPredictor wraps an
    # nn.Sequential into self.net). Strip it so we can load into a bare Sequential.
    stripped = {k.removeprefix("net."): v for k, v in sd.items()}
    net = build_predictor_module(dim, num_layers, bottleneck_dim, dropout)
    net.load_state_dict(stripped)
    net.eval()
    net.to(device)
    return net


def project_states(
    states: Dict[str, np.ndarray],
    predictor: torch.nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Apply predictor(rep) -> rep' to every array in `states`."""
    out: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for k, arr in states.items():
            xs = []
            for i in range(0, arr.shape[0], batch_size):
                t = torch.from_numpy(arr[i : i + batch_size]).to(device).float()
                p_param = next(predictor.parameters(), None)
                if p_param is not None:
                    t = t.to(dtype=p_param.dtype)
                xs.append(predictor(t).float().cpu().numpy())
            out[k] = np.concatenate(xs, axis=0)
    return out


def slice_paired(
    paired: Dict[str, PairedDataset],
    splits: Dict[str, PairSplit],
    indices: Dict[str, np.ndarray],
    view: str,
) -> np.ndarray:
    """Take view-array of each paired dataset, restricted to per-dataset indices."""
    out = []
    for name, ds in paired.items():
        idx = indices[name]
        out.append(ds.views[view][idx])
    return np.vstack(out) if out else np.empty((0, 0))


# ---------- Subsampling ----------


def subsample_train_pairs(
    paired: Dict[str, PairedDataset],
    splits: Dict[str, PairSplit],
    n_per_dataset: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Return one subsampled train-index array per ID-paired dataset.

    Caps at the available train-pool size if n_per_dataset > train_idx.size.
    """
    rng = np.random.default_rng(seed + 31_337)
    out: Dict[str, np.ndarray] = {}
    for name, sp in splits.items():
        n = min(n_per_dataset, sp.train_idx.size)
        chosen = rng.choice(sp.train_idx, size=n, replace=False)
        out[name] = np.sort(chosen)
    return out


def subsample_benign_train(
    benign_train: Dict[str, np.ndarray],
    n_per_source: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 9_191)
    out: Dict[str, np.ndarray] = {}
    for k, arr in benign_train.items():
        n = min(n_per_source, arr.shape[0])
        idx = rng.choice(arr.shape[0], size=n, replace=False)
        out[k] = arr[np.sort(idx)]
    return out


# ---------- Eval-slice construction ----------


def build_eval_slices(
    paired: Dict[str, PairedDataset],
    splits: Dict[str, PairSplit],
    benign_val_arrays: Dict[str, np.ndarray],
    held_out_paired: Dict[str, PairedDataset],
    held_out_benign: Dict[str, np.ndarray],
    jepa_paired: Dict[str, PairedDataset],
    multi_arrays: Dict[str, np.ndarray],
) -> List[EvalSlice]:
    """All OOD eval slices, paired with appropriate benign pools."""
    slices: List[EvalSlice] = []

    # ID val: held-out 20% of ID paired (mal view) + 20% of ID benign
    id_mal = slice_paired(paired, splits, {n: sp.val_idx for n, sp in splits.items()}, ADV_VIEW)
    benign_val_X = (
        np.vstack(list(benign_val_arrays.values())) if benign_val_arrays else np.empty((0, 0))
    )
    id_X = np.vstack([id_mal, benign_val_X])
    id_y = np.concatenate(
        [np.full(id_mal.shape[0], HARMFUL_LABEL), np.full(benign_val_X.shape[0], BENIGN_LABEL)]
    )
    slices.append(EvalSlice(name="id_val", X=id_X, y=id_y))

    # ID paraphrased (val) — paraphrase OOD on the same dataset
    id_para = slice_paired(
        paired, splits, {n: sp.val_idx for n, sp in splits.items()}, PARA_VIEW
    )
    if id_para.size:
        X = np.vstack([id_para, benign_val_X])
        y = np.concatenate(
            [np.full(id_para.shape[0], HARMFUL_LABEL), np.full(benign_val_X.shape[0], BENIGN_LABEL)]
        )
        slices.append(EvalSlice(name="ood_paraphrased_advbench", X=X, y=y))

    # Held-out paired: harmbench × {mal, clean, para}, vs held-out benign (dolly)
    ho_benign_X = (
        np.vstack(list(held_out_benign.values())) if held_out_benign else benign_val_X
    )
    ho_benign_y = np.full(ho_benign_X.shape[0], BENIGN_LABEL)
    for name, ds in held_out_paired.items():
        for view in PAIRED_VIEWS:
            if view not in ds.views:
                continue
            X_mal = ds.views[view]
            y_mal = np.full(X_mal.shape[0], HARMFUL_LABEL)
            X = np.vstack([X_mal, ho_benign_X])
            y = np.concatenate([y_mal, ho_benign_y])
            slices.append(EvalSlice(name=f"ood_heldout_{view}_{name}", X=X, y=y))

    # JEPA attacks: each (attack) is a paired (benign, malicious) source.
    # Pair malicious-attack vs benign-attack within the same attack style so
    # accuracy is not contaminated by domain shift between mal and benign pools.
    for attack, ds in jepa_paired.items():
        if "malicious" not in ds.views or "benign" not in ds.views:
            continue
        X_mal = ds.views["malicious"]
        X_ben = ds.views["benign"]
        X = np.vstack([X_mal, X_ben])
        y = np.concatenate(
            [np.full(X_mal.shape[0], HARMFUL_LABEL), np.full(X_ben.shape[0], BENIGN_LABEL)]
        )
        slices.append(EvalSlice(name=f"ood_jepa_{attack}", X=X, y=y))

    # MULTI: malicious-only. Pair with held-out benign (dolly val).
    for lang, X_mal in multi_arrays.items():
        X = np.vstack([X_mal, ho_benign_X])
        y = np.concatenate(
            [np.full(X_mal.shape[0], HARMFUL_LABEL), np.full(ho_benign_X.shape[0], BENIGN_LABEL)]
        )
        slices.append(EvalSlice(name=f"ood_multi_{lang}", X=X, y=y))

    return slices


# ---------- Probe wrappers ----------


def make_jepa_adapter(name: str, probe: JEPAProbe, device: torch.device) -> ProbeAdapter:
    def _proba(X: np.ndarray) -> np.ndarray:
        return predict_proba_classifier(probe, X, device=device)

    def _pred(X: np.ndarray) -> np.ndarray:
        return _proba(X).argmax(axis=-1)

    return ProbeAdapter(name=name, predict=_pred, predict_proba=_proba)


def split_benign(
    benign: Dict[str, np.ndarray],
    val_frac: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed + 7919)
    train: Dict[str, np.ndarray] = {}
    val: Dict[str, np.ndarray] = {}
    for k, arr in benign.items():
        n = arr.shape[0]
        n_val = max(1, int(round(n * val_frac)))
        perm = rng.permutation(n)
        val[k] = arr[np.sort(perm[:n_val])]
        train[k] = arr[np.sort(perm[n_val:])]
    return train, val


# ---------- One sweep cell ----------


def stack_train_classification(
    paired: Dict[str, PairedDataset],
    train_idx_per_ds: Dict[str, np.ndarray],
    benign_train: Dict[str, np.ndarray],
    adv_view: str,
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for name, ds in paired.items():
        idx = train_idx_per_ds[name]
        Xs.append(ds.views[adv_view][idx])
        ys.append(np.full(idx.size, HARMFUL_LABEL))
    for _, arr in benign_train.items():
        Xs.append(arr)
        ys.append(np.full(arr.shape[0], BENIGN_LABEL))
    return np.vstack(Xs), np.concatenate(ys)


def assemble_pairs_subset(
    paired: Dict[str, PairedDataset],
    train_idx_per_ds: Dict[str, np.ndarray],
    benign_train: Dict[str, np.ndarray],
    adv_view: str,
    clean_view: str,
):
    advs, cleans = [], []
    for name, ds in paired.items():
        idx = train_idx_per_ds[name]
        advs.append(ds.views[adv_view][idx])
        cleans.append(ds.views[clean_view][idx])
    benign = (
        np.vstack(list(benign_train.values()))
        if benign_train
        else np.empty((0, advs[0].shape[1]))
    )

    # Lightweight ad-hoc TrainingTensors-shaped object.
    class _T:
        pass

    t = _T()
    t.adv = np.vstack(advs)
    t.clean = np.vstack(cleans)
    t.benign = benign
    t.dim = t.adv.shape[1]
    return t


def build_probe(cfg: argparse.Namespace, dim: int, jepa_on: bool) -> JEPAProbe:
    return JEPAProbe(
        h_dim=dim,
        encoder_hidden_dims=tuple(cfg.encoder_hidden_dims),
        z_dim=cfg.z_dim,
        predictor_type=cfg.predictor_type if jepa_on else "identity",
        bottleneck_dim=cfg.bottleneck_dim,
        encoder_dropout=0.0,
        target_mode="ema",
        ema_momentum=0.99,
        jepa_loss="cosine",
    )


def run_one_size_seed(
    args: argparse.Namespace,
    paired: Dict[str, PairedDataset],
    splits: Dict[str, PairSplit],
    benign_train_full: Dict[str, np.ndarray],
    eval_slices: List[EvalSlice],
    n_train: int,
    seed: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    train_idx = subsample_train_pairs(paired, splits, n_per_dataset=n_train, seed=seed)
    n_benign = sum(min(n_train, a.shape[0]) for a in benign_train_full.values())
    # Per-source benign cap = floor(n_train / num_sources) but with at least 1
    per_source = max(1, n_train // max(1, len(benign_train_full)))
    benign_train = subsample_benign_train(
        benign_train_full, n_per_source=per_source, seed=seed
    )

    X_train, y_train = stack_train_classification(
        paired, train_idx, benign_train, adv_view=ADV_VIEW
    )

    rows: List[Dict[str, object]] = []
    extra = {"cell": args.cell, "n_train": n_train, "seed": seed}

    if "svm_raw" in args.probes:
        fitted = fit_linear_svm_baseline(X_train, y_train, seed=seed)
        adapter = ProbeAdapter(
            name="svm_raw", predict=fitted.predict, predict_proba=fitted.predict_proba
        )
        for r in evaluate_probe(adapter, eval_slices):
            rows.append({**extra, **r})

    tensors = assemble_pairs_subset(
        paired, train_idx, benign_train, adv_view=ADV_VIEW, clean_view=CLEAN_VIEW
    )

    base_train_cfg = TrainConfig(
        epochs=args.epochs,
        pair_batch_size=min(args.pair_batch_size, max(1, tensors.adv.shape[0])),
        benign_batch_size=min(
            args.benign_batch_size, max(1, tensors.benign.shape[0])
        ),
        lr=args.lr,
        weight_decay=1e-4,
        w_cls=1.0,
        w_jepa=1.0,
        grad_clip=1.0,
        seed=seed,
    )
    no_jepa_cfg = TrainConfig(**{**asdict(base_train_cfg), "w_jepa": 0.0})

    if "mlp_no_jepa" in args.probes:
        probe = build_probe(args, tensors.dim, jepa_on=False)
        train_probe(probe, tensors, no_jepa_cfg, device=device)
        for r in evaluate_probe(make_jepa_adapter("mlp_no_jepa", probe, device), eval_slices):
            rows.append({**extra, **r})

    if "jepa" in args.probes:
        probe = build_probe(args, tensors.dim, jepa_on=True)
        train_probe(probe, tensors, base_train_cfg, device=device)
        for r in evaluate_probe(make_jepa_adapter("jepa", probe, device), eval_slices):
            rows.append({**extra, **r})

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--states_dir", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--train_sizes", nargs="+", type=int, default=[25, 50, 100, 200, 400])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    ap.add_argument(
        "--probes",
        nargs="+",
        default=["svm_raw", "mlp_no_jepa", "jepa"],
        choices=["svm_raw", "mlp_no_jepa", "jepa"],
    )
    ap.add_argument(
        "--predictor_path",
        type=Path,
        default=None,
        help="If set, project every loaded state array through this trained "
             "predictor (jepa_predictor.pt) before training/evaluating probes. "
             "Architecture defaults match the headline_backup_rerun cells.",
    )
    ap.add_argument("--predictor_layers", type=int, default=2)
    ap.add_argument("--predictor_bottleneck_dim", type=int, default=512)
    ap.add_argument("--predictor_dropout", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--encoder_hidden_dims", nargs="+", type=int, default=[1024])
    ap.add_argument("--z_dim", type=int, default=512)
    ap.add_argument("--predictor_type", default="bottleneck_mlp")
    ap.add_argument("--bottleneck_dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pair_batch_size", type=int, default=64)
    ap.add_argument("--benign_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load all states
    paired = load_paired_datasets(
        str(args.states_dir), [ID_PAIR_DATASET], views=PAIRED_VIEWS
    )
    held_out_paired = load_paired_datasets(
        str(args.states_dir), [HELD_OUT_PAIR_DATASET], views=PAIRED_VIEWS
    )
    benign_all = load_unpaired_arrays(
        str(args.states_dir), [ID_BENIGN], prefix="benign"
    )
    held_out_benign = load_unpaired_arrays(
        str(args.states_dir), [HELD_OUT_BENIGN], prefix="benign"
    )

    # JEPA — paired benign/malicious per attack style
    jepa_paired: Dict[str, PairedDataset] = {}
    discovered = discover_npy(str(args.states_dir))
    for atk in JEPA_ATTACKS:
        keys = {v: f"{v}_{atk}" for v in ("benign", "malicious")}
        if not all(k in discovered for k in keys.values()):
            print(f"[skip] missing JEPA states for {atk}")
            continue
        views_arr = {v: load_npy(discovered[k]) for v, k in keys.items()}
        # Don't enforce pair alignment — sizes may differ
        jepa_paired[atk] = type(
            "JEPAPair", (), {"name": atk, "views": views_arr}
        )()

    # MULTI — malicious only per language
    multi_arrays: Dict[str, np.ndarray] = {}
    for lang in MULTI_LANGS:
        key = f"malicious_{lang}"
        if key in discovered:
            multi_arrays[lang] = load_npy(discovered[key])

    # Optionally project every loaded state through a frozen trained predictor.
    if args.predictor_path is not None:
        dim_probe = (
            next(iter(paired.values())).dim
            if paired
            else next(iter(benign_all.values())).shape[1]
        )
        predictor = load_predictor(
            args.predictor_path,
            dim=dim_probe,
            num_layers=args.predictor_layers,
            bottleneck_dim=args.predictor_bottleneck_dim,
            dropout=args.predictor_dropout,
            device=device,
        )
        print(f"[predictor] projecting all states through {args.predictor_path}")
        from jepa.data import PairedDataset as _PD

        def _project_paired(d):
            return {
                n: _PD(name=ds.name, views=project_states(ds.views, predictor, device))
                for n, ds in d.items()
            }

        paired = _project_paired(paired)
        held_out_paired = _project_paired(held_out_paired)
        benign_all = project_states(benign_all, predictor, device)
        held_out_benign = project_states(held_out_benign, predictor, device)
        if jepa_paired:
            for atk, ds in list(jepa_paired.items()):
                ds.views = project_states(ds.views, predictor, device)  # ad-hoc class, not frozen
        if multi_arrays:
            multi_arrays = project_states(multi_arrays, predictor, device)

    splits = make_pair_splits(paired, val_frac=args.val_frac, seed=42)
    benign_train_full, benign_val = split_benign(benign_all, val_frac=args.val_frac, seed=42)

    eval_slices = build_eval_slices(
        paired,
        splits,
        benign_val,
        held_out_paired,
        held_out_benign,
        jepa_paired,
        multi_arrays,
    )

    print(f"[plan] cell={args.cell}")
    print(
        f"[plan] eval slices: "
        + ", ".join(f"{s.name}(n={s.X.shape[0]})" for s in eval_slices)
    )
    print(
        f"[plan] available train pairs: "
        + ", ".join(f"{n}={sp.train_idx.size}" for n, sp in splits.items())
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for n_train in args.train_sizes:
        for seed in args.seeds:
            print(f"[run] n_train={n_train} seed={seed}")
            rows.extend(
                run_one_size_seed(
                    args,
                    paired,
                    splits,
                    benign_train_full,
                    eval_slices,
                    n_train,
                    seed,
                    device,
                )
            )
            # Incremental flush
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[done] {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
