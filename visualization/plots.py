"""Plotting functions for visualizing JEPA defense representations.

Three families of plots:
  1. plot_scatter_2d   — t-SNE / PCA scatter of last-token reps, colored by
                          source label, drawn for both encoder and predictor space.
  2. plot_singular_values
                       — log-spectrum + cumulative-variance of representation
                          matrices, the LoRA delta, and predictor weights.
  3. plot_token_harm_strip / plot_token_harm_lines
                       — per-token harm score visualizations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


_PALETTE = {
    "benign": "#2ca02c",
    "benign_wild": "#98df8a",
    "harmful_clean": "#d62728",
    "jailbreak_reverse": "#1f77b4",
    "jailbreak_wild": "#9467bd",
}


def _color_for(label: str) -> str:
    return _PALETTE.get(label, "#7f7f7f")


# ---------------------------------------------------------------------------
# 2-D scatter plots (t-SNE / PCA) over last-token reps
# ---------------------------------------------------------------------------

def _stack(reps_by_label: Dict[str, Dict[str, torch.Tensor]], space: str
           ) -> Tuple[np.ndarray, np.ndarray]:
    xs, labels = [], []
    for label, rep in reps_by_label.items():
        xs.append(rep[space].numpy())
        labels.extend([label] * rep[space].shape[0])
    return np.concatenate(xs, axis=0), np.array(labels)


def _scatter(ax, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    for label in sorted(set(labels), key=lambda x: list(_PALETTE).index(x) if x in _PALETTE else 99):
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=14, alpha=0.6, label=f"{label} (n={int(mask.sum())})",
            color=_color_for(label), edgecolors="none",
        )
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False, loc="best")


def plot_scatter_2d(
    reps_by_label: Dict[str, Dict[str, torch.Tensor]],
    out_dir: Path,
    title_prefix: str,
    seed: int = 0,
) -> List[Path]:
    """Saves four panels: (encoder, predictor) x (t-SNE, PCA) into one figure
    plus three single-panel figures for closer inspection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for col, space in enumerate(("enc", "pred")):
        X, labels = _stack(reps_by_label, space)

        # t-SNE — perplexity scaled to point count.
        perp = max(5, min(40, X.shape[0] // 6))
        coords_tsne = TSNE(
            n_components=2, init="pca", perplexity=perp,
            random_state=seed, learning_rate="auto",
        ).fit_transform(X)
        _scatter(axes[0, col], coords_tsne, labels, f"t-SNE  ·  {space}")

        coords_pca = PCA(n_components=2, random_state=seed).fit_transform(X)
        _scatter(axes[1, col], coords_pca, labels, f"PCA  ·  {space}")

    fig.suptitle(f"{title_prefix}  ·  last-token representations", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    grid_path = out_dir / "scatter_grid.png"
    fig.savefig(grid_path, dpi=140); plt.close(fig)
    saved.append(grid_path)

    # also: per-space t-SNE big version (handier to share)
    for space in ("enc", "pred"):
        X, labels = _stack(reps_by_label, space)
        perp = max(5, min(40, X.shape[0] // 6))
        coords = TSNE(
            n_components=2, init="pca", perplexity=perp, random_state=seed,
            learning_rate="auto",
        ).fit_transform(X)
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(ax, coords, labels, f"{title_prefix}  ·  t-SNE  ·  {space}")
        fig.tight_layout()
        p = out_dir / f"tsne_{space}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Singular value spectra
# ---------------------------------------------------------------------------

def _svdvals(mat: torch.Tensor) -> np.ndarray:
    # torch.linalg.svdvals is fast on GPU/CPU
    return torch.linalg.svdvals(mat.float()).cpu().numpy()


def plot_rep_singular_values(
    reps_by_label: Dict[str, Dict[str, torch.Tensor]],
    out_dir: Path,
    title_prefix: str,
) -> List[Path]:
    """One figure per space (enc, pred) with two panels:
        log singular values on the left, cumulative explained variance on the right.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for space in ("enc", "pred"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for label, rep in reps_by_label.items():
            X = rep[space] - rep[space].mean(0, keepdim=True)
            sv = _svdvals(X)
            axes[0].plot(sv, label=f"{label} (n={X.shape[0]})", color=_color_for(label))
            cum = np.cumsum(sv ** 2) / max(1e-12, float((sv ** 2).sum()))
            axes[1].plot(cum, label=label, color=_color_for(label))
        axes[0].set_yscale("log"); axes[0].set_xlabel("rank"); axes[0].set_ylabel("singular value (log)")
        axes[0].set_title("singular values"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
        axes[1].set_xlabel("rank"); axes[1].set_ylabel("cumulative variance")
        axes[1].set_title("cumulative variance"); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1.02)
        fig.suptitle(f"{title_prefix}  ·  rep spectra  ·  {space}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p = out_dir / f"singular_values_reps_{space}.png"
        fig.savefig(p, dpi=140); plt.close(fig)
        saved.append(p)
    return saved


def plot_lora_singular_values(
    deltas: Dict[str, torch.Tensor],
    out_dir: Path,
    title_prefix: str,
) -> List[Path]:
    """One figure with all LoRA delta spectra overlaid (log scale).
    Color-codes q_proj vs v_proj.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = {"q_proj": 0, "v_proj": 0}
    for name, mat in deltas.items():
        kind = "q_proj" if "q_proj" in name else "v_proj" if "v_proj" in name else "other"
        color = "#1f77b4" if kind == "q_proj" else "#d62728" if kind == "v_proj" else "#7f7f7f"
        sv = _svdvals(mat)
        ax.plot(sv, color=color, alpha=0.35, lw=1.0,
                label=kind if counts.get(kind, 0) == 0 else None)
        counts[kind] = counts.get(kind, 0) + 1
    ax.set_yscale("log"); ax.set_xlabel("rank"); ax.set_ylabel("singular value (log)")
    ax.set_title(f"{title_prefix}  ·  LoRA delta (B@A) spectra  ·  one curve per layer")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    p = out_dir / "singular_values_lora.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return [p]


def plot_predictor_singular_values(
    weights: Dict[str, torch.Tensor],
    out_dir: Path,
    title_prefix: str,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    items = list(weights.items())
    for i, (name, mat) in enumerate(items):
        sv = _svdvals(mat)
        color = cmap(i / max(1, len(items) - 1))
        ax.plot(sv, color=color, label=f"{name}  {tuple(mat.shape)}")
    ax.set_yscale("log"); ax.set_xlabel("rank"); ax.set_ylabel("singular value (log)")
    ax.set_title(f"{title_prefix}  ·  predictor weight spectra")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out_dir / "singular_values_predictor.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return [p]


# ---------------------------------------------------------------------------
# Per-token harm visualization
# ---------------------------------------------------------------------------

_HARM_CMAP = LinearSegmentedColormap.from_list(
    "harm", ["#2ca02c", "#f0f0f0", "#d62728"], N=256,
)


def plot_token_harm_strip(
    examples: List[Dict[str, object]],
    out_path: Path,
    title: str,
    vmax: float | None = None,
    max_tokens: int = 64,
    annotate: bool = True,
) -> Path:
    """Render a colored strip per example.

    `examples` is a list of dicts with keys:
        'name'   : str   (display name)
        'tokens' : list[str]
        'scores' : torch.Tensor (T,)   (per-token harm score; signed)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if vmax is None:
        all_abs = []
        for ex in examples:
            s = ex["scores"].abs()
            if s.numel():
                all_abs.append(float(s.max().item()))
        vmax = max(all_abs) if all_abs else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    n_tok = max((min(max_tokens, len(ex["tokens"])) for ex in examples), default=1)
    fig_w = max(10.0, 0.30 * n_tok + 4.0)
    fig_h = max(3.0, 0.7 * len(examples) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_tok); ax.set_ylim(-0.5, len(examples) - 0.5)
    ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks(range(len(examples)))
    ax.set_yticklabels([ex["name"] for ex in examples], fontsize=9)

    for row, ex in enumerate(examples):
        tokens = ex["tokens"][:max_tokens]
        scores = ex["scores"].numpy()[:max_tokens]
        for col, (tok, s) in enumerate(zip(tokens, scores)):
            color = _HARM_CMAP(norm(float(s)))
            ax.add_patch(Rectangle((col, row - 0.42), 1.0, 0.84,
                                   facecolor=color, edgecolor="white", linewidth=0.3))
            if not annotate:
                continue
            txt = (tok or "").replace("\n", "\\n").replace(" ", "·")
            # Rotate vertical for readability — strip cells are narrow.
            display = txt[:10] + ("…" if len(txt) > 10 else "")
            ax.text(col + 0.5, row, display, ha="center", va="center",
                    fontsize=6, color="black", rotation=90)

    sm = plt.cm.ScalarMappable(cmap=_HARM_CMAP, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("harm − benign cosine", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def plot_token_harm_lines(
    examples: List[Dict[str, object]],
    out_path: Path,
    title: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for ex in examples:
        s = ex["scores"].numpy()
        ax.plot(np.arange(len(s)), s, label=ex["name"], lw=1.4, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("token position")
    ax.set_ylabel("harm − benign cosine")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path
