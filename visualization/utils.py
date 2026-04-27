"""Loader/encoder helpers for visualizing trained JEPA defenses.

Wraps `scripts.evaluate_jepa_guardrail` so we don't duplicate the manifest /
predictor / centroid logic — but adds per-token rep retrieval, which the
guardrail script doesn't expose.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_jepa_guardrail import (  # noqa: E402
    chat_prompt_ids,
    encode_texts,
    load_cb_by_category,
    load_model_and_predictor,
    load_reverse_jailbreak_prompts,
    load_ultrachat_prompts,
    load_wildjailbreak_ood,
    make_centroids,
)


@dataclass
class Loaded:
    run_dir: Path
    manifest: dict
    tokenizer: object
    model: object
    predictor: torch.nn.Module
    align_layer: int


def load_defense(run_dir: str | Path, device: str = "cuda", dtype: str = "bf16") -> Loaded:
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    rd = Path(run_dir)
    manifest, tok, model, predictor, align_layer = load_model_and_predictor(rd, device, dt)
    return Loaded(rd, manifest, tok, model, predictor, align_layer)


@dataclass
class PromptBundle:
    """Container for one labelled prompt set."""
    label: str
    prompts: List[str]


def gather_prompts(
    cb_path: str,
    reverse_path: str,
    n_benign: int,
    n_harmful: int,
    n_jailbreak: int,
    seed: int = 0,
    use_wildjailbreak: bool = False,
    wildjailbreak_dataset: str | None = None,
) -> List[PromptBundle]:
    """Returns a list of (label, prompts) bundles for visualization.

    - benign  : UltraChat
    - harmful : circuit_breakers_train.json (clean harmful intents)
    - jailbreak: reverse-model generated jailbreak prompts
    - wildjailbreak (optional): adversarial column from allenai/wildjailbreak
    """
    bundles: List[PromptBundle] = []

    bundles.append(PromptBundle("benign", load_ultrachat_prompts(n_benign, seed)))

    cb = load_cb_by_category(cb_path)
    flat: List[str] = []
    # Round-robin across categories so we don't oversample one topic.
    cats = sorted(cb)
    while len(flat) < n_harmful and any(cb[c] for c in cats):
        for c in cats:
            if cb[c]:
                flat.append(cb[c].pop(0))
                if len(flat) >= n_harmful:
                    break
    bundles.append(PromptBundle("harmful_clean", flat))

    bundles.append(
        PromptBundle("jailbreak_reverse", load_reverse_jailbreak_prompts(reverse_path, n_jailbreak))
    )

    if use_wildjailbreak:
        wb_b, wb_h = load_wildjailbreak_ood(
            None, wildjailbreak_dataset or "allenai/wildjailbreak",
            "eval", "train", limit=max(n_benign, n_harmful) * 2, seed=seed,
        )
        if wb_h:
            bundles.append(PromptBundle("jailbreak_wild", wb_h[:n_jailbreak]))
        if wb_b:
            bundles.append(PromptBundle("benign_wild", wb_b[:n_benign]))

    return bundles


def encode_bundles(
    bundles: List[PromptBundle],
    loaded: Loaded,
    max_length: int = 256,
    batch_size: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Returns {label: {'enc': (N,D), 'pred': (N,D)}} using the last-token rep."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for b in bundles:
        if not b.prompts:
            continue
        enc, pred = encode_texts(
            b.prompts, loaded.tokenizer, loaded.model, loaded.predictor,
            loaded.align_layer, max_length, batch_size,
        )
        out[b.label] = {"enc": enc, "pred": pred}
    return out


@torch.no_grad()
def encode_per_token(
    prompts: List[str],
    loaded: Loaded,
    max_length: int = 256,
) -> List[Dict[str, torch.Tensor]]:
    """Returns one dict per prompt with:
       'tokens'  : list[str]  (decoded display tokens)
       'enc'     : (T, D) hidden_states[align_layer] without padding
       'pred'    : (T, D) predictor(enc)
    """
    tok = loaded.tokenizer
    model = loaded.model
    predictor = loaded.predictor
    align_layer = loaded.align_layer
    device = model.device
    pred_param = next(predictor.parameters(), None)
    pred_dtype = pred_param.dtype if pred_param is not None else next(model.parameters()).dtype

    results: List[Dict[str, torch.Tensor]] = []
    for text in prompts:
        item = chat_prompt_ids(tok, text, max_length)
        ids = item["input_ids"].to(device)
        mask = item["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
        layer = align_layer if align_layer >= 0 else len(out.hidden_states) + align_layer
        hidden = out.hidden_states[layer][0]                      # (T, D)
        keep = int(mask.sum().item())
        enc = hidden[:keep].detach()
        pred = predictor(enc.to(dtype=pred_dtype)).detach()
        token_ids = ids[0, :keep].tolist()
        token_strs = tok.convert_ids_to_tokens(token_ids)
        # Pretty-print: convert each token to its rendered string via the tokenizer.
        display = [tok.convert_tokens_to_string([t]) for t in token_strs]
        results.append({
            "tokens": display,
            "raw_tokens": token_strs,
            "enc": enc.float().cpu(),
            "pred": pred.float().cpu(),
        })
    return results


def build_centroids(
    cb_path: str,
    benign_prompts: List[str],
    loaded: Loaded,
    per_category: int = 32,
    max_length: int = 256,
    batch_size: int = 8,
) -> Tuple[List[str], torch.Tensor]:
    """Wraps make_centroids. Returns (labels, centroids[L+1, D])."""
    cb_by_category = load_cb_by_category(cb_path)
    labels, centroids = make_centroids(
        cb_by_category,
        benign_prompts,
        loaded.tokenizer,
        loaded.model,
        loaded.predictor,
        loaded.align_layer,
        max_length,
        batch_size,
        per_category,
    )
    return labels, centroids


def harm_score(reps: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Centroid-based harm score: max cosine to any harmful centroid minus
    cosine to benign centroid (index 0).

    Returns (N,) float.
    """
    c = F.normalize(centroids.float(), dim=-1)
    sims = F.normalize(reps.float(), dim=-1) @ c.T          # (N, L+1)
    benign_sim = sims[:, 0]
    harm_sim = sims[:, 1:].max(dim=-1).values
    return harm_sim - benign_sim


def per_token_harm(
    per_token_reps: List[Dict[str, torch.Tensor]],
    centroids: torch.Tensor,
    space: str = "pred",
) -> List[torch.Tensor]:
    """For each prompt, returns (T,) per-token harm score in the chosen space."""
    return [harm_score(r[space], centroids) for r in per_token_reps]


def collect_lora_delta(model) -> Dict[str, torch.Tensor]:
    """Returns {module_name: BA} for every LoRA-injected layer. We compute
    the delta-weight (B @ A) so we can SVD it. Skips bias / non-LoRA modules.
    """
    deltas: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        # peft LoraLayer exposes lora_A / lora_B as ModuleDicts keyed by adapter
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for adapter_name in module.lora_A.keys():
                a = module.lora_A[adapter_name].weight.detach()  # (r, in)
                b = module.lora_B[adapter_name].weight.detach()  # (out, r)
                # scaling: alpha / r — we'll show the raw BA without scaling
                # because we care about the rank structure, not the magnitude.
                ba = (b @ a).float().cpu()
                deltas[f"{name}::{adapter_name}"] = ba
    return deltas


def collect_predictor_weights(predictor: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Returns {name: weight} for every Linear inside the predictor."""
    out: Dict[str, torch.Tensor] = {}
    for name, p in predictor.named_modules():
        if isinstance(p, torch.nn.Linear):
            out[name or "linear"] = p.weight.detach().float().cpu()
    return out
