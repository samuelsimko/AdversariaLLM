#!/usr/bin/env python3
"""Targeted Latent Adversarial Training (LAT) defense.

Implements the targeted-LAT recipe from Sheshadri et al.\ 2024
(RT-EAT-LAT applied to refusal training).  Defaults audited against the
authors' codebase at github.com/aengusl/latent-adversarial-training
(included as the `latent-adversarial-training/` submodule), specifically
`latent_at/lat_methods.py:projected_gradient_descent` and
`latent_at/laa/attacks.py:GDAdversary`:

    * delta is parameterized as nn.Parameter and optimized with AdamW
      (`inner_learning_rate=5e-2`, `pgd_iterations_per_step=16`).
    * after each AdamW step, delta is masked to prompt positions and
      L2-projected per-(batch, token) onto a ball of radius `pgd_eps`.
    * KL retain term uses reverse KL = KL(p_def || p_base), matching the
      upstream `F.kl_div(base.log_softmax, new.softmax)` call, with the
      gradient-normalization trick from `do_defense_step`.

  Inner adversary (Eq. 3 / App. B.1; PGD on residual-stream perturbations):
    delta* = argmin_{||delta_l||_2 <= eps for each layer l in pgd_layers}
              [ -log P(r | x + delta)               # toward harmful r
              + -log(1 - P(c | x + delta)) ]        # away from harmless c
  with delta perturbing only prompt-token positions, simultaneously at a
  small set of evenly-spaced layers (default {8, 16, 24, 30}).

  Outer defender step (Eq. 4) under delta*:
    L_toward = -log P(c | x + delta*)                                # toward harmless
    L_away   = -log(1 - P(r | x + delta*))                           # away from harmful
    L_benign,KL = KL(p_base || p_def) on benign UltraChat (Eq. 7)    # default Llama-3 setting
    L_benign,SFT = -log P(y | x) on benign (Eq. 6)                   # alt. Llama-2 setting
    L_pra    = MSE predictor(h_adv) -> sg[h_van]                     # OPTIONAL (PRA bolt-on)

    Paper's Eq. 5: L_model = L_defense + L_benign  (no explicit weights)
    We expose weights so a paper-faithful run sets alpha=beta=1 and exactly
    one of {w_benign (SFT), w_kl (KL)} = 1 with the other = 0.

The PGD loop and the defender step both use HuggingFace forward hooks on the
selected transformer layers (`model.model.layers[l]`) to inject delta at
prompt positions.

Output contract (matches experiments/run_experiment.py):
    <output_dir>/
        lora_adapter/      saved LoRA
        jepa_predictor.pt  only when w_jepa > 0
        manifest.json
        hparams.json
        READY              touched on success
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Defaults
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-4
DEFAULT_MAX_STEPS = 1500

# Loss weights
DEFAULT_ALPHA = 1.0   # toward refusal (Eq. 4, paper has no explicit weight -> 1)
DEFAULT_BETA = 1.0    # away from harmful (Eq. 4, paper has no explicit weight -> 1)
# Paper App. B.1 picks ONE benign stabilizer per setting:
#   Llama-2 -> SFT (Eq. 6); Llama-3 -> KL (Eq. 7).
# We default to the Llama-3 recipe since our targets are Llama-3-8B and Qwen3-8B.
DEFAULT_W_BENIGN = 0.0
DEFAULT_W_KL = 1.0

# PGD hyperparameters — match upstream defaults from latent-adversarial-training
# (latent_at/lat_methods.py:ProjectedGradLAT). Paper grid over eps in
# {0.5, 1.0, 2.5, 6.0, 10.0}; we keep 2.5 (middle) as a defensible default.
DEFAULT_PGD_LAYERS = [8, 16, 24, 30]
DEFAULT_PGD_EPS = 2.5
DEFAULT_PGD_STEPS = 16        # upstream `pgd_iterations_per_step` default
DEFAULT_PGD_LR = 5e-2         # upstream `inner_learning_rate` default (used with AdamW)

# PRA bolt-on
DEFAULT_W_JEPA = 0.0
DEFAULT_ALIGN_LAYER = 25
DEFAULT_PREDICTOR_LAYERS = 2
DEFAULT_PREDICTOR_BOTTLENECK_DIM = 512
DEFAULT_PREDICTOR_LR_MULTIPLIER = 5.0

# Refusal pool — fixed strings used as the "toward" target for harmful prompts.
# Sampled deterministically per example; matches the spirit of refusal training
# without needing an external refusal-generation model.
DEFAULT_REFUSALS = [
    "I can't help with that request.",
    "I'm not able to assist with this. If there's something else I can help with, let me know.",
    "Sorry, but I can't help with that.",
    "I won't help with that request, but I'd be glad to help with something else.",
    "That's not something I can assist with.",
]


# ============================================================
# Tokenization
# ============================================================

def _tokenize_chat_pair(prompt: str, response: str, tokenizer: PreTrainedTokenizer, max_length: int):
    """Returns (input_ids, attention_mask, labels) tensors padded to max_length.
    Labels mask prompt tokens (set to -100); response tokens get real ids."""
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False,
        max_length=max_length, truncation=True, return_tensors="pt",
    )
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        max_length=max_length, truncation=True, return_tensors="pt",
    )
    labels = full.clone()
    labels[0, :prompt_only.shape[-1]] = -100

    seq_len = full.shape[-1]
    pad = max_length - seq_len
    input_ids = F.pad(full, (0, pad), value=tokenizer.pad_token_id).squeeze(0)
    labels = F.pad(labels, (0, pad), value=-100).squeeze(0)
    attn = F.pad(torch.ones_like(full), (0, pad), value=0).squeeze(0)
    return input_ids, attn, labels


def tokenize_chat(prompts, responses, tokenizer, max_length=256):
    if len(prompts) != len(responses):
        raise ValueError("prompts/responses length mismatch")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for p, r in zip(prompts, responses):
        input_ids, attn, labels = _tokenize_chat_pair(p, r, tokenizer, max_length)
        out.append({"input_ids": input_ids, "attention_mask": attn, "labels": labels})
    return out


def tokenize_lat_triple(prompts, harmful_responses, refusals, tokenizer, max_length=256):
    """For each harmful prompt, build TWO synced views with the SAME prompt prefix:
       - 'harmful' view: labels score the harmful response
       - 'refusal' view: labels score the refusal response

    Both views share the same input_ids prefix (the prompt), but the suffix
    is the harmful-response tokens for one and the refusal tokens for the other.
    The PGD adversary perturbs prompt positions only, so it sees the same
    positions either way; the labels just decide which response is being
    scored at the response positions.
    """
    if not (len(prompts) == len(harmful_responses) == len(refusals)):
        raise ValueError("prompts/harmful/refusal length mismatch")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for p, ytilde, y in zip(prompts, harmful_responses, refusals):
        h_ids, h_attn, h_labels = _tokenize_chat_pair(p, ytilde, tokenizer, max_length)
        r_ids, r_attn, r_labels = _tokenize_chat_pair(p, y, tokenizer, max_length)
        out.append({
            "harm_input_ids": h_ids, "harm_attention_mask": h_attn, "harm_labels": h_labels,
            "ref_input_ids": r_ids, "ref_attention_mask": r_attn, "ref_labels": r_labels,
        })
    return out


def tokenize_prompt_only(prompts, tokenizer, max_length=256):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = []
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=True, add_generation_prompt=True,
            max_length=max_length, truncation=True, return_tensors="pt",
        )
        seq_len = ids.shape[-1]
        pad = max_length - seq_len
        input_ids = F.pad(ids, (0, pad), value=tokenizer.pad_token_id).squeeze(0)
        attn = F.pad(torch.ones_like(ids), (0, pad), value=0).squeeze(0)
        out.append({"input_ids": input_ids, "attention_mask": attn})
    return out


class ListDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


# ============================================================
# Data loading
# ============================================================

def load_ultrachat(num_samples: int, seed: int = 42):
    ds = (load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
          .shuffle(seed=seed).select(range(num_samples)))
    prompts, responses = [], []
    for item in ds:
        msgs = item["messages"]
        if not msgs:
            continue
        u = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompts.append(u); responses.append(a)
    return prompts, responses


def _read_json_or_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(8192); f.seek(0)
        first = next((c for c in head if not c.isspace()), None)
        if first == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                f.seek(0)
        return [json.loads(s) for s in f if s.strip() and not s.lstrip().startswith(("#", "//"))]


def load_harmful(path: str, limit: Optional[int] = None):
    rec = _read_json_or_jsonl(path)
    if limit is not None:
        rec = rec[:limit]
    prompts = [r.get("prompt", "") for r in rec]
    outputs = [r.get("output", "") for r in rec]
    return prompts, outputs


def load_wj_pairs(path: str, limit: Optional[int] = None):
    rec = _read_json_or_jsonl(path)
    if limit is not None:
        rec = rec[:limit]
    return ([r.get("vanilla", "") for r in rec],
            [r.get("adversarial", "") for r in rec])


# ============================================================
# Hooks for residual-stream perturbation injection
# ============================================================

@contextmanager
def adapters_disabled(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


def _get_decoder_layers(model):
    """Return a Module list of transformer decoder layers regardless of LoRA wrapping."""
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)        # peft.PeftModel.base_model.model -> the HF model
    inner = getattr(base, "model", base)       # the HF model's `.model` (Qwen3Model / LlamaModel)
    if hasattr(inner, "layers"):
        return inner.layers
    raise RuntimeError("Could not locate transformer decoder layers on the given model.")


def attach_lat_hooks(model, deltas, layer_indices, prompt_mask):
    """Register forward hooks on selected layers that add deltas[i] (cast to the
    layer's output dtype) to the residual stream at prompt-token positions only.

    deltas[i]: float tensor [B, T, D] with requires_grad=True.
    prompt_mask: [B, T] bool/long tensor, 1 where the position is in the prompt.
    Returns a list of hook handles to detach later.
    """
    decoder_layers = _get_decoder_layers(model)
    handles = []
    mask = prompt_mask.to(dtype=torch.float32).unsqueeze(-1)
    for di, li in enumerate(layer_indices):
        layer = decoder_layers[li]

        def hook(module, inputs, output, _di=di):
            if isinstance(output, tuple):
                hs = output[0]
                rest = output[1:]
            else:
                hs = output
                rest = None
            # Cast delta to hidden-state dtype/device. Done as part of the autograd
            # graph so gradients flow back to the float32 deltas.
            d = deltas[_di].to(dtype=hs.dtype, device=hs.device)
            hs = hs + d * mask.to(dtype=hs.dtype, device=hs.device)
            return (hs, *rest) if rest is not None else hs

        handles.append(layer.register_forward_hook(hook))
    return handles


def detach_hooks(handles):
    for h in handles:
        h.remove()


# ============================================================
# Loss helpers
# ============================================================

def ce_response_loss(logits, labels):
    """Standard shifted CE; positions with label=-100 are ignored."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def log_one_minus_p_loss(logits, labels, threshold: float = -5.0):
    """Numerically stable per-token -log(1 - p(label)) "away" loss.

    Lifted from latent-adversarial-training/latent_at/utils.py (in turn from
    the HarmBench repo). Computes log(1-p) via logsumexp over the
    non-target logits to avoid catastrophic cancellation when p approaches 1.
    Tokens with log p(label) < `threshold` are masked out (no further pressure
    needed when the model is already very unlikely to emit the target).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    log_sum_exp_all = torch.logsumexp(shift_logits, dim=-1)
    gather_labels = shift_labels.clone()
    gather_labels[shift_labels == -100] = 0
    logits_for_labels = torch.gather(shift_logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    log_p = logits_for_labels - log_sum_exp_all

    mask_target = torch.zeros_like(shift_logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    masked_logits = shift_logits * (1 - mask_target) + mask_target * (-1e10)
    log_sum_exp_without_target = torch.logsumexp(masked_logits, dim=-1)

    log_1_minus_p = log_sum_exp_without_target - log_sum_exp_all
    ignored = (shift_labels == -100)
    log_1_minus_p = log_1_minus_p.masked_fill(ignored, 0.0)
    log_1_minus_p = log_1_minus_p.masked_fill(log_p < threshold, 0.0)
    return -log_1_minus_p.sum() / (~ignored).sum().float().clamp_min(1.0)


def reverse_kl_loss(logits_def, logits_base):
    """Reverse KL: KL(p_def || p_base). Matches upstream LAT's KL term
    (latent_at/lat_helpers.py:do_defense_step), which calls

        F.kl_div(base_logits.log_softmax(-1), new_logits.softmax(-1))

    PyTorch's F.kl_div(input=log_q, target=p) = sum p * (log p - log q) = KL(p || q),
    so with input=p_base and target=p_def this evaluates to KL(p_def || p_base) --
    pulls the defended distribution toward the base on benign inputs.
    """
    return F.kl_div(
        logits_base.log_softmax(dim=-1),
        logits_def.softmax(dim=-1),
        reduction="batchmean",
    )


# ============================================================
# PGD inner attack
# ============================================================

def pgd_attack(
    model,
    harm_input_ids,
    harm_attention_mask,
    harm_labels,
    ref_input_ids,
    ref_attention_mask,
    ref_labels,
    pgd_layers: List[int],
    pgd_eps: float,
    pgd_steps: int,
    pgd_lr: float,
):
    """Two-term LAT attack (Eq. 3) using AdamW on residual-stream perturbations
    at `pgd_layers`, matching the upstream LAT codebase
    (latent-adversarial-training/latent_at/lat_methods.py:projected_gradient_descent
    + laa/attacks.py:GDAdversary).

        L_attack = -log P(r | x + delta)              # toward harmful r
                 + -log(1 - P(c | x + delta))          # away from harmless c

    Per-(batch, token) L2 projection to the eps ball after each AdamW step
    (upstream's GDAdversary.clip_attack). delta is restricted to prompt-token
    positions via a multiplicative mask.
    """
    B, T = harm_input_ids.shape
    hidden_size = getattr(model.config, "hidden_size",
                          model.get_input_embeddings().weight.shape[-1])

    prompt_mask = (harm_labels == -100) & (harm_attention_mask.bool())

    # delta as nn.Parameter so AdamW can manage moments.
    deltas = [nn.Parameter(torch.zeros(B, T, hidden_size,
                                       device=model.device, dtype=torch.float32))
              for _ in pgd_layers]
    adv_optim = torch.optim.AdamW(deltas, lr=pgd_lr)

    def _project():
        with torch.no_grad():
            for d in deltas:
                d.data *= prompt_mask.unsqueeze(-1).to(d.dtype)
                norm = d.data.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
                d.data *= torch.clamp(pgd_eps / norm, max=1.0)

    was_training = model.training
    model.eval()
    pgd_first_loss = None
    pgd_last_loss = None
    try:
        for step in range(pgd_steps):
            adv_optim.zero_grad(set_to_none=True)
            handles = attach_lat_hooks(model, deltas, pgd_layers, prompt_mask)
            try:
                out_h = model(input_ids=harm_input_ids, attention_mask=harm_attention_mask, use_cache=False)
                L_toward_harm = ce_response_loss(out_h.logits, harm_labels)
                out_r = model(input_ids=ref_input_ids, attention_mask=ref_attention_mask, use_cache=False)
                L_away_ref = log_one_minus_p_loss(out_r.logits, ref_labels)
                L_attack = L_toward_harm + L_away_ref
                if step == 0:
                    pgd_first_loss = float(L_attack.detach())
                pgd_last_loss = float(L_attack.detach())
                L_attack.backward()
            finally:
                detach_hooks(handles)
            adv_optim.step()
            _project()
    finally:
        if was_training:
            model.train()

    return [d.detach() for d in deltas], prompt_mask, pgd_first_loss, pgd_last_loss


# ============================================================
# Predictor (PRA bolt-on)
# ============================================================

class Predictor(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, bottleneck_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        if num_layers < 2:
            raise ValueError("predictor_layers must be >= 2")
        layers = [nn.Linear(dim, bottleneck_dim), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers += [nn.Linear(bottleneck_dim, bottleneck_dim), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(bottleneck_dim, dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class JEPADataset(Dataset):
    def __init__(self, vanillas, adversarials, tokenizer, max_length):
        self.van = tokenize_prompt_only(vanillas, tokenizer, max_length=max_length)
        self.adv = tokenize_prompt_only(adversarials, tokenizer, max_length=max_length)
        if len(self.van) != len(self.adv):
            raise ValueError("vanilla/adversarial length mismatch")
    def __len__(self): return len(self.van)
    def __getitem__(self, idx): return {"vanilla": self.van[idx], "adversarial": self.adv[idx]}


# ============================================================
# Loader
# ============================================================

class TripleBatchLoader:
    def __init__(self, benign_loader, harm_loader, jepa_loader=None):
        self.benign_loader = benign_loader
        self.harm_loader = harm_loader
        self.jepa_loader = jepa_loader
        if jepa_loader is None:
            self._len = min(len(benign_loader), len(harm_loader))
        else:
            self._len = min(len(benign_loader), len(harm_loader), len(jepa_loader))
    def __iter__(self):
        if self.jepa_loader is None:
            for b, h in zip(self.benign_loader, self.harm_loader):
                yield (b, h, None)
        else:
            for b, h, j in zip(self.benign_loader, self.harm_loader, self.jepa_loader):
                yield (b, h, j)
    def __len__(self): return self._len


# ============================================================
# Trainer
# ============================================================

class LATTrainer(Trainer):
    def __init__(self, *, benign_ds, harm_ds, jepa_ds, predictor,
                 align_layer, alpha, beta, w_benign, w_kl,
                 pgd_layers, pgd_eps, pgd_steps, pgd_lr,
                 w_jepa, predictor_lr_multiplier, **kwargs):
        super().__init__(**kwargs)
        self.benign_ds = benign_ds
        self.harm_ds = harm_ds
        self.jepa_ds = jepa_ds
        self.predictor = predictor
        self.align_layer = align_layer
        self.alpha = alpha
        self.beta = beta
        self.w_benign = w_benign
        self.w_kl = w_kl
        self.pgd_layers = pgd_layers
        self.pgd_eps = pgd_eps
        self.pgd_steps = pgd_steps
        self.pgd_lr = pgd_lr
        self.w_jepa = w_jepa
        self.predictor_lr_multiplier = predictor_lr_multiplier

    def get_train_dataloader(self):
        bs = self.args.per_device_train_batch_size
        bl = DataLoader(self.benign_ds, batch_size=bs, shuffle=True)
        hl = DataLoader(self.harm_ds, batch_size=bs, shuffle=True)
        jl = DataLoader(self.jepa_ds, batch_size=bs, shuffle=True) if self.jepa_ds is not None else None
        return TripleBatchLoader(bl, hl, jl)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        groups = [{"params": lora_params, "lr": self.args.learning_rate}]
        if self.predictor is not None:
            groups.append({
                "params": list(self.predictor.parameters()),
                "lr": self.args.learning_rate * self.predictor_lr_multiplier,
            })
        self.optimizer = torch.optim.AdamW(groups, lr=self.args.learning_rate)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        benign, harm_pair, jepa = inputs
        device = model.device

        benign = {k: v.to(device) for k, v in benign.items()}
        # harm_pair has both 'harm_*' (prompt + harmful response) and 'ref_*' (prompt + refusal)
        h = {k: v.to(device) for k, v in harm_pair.items()}

        # ---- 1. Inner PGD attack (Eq. 3): toward harmful + away from harmless
        deltas, prompt_mask, pgd_first, pgd_last = pgd_attack(
            model,
            h["harm_input_ids"], h["harm_attention_mask"], h["harm_labels"],
            h["ref_input_ids"],  h["ref_attention_mask"],  h["ref_labels"],
            self.pgd_layers, self.pgd_eps, self.pgd_steps, self.pgd_lr,
        )

        # ---- 2. Defender forward under delta*
        handles = attach_lat_hooks(model, deltas, self.pgd_layers, prompt_mask)
        try:
            ref_out = model(input_ids=h["ref_input_ids"], attention_mask=h["ref_attention_mask"], use_cache=False)
            L_toward = ce_response_loss(ref_out.logits, h["ref_labels"])
            # Away: same prompt+perturbation, but score the harmful labels.
            harm_out = model(input_ids=h["harm_input_ids"], attention_mask=h["harm_attention_mask"], use_cache=False)
            L_away = log_one_minus_p_loss(harm_out.logits, h["harm_labels"])
        finally:
            detach_hooks(handles)

        # ---- 3. Benign LM CE (no perturbation)
        ben_out = model(input_ids=benign["input_ids"], attention_mask=benign["attention_mask"], use_cache=False)
        L_benign = ce_response_loss(ben_out.logits, benign["labels"])

        # ---- 4. Reverse KL retain on benign (matches upstream)
        L_kl = torch.zeros((), device=device)
        if self.w_kl > 0:
            with adapters_disabled(model), torch.no_grad():
                base_out = model(input_ids=benign["input_ids"], attention_mask=benign["attention_mask"], use_cache=False)
            kl_raw = reverse_kl_loss(ben_out.logits, base_out.logits)
            # Upstream gradient-normalization trick (lat_helpers.py:do_defense_step):
            # divide by detached value so the term contributes a unit-magnitude
            # gradient regardless of the absolute KL scale. Reported as kl_raw.
            L_kl = kl_raw / (kl_raw.detach() + 1e-8)
            self._kl_raw_value = float(kl_raw.detach())
        else:
            self._kl_raw_value = 0.0

        # ---- 5. Optional PRA
        L_pra = torch.zeros((), device=device)
        if self.w_jepa > 0 and jepa is not None and self.predictor is not None:
            van = {k: v.to(device) for k, v in jepa["vanilla"].items()}
            adv = {k: v.to(device) for k, v in jepa["adversarial"].items()}
            van_out = model(input_ids=van["input_ids"], attention_mask=van["attention_mask"],
                            output_hidden_states=True, use_cache=False)
            adv_out = model(input_ids=adv["input_ids"], attention_mask=adv["attention_mask"],
                            output_hidden_states=True, use_cache=False)
            van_h = van_out.hidden_states[self.align_layer + 1]
            adv_h = adv_out.hidden_states[self.align_layer + 1]
            van_pooled = (van_h * van["attention_mask"].unsqueeze(-1).float()).sum(1) / \
                         van["attention_mask"].sum(1, keepdim=True).clamp_min(1).float()
            adv_pooled = (adv_h * adv["attention_mask"].unsqueeze(-1).float()).sum(1) / \
                         adv["attention_mask"].sum(1, keepdim=True).clamp_min(1).float()
            L_pra = F.mse_loss(self.predictor(adv_pooled.float()), van_pooled.detach().float())

        loss = (self.alpha * L_toward
                + self.beta * L_away
                + self.w_benign * L_benign
                + self.w_kl * L_kl
                + self.w_jepa * L_pra)

        self.log({
            "loss/lat_toward": L_toward.detach().item(),
            "loss/lat_away": L_away.detach().item(),
            "loss/benign_ce": L_benign.detach().item(),
            "loss/benign_kl_raw": getattr(self, "_kl_raw_value", 0.0),
            "loss/pra": L_pra.detach().item() if isinstance(L_pra, torch.Tensor) else float(L_pra),
            "loss/total": loss.detach().item(),
            # PGD diagnostics: if the attack works, last < first (loss decreases as
            # the adversary finds stronger delta). Flat -> hooks/grads broken.
            "pgd/L_attack_first": pgd_first if pgd_first is not None else float("nan"),
            "pgd/L_attack_last":  pgd_last  if pgd_last  is not None else float("nan"),
        })
        return (loss, None) if return_outputs else loss


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--output_dir", type=str, required=True)

    # Harmful corpus (prompts with harmful responses; the LAT y_tilde target)
    p.add_argument("--cb_path", type=str, required=True,
                   help="JSON list or JSONL with {prompt, output}; output is the harmful response.")
    p.add_argument("--limit_cb", type=int, default=5000)

    # Optional benign side dataset
    p.add_argument("--ultrachat_samples", type=int, default=5000)
    p.add_argument("--extra_benign_path", type=str, default=None)
    p.add_argument("--extra_benign_samples", type=int, default=5000)

    # Optional JEPA pair corpus
    p.add_argument("--pair_path", type=str, default=None)
    p.add_argument("--pair_format", type=str, default="wildjailbreak")
    p.add_argument("--pair_sample_size", type=int, default=8000)

    # Training
    p.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--num_max_steps", type=int, default=DEFAULT_MAX_STEPS)

    # LAT loss weights
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="toward weight")
    p.add_argument("--beta",  type=float, default=DEFAULT_BETA,  help="away weight")
    p.add_argument("--w_benign", type=float, default=DEFAULT_W_BENIGN, help="benign LM CE weight")
    p.add_argument("--w_kl",  type=float, default=DEFAULT_W_KL, help="benign KL retain weight")

    # PGD adversary
    p.add_argument("--pgd_layers", type=int, nargs="*", default=DEFAULT_PGD_LAYERS,
                   help="Transformer-layer indices at which to inject residual-stream perturbations.")
    p.add_argument("--pgd_eps", type=float, default=DEFAULT_PGD_EPS,
                   help="Per-(batch,token) L2 budget on each layer's perturbation.")
    p.add_argument("--pgd_steps", type=int, default=DEFAULT_PGD_STEPS)
    p.add_argument("--pgd_lr", type=float, default=None,
                   help="PGD step size. Defaults to pgd_eps * 0.5 if not set.")

    # PRA bolt-on
    p.add_argument("--w_jepa", type=float, default=DEFAULT_W_JEPA)
    p.add_argument("--align_layer", type=int, default=DEFAULT_ALIGN_LAYER)
    p.add_argument("--predictor_layers", type=int, default=DEFAULT_PREDICTOR_LAYERS)
    p.add_argument("--predictor_bottleneck_dim", type=int, default=DEFAULT_PREDICTOR_BOTTLENECK_DIM)
    p.add_argument("--predictor_lr_multiplier", type=float, default=DEFAULT_PREDICTOR_LR_MULTIPLIER)

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    # Trainer logistics
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to", type=str, default="none")

    args = p.parse_args()
    if args.pgd_lr is None:
        args.pgd_lr = args.pgd_eps * DEFAULT_PGD_LR_FRAC
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Benign data
    benign_prompts, benign_responses = load_ultrachat(args.ultrachat_samples)
    if args.extra_benign_path and os.path.exists(args.extra_benign_path):
        eb = _read_json_or_jsonl(args.extra_benign_path)[: args.extra_benign_samples]
        for r in eb:
            benign_prompts.append(r.get("prompt", ""))
            benign_responses.append(r.get("response", "") or r.get("output", ""))
    benign_data = tokenize_chat(benign_prompts, benign_responses, tokenizer, args.max_length)

    # ----- Harmful data + refusal pairing
    harm_prompts, harm_outputs = load_harmful(args.cb_path, args.limit_cb)
    refusal_pool = DEFAULT_REFUSALS
    refusals = [refusal_pool[i % len(refusal_pool)] for i in range(len(harm_prompts))]
    harm_pair_data = tokenize_lat_triple(harm_prompts, harm_outputs, refusals, tokenizer, args.max_length)

    # ----- Optional JEPA pair data
    jepa_data = None
    if args.w_jepa > 0:
        if not args.pair_path or not os.path.exists(args.pair_path):
            raise ValueError(f"w_jepa>0 requires --pair_path; got {args.pair_path}")
        van, adv = load_wj_pairs(args.pair_path, args.pair_sample_size)
        jepa_data = JEPADataset(van, adv, tokenizer, max_length=args.max_length)

    # ----- Model + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.target_modules.split(","),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    # ----- Predictor
    predictor = None
    if args.w_jepa > 0:
        hidden_dim = model.config.hidden_size
        predictor = Predictor(
            dim=hidden_dim,
            num_layers=args.predictor_layers,
            bottleneck_dim=args.predictor_bottleneck_dim,
        ).to(model.device)  # float32

    # ----- Trainer
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.num_max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_grad_norm=1.0,
        report_to=args.report_to,
        run_name=f"lat_{Path(out_dir).name}",
    )

    trainer = LATTrainer(
        model=model, args=targs, tokenizer=tokenizer,
        benign_ds=ListDataset(benign_data),
        harm_ds=ListDataset(harm_pair_data),
        jepa_ds=jepa_data,
        predictor=predictor,
        align_layer=args.align_layer,
        alpha=args.alpha, beta=args.beta,
        w_benign=args.w_benign, w_kl=args.w_kl,
        pgd_layers=list(args.pgd_layers),
        pgd_eps=args.pgd_eps,
        pgd_steps=args.pgd_steps,
        pgd_lr=args.pgd_lr,
        w_jepa=args.w_jepa,
        predictor_lr_multiplier=args.predictor_lr_multiplier,
    )

    trainer.train()

    # ----- Save outputs
    model.save_pretrained(out_dir / "lora_adapter")
    if predictor is not None:
        torch.save(predictor.state_dict(), out_dir / "jepa_predictor.pt")
    with open(out_dir / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    manifest = {
        "schema_version": "1.0",
        "defense_name": "lat",
        "base_model": args.model,
        "adapter_type": "lora",
        "adapter_path": "lora_adapter",
        "training_completed": True,
        "pgd_layers": args.pgd_layers,
        "pgd_eps": args.pgd_eps,
        "pgd_steps": args.pgd_steps,
        "alpha": args.alpha, "beta": args.beta,
        "w_benign": args.w_benign, "w_kl": args.w_kl,
        "w_jepa": args.w_jepa,
        "align_layer": args.align_layer,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    (out_dir / "READY").touch()


if __name__ == "__main__":
    main()
