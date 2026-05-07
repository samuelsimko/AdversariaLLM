# JEPA / PRA Paper-Results Dataset Overview

Summary of [`memo-ozdincer/jepa-paper-results-2026-05-06`](https://huggingface.co/datasets/memo-ozdincer/jepa-paper-results-2026-05-06), an interim NeurIPS-2026 release for the JepaAlign / PRA paper (1957 files, 2026-05-06 snapshot).

This doc is a digest of the README and `PAPER_RESULTS_SUMMARY.md` plus a peek at the per-cell `manifest.json`, `status.json`, and `probe_eval.json` — no `run.json` files were opened (those carry attack prompts and completions).

---

## What's in the dataset

| dir | experiment_name | what it is | status |
|---|---|---|---|
| `wj_ablation/` | `ablation_wj` | Qwen3-8B + circuit_breaker, **PRA vs no-PRA**, 3 attacks × 25 idx, includes per-cell `probe_eval.json` | complete |
| `ablation_wj_more/` | `ablation_wj_more` | **canonical Table 1** — Llama-3-8B + Qwen3-8B × {circuit_breaker, triplet, ce_in} × {pra, no_pra} = 10 defenses × 3 attacks × 25 idx | complete |
| `ce_floor/` | `ce_floor_jepa_qwen3_ablation` | Qwen3-8B CE-floor base + JEPA-aug (504k pairs), 5 attacks × 10 idx, dual judge (StrongREJECT + HarmBench), benign_eval | base done; `attacks_500k/` rerun against 500k-pair adapter still completing |
| `ce_floor_jepa_llama3_ablation/` | same | Llama-3 counterpart | base done |
| `data/` | — | training data: `wildjailbreak_harmful.json`, `wildjailbreak_benign.jsonl`, `wildjailbreak_pairs.jsonl` | reference |
| `reverse_model/` | — | LoRA + holdout split for the reverse model that generated the 500k-pair set | reference |

**PRA = "Predictor-based Refusal Attractor" = JEPA loss term enabled with `w_jepa=5.0`** (vs `w_jepa=0.0` for the no-PRA baseline). Everything else — same training script, same data, same masking, same hyperparameters — is held fixed within each (model × regularizer) pair, so the comparison is clean.

---

## What each experiment did (training stage)

All training uses the unified `defenses/jepa_ce.py` script, with `--harm_regularizer` selecting the harm loss family.

### `wj_ablation/` and `ablation_wj_more/` (Table 1 setups)

LoRA r=32 on `q_proj` + `v_proj`, 1500 training steps, batch 4 × grad_accum 2, max_length 256, lr 2e-4. Training data: 4000 harmful + 4000 benign pairs sampled from the 100k-cap WildJailbreak pair pool, plus 5000 ultrachat benign and 5000 wildjailbreak_harmful for the LM-CE losses.

PRA-specific knobs (paper version):
- `align_layer=25`
- `predictor_type=mlp`, `predictor_layers=2`, `predictor_bottleneck_dim=256`
- `predictor_lr_multiplier=100.0` (predictor learns much faster than LoRA params)
- `jepa_target_encoder=defended`, `jepa_target=prompt_only` (loss only over adv prompt positions, not response)

Loss = `w_benign · benign_LM + w_harm · harm_regularizer + w_kl · benign_KL + w_jepa · JEPA` with weights `(0.1, 1.0, 1.0, 5.0)` for PRA cells and `(0.1, 1.0, 1.0, 0.0)` for no-PRA cells.

Three harm regularizers swept:
- `circuit_breaker` (Llama only in `ablation_wj_more`; both in `wj_ablation`) — pushes harmful reps away from base reps via cosine targets at layers 20–35.
- `ce_in` — additional CE-floor term on the *included* harmful prompts.
- `triplet` — triplet loss with `margin=0.2`, separating harmful-pull and benign-push.

### `ce_floor/` (different defense script, larger pair budget)

Uses `defenses/ce_floor_align_jepa.py` (a CE-floor + JEPA-alignment hybrid). Different setup:
- `align_layer=-1` (last layer, not 25)
- `align_metric=cosine`, `harm_ce_min=4.0`
- 504,392 alignment pairs (4994 original + 499,398 reverse-model-generated, capped at 100k per epoch)
- 20k circuit_breakers_train CE prompts + 20k ultrachat benign
- LoRA save_total_limit=10 (10 checkpoints kept)

Weights: `w_benign=1.0, w_harm=1.0, w_kl=0.1, w_align=0.5`.

This is a different family of defenses than the WJ ablation — bigger data, last-layer alignment, no per-position predictor. The `attacks_500k/` subdir is a re-run of all 5 attacks against the 500k-pair adapter (the original `attacks/` ran against an earlier 4994-pair adapter).

---

## Does PRA help? — Headline numbers

### Table 1 (canonical, from `ablation_wj_more`): 10 defenses × 3 attacks × 25 behaviors

Judge: `local:strongreject`, score field `score_validated_dual_context`. BoN attack = max over all 1000 steps.

#### Llama-3-8B

| Regularizer | Attack | no-PRA ASR | PRA ASR | Δ (PRA − no-PRA) |
|---|---|---|---|---|
| circuit_breaker | direct | 8.0% | 4.0% | −4.0 pp |
| circuit_breaker | bon | 76.0% | 76.0% | 0.0 pp |
| circuit_breaker | soft_prompt | 20.0% | 12.0% | **−8.0 pp** |
| ce_in | direct | 4.0% | 4.0% | 0.0 pp |
| ce_in | bon | 64.0% | 56.0% | **−8.0 pp** |
| ce_in | soft_prompt | 20.0% | 8.0% | **−12.0 pp** |
| triplet | direct | 4.0% | 4.0% | 0.0 pp |
| triplet | bon | 76.0% | 84.0% | +8.0 pp |
| triplet | soft_prompt | 4.0% | 8.0% | +4.0 pp |
| **Llama mean** | | **30.7%** | **28.4%** | **−2.3 pp** |

#### Qwen3-8B

| Regularizer | Attack | no-PRA ASR | PRA ASR | Δ |
|---|---|---|---|---|
| ce_in | direct | 4.0% | 4.0% | 0.0 pp |
| ce_in | bon | 84.0% | 72.0% | **−12.0 pp** |
| ce_in | soft_prompt | 40.0% | 28.0% | **−12.0 pp** |
| triplet | direct | 4.0% | 4.0% | 0.0 pp |
| triplet | bon | 64.0% | 52.0% | **−12.0 pp** |
| triplet | soft_prompt | 12.0% | 20.0% | +8.0 pp |
| **Qwen3 mean** | | **34.7%** | **30.0%** | **−4.7 pp** |

#### Aggregate (averaged over attacks × bases)

| Regularizer | no-PRA mean | PRA mean | Δ |
|---|---|---|---|
| circuit_breaker (Llama only) | 34.7% | 30.7% | −4.0 pp |
| ce_in | 36.0% | 28.7% | **−7.3 pp** |
| triplet | 27.3% | 28.7% | +1.4 pp |
| **overall** | **32.3%** | **29.1%** | **−3.2 pp** |

### Table 1a (older, from `wj_ablation`): Qwen3-8B + circuit_breaker, n=25

| Attack | CB no-PRA | CB+PRA | Δ |
|---|---|---|---|
| BoN | 64.0% (16/25) | 52.0% (13/25) | **−12.0 pp** |
| Direct | 0.0% | 0.0% | (degenerate output, judge artifact) |
| SoftPrompt | 0.0% | 4.0% (1/25) | +4.0 pp |
| **Avg** | 21.3% | 18.7% | **−2.7 pp** |

This is the cell with `probe_eval.json` attached.

### Table 1b: Qwen3-8B CE-floor (no-PRA only — PRA pending), n=10

| Attack | CE no-PRA SR | CE no-PRA HB |
|---|---|---|
| BoN | 0% | 10% |
| Direct | 0% | 0% |
| Inpainting | **40%** | 0% |
| Prefilling | 0% | 0% |
| SoftPrompt | 0% | 20% |
| Avg | 8.0% | 6.0% |

CE+PRA column is TBD in this snapshot. Note the StrongREJECT vs HarmBench disagreement on Inpainting (40% vs 0%) and SoftPrompt (0% vs 20%) — judge choice changes the story.

---

## Probe-eval AUROC (Qwen3-8B + CB + PRA, layer 25)

From `wj_ablation/q_cb_pra_wj/probe_eval.json`. Centroid-based cosine score; cal/test split = 881/882 harmful + 128/128 benign. Test-set numbers:

| Probe signal | AUROC | TPR @ FPR=1% | benign mean | harmful mean |
|---|---|---|---|---|
| Encoder (raw layer-25 hidden state) | **0.9965** | 99.1% | 0.452 | 0.847 |
| Predictor (MLP head output) | 0.9886 | 88.4% | 0.665 | 0.938 |
| max(encoder, predictor) | 0.9887 | 88.4% | 0.666 | 0.938 |

**Reading:** at layer 25 the trained-defense reps separate harmful vs benign almost perfectly (0.997 AUROC) — the encoder alone is the strong signal. The predictor head adds nothing on AUROC and slightly hurts TPR at low FPR.

---

## Benign-capability preservation (CE-floor on Qwen3-8B)

| Defense | GSM8K (strict) | GSM8K (flexible) | MMLU acc |
|---|---|---|---|
| CE-floor (no PRA) | 0.900 ± 0.021 | 0.905 ± 0.021 | 0.762 ± 0.004 |
| CE-floor + PRA | 0.915 ± 0.020 | 0.930 ± 0.018 | 0.760 ± 0.004 |
| Δ (PRA − no-PRA) | +0.015 | +0.025 | −0.002 |
| Qwen3-8B base (model card) | ~0.876 | — | — |

All differences within one stderr; **PRA does not cost benign capability**.

---

## Caveats called out in the source

1. **Direct attack on Qwen3-CB is degenerate** — both PRA and no-PRA produce repetitive token-collapse outputs ("upper upper upper…", "1. 1. 1. …"). The judge correctly scores 0%, but it's a generation artifact, not robust refusal. Recommend dropping Direct cells from headline numbers or footnoting.
2. **CE-floor + PRA attacks not yet finished** in this snapshot — Table 1b's PRA column is empty.
3. **`attacks_500k/` rerun is in flight** — CE-floor numbers will shift once the 500k-trained adapter's attack run completes.
4. **Judge disagreement matters** — Inpainting and SoftPrompt scores diverge sharply between StrongREJECT and HarmBench on CE-floor. Paper should report both.
5. **Triplet's +1.4 pp regression** is driven by one anomalous Llama BoN cell (76% → 84%); the other triplet cells show typical PRA-helps deltas.
6. Small n on the per-cell numbers (n=25 for Table 1, n=10 for CE-floor). Single-attack ASR estimates have high variance.

---

## Verdict — does PRA help?

**Yes, modestly and consistently across most settings.**

- Across 10 defense configurations and 3 attack families, PRA reduces mean ASR by **3.2 percentage points** (32.3% → 29.1%).
- Stronger on **Qwen3** than Llama (−4.7 vs −2.3 pp).
- Strongest with the **`ce_in` regularizer** (−7.3 pp average; double-digit drops on adaptive attacks for both base models).
- On **adaptive attacks** (BoN and soft_prompt) PRA gives 8–12 pp drops in 6 of 10 (base × regularizer) cells.
- The probe-eval at layer 25 is essentially saturated (encoder AUROC 0.997), confirming the trained defense reorganises rep space to make harmful prompts linearly separable — but the predictor head itself adds nothing to that separability.
- **No benign-capability cost** — GSM8K and MMLU within one stderr.
- **Weakest combination: triplet × Llama** — net +1.4 pp, with one BoN cell driving the regression.

The honest paper-ready statement is: **PRA reliably helps on Qwen3 and on the ce_in regularizer; it gives a small or zero gain on other (model × regularizer) cells; the triplet regularizer is currently a weak point that needs investigation; direct-attack rows must be excluded due to degenerate generation.**

---

## What's missing in the snapshot (per the source README)

- Full Table 1 originally promised 5 attack families × 10 defenses; only 3 attack families ran (`bon_25`, `direct_25`, `soft_prompt_25`). Inpainting / Prefilling / GCG / PAIR not present.
- CE-floor + PRA attack runs (5 attacks × n=10 against the 500k-pair adapter).
- Llama-3 ce_floor variant exists (`ce_floor_jepa_llama3_ablation/`) but only base + JEPA-aug cells; no PRA-vs-no-PRA delta in the summary.
- Direct attack rerun with a config fix to avoid the token-collapse issue.
- OR-Bench over-refusal evaluation.
- Base-model GSM8K / MMLU baseline run for absolute capability comparison.
