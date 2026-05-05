# Phase 1 ablation pipeline

Per-GPU sequential `train → probe → benign_eval` orchestrator for the 60+
cell one-axis ablation around an anchor recipe.

## Files

- `scripts/build_ablation_cells.py` — generates the cell list (writes
  `scripts/ablation_cells.json` by default). 72 cells: 6
  `(harm_reg × backbone)` × 12 `(anchor + matched no-PRA + 10 axis
  variations including a P=Identity baseline)`.
- `scripts/ablation_cells.json` — the generated cell list (commit if you
  want to freeze it; otherwise regen on the target machine).
- `scripts/run_ablation_pipeline.py` — orchestrator. Each GPU thread runs
  cells sequentially through `train → probe → benign_eval`, writes one row
  to `SUMMARY.csv` per completed cell.

## Anchor and axes

```
anchor:  w_jepa=5, predictor_lr_multiplier=1, align_layer=30,
         predictor_layers=2, predictor_bottleneck_dim=256
axes:
  w_jepa                ∈ {0 (no-PRA control), 1, 5, 10, 20}
  predictor_lr_mult     ∈ {1, 100}
  align_layer           ∈ {20, 25, 30}
  predictor_type        ∈ {mlp, identity}     # P=Identity baseline
  predictor_layers      ∈ {2, 3}              # only when type=mlp
  predictor_bottleneck  ∈ {64, 256, 512}      # only when type=mlp
harm_reg ∈ {circuit_breaker, ce_floor, triplet}
backbone ∈ {Llama-3-8B-Instruct, Qwen3-8B}
```

Other training knobs match `experiments/configs/headline_pra_joint_n50.yaml`
(reverse-prompt pair_path, circuit_breakers_train cb_path, 1500 steps,
batch=4 grad_accum=2, etc).

## Data prerequisites

These are not in the repo (large files); the orchestrator expects them:

- `data/circuit_breakers_train.json` — from
  [GraySwan-AI/circuit-breakers](https://github.com/GraySwanAI/circuit-breakers/blob/main/data/circuit_breakers_train.json),
  or copy from another clone of this project.
- `reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl` —
  regenerate via `scripts/generate_reverse_prompts.py` (training a reverse
  model first via `scripts/train_reverse_model.py`), or copy from another
  clone.
- `Why-Probe-Fails/` checkout at `/workspace/AdversariaLLM/Why-Probe-Fails`
  with `data/RS3/{malicious,cleaned,paraphrased}/*.csv` and
  `data/RS1/benign/*.csv`. Override the path with the `WPF_ROOT` constant
  near the top of `run_ablation_pipeline.py`.
- `.env` with `HF_TOKEN` (gated Llama-3 repo). Source it in the parent
  shell before launching.

## Running

```bash
# 1. Generate the cell list
PYTHONPATH=. python scripts/build_ablation_cells.py

# 2. Smoke-test 1 cell first
set -a && source .env && set +a
PYTHONPATH=. python scripts/run_ablation_pipeline.py \
    --cells scripts/ablation_cells.json \
    --num-gpus 1 --gpu-ids 0 \
    --out-root runs/ablation_smoke \
    --gsm8k-limit 50 --limit 1

# 3. Full sweep on 8 GPUs (~8h wall)
PYTHONPATH=. python scripts/run_ablation_pipeline.py \
    --cells scripts/ablation_cells.json \
    --num-gpus 8 \
    --out-root runs/ablation_phase1 \
    --gsm8k-limit 200
```

Cells are round-robin distributed across GPUs. Per-cell budget at
production hyperparameters: ~25 min train + ~5 min probe (5-seed sweep) +
~2 min gsm8k(--limit 200) ≈ **~32 min/cell**. With 8 GPUs and 66 cells:
`66/8 × 32 ≈ 4.4h`.

## Outputs

```
runs/ablation_phase1/
  SUMMARY.csv                        # one row per completed cell
  <cell_name>/
    cell.json                        # the cell config
    cell.json
    lora_adapter/                    # trained LoRA
    jepa_predictor.pt                # trained predictor
    manifest.json
    metrics.csv
    TRAIN_READY                      # sentinel: training done
    probe/
      states/                        # extracted hidden states
      seed_{42,123,777,1234,9999}/
        results.csv                  # per-(probe, slice) AUC
    PROBE_READY
    benign/
      results*.json                  # lm-eval gsm8k results
    BENIGN_READY
    logs/{train,probe,benign}.log
```

`SUMMARY.csv` columns: name, gpu, backbone, harm_regularizer, w_jepa,
predictor_lr_multiplier, align_layer, predictor_layers,
predictor_bottleneck_dim, train_status, train_seconds, probe_status,
probe_seconds, benign_status, benign_seconds, auc_svm_raw,
auc_mlp_no_jepa, auc_jepa, auc_svm_on_jepa_z, gsm8k_em, cell_seconds.

The AUC columns are the 5-seed mean on the
`ood_heldout_paraphrased_harmbench` slice (the JEPD paper's headline
worst-slice).

## Re-running

The orchestrator writes `TRAIN_READY`, `PROBE_READY`, `BENIGN_READY`
sentinels per cell. Re-launching skips cached stages. Safe to Ctrl-C
mid-run; the cell currently in flight will be killed but `SUMMARY.csv`
preserves all completed rows.

## Skipping benign eval

`--skip-benign` to drop the gsm8k stage. Saves ~2 min/cell, ~2h for the
full 66-cell sweep on 1 GPU.
