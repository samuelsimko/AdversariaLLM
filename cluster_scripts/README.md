# cluster_scripts/

One-cell-per-node sbatch dispatch for the headline_rerun experiment. Each
sbatch job runs the full pipeline (train → attacks → benign evals) for a
single cell on a 4-GPU node, syncing results to a private HuggingFace
dataset repo as each stage completes.

> **First-time setup**: read `SETUP.md` end-to-end before running anything.
> It covers the gated-dataset acceptance, the `.env` contents, cluster
> customization (account/partition/etc.), and the recommended smoke-test flow.

## What's here

| File | Purpose |
|---|---|
| `cell.slurm.template` | sbatch template; placeholders filled in by `submit_all.sh`. |
| `run_cell.sh` | The script the sbatch job runs; calls `experiments/run_experiment.py --backend local_gpu --run-pipelines $CELL`. |
| `submit_all.sh` | Generates per-cell sbatch scripts from the template and submits them. |
| `sync_status.py` | Dashboard over local READY sentinels + HF repo state. |
| `generated/` | Per-cell `.slurm` files written by `submit_all.sh`. |
| `logs/` | sbatch stdout/stderr per cell × jobid. |

The orchestrator (`experiments/run_experiment.py`) handles per-stage
fingerprinting, READY sentinels, dependency tracking, and HF sync via
`experiments/hf_sync.py` (called from `experiments/execute_job.py` after
each successful stage). Sbatch retries are safe — completed stages are
skipped via fingerprint+READY.

## Setup (one-time)

1. **Clone and install on the cluster's shared filesystem.**
   ```bash
   git clone <repo> /workspace/AdversariaLLM
   cd /workspace/AdversariaLLM
   git submodule update --init    # strong_reject + latent-adversarial-training
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e . --no-deps
   ```

2. **Generate training data** (if not already on the shared filesystem):
   ```bash
   python scripts/build_wildjailbreak_data.py
   # writes data/wildjailbreak_{pairs,harmful,benign}.{jsonl,json}
   ```

3. **Create the private HF dataset repo:**
   ```bash
   huggingface-cli login    # one-time; or just set HF_TOKEN
   python -c "from huggingface_hub import HfApi; HfApi().create_repo(
       'YOUR_USER/headline-rerun', repo_type='dataset', private=True, exist_ok=True)"
   ```

4. **Smoke-test on one cell** before submitting the full sweep:
   ```bash
   export HF_REPO=YOUR_USER/headline-rerun
   export HF_TOKEN=hf_...
   bash cluster_scripts/submit_all.sh q_cb_pra
   # watch logs/q_cb_pra.<jobid>.out, verify it lands on HF
   python cluster_scripts/sync_status.py
   ```

## Submitting the full sweep

```bash
export HF_REPO=YOUR_USER/headline-rerun
export HF_TOKEN=hf_...
# Optional: override slurm defaults if your account/partition differs.
export ACCOUNT=infra01 PARTITION=normal TIME=04:00:00 GPUS_PER_NODE=4 MEM=460800
bash cluster_scripts/submit_all.sh
```

Submits one job per cell in `experiments/configs/headline_rerun_full.yaml`
(19 cells: 16 trained + 3 reference defenses). Each job is independent;
slurm distributes them across nodes automatically.

To submit a subset of cells:
```bash
bash cluster_scripts/submit_all.sh q_cb_pra q_cb_no_pra q_lat_pra q_lat_no_pra
```

To preview the sbatch scripts without submitting:
```bash
DRY_RUN=1 bash cluster_scripts/submit_all.sh
ls cluster_scripts/generated/
```

## Tracking progress

```bash
python cluster_scripts/sync_status.py
# or HF only (when running from a different machine):
python cluster_scripts/sync_status.py --hf-only
```

Output looks like:

```
Experiment: headline_rerun_full  (19 cells)
================================================================================

LOCAL (runs/experiments/...):
  cell                   train  #attacks  #evals  attacks
  l_cb_pra                 ✓       7        2     bon_50,direct_300,human_jailbreaks_300,inpainting_50,or_bench_overrefusal,prefilling_300,soft_prompt_100
  l_cb_no_pra              ✓       4        2     direct_300,or_bench_overrefusal,prefilling_300,soft_prompt_100
  ...

HF (YOUR_USER/headline-rerun):
  cell                   adapter  #attacks  #evals
  l_cb_pra                  ✓        7        2
  ...
```

## Cells in `headline_rerun_full.yaml`

| Group | Cells | Defense script |
|---|---|---|
| CB ± PRA × {l, q} | l/q_cb_{pra,no_pra} | `defenses/jepa_ce.py` (`harm_regularizer=circuit_breaker`) |
| Triplet ± PRA × {l, q} | l/q_triplet_{pra,no_pra} | `defenses/triplet_simko.py` |
| CE-floor ± PRA × {l, q} | l/q_ce_in_{pra,no_pra} | `defenses/jepa_ce.py` (`harm_regularizer=ce_floor`, `harm_ce_min=3.0`) |
| LAT ± PRA × {l, q} | l/q_lat_{pra,no_pra} | `defenses/lat.py` (PGD on residual stream, layers 8/16/24/30) |
| Reference defenses | ref_grayswan_rr, ref_simko_triplet, ref_lat_l, ref_dat_l | none (attack the published checkpoint) |

Reference defense models:
- `GraySwanAI/Llama-3-8B-Instruct-RR` — Zou et al. circuit-breakers checkpoint.
- `samuelsimko/Meta-Llama-3-8B-Instruct-Triplet` — Simko triplet checkpoint.
- `LLM-LAT/robust-llama3-8b-instruct` — Sheshadri et al. LAT checkpoint.
- `ASSELab/DAT-Llama-3-8B-Instruct` — Hu, Dornbusch, Lüdke, Günnemann, Schwinn
  2026 *Distributional Adversarial Training* (continuous AT against
  diffusion-based / inpainting adversaries). Full checkpoint, not a LoRA.

## Per-cell pipeline

Each cell runs (from `headline_rerun_full.yaml:pipelines`):

1. **train** (single GPU, ~30 min for CB / CE-floor / triplet, ~2 h for LAT)
   - LoRA r=32, rank-stable across cells
   - saves `runs/experiments/headline_rerun_full/<cell>/lora_adapter/`
   - hf_sync hook pushes adapter immediately on `READY`
2. **attacks** (4 GPUs, priority order):
   `soft_prompt_100 → direct_300 → prefilling_300 → human_jailbreaks_300
    → or_bench_overrefusal → bon_50 → inpainting_50`
   - each attack's stage_dir gets pushed on `READY`
3. **benign evals** (4 GPUs, parallel with leftover attack work):
   `gsm8k_200, mmlu_full`

Reference cells skip step 1.

## Costs

Per cell, 1 node, 4×H100:
- CB / CE-floor / triplet: train ~30 min + attacks ~50 min ≈ **80 min**
- LAT: train ~2 h + attacks ~50 min ≈ **2 h 50 min**
- Reference: attacks only ~50 min

With 16 trained + 3 reference cells across 16 nodes:
- LAT cells are the long pole; non-LAT cells finish in ~80 min.
- Wall-clock to all-cells-done ≈ **3 hours** (LAT-bound).
- Total node-hours ≈ 16 cells × 80–170 min + 3 ref × 50 min ≈ **30–40 node-hours**.

## Failure modes & recovery

- **Node dies mid-train**: adapter not yet pushed; re-submit the cell's
  sbatch. Orchestrator's fingerprint+READY check sees no adapter and re-trains.
- **Node dies mid-attack**: completed attacks already on HF; re-submit
  re-runs only the missing attacks (per-attack stage_dir READY skips done ones).
- **HF API hiccup during sync**: `hf_sync.py` catches and logs the exception,
  job still succeeds locally. Re-run `sync_status.py` and manually push any
  missing folders with `python -m experiments.hf_sync push <stage_dir>
  <fingerprint.json> <job_type>`.
- **One cell fails repeatedly**: skip it with `--exclude` or just don't
  re-submit. Other cells unaffected.

## Notes / TODOs

- `human_jailbreaks` attack relies on a per-behavior corpus shipped with
  AdversariaLLM; should work out of the box for `adv_behaviors` indices 0-300.
- `inpainting` attack only works on `jbb_behaviors`; it ships its own
  inpainting-prompt pool. Don't switch its dataset.
- Probing (Para HB) is wired into the YAML's `probes:` block but not
  included in the default `pipelines:` because it requires the
  Why-Probe-Fails repo to be on the node. Add a `{ stage: probe, defense: ...,
  probe: para_hb }` step per pipeline once `WPF_ROOT` is mounted.
- `vllm` should NOT be installed in the venv (CUDA mismatch with our pinned
  torch). The smoke test in step 4 above is the cheapest way to catch this.
