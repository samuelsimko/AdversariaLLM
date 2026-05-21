# TORUN: `qwen3_32b_ce_cb.yaml`

Hand-off doc for running the Qwen3-32B CE-floor vs Circuit-Breakers × identity vs
MLP-PRA grid on Euler (or another slurm cluster with similar conventions).

## What this YAML runs

5 cells, all on **Qwen/Qwen3-32B**:

| cell           | harm regularizer | PRA predictor          |
| -------------- | ---------------- | ---------------------- |
| `q_base`       | (none — no train) | —                      |
| `q_ce_pra_id`  | ce_floor          | identity               |
| `q_ce_pra`     | ce_floor          | mlp 2L, bottleneck=256 |
| `q_cb_pra_id`  | circuit_breaker   | identity               |
| `q_cb_pra`     | circuit_breaker   | mlp 2L, bottleneck=256 |

Per cell:
- **train** (skipped for `q_base`)
- **3 attacks** on `adv_behaviors[0..99]`: `direct`, `prefilling`, `template_jailbreak`
  (the last samples 20 PyRIT templates per behavior — 2000 forward passes/cell)
- **mmlu benign eval**, `--limit 200`

→ 24 sbatch jobs in total: 4 train + 15 attack + 5 benign.

## Prerequisites

### Repo + venv
```bash
# Repo lives at: /cluster/project/schoelkopf/ssimko/AdversariaLLM
# venv:           /cluster/project/schoelkopf/ssimko/AdversariaLLM/venv
# python_bin in YAML is already pinned to that path. If you clone elsewhere,
# update `runtime.python_bin` and `runtime.venv_activate` in the YAML.
```

### Environment file
`./.env` (not committed) must export:
```bash
export HF_TOKEN=...                              # for HF Hub downloads
export STRONGREJECT_PATH=$(pwd)/strong_reject   # vendored submodule
export WANDB_API_KEY=...                         # optional; set report_to=none in YAML if missing
```

### Data files (built once, not committed because they're regenerable)
```bash
# Builds: data/wildjailbreak_{harmful.json,benign.jsonl,pairs.jsonl}
python scripts/build_wildjailbreak_data.py
```

### HF cache pre-population (compute nodes are offline)
**All model + dataset downloads must happen from a login node.** Compute nodes
have no internet — `HF_HUB_OFFLINE=1` is set automatically by `sbatch/common.sh`.

```bash
source .env
export HF_HOME=/cluster/scratch/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub

# Base model (~62 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-32B')"

# Strongreject judge (Gemma-2B LoRA + base ~5 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('qylu4156/strongreject-15k-v1'); snapshot_download('google/gemma-2b')"

# Datasets (small)
python -c "
from datasets import load_dataset
load_dataset('HuggingFaceH4/ultrachat_200k', split='train_sft')
load_dataset('allenai/wildjailbreak', 'train', delimiter='\t', keep_default_na=False)
load_dataset('allenai/wildguardmix', 'wildguardtrain')
"

# MMLU per-subject configs for lm-eval (must enumerate all 57 — don't use 'all')
python -c "
from datasets import load_dataset, get_dataset_config_names
for c in get_dataset_config_names('cais/mmlu'):
    try:
        load_dataset('cais/mmlu', c, split='test')
        load_dataset('cais/mmlu', c, split='dev')
    except Exception as e:
        print(f'WARN {c}: {e}')
"
```

### Apertus chat template (only needed if you also run the Apertus grid)
Skip for this YAML.

## Submitting

From a **login node** (so `sbatch` is available):

```bash
source modules.sh
source venv/bin/activate
source .env
cd /cluster/project/schoelkopf/ssimko/AdversariaLLM
PYTHONPATH=. python experiments/run_experiment.py \
    --config experiments/configs/qwen3_32b_ce_cb.yaml \
    --backend slurm
```

The orchestrator submits each stage as its own sbatch job (chained via
`--dependency=afterok:<train_id>`). Attack and benign stages won't start until
the train job for their cell completes.

Outputs land under `runs/experiments/qwen3_32b_ce_cb/`:

```
runs/experiments/qwen3_32b_ce_cb/
├── q_base/
│   ├── attacks/q_base/{direct_100,prefilling_100,template_jb_100_n20}/...
│   └── benign_eval/q_base/mmlu_smoke/...
├── q_ce_pra_id/
│   ├── lora_adapter/     ← trained defense
│   ├── attacks/...
│   └── benign_eval/...
└── ...  (same for q_ce_pra, q_cb_pra_id, q_cb_pra)
```

## GPU memory math

Qwen3-32B in bf16 = ~64 GB. On a single 80 GB A100 the training stage is tight:

```
weights (bf16):       64 GB
LoRA grads + AdamW:    1 GB
activations bs=1 sl=128: ~4 GB
JEPA double-forward:  ~6 GB
─────────────────────────
total estimate:      ~75 GB     (leaving ~5 GB headroom)
```

If you hit OOM:
1. Cut `max_length: 128 → 96` (and/or `grad_accum: 8 → 16`)
2. Request 2× 80 GB GPUs (`cluster.gpus: "nvidia_a100_80gb_pcie:2"` — model will
   shard automatically via `device_map="auto"`); also bump `cluster.cpus_per_task`
3. Add 4-bit QLoRA support to `defenses/jepa_ce.py` (not currently wired — would
   need `BitsAndBytesConfig` + `prepare_model_for_kbit_training` from PEFT)

## Monitoring

```bash
# Queue depth + per-cell completion
squeue -u $USER
ls runs/experiments/qwen3_32b_ce_cb/*/{READY,FAILED} 2>/dev/null

# ASR + benign once it's done
PYTHONPATH=. python scripts/analyze_run_asr.py qwen3_32b_ce_cb
```

## Quirks to watch for (we hit these on the Apertus grid; fix already in
this repo's slurm backend)

- **Euler `--gres=gpu:<type>:N` silently drops the type.** Use
  `gpus: "<type>:N"` in YAML; backend translates to a working form.
- **Mid-run `BadConstraints` (exit 0:53)** is slurm-side scheduler noise. Just
  resubmit the failing cell (orchestrator skips READY stages so you don't redo
  the rest).
- **Don't manually `rm -rf` a stage_dir while a job is in flight** — you'll
  delete the LoRA adapter the attack job is mid-load on. Cancel the job first.
- **`run_judges.py` skips paths whose `scored_by` already contains the
  classifier.** If you need to re-judge (e.g. because the judge model was
  broken offline-cache first time), use `scripts/rejudge_apertus_grid.py` as a
  template — it walks `runs/experiments/<exp>/**/run.json` and overwrites the
  scores directly.

## Expected timing (rough, A100-80GB)

| stage       | per cell  | total grid |
| ----------- | --------- | ---------- |
| train       | 3–5 h     | 12–20 h    |
| direct_100  | 5–10 min  | 25–50 min  |
| prefilling_100 | 5–10 min | 25–50 min |
| template_jb_100_n20 | 2–4 h | 10–20 h |
| benign mmlu | 30–60 min | 2.5–5 h    |

Total wall-clock if everything queues smoothly: ~24–36 h.
