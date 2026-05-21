# sbatch templates

Generic sbatch wrappers for AdversariaLLM on the ETH Euler cluster.

## Conventions

Every template:

- Uses `--account=ls_schoel` and Euler-style `--gpus=N` (override with `sbatch --gpus=N ...`).
- Sources `sbatch/common.sh`, which loads `modules.sh`, activates `venv/`, points HF caches at `/cluster/scratch/ssimko/`, and exports `CUBLAS_WORKSPACE_CONFIG`.
- Writes logs to `logs/<job-name>-<job-id>.out|.err`.
- Forwards extra CLI args / env vars to the underlying entry point.

## Files

| File | Entry point | Common use |
|------|-------------|------------|
| `common.sh` | (sourced) | shared env setup |
| `smoke.sbatch` | `run_attacks.py` | 30-min, 1-GPU, 2-prompt direct attack on gemma-2-2b — sanity check |
| `attack.sbatch` | `run_attacks.py` | one attack/model/dataset slice |
| `judge.sbatch` | `run_judges.py` | re-score existing `run.json` files |
| `train_defense.sbatch` | `defenses/<script>.py` | train a single defense |
| `experiment.sbatch` | `experiments/run_experiment.py` | stitched train→attack→benign pipeline from a YAML |

## Usage

```bash
# Smoke test (smallest possible job — submit this first after any env change).
sbatch sbatch/smoke.sbatch

# Real attack run (override defaults via sbatch flags + hydra args).
sbatch --time=02:00:00 --gpus=1 sbatch/attack.sbatch \
  attack=gcg attacks.gcg.num_steps=200 \
  model=meta-llama/Meta-Llama-3-8B-Instruct \
  dataset=adv_behaviors 'datasets.adv_behaviors.idx=[0,1,2,3]'

# Train a defense.
DEFENSE_SCRIPT=defenses/jepa_ce.py sbatch --gpus=1 --time=08:00:00 \
  sbatch/train_defense.sbatch \
  --model Qwen/Qwen3-8B \
  --cb_path data/circuit_breakers_train.json \
  --pair_path reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl \
  --pair_format reverse \
  --output_dir runs/jepa_ce_smoke \
  --num_max_steps 50

# Run a multi-stage experiment.
CONFIG=experiments/configs/library_pipeline_example.yaml \
  sbatch --gpus=1 --time=12:00:00 sbatch/experiment.sbatch

# Re-judge with a different classifier.
sbatch sbatch/judge.sbatch classifier=local:strongreject
```

## Notes

- These wrappers pass `hydra/launcher=basic` for `run_attacks.py` / `run_judges.py`, so the Hydra sweep runs inside the single allocated job. To use the cluster-aware submitit launcher (one sbatch per sweep cell), call `python run_attacks.py -m ...` from a login node and let Hydra submit the jobs.
- For >1 GPU, just pass `--gpus=N` to `sbatch`; the defense / experiment scripts honor `accelerate` / `torch.cuda.device_count()` automatically.
- All output / `run.json` files go under `outputs/<date>/<time>/`. The metadata DB lives at `outputs/runs.sqlite3`.
