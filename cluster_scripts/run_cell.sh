#!/usr/bin/env bash
# Run a single cell of an experiment configuration end-to-end on this node.
#
# Invoked from cell.slurm.template (one sbatch job per cell, one node, 4 GPUs).
# Idempotent: relies on the orchestrator's fingerprint+READY skip mechanism, so
# re-submitting a job that crashed mid-run will pick up where the prior attempt
# left off (training adapter saved -> attacks resume; per-attack stage_dir READY
# -> attack skipped).
#
# Required env (set by the slurm template or caller):
#   REPO_DIR          absolute path to the AdversariaLLM checkout
#   CELL              cell/pipeline name from the experiment YAML
#   CONFIG            path to the experiment YAML (relative to REPO_DIR)
#
# Optional env:
#   PYTHON_BIN        python interpreter (default: $REPO_DIR/.venv/bin/python)
#   VENV_ACTIVATE     venv activate script (default: $REPO_DIR/.venv/bin/activate)
#   ENV_FILE          path to a .env to source (default: $REPO_DIR/.env)
#   HF_REPO           HF dataset repo (e.g. user/headline-rerun); enables sync
#   HF_TOKEN          HF write token (read by huggingface_hub)
#   WPF_ROOT          Why-Probe-Fails clone (only needed for probe stages)
#   NUM_GPUS          override GPU count (default: detect from nvidia-smi)
#
# Exit non-zero if the orchestrator failed any non-allow-failure job.

set -euo pipefail

: "${REPO_DIR:?REPO_DIR must be set}"
: "${CELL:?CELL must be set}"
: "${CONFIG:?CONFIG must be set (path to YAML, relative to REPO_DIR)}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO_DIR/.venv/bin/activate}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

cd "$REPO_DIR"

if [ -f "$VENV_ACTIVATE" ]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
if [ -f "$ENV_FILE" ]; then
  set -o allexport
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +o allexport
fi

# These are read by experiments/run_experiment.py via OmegaConf env-var
# interpolation in headline_rerun_full.yaml.
export PYTHON_BIN
export VENV_ACTIVATE
export WPF_ROOT="${WPF_ROOT:-/workspace/Why-Probe-Fails}"
# HF_REPO / HF_TOKEN — used by experiments/hf_sync.py invoked from execute_job.

if [ -n "${NUM_GPUS:-}" ]; then
  echo "[run_cell] NUM_GPUS pinned to $NUM_GPUS via env"
fi

echo "[run_cell] $(date -u +%FT%TZ) cell=$CELL config=$CONFIG repo=$REPO_DIR"
echo "[run_cell] python: $($PYTHON_BIN --version)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Hand off to the orchestrator. local_gpu backend will use all visible GPUs on
# this node (4 by default per the cluster config), assigning one worker per GPU.
# --run-pipelines restricts to this single cell so per-node sbatch dispatch
# doesn't accidentally run other cells.
"$PYTHON_BIN" -m experiments.run_experiment \
  --config "$CONFIG" \
  --backend local_gpu \
  --run-pipelines "$CELL"

echo "[run_cell] $(date -u +%FT%TZ) cell=$CELL done"
