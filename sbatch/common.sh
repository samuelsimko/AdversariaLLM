#!/usr/bin/env bash
# Shared environment for all AdversariaLLM sbatch jobs.
# Sourced from inside each sbatch script after #SBATCH directives.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
cd "${REPO_DIR}"

# Modules + venv.
# shellcheck disable=SC1091
source "${REPO_DIR}/modules.sh"
# shellcheck disable=SC1091
source "${REPO_DIR}/venv/bin/activate"

# Scratch root for all caches/temp. NEVER use /tmp on this cluster.
: "${SCRATCH:=/cluster/scratch/ssimko}"
export SCRATCH
# Unconditional overrides — user's .bashrc exports HF_HOME to a stale
# /cluster/project/sachan/... path; force everything to scratch.
# Do NOT set TRANSFORMERS_CACHE (deprecated, and pointing it elsewhere makes
# transformers ignore the HF_HOME/hub cache where snapshot_download writes).
export HF_HOME="${SCRATCH}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
unset TRANSFORMERS_CACHE
export TORCH_HOME="${SCRATCH}/torch"
export PIP_CACHE_DIR="${SCRATCH}/pip_cache"
export TMPDIR="${TMPDIR:-${SCRATCH}/tmp}"
mkdir -p "${HF_HOME}/hub" "${TORCH_HOME}" "${TMPDIR}" "${REPO_DIR}/logs"

# Compute nodes on this cluster have no internet — force HF offline inside jobs
# so missing-cache turns into a fast, clear error instead of a network timeout.
# Override with HF_FORCE_ONLINE=1 to disable (e.g. when running on a login node).
if [[ -n "${SLURM_JOB_ID:-}" && "${HF_FORCE_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# Determinism: must be set before any torch import.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Optional .env (HF_TOKEN, WANDB_API_KEY, OPENAI_API_KEY, ...).
if [[ -f "${REPO_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a; source "${REPO_DIR}/.env"; set +a
fi

echo "[$(date +%F\ %T)] node=$(hostname) gpus=${SLURM_GPUS_ON_NODE:-?} job=${SLURM_JOB_ID:-local}"
echo "[$(date +%F\ %T)] python=$(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
