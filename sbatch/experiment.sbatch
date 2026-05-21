#!/usr/bin/env bash
#SBATCH --job-name=advllm-exp
#SBATCH --account=ls_schoel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run an experiments/configs/*.yaml end-to-end with experiments/run_experiment.py.
# Backend=local_gpu so all stages execute inside this slurm job.
# (Use --backend slurm only when running from a login node — that backend
# submits each stage as its own sbatch with afterok chains.)
#
# Required env vars:
#   CONFIG  path to a YAML under experiments/configs/
#
# Examples:
#   CONFIG=experiments/configs/library_pipeline_example.yaml \
#     sbatch sbatch/experiment.sbatch
#
#   CONFIG=experiments/configs/jepa_llama3_overnight.yaml \
#     sbatch --gpus=2 --time=48:00:00 sbatch/experiment.sbatch
#
#   # Override backend / extra flags:
#   CONFIG=... BACKEND=local sbatch sbatch/experiment.sbatch --dry-run

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
source "${REPO_DIR}/sbatch/common.sh"

: "${CONFIG:?Set CONFIG=experiments/configs/<name>.yaml}"
BACKEND="${BACKEND:-local_gpu}"

srun python experiments/run_experiment.py --config "${CONFIG}" --backend "${BACKEND}" "$@"
