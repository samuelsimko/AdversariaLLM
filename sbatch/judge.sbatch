#!/usr/bin/env bash
#SBATCH --job-name=advllm-judge
#SBATCH --account=ls_schoel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Re-score existing run.json files with a different judge.
#
# Examples:
#   sbatch sbatch/judge.sbatch classifier=strong_reject
#   sbatch sbatch/judge.sbatch classifier=local:ensemble
#   sbatch sbatch/judge.sbatch classifier=local:gpt_oss

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
source "${REPO_DIR}/sbatch/common.sh"

srun python run_judges.py hydra/launcher=basic "$@"
