#!/usr/bin/env bash
#SBATCH --job-name=advllm-smoke
#SBATCH --account=ls_schoel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Smallest end-to-end sanity check: tiny model, direct attack, 2 prompts.
# Verifies env + modules + venv + HF cache + slurm wiring.

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
source "${REPO_DIR}/sbatch/common.sh"

srun python run_attacks.py hydra/launcher=basic \
  model=Qwen/Qwen2.5-0.5B-Instruct \
  dataset=adv_behaviors \
  'datasets.adv_behaviors.idx=[0,1]' \
  attack=direct \
  classifiers='[]' \
  name=smoke
