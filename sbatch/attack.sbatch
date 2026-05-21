#!/usr/bin/env bash
#SBATCH --job-name=advllm-attack
#SBATCH --account=ls_schoel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Run a single attack via run_attacks.py with the basic (in-process) Hydra launcher.
#
# Examples:
#   sbatch sbatch/attack.sbatch \
#     attack=direct \
#     model=meta-llama/Meta-Llama-3-8B-Instruct \
#     dataset=adv_behaviors \
#     'datasets.adv_behaviors.idx=[0,1,2,3]'
#
#   # Override hyperparameters:
#   sbatch --time=08:00:00 --gpus=1 sbatch/attack.sbatch \
#     attack=gcg attacks.gcg.num_steps=200 \
#     model=Qwen/Qwen3-8B dataset=adv_behaviors \
#     'datasets.adv_behaviors.idx=[0,1,2,3,4,5,6,7]'
#
# Note: This wrapper passes hydra/launcher=basic so the whole sweep runs inside
# THIS slurm job (no nested submitit submission). For a submitit-based sweep,
# call `python run_attacks.py -m ...` from a login node instead.

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
source "${REPO_DIR}/sbatch/common.sh"

srun python run_attacks.py hydra/launcher=basic "$@"
