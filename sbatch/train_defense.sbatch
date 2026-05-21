#!/usr/bin/env bash
#SBATCH --job-name=advllm-train
#SBATCH --account=ls_schoel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Train one defense via defenses/<script>.py.
#
# Required env vars:
#   DEFENSE_SCRIPT  path to a defense training script under defenses/
#                   (e.g. defenses/jepa_ce.py, defenses/honeypot_cb.py)
# Pass-through:    every additional positional arg is forwarded to the script.
#
# Examples:
#   DEFENSE_SCRIPT=defenses/jepa_ce.py sbatch sbatch/train_defense.sbatch \
#     --model Qwen/Qwen3-8B \
#     --cb_path data/circuit_breakers_train.json \
#     --pair_path reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl \
#     --pair_format reverse \
#     --output_dir runs/jepa_ce_smoke \
#     --num_max_steps 50 --batch_size 4 --grad_accum 1 --logging_steps 5
#
# Multi-GPU example (let HF accelerate see them all):
#   DEFENSE_SCRIPT=defenses/jepa_ce.py sbatch --gpus=4 --time=24:00:00 \
#     sbatch/train_defense.sbatch ...

REPO_DIR="${REPO_DIR:-/cluster/project/schoelkopf/ssimko/AdversariaLLM}"
source "${REPO_DIR}/sbatch/common.sh"

: "${DEFENSE_SCRIPT:?Set DEFENSE_SCRIPT=defenses/<script>.py}"

srun python "${DEFENSE_SCRIPT}" "$@"
