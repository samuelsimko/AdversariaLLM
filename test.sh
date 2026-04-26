#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
source ../honeypot_llm_defense/.env

RUN_DIR="runs/jepa_ce_wild_benign_bottleneck64_500step"

python defenses/jepa_ce.py \
  --model Qwen/Qwen3-8B \
  --cb_path data/circuit_breakers_train.json \
  --pair_path reverse_model/cb_train_reverse_prompts_5000_random_temp.jsonl \
  --pair_format reverse \
  --extra_pair_dataset allenai/wildjailbreak \
  --extra_pair_dataset_config train \
  --extra_pair_split train \
  --extra_pair_format wildjailbreak \
  --extra_pair_limit 200000 \
  --output_dir "${RUN_DIR}" \
  --num_max_steps 500 \
  --batch_size 8 \
  --grad_accum 2 \
  --ultrachat_samples 20000 \
  --limit_cb 20000 \
  --pair_limit 1000000 \
  --pair_sample_size 32000 \
  --pair_sample_balanced \
  --max_length 256 \
  --harm_ce_min 5.0 \
  --w_benign 1.0 \
  --w_harm 1.0 \
  --w_jepa 0.5 \
  --w_kl 0.1 \
  --predictor_type mlp \
  --predictor_layers 2 \
  --predictor_bottleneck_dim 64 \
  --report_to wandb \
  --run_name jepa_ce_wild_benign_bottleneck64_500step \
  --save_steps 250 \
  --logging_steps 10

python scripts/evaluate_jepa_guardrail.py \
  --run_dir "${RUN_DIR}" \
  --output_json "${RUN_DIR}/guardrail_wildjailbreak_eval.json" \
  --centroid_per_category 64 \
  --eval_per_category 64 \
  --benign_centroid_samples 512 \
  --benign_eval_samples 1024 \
  --jailbreak_eval_samples 1024 \
  --batch_size 8 \
  --max_length 256 \
  --target_fpr 0.01 \
  --wildjailbreak_dataset allenai/wildjailbreak \
  --wildjailbreak_config eval \
  --wildjailbreak_split train \
  --wildjailbreak_limit 2210
