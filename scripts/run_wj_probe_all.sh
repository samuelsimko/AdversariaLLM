#!/usr/bin/env bash
# Run probe extraction + comparison for all WJ-ablation cells on GPUs 2-7.
# Idempotent: skips cells already done (results.csv exists).

set -uo pipefail
source .env

declare -a CELLS=(
  "/root/AdversariaLLM/runs/experiments/ablation_wj/q_cb_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj/q_cb_no_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_triplet_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_triplet_no_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_ce_in_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_ce_in_no_pra_wj|Qwen/Qwen3-8B"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_cb_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_cb_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_triplet_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_triplet_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_ce_in_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_ce_in_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
)

GPUS=(2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
PIDS=()

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  (
    for j in "${!CELLS[@]}"; do
      if [ $((j % NUM_GPUS)) -ne $i ]; then continue; fi
      entry="${CELLS[$j]}"
      cell="${entry%|*}"
      base="${entry#*|}"
      name=$(basename "$cell")
      bash scripts/run_wj_probe_one.sh "$gpu" "$cell" "$base" "$name"
    done
  ) &
  PIDS+=($!)
done
wait "${PIDS[@]}"
echo "ALL DONE"
