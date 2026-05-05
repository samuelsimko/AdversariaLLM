#!/usr/bin/env bash
# Extract dolly hidden states + re-run compare_probes for all WJ cells, on GPUs 0-7.
# After this, runs/why_probe_fails_wj/runs/<cell>/results.csv will have proper
# harmbench AUCs (since dolly is the held-out benign in the default WPF config).

set -uo pipefail
source .env

WPF=/root/AdversariaLLM_full_2026-04-29/Why-Probe-Fails
OUT_BASE=/root/AdversariaLLM/runs/why_probe_fails_wj
DATA_BENIGN=data/RS1
PROBE_LAYER=-1
PROBE_CFG=configs/rs3_advbench_jepa.json

# (gpu, cell_dir, base_model)
declare -a CELLS=(
  "0|/root/AdversariaLLM/runs/experiments/ablation_wj/q_cb_pra_wj|Qwen/Qwen3-8B"
  "1|/root/AdversariaLLM/runs/experiments/ablation_wj/q_cb_no_pra_wj|Qwen/Qwen3-8B"
  "2|/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_triplet_pra_wj|Qwen/Qwen3-8B"
  "3|/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_triplet_no_pra_wj|Qwen/Qwen3-8B"
  "4|/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_ce_in_pra_wj|Qwen/Qwen3-8B"
  "5|/root/AdversariaLLM/runs/experiments/ablation_wj_more/q_ce_in_no_pra_wj|Qwen/Qwen3-8B"
  "6|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_cb_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "7|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_cb_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "0|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_triplet_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "1|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_triplet_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "2|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_ce_in_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
  "3|/root/AdversariaLLM/runs/experiments/ablation_wj_more/l_ce_in_no_pra_wj|meta-llama/Meta-Llama-3-8B-Instruct"
)

# Per-GPU queues to avoid contention
declare -A GPU_QUEUE
for entry in "${CELLS[@]}"; do
  IFS='|' read -r gpu cell base <<< "$entry"
  GPU_QUEUE[$gpu]+="${cell}|${base}"$'\n'
done

PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
  [[ -z "${GPU_QUEUE[$gpu]:-}" ]] && continue
  (
    while IFS='|' read -r cell base; do
      [[ -z "$cell" ]] && continue
      name=$(basename "$cell")
      ADAPTER="$cell/lora_adapter"
      OUT_STATES="$OUT_BASE/states/$name"
      OUT_STATES_PRED="$OUT_BASE/states_predictor/$name"
      OUT_RUNS="$OUT_BASE/runs/$name"
      OUT_RUNS_PRED="$OUT_BASE/runs_predictor/$name"
      LOG="$OUT_BASE/logs/${name}_dolly.log"
      mkdir -p "$OUT_STATES" "$OUT_RUNS"
      {
        echo "[$(date +%H:%M:%S)] [$name] GPU $gpu — extracting dolly"
        if ! [[ -f "$OUT_STATES/benign_dolly.npy" ]]; then
          cd "$WPF"
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
            --model_path "$base" --adapter_path "$ADAPTER" \
            --data_root "$DATA_BENIGN" --views benign \
            --datasets dolly \
            --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES"
        fi
        # Re-run compare_probes to get harmbench AUCs filled in
        rm -rf "$OUT_RUNS"; mkdir -p "$OUT_RUNS"
        cd "$WPF"
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$WPF" /root/AdversariaLLM/venv/bin/python scripts/compare_probes.py \
          --config "$PROBE_CFG" --states_dir "$OUT_STATES" \
          --out_dir "$OUT_RUNS" --probes svm_raw mlp_no_jepa --with_analyses

        # Predictor pass (only if predictor exists)
        if [[ -f "$cell/jepa_predictor.pt" && -f "$cell/manifest.json" ]]; then
          mkdir -p "$OUT_STATES_PRED"
          cd "$WPF"
          # Re-apply predictor to all .npy files including the new dolly
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/apply_pra_predictor.py \
            --run_dir "$cell" --states_dir "$OUT_STATES" --out_dir "$OUT_STATES_PRED"
          rm -rf "$OUT_RUNS_PRED"; mkdir -p "$OUT_RUNS_PRED"
          cd "$WPF"
          CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$WPF" /root/AdversariaLLM/venv/bin/python scripts/compare_probes.py \
            --config "$PROBE_CFG" --states_dir "$OUT_STATES_PRED" \
            --out_dir "$OUT_RUNS_PRED" --probes svm_raw mlp_no_jepa --with_analyses
        fi
        echo "[$(date +%H:%M:%S)] [$name] DONE"
      } >> "$LOG" 2>&1
    done <<< "${GPU_QUEUE[$gpu]}"
  ) &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "ALL DONE"
