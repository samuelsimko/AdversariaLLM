#!/usr/bin/env bash
# Extract multilingual + JepaData hidden states for all 12 WJ-ablation cells across 8 GPUs.
# After this, runs/why_probe_fails_wj/states/<cell>/ has the full 9-language + 5-attack-family
# coverage matching the CB-trained side, enabling RQ-P3 + RQ-P4 comparisons.

set -uo pipefail
source .env

WPF=/root/AdversariaLLM_full_2026-04-29/Why-Probe-Fails
OUT_BASE=/root/AdversariaLLM/runs/why_probe_fails_wj
DATA_PROBE=data/RS3
DATA_BENIGN=data/RS1
PROBE_LAYER=-1

MULTI_DATASETS="multi_en multi_zh-cn multi_es multi_ja multi_pt multi_id multi_cs multi_ar multi_fy"
JEPA_DATASETS="jepa_direct jepa_prefilling jepa_encoding jepa_bon jepa_distraction"

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
      LOG="$OUT_BASE/logs/${name}_multijepa.log"
      mkdir -p "$OUT_STATES" "$OUT_BASE/logs"
      {
        echo "[$(date +%H:%M:%S)] [$name] GPU $gpu — multi+jepa extraction"

        # Multilingual: malicious + cleaned views
        if ! [[ -f "$OUT_STATES/.extract_done_multi" ]]; then
          cd "$WPF"
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
            --model_path "$base" --adapter_path "$ADAPTER" \
            --data_root "$DATA_PROBE" --views malicious cleaned \
            --datasets $MULTI_DATASETS \
            --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES"
          # Multilingual benign
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
            --model_path "$base" --adapter_path "$ADAPTER" \
            --data_root "$DATA_BENIGN" --views benign \
            --datasets $MULTI_DATASETS \
            --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES" \
            && touch "$OUT_STATES/.extract_done_multi"
        fi

        # JepaData attack families: malicious view (cleaned is self-pair, skip)
        if ! [[ -f "$OUT_STATES/.extract_done_jepa" ]]; then
          cd "$WPF"
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
            --model_path "$base" --adapter_path "$ADAPTER" \
            --data_root "$DATA_PROBE" --views malicious cleaned \
            --datasets $JEPA_DATASETS \
            --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES" \
            && touch "$OUT_STATES/.extract_done_jepa"
        fi

        # Apply PRA predictor to all newly added .npy files (only if predictor exists)
        if [[ -f "$cell/jepa_predictor.pt" && -f "$cell/manifest.json" ]]; then
          mkdir -p "$OUT_STATES_PRED"
          cd "$WPF"
          CUDA_VISIBLE_DEVICES=$gpu /root/AdversariaLLM/venv/bin/python scripts/apply_pra_predictor.py \
            --run_dir "$cell" --states_dir "$OUT_STATES" --out_dir "$OUT_STATES_PRED"
        fi

        echo "[$(date +%H:%M:%S)] [$name] DONE"
      } >> "$LOG" 2>&1
    done <<< "${GPU_QUEUE[$gpu]}"
  ) &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "ALL DONE"
