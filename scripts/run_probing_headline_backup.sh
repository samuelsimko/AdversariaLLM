#!/usr/bin/env bash
# Probe all 8 cells of headline_backup_rerun on a single GPU sequentially.
# Idempotent: skips cells whose results.csv already exists.
#
# Usage:
#   bash scripts/run_probing_headline_backup.sh [gpu_id]    # default gpu_id=0
#
# Outputs per cell:
#   runs/why_probe_fails_headline_backup/states/<cell>/         hidden states
#   runs/why_probe_fails_headline_backup/runs/<cell>/results.csv probe AUC table
#   runs/why_probe_fails_headline_backup/logs/<cell>.log
#
# Probe protocol (from Why-Probe-Fails/configs/rs3_advbench_jepa.json):
#   - Train SVM / MLP on AdvBench malicious vs Alpaca benign at align_layer=25
#   - Eval on held-out HarmBench (malicious / cleaned / paraphrased views)
#     and on paraphrased AdvBench. AUROC drops on paraphrased / OOD =
#     classic "probe-fails" generalization gap.

set -uo pipefail

GPU="${1:-0}"
WPF=/root/AdversariaLLM/Why-Probe-Fails
PYTHON=/root/AdversariaLLM/.venv/bin/python
EXP_ROOT=/root/AdversariaLLM/runs/experiments/headline_backup_rerun
OUT_BASE=/root/AdversariaLLM/runs/why_probe_fails_headline_backup
PROBE_LAYER=25
PROBE_CFG=configs/rs3_advbench_jepa.json
DATA_RS3="$WPF/data/RS3"
DATA_RS1="$WPF/data/RS1"

mkdir -p "$OUT_BASE/states" "$OUT_BASE/runs" "$OUT_BASE/logs"

# Source secrets (HF_TOKEN for adapter downloads if any model is gated).
if [ -f /root/AdversariaLLM/.env ]; then
  set -o allexport
  source /root/AdversariaLLM/.env
  set +o allexport
fi

# (cell, base_model) tuples
declare -a CELLS=(
  "q_cb_pra|Qwen/Qwen3-8B"
  "q_cb_no_pra|Qwen/Qwen3-8B"
  "q_triplet_pra|Qwen/Qwen3-8B"
  "q_triplet_no_pra|Qwen/Qwen3-8B"
  "l_cb_pra|meta-llama/Meta-Llama-3-8B-Instruct"
  "l_cb_no_pra|meta-llama/Meta-Llama-3-8B-Instruct"
  "l_triplet_pra|meta-llama/Meta-Llama-3-8B-Instruct"
  "l_triplet_no_pra|meta-llama/Meta-Llama-3-8B-Instruct"
)

echo "[$(date +%H:%M:%S)] probing ${#CELLS[@]} cells on GPU $GPU"

for entry in "${CELLS[@]}"; do
  cell="${entry%|*}"
  base="${entry#*|}"
  cell_dir="$EXP_ROOT/$cell"
  adapter="$cell_dir/lora_adapter"
  out_states="$OUT_BASE/states/$cell"
  out_runs="$OUT_BASE/runs/$cell"
  log="$OUT_BASE/logs/$cell.log"

  if [ ! -d "$adapter" ]; then
    echo "[$(date +%H:%M:%S)] $cell: SKIP (no adapter at $adapter)" | tee -a "$log"
    continue
  fi
  if [ -f "$out_runs/results.csv" ]; then
    echo "[$(date +%H:%M:%S)] $cell: SKIP (already done)"
    continue
  fi

  mkdir -p "$out_states" "$out_runs"
  echo "[$(date +%H:%M:%S)] $cell: extracting RS3 states (advbench, harmbench × {malicious, cleaned, paraphrased})" | tee -a "$log"
  if [ ! -f "$out_states/.rs3_done" ]; then
    cd "$WPF"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" scripts/extract_states.py \
      --model_path "$base" --adapter_path "$adapter" \
      --data_root "$DATA_RS3" --views malicious cleaned paraphrased \
      --datasets advbench harmbench \
      --layer_idx "$PROBE_LAYER" --out_dir "$out_states" \
      >>"$log" 2>&1 \
      && touch "$out_states/.rs3_done" \
      || { echo "[$(date +%H:%M:%S)] $cell: extract RS3 FAILED" | tee -a "$log"; continue; }
  fi
  echo "[$(date +%H:%M:%S)] $cell: extracting RS1 states (alpaca benign, dolly benign)" | tee -a "$log"
  if [ ! -f "$out_states/.rs1_done" ]; then
    cd "$WPF"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" scripts/extract_states.py \
      --model_path "$base" --adapter_path "$adapter" \
      --data_root "$DATA_RS1" --views benign \
      --datasets alpaca dolly \
      --layer_idx "$PROBE_LAYER" --out_dir "$out_states" \
      >>"$log" 2>&1 \
      && touch "$out_states/.rs1_done" \
      || { echo "[$(date +%H:%M:%S)] $cell: extract RS1 FAILED" | tee -a "$log"; continue; }
  fi

  echo "[$(date +%H:%M:%S)] $cell: running probes (svm_raw, mlp_no_jepa)" | tee -a "$log"
  cd "$WPF"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$WPF" "$PYTHON" scripts/compare_probes.py \
    --config "$PROBE_CFG" --states_dir "$out_states" --out_dir "$out_runs" \
    --probes svm_raw mlp_no_jepa --with_analyses \
    >>"$log" 2>&1 \
    || { echo "[$(date +%H:%M:%S)] $cell: probes FAILED" | tee -a "$log"; continue; }

  echo "[$(date +%H:%M:%S)] $cell: DONE -> $out_runs/results.csv"
done

echo "[$(date +%H:%M:%S)] all cells processed. summary:"
for entry in "${CELLS[@]}"; do
  cell="${entry%|*}"
  res="$OUT_BASE/runs/$cell/results.csv"
  if [ -f "$res" ]; then
    echo "  $cell: $(wc -l < "$res") rows"
  else
    echo "  $cell: MISSING"
  fi
done
