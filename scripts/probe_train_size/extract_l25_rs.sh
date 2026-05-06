#!/usr/bin/env bash
# Extract layer-25 last-token hidden states for the original Wang setup
# (RS1 alpaca/dolly + RS3 advbench/harmbench) for every cell. Output goes
# to a dedicated states_l25/ tree so the existing layer -1 sweep is preserved.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
if [[ -f "$REPO/.env" ]]; then
  set +u; source "$REPO/.env"; set -u
fi
WPF=/workspace/AdversariaLLM/Why-Probe-Fails
PYBIN=/workspace/AdversariaLLM/venv/bin/python
STATES_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states_l25
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/extract_l25_logs
mkdir -p "$LOG_DIR"

LAYER_IDX=${LAYER_IDX:-25}

CELLS=(
  "l_cb_pra|meta-llama/Meta-Llama-3-8B-Instruct|$DEF_ROOT/l_cb_pra/lora_adapter"
  "l_cb_no_pra|meta-llama/Meta-Llama-3-8B-Instruct|$DEF_ROOT/l_cb_no_pra/lora_adapter"
  "l_triplet_pra|meta-llama/Meta-Llama-3-8B-Instruct|$DEF_ROOT/l_triplet_pra/lora_adapter"
  "l_triplet_no_pra|meta-llama/Meta-Llama-3-8B-Instruct|$DEF_ROOT/l_triplet_no_pra/lora_adapter"
  "q_cb_pra|Qwen/Qwen3-8B|$DEF_ROOT/q_cb_pra/lora_adapter"
  "q_cb_no_pra|Qwen/Qwen3-8B|$DEF_ROOT/q_cb_no_pra/lora_adapter"
  "q_triplet_pra|Qwen/Qwen3-8B|$DEF_ROOT/q_triplet_pra/lora_adapter"
  "q_triplet_no_pra|Qwen/Qwen3-8B|$DEF_ROOT/q_triplet_no_pra/lora_adapter"
  "l_base|meta-llama/Meta-Llama-3-8B-Instruct|"
  "q_base|Qwen/Qwen3-8B|"
)

DATA_ROOTS=(
  "data/RS1|benign"
  "data/RS3|malicious cleaned paraphrased"
)
RS1_DATASETS=(alpaca dolly)
RS3_DATASETS=(advbench harmbench)

run_one_cell() {
  local cell="$1" model="$2" adapter="$3" gpu="$4"
  local out="$STATES_ROOT/$cell"
  mkdir -p "$out"
  local log="$LOG_DIR/${cell}.log"
  echo "[gpu$gpu] cell=$cell model=$model adapter=${adapter:-none} layer=$LAYER_IDX" | tee -a "$log"
  for entry in "${DATA_ROOTS[@]}"; do
    local data_path="${entry%|*}"
    local views="${entry#*|}"
    local cmd=(env CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" "$WPF/scripts/extract_states.py"
      --model_path "$model"
      --data_root "$data_path"
      --views $views
      --layer_idx "$LAYER_IDX"
      --out_dir "$out")
    if [[ "$data_path" == "data/RS1" ]]; then
      cmd+=(--datasets "${RS1_DATASETS[@]}")
    elif [[ "$data_path" == "data/RS3" ]]; then
      cmd+=(--datasets "${RS3_DATASETS[@]}")
    fi
    if [[ -n "$adapter" ]]; then
      cmd+=(--adapter_path "$adapter")
    fi
    echo "[gpu$gpu] $cell <- $data_path ($views)" | tee -a "$log"
    (cd "$WPF" && "${cmd[@]}") >>"$log" 2>&1
  done
  echo "[gpu$gpu] cell=$cell DONE" | tee -a "$log"
}

GPU_COUNT=${GPU_COUNT:-2}
# H200s have 144GB; an 8B model loads at ~16GB → fit ~5 cells per GPU.
# Fire all cells in parallel, pinned round-robin to GPUs.
for i in "${!CELLS[@]}"; do
  IFS='|' read -r cell model adapter <<<"${CELLS[$i]}"
  gpu=$(( i % GPU_COUNT ))
  run_one_cell "$cell" "$model" "$adapter" "$gpu" &
done
wait
echo "All layer-$LAYER_IDX extractions complete."
