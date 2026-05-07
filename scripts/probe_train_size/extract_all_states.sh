#!/usr/bin/env bash
# Extract layer -1 hidden states for every (cell, data_root) we need.
# extract_states.py auto-skips files that already exist, so this is safe to re-run.
#
# Cells = 8 defenses + 2 bases. Data roots = RS1 (benign), RS3 (paired),
# JEPA (paired benign+malicious), MULTI (malicious-only).
#
# Cells are assigned round-robin across GPUs; each GPU process loads its model
# once per cell and walks through all data roots before moving on.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
# Load HF_TOKEN (gated repo access for Llama-3) and other secrets.
# .env contains $PYTHONPATH expansion which trips set -u; bracket the source.
if [[ -f "$REPO/.env" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$REPO/.env"
  set -u
fi
WPF=/workspace/AdversariaLLM/Why-Probe-Fails
PYBIN=/workspace/AdversariaLLM/venv/bin/python
STATES_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/extract_logs
mkdir -p "$LOG_DIR"

# cell|model_id|adapter_path
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

# data_root_path|views (space-separated)
DATA_ROOTS=(
  "data/RS1|benign"
  "data/RS3|malicious cleaned paraphrased"
  "data/JEPA|benign malicious"
  "data/MULTI|malicious"
)

# RS1 datasets used by existing config + new ones; restrict to the ones we want.
# For consistency with the existing run, only extract alpaca + dolly from RS1.
RS1_DATASETS=(alpaca dolly)
RS3_DATASETS=(advbench harmbench)

run_one_cell() {
  local cell="$1" model="$2" adapter="$3" gpu="$4"
  local out="$STATES_ROOT/$cell"
  mkdir -p "$out"
  local log="$LOG_DIR/${cell}.log"
  echo "[gpu$gpu] cell=$cell model=$model adapter=${adapter:-none}" | tee -a "$log"
  for entry in "${DATA_ROOTS[@]}"; do
    local data_path="${entry%|*}"
    local views="${entry#*|}"
    local cmd=(env CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" "$WPF/scripts/extract_states.py"
      --model_path "$model"
      --data_root "$data_path"
      --views $views
      --layer_idx -1
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
PIDS=()
for i in "${!CELLS[@]}"; do
  IFS='|' read -r cell model adapter <<<"${CELLS[$i]}"
  gpu=$(( i % GPU_COUNT ))
  # Wait if this GPU is already busy.
  while [[ -n "${GPU_PID[$gpu]:-}" ]] && kill -0 "${GPU_PID[$gpu]}" 2>/dev/null; do
    sleep 2
  done
  run_one_cell "$cell" "$model" "$adapter" "$gpu" &
  GPU_PID[$gpu]=$!
  PIDS+=("$!")
done
echo "Waiting for ${#PIDS[@]} cell processes..."
wait
echo "All extractions complete."
