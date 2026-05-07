#!/usr/bin/env bash
# End-to-end Wang-faithful probe pipeline for one or more cells.
#
# Inputs: cell directories under runs/experiments/headline_backup_rerun/<cell>/
#         (each containing lora_adapter/ and optional jepa_predictor.pt).
# Outputs: states_wang/L{-1,25}/<cell>/*.npy hidden states,
#          classify_wang/<cell>_L<layer>_{raw,pred}.csv probe results,
#          assets/figures/probe_wang_faithful/{WANG_FAITHFUL_RESULTS.md,
#                                              wide_table_wang.csv,
#                                              all_results_wang.csv}.
#
# Usage:
#   bash scripts/probe_train_size/run_wang_pipeline.sh                 # run on all 10 default cells
#   bash scripts/probe_train_size/run_wang_pipeline.sh l_cb_pra        # one cell
#   bash scripts/probe_train_size/run_wang_pipeline.sh l_cb_pra l_cb_no_pra  # subset
#
# Optional env:
#   GPU_COUNT (default 2), PER_GPU (default 5)
#   LAYERS    (default "-1 25")
#   ID_MALICIOUS (default "beaver"), ID_BENIGN (default "alpaca dolly")
#   SKIP_EXTRACT=1  to assume states are already extracted
#   SKIP_CLASSIFY=1 to only rebuild the table from existing CSVs
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
if [[ -f "$REPO/.env" ]]; then
  set +u; source "$REPO/.env"; set -u
fi

PYBIN=${PYBIN:-/workspace/AdversariaLLM/venv/bin/python}
WPF=${WPF:-/workspace/AdversariaLLM/Why-Probe-Fails}
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
WANG_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states_wang
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/wang_pipeline_logs
CLASSIFY_OUT=$REPO/runs/experiments/headline_backup_rerun_probing/classify_wang
mkdir -p "$WANG_ROOT" "$LOG_DIR" "$CLASSIFY_OUT"

GPU_COUNT=${GPU_COUNT:-2}
PER_GPU=${PER_GPU:-5}
LAYERS=${LAYERS:-"-1 25"}
ID_MALICIOUS=${ID_MALICIOUS:-"beaver"}
ID_BENIGN=${ID_BENIGN:-"alpaca dolly"}
PRED_LAYERS=${PRED_LAYERS:-2}
PRED_BOTTLENECK=${PRED_BOTTLENECK:-512}

# Default cell list (10 standard cells in headline_backup_rerun)
DEFAULT_CELLS=(
  l_cb_pra l_cb_no_pra l_triplet_pra l_triplet_no_pra
  q_cb_pra q_cb_no_pra q_triplet_pra q_triplet_no_pra
  l_base q_base
)

if [[ $# -eq 0 ]]; then
  CELLS=("${DEFAULT_CELLS[@]}")
else
  CELLS=("$@")
fi

# Resolve each cell to (model_id, adapter_path)
resolve_cell() {
  local cell="$1"
  case "$cell" in
    l_*) model="meta-llama/Meta-Llama-3-8B-Instruct" ;;
    q_*) model="Qwen/Qwen3-8B" ;;
    *) echo "unknown cell prefix: $cell" >&2; exit 1 ;;
  esac
  if [[ "$cell" == *_base ]]; then
    adapter=""
  else
    adapter="$DEF_ROOT/$cell/lora_adapter"
    if [[ ! -d "$adapter" ]]; then
      echo "missing lora_adapter at $adapter" >&2; exit 1
    fi
  fi
  echo "$model|$adapter"
}

DATA_ROOTS=(
  "data/RS1|benign"
  "data/RS1|malicious"
  "data/RS2|malicious cleaned"
  "data/RS3|malicious cleaned paraphrased"
  "data/JEPA|benign malicious"
  "data/MULTI|malicious"
)

extract_one_cell_layer() {
  local cell="$1" model="$2" adapter="$3" gpu="$4" layer="$5"
  local out="$WANG_ROOT/L${layer}/$cell"
  mkdir -p "$out"
  local log="$LOG_DIR/extract_${cell}_L${layer}.log"
  echo "[gpu$gpu] cell=$cell layer=$layer" | tee -a "$log"
  for entry in "${DATA_ROOTS[@]}"; do
    local data_path="${entry%|*}"
    local views="${entry#*|}"
    local cmd=(env CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" "$WPF/scripts/extract_states.py"
      --model_path "$model"
      --data_root "$data_path"
      --views $views
      --layer_idx "$layer"
      --out_dir "$out")
    if [[ -n "$adapter" ]]; then
      cmd+=(--adapter_path "$adapter")
    fi
    echo "[gpu$gpu] $cell L${layer} <- $data_path ($views)" | tee -a "$log"
    (cd "$WPF" && "${cmd[@]}") >>"$log" 2>&1
  done
  echo "[gpu$gpu] cell=$cell L${layer} EXTRACT DONE" | tee -a "$log"
}

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  echo "=== EXTRACT ($((${#CELLS[@]} * $(echo $LAYERS | wc -w))) jobs) ==="
  declare -A PID_GPU
  GPU_PIDS_LIST=()
  i=0
  for layer in $LAYERS; do
    for cell in "${CELLS[@]}"; do
      info=$(resolve_cell "$cell")
      model="${info%|*}"
      adapter="${info#*|}"
      gpu=$(( i % GPU_COUNT ))
      while :; do
        load=0
        for pid in ${GPU_PIDS_LIST[@]:-}; do
          if kill -0 "$pid" 2>/dev/null; then
            if [[ "${PID_GPU[$pid]:-}" == "$gpu" ]]; then load=$((load+1)); fi
          fi
        done
        if (( load < PER_GPU )); then break; fi
        sleep 2
      done
      extract_one_cell_layer "$cell" "$model" "$adapter" "$gpu" "$layer" &
      pid=$!
      PID_GPU[$pid]=$gpu
      GPU_PIDS_LIST+=($pid)
      i=$((i+1))
    done
  done
  wait
  echo "=== EXTRACT DONE ==="
fi

if [[ "${SKIP_CLASSIFY:-0}" != "1" ]]; then
  echo "=== CLASSIFY ==="
  for layer in $LAYERS; do
    for cell in "${CELLS[@]}"; do
      states="$WANG_ROOT/L${layer}/$cell"
      if [[ ! -d "$states" ]]; then
        echo "skip $cell L$layer (no states)"
        continue
      fi
      "$PYBIN" "$REPO/scripts/probe_train_size/classify_wang.py" \
        --cell "$cell" --layer "$layer" --states_dir "$states" \
        --out_csv "$CLASSIFY_OUT/${cell}_L${layer}_raw.csv" \
        --id_malicious $ID_MALICIOUS --id_benign $ID_BENIGN
      pred="$DEF_ROOT/$cell/jepa_predictor.pt"
      if [[ -f "$pred" ]]; then
        "$PYBIN" "$REPO/scripts/probe_train_size/classify_wang.py" \
          --cell "$cell" --layer "$layer" --states_dir "$states" \
          --out_csv "$CLASSIFY_OUT/${cell}_L${layer}_pred.csv" \
          --id_malicious $ID_MALICIOUS --id_benign $ID_BENIGN \
          --predictor_path "$pred" \
          --predictor_layers "$PRED_LAYERS" \
          --predictor_bottleneck_dim "$PRED_BOTTLENECK"
      fi
    done
  done
  echo "=== CLASSIFY DONE ==="
fi

echo "=== TABLE ==="
"$PYBIN" "$REPO/scripts/probe_train_size/build_wang_table.py"
echo "=== ALL DONE ==="
echo "Outputs: assets/figures/probe_wang_faithful/"
