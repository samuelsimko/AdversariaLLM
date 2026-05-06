#!/usr/bin/env bash
# Run the probe-train-size sweep on layer-25 states, twice per cell when a
# predictor exists: once raw, once projected through jepa_predictor.pt.
# Wang setup only — RS1+RS3 slices, low train sizes.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
if [[ -f "$REPO/.env" ]]; then
  set +u; source "$REPO/.env"; set -u
fi

PYBIN=/workspace/AdversariaLLM/venv/bin/python
STATES_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states_l25
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
SWEEP_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/sweep_l25
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/sweep_l25_logs
mkdir -p "$SWEEP_ROOT" "$LOG_DIR"

CELLS=(
  l_cb_pra l_cb_no_pra l_triplet_pra l_triplet_no_pra
  q_cb_pra q_cb_no_pra q_triplet_pra q_triplet_no_pra
  l_base q_base
)

GPU_COUNT=${GPU_COUNT:-2}
TRAIN_SIZES=${TRAIN_SIZES:-"10 25 50 100"}
SEEDS=${SEEDS:-"42 123 777 1234 9999"}
PROBES=${PROBES:-"svm_raw mlp_no_jepa jepa"}

run_sweep() {
  local cell="$1" tag="$2" gpu="$3" extra_args=("${@:4}")
  local out="$SWEEP_ROOT/${cell}_${tag}.csv"
  local log="$LOG_DIR/${cell}_${tag}.log"
  echo "[gpu$gpu] sweep cell=$cell tag=$tag" | tee -a "$log"
  (env CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" "$REPO/scripts/probe_train_size/sweep.py" \
      --cell "$cell" \
      --states_dir "$STATES_ROOT/$cell" \
      --out_csv "$out" \
      --train_sizes $TRAIN_SIZES \
      --seeds $SEEDS \
      --probes $PROBES \
      "${extra_args[@]}" \
      >>"$log" 2>&1 && echo "[gpu$gpu] $cell/$tag DONE" >>"$log")
}

declare -A GPU_PID
schedule() {
  local cell="$1" tag="$2"; shift 2
  while :; do
    for gpu in $(seq 0 $((GPU_COUNT-1))); do
      if [[ -z "${GPU_PID[$gpu]:-}" ]] || ! kill -0 "${GPU_PID[$gpu]}" 2>/dev/null; then
        run_sweep "$cell" "$tag" "$gpu" "$@" &
        GPU_PID[$gpu]=$!
        return
      fi
    done
    sleep 2
  done
}

for cell in "${CELLS[@]}"; do
  schedule "$cell" "raw"
  pred="$DEF_ROOT/$cell/jepa_predictor.pt"
  if [[ -f "$pred" ]]; then
    schedule "$cell" "pred" --predictor_path "$pred"
  fi
done

wait
echo "All layer-25 sweeps complete."
