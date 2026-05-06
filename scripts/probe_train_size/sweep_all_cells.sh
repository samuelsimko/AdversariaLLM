#!/usr/bin/env bash
# Run probe_train_size/sweep.py across all 10 cells, parallel across 2 GPUs.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
if [[ -f "$REPO/.env" ]]; then
  set +u; source "$REPO/.env"; set -u
fi

PYBIN=/workspace/AdversariaLLM/venv/bin/python
STATES_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states
SWEEP_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/sweep
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/sweep_logs
mkdir -p "$SWEEP_ROOT" "$LOG_DIR"

CELLS=(
  l_cb_pra l_cb_no_pra l_triplet_pra l_triplet_no_pra
  q_cb_pra q_cb_no_pra q_triplet_pra q_triplet_no_pra
  l_base q_base
)

GPU_COUNT=${GPU_COUNT:-2}
TRAIN_SIZES=${TRAIN_SIZES:-"25 50 100 200 400"}
SEEDS=${SEEDS:-"42 123 777"}
PROBES=${PROBES:-"svm_raw mlp_no_jepa jepa"}

declare -A GPU_PID
for i in "${!CELLS[@]}"; do
  cell="${CELLS[$i]}"
  gpu=$(( i % GPU_COUNT ))
  while [[ -n "${GPU_PID[$gpu]:-}" ]] && kill -0 "${GPU_PID[$gpu]}" 2>/dev/null; do
    sleep 2
  done
  log="$LOG_DIR/${cell}.log"
  out="$SWEEP_ROOT/${cell}.csv"
  echo "[gpu$gpu] sweep cell=$cell" | tee -a "$log"
  (env CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" "$REPO/scripts/probe_train_size/sweep.py" \
      --cell "$cell" \
      --states_dir "$STATES_ROOT/$cell" \
      --out_csv "$out" \
      --train_sizes $TRAIN_SIZES \
      --seeds $SEEDS \
      --probes $PROBES \
      >>"$log" 2>&1 && echo "[gpu$gpu] $cell DONE" >>"$log") &
  GPU_PID[$gpu]=$!
done

wait
echo "All sweeps complete."
