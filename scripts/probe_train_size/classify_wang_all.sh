#!/usr/bin/env bash
# Run Wang-faithful classifier per (cell × layer × {raw, pred}). All CPU work
# (predictor projection runs on GPU when available). ~seconds per call.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
PYBIN=/workspace/AdversariaLLM/venv/bin/python
WANG_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states_wang
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
OUT_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/classify_wang
mkdir -p "$OUT_ROOT"

CELLS=(
  l_cb_pra l_cb_no_pra l_triplet_pra l_triplet_no_pra
  q_cb_pra q_cb_no_pra q_triplet_pra q_triplet_no_pra
  l_base q_base
)

LAYERS=(-1 25)

# Wang's chosen ID — beaver malicious + alpaca/dolly benign.
ID_MALICIOUS="beaver"
ID_BENIGN="alpaca dolly"

# Predictor architecture (matches our local cells; paper cells use 256).
PRED_LAYERS=${PRED_LAYERS:-2}
PRED_BOTTLENECK=${PRED_BOTTLENECK:-512}

for layer in "${LAYERS[@]}"; do
  for cell in "${CELLS[@]}"; do
    states="$WANG_ROOT/L${layer}/$cell"
    if [[ ! -d "$states" ]]; then
      echo "skip $cell L$layer (no states dir)"
      continue
    fi

    # Raw (Wang-faithful, no projection)
    "$PYBIN" "$REPO/scripts/probe_train_size/classify_wang.py" \
      --cell "$cell" \
      --layer "$layer" \
      --states_dir "$states" \
      --out_csv "$OUT_ROOT/${cell}_L${layer}_raw.csv" \
      --id_malicious $ID_MALICIOUS \
      --id_benign $ID_BENIGN

    # Predictor-projected (only if jepa_predictor.pt exists)
    pred="$DEF_ROOT/$cell/jepa_predictor.pt"
    if [[ -f "$pred" ]]; then
      "$PYBIN" "$REPO/scripts/probe_train_size/classify_wang.py" \
        --cell "$cell" \
        --layer "$layer" \
        --states_dir "$states" \
        --out_csv "$OUT_ROOT/${cell}_L${layer}_pred.csv" \
        --id_malicious $ID_MALICIOUS \
        --id_benign $ID_BENIGN \
        --predictor_path "$pred" \
        --predictor_layers "$PRED_LAYERS" \
        --predictor_bottleneck_dim "$PRED_BOTTLENECK"
    fi
  done
done

echo "All Wang-faithful classifications complete (raw + pred)."
