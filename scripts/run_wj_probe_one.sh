#!/usr/bin/env bash
# Run probe extraction + comparison on a single WJ-ablation cell on a single GPU.
# Mirrors the relevant steps from run_why_probe_fails.sh but takes the cell + base_model
# explicitly so we don't need to glob a single CKPT_BASE.
#
# Usage:  bash scripts/run_wj_probe_one.sh <gpu> <cell_dir> <base_model>
#   <cell_dir>  = absolute path to the cell (must have lora_adapter/ + manifest.json + READY)
#   <base_model> = HF model id used as the base
#   <name>      = name to use under OUT_BASE (default: basename of cell_dir)

set -uo pipefail

GPU="${1:?gpu}"
CELL_DIR="${2:?cell_dir}"
BASE_MODEL="${3:?base_model}"
NAME="${4:-$(basename "$CELL_DIR")}"

WPF=/root/AdversariaLLM_full_2026-04-29/Why-Probe-Fails
OUT_BASE=/root/AdversariaLLM/runs/why_probe_fails_wj
PROBE_LAYER=-1
PROBE_CFG=configs/rs3_advbench_jepa.json
DATA_PROBE=data/RS3
DATA_BENIGN=data/RS1

source .env

OUT_STATES="$OUT_BASE/states/$NAME"
OUT_RUNS="$OUT_BASE/runs/$NAME"
OUT_STATES_PRED="$OUT_BASE/states_predictor/$NAME"
OUT_RUNS_PRED="$OUT_BASE/runs_predictor/$NAME"
LOG="$OUT_BASE/logs/${NAME}.log"
mkdir -p "$OUT_STATES" "$OUT_RUNS" "$OUT_BASE/logs"

ADAPTER="$CELL_DIR/lora_adapter"
[[ -d "$ADAPTER" ]] || { echo "no adapter at $ADAPTER" >&2; exit 1; }

{
  echo "[$(date +%H:%M:%S)] [$NAME] extracting RS3 advbench+harmbench states on GPU $GPU"
  if ! [[ -f "$OUT_STATES/.extract_done_rs3_min" ]]; then
    cd "$WPF"
    CUDA_VISIBLE_DEVICES=$GPU /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
      --model_path "$BASE_MODEL" --adapter_path "$ADAPTER" \
      --data_root "$DATA_PROBE" --views malicious cleaned paraphrased \
      --datasets advbench harmbench \
      --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES" \
      && touch "$OUT_STATES/.extract_done_rs3_min"
  fi
  if ! [[ -f "$OUT_STATES/.extract_done_rs1_min" ]]; then
    cd "$WPF"
    CUDA_VISIBLE_DEVICES=$GPU /root/AdversariaLLM/venv/bin/python scripts/extract_states.py \
      --model_path "$BASE_MODEL" --adapter_path "$ADAPTER" \
      --data_root "$DATA_BENIGN" --views benign \
      --datasets alpaca \
      --layer_idx "$PROBE_LAYER" --out_dir "$OUT_STATES" \
      && touch "$OUT_STATES/.extract_done_rs1_min"
  fi
  echo "[$(date +%H:%M:%S)] [$NAME] running raw-state probe"
  if ! [[ -f "$OUT_RUNS/results.csv" ]]; then
    cd "$WPF"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$WPF" /root/AdversariaLLM/venv/bin/python scripts/compare_probes.py \
      --config "$PROBE_CFG" --states_dir "$OUT_STATES" \
      --out_dir "$OUT_RUNS" --probes svm_raw mlp_no_jepa --with_analyses
  fi
  if [[ -f "$CELL_DIR/jepa_predictor.pt" && -f "$CELL_DIR/manifest.json" ]]; then
    mkdir -p "$OUT_STATES_PRED" "$OUT_RUNS_PRED"
    if ! [[ -f "$OUT_STATES_PRED/.predictor_done" ]]; then
      cd "$WPF"
      CUDA_VISIBLE_DEVICES=$GPU /root/AdversariaLLM/venv/bin/python scripts/apply_pra_predictor.py \
        --run_dir "$CELL_DIR" --states_dir "$OUT_STATES" --out_dir "$OUT_STATES_PRED" \
        && touch "$OUT_STATES_PRED/.predictor_done"
    fi
    if ! [[ -f "$OUT_RUNS_PRED/results.csv" ]]; then
      cd "$WPF"
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$WPF" /root/AdversariaLLM/venv/bin/python scripts/compare_probes.py \
        --config "$PROBE_CFG" --states_dir "$OUT_STATES_PRED" \
        --out_dir "$OUT_RUNS_PRED" --probes svm_raw mlp_no_jepa --with_analyses
    fi
  fi
  echo "[$(date +%H:%M:%S)] [$NAME] done"
} >>"$LOG" 2>&1
