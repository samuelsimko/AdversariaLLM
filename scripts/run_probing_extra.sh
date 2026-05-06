#!/usr/bin/env bash
# Extension to run_probing_headline_backup.sh: probe each cell on the
# multilingual jailbreak (wow2000) and JEPA-attack-family (memo-ozdincer/jepadata)
# slices, then aggregate AUCs into a per-cell CSV.
#
# Prereqs:
#   1. Build the per-slice CSVs under WPF/data/multi and WPF/data/jepa:
#         python scripts/build_extra_probe_data.py
#      (idempotent; reads HF_TOKEN from .env for wow2000 gated dataset).
#   2. Cells must already have their RS3+RS1 states extracted by
#      scripts/run_probing_headline_backup.sh (we reuse the AdvBench-en
#      malicious states as the SVM's ID positives and the Alpaca-en benign
#      states as negatives — same anchor as the headline run).
#
# Usage:
#   bash scripts/run_probing_extra.sh [gpu_id]   # default gpu_id=0
#
# Outputs per cell (under runs/why_probe_fails_headline_backup/):
#   states/<cell>/multi_malicious_<lang>.npy        per-language states
#   states/<cell>/jepa_malicious_<family>.npy        per-family states
#   states/<cell>/jepa_benign_<family>.npy
#   runs_extra/<cell>/multi.csv                      AUC per language
#   runs_extra/<cell>/jepa.csv                       AUC per attack family

set -uo pipefail
GPU="${1:-0}"
WPF=/root/AdversariaLLM/Why-Probe-Fails
PYTHON=/root/AdversariaLLM/.venv/bin/python
EXP_ROOT=/root/AdversariaLLM/runs/experiments/headline_backup_rerun
OUT_BASE=/root/AdversariaLLM/runs/why_probe_fails_headline_backup
PROBE_LAYER=25

if [ -f /root/AdversariaLLM/.env ]; then
  set -o allexport; source /root/AdversariaLLM/.env; set +o allexport
fi

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

LANGS=(en zh-cn es ja ar fy cs id pt)
JEPA_FAMS=(encoding prefilling persona inpainting)

for entry in "${CELLS[@]}"; do
  cell="${entry%|*}"; base="${entry#*|}"
  cell_dir="$EXP_ROOT/$cell"
  adapter="$cell_dir/lora_adapter"
  out_states="$OUT_BASE/states/$cell"
  out_runs="$OUT_BASE/runs_extra/$cell"
  log="$OUT_BASE/logs/${cell}_extra.log"
  mkdir -p "$out_states" "$out_runs"

  echo "[$(date +%H:%M:%S)] $cell: extract multi (${#LANGS[@]} langs)" | tee -a "$log"
  cd "$WPF"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/extract_states.py \
    --model_path "$base" --adapter_path "$adapter" \
    --data_root "$WPF/data/multi" --views malicious \
    --datasets "${LANGS[@]}" \
    --layer_idx $PROBE_LAYER --out_dir "$out_states" >>"$log" 2>&1

  echo "[$(date +%H:%M:%S)] $cell: extract jepa (${#JEPA_FAMS[@]} families × 2 sides)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/extract_states.py \
    --model_path "$base" --adapter_path "$adapter" \
    --data_root "$WPF/data/jepa" --views malicious benign \
    --datasets "${JEPA_FAMS[@]}" \
    --layer_idx $PROBE_LAYER --out_dir "$out_states" >>"$log" 2>&1

  echo "[$(date +%H:%M:%S)] $cell: SVM unified-anchor eval" | tee -a "$log"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON /root/AdversariaLLM/scripts/eval_extra_probes.py \
    --states_dir "$out_states" --out_dir "$out_runs" \
    --langs "${LANGS[@]}" --jepa_fams "${JEPA_FAMS[@]}" >>"$log" 2>&1

  echo "[$(date +%H:%M:%S)] $cell: DONE" | tee -a "$log"
done

echo "[$(date +%H:%M:%S)] all cells: $OUT_BASE/runs_extra/*/{multi,jepa}.csv"
