#!/usr/bin/env bash
# Wang-faithful extraction: layer -1 AND layer 25 hidden states for every cell,
# across the full Wang dataset (RS1 benign+malicious, RS2 malicious+cleaned,
# RS3 all 3 views). All 10 cells fired in parallel, pinned round-robin to GPUs.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO=$(pwd)
if [[ -f "$REPO/.env" ]]; then
  set +u; source "$REPO/.env"; set -u
fi

WPF=/workspace/AdversariaLLM/Why-Probe-Fails
PYBIN=/workspace/AdversariaLLM/venv/bin/python
DEF_ROOT=$REPO/runs/experiments/headline_backup_rerun
WANG_ROOT=$REPO/runs/experiments/headline_backup_rerun_probing/states_wang
LOG_DIR=$REPO/runs/experiments/headline_backup_rerun_probing/extract_wang_logs
mkdir -p "$LOG_DIR"

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

# Wang's full dataset + augmented OOD families (multilingual + JEPA attack styles).
DATA_ROOTS=(
  "data/RS1|benign"            # alpaca, dolly, nq, simpleqa
  "data/RS1|malicious"         # beaver, maliciousinstruct
  "data/RS2|malicious cleaned" # advbench, harmbench, jailbreakbench, maliciousinstruct
  "data/RS3|malicious cleaned paraphrased"  # advbench, harmbench
  "data/JEPA|benign malicious" # encoding, inpainting, persona, prefilling (paired)
  "data/MULTI|malicious"       # ar, cs, en, es, fy, id, ja, pt, zh-cn (multilingual)
)

LAYERS=(-1 25)

run_one_cell_layer() {
  local cell="$1" model="$2" adapter="$3" gpu="$4" layer="$5"
  local out="$WANG_ROOT/L${layer}/$cell"
  mkdir -p "$out"
  local log="$LOG_DIR/${cell}_L${layer}.log"
  echo "[gpu$gpu] cell=$cell layer=$layer model=$model" | tee -a "$log"
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
  echo "[gpu$gpu] cell=$cell L${layer} DONE" | tee -a "$log"
}

GPU_COUNT=${GPU_COUNT:-2}
PER_GPU=${PER_GPU:-5}  # H200 144GB / 16GB-per-8B = 9 max safe; 5 leaves headroom.
# 10 cells × 2 layers = 20 jobs. Throttle to PER_GPU concurrent per GPU.
declare -A GPU_LOAD
i=0
for layer in "${LAYERS[@]}"; do
  for entry in "${CELLS[@]}"; do
    IFS='|' read -r cell model adapter <<<"$entry"
    gpu=$(( i % GPU_COUNT ))
    # Block until this GPU has a free slot.
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
    run_one_cell_layer "$cell" "$model" "$adapter" "$gpu" "$layer" &
    pid=$!
    PID_GPU[$pid]=$gpu
    GPU_PIDS_LIST+=($pid)
    i=$((i+1))
  done
done
wait
echo "All Wang-faithful extractions complete."
