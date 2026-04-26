#!/usr/bin/env bash
set -euo pipefail

source venv/bin/activate
source ../honeypot_llm_defense/.env

CONFIG="${CONFIG:-experiments/configs/jepa_llama3_overnight.yaml}"
BACKEND="${BACKEND:-local}"
EXP_ROOT="${EXP_ROOT:-runs/experiments/jepa_llama3_overnight}"

# Guardrail / OOD AUROC settings.
RUN_GUARDRAIL_EVAL="${RUN_GUARDRAIL_EVAL:-1}"
WAIT_FOR_TRAINING="${WAIT_FOR_TRAINING:-1}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-86400}"

DEFENSES=(
  jepa_ce_no_jepa_llama3
  jepa_ce_bottleneck64_llama3
  jepa_ce_bottleneck256_llama3
  jepa_ce_identity_llama3
)

echo "[overnight] config=${CONFIG}"
echo "[overnight] backend=${BACKEND}"
echo "[overnight] output=${EXP_ROOT}"
echo "[overnight] attacks=direct_full,prefilling_full,soft_prompt_50,bon_50,inpainting_50"
echo "[overnight] guardrail_eval=${RUN_GUARDRAIL_EVAL}"

python experiments/run_experiment.py \
  --config "${CONFIG}" \
  --backend "${BACKEND}"

if [[ "${RUN_GUARDRAIL_EVAL}" != "1" ]]; then
  echo "[overnight] skipping guardrail OOD AUROC eval"
  exit 0
fi

if [[ "${WAIT_FOR_TRAINING}" == "1" ]]; then
  echo "[overnight] waiting for training READY files before guardrail eval"
  start_ts="$(date +%s)"
  while true; do
    missing=()
    for defense in "${DEFENSES[@]}"; do
      if [[ ! -f "${EXP_ROOT}/${defense}/READY" ]]; then
        missing+=("${defense}")
      fi
    done

    if [[ "${#missing[@]}" -eq 0 ]]; then
      echo "[overnight] all training runs are READY"
      break
    fi

    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"
    if (( elapsed > MAX_WAIT_SECONDS )); then
      echo "[overnight] timed out waiting for training READY files after ${elapsed}s"
      echo "[overnight] still missing: ${missing[*]}"
      exit 1
    fi

    echo "[overnight] missing training READY: ${missing[*]}"
    sleep "${POLL_SECONDS}"
  done
fi

for defense in "${DEFENSES[@]}"; do
  run_dir="${EXP_ROOT}/${defense}"
  if [[ ! -f "${run_dir}/manifest.json" ]]; then
    echo "[overnight] skipping guardrail eval for ${defense}: missing ${run_dir}/manifest.json"
    continue
  fi

  echo "[overnight] guardrail OOD AUROC eval: ${defense}"
  python scripts/evaluate_jepa_guardrail.py \
    --run_dir "${run_dir}" \
    --output_json "${run_dir}/guardrail_wildjailbreak_eval.json" \
    --centroid_per_category 64 \
    --eval_per_category 64 \
    --benign_centroid_samples 512 \
    --benign_eval_samples 1024 \
    --jailbreak_eval_samples 1024 \
    --batch_size 8 \
    --max_length 256 \
    --target_fpr 0.01 \
    --wildjailbreak_dataset allenai/wildjailbreak \
    --wildjailbreak_config eval \
    --wildjailbreak_split train \
    --wildjailbreak_limit 2210
done

echo "[overnight] done"
