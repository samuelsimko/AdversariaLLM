#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   scripts/run_reverse_model_pipeline.sh
#   SKIP_TRAIN=1 ADAPTER_DIR=reverse_model/lora scripts/run_reverse_model_pipeline.sh
#   SKIP_TRAIN=1 ADAPTER_DIR=reverse_model/checkpoint-100 scripts/run_reverse_model_pipeline.sh

PYTHON_BIN="${PYTHON_BIN:-/workspace/AdversariaLLM/venv/bin/python}"
CB_PATH="${CB_PATH:-data/circuit_breakers_train.json}"
REVERSE_DIR="${REVERSE_DIR:-reverse_model}"
ADAPTER_DIR="${ADAPTER_DIR:-${REVERSE_DIR}/lora}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"

CB_LIMIT="${CB_LIMIT:-5000}"
ULTRA_SAMPLES="${ULTRA_SAMPLES:-20000}"
HOLDOUT="${HOLDOUT:-500}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-4}"
TRAIN_LR="${TRAIN_LR:-0.0002}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-512}"

GEN_LIMIT="${GEN_LIMIT:-100}"
NUM_GENERATIONS="${NUM_GENERATIONS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
INITIAL_BATCH_SIZE="${INITIAL_BATCH_SIZE:-64}"

GENERATION_MODE="${GENERATION_MODE:-single}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TEMPERATURES="${TEMPERATURES:-}"
RANDOM_TEMPERATURE_COUNT="${RANDOM_TEMPERATURE_COUNT:-}"
TEMPERATURE_MIN="${TEMPERATURE_MIN:-0.0}"
TEMPERATURE_MAX="${TEMPERATURE_MAX:-3.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
USE_CACHE="${USE_CACHE:-1}"
BACKUP_EVERY="${BACKUP_EVERY:-10}"

OUTPUT_PATH="${OUTPUT_PATH:-${REVERSE_DIR}/cb_train_reverse_prompts.jsonl}"

echo "Python: ${PYTHON_BIN}"
echo "Circuit Breakers path: ${CB_PATH}"
echo "Reverse dir: ${REVERSE_DIR}"
echo "Adapter dir: ${ADAPTER_DIR}"
echo "Generation mode: ${GENERATION_MODE}"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "Training reverse model..."
  "${PYTHON_BIN}" scripts/train_reverse_model.py \
    --base_model "${BASE_MODEL}" \
    --cb_path "${CB_PATH}" \
    --output_dir "${REVERSE_DIR}" \
    --cb_limit "${CB_LIMIT}" \
    --ultra_samples "${ULTRA_SAMPLES}" \
    --holdout "${HOLDOUT}" \
    --max_len "${TRAIN_MAX_LEN}" \
    --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${TRAIN_GRAD_ACCUM}" \
    --learning_rate "${TRAIN_LR}" \
    --max_steps "${TRAIN_MAX_STEPS}"
else
  echo "Skipping training and reusing adapter at ${ADAPTER_DIR}"
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "Adapter directory not found: ${ADAPTER_DIR}" >&2
  exit 1
fi

GEN_ARGS=(
  --base-model "${BASE_MODEL}"
  --adapter-dir "${ADAPTER_DIR}"
  --cb-path "${CB_PATH}"
  --limit "${GEN_LIMIT}"
  --num-generations "${NUM_GENERATIONS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --initial-batch-size "${INITIAL_BATCH_SIZE}"
  --generation-mode "${GENERATION_MODE}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --backup-every "${BACKUP_EVERY}"
  --output-path "${OUTPUT_PATH}"
)

if [[ "${USE_CACHE}" == "1" ]]; then
  GEN_ARGS+=(--use-cache)
else
  GEN_ARGS+=(--no-cache)
fi

if [[ -n "${TEMPERATURES}" ]]; then
  # shellcheck disable=SC2206
  TEMP_ARRAY=(${TEMPERATURES})
  GEN_ARGS+=(--temperatures "${TEMP_ARRAY[@]}")
fi

if [[ -n "${RANDOM_TEMPERATURE_COUNT}" ]]; then
  GEN_ARGS+=(
    --random-temperature-count "${RANDOM_TEMPERATURE_COUNT}"
    --temperature-min "${TEMPERATURE_MIN}"
    --temperature-max "${TEMPERATURE_MAX}"
  )
fi

echo "Generating reverse prompts..."
"${PYTHON_BIN}" scripts/generate_reverse_prompts.py "${GEN_ARGS[@]}"

echo "Done. Output written to ${OUTPUT_PATH}"
