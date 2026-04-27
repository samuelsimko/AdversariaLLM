#!/usr/bin/env bash
# Run scripts/evaluate_jepa_attack_guardrail.py on every JEPA defense under
# EXP_DIR. A "defense" is any direct subdirectory of EXP_DIR that contains both
# a manifest.json and a jepa_predictor.pt (i.e. anything trained by
# defenses/jepa_ce.py, including w_jepa=0 baselines whose predictor is just
# instantiated and never trained).
#
# Usage:
#   ./predict.sh
#   EXP_DIR=runs/experiments/jepa_byol_llama3 ./predict.sh
#   ./predict.sh --target_fpr 0.05                    # extra args go to the eval script
#   ONLY="no_jepa_500 jepa_byol_b64_500" ./predict.sh # restrict to a subset (space-separated)
#   SKIP_EXISTING=1 ./predict.sh                      # skip defenses with an existing JSON output
set -euo pipefail

EXP_DIR="${EXP_DIR:-runs/experiments/jepa_llama3_overnight}"
ONLY="${ONLY:-}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [[ ! -d "${EXP_DIR}" ]]; then
  echo "[predict.sh] EXP_DIR=${EXP_DIR} does not exist" >&2
  exit 1
fi

# Build the keep-set if ONLY is given (whitespace-separated names).
declare -A keep_set=()
if [[ -n "${ONLY}" ]]; then
  for name in ${ONLY}; do
    keep_set["${name}"]=1
  done
fi

# Discover defenses by scanning EXP_DIR for subdirs with the required artifacts.
DEFENSES=()
shopt -s nullglob
for run_dir in "${EXP_DIR}"/*/; do
  d="$(basename "${run_dir}")"
  if [[ -n "${ONLY}" && -z "${keep_set[${d}]:-}" ]]; then
    continue
  fi
  if [[ ! -f "${run_dir}manifest.json" ]]; then
    continue
  fi
  if [[ ! -f "${run_dir}jepa_predictor.pt" ]]; then
    continue
  fi
  DEFENSES+=("${d}")
done
shopt -u nullglob

if [[ ${#DEFENSES[@]} -eq 0 ]]; then
  echo "[predict.sh] no defenses with manifest.json + jepa_predictor.pt under ${EXP_DIR}" >&2
  if [[ -n "${ONLY}" ]]; then
    echo "[predict.sh] (ONLY=${ONLY} may have filtered them all out)" >&2
  fi
  exit 1
fi

echo "[predict.sh] EXP_DIR : ${EXP_DIR}"
echo "[predict.sh] found ${#DEFENSES[@]} defense(s):"
for d in "${DEFENSES[@]}"; do echo "  - ${d}"; done

EXTRA_ARGS=("$@")

for d in "${DEFENSES[@]}"; do
  run_dir="${EXP_DIR}/${d}"
  out_json="${run_dir}/guardrail_attack_eval.json"
  log_file="${run_dir}/guardrail_attack_eval.log"

  if [[ ! -d "${run_dir}/attacks" ]]; then
    echo "[predict.sh] skip ${d}: no attacks/ subdir"
    continue
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
    echo "[predict.sh] skip ${d}: ${out_json} already exists (SKIP_EXISTING=1)"
    continue
  fi

  echo
  echo "=================================================================="
  echo "[predict.sh] ${d}"
  echo "  run_dir : ${run_dir}"
  echo "  output  : ${out_json}"
  echo "  log     : ${log_file}"
  echo "=================================================================="

  python scripts/evaluate_jepa_attack_guardrail.py \
    --run_dir "${run_dir}" \
    --output_json "${out_json}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${log_file}"
done

echo
echo "[predict.sh] done. Per-model JSON summaries:"
for d in "${DEFENSES[@]}"; do
  echo "  ${EXP_DIR}/${d}/guardrail_attack_eval.json"
done
