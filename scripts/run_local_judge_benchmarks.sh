#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/venv/bin/python}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/runs/judge_benchmarks/$(date -u +%Y%m%dT%H%M%SZ)}"
INCLUDE_GPT_OSS="${INCLUDE_GPT_OSS:-0}"
STRONGREJECT_COUNT="${STRONGREJECT_COUNT:-128}"
HARMBENCH_COUNT="${HARMBENCH_COUNT:-128}"
GPT_OSS_COUNT="${GPT_OSS_COUNT:-32}"
ROUNDS="${ROUNDS:-1}"

# Hugging Face Xet downloads can be flaky in some environments; prefer plain HTTP downloads.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "$OUT_DIR"

failures=0

run_bench() {
  local classifier="$1"
  local count="$2"
  local rounds="${3:-1}"
  local log_path="$OUT_DIR/${classifier}_count${count}.log"

  echo "==> classifier=${classifier} count=${count} rounds=${rounds}"
  if "$PYTHON_BIN" "$ROOT_DIR/scripts/benchmark_local_judges.py" \
    --classifier "$classifier" \
    --count "$count" \
    --rounds "$rounds" 2>&1 | tee "$log_path"; then
    echo "saved_log=$log_path"
  else
    echo "benchmark_failed classifier=${classifier} log=$log_path"
    failures=$((failures + 1))
  fi
}

run_bench strongreject "$STRONGREJECT_COUNT" "$ROUNDS"
run_bench harmbench "$HARMBENCH_COUNT" "$ROUNDS"

if [[ "$INCLUDE_GPT_OSS" == "1" ]]; then
  run_bench gpt_oss "$GPT_OSS_COUNT" "$ROUNDS"
else
  echo "==> skipping gpt_oss by default (set INCLUDE_GPT_OSS=1 to enable)"
fi

echo "all_logs_dir=$OUT_DIR"
echo "failures=$failures"

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
