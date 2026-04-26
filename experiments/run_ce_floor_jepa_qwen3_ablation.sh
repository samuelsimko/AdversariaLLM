#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-experiments/configs/ce_floor_jepa_qwen3_ablation.yaml}"

/workspace/AdversariaLLM/venv/bin/python /workspace/AdversariaLLM/experiments/run_experiment.py --config "${CONFIG}"
