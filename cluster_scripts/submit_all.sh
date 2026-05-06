#!/usr/bin/env bash
# Generate per-cell sbatch scripts from cell.slurm.template and submit them.
#
# Usage:
#   bash cluster_scripts/submit_all.sh                       # all cells in CONFIG
#   bash cluster_scripts/submit_all.sh q_cb_pra l_cb_pra     # specific cells
#   DRY_RUN=1 bash cluster_scripts/submit_all.sh             # generate scripts, don't submit
#
# Configurable via env vars (with defaults):
#   CONFIG        experiments/configs/headline_rerun_full.yaml
#   REPO_DIR      $(git rev-parse --show-toplevel)
#   ACCOUNT       infra01
#   PARTITION     normal
#   TIME          04:00:00
#   GPUS_PER_NODE 4
#   MEM           460800
#   LOGS_DIR      $REPO_DIR/cluster_scripts/logs
#   HF_REPO       (must export this before running for HF sync to work)
#   HF_TOKEN      (must export this before running for HF sync to work)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel)}"
CONFIG="${CONFIG:-experiments/configs/headline_rerun_full.yaml}"
ACCOUNT="${ACCOUNT:-infra01}"
PARTITION="${PARTITION:-normal}"
TIME_LIMIT="${TIME:-04:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MEM="${MEM:-460800}"
LOGS_DIR="${LOGS_DIR:-$REPO_DIR/cluster_scripts/logs}"
GENERATED_DIR="${GENERATED_DIR:-$REPO_DIR/cluster_scripts/generated}"

mkdir -p "$LOGS_DIR" "$GENERATED_DIR"

if [ ! -f "$REPO_DIR/$CONFIG" ]; then
  echo "ERROR: config not found: $REPO_DIR/$CONFIG" >&2
  exit 1
fi

# Determine which cells to submit. If args given, use them; else read all
# pipeline names from the YAML.
if [ "$#" -gt 0 ]; then
  CELLS=("$@")
else
  mapfile -t CELLS < <(
    "${PYTHON_BIN:-python}" - <<EOF
import sys, yaml
with open("$REPO_DIR/$CONFIG") as f:
    cfg = yaml.safe_load(f)
for name in cfg.get("pipelines", {}):
    print(name)
EOF
  )
fi

if [ "${#CELLS[@]}" -eq 0 ]; then
  echo "ERROR: no cells to submit (config has no pipelines?)" >&2
  exit 1
fi

echo "[submit_all] $(date -u +%FT%TZ) submitting ${#CELLS[@]} cells from $CONFIG"
echo "[submit_all] sbatch defaults: account=$ACCOUNT partition=$PARTITION time=$TIME_LIMIT gpus=$GPUS_PER_NODE mem=$MEM"
[ -z "${HF_REPO:-}" ] && echo "[submit_all] WARNING: HF_REPO not set in env; sync will be a no-op" >&2
[ -z "${HF_TOKEN:-}" ] && echo "[submit_all] WARNING: HF_TOKEN not set in env; sync will be a no-op" >&2

TEMPLATE="$REPO_DIR/cluster_scripts/cell.slurm.template"
for CELL in "${CELLS[@]}"; do
  OUT="$GENERATED_DIR/${CELL}.slurm"
  sed \
    -e "s|{CELL}|$CELL|g" \
    -e "s|{ACCOUNT}|$ACCOUNT|g" \
    -e "s|{PARTITION}|$PARTITION|g" \
    -e "s|{TIME}|$TIME_LIMIT|g" \
    -e "s|{GPUS_PER_NODE}|$GPUS_PER_NODE|g" \
    -e "s|{MEM}|$MEM|g" \
    -e "s|{LOGS_DIR}|$LOGS_DIR|g" \
    -e "s|{REPO_DIR}|$REPO_DIR|g" \
    -e "s|{CONFIG}|$CONFIG|g" \
    "$TEMPLATE" > "$OUT"
  chmod +x "$OUT"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[submit_all] (dry-run) would submit: $OUT"
  else
    JOBID=$(sbatch --parsable "$OUT")
    echo "[submit_all] $CELL -> job $JOBID ($OUT)"
  fi
done

echo "[submit_all] done. Per-cell logs: $LOGS_DIR/<cell>.<jobid>.{out,err}"
echo "[submit_all] Track progress: bash cluster_scripts/sync_status.sh"
