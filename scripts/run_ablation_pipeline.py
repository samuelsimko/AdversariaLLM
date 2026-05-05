"""Per-GPU sequential pipeline orchestrator for the Phase 1 ablation.

Each GPU thread runs cells sequentially through the full pipeline:
  for cell in my_cells:
    1. train (defenses/jepa_ce.py)
    2. probe   (Why-Probe-Fails extract_states + compare_probes)
    3. benign  (benign_capabilities/run_benign_eval.py, gsm8k --limit)
    4. write SUMMARY.csv row

Cells are distributed round-robin across GPUs. CSV is appended atomically
(threading.Lock) so partial results are usable any time. Per-cell stages
write a READY sentinel at the end so re-launches skip completed cells.

Usage:
  PYTHONPATH=. python scripts/run_ablation_pipeline.py \\
      --cells scripts/ablation_cells.json \\
      --num-gpus 3 \\
      --out-root runs/ablation_phase1 \\
      --gsm8k-limit 200

Stop at any time with Ctrl-C; the running cell will be killed but the CSV
is preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/workspace/AdversariaLLM/venv/bin/python"
WPF_ROOT = Path("/workspace/AdversariaLLM/Why-Probe-Fails")
ENV_FILE = REPO_ROOT / ".env"

CB_PATH = REPO_ROOT / "data" / "circuit_breakers_train.json"
PAIR_PATH = REPO_ROOT / "reverse_model" / "cb_train_reverse_prompts_5000_random_temp.jsonl"

PROBE_CONFIG = "configs/rs3_advbench_jepa.json"
PROBE_DATA_ROOTS = [
    ("data/RS3", ["malicious", "cleaned", "paraphrased"]),
    ("data/RS1", ["benign"]),
]
PROBE_SEEDS = [42, 123, 777, 1234, 9999]
PROBE_NAMES = ["svm_raw", "mlp_no_jepa", "jepa", "svm_on_jepa_z"]
HEADLINE_SLICE = "ood_heldout_paraphrased_harmbench"

# Mirror headline_pra_joint_n50 training knobs (only the ablation knobs vary).
TRAIN_DEFAULTS = dict(
    num_max_steps=1500,
    batch_size=4,
    grad_accum=2,
    ultrachat_samples=5000,
    limit_cb=5000,
    pair_limit=100000,
    pair_sample_size=8000,
    pair_sample_balanced=True,
    max_length=256,
    lr=0.0002,
    jepa_target_encoder="defended",
    predictor_type="mlp",
    predictor_dropout=0.0,
    jepa_target="prompt_only",
    report_to="wandb",
    save_total_limit=1,
    logging_steps=50,
    save_steps=1500,
    lora_r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="q_proj,v_proj",
    pair_path=str(PAIR_PATH),
    pair_format="reverse",
    w_benign=0.1,
    w_harm=1.0,
    w_kl=1.0,
    harm_ce_min=5.0,
    include_pair_harmful_in_harm_ce=True,
)

CB_EXTRA = dict(cb_beta_start_mult=10.0, cb_beta_decay_fraction=1.0)

CSV_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with PRINT_LOCK:
        print(f"[{ts}] {msg}", flush=True)


def run_step(name: str, cmd: list[str], log_path: Path, env: dict, cwd: Path | None = None) -> tuple[bool, float]:
    """Run one stage; return (ok, seconds_elapsed). Logs stdout+stderr to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(f"  [{name}] start  log={log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = proc.returncode == 0
    log(f"  [{name}] {'OK' if ok else 'FAIL'} rc={proc.returncode} dt={dt:.1f}s")
    return ok, dt


def train_cmd(cell: dict, out_dir: Path) -> list[str]:
    args = dict(TRAIN_DEFAULTS)
    args.update(
        align_layer=cell["align_layer"],
        predictor_layers=cell["predictor_layers"],
        predictor_bottleneck_dim=cell["predictor_bottleneck_dim"],
        predictor_lr_multiplier=cell["predictor_lr_multiplier"],
        w_jepa=cell["w_jepa"],
        harm_regularizer=cell["harm_regularizer"],
        run_name=f"ablation_{cell['name']}",
    )
    if cell["harm_regularizer"] == "circuit_breaker":
        args.update(CB_EXTRA)

    cmd = [
        PYTHON_BIN,
        str(REPO_ROOT / "defenses" / "jepa_ce.py"),
        "--model", cell["base_model"],
        "--cb_path", str(CB_PATH),
        "--output_dir", str(out_dir),
    ]
    for k, v in args.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))
    return cmd


def probe_shell(cell: dict, out_dir: Path) -> str:
    """Build the chained extract_states + compare_probes shell command."""
    states_dir = out_dir / "probe" / "states"
    parts = [f"cd {shlex.quote(str(WPF_ROOT))}"]
    lora_path = out_dir / "lora_adapter"
    for dr_path, dr_views in PROBE_DATA_ROOTS:
        cmd = [
            shlex.quote(PYTHON_BIN),
            "scripts/extract_states.py",
            "--model_path", shlex.quote(cell["base_model"]),
            "--data_root", shlex.quote(dr_path),
            "--layer_idx", str(cell["align_layer"]),
            "--out_dir", shlex.quote(str(states_dir)),
            "--views", *(shlex.quote(v) for v in dr_views),
            "--adapter_path", shlex.quote(str(lora_path)),
        ]
        parts.append(" ".join(cmd))
    for seed in PROBE_SEEDS:
        seed_dir = out_dir / "probe" / f"seed_{seed}"
        cmd = [
            shlex.quote(PYTHON_BIN),
            "scripts/compare_probes.py",
            "--config", shlex.quote(PROBE_CONFIG),
            "--states_dir", shlex.quote(str(states_dir)),
            "--out_dir", shlex.quote(str(seed_dir)),
            "--seed", str(seed),
            "--probes", *(shlex.quote(p) for p in PROBE_NAMES),
        ]
        parts.append(" ".join(cmd))
    return " && ".join(parts)


def benign_cmd(cell: dict, out_dir: Path, gsm8k_limit: int) -> list[str]:
    return [
        PYTHON_BIN,
        str(REPO_ROOT / "benign_capabilities" / "run_benign_eval.py"),
        "--model", cell["base_model"],
        "--lora", str(out_dir / "lora_adapter"),
        "--tasks", "gsm8k",
        "--limit", str(gsm8k_limit),
        "--output-dir", str(out_dir / "benign"),
    ]


def parse_probe_aucs(out_dir: Path) -> dict[str, float]:
    """Mean-across-seeds AUC for the headline slice, per probe."""
    by_probe: dict[str, list[float]] = {p: [] for p in PROBE_NAMES}
    for seed in PROBE_SEEDS:
        f = out_dir / "probe" / f"seed_{seed}" / "results.csv"
        if not f.exists():
            continue
        for row in csv.DictReader(open(f)):
            if row.get("slice") != HEADLINE_SLICE:
                continue
            p = row.get("probe")
            if p in by_probe:
                try:
                    by_probe[p].append(float(row["auc"]))
                except Exception:
                    pass
    return {p: (mean(vs) if vs else float("nan")) for p, vs in by_probe.items()}


def parse_gsm8k(out_dir: Path) -> float | None:
    """Pull gsm8k flexible-extract acc from lm-eval's results.json."""
    benign_dir = out_dir / "benign"
    if not benign_dir.exists():
        return None
    candidates = list(benign_dir.glob("**/results*.json"))
    for c in candidates:
        try:
            data = json.loads(c.read_text())
        except Exception:
            continue
        results = data.get("results") or {}
        gsm = results.get("gsm8k") or results.get("gsm8k,strict-match") or {}
        for k in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match"):
            if k in gsm:
                try:
                    return float(gsm[k])
                except Exception:
                    pass
    return None


def write_csv_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with CSV_LOCK:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def gpu_worker(gpu_id: str, cells: list[dict], out_root: Path, csv_path: Path,
               gsm8k_limit: int, skip_benign: bool) -> None:
    log(f"[GPU {gpu_id}] starting on {len(cells)} cells")
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    base_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    base_env["TOKENIZERS_PARALLELISM"] = "false"
    # HF token already loaded in parent env (we source .env before launching).

    for i, cell in enumerate(cells):
        cell_t0 = time.time()
        out_dir = out_root / cell["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"[GPU {gpu_id}] ({i+1}/{len(cells)}) cell={cell['name']}")

        # Persist the cell config alongside its outputs.
        (out_dir / "cell.json").write_text(json.dumps(cell, indent=2))

        row = dict(
            name=cell["name"],
            gpu=gpu_id,
            backbone=cell["backbone"],
            harm_regularizer=cell["harm_regularizer"],
            w_jepa=cell["w_jepa"],
            predictor_lr_multiplier=cell["predictor_lr_multiplier"],
            align_layer=cell["align_layer"],
            predictor_layers=cell["predictor_layers"],
            predictor_bottleneck_dim=cell["predictor_bottleneck_dim"],
            train_status="skip", train_seconds=0.0,
            probe_status="skip", probe_seconds=0.0,
            benign_status="skip", benign_seconds=0.0,
            auc_svm_raw=float("nan"), auc_mlp_no_jepa=float("nan"),
            auc_jepa=float("nan"), auc_svm_on_jepa_z=float("nan"),
            gsm8k_em=float("nan"),
            cell_seconds=0.0,
        )

        # ---- train ----
        train_ready = out_dir / "TRAIN_READY"
        if train_ready.exists():
            log(f"  [train] skip (TRAIN_READY exists)")
            row["train_status"] = "cached"
        else:
            cmd = train_cmd(cell, out_dir)
            ok, dt = run_step("train", cmd, out_dir / "logs" / "train.log", base_env)
            row["train_seconds"] = round(dt, 1)
            row["train_status"] = "ok" if ok else "fail"
            if ok:
                train_ready.touch()
            else:
                row["cell_seconds"] = round(time.time() - cell_t0, 1)
                write_csv_row(csv_path, row)
                continue

        # ---- probe ----
        probe_ready = out_dir / "PROBE_READY"
        if probe_ready.exists():
            log(f"  [probe] skip (PROBE_READY exists)")
            row["probe_status"] = "cached"
        else:
            sh = probe_shell(cell, out_dir)
            cmd = ["bash", "-lc", sh]
            ok, dt = run_step("probe", cmd, out_dir / "logs" / "probe.log", base_env)
            row["probe_seconds"] = round(dt, 1)
            row["probe_status"] = "ok" if ok else "fail"
            if ok:
                probe_ready.touch()
        if (out_dir / "PROBE_READY").exists():
            aucs = parse_probe_aucs(out_dir)
            row["auc_svm_raw"]       = aucs.get("svm_raw", float("nan"))
            row["auc_mlp_no_jepa"]   = aucs.get("mlp_no_jepa", float("nan"))
            row["auc_jepa"]          = aucs.get("jepa", float("nan"))
            row["auc_svm_on_jepa_z"] = aucs.get("svm_on_jepa_z", float("nan"))

        # ---- benign ----
        if skip_benign:
            row["benign_status"] = "skip"
        else:
            benign_ready = out_dir / "BENIGN_READY"
            if benign_ready.exists():
                log(f"  [benign] skip (BENIGN_READY exists)")
                row["benign_status"] = "cached"
            else:
                cmd = benign_cmd(cell, out_dir, gsm8k_limit)
                ok, dt = run_step("benign", cmd, out_dir / "logs" / "benign.log", base_env)
                row["benign_seconds"] = round(dt, 1)
                row["benign_status"] = "ok" if ok else "fail"
                if ok:
                    benign_ready.touch()
            row["gsm8k_em"] = parse_gsm8k(out_dir) or float("nan")

        row["cell_seconds"] = round(time.time() - cell_t0, 1)
        write_csv_row(csv_path, row)
        log(f"[GPU {gpu_id}] cell {cell['name']} done in {row['cell_seconds']:.0f}s "
            f"(train={row['train_seconds']:.0f}s probe={row['probe_seconds']:.0f}s "
            f"benign={row['benign_seconds']:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--num-gpus", type=int, default=3)
    ap.add_argument("--gpu-ids", default=None, help="Comma-separated explicit GPU ids (overrides --num-gpus).")
    ap.add_argument("--out-root", default="runs/ablation_phase1")
    ap.add_argument("--gsm8k-limit", type=int, default=200)
    ap.add_argument("--skip-benign", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N cells (for smoke tests).")
    args = ap.parse_args()

    cells = json.loads(Path(args.cells).read_text())
    if args.limit:
        cells = cells[:args.limit]
    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    else:
        gpu_ids = [str(i) for i in range(args.num_gpus)]
    n_gpus = len(gpu_ids)

    # Resolve to absolute: probe stage `cd`s into Why-Probe-Fails before
    # invoking extract_states/compare_probes, so any relative paths break.
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "SUMMARY.csv"
    log(f"orchestrator: {len(cells)} cells, {n_gpus} GPUs, out_root={out_root}, csv={csv_path}")

    # Round-robin distribute.
    groups: list[list[dict]] = [[] for _ in range(n_gpus)]
    for i, cell in enumerate(cells):
        groups[i % n_gpus].append(cell)
    for gpu, g in zip(gpu_ids, groups):
        log(f"  GPU {gpu}: {len(g)} cells -> {[c['name'] for c in g][:3]} ...")

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = []
        for gpu, g in zip(gpu_ids, groups):
            futures.append(pool.submit(gpu_worker, gpu, g, out_root, csv_path,
                                       args.gsm8k_limit, args.skip_benign))
        for f in futures:
            f.result()
    log("orchestrator: all workers finished")


if __name__ == "__main__":
    main()
