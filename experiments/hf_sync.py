"""HuggingFace incremental sync for the headline-backup-rerun experiment.

Called by execute_job.py after a stage completes successfully. Pushes
the relevant artifact to a private HF dataset repo so a node failure
costs at most one cell. Idempotent — re-uploading the same file is a
no-op on HF's side when the SHA matches.

Layout in the repo:
    adapters/<cell>/                 LoRA adapter + jepa_predictor.pt
    attack_results/<cell>/<attack>/  attack stage_dir contents (run.json, status, etc.)
    eval_results/<cell>/<eval>/      benign-eval results
    eval_results/<cell>/probe_<p>/   probe results
    manifest/<cell>.json             append-only completion log per cell

Manifest design: one JSON file per cell, append-only entries. We never
read-modify-write a shared manifest because 8 cells run in parallel and
HF's API isn't transactional. Each cell only writes its own file → no races.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _disabled() -> bool:
    return os.environ.get("HF_SYNC_ENABLED", "1") not in ("1", "true", "True", "yes")


def _repo_id() -> str:
    repo = os.environ.get("HF_REPO", "").strip()
    # The spec literally writes HF_REPO=/headline-backup-rerun (missing namespace).
    # Tolerate that by prefixing the authenticated user.
    if repo.startswith("/"):
        repo = repo.lstrip("/")
        from huggingface_hub import whoami
        try:
            user = whoami()["name"]
            repo = f"{user}/{repo}"
        except Exception:
            pass
    return repo


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def _upload_folder(local: Path, repo_path: str, repo_id: str, commit: str):
    from huggingface_hub import upload_folder
    upload_folder(
        folder_path=str(local),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit,
        token=os.environ.get("HF_TOKEN"),
        # Ignore optimizer state, raw checkpoints (just the final adapter is enough)
        ignore_patterns=[
            "checkpoint-*/optimizer.pt",
            "checkpoint-*/scheduler.pt",
            "checkpoint-*/rng_state*",
            "checkpoint-*/training_args.bin",
            "*.tmp",
            "*.lock",
            "wandb/**",
        ],
    )


def _upload_file(local: Path, repo_path: str, repo_id: str, commit: str):
    from huggingface_hub import upload_file
    upload_file(
        path_or_fileobj=str(local),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit,
        token=os.environ.get("HF_TOKEN"),
    )


def _append_manifest(repo_id: str, cell: str, entry: dict[str, Any]):
    """Append one entry to manifest/<cell>.json. We pull-modify-push only this
    cell's file, which has a single writer (this cell), so it's race-free.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

    rel = f"manifest/{cell}.json"
    existing: list[dict[str, Any]] = []
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            filename=rel,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        existing = json.loads(Path(local).read_text())
        if not isinstance(existing, list):
            existing = []
    except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError, FileNotFoundError):
        existing = []

    existing.append(entry)

    tmp = Path("/tmp") / f"manifest-{cell}-{os.getpid()}.json"
    tmp.write_text(json.dumps(existing, indent=2))
    _upload_file(tmp, rel, repo_id, f"manifest: {cell} += {entry.get('key','?')}")
    try:
        tmp.unlink()
    except Exception:
        pass


def _cell_from_fingerprint(fp: dict[str, Any]) -> str | None:
    return fp.get("defense") or fp.get("cell")


def sync_after_stage(stage_dir: Path, fingerprint: dict[str, Any], job_type: str) -> None:
    """Main entry. Decides what to push based on job_type and fingerprint."""
    if _disabled():
        return
    repo_id = _repo_id()
    if not repo_id:
        print("[hf_sync] HF_REPO not set; skipping", file=sys.stderr)
        return

    cell = _cell_from_fingerprint(fingerprint)
    if not cell:
        print(f"[hf_sync] no cell in fingerprint; skipping (job_type={job_type})", file=sys.stderr)
        return

    try:
        if job_type == "train":
            _sync_train(stage_dir, cell, repo_id)
        elif job_type == "attack":
            _sync_attack(stage_dir, fingerprint, cell, repo_id)
        elif job_type == "benign_eval":
            _sync_benign(stage_dir, fingerprint, cell, repo_id)
        elif job_type == "probe":
            _sync_probe(stage_dir, fingerprint, cell, repo_id)
        else:
            print(f"[hf_sync] unknown job_type={job_type}; skipping", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - never fail the job for a sync hiccup
        print(f"[hf_sync] WARN: sync failed for {job_type} {cell}: {exc}", file=sys.stderr)
        traceback.print_exc()


def _sync_train(stage_dir: Path, cell: str, repo_id: str):
    """stage_dir is out_root/<cell>; the adapter lives at <stage_dir>/lora_adapter."""
    adapter = stage_dir / "lora_adapter"
    if not adapter.exists():
        print(f"[hf_sync] no lora_adapter at {adapter}; skipping", file=sys.stderr)
        return
    _upload_folder(adapter, f"adapters/{cell}", repo_id, f"train {cell}")
    # Upload jepa_predictor.pt + manifest.json if they exist alongside (not always inside lora_adapter).
    for extra in ("jepa_predictor.pt", "manifest.json", "metrics.csv"):
        p = stage_dir / extra
        if p.exists():
            _upload_file(p, f"adapters/{cell}/{extra}", repo_id, f"train {cell}: {extra}")
    _append_manifest(repo_id, cell, {"ts": _now(), "kind": "train", "key": "adapter", "status": "complete"})
    print(f"[hf_sync] pushed adapter for {cell}")


def _sync_attack(stage_dir: Path, fp: dict[str, Any], cell: str, repo_id: str):
    attack_name = fp.get("attack") or fp.get("attack_name") or stage_dir.name
    _upload_folder(stage_dir, f"attack_results/{cell}/{attack_name}", repo_id, f"attack {cell}/{attack_name}")
    _append_manifest(repo_id, cell, {
        "ts": _now(), "kind": "attack", "key": attack_name, "status": "complete"
    })
    print(f"[hf_sync] pushed attack {cell}/{attack_name}")


def _sync_benign(stage_dir: Path, fp: dict[str, Any], cell: str, repo_id: str):
    eval_name = fp.get("benign_eval") or stage_dir.name
    _upload_folder(stage_dir, f"eval_results/{cell}/{eval_name}", repo_id, f"benign {cell}/{eval_name}")
    _append_manifest(repo_id, cell, {
        "ts": _now(), "kind": "benign_eval", "key": eval_name, "status": "complete"
    })
    print(f"[hf_sync] pushed benign {cell}/{eval_name}")


def _sync_probe(stage_dir: Path, fp: dict[str, Any], cell: str, repo_id: str):
    probe_name = fp.get("probe") or stage_dir.name
    _upload_folder(stage_dir, f"eval_results/{cell}/probe_{probe_name}", repo_id, f"probe {cell}/{probe_name}")
    _append_manifest(repo_id, cell, {
        "ts": _now(), "kind": "probe", "key": probe_name, "status": "complete"
    })
    print(f"[hf_sync] pushed probe {cell}/{probe_name}")


def _cli():
    """Manual CLI entry: `python -m experiments.hf_sync test`, `... push <stage_dir> <fingerprint.json> <job_type>`."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("test")
    p = sub.add_parser("push")
    p.add_argument("stage_dir")
    p.add_argument("fingerprint_json")
    p.add_argument("job_type")
    args = parser.parse_args()

    if args.cmd == "test":
        repo_id = _repo_id()
        if not repo_id:
            print("HF_REPO unset", file=sys.stderr)
            sys.exit(2)
        tmp = Path("/tmp") / f"hf_sync_smoketest_{os.getpid()}.txt"
        tmp.write_text(f"smoketest {_now()}\n")
        _upload_file(tmp, "smoketest.txt", repo_id, "hf_sync smoketest")
        tmp.unlink()
        print(f"OK pushed smoketest.txt to {repo_id}")
        return

    if args.cmd == "push":
        fp = json.loads(Path(args.fingerprint_json).read_text())
        sync_after_stage(Path(args.stage_dir), fp, args.job_type)
        return


if __name__ == "__main__":
    _cli()
