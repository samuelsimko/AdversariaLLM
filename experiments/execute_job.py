#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute a stage command and write experiment job metadata.")
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--job-type", required=True)
    parser.add_argument("--fingerprint-json", default=None)
    parser.add_argument("--fingerprint-b64", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("No command provided to execute_job.py")

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if args.fingerprint_b64:
        fingerprint = json.loads(base64.b64decode(args.fingerprint_b64).decode("utf-8"))
    elif args.fingerprint_json:
        fingerprint = json.loads(args.fingerprint_json)
    else:
        raise ValueError("One of --fingerprint-json or --fingerprint-b64 is required")

    fingerprint_path = stage_dir / "fingerprint.json"
    ready_path = stage_dir / "READY"
    failed_path = stage_dir / "FAILED"
    status_path = stage_dir / "status.json"
    command_path = stage_dir / "command.txt"

    if failed_path.exists():
        failed_path.unlink()
    if ready_path.exists():
        ready_path.unlink()

    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")
    fingerprint_path.write_text(json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8")

    status = {
        "job_name": args.job_name,
        "job_type": args.job_type,
        "allow_failure": args.allow_failure,
        "command": command,
        "fingerprint": fingerprint,
        "started_at": utc_now(),
        "cwd": args.cwd or os.getcwd(),
        "status": "running",
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(command, cwd=args.cwd, check=False)

    status["finished_at"] = utc_now()
    status["returncode"] = completed.returncode
    status["status"] = "ok" if completed.returncode == 0 else "failed"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    if completed.returncode == 0:
        ready_path.touch()
        try:
            repo_root = str(Path(__file__).resolve().parents[1])
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from experiments.hf_sync import sync_after_stage
            sync_after_stage(stage_dir, fingerprint, args.job_type)
        except Exception as exc:  # never fail the job for a sync hiccup
            print(f"[execute_job] hf_sync error: {exc}", file=sys.stderr)
    else:
        failed_path.touch()

    if completed.returncode != 0 and args.allow_failure:
        return 0
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
