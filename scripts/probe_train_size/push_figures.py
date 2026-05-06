"""Upload assets/figures/probe_train_size/ to a private HF dataset.

Usage:
    python scripts/probe_train_size/push_figures.py [--repo samuelsimko/adversariallm-figures]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--repo", default="samuelsimko/adversariallm-figures")
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("assets/figures/probe_train_size"),
    )
    ap.add_argument(
        "--path_in_repo",
        default="probe_train_size",
        help="Subfolder inside the HF dataset.",
    )
    ap.add_argument("--commit_message", default="probe-train-size sweep figures")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"src dir does not exist: {args.src}")

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"[push] {args.src} -> dataset:{args.repo}:{args.path_in_repo}")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(args.src),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        ignore_patterns=["*.npz", "*.cache", "**/__pycache__/**"],
    )
    print("[done]")


if __name__ == "__main__":
    main()
