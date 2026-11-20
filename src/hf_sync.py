"""Share / restore the processed data through a Hugging Face dataset repo.

Data files are not committed to GitHub. Instead:

    python -m src.hf_sync push <user>/<repo>    # needs `huggingface-cli login` (write token)
    python -m src.hf_sync pull <user>/<repo>    # public repo, no login needed

Uploads data/processed/dataset.parquet and the trained models (models/*.joblib).
`pull` restores them in place, so `python run_pipeline.py --mode file` works
without downloading 3 years of klines.
"""
from __future__ import annotations

import sys

from huggingface_hub import HfApi, snapshot_download

from . import config as cfg

MODELS = cfg.ROOT / "models"
FILES = [(cfg.PROCESSED_DIR / "dataset.parquet", "data/processed/dataset.parquet")]


def push(repo: str) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    for local, remote in FILES:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(MODELS), path_in_repo="models", repo_id=repo, repo_type="dataset",
                      allow_patterns=["*.joblib"])
    print(f"uploaded -> https://huggingface.co/datasets/{repo}")


def pull(repo: str) -> None:
    snapshot_download(repo, repo_type="dataset", local_dir=str(cfg.ROOT),
                      allow_patterns=["data/processed/dataset.parquet", "models/*.joblib"])
    print("restored data/processed/dataset.parquet and models/")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("push", "pull"):
        sys.exit(__doc__)
    {"push": push, "pull": pull}[sys.argv[1]](sys.argv[2])
