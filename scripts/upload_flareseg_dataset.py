from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id, e.g. user/flareseg")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--commit-message", default="Upload FlareSeg synthetic dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError("Install huggingface_hub before upload: pip install huggingface_hub") from e

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"uploaded {dataset_dir} to hf://datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

