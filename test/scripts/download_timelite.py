from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from .utils import DATA_DIR, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TIME-Lite dataset to cache dir")
    parser.add_argument("--split", type=str, default=None, help="split name (default: auto)")
    args = parser.parse_args()

    ensure_dirs()
    cache_dir = DATA_DIR / "time_lite"
    ds = load_dataset("SylvainWei/TIME-Lite", cache_dir=str(cache_dir))
    split_name = args.split
    if split_name and split_name not in ds:
        raise SystemExit(f"Split {split_name} not in dataset; available: {list(ds.keys())}")
    if split_name:
        _ = ds[split_name]
    else:
        for name in ("test", "validation", "train"):
            if name in ds:
                _ = ds[name]
                break
        else:
            first = list(ds.keys())[0]
            _ = ds[first]
    print(f"Downloaded TIME-Lite to {cache_dir}")


if __name__ == "__main__":
    main()
