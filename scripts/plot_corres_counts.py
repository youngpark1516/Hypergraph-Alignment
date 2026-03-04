#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def collect_counts(root: Path):
    files = sorted(root.glob("**/*.npz"))
    if not files:
        raise SystemExit(f"No .npz files found under {root}")
    counts = []
    for path in files:
        data = np.load(path)
        if "corres" not in data:
            raise ValueError(f"'corres' missing in {path}")
        counts.append(int(data["corres"].shape[0]))
    return counts, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root folder with .npz files")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    parser.add_argument("--save", default="", help="Optional path to save figure")
    parser.add_argument(
        "--remove_outliers",
        action="store_true",
        help="Remove outliers using percentile trimming",
    )
    parser.add_argument(
        "--outlier_pct",
        type=float,
        default=1.0,
        help="Percent to trim from each tail when removing outliers",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Filter out counts greater than this value (0 = disabled)",
    )
    args = parser.parse_args()

    counts, _ = collect_counts(Path(args.root))
    if args.remove_outliers:
        if args.outlier_pct < 0 or args.outlier_pct >= 50:
            raise SystemExit("--outlier_pct must be in [0, 50)")
        lower = np.percentile(counts, args.outlier_pct)
        upper = np.percentile(counts, 100.0 - args.outlier_pct)
        counts = [c for c in counts if lower <= c <= upper]
    if args.max > 0:
        counts = [c for c in counts if c <= args.max]

    plt.figure(figsize=(7, 4))
    plt.hist(counts, bins=args.bins, color="#2c7fb8", edgecolor="white")
    plt.title("Distribution of corres length")
    plt.xlabel("data['corres'].shape[0]")
    plt.ylabel("Number of files")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
