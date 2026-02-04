from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path

from project.utils.io import load_yaml, save_yaml
from project.utils.seed import set_seed
from project.utils.logging import setup_logger

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    run_cfg = cfg.get("run", {})
    seed = int(run_cfg.get("seed", 42))
    out_root = Path(run_cfg.get("out_root", "results/runs"))
    base_name = args.run_name or run_cfg.get("name", "run")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{timestamp}_{base_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Config: {args.config}")

    # Save resolved config for reproducibility
    save_yaml(cfg, run_dir / "config_resolved.yaml")

    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    data_cfg = cfg.get("data", {})
    processed_root = data_cfg.get("processed_root", "data/processed")
    dataset = str(data_cfg.get("dataset", "deformingthings4d")).lower()
    split = str(data_cfg.get("split", "train")).lower()

    # Load dataset index (counts processed files)
    if dataset == "deformingthings4d":
        from project.data.datasets.deformingthings4d import discover_processed_files
    elif dataset == "faust":
        from project.data.datasets.faust import discover_processed_files
    elif dataset == "sapien":
        from project.data.datasets.sapien import discover_processed_files
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    index = discover_processed_files(processed_root, dataset, split)
    logger.info(f"Dataset={dataset}, split={split}")
    logger.info(f"Processed root: {processed_root}")
    logger.info(f"Found {len(index)} processed files")

    k = int(cfg.get("sanity", {}).get("max_examples_to_print", 3))
    if len(index) == 0:
        logger.warning("No processed files found. This is OK for Week 1 if preprocessing isn't done yet.")
        logger.warning("Expected: data/processed/<dataset>/<split>/*.npz (or .pt)")
    else:
        logger.info("Sample files:")
        for p in index.files[:k]:
            logger.info(f" - {p}")

    logger.info("Sanity run complete ✅")

if __name__ == "__main__":
    main()
