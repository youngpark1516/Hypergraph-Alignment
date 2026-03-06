"""Run ICP full-eval over multiple (dataset, root) configurations and aggregate results.

Edit the CONFIGS list below to define your sweep, then run:
    python scripts/icp_sweep.py [--out results/icp_sweep.csv] [--workers 12] [--repeats 5]

Each config is run `n_repeats` times (varying split_seed) and the aggregate CSV
reports mean ± std across repeats.  A second CSV (<out_stem>_runs.csv) stores
every individual repeat for inspection.

Each config produces its own per-pair summary CSV under results/<name>_seedN/,
and all aggregate metrics are collected into a single comparison CSV.
"""

import argparse
import csv
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make project importable when run from workspace root
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.project.models.baselines.icp import batch_run_icp_full_eval  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """One ICP evaluation run."""
    name: str                          # short label for the results table
    dataset: str                       # "faust" or "partnet"
    root: str                          # path to directory of .npz pair files

    # ---- dataset / loader ----
    val_ratio: float = 0.3
    use_features: bool = True
    feature_dim: int = 33
    num_node: int = 512
    full_data: bool = False
    neg_ratio: float = 2.0
    max_corr: int = 0
    num_workers: int = 12
    split_seed: int = 0

    # ---- eval filtering ----
    eval_num: int = 0
    eval_seed: int = 0
    eval_all_pairs: bool = False
    fit_all_pairs: bool = False

    # ---- ICP ----
    icp_threshold: float = float("inf")
    init_from_fpfh: bool = True

    # ---- output ----
    out: Optional[str] = None          # None → auto-named under results/

    # ---- internal (not exposed to user) ----
    eval_snapshot: str = ""
    mode: str = "train"
    config: str = ""

    # ---- repetitions ----
    n_repeats: int = 10               # how many times to re-run with different seeds

    # extra kwargs forwarded verbatim
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EDIT THIS LIST to define your sweep
# ============================================================
CONFIGS: List[RunConfig] = [
    RunConfig(
        name="faust_fpfh",
        dataset="faust",
        root="data/faust/fpfh",
    ),
    RunConfig(
        name="partnet_fpfh_rigid",
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
    ),
    RunConfig(
        name="partnet_fpfh_affine_1-5",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
    ),
    RunConfig(
        name="partnet_fpfh_affine_2-0",
        dataset="partnet",
        root="data/partnet/fpfh_affine_2-0",
    ),
]
# ============================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_to_args(
    cfg: RunConfig,
    global_workers: Optional[int],
    seed_offset: int = 0,
    repeat_idx: int = 0,
) -> argparse.Namespace:
    """Convert a RunConfig into the args namespace expected by batch_run_icp_full_eval.

    seed_offset is added to cfg.split_seed so each repeat uses a different random split.
    """
    split_seed = cfg.split_seed + seed_offset
    run_out = cfg.out or f"results/{cfg.name}_seed{split_seed}"
    ns = argparse.Namespace(
        dataset=cfg.dataset,
        root=cfg.root,
        val_ratio=cfg.val_ratio,
        use_features=cfg.use_features,
        feature_dim=cfg.feature_dim,
        num_node=cfg.num_node,
        full_data=cfg.full_data,
        neg_ratio=cfg.neg_ratio,
        max_corr=cfg.max_corr,
        num_workers=global_workers if global_workers is not None else cfg.num_workers,
        split_seed=split_seed,
        eval_num=cfg.eval_num,
        eval_seed=cfg.eval_seed + seed_offset,
        eval_snapshot=cfg.eval_snapshot,
        mode=cfg.mode,
        config=cfg.config,
        icp_threshold=cfg.icp_threshold,
        init_from_fpfh=cfg.init_from_fpfh,
        eval_all_pairs=cfg.eval_all_pairs,
        fit_all_pairs=cfg.fit_all_pairs,
        out=run_out,
    )
    for k, v in cfg.extra.items():
        setattr(ns, k, v)
    return ns


def _read_aggregate(summary_csv: str) -> Dict[str, Any]:
    """Read per-pair summary CSV and compute aggregate metrics."""
    rows = []
    with open(summary_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def _mean(key):
        vals = _vals(key)
        return float(np.mean(vals)) if vals else float("nan")

    def _vals(key):
        out = []
        for r in rows:
            v = r.get(key, "")
            try:
                fv = float(v)
                if not math.isnan(fv):
                    out.append(fv)
            except (ValueError, TypeError):
                pass
        return out

    n_total = len(rows)
    n_ok = sum(1 for r in rows if not r.get("error", "").strip())

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "mean_fitness": _mean("fitness"),
        "mean_inlier_rmse": _mean("inlier_rmse"),
        "mean_init_abs_err": _mean("init_abs_err"),
        "mean_init_l2_err": _mean("init_l2_err"),
        "mean_abs_err": _mean("mean_abs_err"),
        "mean_l2_err": _mean("mean_l2_err"),
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "mean_fitness", "mean_inlier_rmse",
    "mean_init_abs_err", "mean_init_l2_err",
    "mean_abs_err", "mean_l2_err",
]

_RUN_FIELDS = [
    "name", "repeat", "split_seed", "dataset", "root",
    "icp_threshold", "init_from_fpfh", "eval_all_pairs", "fit_all_pairs",
    "n_total", "n_ok",
] + _METRIC_KEYS + ["error"]

_SUMMARY_FIELDS = [
    "name", "dataset", "root", "icp_threshold", "init_from_fpfh",
    "eval_all_pairs", "fit_all_pairs",
    "n_repeats", "n_total", "n_ok",
] + [
    col
    for key in _METRIC_KEYS
    for col in (key, key.replace("mean_", "std_", 1) if key.startswith("mean_") else f"{key}_std")
] + ["error"]


def _aggregate_repeats(repeat_aggs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Given a list of per-repeat aggregate dicts, compute overall mean and std."""
    out: Dict[str, Any] = {}
    ok_repeats = [r for r in repeat_aggs if not r.get("_error")]
    out["n_repeats"] = len(ok_repeats)
    out["n_total"] = int(np.mean([r["n_total"] for r in ok_repeats])) if ok_repeats else 0
    out["n_ok"] = int(np.mean([r["n_ok"] for r in ok_repeats])) if ok_repeats else 0
    for key in _METRIC_KEYS:
        vals = [r[key] for r in ok_repeats if not math.isnan(r.get(key, float("nan")))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
        std_key = key.replace("mean_", "std_", 1) if key.startswith("mean_") else f"{key}_std"
        out[std_key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return out


def run_sweep(
    configs: List[RunConfig],
    out_csv: str,
    workers: Optional[int] = None,
    global_repeats: Optional[int] = None,
) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    runs_csv = os.path.splitext(out_csv)[0] + "_runs.csv"

    summary_rows: List[Dict[str, Any]] = []
    all_run_rows: List[Dict[str, Any]] = []

    for i, cfg in enumerate(configs):
        n_repeats = global_repeats if global_repeats is not None else cfg.n_repeats
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}]  {cfg.name}  ({cfg.dataset} / {cfg.root})  repeats={n_repeats}")
        print(f"{'='*70}")

        repeat_aggs: List[Dict[str, Any]] = []

        for rep in range(n_repeats):
            seed_offset = rep  # repeat 0 → base seed, repeat 1 → base+1, ...
            split_seed = cfg.split_seed + seed_offset
            print(f"  -- repeat {rep+1}/{n_repeats}  (split_seed={split_seed}) --")

            run_row: Dict[str, Any] = {
                "name": cfg.name,
                "repeat": rep,
                "split_seed": split_seed,
                "dataset": cfg.dataset,
                "root": cfg.root,
                "icp_threshold": cfg.icp_threshold,
                "init_from_fpfh": cfg.init_from_fpfh,
                "eval_all_pairs": cfg.eval_all_pairs,
                "fit_all_pairs": cfg.fit_all_pairs,
                "error": "",
            }

            try:
                args = _cfg_to_args(cfg, workers, seed_offset=seed_offset, repeat_idx=rep)
                csv_path = batch_run_icp_full_eval(args)
                agg = _read_aggregate(csv_path)
                run_row.update(agg)
                agg["_error"] = False
            except Exception as e:
                tb = traceback.format_exc()
                print(f"    ERROR: {e}\n{tb}")
                run_row["error"] = str(e)
                agg = {k: float("nan") for k in _METRIC_KEYS}
                agg["n_total"] = 0
                agg["n_ok"] = 0
                agg["_error"] = True

            repeat_aggs.append(agg)
            all_run_rows.append(run_row)

            # Write runs CSV incrementally
            with open(runs_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_RUN_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for r in all_run_rows:
                    writer.writerow(r)

        # Aggregate across repeats
        agg_across = _aggregate_repeats(repeat_aggs)
        errors = [r["error"] for r in all_run_rows if r["name"] == cfg.name and r.get("error")]

        summary_row: Dict[str, Any] = {
            "name": cfg.name,
            "dataset": cfg.dataset,
            "root": cfg.root,
            "icp_threshold": cfg.icp_threshold,
            "init_from_fpfh": cfg.init_from_fpfh,
            "eval_all_pairs": cfg.eval_all_pairs,
            "fit_all_pairs": cfg.fit_all_pairs,
            "error": "; ".join(errors) if errors else "",
        }
        summary_row.update(agg_across)
        summary_rows.append(summary_row)

        # Write summary CSV incrementally
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)

        print(f"  -> results written to {out_csv}  (runs: {runs_csv})")

    # Final console table
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE  ({len(summary_rows)} configs)")
    print(f"{'='*70}")
    col_w = 30
    hdr = (
        f"{'name':<{col_w}}  {'mean_abs_err':>22}  {'mean_l2_err':>20}  "
        f"{'mean_fitness':>12}  {'n_ok':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        if r.get("error") and r.get("n_repeats", 0) == 0:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
        else:
            abs_val = r.get("mean_abs_err", float("nan"))
            abs_std = r.get("std_abs_err", float("nan"))
            l2_val  = r.get("mean_l2_err", float("nan"))
            l2_std  = r.get("std_l2_err", float("nan"))
            fit_val = r.get("mean_fitness", float("nan"))
            abs_str = f"{abs_val:.6f} ± {abs_std:.6f}" if not math.isnan(abs_std) else f"{abs_val:.6f}"
            l2_str  = f"{l2_val:.6f} ± {l2_std:.6f}" if not math.isnan(l2_std) else f"{l2_val:.6f}"
            print(
                f"{r['name']:<{col_w}}  "
                f"{abs_str:>22}  "
                f"{l2_str:>20}  "
                f"{fit_val:>12.4f}  "
                f"{r.get('n_ok', 0):>6}"
            )
    print(f"\nAggregate CSV: {out_csv}")
    print(f"Per-run CSV:   {runs_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep ICP full-eval over multiple configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default="results/icp_sweep.csv",
        help="Path for the aggregate comparison CSV",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override num_workers for all runs (default: use per-config value)",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Run only the named configs (subset of CONFIGS)",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Override n_repeats for all configs (each repeat uses split_seed + repeat_index)",
    )
    cli = parser.parse_args()

    chosen = CONFIGS
    if cli.names:
        name_set = set(cli.names)
        chosen = [c for c in CONFIGS if c.name in name_set]
        missing = name_set - {c.name for c in chosen}
        if missing:
            print(f"Warning: unknown config names: {missing}")

    run_sweep(chosen, cli.out, workers=cli.workers, global_repeats=cli.repeats)
