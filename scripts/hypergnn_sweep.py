"""Sweep HyperGNN eval_snapshot mode over multiple configurations with repeats.

Edit the CONFIGS list below, then run:
    python scripts/hypergnn_sweep.py [--out results/hypergnn_sweep.csv] [--repeats 5]

Each config is run n_repeats times (varying split_seed).
Results are aggregated into a single CSV with mean ± std across repeats,
plus a _runs.csv with every individual repeat.
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
# Project imports
# ---------------------------------------------------------------------------
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_SRC_ROOT = str(Path(_REPO_ROOT) / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from src.project.hypergnn_main import run_eval_from_args  # noqa: E402
from src.project.utils.io import load_yaml               # noqa: E402


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# Base defaults loaded from the eval YAML — overridden per-config below
_BASE_YAML = "src/project/config/hypergnn_partnet_eval.yaml"

# All fields that are NOT in the eval yaml but are needed by the trainer
_EXTRA_DEFAULTS = dict(
    # optimizer / scheduler
    optimizer="ADAM",
    lr=1e-4,
    weight_decay=1e-6,
    momentum=0.9,
    scheduler="ExpLR",
    scheduler_gamma=0.99,
    scheduler_interval=1,
    max_epoch=50,
    training_max_iter=3500,
    val_max_iter=1000,
    evaluate_interval=1,
    # loss weights
    balanced=False,
    weight_classification=1.0,
    weight_spectralmatching=1.0,
    weight_hypergraph=1.0,
    weight_transformation=0.0,
    transformation_loss_start_epoch=0,
    re_thre=15.0,
    te_thre=30.0,
    # data
    augment_axis=3,
    augment_rotation=1.0,
    augment_translation=0.5,
    full_data=False,
    # misc
    gpu_mode=True,
    verbose=True,
    pretrain="",
    weights_fixed=False,
    force_all_labels=False,
    skip_gt_trans=True,
    split_seed=0,
    eval_num=0,
    eval_seed=0,
    generation_method="spectral-2",
    generation_min_score=0.4,
    generation_min_confidence=0.9,
    use_wandb=False,
    wandb_project="3d-alignment",
    wandb_entity="",
    wandb_run_name=None,
    # snapshot dirs (unused in eval but required by Trainer init)
    exp_id="sweep_eval",
    snapshot_dir="snapshot/sweep_eval",
    tboard_dir="tensorboard/sweep_eval",
    save_dir="snapshot/sweep_eval/models",
    # matching output dir
    matching_dir="results/sweep_matching",
    snapshot_interval=1,
)


@dataclass
class EvalConfig:
    """One HyperGNN evaluation run."""
    name: str
    snapshot: str           # path to .pkl
    dataset: str
    root: str

    pooling_layer_idx: str = "-1"   # e.g. "0,1" or "0,1,2,3,4,5"
    generation_method: str = "spectral-2"
    config_yaml: str = _BASE_YAML

    # split / repeat control
    split_seed: int = 0
    n_repeats: int = 10

    # any extra overrides
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EDIT THIS LIST to add / remove runs
# ============================================================
CONFIGS: List[EvalConfig] = [
    # ---- PartNet rigid ----
    # EvalConfig(
    #     name="WHNN-first-PN-R",
    #     snapshot="snapshot/WHNN-first-PN-R/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_rigid",
    #     pooling_layer_idx="0",
    # ),
    EvalConfig(
        name="WHNN-first2-PN-R",
        snapshot="snapshot/WHNN-first2-PN-R/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
        pooling_layer_idx="0,1",
    ),
    EvalConfig(
        name="WHNN-last2-PN-R",
        snapshot="snapshot/WHNN-last2-PN-R/models/model_best.pkl",  # inferred from pattern
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
        pooling_layer_idx="4,5",
    ),
    EvalConfig(
        name="WHNN-firstlast-PN-R",
        snapshot="snapshot/WHNN-firstlast-PN-R/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
        pooling_layer_idx="0,5",
    ),
    EvalConfig(
        name="WHNN-last-PN-R",
        snapshot="snapshot/WHNN-last-PN-R/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
        pooling_layer_idx="5",
    ),
    # EvalConfig(
    #     name="WHNN-full-PN-R",
    #     snapshot="snapshot/WHNN-full-PN-R/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_rigid",
    #     pooling_layer_idx="0,1,2,3,4,5",
    # ),
    EvalConfig(
        name="HCT-PN-R",
        snapshot="snapshot/HCT-PN-R/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_rigid",
        pooling_layer_idx="-1",
    ),
    # ---- PartNet affine 1.5 ----
    # EvalConfig(
    #     name="WHNN-first2-PN-1.5",
    #     snapshot="snapshot/WHNN-first2-PN-1.5/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_affine_1-5",
    #     pooling_layer_idx="0,1",
    # ),
    EvalConfig(
        name="WHNN-last2-PN-1.5",
        snapshot="snapshot/WHNN-last2-PN-1.5/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
        pooling_layer_idx="4,5",
    ),
    EvalConfig(
        name="WHNN-firstlast-PN-1.5",
        snapshot="snapshot/WHNN-firstlast-PN-1.5/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
        pooling_layer_idx="0,5",
    ),
    EvalConfig(
        name="WHNN-last-PN-1.5",
        snapshot="snapshot/WHNN-last-PN-1.5/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
        pooling_layer_idx="5",
    ),
    EvalConfig(
        name="WHNN-first-PN-1.5",
        snapshot="snapshot/WHNN-first-PN-1.5/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
        pooling_layer_idx="0",
    ),
    # EvalConfig(
    #     name="WHNN-full-PN-1.5",
    #     snapshot="snapshot/WHNN-full-PN-1.5/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_affine_1-5",
    #     pooling_layer_idx="0,1,2,3,4,5",
    # ),
    EvalConfig(
        name="HCT-PN-1.5",
        snapshot="snapshot/HCT-PN-1.5/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_1-5",
        pooling_layer_idx="-1",
    ),
    # ---- PartNet affine 2.0 ----
    # EvalConfig(
    #     name="WHNN-first-PN-2.0",
    #     snapshot="snapshot/WHNN-first-PN-2.0/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_affine_2-0",
    #     pooling_layer_idx="0",
    # ),

    EvalConfig(
        name="WHNN-first2-PN-2.0",
        snapshot="snapshot/WHNN-first2-PN-2.0/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_2-0",
        pooling_layer_idx="0,1",
    ),

    # EvalConfig(
    #     name="WHNN-last2-PN-2.0",
    #     snapshot="snapshot/WHNN-last2-PN-2.0/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_affine_2-0",
    #     pooling_layer_idx="4,5",
    # ),
    EvalConfig(
        name="WHNN-firstlast-PN-2.0",
        snapshot="snapshot/WHNN-firstlast-PN-2.0/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_2-0",
        pooling_layer_idx="0,5",
    ),
    EvalConfig(
        name="WHNN-last-PN-2.0",
        snapshot="snapshot/WHNN-last-PN-2.0/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_2-0",
        pooling_layer_idx="5",
    ),
    # EvalConfig(
    #     name="WHNN-full-PN-2.0",
    #     snapshot="snapshot/WHNN-full-PN-2.0/models/model_best.pkl",
    #     dataset="partnet",
    #     root="data/partnet/fpfh_affine_2-0",
    #     pooling_layer_idx="0,1,2,3,4,5",
    # ),
    EvalConfig(
        name="HCT-PN-2.0",
        snapshot="snapshot/HCT-PN-2.0/models/model_best.pkl",
        dataset="partnet",
        root="data/partnet/fpfh_affine_2-0",
        pooling_layer_idx="-1",
    ),
    # ---- FAUST ----
    # EvalConfig(
    #     name="WHNN-full-FAUST",
    #     snapshot="snapshot/WHNN-full-FAUST/models/model_best.pkl",
    #     dataset="faust",
    #     root="data/faust/fpfh",
    #     pooling_layer_idx="0,1,2,3,4,5",
    # ),
    EvalConfig(
        name="WHNN-first2-FAUST",
        snapshot="snapshot/WHNN-first2-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="0,1",
    ),
    EvalConfig(
        name="WHNN-last2-FAUST",
        snapshot="snapshot/WHNN-last2-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="4,5",
    ),
    EvalConfig(
        name="WHNN-firstlast-FAUST",
        snapshot="snapshot/WHNN-firstlast-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="0,5",
    ),
    EvalConfig(
        name="WHNN-last-FAUST",
        snapshot="snapshot/WHNN-last-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="5",
    ),
    EvalConfig(
        name="WHNN-first-FAUST",
        snapshot="snapshot/WHNN-first-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="0",
    ),
    EvalConfig(
        name="HCT-FAUST",
        snapshot="snapshot/HCT-FAUST/models/model_best.pkl",
        dataset="faust",
        root="data/faust/fpfh",
        pooling_layer_idx="-1",
    ),
]
# ============================================================


# ---------------------------------------------------------------------------
# Namespace builder
# ---------------------------------------------------------------------------

def _build_namespace(cfg: EvalConfig, seed_offset: int = 0) -> argparse.Namespace:
    """Merge YAML defaults → _EXTRA_DEFAULTS → per-config overrides into a Namespace."""
    base = {}

    # 1. Load YAML
    yaml_path = Path(cfg.config_yaml)
    if yaml_path.exists():
        yaml_data = load_yaml(yaml_path)
        if isinstance(yaml_data, dict):
            base.update(yaml_data)

    # 2. Extra defaults for fields not typically in the YAML
    for k, v in _EXTRA_DEFAULTS.items():
        base.setdefault(k, v)

    # 3. Per-config overrides
    split_seed = cfg.split_seed + seed_offset
    base.update(
        dataset=cfg.dataset,
        root=cfg.root,
        eval_snapshot=cfg.snapshot,
        pooling_layer_idx=cfg.pooling_layer_idx,
        generation_method=cfg.generation_method,
        split_seed=split_seed,
        eval_seed=cfg.split_seed + seed_offset,
        exp_id=f"sweep_{cfg.name}_seed{split_seed}",
        snapshot_dir=f"snapshot/sweep_{cfg.name}_seed{split_seed}",
        tboard_dir=f"tensorboard/sweep_{cfg.name}_seed{split_seed}",
        save_dir=f"snapshot/sweep_{cfg.name}_seed{split_seed}/models",
        matching_dir=f"results/sweep_matching/{cfg.name}_seed{split_seed}",
    )
    base.update(cfg.extra)

    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Metric keys
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "sm_loss", "class_loss", "graph_loss", "f1", "reg_recall",
    "precision", "recall", "re", "te",
    "init_abs_err", "init_l2_err",
    "selected_abs_err", "selected_l2_err",
]

_RUN_FIELDS = [
    "name", "repeat", "split_seed", "snapshot", "dataset", "root",
    "pooling_layer_idx", "generation_method",
] + _METRIC_KEYS + ["n_eval", "n_selected", "error"]

_SUMMARY_FIELDS = (
    ["name", "snapshot", "dataset", "root", "pooling_layer_idx", "generation_method", "n_repeats"]
    + [
        col
        for key in _METRIC_KEYS
        for col in (key, "std_" + key[len("mean_"):] if key.startswith("mean_") else key + "_std")
    ]
    + ["error"]
)


def _aggregate_repeats(repeat_res: List[Dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    ok = [r for r in repeat_res if not r.get("_error")]
    out["n_repeats"] = len(ok)

    for key in _METRIC_KEYS + ["n_eval", "n_selected"]:
        vals = []
        for r in ok:
            v = r.get(key, float("nan"))
            try:
                fv = float(v)
                if not math.isnan(fv):
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
        out[key] = float(np.mean(vals)) if vals else float("nan")
        std_key = "std_" + key[len("mean_"):] if key.startswith("mean_") else key + "_std"
        out[std_key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")

    return out


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    configs: List[EvalConfig],
    out_csv: str,
    global_repeats: Optional[int] = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    runs_csv = os.path.splitext(out_csv)[0] + "_runs.csv"

    summary_rows: List[Dict] = []
    all_run_rows: List[Dict] = []

    for i, cfg in enumerate(configs):
        n_repeats = global_repeats if global_repeats is not None else cfg.n_repeats
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(configs)}]  {cfg.name}  repeats={n_repeats}")
        print(f"  snapshot: {cfg.snapshot}")
        print(f"  {cfg.dataset} / {cfg.root}  pooling={cfg.pooling_layer_idx}")
        print(f"{'='*70}")

        repeat_res: List[Dict] = []

        for rep in range(n_repeats):
            split_seed = cfg.split_seed + rep
            print(f"\n  -- repeat {rep+1}/{n_repeats}  (split_seed={split_seed}) --")

            run_row: Dict[str, Any] = {
                "name": cfg.name,
                "repeat": rep,
                "split_seed": split_seed,
                "snapshot": cfg.snapshot,
                "dataset": cfg.dataset,
                "root": cfg.root,
                "pooling_layer_idx": cfg.pooling_layer_idx,
                "generation_method": cfg.generation_method,
                "error": "",
            }

            try:
                ns = _build_namespace(cfg, seed_offset=rep)
                res = run_eval_from_args(ns)
                run_row.update({k: res.get(k, float("nan")) for k in _METRIC_KEYS})
                run_row["n_eval"] = res.get("n_eval", 0)
                run_row["n_selected"] = res.get("n_selected", 0)
                res["_error"] = False
                repeat_res.append(res)
                print(
                    f"    sel_abs={res.get('selected_abs_err', float('nan')):.6f}  "
                    f"sel_l2={res.get('selected_l2_err', float('nan')):.6f}  "
                    f"f1={res.get('f1', float('nan')):.4f}"
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"    ERROR: {e}\n{tb}")
                run_row["error"] = str(e)
                err_res = {k: float("nan") for k in _METRIC_KEYS}
                err_res["n_eval"] = 0
                err_res["n_selected"] = 0
                err_res["_error"] = True
                repeat_res.append(err_res)

            all_run_rows.append(run_row)

            # Write runs CSV incrementally
            with open(runs_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_RUN_FIELDS, extrasaction="ignore")
                writer.writeheader()
                for r in all_run_rows:
                    writer.writerow(r)

        # Aggregate across repeats
        agg = _aggregate_repeats(repeat_res)
        errors = [r["error"] for r in all_run_rows if r["name"] == cfg.name and r.get("error")]

        summary_row: Dict[str, Any] = {
            "name": cfg.name,
            "snapshot": cfg.snapshot,
            "dataset": cfg.dataset,
            "root": cfg.root,
            "pooling_layer_idx": cfg.pooling_layer_idx,
            "generation_method": cfg.generation_method,
            "error": "; ".join(errors) if errors else "",
        }
        summary_row.update(agg)
        summary_rows.append(summary_row)

        # Write summary CSV incrementally
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)

        print(f"\n  -> results written to {out_csv}  (runs: {runs_csv})")

    # Final console table
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE  ({len(summary_rows)} configs)")
    print(f"{'='*70}")
    col_w = 26
    hdr = (
        f"{'name':<{col_w}}  {'sel_abs_err':>24}  {'sel_l2_err':>22}  {'f1':>10}  {'n_rep':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in summary_rows:
        if r.get("error") and r.get("n_repeats", 0) == 0:
            print(f"{r['name']:<{col_w}}  ERROR: {r['error']}")
            continue
        abs_v = r.get("selected_abs_err", float("nan"))
        abs_s = r.get("selected_abs_err_std", float("nan"))
        l2_v  = r.get("selected_l2_err", float("nan"))
        l2_s  = r.get("selected_l2_err_std", float("nan"))
        f1_v  = r.get("f1", float("nan"))
        f1_s  = r.get("f1_std", float("nan"))
        abs_str = f"{abs_v:.6f} ± {abs_s:.6f}" if not math.isnan(abs_s) else f"{abs_v:.6f}"
        l2_str  = f"{l2_v:.6f} ± {l2_s:.6f}"  if not math.isnan(l2_s)  else f"{l2_v:.6f}"
        f1_str  = f"{f1_v:.4f} ± {f1_s:.4f}"  if not math.isnan(f1_s)  else f"{f1_v:.4f}"
        print(
            f"{r['name']:<{col_w}}  "
            f"{abs_str:>24}  "
            f"{l2_str:>22}  "
            f"{f1_str:>10}  "
            f"{r.get('n_repeats', 0):>5}"
        )
    print(f"\nAggregate CSV: {out_csv}")
    print(f"Per-run CSV:   {runs_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep HyperGNN eval_snapshot over multiple configs with repeats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", default="results/hypergnn_sweep.csv",
        help="Path for the aggregate comparison CSV",
    )
    parser.add_argument(
        "--repeats", type=int, default=None,
        help="Override n_repeats for all configs (each repeat uses split_seed + repeat_index)",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Run only the named configs (subset of CONFIGS)",
    )
    cli = parser.parse_args()

    chosen = CONFIGS
    if cli.names:
        name_set = set(cli.names)
        chosen = [c for c in CONFIGS if c.name in name_set]
        missing = name_set - {c.name for c in chosen}
        if missing:
            print(f"Warning: unknown config names: {missing}")

    run_sweep(chosen, cli.out, global_repeats=cli.repeats)
