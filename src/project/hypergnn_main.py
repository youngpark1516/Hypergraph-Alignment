import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project.dataset import AlignmentDataset, FullAlignmentDataset
from project.config.hypergnn_config import get_config
from project.models.hypergnn.loss import (
    ClassificationLoss,
    EdgeLoss,
    SpectralMatchingLoss,
    TransformationLoss,
)
from project.models.hypergnn.model import HGNN, HyperGCT
from project.trainers.hypergnn_trainer import Trainer


def build_dataloaders(args):
    root = Path(args.root)
    print('Loading data from:', root)
    eval_only = bool(args.eval_snapshot)
    files = sorted(root.glob("**/*.npz"))
    if not files:
        raise ValueError(f"No .npz files found under {root}")

    rng = np.random.default_rng(args.split_seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    split = int(len(files) * (1.0 - args.val_ratio))
    train_files = [files[i] for i in indices[:split]]
    val_files = [files[i] for i in indices[split:]]
    if (args.eval_snapshot or args.mode == "test") and args.eval_num > 0:
        if not val_files:
            raise ValueError(f"No .npz files available under {root} for evaluation.")
        eval_count = min(args.eval_num, len(val_files))
        eval_rng = np.random.default_rng(args.eval_seed)
        chosen_idx = eval_rng.choice(len(val_files), size=eval_count, replace=False)
        val_files = [val_files[i] for i in chosen_idx]
        print(
            f"eval_num enabled, randomly selected {len(val_files)} validation files "
            f"(eval_seed={args.eval_seed})"
        )

    include_gt_trans = not args.skip_gt_trans
    if args.full_data:
        print("Using FullAlignmentDataset")
        train_set = None if eval_only else FullAlignmentDataset(
            train_files,
            num_corr=args.num_node,
            use_features=args.use_features,
            feature_dim=args.feature_dim,
            include_gt_trans=include_gt_trans,
            seed=args.split_seed,
            force_add_true_pairs=True,
            max_corr=args.max_corr,
        )
        val_set = FullAlignmentDataset(
            val_files,
            num_corr=args.num_node,
            use_features=args.use_features,
            feature_dim=args.feature_dim,
            include_gt_trans=include_gt_trans,
            seed=args.split_seed + 1,
            force_add_true_pairs=True,
            max_corr=args.max_corr,
        )
    else:
        train_set = None if eval_only else AlignmentDataset(
            train_files,
            num_corr=args.num_node,
            use_features=args.use_features,
            feature_dim=args.feature_dim,
            neg_ratio=args.neg_ratio,
            include_gt_trans=include_gt_trans,
            seed=args.split_seed,
        )
        val_set = AlignmentDataset(
            val_files,
            num_corr=args.num_node,
            use_features=args.use_features,
            feature_dim=args.feature_dim,
            neg_ratio=args.neg_ratio,
            include_gt_trans=include_gt_trans,
            seed=args.split_seed + 1,
        )

    train_loader = None if eval_only else DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if hasattr(val_set, "summarize_true_pair_coverage"):
        val_cov = val_set.summarize_true_pair_coverage()
        print("Val true-pair stats:")
        for key in sorted(val_cov.keys()):
            value = val_cov[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    elif hasattr(val_set, "summarize_selection_stats"):
        val_sel = val_set.summarize_selection_stats()
        print("Val selection stats:")
        for key in sorted(val_sel.keys()):
            value = val_sel[key]
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    return train_loader, val_loader


def build_model(args):
    model = HyperGCT(args)
    corr_feat_dim = 6 + (2 * args.feature_dim if args.use_features else 0)
    if corr_feat_dim != 6:
        model.encoder = HGNN(n_emb_dims=model.num_channels, in_channel=corr_feat_dim, pooling_layer_idx=args.pooling_layer_idx)
    return model


def build_optimizer(args, model):
    if args.optimizer == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.scheduler == "ExpLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def run_eval_from_args(args) -> dict:
    """Run a single eval pass given a fully-constructed args namespace.

    Calling code is responsible for setting all required fields.
    Returns the full metrics dict from trainer.evaluate().
    """
    args.generate = True
    args.mode = "test"
    args.pretrain = args.eval_snapshot
    args.batch_size = 1

    # Parse pooling_layer_idx if it came in as a string
    if isinstance(args.pooling_layer_idx, str):
        text = args.pooling_layer_idx.strip()
        args.pooling_layer_idx = (
            [int(v.strip()) for v in text.split(",") if v.strip()]
            if text else []
        )
    elif isinstance(args.pooling_layer_idx, (int, float)):
        args.pooling_layer_idx = [int(args.pooling_layer_idx)]

    _, val_loader = build_dataloaders(args)

    model = build_model(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=args.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=args.balanced),
        "HypergraphLoss": EdgeLoss(),
        "TransformationLoss": TransformationLoss(args.re_thre, args.te_thre),
    }
    metric_weight = {
        "ClassificationLoss": args.weight_classification,
        "SpectralMatchingLoss": args.weight_spectralmatching,
        "HypergraphLoss": args.weight_hypergraph,
        "TransformationLoss": args.weight_transformation,
    }

    args.model = model
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.evaluate_metric = evaluate_metric
    args.metric_weight = metric_weight
    args.train_loader = None
    args.val_loader = val_loader

    trainer = Trainer(args)
    res = trainer.evaluate(epoch=0)

    # Clean up GPU memory and TensorBoard writer so repeated calls don't accumulate resources
    trainer.writer.close()
    del trainer.model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def main():
    args = get_config()
    args.generate = bool(args.eval_snapshot)
    if args.eval_snapshot:
        args.mode = "test"
        args.pretrain = args.eval_snapshot
    if args.mode == "test":
        args.batch_size = 1
    if not args.mode == "test":
        os.makedirs(args.snapshot_dir, exist_ok=True)
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.tboard_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)

    model = build_model(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    evaluate_metric = {
        "ClassificationLoss": ClassificationLoss(balanced=args.balanced),
        "SpectralMatchingLoss": SpectralMatchingLoss(balanced=args.balanced),
        "HypergraphLoss": EdgeLoss(),
        "TransformationLoss": TransformationLoss(args.re_thre, args.te_thre),
    }
    metric_weight = {
        "ClassificationLoss": args.weight_classification,
        "SpectralMatchingLoss": args.weight_spectralmatching,
        "HypergraphLoss": args.weight_hypergraph,
        "TransformationLoss": args.weight_transformation,
    }

    args.model = model
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.evaluate_metric = evaluate_metric
    args.metric_weight = metric_weight
    args.train_loader = train_loader
    args.val_loader = val_loader

    trainer = Trainer(args)
    if args.eval_snapshot:
        res = trainer.evaluate(epoch=0)
        print(
            f'Evaluation: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} '
            f'Graph Loss {res["graph_loss"]:.2f} F1 {res["f1"]:.2f} Recall {res["reg_recall"]:.2f}'
        )

        # --- save results to CSV ---
        snapshot_stem = Path(args.eval_snapshot).stem
        default_csv = f"results/{snapshot_stem}_{args.dataset}_eval.csv"
        out_csv = args.eval_out or default_csv
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

        row = {
            "snapshot": args.eval_snapshot,
            "dataset": args.dataset,
            "root": args.root,
            "split_seed": args.split_seed,
            "eval_num": args.eval_num,
            "generation_method": args.generation_method,
            **{k: v for k, v in res.items()},
        }
        write_header = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Eval results saved to {out_csv}")
        return
    trainer.train(resume=False, start_epoch=0, best_reg_recall=0.0, best_F1=0.0)


if __name__ == "__main__":
    main()
