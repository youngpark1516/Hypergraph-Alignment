import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project.dataset import AlignmentDataset, FullAlignmentDataset
from project.config.whnn_config import get_config
from project.models.hypergnn.loss import (
    ClassificationLoss,
    EdgeLoss,
    SpectralMatchingLoss,
    TransformationLoss,
)
from project.models.whnn import WHNN
from project.trainers.whnn_trainer import Trainer


def build_dataloaders(args):
    root = Path(args.root)
    print("Loading data from:", root)
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
    return WHNN(args)


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
        return
    trainer.train(resume=False, start_epoch=0, best_reg_recall=0.0, best_F1=0.0)


if __name__ == "__main__":
    main()
