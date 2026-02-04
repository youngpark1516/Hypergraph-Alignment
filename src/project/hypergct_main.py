import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project.config.hypergct_config import get_config
from project.models.hypergct.loss import (
    ClassificationLoss,
    EdgeLoss,
    SpectralMatchingLoss,
    TransformationLoss,
)
from project.models.hypergct.model import HGNN, HyperGCT
from project.trainers.hypergct_trainer import Trainer
from project.dataset import AlignmentDataset

def build_dataloaders(args):
    root = Path(args.root)
    print('Loading data from:', root)
    files = sorted(root.glob("**/*.npz"))
    if not files:
        raise ValueError(f"No .npz files found under {root}")

    rng = np.random.default_rng(args.split_seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)
    split = int(len(files) * (1.0 - args.val_ratio))
    train_files = [files[i] for i in indices[:split]]
    val_files = [files[i] for i in indices[split:]]

    include_gt_trans = not args.skip_gt_trans
    train_set = AlignmentDataset(
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

    train_loader = DataLoader(
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
    return train_loader, val_loader


def build_model(args):
    model = HyperGCT(args)
    corr_feat_dim = 6 + (2 * args.feature_dim if args.use_features else 0)
    if corr_feat_dim != 6:
        model.encoder = HGNN(n_emb_dims=model.num_channels, in_channel=corr_feat_dim)
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


def main():
    args = get_config()
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
    trainer.train(resume=False, start_epoch=0, best_reg_recall=0.0, best_F1=0.0)


if __name__ == "__main__":
    main()
