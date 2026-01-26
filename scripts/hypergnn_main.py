import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from project.config.hypergnn_config import get_config
from project.models.hypergnn.loss import (
    ClassificationLoss,
    EdgeLoss,
    SpectralMatchingLoss,
    TransformationLoss,
)
from project.models.hypergnn.model import HGNN, MethodName
from project.runners.hypergnn_trainer import Trainer


class PartNetRigidDataset(Dataset):
    def __init__(
        self,
        files,
        num_corr,
        use_features,
        feature_dim,
        neg_ratio,
        include_gt_trans,
        seed,
    ):
        self.files = files
        self.num_corr = num_corr
        self.use_features = use_features
        self.feature_dim = feature_dim
        self.neg_ratio = neg_ratio
        self.include_gt_trans = include_gt_trans
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def _sample_indices(self, total, rng):
        if self.num_corr is None or self.num_corr <= 0:
            return np.arange(total, dtype=np.int64)
        replace = total < self.num_corr
        return rng.choice(total, size=self.num_corr, replace=replace).astype(np.int64)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        xyz0 = data["xyz0"].astype(np.float32)
        xyz1 = data["xyz1"].astype(np.float32)
        normal0 = data["normal0"].astype(np.float32)
        normal1 = data["normal1"].astype(np.float32)
        corres = data["corres"].astype(np.int64)

        if corres.shape[0] != xyz0.shape[0]:
            raise ValueError(f"corres size mismatch in {path}")
        if corres.max() >= xyz1.shape[0]:
            raise ValueError(f"corres index out of range in {path}")

        rng = np.random.default_rng(self.seed + idx)
        src_idx = self._sample_indices(corres.shape[0], rng)
        tgt_idx = corres[src_idx]

        num_pos = src_idx.shape[0]
        num_neg = int(round(num_pos * self.neg_ratio))
        if num_neg > 0 and xyz1.shape[0] > 1:
            neg_src_idx = rng.choice(src_idx, size=num_neg, replace=True)
            neg_tgt_idx = rng.integers(0, xyz1.shape[0], size=num_neg)
            bad = neg_tgt_idx == corres[neg_src_idx]
            while bad.any():
                neg_tgt_idx[bad] = rng.integers(0, xyz1.shape[0], size=bad.sum())
                bad = neg_tgt_idx == corres[neg_src_idx]
            all_src_idx = np.concatenate([src_idx, neg_src_idx])
            all_tgt_idx = np.concatenate([tgt_idx, neg_tgt_idx])
            gt_labels = np.concatenate(
                [np.ones(num_pos, dtype=np.float32), np.zeros(num_neg, dtype=np.float32)]
            )
        else:
            all_src_idx = src_idx
            all_tgt_idx = tgt_idx
            gt_labels = np.ones(num_pos, dtype=np.float32)

        src_keypts = xyz0[all_src_idx]
        tgt_keypts = xyz1[all_tgt_idx]
        src_normal = normal0[all_src_idx]
        tgt_normal = normal1[all_tgt_idx]

        corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=1)
        if self.use_features:
            if "features0" not in data or "features1" not in data:
                raise ValueError(f"features0/features1 missing in {path}")
            feat0 = data["features0"].astype(np.float32)
            feat1 = data["features1"].astype(np.float32)
            if feat0.shape[1] != self.feature_dim or feat1.shape[1] != self.feature_dim:
                raise ValueError(f"feature_dim mismatch in {path}")
            corr_pos = np.concatenate([corr_pos, feat0[all_src_idx], feat1[all_tgt_idx]], axis=1)

        corr_pos = torch.from_numpy(corr_pos)
        src_keypts = torch.from_numpy(src_keypts)
        tgt_keypts = torch.from_numpy(tgt_keypts)
        src_normal = torch.from_numpy(src_normal)
        tgt_normal = torch.from_numpy(tgt_normal)
        gt_labels = torch.from_numpy(gt_labels)

        if self.include_gt_trans and "gt_trans" in data:
            gt_trans = data["gt_trans"].astype(np.float32)
            gt_trans = torch.from_numpy(gt_trans)
            return (
                corr_pos,
                src_keypts,
                tgt_keypts,
                src_normal,
                tgt_normal,
                gt_trans,
                gt_labels,
            )

        return (
            corr_pos,
            src_keypts,
            tgt_keypts,
            src_normal,
            tgt_normal,
            gt_labels,
        )


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
    train_set = PartNetRigidDataset(
        train_files,
        num_corr=args.num_node,
        use_features=args.use_features,
        feature_dim=args.feature_dim,
        neg_ratio=args.neg_ratio,
        include_gt_trans=include_gt_trans,
        seed=args.split_seed,
    )
    val_set = PartNetRigidDataset(
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
    model = MethodName(args)
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
