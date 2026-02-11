from torch.utils.data import Dataset
import numpy as np
import torch

from project.utils.build_correspondences import fpfh_knn_correspondences

class AlignmentDataset(Dataset):
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
        feat0 = None
        feat1 = None
        if self.use_features:
            if "features0" not in data or "features1" not in data:
                raise ValueError(f"features0/features1 missing in {path}")
            feat0 = data["features0"].astype(np.float32)
            feat1 = data["features1"].astype(np.float32)
            if feat0.shape[1] != self.feature_dim or feat1.shape[1] != self.feature_dim:
                raise ValueError(f"feature_dim mismatch in {path}")

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
        src_indices = torch.from_numpy(all_src_idx.astype(np.int64))
        tgt_indices = torch.from_numpy(all_tgt_idx.astype(np.int64))

        corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=1)
        if self.use_features:
            corr_pos = np.concatenate([corr_pos, feat0[all_src_idx], feat1[all_tgt_idx]], axis=1)

        corr_pos = torch.from_numpy(corr_pos)
        src_keypts = torch.from_numpy(src_keypts)
        tgt_keypts = torch.from_numpy(tgt_keypts)
        src_normal = torch.from_numpy(src_normal)
        tgt_normal = torch.from_numpy(tgt_normal)
        gt_labels = torch.from_numpy(gt_labels)

        assert not self.include_gt_trans, "include_gt_trans is not supported; expected 9-item batch"
        return (
            corr_pos,
            src_keypts,
            tgt_keypts,
            src_normal,
            tgt_normal,
            src_indices,
            tgt_indices,
            gt_labels,
            str(path),
        )
