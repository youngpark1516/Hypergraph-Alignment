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

    def summarize_selection_stats(self):
        if len(self.files) == 0:
            return {
                "num_samples": 0,
                "mean_src_selected_pct": 0.0,
                "mean_tgt_selected_pct": 0.0,
            }

        src_pcts = []
        tgt_pcts = []
        for idx, path in enumerate(self.files):
            data = np.load(path)
            xyz0 = data["xyz0"].astype(np.float32)
            xyz1 = data["xyz1"].astype(np.float32)
            corres = data["corres"].astype(np.int64)
            if corres.shape[0] != xyz0.shape[0]:
                raise ValueError(f"corres size mismatch in {path}")
            if corres.max() >= xyz1.shape[0]:
                raise ValueError(f"corres index out of range in {path}")

            rng = np.random.default_rng(self.seed + idx)
            src_idx = self._sample_indices(corres.shape[0], rng)
            tgt_idx = corres[src_idx]
            if idx % 100 == 0:
                print(f"Processed {idx} / {len(self.files)} files")
                print(f"  {src_idx.shape[0]} source indices sampled out of {xyz0.shape[0]} total")
            src_selected_pct = float(src_idx.shape[0] / xyz0.shape[0]) if xyz0.shape[0] > 0 else 0.0
            tgt_selected_unique = int(np.unique(tgt_idx).shape[0]) if tgt_idx.size > 0 else 0
            tgt_selected_pct = float(tgt_selected_unique / xyz1.shape[0]) if xyz1.shape[0] > 0 else 0.0
            src_pcts.append(src_selected_pct)
            tgt_pcts.append(tgt_selected_pct)

        return {
            "num_samples": len(self.files),
            "mean_src_selected_pct": float(np.mean(src_pcts)),
            "mean_tgt_selected_pct": float(np.mean(tgt_pcts)),
        }


class FullAlignmentDataset(Dataset):
    def __init__(
        self,
        files,
        num_corr,
        use_features,
        feature_dim,
        include_gt_trans,
        seed,
        k=10,
        force_add_true_pairs=False,
        max_corr=-1,
    ):
        self.files = list(files)
        self.num_corr = num_corr
        self.use_features = use_features
        self.feature_dim = feature_dim
        self.include_gt_trans = include_gt_trans
        self.seed = seed
        self.k = int(k)
        self.force_add_true_pairs = bool(force_add_true_pairs)
        self.max_corr = int(max_corr) if max_corr is not None else 0
        self._knn_cache = {}
        self._coverage_cache = None
        if self.max_corr > 0:
            self.files = self._filter_max_corr(self.files, self.max_corr)

    def __len__(self):
        return len(self.files)

    def _filter_max_corr(self, files, max_corr):
        kept = []
        for path in files:
            data = np.load(path)
            if "corres" not in data:
                raise ValueError(f"'corres' missing in {path}")
            if int(data["corres"].shape[0]) <= max_corr:
                kept.append(path)
        return kept

    def _sample_indices(self, total, rng):
        if self.num_corr is None or self.num_corr <= 0:
            return np.arange(total, dtype=np.int64)
        replace = total < self.num_corr
        return rng.choice(total, size=self.num_corr, replace=replace).astype(np.int64)

    def _compute_knn_membership(self, idx, corres, feat0, feat1):
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if feat1.shape[0] == 0:
            raise ValueError("features1 is empty")

        rng = np.random.default_rng(self.seed + idx)
        src_idx = self._sample_indices(corres.shape[0], rng)
        k_eff = min(self.k, feat1.shape[0])
        knn_tgt = fpfh_knn_correspondences(
            feat0[src_idx],
            feat1,
            k=k_eff,
            return_distances=False,
            squeeze=False,
        )
        true_in_knn = (knn_tgt == corres[src_idx][:, None]).any(axis=1)
        return src_idx, knn_tgt, true_in_knn

    def _get_knn_membership(self, idx, corres, feat0, feat1):
        if idx in self._knn_cache:
            return self._knn_cache[idx]
        result = self._compute_knn_membership(idx, corres, feat0, feat1)
        self._knn_cache[idx] = result
        return result

    def true_pair_coverage_for_index(self, idx):
        path = self.files[idx]
        data = np.load(path)
        xyz0 = data["xyz0"].astype(np.float32)
        xyz1 = data["xyz1"].astype(np.float32)
        corres = data["corres"].astype(np.int64)
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

        src_idx, knn_tgt, true_in_knn = self._get_knn_membership(idx, corres, feat0, feat1)
        num_sampled_sources = int(src_idx.shape[0])
        num_true_in_knn = int(true_in_knn.sum())
        coverage = float(num_true_in_knn / num_sampled_sources) if num_sampled_sources > 0 else 0.0
        coverage_after_force_add = 1.0 if (self.force_add_true_pairs and num_sampled_sources > 0) else coverage

        k_eff = knn_tgt.shape[1]
        src_rep = np.repeat(src_idx, k_eff)
        tgt_flat = knn_tgt.reshape(-1).astype(np.int64, copy=False)
        if k_eff != self.k:
            print(
                f"Warning: effective k ({k_eff}) != configured k ({self.k}) "
                f"for {path} (feat1 has {feat1.shape[0]} points)"
            )
        if self.force_add_true_pairs:
            gt_tgt = corres[src_idx]
            missing_true = ~true_in_knn
            if missing_true.any():
                add_src = src_idx[missing_true]
                add_tgt = gt_tgt[missing_true]
                # Replace one existing pair per missing source to keep pair count stable
                src_rep = np.concatenate([src_rep, add_src])
                tgt_flat = np.concatenate([tgt_flat, add_tgt])
                for src_val in add_src:
                    idxs = np.flatnonzero(src_rep == src_val)
                    if idxs.size <= 1:
                        continue
                    drop_idx = int(idxs[0])
                    src_rep = np.delete(src_rep, drop_idx)
                    tgt_flat = np.delete(tgt_flat, drop_idx)

        num_pairs = int(tgt_flat.shape[0])
        expected_pairs = int(num_sampled_sources * k_eff)
        if num_pairs != expected_pairs:
            print(
                f"Warning: pair count mismatch for {path}: "
                f"expected {expected_pairs} (num_sampled_sources={num_sampled_sources}, k_eff={k_eff}) "
                f"but got {num_pairs} after force_add_true_pairs"
            )
        src_selected_pct = float(num_sampled_sources / xyz0.shape[0]) if xyz0.shape[0] > 0 else 0.0
        tgt_selected_unique = int(np.unique(tgt_flat).shape[0]) if tgt_flat.size > 0 else 0
        tgt_selected_pct = float(tgt_selected_unique / xyz1.shape[0]) if xyz1.shape[0] > 0 else 0.0

        if idx % 100 == 0:
            print(f"Processed {idx} / {len(self.files)} files")
            print(f"  {src_idx.shape[0]} source indices sampled out of {xyz0.shape[0]} total")
        return {
            "index": int(idx),
            "path": str(path),
            "num_sampled_sources": num_sampled_sources,
            "num_true_in_knn": num_true_in_knn,
            "coverage": coverage,
            "coverage_after_force_add": float(coverage_after_force_add),
            "num_pairs": num_pairs,
            "src_selected_pct": src_selected_pct,
            "tgt_selected_pct": tgt_selected_pct,
        }

    def summarize_true_pair_coverage(self):
        if self._coverage_cache is not None:
            return dict(self._coverage_cache)

        if len(self.files) == 0:
            summary = {
                "num_samples": 0,
                "total_sampled_sources": 0,
                "total_true_in_knn": 0,
                "total_true_after_force": 0,
                "overall_coverage": 0.0,
                "mean_coverage": 0.0,
                "std_coverage": 0.0,
                "mean_num_pairs": 0.0,
                "mean_src_selected_pct": 0.0,
                "mean_tgt_selected_pct": 0.0,
            }
            self._coverage_cache = summary
            return dict(summary)

        per_sample_coverage = []
        per_sample_num_pairs = []
        per_sample_src_selected_pct = []
        per_sample_tgt_selected_pct = []
        total_sampled_sources = 0
        total_true_in_knn = 0
        total_true_after_force = 0
        for idx in range(len(self.files)):
            stats = self.true_pair_coverage_for_index(idx)
            per_sample_coverage.append(stats["coverage"])
            per_sample_num_pairs.append(stats["num_pairs"])
            per_sample_src_selected_pct.append(stats["src_selected_pct"])
            per_sample_tgt_selected_pct.append(stats["tgt_selected_pct"])
            total_sampled_sources += stats["num_sampled_sources"]
            total_true_in_knn += stats["num_true_in_knn"]
            if self.force_add_true_pairs:
                total_true_after_force += stats["num_sampled_sources"]
            else:
                total_true_after_force += stats["num_true_in_knn"]

        summary = {
            "num_samples": len(self.files),
            "total_sampled_sources": int(total_sampled_sources),
            "total_true_in_knn": int(total_true_in_knn),
            "total_true_after_force": int(total_true_after_force),
            "mean_coverage": float(np.mean(per_sample_coverage)),
            "mean_num_pairs": float(np.mean(per_sample_num_pairs)),
            "mean_src_selected_pct": float(np.mean(per_sample_src_selected_pct)),
            "mean_tgt_selected_pct": float(np.mean(per_sample_tgt_selected_pct)),
        }
        self._coverage_cache = summary
        return dict(summary)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        xyz0 = data["xyz0"].astype(np.float32)
        xyz1 = data["xyz1"].astype(np.float32)
        normal0 = data["normal0"].astype(np.float32)
        normal1 = data["normal1"].astype(np.float32)
        corres = data["corres"].astype(np.int64)

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
        src_idx, knn_tgt, true_in_knn = self._get_knn_membership(idx, corres, feat0, feat1)
        k_eff = knn_tgt.shape[1]

        src_rep = np.repeat(src_idx, k_eff)
        tgt_flat = knn_tgt.reshape(-1).astype(np.int64, copy=False)
        if k_eff != self.k:
            print(
                f"Warning: effective k ({k_eff}) != configured k ({self.k}) "
                f"for {path} (feat1 has {feat1.shape[0]} points)"
            )

        if self.force_add_true_pairs:
            gt_tgt = corres[src_idx]
            missing_true = ~true_in_knn
            if missing_true.any():
                add_src = src_idx[missing_true]
                add_tgt = gt_tgt[missing_true]
                src_rep = np.concatenate([src_rep, add_src])
                tgt_flat = np.concatenate([tgt_flat, add_tgt])
                for src_val in add_src:
                    idxs = np.flatnonzero(src_rep == src_val)
                    if idxs.size <= 1:
                        continue
                    drop_idx = int(idxs[0])
                    src_rep = np.delete(src_rep, drop_idx)
                    tgt_flat = np.delete(tgt_flat, drop_idx)

        expected_pairs = int(src_idx.shape[0] * k_eff)
        if int(tgt_flat.shape[0]) != expected_pairs:
            print(
                f"Warning: pair count mismatch for {path}: "
                f"expected {expected_pairs} (num_sampled_sources={src_idx.shape[0]}, k_eff={k_eff}) "
                f"but got {tgt_flat.shape[0]} after force_add_true_pairs"
            )
        gt_labels = (tgt_flat == corres[src_rep]).astype(np.float32)

        src_keypts = xyz0[src_rep]
        tgt_keypts = xyz1[tgt_flat]
        src_normal = normal0[src_rep]
        tgt_normal = normal1[tgt_flat]
        src_indices = torch.from_numpy(src_rep.astype(np.int64, copy=False))
        tgt_indices = torch.from_numpy(tgt_flat.astype(np.int64, copy=False))

        corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=1)
        if self.use_features:
            corr_pos = np.concatenate([corr_pos, feat0[src_rep], feat1[tgt_flat]], axis=1)

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
