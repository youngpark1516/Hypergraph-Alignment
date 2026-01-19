from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def fpfh_knn_correspondences(
    features0: np.ndarray,
    features1: np.ndarray,
    k: int = 1,
    return_distances: bool = False,
    squeeze: bool = True,
):
    """Find k-NN correspondences in FPFH feature space.

    Args:
        features0: (N0, F) source FPFH features.
        features1: (N1, F) target FPFH features.
        k: number of nearest neighbors per source point.
        return_distances: whether to return L2 distances.
        squeeze: if True and k == 1, return 1D arrays of shape (N0,).

    Returns:
        indices: (N0, k) int64 indices into features1 (or (N0,) if squeezed).
        distances (optional): L2 distances with the same shape as indices.
    """
    features0 = np.asarray(features0, dtype=np.float32)
    features1 = np.asarray(features1, dtype=np.float32)

    if features0.ndim != 2 or features1.ndim != 2:
        raise ValueError("features0 and features1 must be 2D arrays.")
    if features0.shape[1] != features1.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: {features0.shape[1]} vs {features1.shape[1]}"
        )
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if features1.shape[0] == 0:
        raise ValueError("features1 is empty; cannot build correspondences.")
    if k > features1.shape[0]:
        raise ValueError("k cannot exceed the number of target features.")

    if features0.shape[0] == 0:
        empty_idx = np.empty((0, k), dtype=np.int64)
        if return_distances:
            empty_dist = np.empty((0, k), dtype=np.float32)
            return (empty_idx.squeeze(-1) if squeeze and k == 1 else empty_idx), (
                empty_dist.squeeze(-1) if squeeze and k == 1 else empty_dist
            )
        return empty_idx.squeeze(-1) if squeeze and k == 1 else empty_idx

    tree = cKDTree(features1)
    distances, indices = tree.query(features0, k=k)

    if k == 1:
        indices = indices.astype(np.int64, copy=False)
        distances = distances.astype(np.float32, copy=False)
        if squeeze:
            if return_distances:
                return indices, distances
            return indices
        indices = indices.reshape(-1, 1)
        distances = distances.reshape(-1, 1)
    else:
        indices = indices.astype(np.int64, copy=False)
        distances = distances.astype(np.float32, copy=False)

    if return_distances:
        return indices, distances
    return indices
