import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_pair_npz(path: Path):
    data = np.load(path)
    required = ["xyz0", "xyz1", "corres"]
    for key in required:
        if key not in data:
            raise KeyError(f"{path} missing required key '{key}'")
    xyz0 = np.asarray(data["xyz0"], dtype=np.float32)
    xyz1 = np.asarray(data["xyz1"], dtype=np.float32)
    corres = np.asarray(data["corres"], dtype=np.int64)
    if xyz0.ndim != 2 or xyz0.shape[1] != 3:
        raise ValueError(f"Expected xyz0 shape (N, 3), got {xyz0.shape}")
    if xyz1.ndim != 2 or xyz1.shape[1] != 3:
        raise ValueError(f"Expected xyz1 shape (M, 3), got {xyz1.shape}")
    if corres.shape[0] != xyz0.shape[0]:
        raise ValueError(f"corres length {corres.shape[0]} does not match xyz0 length {xyz0.shape[0]}")
    if np.any(corres < 0) or np.any(corres >= xyz1.shape[0]):
        raise ValueError("corres contains indices outside xyz1 range")
    return xyz0, xyz1, corres


def sample_indices(n: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise ValueError("Need at least one point to sample")
    k = max(1, int(n * frac))
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


def build_lines(points_a: np.ndarray, points_b: np.ndarray) -> o3d.geometry.LineSet:
    combined = np.vstack([points_a, points_b])
    lines = np.array([[i, i + points_a.shape[0]] for i in range(points_a.shape[0])], dtype=np.int32)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(combined.astype(np.float64))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([0.6, 0.6, 0.6])
    return line_set


def sample_line_pairs(points_a: np.ndarray, points_b: np.ndarray, frac: float, rng: np.random.Generator):
    if frac <= 0.0:
        return None
    n = points_a.shape[0]
    if n == 0:
        return None
    frac = min(1.0, frac)
    k = max(1, int(n * frac))
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return points_a[idx], points_b[idx]


def main():
    parser = argparse.ArgumentParser(description="Visualize FAUST correspondences from a paired .npz")
    parser.add_argument("pair", type=Path, help="Path to .npz produced by data_utils/faust/align.py")
    parser.add_argument("--fraction", type=float, default=0.001, help="Fraction of xyz0 indices to sample")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling")
    parser.add_argument(
        "--line-fraction",
        type=float,
        default=1.0,
        help="Fraction of sampled correspondences to draw as lines (0 disables lines)",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="Translate xyz1 by this amount along +X for visibility (0 disables)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    xyz0, xyz1, corres = load_pair_npz(args.pair)
    idx = sample_indices(xyz0.shape[0], args.fraction, rng)

    xyz0_sel = xyz0[idx]
    xyz1_sel = xyz1[corres[idx]]
    if args.offset != 0.0:
        xyz1_sel = xyz1_sel + np.array([args.offset, 0.0, 0.0], dtype=np.float32)

    pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz0_sel.astype(np.float64)))
    pcd0.paint_uniform_color([0.2, 0.6, 1.0])
    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz1_sel.astype(np.float64)))
    pcd1.paint_uniform_color([1.0, 0.4, 0.2])

    geoms = [pcd0, pcd1]
    line_points = sample_line_pairs(xyz0_sel, xyz1_sel, args.line_fraction, rng)
    if line_points is not None:
        line_a, line_b = line_points
        geoms.append(build_lines(line_a, line_b))

    window_title = f"FAUST correspondences: {args.pair}"
    o3d.visualization.draw_geometries(geoms, window_name=window_title)


if __name__ == "__main__":
    main()
