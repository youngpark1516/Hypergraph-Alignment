import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm


def compute_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate normals.")
    k = max(3, min(k, points.shape[0] - 1))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_norm = 1.0
    axis = axis / axis_norm
    angle = rng.uniform(0, np.pi)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle).astype(np.float32)


def random_se3(translation_dist: float, rng: np.random.Generator):
    # Random rotation via axis-angle
    A = random_rotation_matrix(rng)

    # Random translation with fixed magnitude
    t_dir = rng.normal(size=3)
    t_dir_norm = np.linalg.norm(t_dir)
    t_dist = rng.uniform(0, 1) * translation_dist
    if t_dir_norm < 1e-8:
        t_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t_dir_norm = 1.0
    t = (t_dir / t_dir_norm * t_dist).astype(np.float32)
    gt = np.eye(4, dtype=np.float32)
    gt[:3, :3] = A
    gt[:3, 3] = t
    return A, t, gt


def random_affine_spd(translation_dist: float, sigma_min: float, sigma_max: float, rng: np.random.Generator):
    # Random symmetric positive-definite matrix with singular value bounds
    Q = random_rotation_matrix(rng)
    sigma = rng.uniform(sigma_min, sigma_max, size=3).astype(np.float32)
    sigma = sigma / np.power(np.prod(sigma), 1.0 / 3.0)  # normalize to det=1
    D = np.diag(sigma.astype(np.float32))
    S = (Q @ D @ Q.T).astype(np.float32)

    t_dir = rng.normal(size=3)
    t_dir_norm = np.linalg.norm(t_dir)
    t_dist = rng.uniform(0, 1) * translation_dist
    if t_dir_norm < 1e-8:
        t_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t_dir_norm = 1.0
    t = (t_dir / t_dir_norm * t_dist).astype(np.float32)

    R = random_rotation_matrix(rng)
    A = (R @ S).astype(np.float32)

    gt = np.eye(4, dtype=np.float32)
    gt[:3, :3] = A
    gt[:3, 3] = t
    return A, t, gt


def load_cloud(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 array in {path}, got {arr.shape}")
    return arr.astype(np.float32)


def augment_cloud(
    path: Path,
    translation_dist: float,
    normal_k: int,
    rng: np.random.Generator,
    use_affine: bool,
    sigma_min: float,
    sigma_max: float,
):
    xyz0 = load_cloud(path)
    normal0 = compute_normals(xyz0, k=normal_k)

    if use_affine:
        A, t, gt_trans = random_affine_spd(translation_dist, sigma_min, sigma_max, rng)
    else:
        A, t, gt_trans = random_se3(translation_dist, rng)

    xyz1 = (xyz0 @ A.T) + t

    # Transform normals via inverse-transpose of the linear part, then renormalize.
    A_inv_T = np.linalg.inv(A).T
    normal1 = (A_inv_T @ normal0.T).T
    normal1 = normal1 / np.clip(np.linalg.norm(normal1, axis=1, keepdims=True), 1e-12, None)

    perm = rng.permutation(xyz0.shape[0])
    xyz1 = xyz1[perm]
    normal1 = normal1[perm]
    corres = np.empty_like(perm)
    corres[perm] = np.arange(perm.shape[0])

    return {
        "xyz0": xyz0,
        "xyz1": xyz1.astype(np.float32),
        "normal0": normal0,
        "normal1": normal1.astype(np.float32),
        "gt_trans": gt_trans,
        "corres": corres.astype(np.int64),
    }


def find_clouds(root: Path):
    return sorted(root.glob("**/*.npy"))


def main():
    parser = argparse.ArgumentParser(description="Generate augmented point cloud pairs with SE(3) or affine transforms.")
    parser.add_argument("--pc-root", type=Path, default=Path("point_clouds"), help="Root directory containing source .npy point clouds")
    parser.add_argument("--out-root", type=Path, default=Path("augmented_point_clouds"), help="Output root for .npz files")
    parser.add_argument("--translation", type=float, default=10.0, help="Max translation distance magnitude for the random transform")
    parser.add_argument("--normal-k", type=int, default=10, help="KNN neighbors for normal estimation")
    parser.add_argument("--affine", action="store_true", help="Use affine SPD transform with bounded singular values instead of rigid SE(3)")
    parser.add_argument("--sigma-max", type=float, default=1.2, help="Maximum singular value for affine transform")
    parser.add_argument("--sigma-min", type=float, help="Minimum singular value for affine transform (default: 1/sigma-max)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    sigma_min = args.sigma_min if args.sigma_min is not None else 1.0 / args.sigma_max

    rng = np.random.default_rng(args.seed)
    pc_paths = find_clouds(args.pc_root)
    if not pc_paths:
        raise ValueError(f"No .npy point clouds found under {args.pc_root}")

    for path in tqdm(pc_paths, desc="Augmenting", unit="pc"):
        rel = path.relative_to(args.pc_root)
        out_path = (args.out_root / rel).with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = augment_cloud(
                path,
                args.translation,
                args.normal_k,
                rng,
                args.affine,
                sigma_min,
                args.sigma_max,
            )
            np.savez(out_path, **data)
        except Exception as exc:
            print(f"Failed on {path}: {exc}")


if __name__ == "__main__":
    main()
