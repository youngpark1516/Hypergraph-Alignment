import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_cloud(path: Path) -> np.ndarray:
    cloud = np.load(path)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError(f"Expected array of shape (N, 3), got {cloud.shape}")
    return cloud


def load_pair_npz(path: Path):
    data = np.load(path)
    required = ["xyz0", "xyz1"]
    for key in required:
        if key not in data:
            raise KeyError(f"{path} missing required key '{key}'")
    gt_trans = data.get("gt_trans")
    return np.asarray(data["xyz0"]), np.asarray(data["xyz1"]), gt_trans


def main():
    parser = argparse.ArgumentParser(description="Visualize point clouds stored as .npy or paired .npz from augment_point_clouds.py")
    parser.add_argument("cloud", type=Path, nargs="?", help="Path to source .npy point cloud (shape: N x 3)")
    parser.add_argument("--pair", type=Path, help="Path to .npz produced by augment_point_clouds.py (uses xyz0/xyz1)")
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Optional voxel downsample size (0 to disable)",
    )
    args = parser.parse_args()

    tgt_points = None
    gt_trans = None
    if args.pair:
        if args.cloud is not None:
            raise ValueError("When using --pair, do not also pass positional cloud or --tgt.")
        src_points, tgt_points, gt_trans = load_pair_npz(args.pair)
        if gt_trans is not None:
            print("gt_trans:\n", gt_trans)
        else:
            print("gt_trans not found in npz; proceeding without displaying transform.")
    else:
        if args.cloud is None:
            raise ValueError("Provide a source cloud .npy or --pair .npz to visualize.")
        src_points = load_cloud(args.cloud)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_points))

    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel)

    # Give everything a neutral color so it is visible.
    pcd.paint_uniform_color([0.2, 0.6, 1.0])

    geoms = [pcd]

    if tgt_points is not None:
        tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_points))
        if args.voxel and args.voxel > 0:
            tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=args.voxel)
        tgt_pcd.paint_uniform_color([1.0, 0.4, 0.2])
        geoms.append(tgt_pcd)

    window_title = f"Point Cloud(s): {args.cloud if args.cloud else args.pair}"
    o3d.visualization.draw_geometries(geoms, window_name=window_title)


if __name__ == "__main__":
    main()
