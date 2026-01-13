import os
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
import argparse


def compute_fpfh(points, voxel_size):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    fpfh_np = np.array(fpfh.data).T
    normals = np.asarray(pcd.normals)
    return np.asarray(pcd.points), fpfh_np, normals


def load_pair_xyz(pair_path):
    data = np.load(pair_path)
    xyz0 = data["xyz0"]
    xyz1 = data["xyz1"]
    return xyz0, xyz1


def save_pair_features(filename, xyz0, xyz1, fpfh0, fpfh1, normals0, normals1):
    np.savez_compressed(
        filename,
        xyz0=np.asarray(xyz0).astype(np.float32),
        xyz1=np.asarray(xyz1).astype(np.float32),
        features0=fpfh0.astype(np.float32),
        features1=fpfh1.astype(np.float32),
        normal0=normals0.astype(np.float32),
        normal1=normals1.astype(np.float32),
    )


def process_pairs(pair_files, pair_root, save_root, voxel_size):
    for pair_path in tqdm(pair_files, ncols=50):
        xyz0, xyz1 = load_pair_xyz(pair_path)
        if xyz0.shape[0] == 0 or xyz1.shape[0] == 0:
            print(f"{pair_path} error: empty point cloud.")
            continue

        xyz0_ds, fpfh_np0, normals0 = compute_fpfh(xyz0, voxel_size)
        xyz1_ds, fpfh_np1, normals1 = compute_fpfh(xyz1, voxel_size)

        rel_path = os.path.relpath(pair_path, pair_root)
        filename = os.path.join(save_root, rel_path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_pair_features(filename, xyz0_ds, xyz1_ds, fpfh_np0, fpfh_np1, normals0, normals1)
        # print(pair_path, fpfh_np0.shape, fpfh_np1.shape)


def process_partnet(voxel_size=0.02, pair_root=None, save_root=None):
    pair_root = pair_root or "data_utils/partnet/output/rigid_pc"
    save_root = save_root or "data_utils/partnet/output/fpfh_rigid"
    os.makedirs(save_root, exist_ok=True)

    pair_files = sorted(glob.glob(os.path.join(pair_root, "**", "*.npz"), recursive=True))
    if len(pair_files) == 0:
        raise ValueError(f"No pair files found under {pair_root}")

    process_pairs(pair_files, pair_root, save_root, voxel_size)


def process_faust(voxel_size=0.02, pair_root=None, save_root=None):
    pair_root = pair_root or "data_utils/faust/data/processed/faust/corres/pairs"
    save_root = save_root or "data_utils/faust/data/processed/faust/fpfh"
    os.makedirs(save_root, exist_ok=True)

    pair_files = sorted(glob.glob(os.path.join(pair_root, "*.npz")))
    if len(pair_files) == 0:
        raise ValueError(f"No pair files found under {pair_root}")

    process_pairs(pair_files, pair_root, save_root, voxel_size)


def main():
    parser = argparse.ArgumentParser(description="Compute FPFH features for paired point clouds.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["faust", "partnet"],
        required=True,
        help="Dataset to process",
    )
    parser.add_argument(
        "--pair-root",
        type=str,
        default=None,
        help="Root directory of paired point clouds (overrides default paths)",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Directory to save computed features (overrides default paths)",
    )
    parser.add_argument("--voxel-size", type=float, default=0.1)
    args = parser.parse_args()

    if args.dataset == "faust":
        process_faust(voxel_size=args.voxel_size, pair_root=args.pair_root, save_root=args.save_root)
    elif args.dataset == "partnet":
        process_partnet(voxel_size=args.voxel_size, pair_root=args.pair_root, save_root=args.save_root)


if __name__ == '__main__':
    main()
