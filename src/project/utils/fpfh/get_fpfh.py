import os
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm
import argparse


def compute_fpfh(points, voxel_size, downsample=True, interp_k=1):
    points = np.asarray(points, dtype=np.float32)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if not downsample:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        fpfh_np = np.array(fpfh.data).T
        normals = np.asarray(pcd.normals)
        return points, fpfh_np, normals

    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    if len(pcd_down.points) == 0:
        raise ValueError("Downsampled point cloud is empty.")

    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    fpfh_down = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    fpfh_down_np = np.array(fpfh_down.data).T
    normals_down = np.asarray(pcd_down.normals)

    tree = o3d.geometry.KDTreeFlann(pcd_down)
    features = np.empty((points.shape[0], fpfh_down_np.shape[1]), dtype=fpfh_down_np.dtype)
    normals = np.empty((points.shape[0], 3), dtype=normals_down.dtype)
    k = max(1, int(interp_k))
    for idx, pt in enumerate(points):
        _, idxs, _ = tree.search_knn_vector_3d(pt, k)
        if len(idxs) == 1:
            sel = idxs[0]
            features[idx] = fpfh_down_np[sel]
            normals[idx] = normals_down[sel]
        else:
            sel = np.asarray(idxs, dtype=np.int64)
            features[idx] = fpfh_down_np[sel].mean(axis=0)
            normals[idx] = normals_down[sel].mean(axis=0)

    return points, features, normals


def load_pair_xyz(data):
    xyz0 = data["xyz0"]
    xyz1 = data["xyz1"]
    return xyz0, xyz1


def save_pair_features(filename, xyz0, xyz1, fpfh0, fpfh1, normals0, normals1, corres):
    np.savez_compressed(
        filename,
        xyz0=np.asarray(xyz0).astype(np.float32),
        xyz1=np.asarray(xyz1).astype(np.float32),
        features0=fpfh0.astype(np.float32),
        features1=fpfh1.astype(np.float32),
        normal0=normals0.astype(np.float32),
        normal1=normals1.astype(np.float32),
        corres=corres.astype(np.int32)
    )


def process_pairs(pair_files, pair_root, save_root, voxel_size, downsample=True, interp_k=1):
    for pair_path in tqdm(pair_files, ncols=50):
        with np.load(pair_path) as data:
            xyz0, xyz1 = load_pair_xyz(data)
            corres = data["corres"]
        if xyz0.shape[0] == 0 or xyz1.shape[0] == 0:
            print(f"{pair_path} error: empty point cloud.")
            continue

        xyz0_out, fpfh_np0, normals0 = compute_fpfh(
            xyz0, voxel_size, downsample=downsample, interp_k=interp_k
        )
        xyz1_out, fpfh_np1, normals1 = compute_fpfh(
            xyz1, voxel_size, downsample=downsample, interp_k=interp_k
        )

        rel_path = os.path.relpath(pair_path, pair_root)
        filename = os.path.join(save_root, rel_path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_pair_features(
            filename, xyz0_out, xyz1_out, fpfh_np0, fpfh_np1, normals0, normals1, corres
        )
        # print(pair_path, fpfh_np0.shape, fpfh_np1.shape)


def process_partnet(voxel_size=0.02, pair_root=None, save_root=None, downsample=True, interp_k=1):
    pair_root = pair_root or "data_utils/partnet/output/rigid_pc"
    save_root = save_root or "data_utils/partnet/output/fpfh_rigid"
    os.makedirs(save_root, exist_ok=True)

    pair_files = sorted(glob.glob(os.path.join(pair_root, "**", "*.npz"), recursive=True))
    if len(pair_files) == 0:
        raise ValueError(f"No pair files found under {pair_root}")

    process_pairs(
        pair_files, pair_root, save_root, voxel_size, downsample=downsample, interp_k=interp_k
    )


def process_faust(voxel_size=0.02, pair_root=None, save_root=None, downsample=True, interp_k=1):
    pair_root = pair_root or "data_utils/faust/data/processed/faust/corres/pairs"
    save_root = save_root or "data_utils/faust/data/processed/faust/fpfh"
    os.makedirs(save_root, exist_ok=True)

    pair_files = sorted(glob.glob(os.path.join(pair_root, "*.npz")))
    if len(pair_files) == 0:
        raise ValueError(f"No pair files found under {pair_root}")

    process_pairs(
        pair_files, pair_root, save_root, voxel_size, downsample=downsample, interp_k=interp_k
    )


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
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Compute FPFH on full point clouds without voxel downsampling.",
    )
    parser.add_argument(
        "--interp-k",
        type=int,
        default=1,
        help="Number of nearest downsampled points to average when interpolating.",
    )
    args = parser.parse_args()
    downsample = not args.no_downsample

    if args.dataset == "faust":
        process_faust(
            voxel_size=args.voxel_size,
            pair_root=args.pair_root,
            save_root=args.save_root,
            downsample=downsample,
            interp_k=args.interp_k,
        )
    elif args.dataset == "partnet":
        process_partnet(
            voxel_size=args.voxel_size,
            pair_root=args.pair_root,
            save_root=args.save_root,
            downsample=downsample,
            interp_k=args.interp_k,
        )


if __name__ == '__main__':
    main()
