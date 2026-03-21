import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_cloud_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("cloud_bin_"):
        raise ValueError(f"Unexpected cloud filename: {path.name}")
    return int(stem.split("_")[-1])


def parse_gt_log(gt_log_path: Path):
    lines = [line.strip() for line in gt_log_path.read_text().splitlines() if line.strip()]
    entries = []
    i = 0
    while i + 4 < len(lines):
        header = lines[i].split()
        if len(header) < 2:
            raise ValueError(f"Invalid gt.log header at line {i + 1} in {gt_log_path}")
        src_idx = int(header[0])
        tgt_idx = int(header[1])
        mat_vals = []
        for row in lines[i + 1 : i + 5]:
            vals = [float(x) for x in row.split()]
            if len(vals) != 4:
                raise ValueError(f"Invalid transform row in {gt_log_path}: '{row}'")
            mat_vals.extend(vals)
        transform = np.asarray(mat_vals, dtype=np.float64).reshape(4, 4)
        entries.append((src_idx, tgt_idx, transform))
        i += 5
    return entries


def load_points(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"Invalid point cloud in {path}")
    return pts


def compute_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate normals.")
    k = max(3, min(int(k), points.shape[0] - 1))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    return np.asarray(pcd.normals, dtype=np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return (points @ rot.T) + trans


def build_pair_npz(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    src_normals: np.ndarray,
    tgt_normals: np.ndarray,
    t_tgt_src: np.ndarray,
    match_radius: float,
    min_corr: int,
):
    src_in_tgt = transform_points(src_points, t_tgt_src)

    tgt_tree = cKDTree(tgt_points.astype(np.float64))
    dists, nn_idx = tgt_tree.query(src_in_tgt.astype(np.float64), k=1, workers=-1)
    dists = np.asarray(dists, dtype=np.float64)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)

    valid = dists <= float(match_radius)
    src_idx = np.flatnonzero(valid).astype(np.int64, copy=False)
    if src_idx.shape[0] < int(min_corr):
        return None

    corres = nn_idx[src_idx]
    xyz0 = src_points[src_idx]
    xyz1 = tgt_points
    normal0 = src_normals[src_idx]
    normal1 = tgt_normals

    return {
        "xyz0": xyz0.astype(np.float32),
        "xyz1": xyz1.astype(np.float32),
        "normal0": normal0.astype(np.float32),
        "normal1": normal1.astype(np.float32),
        "corres": corres.astype(np.int64),
    }


def scene_files(pc_scene_dir: Path):
    return sorted(pc_scene_dir.glob("cloud_bin_*.ply"), key=parse_cloud_index)


def process_scene(
    pc_scene_dir: Path,
    eval_scene_dir: Path,
    out_scene_dir: Path,
    max_frame_gap: int,
    max_pairs_per_scene: int,
    match_radius: float,
    normal_k: int,
    min_corr: int,
):
    clouds = scene_files(pc_scene_dir)
    if not clouds:
        return {"saved": 0, "skipped": 0, "pairs": 0}

    gt_log = eval_scene_dir / "gt.log"
    if not gt_log.exists():
        print(f"Skipping {pc_scene_dir.name}: missing {gt_log}")
        return {"saved": 0, "skipped": 0, "pairs": 0}

    gt_entries = parse_gt_log(gt_log)
    if max_frame_gap > 0:
        gt_entries = [e for e in gt_entries if abs(e[1] - e[0]) <= max_frame_gap]
    if max_pairs_per_scene > 0:
        gt_entries = gt_entries[:max_pairs_per_scene]

    out_scene_dir.mkdir(parents=True, exist_ok=True)
    cloud_map = {parse_cloud_index(p): p for p in clouds}
    points_cache = {}
    normals_cache = {}

    saved = 0
    skipped = 0

    for src_idx, tgt_idx, t_tgt_src in tqdm(gt_entries, desc=f"{pc_scene_dir.name}", leave=False):
        src_path = cloud_map.get(src_idx)
        tgt_path = cloud_map.get(tgt_idx)
        if src_path is None or tgt_path is None:
            skipped += 1
            continue
        src_name = src_path.stem
        tgt_name = tgt_path.stem

        if src_path not in points_cache:
            points_cache[src_path] = load_points(src_path)
            normals_cache[src_path] = compute_normals(points_cache[src_path], k=normal_k)
        if tgt_path not in points_cache:
            points_cache[tgt_path] = load_points(tgt_path)
            normals_cache[tgt_path] = compute_normals(points_cache[tgt_path], k=normal_k)

        pair = build_pair_npz(
            src_points=points_cache[src_path],
            tgt_points=points_cache[tgt_path],
            src_normals=normals_cache[src_path],
            tgt_normals=normals_cache[tgt_path],
            t_tgt_src=t_tgt_src,
            match_radius=match_radius,
            min_corr=min_corr,
        )
        if pair is None:
            skipped += 1
            continue

        out_path = out_scene_dir / f"{src_name}__{tgt_name}.npz"
        np.savez_compressed(out_path, **pair)
        saved += 1

    return {"saved": saved, "skipped": skipped, "pairs": len(gt_entries)}


def discover_scenes(root: Path):
    scenes = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.endswith("-evaluation"):
            continue
        if list(p.glob("cloud_bin_*.ply")):
            scenes.append(p)
    return scenes


def main():
    parser = argparse.ArgumentParser(description="Build 3DMatch pair correspondences in Hypergraph-Alignment format")
    parser.add_argument("--pc-root", type=Path, default=Path("data/3dmatch"), help="3DMatch root containing scene folders with cloud_bin_*.ply")
    parser.add_argument("--eval-root", type=Path, default=Path("data/3dmatch"), help="Root containing <scene>-evaluation/gt.log files")
    parser.add_argument("--out-root", type=Path, default=Path("data/3dmatch/pairs"), help="Output root for generated pair .npz files")
    parser.add_argument("--scenes", type=str, default="", help="Comma-separated scene names; empty means all scenes under --pc-root")
    parser.add_argument("--max-frame-gap", type=int, default=0, help="Optional cap on |target_idx - source_idx| for gt.log pairs; 0 keeps all official pairs")
    parser.add_argument("--max-pairs-per-scene", type=int, default=0, help="Optional cap on number of gt.log pairs processed per scene; 0 means no cap")
    parser.add_argument("--match-radius", type=float, default=0.10, help="Distance threshold (meters) for correspondence acceptance")
    parser.add_argument("--normal-k", type=int, default=30, help="KNN used for normal estimation")
    parser.add_argument("--min-corr", type=int, default=256, help="Minimum accepted correspondences per pair")
    args = parser.parse_args()

    if args.scenes.strip():
        scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
        scenes = [args.pc_root / name for name in scene_names]
    else:
        scenes = discover_scenes(args.pc_root)

    total_saved = 0
    total_skipped = 0
    total_pairs = 0

    for scene_dir in tqdm(scenes, desc="Scenes"):
        if not scene_dir.exists():
            print(f"Skipping missing scene dir: {scene_dir}")
            continue
        eval_scene_dir = args.eval_root / f"{scene_dir.name}-evaluation"
        if not eval_scene_dir.exists():
            print(f"Skipping {scene_dir.name}: eval directory not found at {eval_scene_dir}")
            continue

        stats = process_scene(
            pc_scene_dir=scene_dir,
            eval_scene_dir=eval_scene_dir,
            out_scene_dir=args.out_root / scene_dir.name,
            max_frame_gap=args.max_frame_gap,
            max_pairs_per_scene=args.max_pairs_per_scene,
            match_radius=args.match_radius,
            normal_k=args.normal_k,
            min_corr=args.min_corr,
        )
        total_saved += stats["saved"]
        total_skipped += stats["skipped"]
        total_pairs += stats["pairs"]
        print(
            f"[{scene_dir.name}] candidate_pairs={stats['pairs']} saved={stats['saved']} skipped={stats['skipped']}"
        )

    print("Done.")
    print(f"Total candidate pairs: {total_pairs}")
    print(f"Saved pairs: {total_saved}")
    print(f"Skipped pairs: {total_skipped}")
    print(f"Output root: {args.out_root}")


if __name__ == "__main__":
    main()
