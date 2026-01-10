from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
from tqdm import tqdm
import itertools


def load_scan_points(scan_path: Path) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(scan_path))
    if len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(str(scan_path))
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
    return pts


def load_registration_vertices(reg_path: Path) -> np.ndarray:
    reg = o3d.io.read_triangle_mesh(str(reg_path))
    pts = np.asarray(reg.vertices, dtype=np.float32)
    return pts


def load_gt_flags(gt_path: Path) -> np.ndarray:
    return np.loadtxt(gt_path, dtype=np.int8).astype(bool)


def map_template_to_scan(reg_path: Path, scan_pts: np.ndarray) -> np.ndarray:
    reg_pts = load_registration_vertices(reg_path)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    nearest = np.empty((reg_pts.shape[0],), dtype=np.int64)
    for i, v in enumerate(reg_pts):
        _, idx, _ = kdt.search_knn_vector_3d(v, 1)
        nearest[i] = idx[0]
    return nearest


def build_maps(raw_root: Path, out_root: Path, split: str = "training"):
    scan_dir = raw_root / split / "scans"
    reg_dir = raw_root / split / "registrations"
    gt_dir = raw_root / "training" / "ground_truth_vertices"
    out_maps = out_root / "maps"
    out_maps.mkdir(parents=True, exist_ok=True)

    scans = sorted(scan_dir.glob("*.ply"))
    for p in tqdm(scans, desc="Building maps"):
        name = p.stem  # e.g., tr_scan_000
        idx = name.split("_")[-1]
        reg_path = reg_dir / f"tr_reg_{idx}.ply"
        gt_path = gt_dir / f"tr_gt_{idx}.txt"
        if not reg_path.exists() or not gt_path.exists():
            print(f"Skipping {name}: missing registration or gt")
            continue
        scan_pts = load_scan_points(p)
        map_idx = map_template_to_scan(reg_path, scan_pts)
        gt_flags = load_gt_flags(gt_path)
        # compute template reliability mask: gt_flags mapped by template
        template_valid = gt_flags[map_idx]
        np.save(out_maps / f"{idx}_map.npy", map_idx)
        np.save(out_maps / f"{idx}_mask.npy", template_valid.astype(np.uint8))


def build_pair(a: str, b: str, raw_root: Path, out_root: Path):
    # a,b are zero-padded ids like '000' or '001'
    scans_dir = raw_root / "training" / "scans"
    out_pairs = out_root / "pairs"
    out_pairs.mkdir(parents=True, exist_ok=True)
    maps_dir = out_root / "maps"

    scanA_path = scans_dir / f"tr_scan_{a}.ply"
    scanB_path = scans_dir / f"tr_scan_{b}.ply"
    mapA_path = maps_dir / f"{a}_map.npy"
    mapB_path = maps_dir / f"{b}_map.npy"
    maskA_path = maps_dir / f"{a}_mask.npy"
    maskB_path = maps_dir / f"{b}_mask.npy"

    if not (scanA_path.exists() and scanB_path.exists() and mapA_path.exists() and mapB_path.exists() and maskA_path.exists() and maskB_path.exists()):
        raise FileNotFoundError("Required files missing: ensure maps are built first with --build-maps")

    scanA = load_scan_points(scanA_path)
    scanB = load_scan_points(scanB_path)
    mapA = np.load(mapA_path)
    mapB = np.load(mapB_path)
    maskA = np.load(maskA_path).astype(bool)
    maskB = np.load(maskB_path).astype(bool)

    # template indices that are valid on both scans
    valid_t = np.where(maskA & maskB)[0]
    src_inds = mapA[valid_t]
    tgt_inds = mapB[valid_t]
    src_coords = scanA[src_inds]
    tgt_coords = scanB[tgt_inds]

    out_path = out_pairs / f"{a}_{b}.npz"
    np.savez_compressed(out_path, template_inds=valid_t, src_inds=src_inds, tgt_inds=tgt_inds, src_coords=src_coords, tgt_coords=tgt_coords)
    print(f"Saved pair correspondences: {out_path} (N={valid_t.shape[0]})")


def build_all_pairs(raw_root: Path, out_root: Path):
    scans_dir = raw_root / "training" / "scans"
    ids = [p.stem.split("_")[-1] for p in sorted(scans_dir.glob("tr_scan_*.ply"))]
    # build unordered pairs
    for a, b in tqdm(itertools.combinations(ids, 2), desc="Building all pairs"):
        try:
            build_pair(a, b, raw_root, out_root)
        except Exception as exc:
            print(f"Failed pair {a}_{b}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Build FAUST template->scan maps and per-pair correspondences")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/MPI-FAUST"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/faust/corres"))
    parser.add_argument("--build-maps", action="store_true")
    parser.add_argument("--build-pair", nargs=2, help="Build a single pair: provide two zero-padded ids (e.g. 000 001)")
    parser.add_argument("--build-all-pairs", action="store_true", help="Build correspondences for all unordered training pairs")
    args = parser.parse_args()

    if args.build_maps:
        build_maps(args.raw_root, args.out_root)
    if args.build_pair:
        a, b = args.build_pair
        build_pair(a, b, args.raw_root, args.out_root)
    if args.build_all_pairs:
        build_all_pairs(args.raw_root, args.out_root)


if __name__ == "__main__":
    main()
