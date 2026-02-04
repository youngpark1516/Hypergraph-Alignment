import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_ply_points(path: Path) -> np.ndarray:
    # Try reading as point cloud first, fall back to mesh vertices
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if pts.size == 0:
        # try triangle mesh
        mesh = o3d.io.read_triangle_mesh(str(path))
        if mesh.is_empty():
            raise RuntimeError(f"Failed to read points or mesh from {path}")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        return verts
    return pts


def find_scans(raw_root: Path, split: str):
    # FAUST raw layout has `training/scans` and `test/scans` directories
    if split == "train":
        scan_dir = raw_root / "training" / "scans"
    elif split == "test":
        scan_dir = raw_root / "test" / "scans"
    else:
        scan_dir = raw_root / split / "scans"
    if not scan_dir.exists():
        return []
    return sorted(scan_dir.glob("*.ply"))


def process_split(raw_root: Path, out_root: Path, split: str):
    scans = find_scans(raw_root, split)
    if not scans:
        print(f"No scans found for split '{split}' under {raw_root}")
        return
    for p in tqdm(scans, desc=f"Processing {split}"):
        rel_name = p.stem
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (rel_name + ".npy")
        try:
            pts = load_ply_points(p)
            np.save(out_path, pts)
        except Exception as exc:
            print(f"Failed to process {p}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Convert FAUST .ply scans to .npy point clouds")
    parser.add_argument("--raw-root", type=Path, default=Path("data/faust"), help="Root of FAUST raw files")
    parser.add_argument("--out-root", type=Path, default=Path("data/faust/point_clouds"), help="Output root for .npy point clouds")
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to process (train,test,val)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits:
        process_split(args.raw_root, args.out_root, s)


if __name__ == "__main__":
    main()
