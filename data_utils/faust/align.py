import argparse
from operator import gt
import numpy as np
import os
from pathlib import Path
import re
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

def read_ply(path):
    """Read a PLY file and return a dict with 'vertices' (Nx3 numpy array) and optional 'faces'.
    - Supports ASCII PLY (simple parser here).
    - For binary PLY, attempts to use trimesh or open3d if available.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PLY file not found: {path}")

    with open(path, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Unexpected EOF while reading PLY header')
            line = line.decode('utf-8').rstrip('\n')
            header_lines.append(line)
            if line.strip() == 'end_header':
                break

        # Parse header
        fmt_line = [l for l in header_lines if l.startswith('format')]
        fmt = fmt_line[0].split()[1] if fmt_line else 'ascii'

        vertex_el = [l for l in header_lines if l.startswith('element vertex')]
        n_verts = int(vertex_el[0].split()[-1]) if vertex_el else 0

        face_el = [l for l in header_lines if l.startswith('element face')]
        n_faces = int(face_el[0].split()[-1]) if face_el else 0

        try:
            mesh = trimesh.load(path, process=False)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces) if hasattr(mesh, 'faces') and mesh.faces is not None else None
            return {'vertices': verts, 'faces': faces}
        except Exception:
            try:
                import open3d as o3d
                # try as point cloud
                try:
                    pcd = o3d.io.read_point_cloud(path)
                    verts = np.asarray(pcd.points)
                    return {'vertices': verts, 'faces': None}
                except Exception:
                    mesh = o3d.io.read_triangle_mesh(path)
                    verts = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.triangles) if len(mesh.triangles) > 0 else None
                    return {'vertices': verts, 'faces': faces}
            except Exception:
                raise RuntimeError('Binary PLY reading requires trimesh or open3d (not available)')
            

def load_mesh_vertices_normals(path: Path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise RuntimeError(f"No geometry in {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not hasattr(mesh, "vertices"):
        raise RuntimeError(f"Expected mesh with vertices in {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != verts.shape:
        raise RuntimeError(f"Normals shape mismatch for {path}: {normals.shape} vs {verts.shape}")
    return verts, normals


def load_gt_vertices(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.int8)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr.astype(bool)


def align(scan1, scan2, reg1, reg2, gt_vert1, gt_vert2):
    """
    Align two point clouds based on their registrations and ground truth vertices.
    
    :param scan1: point cloud 1 as numpy array of shape (N, 3)
    :param scan2: point cloud 2 as numpy array of shape (M, 3)
    :param reg1: registration of point cloud 1 as numpy array of shape (N, 3)
    :param reg2: registration of point cloud 2 as numpy array of shape (M, 3)
    :param gt_vert1: ground truth vertices for point cloud 1 as numpy array of shape (N,)
    :param gt_vert2: ground truth vertices for point cloud 2 as numpy array of shape (M,)
    """
    gt_vert1 = gt_vert1.astype(bool)
    gt_vert2 = gt_vert2.astype(bool)
    if gt_vert1.shape[0] != scan1.shape[0]:
        raise ValueError("gt_vert1 must match scan1 length")
    if gt_vert2.shape[0] != scan2.shape[0]:
        raise ValueError("gt_vert2 must match scan2 length")

    scan1 = scan1[gt_vert1]

    max_dist = 5e-2
    # For each point in scan1, find the closest point in reg1
    tree1 = cKDTree(reg1)
    dists1, indices1 = tree1.query(scan1)
    if not np.all(dists1 < max_dist):
        print(f"scan1->reg1 max distance: {max(dists1)}")
        raise ValueError(f"scan1->reg1 distances must be < {max_dist}")
    
    # For each point in scan2, find the closest point in reg2
    tree2 = cKDTree(reg2)
    dists2, indices2 = tree2.query(scan2)
    if not np.all(dists2[gt_vert2] < max_dist):
        print(f"scan2->reg2 max distance: {max(dists2[gt_vert2])}")
        raise ValueError(f"scan2->reg2 distances must be < {max_dist}")
    # Reverse indices
    reverse_indices2 = np.zeros_like(indices2)
    reverse_indices2[indices2] = np.arange(len(indices2))

    return reverse_indices2[indices1]


def find_training_indices(scan_dir: Path):
    pattern = re.compile(r"tr_scan_(\d+)\.ply$")
    indices = []
    for p in scan_dir.glob("tr_scan_*.ply"):
        match = pattern.match(p.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def group_by_body(indices):
    bodies = {}
    for idx in indices:
        body_id = idx // 10
        bodies.setdefault(body_id, []).append(idx)
    return bodies


def sample_pairs(indices, rng, pairs_per_body: int):
    if len(indices) < 2 * pairs_per_body:
        raise ValueError(f"Need at least {2 * pairs_per_body} scans, got {len(indices)}")
    perm = rng.permutation(indices)
    pairs = []
    for i in range(pairs_per_body):
        a = int(perm[2 * i])
        b = int(perm[2 * i + 1])
        pairs.append((a, b))
    return pairs


def build_paths(raw_root: Path, idx: int):
    scan_path = raw_root / "training" / "scans" / f"tr_scan_{idx:03d}.ply"
    reg_path = raw_root / "training" / "registrations" / f"tr_reg_{idx:03d}.ply"
    gt_path = raw_root / "training" / "ground_truth_vertices" / f"tr_gt_{idx:03d}.txt"
    return scan_path, reg_path, gt_path


def main():
    parser = argparse.ArgumentParser(description="Build FAUST scan pairs with correspondences.")
    parser.add_argument("--raw-root", type=Path, default=Path("data_utils/faust/data/raw/MPI-FAUST"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed/faust/train"))
    parser.add_argument("--pairs-per-body", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scan_dir = args.raw_root / "training" / "scans"
    indices = find_training_indices(scan_dir)
    if not indices:
        raise ValueError(f"No scans found under {scan_dir}")

    rng = np.random.default_rng(args.seed)
    bodies = group_by_body(indices)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    for body_id in sorted(bodies.keys()):
        body_indices = sorted(bodies[body_id])
        pairs = sample_pairs(body_indices, rng, args.pairs_per_body)
        print(f'Processing body {body_id}')
        for a, b in tqdm(pairs, desc="Aligning", unit="pc"):
            scan_path1, reg_path1, gt_path1 = build_paths(args.raw_root, a)
            scan_path2, reg_path2, gt_path2 = build_paths(args.raw_root, b)

            scan1, normals1 = load_mesh_vertices_normals(scan_path1)
            scan2, normals2 = load_mesh_vertices_normals(scan_path2)
            reg1 = read_ply(str(reg_path1))["vertices"].astype(np.float32)
            reg2 = read_ply(str(reg_path2))["vertices"].astype(np.float32)
            gt_vert1 = load_gt_vertices(gt_path1)
            gt_vert2 = load_gt_vertices(gt_path2)

            corres = align(scan1, scan2, reg1, reg2, gt_vert1, gt_vert2)
            xyz0 = scan1[gt_vert1]
            normal0 = normals1[gt_vert1]
            xyz1 = scan2
            normal1 = normals2

            out_path = out_root / f"tr_{a:03d}_{b:03d}.npz"
            np.savez(
                out_path,
                xyz0=xyz0.astype(np.float32),
                xyz1=xyz1.astype(np.float32),
                normal0=normal0.astype(np.float32),
                normal1=normal1.astype(np.float32),
                corres=corres.astype(np.int64),
            )
            saved += 1

    print(f"Saved {saved} pair files to {out_root}")


if __name__ == "__main__":
    main()
