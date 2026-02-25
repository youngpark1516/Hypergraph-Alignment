"""ICP (Iterative Closest Point) baseline wrapper using Open3D.

Supports single-pair and batched execution over FAUST and PartNet FPFH datasets.

Usage examples:
    # Single pair
    python -m src.project.models.baselines.icp --pair data/faust/fpfh/000_001.npz --init-from-fpfh

    # Batch over FAUST
    python -m src.project.models.baselines.icp --dataset faust --data-dir data/faust/fpfh --out results/icp_faust_fpfh

    # Batch over PartNet (flat or category subdirs)
    python -m src.project.models.baselines.icp --dataset partnet --data-dir data/partnet/fpfh_rigid --out results/icp_partnet_fpfh

    # Both
    python -m src.project.models.baselines.icp --dataset both
"""

import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

try:
    from src.project.utils.build_correspondences import fpfh_knn_correspondences
except ImportError:
    fpfh_knn_correspondences = None  # only needed when --init-from-fpfh is used with external features

def _estimate_rigid_transform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
	"""Estimate rigid transform (R,t) mapping src_pts -> tgt_pts via SVD."""
	if src_pts.shape != tgt_pts.shape:
		raise ValueError("src_pts and tgt_pts must have the same shape.")
	if src_pts.ndim != 2 or src_pts.shape[1] != 3:
		raise ValueError("Expected Nx3 arrays for src_pts and tgt_pts.")
	if src_pts.shape[0] < 3:
		raise ValueError("Need at least 3 correspondences for a stable transform.")

	src = src_pts.astype(np.float64, copy=False)
	tgt = tgt_pts.astype(np.float64, copy=False)

	src_centroid = src.mean(axis=0)
	tgt_centroid = tgt.mean(axis=0)
	src_centered = src - src_centroid
	tgt_centered = tgt - tgt_centroid

	H = src_centered.T @ tgt_centered
	U, _, Vt = np.linalg.svd(H)
	R = Vt.T @ U.T
	if np.linalg.det(R) < 0:
		Vt[-1, :] *= -1
		R = Vt.T @ U.T
	t = tgt_centroid - R @ src_centroid

	trans = np.eye(4, dtype=np.float64)
	trans[:3, :3] = R
	trans[:3, 3] = t
	return trans


def _select_correspondence_pairs(
	src_pts: np.ndarray,
	tgt_pts: np.ndarray,
	corres: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	corres = np.asarray(corres)
	if corres.ndim == 1:
		src_idx = np.arange(corres.shape[0], dtype=np.int64)
		tgt_idx = corres.astype(np.int64)
	elif corres.ndim == 2:
		if corres.shape[1] == 2:
			src_idx = corres[:, 0].astype(np.int64)
			tgt_idx = corres[:, 1].astype(np.int64)
		else:
			src_idx = np.arange(corres.shape[0], dtype=np.int64)
			tgt_idx = corres[:, 0].astype(np.int64)
	else:
		raise ValueError("corres must be 1D or 2D.")

	if src_idx.size == 0:
		raise ValueError("corres is empty.")
	if np.any(src_idx < 0) or np.any(src_idx >= src_pts.shape[0]):
		raise ValueError("corres contains source indices outside src_pts range.")
	if np.any(tgt_idx < 0) or np.any(tgt_idx >= tgt_pts.shape[0]):
		raise ValueError("corres contains target indices outside tgt_pts range.")

	return src_pts[src_idx], tgt_pts[tgt_idx]


def _make_colored_pcd(points: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
	colors = np.tile(np.asarray(color, dtype=np.float64), (points.shape[0], 1))
	pcd.colors = o3d.utility.Vector3dVector(colors)
	return pcd


def _visualize_alignment(src_pts: np.ndarray, tgt_pts: np.ndarray, trans: np.ndarray) -> None:
	src_trans = (trans[:3, :3] @ src_pts.T + trans[:3, 3:4]).T
	geoms = [
		_make_colored_pcd(src_pts, (0.1, 0.6, 0.9)),   # src: blue
		_make_colored_pcd(tgt_pts, (0.9, 0.2, 0.2)),   # tgt: red
		_make_colored_pcd(src_trans, (0.2, 0.8, 0.2)), # src_transformed: green
	]
	o3d.visualization.draw_geometries(geoms, window_name="ICP Alignment (src / tgt / src_transformed)")


def run_icp_on_pair(
	pair_npz: str,
	out_dir: str,
	threshold: float = float("inf"),
	init: Optional[np.ndarray] = None,
	init_from_fpfh: bool = True,
	corres_key: str = "corres",
	visualize: bool = False,
	verbose: bool = False,
) -> Dict:
	"""Run Open3D point-to-point ICP on a single fixed-N pair file.

	Args:
		pair_npz: path to .npz with `xyz0`/`xyz1` (or `src_coords`/`tgt_coords`) arrays.
		          For FPFH init, the file should also contain `features0`/`features1`
		          or a `corres` key with ground-truth / pre-computed correspondences.
		out_dir: output directory to save results
		threshold: max correspondence distance for ICP (inf = unconstrained)
		init: initial 4x4 transformation; overridden when init_from_fpfh is True
		init_from_fpfh: initialise transform from stored correspondences or FPFH kNN
		corres_key: key for pre-computed correspondence indices in the pair file
		visualize: show point clouds after alignment
		verbose: print transform / metrics

	Returns:
		dict with keys: out_path, fitness, inlier_rmse, pair_name
	"""
	if not os.path.exists(pair_npz):
		raise FileNotFoundError(pair_npz)
	os.makedirs(out_dir, exist_ok=True)

	data = np.load(pair_npz)

	# Load point clouds — prefer xyz0/xyz1, fall back to src_coords/tgt_coords
	if 'xyz0' in data and 'xyz1' in data:
		src_pts = data['xyz0']
		tgt_pts = data['xyz1']
	elif 'src_coords' in data and 'tgt_coords' in data:
		src_pts = data['src_coords']
		tgt_pts = data['tgt_coords']
	else:
		raise ValueError(f"Pair file {pair_npz} missing point arrays (expected xyz0/xyz1 or src_coords/tgt_coords)")

	# Build initial transform
	if init_from_fpfh:
		if corres_key in data:
			# Use pre-computed correspondences stored in the file
			src_sel, tgt_sel = _select_correspondence_pairs(src_pts, tgt_pts, data[corres_key])
			init_mat = _estimate_rigid_transform(src_sel, tgt_sel)
		elif 'features0' in data and 'features1' in data:
			# Fall back to kNN on FPFH features
			if fpfh_knn_correspondences is None:
				raise ImportError("fpfh_knn_correspondences not available; ensure src.project.utils is on path")
			corres = fpfh_knn_correspondences(data["features0"], data["features1"], k=1)
			src_sel, tgt_sel = _select_correspondence_pairs(src_pts, tgt_pts, corres)
			init_mat = _estimate_rigid_transform(src_sel, tgt_sel)
		else:
			raise ValueError(f"init_from_fpfh=True but no '{corres_key}' or features0/features1 in {pair_npz}")
	elif init is not None:
		init_mat = np.asarray(init, dtype=np.float64)
		if init_mat.shape != (4, 4):
			raise ValueError("init must be a 4x4 matrix.")
	else:
		init_mat = np.eye(4, dtype=np.float64)

	src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts.astype(np.float64)))
	tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts.astype(np.float64)))

	try:
		src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
		tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
	except Exception:
		pass

	res = o3d.pipelines.registration.registration_icp(
		src_pcd, tgt_pcd, threshold, init_mat,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
	)

	if verbose:
		print(f"[{os.path.basename(pair_npz)}] fitness={res.fitness:.4f}  inlier_rmse={res.inlier_rmse:.6f}")

	if visualize:
		_visualize_alignment(src_pts, tgt_pts, res.transformation)

	pair_name = os.path.splitext(os.path.basename(pair_npz))[0]
	out_path = os.path.join(out_dir, pair_name + "_icp_result.npz")
	np.savez_compressed(
		out_path,
		trans=res.transformation.astype(np.float32),
		fitness=float(res.fitness),
		inlier_rmse=float(res.inlier_rmse),
	)
	return {"out_path": out_path, "fitness": float(res.fitness), "inlier_rmse": float(res.inlier_rmse), "pair_name": pair_name}


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def _collect_npz_files(data_dir: str) -> List[str]:
	"""Recursively collect all .npz files under data_dir (handles flat dirs and category subdirs)."""
	files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
	return files


def _worker(args):
	pair_npz, out_dir, threshold, init_from_fpfh = args
	try:
		return run_icp_on_pair(pair_npz, out_dir, threshold=threshold, init_from_fpfh=init_from_fpfh, verbose=False)
	except Exception as e:
		return {"pair_name": os.path.splitext(os.path.basename(pair_npz))[0], "error": str(e), "fitness": None, "inlier_rmse": None}


def batch_run_icp(
	data_dir: str,
	out_dir: str,
	threshold: float = float("inf"),
	init_from_fpfh: bool = True,
	workers: int = 4,
	skip_existing: bool = True,
) -> str:
	"""Run ICP over all .npz pairs in data_dir (recursively).

	Args:
		data_dir: directory containing .npz pair files (can have category subdirs)
		out_dir: output directory for result .npz files and summary CSV
		threshold: ICP correspondence threshold
		init_from_fpfh: initialise from stored correspondences / FPFH features
		workers: number of parallel worker processes
		skip_existing: skip pairs whose output already exists

	Returns:
		path to the summary CSV file
	"""
	files = _collect_npz_files(data_dir)
	if not files:
		raise FileNotFoundError(f"No .npz files found under {data_dir}")

	os.makedirs(out_dir, exist_ok=True)

	if skip_existing:
		before = len(files)
		files = [f for f in files if not os.path.exists(
			os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + "_icp_result.npz")
		)]
		print(f"Skipping {before - len(files)} already-done pairs; {len(files)} remaining.")

	print(f"Running ICP on {len(files)} pairs from {data_dir} -> {out_dir}  (workers={workers})")

	work = [(f, out_dir, threshold, init_from_fpfh) for f in files]
	results = []

	if workers > 1:
		with ProcessPoolExecutor(max_workers=workers) as pool:
			futures = {pool.submit(_worker, w): w[0] for w in work}
			done = 0
			for future in as_completed(futures):
				r = future.result()
				results.append(r)
				done += 1
				if done % 50 == 0 or done == len(files):
					print(f"  {done}/{len(files)} done")
	else:
		for i, w in enumerate(work):
			r = _worker(w)
			results.append(r)
			if (i + 1) % 50 == 0 or (i + 1) == len(files):
				print(f"  {i+1}/{len(files)} done")

	# Write summary CSV
	csv_path = os.path.join(out_dir, "summary.csv")
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["pair_name", "fitness", "inlier_rmse", "error"])
		writer.writeheader()
		for r in sorted(results, key=lambda x: x["pair_name"]):
			writer.writerow({
				"pair_name": r["pair_name"],
				"fitness": r.get("fitness", ""),
				"inlier_rmse": r.get("inlier_rmse", ""),
				"error": r.get("error", ""),
			})

	ok = [r for r in results if r.get("fitness") is not None]
	if ok:
		mean_fitness = np.mean([r["fitness"] for r in ok])
		mean_rmse = np.mean([r["inlier_rmse"] for r in ok])
		print(f"\nDone. {len(ok)}/{len(results)} succeeded.")
		print(f"  mean fitness:     {mean_fitness:.4f}")
		print(f"  mean inlier RMSE: {mean_rmse:.6f}")
		print(f"  summary CSV:      {csv_path}")
	else:
		print("All pairs failed — check errors in summary CSV.")

	return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(
		description="Run FPFH-initialised ICP baseline on FAUST / PartNet datasets.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	mode = parser.add_mutually_exclusive_group(required=True)
	mode.add_argument('--pair', help='Single .npz pair file (single-pair mode)')
	mode.add_argument('--dataset', choices=['faust', 'partnet', 'both'],
		help='Dataset to batch over (batch mode)')

	# Batch options
	parser.add_argument('--faust-dir',   default='data/faust/fpfh',         help='FAUST fpfh dir')
	parser.add_argument('--partnet-dir', default='data/partnet/fpfh_rigid',  help='PartNet fpfh_rigid dir')
	parser.add_argument('--faust-out',   default='results/icp_faust_fpfh',   help='FAUST output dir')
	parser.add_argument('--partnet-out', default='results/icp_partnet_fpfh', help='PartNet output dir')
	parser.add_argument('--workers', type=int, default=4, help='Parallel worker processes')
	parser.add_argument('--no-skip', action='store_true', help='Re-run pairs that already have output')

	# Shared options
	parser.add_argument('--threshold', type=float, default=float("inf"), help='ICP correspondence threshold')
	parser.add_argument('--no-fpfh-init', action='store_true', help='Use identity init instead of FPFH/corres init')

	# Single-pair extras
	parser.add_argument('--out', default='results/icp_out', help='Output dir (single-pair mode)')
	parser.add_argument('--visualize', action='store_true', help='Visualise alignment (single-pair mode)')
	parser.add_argument('--verbose', action='store_true')
	args = parser.parse_args()

	init_from_fpfh = not args.no_fpfh_init

	if args.pair:
		# Single-pair mode
		result = run_icp_on_pair(
			args.pair,
			args.out,
			threshold=args.threshold,
			init_from_fpfh=init_from_fpfh,
			visualize=args.visualize,
			verbose=True,
		)
		print('ICP result saved to', result["out_path"])
	else:
		# Batch mode
		skip_existing = not args.no_skip
		if args.dataset in ('faust', 'both'):
			batch_run_icp(args.faust_dir, args.faust_out,
				threshold=args.threshold, init_from_fpfh=init_from_fpfh,
				workers=args.workers, skip_existing=skip_existing)
		if args.dataset in ('partnet', 'both'):
			batch_run_icp(args.partnet_dir, args.partnet_out,
				threshold=args.threshold, init_from_fpfh=init_from_fpfh,
				workers=args.workers, skip_existing=skip_existing)
