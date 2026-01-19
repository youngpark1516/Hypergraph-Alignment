"""ICP (Iterative Closest Point) baseline wrapper using Open3D.

Provides a small helper to run point-to-point ICP on a preprocessed fixed-N
pair `.npz` (expects `src_coords` and `tgt_coords` arrays) and save the
resulting transformation and metrics.

This is intended as a quick baseline and smoke-test for the preprocessing
pipeline; it is not a full training/evaluation integration.
"""

import os
import numpy as np
import open3d as o3d
from typing import Optional, Tuple

from src.project.utils.build_correspondences import fpfh_knn_correspondences

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
			# Treat as kNN list per source; use the closest neighbor.
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
	print(src_pts.shape, tgt_pts.shape)
	print(len(np.unique(src_pts, axis=0)), len(np.unique(tgt_pts, axis=0)))
	src_trans = (trans[:3, :3] @ src_pts.T + trans[:3, 3:4]).T
	geoms = [
		_make_colored_pcd(src_pts, (0.1, 0.6, 0.9)), # src: blue
		_make_colored_pcd(tgt_pts, (0.9, 0.2, 0.2)), # tgt: red
		_make_colored_pcd(src_trans, (0.2, 0.8, 0.2)), # src_transformed: green
	]
	o3d.visualization.draw_geometries(geoms, window_name="ICP Alignment (src / tgt / src_transformed)")


def run_icp_on_pair(
	pair_npz: str,
	out_dir: str,
	threshold: float = 10.0,
	init: Optional[np.ndarray] = None,
	init_from_fpfh: bool = False,
	corres_key: str = "corres",
	visualize: bool = False,
) -> str:
	"""Run Open3D point-to-point ICP on a single fixed-N pair file.

	Args:
		pair_npz: path to fixed-N pair .npz (expects `src_coords`, `tgt_coords`)
		out_dir: output directory to save results
		threshold: max correspondence distance for ICP
		init: initial 4x4 transformation (overridden if init_from_fpfh is True)
		init_from_fpfh: initialize transform using FPFH feature correspondences
		corres_key: key for correspondence indices in the pair file
		visualize: show src, tgt, and transformed src point clouds

	Returns:
		path to saved npz with fields `trans`, `fitness`, `inlier_rmse`.
	"""
	if not os.path.exists(pair_npz):
		raise FileNotFoundError(pair_npz)
	os.makedirs(out_dir, exist_ok=True)

	data = np.load(pair_npz)
	if 'src_coords' in data and 'tgt_coords' in data:
		src_pts = data['src_coords']
		tgt_pts = data['tgt_coords']
	else:
		# fallback: try `xyz0` / `xyz1` or `points` keys
		if 'xyz0' in data and 'xyz1' in data:
			src_pts = data['xyz0']
			tgt_pts = data['xyz1']
		elif 'points' in data:
			pts = data['points']
			# if only single scan present, do identity
			raise ValueError('Pair file lacks `src_coords`/`tgt_coords` and `xyz0`/`xyz1`.')
		else:
			raise ValueError('Pair file missing required point arrays')

	if init is None:
		init_mat = np.eye(4, dtype=np.float64)
	else:
		init_mat = np.asarray(init, dtype=np.float64)
		if init_mat.shape != (4, 4):
			raise ValueError("init must be a 4x4 matrix.")

	if init_from_fpfh:
		if "features0" not in data or "features1" not in data:
			raise ValueError("Pair file missing FPFH features: expected features0 and features1.")
		corres = fpfh_knn_correspondences(data["features0"], data["features1"], k=1)
		print(f"Found {corres.shape[0]} FPFH correspondences for initialization.")
		src_sel, tgt_sel = _select_correspondence_pairs(src_pts, tgt_pts, corres)
		init_mat = _estimate_rigid_transform(src_sel, tgt_sel)
	print('Initial transform from FPFH correspondences:\n', init_mat)

	src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts.astype(np.float64)))
	tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts.astype(np.float64)))

	# estimate normals for robustness (optional)
	try:
		src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
		tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
	except Exception:
		pass

	res = o3d.pipelines.registration.registration_icp(
		src_pcd, tgt_pcd, threshold, init_mat,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
	)
	print('Final ICP transformation:\n', res.transformation)
	print(f'ICP fitness: {res.fitness:.4f}, inlier RMSE: {res.inlier_rmse:.4f}')
	if visualize:
		_visualize_alignment(src_pts, tgt_pts, res.transformation)

	out_path = os.path.join(out_dir, os.path.basename(pair_npz).replace('.npz', '_icp_result.npz'))
	np.savez_compressed(out_path, trans=res.transformation.astype(np.float32), fitness=float(res.fitness), inlier_rmse=float(res.inlier_rmse))
	return out_path


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Run ICP baseline on fixed-N pair .npz')
	parser.add_argument('--pair', required=True, help='Fixed-N pair .npz file')
	parser.add_argument('--out', default='results/icp_out', help='Output directory')
	parser.add_argument('--threshold', type=float, default=float("inf"), help='ICP correspondence threshold')
	parser.add_argument('--init-from-fpfh', action='store_true', help='Initialize transform from FPFH feature correspondences')
	parser.add_argument('--corres-key', default='corres', help='Key for correspondence indices when using --init-from-corres')
	parser.add_argument('--visualize', action='store_true', help='Plot src, tgt, and transformed src point clouds')
	args = parser.parse_args()

	try:
		out = run_icp_on_pair(
			args.pair,
			args.out,
			threshold=args.threshold,
			init_from_fpfh=args.init_from_fpfh,
			corres_key=args.corres_key,
			visualize=args.visualize,
		)
		print('ICP result saved to', out)
	except Exception as e:
		print('ICP failed:', str(e))
		raise SystemExit(2)
