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
from typing import Optional


def run_icp_on_pair(pair_npz: str, out_dir: str, threshold: float = 0.05, init=np.eye(4)) -> str:
	"""Run Open3D point-to-point ICP on a single fixed-N pair file.

	Args:
		pair_npz: path to fixed-N pair .npz (expects `src_coords`, `tgt_coords`)
		out_dir: output directory to save results
		threshold: max correspondence distance for ICP
		init: initial 4x4 transformation

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

	src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts.astype(np.float64)))
	tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts.astype(np.float64)))

	# estimate normals for robustness (optional)
	try:
		src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
		tgt_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=threshold * 2, max_nn=30))
	except Exception:
		pass

	res = o3d.pipelines.registration.registration_icp(
		src_pcd, tgt_pcd, threshold, init,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
	)

	out_path = os.path.join(out_dir, os.path.basename(pair_npz).replace('.npz', '_icp_result.npz'))
	np.savez_compressed(out_path, trans=res.transformation.astype(np.float32), fitness=float(res.fitness), inlier_rmse=float(res.inlier_rmse))
	return out_path


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Run ICP baseline on fixed-N pair .npz')
	parser.add_argument('--pair', required=True, help='Fixed-N pair .npz file')
	parser.add_argument('--out', default='icp_out', help='Output directory')
	parser.add_argument('--threshold', type=float, default=0.05, help='ICP correspondence threshold')
	args = parser.parse_args()

	try:
		out = run_icp_on_pair(args.pair, args.out, threshold=args.threshold)
		print('ICP result saved to', out)
	except Exception as e:
		print('ICP failed:', str(e))
		raise SystemExit(2)

