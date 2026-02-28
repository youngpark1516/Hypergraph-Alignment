"""ICP (Iterative Closest Point) baseline wrapper using Open3D.

Supports single-pair and batched execution over FAUST and PartNet FPFH datasets.

Legacy usage:
    # Single pair
    python -m src.project.models.baselines.icp --pair data/faust/fpfh/000_001.npz

    # Batch (old-style)
    python -m src.project.models.baselines.icp --dataset faust --data-dir data/faust/fpfh --out results/icp_faust_fpfh

Spectral-ICP (HyperGNN-style, no model required):
    python -m src.project.models.baselines.icp \\
        --use_features true --dataset faust --root data/faust/fpfh \\
        --config src/project/config/hypergnn_partnet_eval.yaml \\
        --generation_method spectral-2 --full_data

    python -m src.project.models.baselines.icp \\
        --use_features true --dataset partnet --root data/partnet/fpfh_rigid \\
        --config src/project/config/hypergnn_partnet_eval.yaml \\
        --generation_method spectral-2 --full_data
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

# Torch and project imports are lazy (inside functions) so the file stays
# importable even without a GPU / full project install.
_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass

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


def _compute_l1_error(src_pts: np.ndarray, tgt_pts: np.ndarray, trans: np.ndarray) -> float:
	"""Mean L1 (MAE) distance from each transformed src point to its nearest tgt point."""
	src_transformed = (trans[:3, :3] @ src_pts.astype(np.float64).T + trans[:3, 3:4]).T
	tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_pts.astype(np.float64)))
	kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
	distances = []
	for pt in src_transformed:
		_, _, dist_sq = kdtree.search_knn_vector_3d(pt, 1)
		distances.append(np.sqrt(max(dist_sq[0], 0.0)))
	return float(np.mean(distances))


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

	l1_before = _compute_l1_error(src_pts, tgt_pts, init_mat)

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

	l1_after = _compute_l1_error(src_pts, tgt_pts, res.transformation)

	if verbose:
		print(f"[{os.path.basename(pair_npz)}] fitness={res.fitness:.4f}  inlier_rmse={res.inlier_rmse:.6f}")
		print(f"  L1 before: {l1_before:.6f}  L1 after: {l1_after:.6f}  (delta: {l1_before - l1_after:+.6f})")

	if visualize:
		_visualize_alignment(src_pts, tgt_pts, res.transformation)

	pair_name = os.path.splitext(os.path.basename(pair_npz))[0]
	out_path = os.path.join(out_dir, pair_name + "_icp_result.npz")
	np.savez_compressed(
		out_path,
		trans=res.transformation.astype(np.float32),
		fitness=float(res.fitness),
		inlier_rmse=float(res.inlier_rmse),
		l1_before=float(l1_before),
		l1_after=float(l1_after),
	)
	return {
		"out_path": out_path,
		"fitness": float(res.fitness),
		"inlier_rmse": float(res.inlier_rmse),
		"l1_before": float(l1_before),
		"l1_after": float(l1_after),
		"pair_name": pair_name,
	}


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
		writer = csv.DictWriter(f, fieldnames=["pair_name", "fitness", "inlier_rmse", "l1_before", "l1_after", "error"])
		writer.writeheader()
		for r in sorted(results, key=lambda x: x["pair_name"]):
			writer.writerow({
				"pair_name": r["pair_name"],
				"fitness": r.get("fitness", ""),
				"inlier_rmse": r.get("inlier_rmse", ""),
				"l1_before": r.get("l1_before", ""),
				"l1_after": r.get("l1_after", ""),
				"error": r.get("error", ""),
			})

	ok = [r for r in results if r.get("fitness") is not None]
	if ok:
		mean_fitness = np.mean([r["fitness"] for r in ok])
		mean_rmse = np.mean([r["inlier_rmse"] for r in ok])
		mean_l1_before = np.mean([r["l1_before"] for r in ok])
		mean_l1_after = np.mean([r["l1_after"] for r in ok])
		print(f"\nDone. {len(ok)}/{len(results)} succeeded.")
		print(f"  mean fitness:     {mean_fitness:.4f}")
		print(f"  mean inlier RMSE: {mean_rmse:.6f}")
		print(f"  mean L1 before:   {mean_l1_before:.6f}")
		print(f"  mean L1 after:    {mean_l1_after:.6f}")
		print(f"  summary CSV:      {csv_path}")
	else:
		print("All pairs failed — check errors in summary CSV.")

	return csv_path


# ---------------------------------------------------------------------------
# Spectral-ICP pipeline  –  mirrors HyperGNN trainer evaluate loop.
# M and H are built from raw FPFH features; no trained model required.
# ---------------------------------------------------------------------------

def _spectral_build_M(corr_feat_t, sigma: float):
	"""Feature-similarity matrix M from [1, N, D] correspondence features."""
	import torch
	import torch.nn.functional as F
	n = corr_feat_t.shape[1]
	fn = F.normalize(corr_feat_t, dim=-1)
	M = torch.bmm(fn, fn.transpose(1, 2))
	M = torch.clamp(1.0 - (1.0 - M) / (sigma ** 2), min=0.0, max=1.0)
	M[:, torch.arange(n), torch.arange(n)] = 0.0
	return M


def _spectral_build_fcg(src_kp, tgt_kp, sigma_d: float, fcg_k_ratio: float = 0.1):
	"""Spatial Feature Compatibility Graph H (FCG) from keypoint coords. Returns [1, N, N]."""
	import torch
	bs, n, _ = src_kp.shape
	k = max(1, int(n * fcg_k_ratio))
	chunk = 128
	FCG = torch.zeros(bs, n, n, device=src_kp.device, dtype=src_kp.dtype)
	with torch.no_grad():
		for s in range(0, n, chunk):
			e = min(s + chunk, n)
			sd = ((src_kp[:, s:e, None, :] - src_kp[:, None, :, :]) ** 2).sum(-1) ** 0.5
			td = ((tgt_kp[:, s:e, None, :] - tgt_kp[:, None, :, :]) ** 2).sum(-1) ** 0.5
			FCG[:, s:e, :] = torch.clamp(1.0 - (sd - td) ** 2 / (sigma_d ** 2), min=0.0)
		FCG[:, torch.arange(n), torch.arange(n)] = 0.0
		topk_val, _ = torch.topk(FCG, k, dim=2)
		thresh = topk_val.reshape(bs, -1).mean(dim=1, keepdim=True).unsqueeze(2)
		H = (FCG >= thresh).float()
		H[:, torch.arange(n), torch.arange(n)] = 0.0
	return H


def _spectral_build_knn_idx(corr_feat_t, H, k: int = 10):
	"""Build kNN index masked by H for two-stage spectral matching. Returns [1, N, k]."""
	import torch
	import torch.nn.functional as F
	n = corr_feat_t.shape[1]
	fn = F.normalize(corr_feat_t, dim=-1)
	dist = 1.0 - torch.bmm(fn, fn.transpose(1, 2))
	masked = dist * H.float() + (1.0 - H.float()) * 1e9
	k_eff = min(k, n - 1)
	return torch.topk(masked, k_eff, largest=False, dim=2)[1]


def _spectral_select_correspondences(
	M, H, seeds, corr_feat_t, src_indices_t, tgt_indices_t,
	generation_method: str, generation_min_score,
	num_iterations: int = 10, max_ratio: float = 0.4, max_ratio_seeds: float = 0.4,
):
	"""Apply spectral / greedy generation method and return (selected_idx, selected_mask)."""
	import torch
	try:
		from project.models.hypothesis_generation import (
			spectral_matching_greedy, two_stage_spectral_matching_greedy, greedy_compatibility_expansion,
		)
	except ImportError:
		from src.project.models.hypothesis_generation import (
			spectral_matching_greedy, two_stage_spectral_matching_greedy, greedy_compatibility_expansion,
		)

	if generation_method == "spectral-2":
		k = min(10, corr_feat_t.shape[1] - 1)
		knn_idx = _spectral_build_knn_idx(corr_feat_t, H, k=k)
		return two_stage_spectral_matching_greedy(
			M.squeeze(0), seeds.squeeze(0), knn_idx.squeeze(0),
			num_iterations=num_iterations, max_ratio=max_ratio,
			max_ratio_seeds=max_ratio_seeds, min_score=generation_min_score,
		)
	elif generation_method == "spectral":
		return spectral_matching_greedy(
			M.squeeze(0), max_ratio=max_ratio,
			num_iterations=num_iterations, min_score=generation_min_score,
		)
	elif generation_method == "greedy":
		confidence = M.squeeze(0).sum(dim=1)  # feature-degree as proxy confidence
		return greedy_compatibility_expansion(
			M.squeeze(0), confidence, seeds.squeeze(0), max_ratio=max_ratio,
			src_idx=src_indices_t.squeeze(0) if src_indices_t is not None else None,
			tgt_idx=tgt_indices_t.squeeze(0) if tgt_indices_t is not None else None,
		)
	else:
		raise ValueError(f"Unknown generation_method: {generation_method}")


def _run_spectral_icp_on_batch(
	batch,
	sigma: float,
	sigma_d: float,
	seed_ratio: float,
	generation_method: str,
	generation_min_score,
	icp_threshold: float,
	matching_dir: str,
	feature_dim: int,
	num_iterations: int = 10,
) -> Dict:
	"""Run spectral correspondence selection + ICP on one DataLoader batch (bs=1).

	Mirrors the HyperGNN trainer evaluate loop but replaces the model with
	FPFH-feature-based M / H construction.
	"""
	import torch
	import torch.nn.functional as F

	(corr_pos, src_keypts, tgt_keypts, src_normal, tgt_normal,
	 src_indices, tgt_indices, gt_labels, file_name) = batch

	assert corr_pos.shape[0] == 1, "batch_size must be 1 for spectral-ICP"
	n = corr_pos.shape[1]

	# Extract features from corr_pos: [1, N, 6+2D]  (src_xyz|tgt_xyz|src_feat|tgt_feat)
	if feature_dim > 0 and corr_pos.shape[-1] >= 6 + 2 * feature_dim:
		src_feat = corr_pos[:, :, 6:6 + feature_dim].float()
		tgt_feat = corr_pos[:, :, 6 + feature_dim:6 + 2 * feature_dim].float()
		corr_feat_t = torch.cat([src_feat, tgt_feat], dim=-1)  # [1, N, 2D]
	else:
		corr_feat_t = corr_pos[:, :, :6].float()
		src_feat = corr_feat_t[:, :, :3]
		tgt_feat = corr_feat_t[:, :, 3:]

	# Proxy confidence: per-correspondence feature similarity
	sf = F.normalize(src_feat.squeeze(0), dim=-1)
	tf = F.normalize(tgt_feat.squeeze(0), dim=-1)
	confidence = (sf * tf).sum(dim=-1).unsqueeze(0)  # [1, N]

	# Seeds = top-confidence correspondences (mirrors graph_filter without model)
	num_seeds = max(1, int(n * seed_ratio))
	seeds = torch.argsort(confidence, dim=1, descending=True)[:, :num_seeds]  # [1, S]

	# Build M (feature similarity) and H (spatial FCG)
	src_kp = src_keypts.float()
	tgt_kp = tgt_keypts.float()
	with torch.no_grad():
		M = _spectral_build_M(corr_feat_t, sigma)
		H = _spectral_build_fcg(src_kp, tgt_kp, sigma_d)

	# Spectral / greedy correspondence selection
	with torch.no_grad():
		sel_idx, sel_mask = _spectral_select_correspondences(
			M=M, H=H, seeds=seeds, corr_feat_t=corr_feat_t,
			src_indices_t=src_indices, tgt_indices_t=tgt_indices,
			generation_method=generation_method, generation_min_score=generation_min_score,
			num_iterations=num_iterations,
		)
		sel_idx = sel_idx[sel_mask]

	selected_total = int(sel_idx.numel())
	if selected_total > 0:
		selected_true = int((gt_labels.squeeze(0)[sel_idx] > 0.5).sum().item())
	else:
		selected_true = 0
	true_ratio = (selected_true / selected_total) if selected_total > 0 else 0.0
	print(f"  selected true correspondences: {selected_true}/{selected_total}  (precision={100.0 * true_ratio:.2f}%)")

	# Resolve file stem
	fn_single = file_name[0] if isinstance(file_name, (list, tuple)) else str(file_name)
	stem = os.path.splitext(os.path.basename(fn_single))[0]

	# Correspondence error metrics (same as HyperGNN trainer)
	initial_abs_err = initial_l2_err = selected_abs_err = selected_l2_err = float("nan")
	src_all_np = src_indices.squeeze(0).cpu().numpy().astype(np.int64)
	tgt_all_np = tgt_indices.squeeze(0).cpu().numpy().astype(np.int64)
	try:
		eval_data = np.load(fn_single)
		xyz1_np = eval_data["xyz1"].astype(np.float32)
		corres_np = eval_data["corres"].astype(np.int64)
		diff_all = xyz1_np[tgt_all_np] - xyz1_np[corres_np[src_all_np]]
		initial_abs_err = float(np.abs(diff_all).mean())
		initial_l2_err = float(np.linalg.norm(diff_all, axis=1).mean())
		if selected_total > 0:
			sel_np = sel_idx.detach().cpu().numpy()
			diff_sel = xyz1_np[tgt_all_np[sel_np]] - xyz1_np[corres_np[src_all_np[sel_np]]]
			selected_abs_err = float(np.abs(diff_sel).mean())
			selected_l2_err = float(np.linalg.norm(diff_sel, axis=1).mean())
			print(f"  selected target distance error: mean|y_pred-y_gt|={selected_abs_err:.6f}, mean_l2={selected_l2_err:.6f}")
		print(f"  initial dataset target distance error: mean|y_pred-y_gt|={initial_abs_err:.6f}, mean_l2={initial_l2_err:.6f}")
	except Exception as exc:
		print(f"  target distance error unavailable: {exc}")

	# Save per-pair matching CSV (same format as HyperGNN trainer)
	if selected_total > 0:
		sel_np = sel_idx.detach().cpu().numpy()
		src_sel_out = src_all_np[sel_np]
		tgt_sel_out = tgt_all_np[sel_np]
		os.makedirs(matching_dir, exist_ok=True)
		with open(os.path.join(matching_dir, f"{stem}.csv"), "w", newline="") as f:
			w = csv.writer(f)
			w.writerow(["src_idx", "tgt_idx"])
			for s, t in zip(src_sel_out.tolist(), tgt_sel_out.tolist()):
				w.writerow([s, t])

	# Kabsch init from selected correspondences → ICP
	src_kp_np = src_kp.squeeze(0).numpy()
	tgt_kp_np = tgt_kp.squeeze(0).numpy()
	if selected_total >= 3:
		sel_np = sel_idx.detach().cpu().numpy()
		init_mat = _estimate_rigid_transform(src_kp_np[sel_np], tgt_kp_np[sel_np])
	elif n >= 3:
		init_mat = _estimate_rigid_transform(src_kp_np, tgt_kp_np)
	else:
		init_mat = np.eye(4, dtype=np.float64)

	l1_before = _compute_l1_error(src_kp_np, tgt_kp_np, init_mat)
	src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_kp_np.astype(np.float64)))
	tgt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt_kp_np.astype(np.float64)))
	res = o3d.pipelines.registration.registration_icp(
		src_pcd, tgt_pcd, icp_threshold, init_mat,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
	)
	l1_after = _compute_l1_error(src_kp_np, tgt_kp_np, res.transformation)
	print(f"  ICP: fitness={res.fitness:.4f}  inlier_rmse={res.inlier_rmse:.6f}  L1_before={l1_before:.6f}  L1_after={l1_after:.6f}")

	return {
		"pair_name": stem,
		"selected_total": selected_total,
		"selected_true": selected_true,
		"precision": true_ratio,
		"initial_abs_err": initial_abs_err,
		"initial_l2_err": initial_l2_err,
		"selected_abs_err": selected_abs_err,
		"selected_l2_err": selected_l2_err,
		"fitness": float(res.fitness),
		"inlier_rmse": float(res.inlier_rmse),
		"l1_before": l1_before,
		"l1_after": l1_after,
	}


def batch_spectral_icp(args) -> str:
	"""Run spectral-ICP pipeline over a full dataset, matching HyperGNN eval output.

	Args:
		args: namespace from get_config() with fields:
		      root, dataset, generation_method, generation_min_score,
		      num_node, use_features, feature_dim, full_data,
		      val_ratio, split_seed, skip_gt_trans, inlier_threshold,
		      seed_ratio, num_workers
	Returns:
		path to summary CSV
	"""
	import sys
	from pathlib import Path

	_src_root = str(Path(__file__).resolve().parents[3])
	if _src_root not in sys.path:
		sys.path.insert(0, _src_root)

	import torch  # noqa: F401
	from torch.utils.data import DataLoader

	try:
		from project.dataset import AlignmentDataset, FullAlignmentDataset
	except ImportError:
		from src.project.dataset import AlignmentDataset, FullAlignmentDataset

	from pathlib import Path as _Path
	root = _Path(args.root)
	files = sorted(root.glob("**/*.npz"))
	if not files:
		raise FileNotFoundError(f"No .npz files found under {root}")
	print(f"Found {len(files)} files under {root}")

	# Train/val split identical to HyperGNN build_dataloaders
	rng = np.random.default_rng(args.split_seed)
	indices = np.arange(len(files))
	rng.shuffle(indices)
	split = int(len(files) * (1.0 - args.val_ratio))
	val_files = [files[i] for i in indices[split:]]
	if not val_files:
		val_files = list(files)
		print("Warning: val split is empty — using all files.")

	feature_dim = int(getattr(args, "feature_dim", 33))
	use_features = bool(getattr(args, "use_features", True))
	include_gt_trans = not getattr(args, "skip_gt_trans", True)

	DatasetCls = FullAlignmentDataset if getattr(args, "full_data", False) else AlignmentDataset
	ds_kwargs = dict(
		files=val_files,
		num_corr=getattr(args, "num_node", 512),
		use_features=use_features,
		feature_dim=feature_dim,
		include_gt_trans=include_gt_trans,
		seed=getattr(args, "split_seed", 0) + 1,
	)
	if DatasetCls is AlignmentDataset:
		ds_kwargs["neg_ratio"] = getattr(args, "neg_ratio", 0.0)
	else:
		ds_kwargs["force_add_true_pairs"] = True

	val_set = DatasetCls(**ds_kwargs)
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
							num_workers=getattr(args, "num_workers", 4))

	root_stem = root.name
	gen_tag = args.generation_method.replace("-", "")
	out_dir = getattr(args, "out", None) or f"results/icp_{args.dataset}_{root_stem}_{gen_tag}"
	matching_dir = os.path.join(out_dir, "matching")
	os.makedirs(out_dir, exist_ok=True)

	sigma = float(getattr(args, "sigma", 0.5))
	sigma_d = float(getattr(args, "inlier_threshold", 0.1))
	seed_ratio = float(getattr(args, "seed_ratio", 0.2))
	icp_threshold = float(getattr(args, "icp_threshold", float("inf")))
	generation_min_score = getattr(args, "generation_min_score", None)

	results = []
	for i, batch in enumerate(val_loader):
		fn = batch[8] if len(batch) == 9 else f"pair_{i:04d}"
		stem = os.path.splitext(os.path.basename(
			fn[0] if isinstance(fn, (list, tuple)) else str(fn)
		))[0]
		print(f"\n[{i+1}/{len(val_files)}] {stem}")
		try:
			r = _run_spectral_icp_on_batch(
				batch=batch,
				sigma=sigma, sigma_d=sigma_d, seed_ratio=seed_ratio,
				generation_method=args.generation_method,
				generation_min_score=generation_min_score,
				icp_threshold=icp_threshold,
				matching_dir=matching_dir,
				feature_dim=feature_dim,
			)
		except Exception as exc:
			print(f"  ERROR: {exc}")
			r = {"pair_name": stem, "error": str(exc)}
		results.append(r)

	# Summary CSV
	csv_fields = [
		"pair_name", "selected_total", "selected_true", "precision",
		"initial_abs_err", "initial_l2_err", "selected_abs_err", "selected_l2_err",
		"fitness", "inlier_rmse", "l1_before", "l1_after", "error",
	]
	csv_path = os.path.join(out_dir, "summary.csv")
	with open(csv_path, "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
		writer.writeheader()
		for r in sorted(results, key=lambda x: x.get("pair_name", "")):
			writer.writerow({k: r.get(k, "") for k in csv_fields})

	ok = [r for r in results if isinstance(r.get("fitness"), float)]
	if ok:
		def _mean(key):
			vals = [r[key] for r in ok if isinstance(r.get(key), float) and not np.isnan(r[key])]
			return float(np.mean(vals)) if vals else float("nan")
		print(f"\n{'='*60}")
		print(f"Done. {len(ok)}/{len(results)} succeeded  |  dataset={args.dataset}  method={args.generation_method}")
		print()
		# Labels match HyperGNN trainer output exactly for direct comparison
		print(f"selected target distance error (mean over {len(ok)} files): "
		      f"mean|y_pred-y_gt|={_mean('selected_abs_err'):.6f}, "
		      f"mean_l2={_mean('selected_l2_err'):.6f}")
		print(f"initial dataset target distance error (mean over {len(ok)} files): "
		      f"mean|y_pred-y_gt|={_mean('initial_abs_err'):.6f}, "
		      f"mean_l2={_mean('initial_l2_err'):.6f}")
		print()
		# ICP-specific metrics (no HyperGNN equivalent)
		print(f"  mean precision:         {_mean('precision'):.4f}")
		print(f"  mean ICP fitness:       {_mean('fitness'):.4f}")
		print(f"  mean inlier RMSE:       {_mean('inlier_rmse'):.6f}")
		print(f"  mean L1 before (init):  {_mean('l1_before'):.6f}  [nearest-nbr dist after Kabsch init]")
		print(f"  mean L1 after  (ICP):   {_mean('l1_after'):.6f}  [nearest-nbr dist after ICP]")
		print(f"  summary CSV:      {csv_path}")
		print(f"  matching CSVs:    {matching_dir}/")
	else:
		print("All pairs failed — check errors in summary CSV.")

	return csv_path


def _spectral_icp_main():
	"""Entry point for the HyperGNN-style spectral-ICP command (--root / --generation_method)."""
	import sys
	from pathlib import Path

	_src_root = str(Path(__file__).resolve().parents[3])
	if _src_root not in sys.path:
		sys.path.insert(0, _src_root)

	try:
		from project.config.hypergnn_config import get_config
	except ImportError:
		from src.project.config.hypergnn_config import get_config

	args = get_config()

	# ICP-specific extras not present in hypergnn_config
	import argparse as _ap
	_extra = _ap.ArgumentParser(add_help=False)
	_extra.add_argument("--icp_threshold", type=float, default=float("inf"))
	_extra.add_argument("--sigma", type=float, default=0.5)
	_extra.add_argument("--out", type=str, default=None)
	_known, _ = _extra.parse_known_args()
	args.icp_threshold = _known.icp_threshold
	args.sigma = _known.sigma
	args.out = _known.out

	batch_spectral_icp(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
	import sys

	# ── Dispatch to spectral-ICP pipeline when --root or --generation_method present ──
	if '--root' in sys.argv or '--generation_method' in sys.argv:
		_spectral_icp_main()
		sys.exit(0)

	# ── Legacy CLI ────────────────────────────────────────────────────────────
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
