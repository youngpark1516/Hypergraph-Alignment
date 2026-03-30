import csv
import os

import numpy as np
import torch

from project.models.hypothesis_generation import (
    greedy_compatibility_expansion,
    spectral_matching_greedy,
    two_stage_spectral_matching_greedy,
)


def run_generation_step(
    *,
    res,
    gt_labels,
    src_indices,
    tgt_indices,
    file_name,
    epoch,
    iter_idx,
    model,
    generation_method,
    generation_min_score,
    generation_min_confidence,
    matching_dir,
    selected_abs_err_meter,
    selected_l2_err_meter,
    initial_abs_err_meter,
    initial_l2_err_meter,
):
    corr_features = res["corr_feats"].permute(0, 2, 1)
    M = res.get("M")
    if M is None:
        M = torch.matmul(corr_features.permute(0, 2, 1), corr_features)
        M = torch.clamp(1 - (1 - M) / model.sigma ** 2, min=0, max=1)
        M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
    _, num_corr, _ = corr_features.shape

    if generation_method == "spectral-2":
        k = min(10, num_corr - 1)
        H = res["hypergraph"]
        feature_distance = 1 - torch.matmul(corr_features, corr_features.transpose(2, 1))
        masked_feature_distance = feature_distance * H.float() + (1 - H.float()) * float(1e9)
        knn_idx = torch.topk(masked_feature_distance, k, largest=False)[1]
        selected_idx, selected_mask = two_stage_spectral_matching_greedy(
            M.squeeze(0),
            res["seeds"].squeeze(0),
            knn_idx.squeeze(0),
            num_iterations=model.num_iterations,
            max_ratio=0.2,
            max_ratio_seeds=0.8,
            min_score=generation_min_score,
        )
    elif generation_method == "spectral":
        selected_idx, selected_mask = spectral_matching_greedy(
            M.squeeze(0),
            max_ratio=0.4,
            num_iterations=model.num_iterations,
            min_score=generation_min_score,
        )
    elif generation_method == "greedy":
        selected_idx, selected_mask = greedy_compatibility_expansion(
            M.squeeze(0),
            res["confidence"].squeeze(0),
            res["seeds"].squeeze(0),
            max_ratio=0.4,
            src_idx=src_indices.squeeze(0) if src_indices is not None else None,
            tgt_idx=tgt_indices.squeeze(0) if tgt_indices is not None else None,
            topk_each_iter=4,
            min_confidence=generation_min_confidence,
        )
    else:
        raise ValueError(f"Unknown generation_method: {generation_method}")

    selected_idx = selected_idx[selected_mask]
    selected_total = int(selected_idx.numel())
    if selected_total > 0:
        selected_true = int((gt_labels.squeeze(0)[selected_idx] > 0.5).sum().item())
    else:
        selected_true = 0
    true_ratio = (selected_true / selected_total) if selected_total > 0 else 0.0
    print(
        f"selected true correspondences: {selected_true}/{selected_total} "
        f"(precision = {100.0 * true_ratio:.2f}%)"
    )

    if src_indices is None or tgt_indices is None:
        return

    selected_idx_cpu = selected_idx.detach().cpu()
    src_sel = src_indices.squeeze(0).detach().cpu()[selected_idx_cpu]
    tgt_sel = tgt_indices.squeeze(0).detach().cpu()[selected_idx_cpu]
    pairs = list(zip(src_sel.tolist(), tgt_sel.tolist()))
    file_name_single = None
    stem = None
    if file_name is not None:
        if isinstance(file_name, (list, tuple)):
            file_name_single = file_name[0]
        else:
            file_name_single = file_name
        base = os.path.basename(str(file_name_single))
        stem = os.path.splitext(base)[0]

    if file_name_single is not None:
        try:
            eval_data = np.load(str(file_name_single))
            xyz1_np = eval_data["xyz1"].astype(np.float32)
            corres_np = eval_data["corres"].astype(np.int64)

            src_all_np = src_indices.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
            tgt_all_np = tgt_indices.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)
            gt_all_idx = corres_np[src_all_np]
            pred_all_xyz = xyz1_np[tgt_all_np]
            gt_all_xyz = xyz1_np[gt_all_idx]
            diff_all = pred_all_xyz - gt_all_xyz
            mean_abs_err_all = float(np.abs(diff_all).mean())
            mean_l2_err_all = float(np.linalg.norm(diff_all, axis=1).mean())
            initial_abs_err_meter.update(mean_abs_err_all)
            initial_l2_err_meter.update(mean_l2_err_all)

            if selected_total > 0:
                src_sel_np = src_sel.numpy().astype(np.int64, copy=False)
                tgt_sel_np = tgt_sel.numpy().astype(np.int64, copy=False)
                gt_tgt_idx = corres_np[src_sel_np]
                pred_tgt_xyz = xyz1_np[tgt_sel_np]
                gt_tgt_xyz = xyz1_np[gt_tgt_idx]
                diff = pred_tgt_xyz - gt_tgt_xyz
                mean_abs_err = float(np.abs(diff).mean())
                mean_l2_err = float(np.linalg.norm(diff, axis=1).mean())
                selected_abs_err_meter.update(mean_abs_err)
                selected_l2_err_meter.update(mean_l2_err)
            print(
                f"initial dataset target distance error: "
                f"mean|y_pred-y_gt|={mean_abs_err_all:.6f}, "
                f"mean_l2={mean_l2_err_all:.6f}"
            )
            if selected_total > 0:
                print(
                    f"selected target distance error: "
                    f"mean|y_pred-y_gt|={mean_abs_err:.6f}, "
                    f"mean_l2={mean_l2_err:.6f}"
                )
        except Exception as exc:
            print(f"selected target distance error unavailable: {exc}")

    prefix = stem or f"matching_plan_epoch{epoch}_iter{iter_idx}"
    os.makedirs(matching_dir, exist_ok=True)
    csv_path = os.path.join(matching_dir, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_idx", "tgt_idx"])
        for s, t in pairs:
            writer.writerow([s, t])
