import torch
from project.utils.SE3 import transform, integrate_trans

def kabsch(A, B, weights=None, weight_threshold=0):
    """ 
    Input:
        - A:       [bs, num_corr, 3], source point cloud
        - B:       [bs, num_corr, 3], target point cloud
        - weights: [bs, num_corr]     weight for each correspondence 
        - weight_threshold: float,    clips points with weight below threshold
    Output:
        - R, t 
    """
    bs = A.shape[0]
    if weights is None:
        weights = torch.ones_like(A[:, :, 0])
    weights[weights < weight_threshold] = 0
    # weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)

    # find mean of point cloud
    centroid_A = torch.sum(A * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)
    centroid_B = torch.sum(B * weights[:, :, None], dim=1, keepdim=True) / (torch.sum(weights, dim=1, keepdim=True)[:, :, None] + 1e-6)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    # construct weight covariance matrix
    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm

    # find rotation
    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.linalg.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(A.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    t = centroid_B.permute(0,2,1) - R @ centroid_A.permute(0,2,1)
    # warp_A = transform(A, integrate_trans(R,t))
    # RMSE = torch.sum( (warp_A - B) ** 2, dim=-1).mean()
    return integrate_trans(R, t)

def cal_leading_eigenvector(M, num_iterations, method='power'):
    """
    Calculate the leading eigenvector using power iteration algorithm or torch.symeig
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
        - method: select different method for calculating the learding eigenvector.
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    if method == 'power':
        # power iteration algorithm
        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for _ in range(num_iterations):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    if method == 'eig':  # cause NaN during back-prop
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    exit(-1)


def spectral_matching_greedy(M, src_idx=None, tgt_idx=None, max_ratio=None, num_iterations=10, method='power'):
    """
    Spectral matching (leading eigenvector) + greedy one-to-one selection.
    Args:
        M:         [num_corr, num_corr] or [bs, num_corr, num_corr] compatibility matrix
        src_idx:   [num_corr] source indices for each correspondence
        tgt_idx:   [num_corr] target indices for each correspondence
        max_ratio: optional max ratio of matches to return (0..1)
        num_iterations: power-iteration steps for leading eigenvector
        method:    'power' or 'eig'
    Returns:
        selected_idx: LongTensor [bs, max_len] (padded with -1)
        selected_mask: BoolTensor [bs, max_len] indicating valid entries
    """
    if M.dim() == 3:
        if M.shape[0] != 1:
            raise ValueError("Only bs==1 is supported")
        M = M.squeeze(0)
    if M.dim() != 2:
        raise ValueError(f"M must have shape (num_corr, num_corr) (bs==1), got {M.shape}")
    num_corr = M.shape[0]
    if src_idx is None:
        src_idx = torch.arange(num_corr, device=M.device, dtype=torch.long)
    if tgt_idx is None:
        tgt_idx = torch.arange(num_corr, device=M.device, dtype=torch.long)
    if src_idx.numel() != num_corr or tgt_idx.numel() != num_corr:
        raise ValueError("src_idx/tgt_idx must have length num_corr")

    scores = cal_leading_eigenvector(M.unsqueeze(0), num_iterations, method=method).squeeze(0)  # [num_corr]
    max_possible = min(int(torch.unique(src_idx).numel()), int(torch.unique(tgt_idx).numel()))
    if max_ratio is None:
        max_num = max_possible
    else:
        max_num = int(max_possible * max_ratio)
        if max_ratio > 0 and max_num == 0:
            max_num = 1
        max_num = min(max_num, max_possible)

    order = torch.argsort(scores, descending=True)
    used_src = set()
    used_tgt = set()
    picks = []
    for idx in order.tolist():
        s = int(src_idx[idx].item())
        t = int(tgt_idx[idx].item())
        if s in used_src or t in used_tgt:
            continue
        used_src.add(s)
        used_tgt.add(t)
        picks.append(idx)
        if len(picks) >= max_num:
            break

    if len(picks) == 0:
        selected_idx = torch.full((0,), -1, device=M.device, dtype=torch.long)
        selected_mask = torch.zeros((0,), device=M.device, dtype=torch.bool)
        return selected_idx, selected_mask

    selected_idx = torch.tensor(picks, device=M.device, dtype=torch.long)
    selected_mask = torch.ones_like(selected_idx, dtype=torch.bool)
    return selected_idx, selected_mask


def two_stage_spectral_matching_greedy(
    M,
    seeds,
    knn_idx,
    src_idx=None,
    tgt_idx=None,
    max_ratio_seeds=None,
    max_ratio=None,
    num_iterations=10,
    method='power',
):
    """
    Two-stage spectral matching with subgraph restriction:
      1) run spectral matching on seeds only
      2) expand to all kNN of selected seeds, then run spectral matching again
    Args:
        M:         [num_corr, num_corr] or [bs, num_corr, num_corr] compatibility matrix
        seeds:     [num_seeds] or [bs, num_seeds] seed correspondence indices
        knn_idx:   [num_corr, k] or [bs, num_corr, k] kNN indices per correspondence
        src_idx:   [num_corr] optional source indices per correspondence
        tgt_idx:   [num_corr] optional target indices per correspondence
        max_ratio_seeds: optional max ratio in stage 1
        max_ratio: optional max ratio in stage 2
    Returns:
        selected_idx: LongTensor [bs, max_len] (padded with -1)
        selected_mask: BoolTensor [bs, max_len] indicating valid entries
    """
    if M.dim() == 3:
        if M.shape[0] != 1:
            raise ValueError("Only bs==1 is supported")
        M = M.squeeze(0)
    if M.dim() != 2:
        raise ValueError(f"M must have shape (num_corr, num_corr) (bs==1), got {M.shape}")
    num_corr = M.shape[0]

    if seeds.dim() != 1 or knn_idx.dim() != 2:
        raise ValueError("seeds must be [num_seeds] and knn_idx must be [num_corr, k]")

    if src_idx is not None and src_idx.numel() != num_corr:
        raise ValueError("src_idx must have length num_corr")
    if tgt_idx is not None and tgt_idx.numel() != num_corr:
        raise ValueError("tgt_idx must have length num_corr")

    M_seed = M[seeds][:, seeds]
    src_seed = src_idx[seeds] if src_idx is not None else None
    tgt_seed = tgt_idx[seeds] if tgt_idx is not None else None

    sel_seed_idx, sel_seed_mask = spectral_matching_greedy(
        M_seed, src_seed, tgt_seed, max_ratio=max_ratio_seeds,
        num_iterations=num_iterations, method=method
    )
    sel_seed_idx = sel_seed_idx[sel_seed_mask]
    if sel_seed_idx.numel() == 0:
        selected_idx = torch.full((0,), -1, device=M.device, dtype=torch.long)
        selected_mask = torch.zeros((0,), device=M.device, dtype=torch.bool)
        return selected_idx, selected_mask

    selected_seeds = seeds[sel_seed_idx]
    neighbors = knn_idx[selected_seeds].reshape(-1)
    candidates = torch.unique(torch.cat([selected_seeds, neighbors], dim=0))

    M_cand = M[candidates][:, candidates]
    src_cand = src_idx[candidates] if src_idx is not None else None
    tgt_cand = tgt_idx[candidates] if tgt_idx is not None else None

    sel_idx, sel_mask = spectral_matching_greedy(
        M_cand, src_cand, tgt_cand, max_ratio=max_ratio,
        num_iterations=num_iterations, method=method
    )
    sel_idx = sel_idx[sel_mask]
    selected = candidates[sel_idx]
    selected_mask = torch.ones_like(selected, dtype=torch.bool)
    return selected, selected_mask


def build_seed_knn_weights(seeds, corr_features, H, src_keypts, tgt_keypts, sigma, sigma_d, num_iterations):
    # 1、Generate initial hypotheses from each seed
    bs, num_corr, num_channels = corr_features.size()
    _, num_seeds = seeds.size()
    assert num_seeds > 0
    k = min(10, num_corr - 1)
    mask = H  # graph_matrix_reconstruct(H, 0, None)
    feature_distance = 1 - torch.matmul(corr_features, corr_features.transpose(2, 1))  # normalized
    masked_feature_distance = feature_distance * mask.float() + (1 - mask.float()) * float(1e9)
    knn_idx = torch.topk(masked_feature_distance, k, largest=False)[1]
    knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k))  # [bs, num_seeds, k]
    #################################
    # construct the feature consistency matrix of each correspondence subset.
    #################################
    knn_features = corr_features.gather(dim=1,
                                        index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels))\
                                        .view([bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
    knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
    knn_M = torch.clamp(1 - (1 - knn_M) / sigma ** 2, min=0)
    knn_M = knn_M.view([-1, k, k])
    feature_knn_M = knn_M

    #################################
    # construct the spatial consistency matrix of each correspondence subset.
    #################################
    src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
        [bs, -1, k, 3])  # [bs, num_seeds, k, 3]
    tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
        [bs, -1, k, 3])
    knn_M = ((src_knn[:, :, :, None, :] - src_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5 - (
            (tgt_knn[:, :, :, None, :] - tgt_knn[:, :, None, :, :]) ** 2).sum(-1) ** 0.5
    knn_M = torch.clamp(1 - knn_M ** 2 / sigma_d ** 2, min=0)
    knn_M = knn_M.view([-1, k, k])
    spatial_knn_M = knn_M

    #################################
    # Power iteratation to get the inlier probability
    #################################
    total_knn_M = feature_knn_M * spatial_knn_M
    total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
    total_weight = cal_leading_eigenvector(total_knn_M, num_iterations, method='power')
    total_weight = total_weight.view([bs, -1, k])
    total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
    total_weight = total_weight.view([-1, k])

    return src_knn, tgt_knn, total_weight


def evaluate_hypotheses(seeds, H, src_knn, tgt_knn, total_weight, src_keypts, tgt_keypts,
                        src_normal, tgt_normal, inlier_threshold, cal_inliers_normal):
    #################################
    # calculate the transformation by weighted least-squares for each subsets in parallel
    #################################
    bs, num_corr, _ = src_keypts.size()
    _, num_seeds = seeds.size()
    k = src_knn.size(2)
    mask = H
    corr_mask = (mask.sum(dim=1) > 0).float()[:, None, :]
    seed_mask = mask.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, num_corr))  # [bs, num_seeds, num_corr]
    # 2、Inlier estimation for each hypothesis
    src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
    seedwise_trans = kabsch(src_knn, tgt_knn, total_weight)  # weight 有用
    seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
    pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                 src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3, 3:4]  # [bs, num_seeds, num_corr, 3]
    pred_position = pred_position.permute(0, 1, 3, 2)
    L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
    seed_L2_dis = L2_dis * corr_mask + (1 - corr_mask) * float(1e9)
    fitness = cal_inliers_normal(seed_L2_dis < inlier_threshold, seedwise_trans, src_normal, tgt_normal)

    h = int(num_seeds * 0.1)
    hypo_inliers_idx = torch.topk(fitness, h, -1, largest=True)[1]  # [bs, h]
    #seeds = seeds.gather(1, hypo_inliers_idx)  # [bs, h]
    seed_mask = seed_mask.gather(dim=1,
                                 index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]
    L2_dis = L2_dis.gather(dim=1, index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]

    # Get the maximum sliding pos
    max_length = seed_mask.sum(dim=2).min()

    # prepare for sampling
    best_score, best_trans, best_labels = None, None, None
    L2_dis = L2_dis * seed_mask + (1 - seed_mask) * float(1e9)  # [bs, num_seeds, num_corr]
    # 3、Sliding window sampling
    s = 6  # window size
    m = 3  # stride
    iters = 15 # sampling times
    max_iters = int((max_length - s) / m)
    if max_length > s + m:
        iters = min(max_iters, iters)

    dis, idx = torch.topk(L2_dis, s + iters * m, -1, largest=False)
    # corr_mask
    #corr_mask = (seed_mask.sum(dim=1) > 0).float()[:, None, :]
    sampled_list = []
    for i in range(iters + 1):
        knn_idx = idx[:, :, i * m: s + i * m].contiguous()
        src_knn = src_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
            [bs, -1, s, 3])
        tgt_knn = tgt_keypts.gather(dim=1, index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, 3)).view(
            [bs, -1, s, 3])
        src_knn, tgt_knn = src_knn.view([-1, s, 3]), tgt_knn.view([-1, s, 3])
        sampled_trans = kabsch(src_knn, tgt_knn)
        sampled_trans = sampled_trans.view([bs, -1, 4, 4])

        sampled_list.append(sampled_trans[0])

        pred_position = torch.einsum('bsnm,bmk->bsnk', sampled_trans[:, :, :3, :3],
                                     src_keypts.permute(0, 2, 1)) + sampled_trans[:, :, :3,
                                                                    3:4]  # [bs, num_seeds, num_corr, 3]
        pred_position = pred_position.permute(0, 1, 3, 2)
        sampled_L2_dis = torch.norm(pred_position - tgt_keypts[:, None, :, :], dim=-1)  # [bs, num_seeds, num_corr]
        sampled_L2_dis = sampled_L2_dis * corr_mask + (1 - corr_mask) * float(1e9)
        MAE_score = (inlier_threshold - sampled_L2_dis) / inlier_threshold
        fitness = torch.sum(MAE_score * (sampled_L2_dis < inlier_threshold), dim=-1)
        sampled_best_guess = fitness.argmax(dim=1)  # [bs, 1]
        sampled_best_score = fitness.gather(dim=1, index=sampled_best_guess[:, None]).squeeze(1)  # [bs, 1]

        sampled_best_trans = sampled_trans.gather(dim=1,
                                                  index=sampled_best_guess[:, None, None, None].expand(-1, -1, 4,
                                                                                                       4)).squeeze(
            1)  # [bs, 4, 4]

        if i == 0:
            best_score = sampled_best_score
            best_trans = sampled_best_trans
            #best_labels = sampled_best_labels
        else:
            update_mask = sampled_best_score > best_score
            best_score = torch.where(update_mask, sampled_best_score, best_score)
            best_trans = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), sampled_best_trans, best_trans)
            #best_labels = torch.where(update_mask.unsqueeze(-1), sampled_best_labels, best_labels)

    final_trans = best_trans
    #final_labels = (best_labels < inlier_threshold).float()
    return torch.stack(sampled_list), final_trans
