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
                                        index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view(
        [bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
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
    total_knn_M = feature_knn_M * spatial_knn_M  # 有用
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
    # 2、每个假设得到pred inliers
    src_knn, tgt_knn = src_knn.view([-1, k, 3]), tgt_knn.view([-1, k, 3])
    seedwise_trans = kabsch(src_knn, tgt_knn, total_weight)  # weight 有用
    seedwise_trans = seedwise_trans.view([bs, -1, 4, 4])
    pred_position = torch.einsum('bsnm,bmk->bsnk', seedwise_trans[:, :, :3, :3],
                                 src_keypts.permute(0, 2, 1)) + seedwise_trans[:, :, :3,
                                                                3:4]  # [bs, num_seeds, num_corr, 3]
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
