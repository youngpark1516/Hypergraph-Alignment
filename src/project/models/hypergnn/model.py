import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from project.utils.SE3 import transform, integrate_trans
from project.utils.timer import Timer
from project.models.hypergnn.pooling import FPSWE_pool, interp1d, sparse_sort
from project.models.hypothesis_generation import build_seed_knn_weights, evaluate_hypotheses, kabsch
from torch_scatter import scatter_add
import math
import time

def distance(x):  # bs, channel, num_points
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = xx + inner + xx.transpose(2, 1).contiguous()  # bs, num_points, num_points
    return distance


def feature_knn(x, k):
    dis = -distance(x)
    idx = dis.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def mask_score(score, H0, iter, num_layer, choice='topk'):
    bs, num, _ = H0.size()
    # score: row->V, col->E
    if choice == 'auto':
        W = score * H0
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    elif choice == 'topk':
        k = max(1, round(num * 0.1 * (num_layer - 1 - iter)))  # Ensure k >= 1
        topk, _ = torch.topk(score, k=k, dim=-1)  # 每个节点选出topk概率的超边，该超边包含该节点
        a_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, 1, num)
        W = torch.where(torch.greater_equal(score, a_min), score, torch.zeros_like(score))
        H = torch.where(torch.greater(W, 0), H0, torch.zeros_like(H0))

    else:
        raise NotImplementedError

    return H, W

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


def knn(x, k, ignore_self=False, normalized=True):
    """ find feature space knn neighbor of x 
    Input:
        - x:       [bs, num_corr, num_channels],  input features
        - k:       
        - ignore_self:  True/False, return knn include self or not.
        - normalized:   True/False, if the feature x normalized.
    Output:
        - idx:     [bs, num_corr, k], the indices of knn neighbors
    """
    inner = 2 * torch.matmul(x, x.transpose(2, 1))
    if normalized:
        pairwise_distance = 2 - inner
    else:
        xx = torch.sum(x ** 2, dim=-1, keepdim=True)
        pairwise_distance = xx - inner + xx.transpose(2, 1)

    if ignore_self is False:
        idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]  # (batch_size, num_points, k)
    else:
        idx = pairwise_distance.topk(k=k + 1, dim=-1, largest=False)[1][:, :, 1:]
    return idx


class GraphUpdate(nn.Module):
    def __init__(self, num_channels, num_heads, num_layer):
        super(GraphUpdate, self).__init__()
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads
        self.num_layer = num_layer
        self.make_score_choice = 'topk'

    def forward(self, H0, vertex_feat, edge_feat, iter):
        # Update H according to vertex_feat and edge_feat (attention)
        bs, num_vertices, _ = H0.size()
        Q = self.projection_q(vertex_feat).view([bs, self.head, self.num_channels // self.head, num_vertices])  # row
        K = self.projection_k(edge_feat).view([bs, self.head, self.num_channels // self.head, num_vertices])  # col

        attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5
        
        # Free Q, K immediately after use
        del Q, K

        # mask attention using SC2 prior
        attention_mask = 1 - H0  # Fix the hypergraph
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)
        score = torch.sigmoid(attention + attention_mask[:, None, :, :])
        
        # Free attention and mask after use
        del attention, attention_mask

        score = (torch.sum(score, dim=1) / self.head).view(bs, num_vertices, num_vertices)  # Mean over heads
        H, W = mask_score(score, H0, iter, self.num_layer, choice=self.make_score_choice)
        
        # Free score after mask_score
        del score

        # update D_n_1, W_edge according to new H
        degree_E = H.sum(dim=1)  # [bs, num_vertices]
        # Compute inverse degrees as vectors
        inv_deg_E = 1.0 / (degree_E + 1e-10)
        inv_deg_E[torch.isinf(inv_deg_E)] = 0
        De_n_1 = inv_deg_E.unsqueeze(-1)  # [bs, num_vertices, 1]

        degree_V = H.sum(dim=2)  # [bs, num_vertices]
        inv_deg_V = 1.0 / (degree_V + 1e-10)
        inv_deg_V[torch.isinf(inv_deg_V)] = 0
        Dv_n_1 = inv_deg_V.unsqueeze(-1)  # [bs, num_vertices, 1]
        
        del degree_E, degree_V, inv_deg_E, inv_deg_V

        W_edge = W.sum(dim=1)  # [bs, num]
        W_edge = F.normalize(W_edge, dim=1)
        W_edge = W_edge.view([bs, num_vertices, 1])
        return H, W, De_n_1, Dv_n_1, W_edge


class NonLocalBlock(nn.Module):
    def __init__(self, num_channels=128, num_heads=1):
        super(NonLocalBlock, self).__init__()
        self.fc_message = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels // 2, kernel_size=1),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_channels // 2, num_channels, kernel_size=1),
        )
        self.projection_q = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_k = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.projection_v = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.num_channels = num_channels
        self.head = num_heads

    def forward(self, feat, attention, H):
        """
        Input:
            - feat:     [bs, num_channels, num_corr]  input feature
            - attention [bs, num_corr, num_corr]      spatial consistency matrix
        Output:
            - res:      [bs, num_channels, num_corr]  updated feature
        """
        bs, num_corr = feat.shape[0], feat.shape[-1]
        Q = self.projection_q(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        K = self.projection_k(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        V = self.projection_v(feat).view([bs, self.head, self.num_channels // self.head, num_corr])
        feat_attention = torch.einsum('bhco, bhci->bhoi', Q, K) / (self.num_channels // self.head) ** 0.5  
        # [bs, head, num_corr, num_corr]
        attention_mask = 1 - H
        attention_mask = attention_mask.masked_fill(attention_mask.bool(), -1e9)

        # combine the feature similarity with spatial consistency
        weight = torch.softmax(attention[:, None, :, :] * feat_attention + attention_mask[:, None, :, :],
                               dim=-1)  #[bs, head, num_corr, num_corr]
        message = torch.einsum('bhoi, bhci-> bhco', weight, V).reshape([bs, -1, num_corr])  # [bs, dim, num_corr]
        message = self.fc_message(message)
        res = feat + message
        return res


class FPSWE_pool(nn.Module):
    def __init__(self, d_in, num_anchors=1024, num_projections=1024, anch_freeze=True, out_type='linear'):
        '''
        The PSWE and LPSWE module that produces 
        fixed-dimensional permutation-invariant embeddings 
        for input sets of arbitrary size.
        '''

        super(FPSWE_pool, self).__init__()
        self.d_in = d_in # the dimensionality of the space that each set sample belongs to
        self.num_ref_points = num_anchors # number of points in the reference set
        self.num_projections = num_projections # number of slices
        self.anch_freeze = anch_freeze # if True the reference set and the theta are not learnable

        uniform_ref = torch.linspace(-1, 1, num_anchors).unsqueeze(1).repeat(1, num_projections) #num_anchors x num_preojections
        self.reference = nn.Parameter(uniform_ref, requires_grad=not self.anch_freeze)

        # slicer
        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_projections, bias=False), dim=0)
        if num_projections <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        self.theta.weight_v.requires_grad = not self.anch_freeze

        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Parameter(torch.zeros(num_projections, num_anchors))
        nn.init.xavier_uniform_(self.weight)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        device = self.weight.device

        if self.anch_freeze == False:
            uniform_ref = torch.linspace(-1, 1, self.num_ref_points).unsqueeze(1).repeat(1, self.num_projections).to(device) #num_anchors x num_preojections
            self.reference.data = uniform_ref

        if self.num_projections <= self.d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
        

    def double_self_loops(self, features, index):
        '''
        for the isolated nodes double them because pooling only works
            for minimum 2 elements in a set
        '''
        # Find indices where the group appears only once
        counts = torch.bincount(index)
        unique_mask = counts[index] == 1

        # Get elements to be duplicated
        unique_features = features[unique_mask]
        unique_groups = index[unique_mask]
        
        # Stack original tensor with duplicated unique elements
        new_features = torch.cat((features, unique_features), dim=0)
        new_index = torch.cat((index, unique_groups), dim=0)
        
        # Cleanup intermediate tensors
        del counts, unique_mask, unique_features, unique_groups
        
        return new_features, new_index

    def forward(self, X, hyperedge_index, data=None, name=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  N x dn tensor containing N samples in a dn-dimensional space
        Output:
            weighted_embeddings: E x num_projections tensor, containing an embedding of dimension "num_projections" (i.e., number of slices)
        '''
   
        # Step 1: project samples into the 1D slices
        N, dn = X.shape
        Xslices = self.get_slice(X) # N x num_projections

        # for the self-loops double the node to be able to apply the pooling 
        Xslices, hyperedge_index_new = self.double_self_loops(Xslices, hyperedge_index)
        Xslices_sorted, Xind = sparse_sort(Xslices, hyperedge_index_new)

        # regardless of the column sorting, all of them should have the same resorted index
        hyperedge_index_1_sorted = hyperedge_index_new[Xind[:,0]]
        M, dm = self.reference.shape

        eps = 0.00001
        #this should allow a correct interpolation when M>N
        margin_up = 0.9999
        assert (margin_up+eps < 1)

        # Compute constants for this forward pass (no caching to support varying batch structures)
        deg_helper = torch.ones_like(hyperedge_index_1_sorted)
        R = torch.arange(hyperedge_index_1_sorted.shape[0]).to(X.device).to(X.dtype)+1
        pad = torch.tensor([0.0]).to(X.device)
        edges = torch.sort(torch.unique(hyperedge_index_1_sorted))[0]
        hyperedge_index_anchors_1 = edges.repeat_interleave(M)
        num_edges = edges.shape[0]
        xnew = torch.linspace(0, 1, M).repeat(num_edges).to(X.device).to(X.dtype)
        xnew = xnew * 0.99998+eps

        ynew = torch.zeros((self.num_projections, M*num_edges)).to(X.device)

        # compute the degree
        D1 = scatter_add(deg_helper, hyperedge_index_1_sorted) #E
        D = torch.index_select(D1, 0,  hyperedge_index_1_sorted)

        # Step 2: interpolate 

        # compute the x indices to be used as positions for interpolation
        # they are computer for each hyperedge in parallel and are uniformly arranged
        ptr = torch.cat((pad,torch.cumsum(D1, dim=0)))
        P = torch.index_select(ptr, 0,  hyperedge_index_1_sorted)
        assert (D.min() >= 2)
        x = (R-P-1)/(D-1)*0.99999+eps +hyperedge_index_1_sorted
        x = x.unsqueeze(0).repeat(self.num_projections, 1)

        xnew = xnew + hyperedge_index_anchors_1
        xnew = xnew.unsqueeze(0).repeat(self.num_projections, 1)

        #this still correspond to hyperedge_index_1_sorted
        y = torch.transpose(Xslices_sorted, 0, 1).reshape(self.num_projections, -1)
        
        # interpolate y based on the x values
        Xslices_sorted_interpolated = interp1d(x, y, xnew, hyperedge_index_1_sorted).view(self.num_projections, -1)
        Xslices_sorted_interpolated = torch.transpose(Xslices_sorted_interpolated, 0, 1)

        # reshape the (projected) references. no need for projection since we sample them already projected
        Rslices = self.reference.unsqueeze(0).repeat(num_edges,1,1)#.to(X.device) # num_edges x M x num_projections
        Rslices = Rslices.reshape(num_edges*M,-1) # num_edges x  x num_projections

        # Step 3: sort references and compute proper Wasserstein distance
        _, Rind = sparse_sort(Rslices, hyperedge_index_anchors_1)
        
        # Compute distance between sorted samples (this IS the 1D Wasserstein distance)
        embeddings = Rslices - torch.gather(Xslices_sorted_interpolated, dim=0, index=Rind)
        del Rind, Rslices, Xslices_sorted_interpolated  # Free immediately
        
        embeddings = embeddings.transpose(0, 1)  # [num_projections, num_edges*M]
        embeddings = embeddings.reshape(self.num_projections, num_edges, M)  # [num_projections, num_edges, M]
        
        # Step 4: weighted sum (learned weights are important for Wasserstein!)
        w = self.weight.unsqueeze(1).repeat(1, num_edges, 1)  # [num_projections, num_edges, M]
        weighted_embeddings = (w * embeddings).mean(-1)  # [num_projections, num_edges]
        del w, embeddings  # Free
        
        out = weighted_embeddings.transpose(0, 1)  # [num_edges, num_projections]
        del weighted_embeddings
        
        # Cleanup remaining tensors
        del Xslices, Xslices_sorted, Xind, hyperedge_index_new, hyperedge_index_1_sorted
        del deg_helper, R, pad, hyperedge_index_anchors_1, xnew, ynew
        del D1, D, ptr, P, x, y
        
        # Return compact output - caller handles index mapping
        return out.contiguous(), edges
        

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)


class feature_aggregation_layer(nn.Module):
    def __init__(self,
                 k=20,
                 num_channels=128,
                 head=1,
                 use_knn=False,
                 aggr='mean'):
        super(feature_aggregation_layer, self).__init__()
        self.k = k
        self.use_knn = use_knn
        self.head = head
        self.num_channels = num_channels
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * num_channels, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )
        if self.aggr == 'self_attention':
            self.fc = nn.Sequential(
                nn.Conv1d(num_channels, num_channels // 4, kernel_size=1),
                nn.BatchNorm1d(num_channels // 4),
                nn.LeakyReLU(inplace=True, negative_slope=0.2),
                nn.Conv1d(num_channels // 4, 1, kernel_size=1),
            )

    def forward(self, vertex_feat, edge_feat, edge_weight, incidence, inv_edge_degree, inv_vertex_degree, 
                edge_scale, knn_k):
        # Renames: x->vertex_feat, y->edge_feat, W->edge_weight, H->incidence,
        # De_n_1->inv_edge_degree, Dv_n_1->inv_vertex_degree, W_edge->edge_scale, k->knn_k
        batch_size, num_channels, num_nodes = vertex_feat.size()
        edge_feat_out = None
        if self.use_knn:
            vertex_pos = vertex_feat  # (bs, dim, num_corr)
            idx = feature_knn(vertex_pos, k=knn_k + 1)  # (bs, num_corr, k)
            idx = idx[:, :, 1:]  # ignore the center point

            # bs, num_points, _ = idx.size()
            idx_base = torch.arange(batch_size).to(vertex_feat.device).view(-1, 1, 1) * num_nodes  #
            idx = idx + idx_base  #

            idx = idx.view(-1)
            # _, num_channels, _ = vertex_pos.size()

            vertex_pos = vertex_pos.transpose(2, 1).contiguous()
            neighbor_pos = vertex_pos.view(batch_size * num_nodes, -1)[idx, :]
            neighbor_pos = neighbor_pos.view(batch_size, num_nodes, knn_k, num_channels)
            vertex_pos = vertex_pos.view(batch_size, num_nodes, 1, num_channels)

            neighbor_pos = (neighbor_pos - vertex_pos).permute(0, 3, 1, 2)

            neighbor_pos = neighbor_pos.max(dim=-1, keepdim=True)[
                0]  # attention here [bs, num_dim, num_point, k] -> [bs, num_dim, num_point, 1]

            neighbor_pos = neighbor_pos.view(batch_size, num_channels, num_nodes)
            vertex_feat_out = vertex_feat + neighbor_pos
            # vertex_feat_out = torch.cat((vertex_feat, neighbor_pos), dim=1)  # bs, dim*2, num_corr, k

        else:
            # feature aggregation using softmax or more complicate ways (attention)
            if self.aggr == 'mean':
                # v->e
                # aggregation message from v to e
                feature = vertex_feat.permute(0, 2, 1)  # [bs, num, dim]
                feature = torch.bmm(incidence.permute(0, 2, 1), feature)
                # Element-wise multiply instead of bmm (inv_edge_degree is [bs, num, 1])
                edge_feat_out = feature * inv_edge_degree

                if edge_feat is not None:
                    edge_feat_out = self.mlp(torch.cat((edge_feat, edge_feat_out.permute(0, 2, 1)), dim=1)).permute(0, 2,
                                                                                                          1)  # [bs, num, dim]

                # update message of e
                feature = edge_scale * edge_feat_out  # [bs, num, 1] * [bs, num, dim]

                # aggregation message from e to v
                feature = torch.bmm(incidence, feature)
                # Element-wise multiply instead of bmm
                feature = feature * inv_vertex_degree

                vertex_feat_out = feature.permute(0, 2, 1)  # [bs, dim, num]
                edge_feat_out = edge_feat_out.permute(0, 2, 1)

            elif self.aggr == 'self_attention':
                # e->v nonlocal net using weight matrix W
                feature = vertex_feat.permute(0, 2, 1)
                feature = torch.bmm(incidence.permute(0, 2, 1), feature)  # H^T=H
                # Element-wise multiply instead of bmm
                edge_feat_out = feature * inv_edge_degree  # [bs, num, dim]

                feat = edge_feat_out.permute(0, 2, 1)  # [bs, dim, num]
                score = torch.softmax(self.fc(feat).permute(0, 2, 1), dim=1)  # [bs, num, 1]
                feature = score * edge_scale * feature

                feature = torch.bmm(incidence, feature)
                # Element-wise multiply instead of bmm
                feature = feature * inv_vertex_degree

                vertex_feat_out = feature.permute(0, 2, 1)
                edge_feat_out = edge_feat_out.permute(0, 2, 1)

            else:
                raise NotImplementedError

        return vertex_feat_out, edge_feat_out

class WHNN_aggregation_layer(nn.Module):
    """
    Feature aggregation using FPSWE pooling (WHNN-style).
    Returns (vertex_feat_out, edge_feat_out) with shapes [bs, C, N].
    """

    def __init__(self, num_channels=64, num_anchors=128, num_projections=32, anch_freeze=True, topk_per_vertex=2):
        super(WHNN_aggregation_layer, self).__init__()
        self.num_channels = num_channels
        self.num_projections = num_projections
        self.topk_per_vertex = topk_per_vertex
        self.pool_v2e = FPSWE_pool(
            d_in=num_channels,
            num_anchors=num_anchors,
            num_projections=num_projections,
            anch_freeze=anch_freeze,
        )
        self.pool_e2v = FPSWE_pool(
            d_in=num_channels,
            num_anchors=num_anchors,
            num_projections=num_projections,
            anch_freeze=anch_freeze,
        )
        if num_projections == num_channels:
            self.edge_proj = nn.Identity()
            self.vertex_proj = nn.Identity()
        else:
            self.edge_proj = nn.Conv1d(num_projections, num_channels, kernel_size=1, bias=False)
            self.vertex_proj = nn.Conv1d(num_projections, num_channels, kernel_size=1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(2 * num_channels, num_channels, kernel_size=1),
            nn.BatchNorm1d(num_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
        )

    def _build_incidence_index(self, incidence, edge_weight):
        # incidence/edge_weight: [N, N] dense; keep top-k per vertex
        scores = edge_weight if edge_weight is not None else incidence
        if incidence is not None:
            scores = scores.masked_fill(incidence <= 0, float("-inf"))
        num_nodes = scores.size(0)
        k = min(self.topk_per_vertex, scores.size(1))
        if k <= 0:
            return None, None
        _, topk_idx = torch.topk(scores, k=k, dim=1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        row_idx = torch.arange(num_nodes, device=scores.device).unsqueeze(1).expand(-1, k)
        mask[row_idx, topk_idx] = True
        # if incidence is not None:
        mask = mask & (incidence > 0)
        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return None, None
        edge_counts = torch.bincount(idx[:, 1], minlength=num_nodes)
        valid_edges = edge_counts > 1
        keep = valid_edges[idx[:, 1]]
        idx = idx[keep]
        if idx.numel() == 0:
            return None, None
        return idx[:, 0], idx[:, 1]

    def _pool_grouped(self, X, group_index, num_groups, pool):
        # X: [M, C], group_index: [M] with values in [0, num_groups)
        if X.numel() == 0:
            return X.new_zeros((num_groups, self.num_projections))
        pooled, groups = pool(X, group_index)
        out = X.new_zeros((num_groups, pooled.size(-1)))
        out[groups] = pooled
        return out

    def forward(
        self,
        vertex_feat,
        edge_feat,
        edge_weight,
        incidence,
        inv_edge_degree,
        inv_vertex_degree,
        edge_scale,
        knn_k,
    ):
        # Keep signature aligned with feature_aggregation_layer; unused args are ignored.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        batch_size, num_channels, num_nodes = vertex_feat.size()
        vertex_out_list = []
        edge_out_list = []

        for b in range(batch_size):
            incidence_b = incidence[b]
            v_idx, e_idx = self._build_incidence_index(incidence_b, edge_weight[b] if edge_weight is not None else None)
            if v_idx is None:
                edge_out_b = vertex_feat.new_zeros((num_nodes, num_channels))
                vertex_out_b = vertex_feat.new_zeros((num_nodes, num_channels))
                edge_out_list.append(edge_out_b)
                vertex_out_list.append(vertex_out_b)
                continue
            edge_sizes = torch.bincount(e_idx, minlength=num_nodes)
            edge_sizes = edge_sizes[edge_sizes > 0]
            # print(
            #     f"[WHNN] batch {b}: hyperedges={edge_sizes.numel()}, "
            #     f"size mean={edge_sizes.float().mean().item():.2f}, size std={edge_sizes.float().std().item():.2f}"
            # )

            # V -> E pooling: group by edge index
            v_feat_b = vertex_feat[b].transpose(0, 1)  # [N, C]
            v_samples = v_feat_b.index_select(0, v_idx)  # [M, C]
            edge_out_b = self._pool_grouped(v_samples, e_idx, num_nodes, self.pool_v2e)
            edge_out_list.append(edge_out_b)

        edge_feat_out = torch.stack(edge_out_list, dim=0).permute(0, 2, 1)  # [bs, P, N]
        edge_feat_out = self.edge_proj(edge_feat_out)

        if edge_feat is not None:
            edge_feat_out = self.mlp(torch.cat((edge_feat, edge_feat_out), dim=1))

        for b in range(batch_size):
            incidence_b = incidence[b]
            v_idx, e_idx = self._build_incidence_index(incidence_b, edge_weight[b] if edge_weight is not None else None)
            if v_idx is None:
                vertex_out_b = vertex_feat.new_zeros((num_nodes, num_channels))
                vertex_out_list.append(vertex_out_b)
                continue

            # E -> V pooling: group by vertex index
            e_feat_b = edge_feat_out[b].transpose(0, 1)  # [N, C]
            e_samples = e_feat_b.index_select(0, e_idx)  # [M, C]

            if edge_scale is not None:
                scale_b = edge_scale[b].index_select(0, e_idx)  # [M, 1]
                e_samples = e_samples * scale_b

            vertex_out_b = self._pool_grouped(e_samples, v_idx, num_nodes, self.pool_e2v)
            vertex_out_list.append(vertex_out_b)

        vertex_feat_out = torch.stack(vertex_out_list, dim=0).permute(0, 2, 1)  # [bs, P, N]
        vertex_feat_out = self.vertex_proj(vertex_feat_out)
        return vertex_feat_out, edge_feat_out


class HGNN_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        residual_connection=False,
        use_edge_feature=True,
        use_whnn=False,
        whnn_num_anchors=128,
        whnn_num_projections=32,
        whnn_anch_freeze=True,
        whnn_topk_per_vertex=2,
    ):
        super(HGNN_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        if use_whnn:
            self.feat_agg = WHNN_aggregation_layer(
                num_channels=in_channels,
                num_anchors=whnn_num_anchors,
                num_projections=whnn_num_projections,
                anch_freeze=whnn_anch_freeze,
                topk_per_vertex=whnn_topk_per_vertex,
            )
        else:
            self.feat_agg = feature_aggregation_layer(k=20, num_channels=in_channels)
        self.residual_connection = residual_connection
        self.use_edge_feature = use_edge_feature

    def forward(self, vertex_feat, edge_feat, edge_weight, incidence, inv_edge_degree, inv_vertex_degree, 
                edge_scale, alpha, theta, vertex_feat0, knn_k):
        bs, num_vertex, num_edge = incidence.size()
        vertex_feat_out, edge_feat_out = self.feat_agg(vertex_feat, edge_feat, edge_weight, incidence, 
                                                 inv_edge_degree, inv_vertex_degree, edge_scale, knn_k)  
            # vertex_feat_out: feature of vertex in n+1 layer; edge_feat_out: feature of edge in n layer
        # update feature of vertex. self.conv is a linear layer over channel dimension
        if self.residual_connection:
            vertex_feat_residual = vertex_feat_out * (1 - alpha) + alpha * vertex_feat0
            vertex_feat = (1 - theta) * vertex_feat_residual + theta * self.bn(self.conv(vertex_feat_residual))
            del vertex_feat_residual
        else:  
            vertex_feat_conv = self.bn(self.conv(vertex_feat_out))
            vertex_feat = vertex_feat_conv + vertex_feat
            del vertex_feat_conv
        
        del vertex_feat_out  # Free after use
        vertex_feat = F.leaky_relu(vertex_feat, negative_slope=0.2)
        
        if not self.use_edge_feature:
            edge_feat_out = None
        return vertex_feat, edge_feat_out


class HGNN(nn.Module):
    def __init__(self, in_channel=6,
                 n_emb_dims=128,
                 k=20,
                 num_layers=6,
                 lamda=0.5,
                 alpha=0.1,
                 pooling_layer_idx=-1,):
        super(HGNN, self).__init__()
        self.k = k
        self.lamda = lamda
        self.alpha = alpha
        self.num_layers = num_layers
        self.change_H0 = False
        dim = [n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims,
               n_emb_dims]
        self.layer0 = nn.Conv1d(in_channel, dim[0], kernel_size=1, bias=True)
        if isinstance(pooling_layer_idx, (list, tuple)):
            pooling_layer_indices = [int(v) for v in pooling_layer_idx]
        else:
            pooling_layer_indices = [int(pooling_layer_idx)]
        normalized_indices = []
        for idx in pooling_layer_indices:
            if idx < 0:
                idx = num_layers - 1
            if 0 <= idx < num_layers and idx not in normalized_indices:
                normalized_indices.append(idx)
        self.pooling_layer_indices = normalized_indices

        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            use_whnn = i in self.pooling_layer_indices
            self.blocks[f'GNN_layer_{i}'] = HGNN_layer(
                in_channels=dim[i],
                out_channels=dim[i + 1],
                use_whnn=use_whnn,
            )
            self.blocks[f'NonLocal_{i}'] = NonLocalBlock(dim[i + 1])
            if i < num_layers - 1:
                self.blocks[f'update_graph_{i}'] = GraphUpdate(num_channels=128, num_heads=1, num_layer=self.num_layers)

    def forward(self, vertex_feat, edge_weight):
        global edge_score
        batch_size, num_dims, num_points = vertex_feat.size()  # bs, 12, num_corr

        # 1. edge_weight[edge_weight>0] = 1
        incidence = edge_weight.clone()  # Clone to avoid modifying input
        incidence[incidence > 0] = 1.0
        # 2. Degree of V, E.
        degree = incidence.sum(dim=1)  # [bs, num_points]

        raw_incidence = incidence.clone()  # Keep a copy
        # Compute inverse degrees as vectors, not full diagonal matrices
        inv_degree_vec = 1.0 / (degree + 1e-10)  # [bs, num_points]
        inv_degree_vec[torch.isinf(inv_degree_vec)] = 0
        # Store as [bs, num_points, 1] for broadcasting in bmm
        inv_edge_degree = inv_degree_vec.unsqueeze(-1)
        inv_vertex_degree = inv_edge_degree  # The initial incidence matrix is symmetric
        del degree, inv_degree_vec  # Free after use

        # 3. edge weight = sum(W, dim=0)
        edge_scale = edge_weight.sum(dim=1)  # [bs, num]
        edge_scale = F.normalize(edge_scale, p=2, dim=1)
        edge_scale = edge_scale.view([batch_size, num_points, 1])

        feat = self.layer0(vertex_feat)
        feat0 = feat.clone()  # Keep initial features
        edge_feat = None
        
        for i in range(self.num_layers):
            theta = math.log(self.lamda / (i + 1) + 1)
            
            # Store previous features for cleanup
            feat_prev = feat
            edge_feat_prev = edge_feat
            
            feat, edge_feat = self.blocks[f'GNN_layer_{i}'](feat, edge_feat, edge_weight, incidence, 
                                                            inv_edge_degree, inv_vertex_degree, edge_scale, 
                                                            self.alpha, theta, feat0, self.k)
            
            # Free previous layer's features
            if i > 0:
                del feat_prev
                if edge_feat_prev is not None:
                    del edge_feat_prev
            
            feat = self.blocks[f'NonLocal_{i}'](feat, edge_weight, raw_incidence)
            feat = F.normalize(feat, p=2, dim=1)
            if edge_feat is not None:
                edge_feat = F.normalize(edge_feat, p=2, dim=1)
            
            # change hypergraph dynamically
            if i < self.num_layers - 1:
                # Store old values for cleanup
                old_incidence = incidence
                old_inv_edge_degree = inv_edge_degree
                old_inv_vertex_degree = inv_vertex_degree
                old_edge_scale = edge_scale
                
                incidence, edge_score, inv_edge_degree, inv_vertex_degree, edge_scale = self.blocks[f'update_graph_{i}'](incidence, feat, edge_feat, i)
                
                # Free old values
                del old_incidence, old_inv_edge_degree, old_inv_vertex_degree, old_edge_scale
        
        # Final cleanup
        del feat0, inv_edge_degree, inv_vertex_degree, edge_scale, incidence
        
        return raw_incidence, raw_incidence.clone(), edge_score, feat


class HyperGCT(nn.Module):
    def __init__(self, config):
        super(HyperGCT, self).__init__()
        self.config = config
        self.inlier_threshold = config.inlier_threshold
        self.num_iterations = 10
        self.num_channels = 128
        self.encoder = HGNN(n_emb_dims=self.num_channels, in_channel=6, pooling_layer_idx=config.pooling_layer_idx)
        self.nms_radius = config.inlier_threshold  # only used during testing
        self.ratio = config.seed_ratio
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True) # changes during training
        self.sigma_d = config.inlier_threshold
        self.k = 40  # neighborhood number in NSM module.

        self.classification = nn.Sequential(
            nn.Conv1d(self.num_channels, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True),
        )

        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        corr, src_pts, tgt_pts, src_normal, tgt_normal = (
            input_data['corr_pos'], input_data['src_keypts'], input_data['tgt_keypts'], input_data['src_normal'],
            input_data['tgt_normal'])
        
        bs, num_corr, num_dim = corr.size()
        FCG_K = int(num_corr * 0.1)
        
        with torch.no_grad():
            # Compute pairwise distances in chunks to avoid OOM
            # Instead of creating [bs, num_corr, num_corr] all at once
            chunk_size = 128  # Process 128 correspondences at a time
            FCG = torch.zeros(bs, num_corr, num_corr, device=src_pts.device, dtype=src_pts.dtype)
            
            for start_idx in range(0, num_corr, chunk_size):
                end_idx = min(start_idx + chunk_size, num_corr)
                
                # Compute distances for chunk: [bs, chunk, num_corr]
                src_chunk = src_pts[:, start_idx:end_idx, None, :]  # [bs, chunk, 1, 3]
                src_dist_chunk = ((src_chunk - src_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5
                
                tgt_chunk = tgt_pts[:, start_idx:end_idx, None, :]
                tgt_dist_chunk = ((tgt_chunk - tgt_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5
                
                pairwise_dist_chunk = src_dist_chunk - tgt_dist_chunk
                FCG[:, start_idx:end_idx, :] = torch.clamp(1 - pairwise_dist_chunk ** 2 / self.sigma_d ** 2, min=0)
                
                del src_chunk, src_dist_chunk, tgt_chunk, tgt_dist_chunk, pairwise_dist_chunk
            
            FCG[:, torch.arange(FCG.shape[1]), torch.arange(FCG.shape[1])] = 0

            # Keep top matches for each row
            sorted_value, _ = torch.topk(FCG, FCG_K, dim=2, largest=True, sorted=False)
            sorted_value = sorted_value.reshape(bs, -1)
            thresh = sorted_value.mean(dim=1, keepdim=True).unsqueeze(2)
            del sorted_value  # Free memory

            # Apply threshold
            FCG = torch.where(FCG < thresh, torch.tensor(0.0, device=FCG.device), FCG)
            del thresh  # Free memory

            # Compute W
            W = torch.matmul(FCG, FCG) * FCG
            del FCG

        F0 = corr
        raw_H, H, edge_score, corr_feats = self.encoder(F0.permute(0, 2, 1), W)  # bs, dim, num_corr
        del W, F0  # Free after encoder
        
        confidence = self.classification(corr_feats).squeeze(1)  # bs, 1, num_corr-> bs, num_corr loss has sigmoid

        # M = distance(normed_corr_feats)
        # construct the feature similarity matrix M for loss calculation
        M = torch.matmul(corr_feats.permute(0, 2, 1), corr_feats)
        M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
        # # set diagnal of M to zero
        M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0

        if not self.training and bs == 1:
            seeds = self.graph_filter(H=H, confidence=confidence, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]

        if not self.training:
            sampled_trans, pred_trans = self.hypo_sampling_and_evaluation(seeds, corr_feats.permute(0, 2, 1), H,
                                                                        src_pts, tgt_pts, src_normal, tgt_normal)
            sampled_trans = sampled_trans.view([-1, 4, 4])
        else:
            sampled_trans, pred_trans = None, None

        #post refinement (only used during testing and bs == 1)
        # if self.config.mode == "test":
        #     pred_trans = self.post_refinement(H, pred_trans, src_pts, tgt_pts)
        #     frag1_warp = transform(src_pts, pred_trans)
        #     distance = torch.sum((frag1_warp - tgt_pts) ** 2, dim=-1) ** 0.5
        #     pred_labels = (distance < self.inlier_threshold).float()
        #     del frag1_warp, distance  # Free test-only tensors

        # if self.config.mode != "test":
        pred_labels = confidence
        
        res = {
            "raw_H": raw_H,
            "hypergraph": H,
            "edge_score": edge_score,
            "final_trans": pred_trans,
            "final_labels": pred_labels,
            "M": M,
            "seeds": seeds,
            "confidence": confidence,
            "sampled_trans": sampled_trans,
            "corr_feats": corr_feats
        }
        return res

    def graph_matrix_reconstruct(self, H, thresh=1, seeds=None):
        """
        Reconstruct the graph matrix from H.
        Input:
            - H: [bs, num_corr, num_corr]
            - seeds: [bs, num_seeds] the index to the seeding correspondence
            - thresh: 0 or 1 loose or tight
        Output:
            - seed_matrix: [bs, num_seeds, num_seeds] or [bs, num_corr, num_corr]
        """
        num_seed = None
        if seeds is not None:
            if len(seeds.size()) == 2:
                _, num_seed = seeds.size()
            else:
                num_seed = len(seeds)

        if len(H.size()) == 3:
            bs, num_corr, _ = H.size()
            merge_matrix = H + H.permute(0, 2, 1)
            merge_matrix = (merge_matrix > thresh).float()

            if seeds is not None:
                seeds_matrix = merge_matrix.gather(1, seeds.unsqueeze(-1).expand(-1, -1, num_corr)).gather(
                    2, seeds.unsqueeze(-2).expand(-1, num_seed, -1))
            else:
                seeds_matrix = merge_matrix

        else:
            num_corr, _ = H.size()
            merge_matrix = H + H.t()
            merge_matrix = (merge_matrix > thresh).float()

            if seeds is not None:
                seeds_matrix = merge_matrix[seeds][:, seeds]
            else:
                seeds_matrix = merge_matrix

        return seeds_matrix.detach()

    def graph_filter(self, H, confidence, max_num):
        assert confidence.shape[0] == 1
        assert H.shape[0] == 1
        # H0 = H
        H = self.graph_matrix_reconstruct(H, 1, None)
        bs, num_corr, _ = H.size()
        D = torch.sum(H, dim=-1) # [bs, num_corr]
        L = torch.diag_embed(D) - H # [bs, num_corr, num_corr]
        xyz = torch.bmm(D[:, :, None].permute(0, 2, 1), L)
        Lscore = torch.norm(xyz, dim=1)  # [bs, num] 绝对值
        #Lscore = xyz.reshape(bs, num_corr)
        low, _ = Lscore.min(dim=1, keepdim=True)
        up, _ = Lscore.max(dim=1, keepdim=True)
        Lscore = (Lscore - low) / (up - low) * (D > 0).float()  # ignore the node that degree == 0

        #parallel Non Maximum Suppression (more efficient)
        score_relation = confidence.T >= confidence  # [num_corr, num_corr], save the relation of leading_eig
        masked_score_relation = score_relation.masked_fill(H == 0, float('inf'))  # ignore the relation that H == 0
        is_local_max = masked_score_relation.min(dim=-1)[0].float() * (
                D > 0).float()  # ignore the node that degree == 0

        score = Lscore * is_local_max
        seed1 = torch.argsort(score, dim=1, descending=True)
        seed2 = torch.argsort(Lscore, dim=1, descending=True)
        sel_len1 = min(max_num, (score > 0).sum().item())  # preserve the seed where score > 0
        set_seed1 = set(seed1[0][:sel_len1].tolist())
        unique_seed2 = [e for e in seed2[0].tolist() if e not in set_seed1][:max_num-sel_len1]

        appended_seed1 = list(set_seed1) + unique_seed2
        return torch.tensor([appended_seed1], device=H.device).detach()

    def cal_inliers_normal(self, inliers_mask, trans, src_normal, tgt_normal):

        pred_src_normal = torch.einsum('bsnm,bmk->bsnk', trans[:, :, :3, :3],
                                       src_normal.permute(0, 2, 1))  # [bs, num_seeds, num_corr, 3]
        pred_src_normal = pred_src_normal.permute(0, 1, 3, 2)
        normal_similarity = (pred_src_normal * tgt_normal[:, None, :, :]).sum(-1)
        normal_similarity = (normal_similarity > 0.7).float()
        normal_similarity = (normal_similarity * inliers_mask).mean(-1)
        return normal_similarity

    def hypo_sampling_and_evaluation(self, seeds, corr_features, H, src_keypts, tgt_keypts, src_normal, tgt_normal):
        src_knn, tgt_knn, total_weight = build_seed_knn_weights(
            seeds, corr_features, H, src_keypts, tgt_keypts, self.sigma, self.sigma_d, self.num_iterations
        )
        return evaluate_hypotheses(
            seeds, H, src_knn, tgt_knn, total_weight, src_keypts, tgt_keypts, src_normal, tgt_normal,
            self.inlier_threshold, self.cal_inliers_normal
        )


    def post_refinement(self, H, initial_trans, src_keypts, tgt_keypts, weights=None):
        """
        Perform post refinement using the initial transformation matrix, only adopted during testing.
        Input
            - initial_trans: [bs, 4, 4]
            - src_keypts:    [bs, num_corr, 3]
            - tgt_keypts:    [bs, num_corr, 3]
            - weights:       [bs, num_corr]
        Output:
            - final_trans:   [bs, 4, 4]
        """
        assert initial_trans.shape[0] == 1
        if self.inlier_threshold == 0.10:  # for 3DMatch
            inlier_threshold_list = [0.10] * 20
        else:  # for KITTI
            inlier_threshold_list = [1.2] * 20
        mask = H
        degree = mask.sum(dim=1)  # [bs, num]
        corr_mask = (degree > 0).float()

        previous_inlier_num = 0
        for inlier_threshold in inlier_threshold_list:
            warped_src_keypts = transform(src_keypts, initial_trans)
            L2_dis = torch.norm(warped_src_keypts - tgt_keypts, dim=-1)
            L2_dis = L2_dis * corr_mask.float() + (1 - corr_mask.float()) * float(1e9)
            MAE_score = (inlier_threshold - L2_dis) / inlier_threshold
            inlier_num = torch.sum(MAE_score * (L2_dis < inlier_threshold), dim=-1)[0]
            pred_inlier = (L2_dis < inlier_threshold)[0]  # assume bs = 1
            if inlier_num <= previous_inlier_num:
                break
            else:
                previous_inlier_num = inlier_num
            initial_trans = kabsch(
                A=src_keypts[:, pred_inlier, :],
                B=tgt_keypts[:, pred_inlier, :],
                ## https://link.springer.com/article/10.1007/s10589-014-9643-2
                # weights=None,
                weights=1 / (1 + (L2_dis / inlier_threshold) ** 2)[:, pred_inlier],
                # weights=((1-L2_dis/inlier_threshold)**2)[:, pred_inlier]
            )
        return initial_trans
