import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from project.utils.SE3 import transform, integrate_trans
from project.utils.timer import Timer
import math
from torch_scatter import scatter_add


def sparse_sort(values, index):
    """
    Sort values within groups - VECTORIZED for speed
    """
    device = values.device
    N, num_proj = values.shape
    
    # Create sorting key: combine group index with value
    # This allows us to sort all groups simultaneously
    max_val = values.abs().max().item() + 1
    sort_keys = index[:, None].float() * max_val + values
    
    # Sort all projections at once
    sorted_keys, sort_idx = torch.sort(sort_keys, dim=0)
    
    # Gather sorted values
    batch_idx = torch.arange(num_proj, device=device).unsqueeze(0).expand(N, -1)
    sorted_values = torch.gather(values, 0, sort_idx)
    
    del sort_keys, batch_idx
    return sorted_values, sort_idx


def interp1d(x, y, xnew, ynew, index):
    """
    Ultra memory-efficient 1D interpolation - process one projection at a time
    """
    num_proj, M = xnew.shape
    num_proj, N = x.shape
    
    # Process each projection separately to minimize memory
    for proj_idx in range(num_proj):
        x_proj = x[proj_idx]  # [N]
        y_proj = y[proj_idx]  # [N]
        xnew_proj = xnew[proj_idx]  # [M]
        
        # Process in small chunks
        chunk_size = 64  # Very small chunks
        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            xnew_chunk = xnew_proj[start_idx:end_idx]  # [chunk_size]
            
            # Find nearest neighbor: [chunk_size, 1] - [1, N] = [chunk_size, N]
            dists = torch.abs(xnew_chunk.unsqueeze(1) - x_proj.unsqueeze(0))
            nearest_idx = torch.argmin(dists, dim=1)  # [chunk_size]
            
            # Gather y values
            ynew[proj_idx, start_idx:end_idx] = y_proj[nearest_idx]
            
            del dists, nearest_idx, xnew_chunk
        
        del x_proj, y_proj, xnew_proj
    
    return ynew


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
        k = round(num * 0.1 * (num_layer - 1 - iter))
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
        degree_E = H.sum(dim=1)
        De = torch.diag_embed(degree_E)  # torch.sparse_matrix
        De_n_1 = De ** -1
        De_n_1[torch.isinf(De_n_1)] = 0

        degree_V = H.sum(dim=2)
        Dv = torch.diag_embed(degree_V)
        Dv_n_1 = Dv ** -1
        Dv_n_1[torch.isinf(Dv_n_1)] = 0

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
        Xslices_sorted_interpolated = interp1d(x, y, xnew, ynew, hyperedge_index_1_sorted).view(self.num_projections, -1)
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
        elif self.aggr == 'wasserstein':
            # Use much smaller parameters for GPU memory efficiency
            self.v2e_pool = FPSWE_pool(d_in=num_channels, num_anchors=16, num_projections=32, anch_freeze=True)
            self.e2v_pool = FPSWE_pool(d_in=num_channels, num_anchors=16, num_projections=32, anch_freeze=True)
            # Project back to num_channels
            self.v2e_proj = nn.Linear(32, num_channels)
            self.e2v_proj = nn.Linear(32, num_channels)

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
                edge_feat_out = torch.bmm(inv_edge_degree, feature)

                if edge_feat is not None:
                    edge_feat_out = self.mlp(torch.cat((edge_feat, edge_feat_out.permute(0, 2, 1)), dim=1)).permute(0, 2,
                                                                                                          1)  # [bs, num, dim]

                # update message of e
                feature = edge_scale * edge_feat_out  # [bs, num, 1] * [bs, num, dim]

                # aggregation message from e to v
                feature = torch.bmm(incidence, feature)
                feature = torch.bmm(inv_vertex_degree, feature)

                vertex_feat_out = feature.permute(0, 2, 1)  # [bs, dim, num]
                edge_feat_out = edge_feat_out.permute(0, 2, 1)

            elif self.aggr == 'self_attention':
                # e->v nonlocal net using weight matrix W
                feature = vertex_feat.permute(0, 2, 1)
                feature = torch.bmm(incidence.permute(0, 2, 1), feature)  # H^T=H
                edge_feat_out = torch.bmm(inv_edge_degree, feature)  # [bs, num, dim]

                feat = edge_feat_out.permute(0, 2, 1)  # [bs, dim, num]
                score = torch.softmax(self.fc(feat).permute(0, 2, 1), dim=1)  # [bs, num, 1]
                feature = score * edge_scale * feature

                feature = torch.bmm(incidence, feature)
                feature = torch.bmm(inv_vertex_degree, feature)

                vertex_feat_out = feature.permute(0, 2, 1)
                edge_feat_out = edge_feat_out.permute(0, 2, 1)

            elif self.aggr == 'wasserstein':
                # v->e: Use Wasserstein aggregation - OPTIMIZED
                vertex_feat_t = vertex_feat.permute(0, 2, 1)  # [bs, num_nodes, dim]
                
                # CRITICAL FIX: Cap memberships per vertex to prevent OOM
                k_edges = min(20, num_nodes)  # Keep top-k strongest edges per vertex
                top_k_values, top_k_indices = torch.topk(incidence, k=k_edges, dim=2, largest=True, sorted=False)
                top_k_mask = torch.zeros_like(incidence, dtype=torch.bool)
                top_k_mask.scatter_(2, top_k_indices, True)
                incidence_sparse = incidence * top_k_mask  # Zero out non-top-k entries
                
                # Free intermediate tensors
                del top_k_values, top_k_indices, top_k_mask
                
                # v->e: Find all (batch, vertex, edge) tuples where incidence > 0
                nonzero_mask = incidence_sparse > 0  # [bs, num_nodes, num_nodes]
                batch_idx, vertex_idx, edge_idx = torch.nonzero(nonzero_mask, as_tuple=True)
                
                # Create global indices by adding batch offsets
                global_vertex_idx = batch_idx * num_nodes + vertex_idx
                global_edge_idx = batch_idx * num_nodes + edge_idx
                
                # Free indices we don't need anymore
                del batch_idx, vertex_idx, edge_idx
                
                # Gather vertex features for all connections across all batches
                vertex_feat_flat = vertex_feat_t.reshape(batch_size * num_nodes, num_channels)
                vertex_feat_expanded = vertex_feat_flat[global_vertex_idx]  # [total_connections, dim]
                
                # Free intermediate tensors
                del vertex_feat_flat, global_vertex_idx
                
                # Wasserstein pooling with global edge indices - returns compact output
                edge_feat_compact, edge_indices = self.v2e_pool(vertex_feat_expanded, global_edge_idx)  # [num_edges, 32]
                del vertex_feat_expanded, global_edge_idx
                
                edge_feat_compact = self.v2e_proj(edge_feat_compact)  # [num_edges, num_channels]
                
                # Map compact output back to full [bs*num_nodes] space
                edge_feat_pooled = torch.zeros(batch_size * num_nodes, num_channels,
                                               device=edge_feat_compact.device, dtype=edge_feat_compact.dtype)
                edge_feat_pooled[edge_indices] = edge_feat_compact
                del edge_feat_compact, edge_indices
                
                # Reshape back to batch
                edge_feat_out = edge_feat_pooled.reshape(batch_size, num_nodes, num_channels)
                del edge_feat_pooled
                
                if edge_feat is not None:
                    edge_feat_out = self.mlp(torch.cat((edge_feat, edge_feat_out.permute(0, 2, 1)), dim=1)).permute(0, 2, 1)
                
                # e->v: Use Wasserstein aggregation from edges to vertices
                feature = edge_scale * edge_feat_out  # [bs, num_nodes, dim]
                
                # e->v: Find all (batch, edge, vertex) tuples (transpose of incidence)
                batch_idx, edge_idx, vertex_idx = torch.nonzero(nonzero_mask.transpose(1, 2), as_tuple=True)
                del nonzero_mask, incidence_sparse  # Free sparse incidence
                
                # Create global indices
                global_edge_idx = batch_idx * num_nodes + edge_idx
                global_vertex_idx = batch_idx * num_nodes + vertex_idx
                del batch_idx, edge_idx, vertex_idx
                
                # Gather edge features for all connections across all batches
                feature_flat = feature.reshape(batch_size * num_nodes, num_channels)
                feature_expanded = feature_flat[global_edge_idx]  # [total_connections, dim]
                
                # Free intermediate tensors
                del feature_flat, global_edge_idx
                
                # Wasserstein pooling with global vertex indices - returns compact output
                vertex_feat_compact, vertex_indices = self.e2v_pool(feature_expanded, global_vertex_idx)  # [num_vertices, 32]
                del feature_expanded, global_vertex_idx
                
                vertex_feat_compact = self.e2v_proj(vertex_feat_compact)  # [num_vertices, num_channels]
                
                # Map compact output back to full [bs*num_nodes] space
                vertex_feat_pooled = torch.zeros(batch_size * num_nodes, num_channels,
                                                 device=vertex_feat_compact.device, dtype=vertex_feat_compact.dtype)
                vertex_feat_pooled[vertex_indices] = vertex_feat_compact
                del vertex_feat_compact, vertex_indices
                
                # Reshape back to batch
                vertex_feat_out = vertex_feat_pooled.reshape(batch_size, num_nodes, num_channels).permute(0, 2, 1)
                del vertex_feat_pooled
                
                edge_feat_out = edge_feat_out.permute(0, 2, 1)

            else:
                raise NotImplementedError

        return vertex_feat_out, edge_feat_out


class HGNN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, residual_connection=False, use_edge_feature=True, aggr='mean'):
        super(HGNN_layer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.feat_agg = feature_aggregation_layer(k=20, num_channels=out_channels, aggr=aggr)
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
                 aggr='wasserstein'):
        super(HGNN, self).__init__()
        self.k = k
        self.lamda = lamda
        self.alpha = alpha
        self.num_layers = num_layers
        self.change_H0 = False
        self.aggr = aggr
        dim = [n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims, n_emb_dims,
               n_emb_dims]
        self.layer0 = nn.Conv1d(in_channel, dim[0], kernel_size=1, bias=True)
        self.blocks = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks[f'GNN_layer_{i}'] = HGNN_layer(in_channels=dim[i], out_channels=dim[i + 1], aggr=aggr)
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
        degree = incidence.sum(dim=1)

        raw_incidence = incidence.clone()  # Keep a copy
        D = torch.diag_embed(degree)  # torch.sparse_matrix
        del degree  # Free after use
        
        inv_edge_degree = D ** -1
        inv_edge_degree[torch.isinf(inv_edge_degree)] = 0
        inv_vertex_degree = inv_edge_degree  # The initial incidence matrix is symmetric
        del D  # Free after use

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


class MethodName(nn.Module):
    def __init__(self, config):
        super(MethodName, self).__init__()
        self.config = config
        self.inlier_threshold = config.inlier_threshold
        self.num_iterations = 10
        self.num_channels = 128
        self.encoder = HGNN(n_emb_dims=self.num_channels, in_channel=6)
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
            # pairwise distance compute

            src_dist = ((src_pts[:, :, None, :] - src_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5
            tgt_dist = ((tgt_pts[:, :, None, :] - tgt_pts[:, None, :, :]) ** 2).sum(-1) ** 0.5

            pairwise_dist = src_dist - tgt_dist
            del src_dist, tgt_dist
            FCG = torch.clamp(1 - pairwise_dist ** 2 / self.sigma_d ** 2, min=0)
            del pairwise_dist
            FCG[:, torch.arange(FCG.shape[1]), torch.arange(FCG.shape[1])] = 0

            # Remain top matches for each row
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


        if self.config.mode == "test":
            M = None
        else:
            # M = distance(normed_corr_feats)
            # construct the feature similarity matrix M for loss calculation
            M = torch.matmul(corr_feats.permute(0, 2, 1), corr_feats)
            M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
            # # set diagnal of M to zero
            M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0

        if self.config.mode == "test":
            seeds = self.graph_filter(H=H, confidence=confidence, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]

        sampled_trans, pred_trans = self.hypo_sampling_and_evaluation(seeds, corr_feats.permute(0, 2, 1), H,
                                                                    src_pts, tgt_pts, src_normal, tgt_normal)
        sampled_trans = sampled_trans.view([-1, 4, 4])

        #post refinement (only used during testing and bs == 1)
        if self.config.mode == "test":
            pred_trans = self.post_refinement(H, pred_trans, src_pts, tgt_pts)
            frag1_warp = transform(src_pts, pred_trans)
            distance = torch.sum((frag1_warp - tgt_pts) ** 2, dim=-1) ** 0.5
            pred_labels = (distance < self.inlier_threshold).float()
            del frag1_warp, distance  # Free test-only tensors

        if self.config.mode != "test":
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
            "sampled_trans": sampled_trans
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
        # 1、每个seed生成初始假设
        bs, num_corr, num_channels = corr_features.size()
        _, num_seeds = seeds.size()
        assert num_seeds > 0
        k = min(10, num_corr - 1)
        mask = H  # self.graph_matrix_reconstruct(H, 0, None)
        feature_distance = 1 - torch.matmul(corr_features, corr_features.transpose(2, 1))  # normalized
        masked_feature_distance = feature_distance * mask.float() + (1 - mask.float()) * float(1e9)
        knn_idx = torch.topk(masked_feature_distance, k, largest=False)[1]
        knn_idx = knn_idx.gather(dim=1, index=seeds[:, :, None].expand(-1, -1, k))  # [bs, num_seeds, k]
        # 初始假设更精确
        #################################
        # construct the feature consistency matrix of each correspondence subset.
        #################################
        knn_features = corr_features.gather(dim=1,
                                            index=knn_idx.view([bs, -1])[:, :, None].expand(-1, -1, num_channels)).view(
            [bs, -1, k, num_channels])  # [bs, num_seeds, k, num_channels]
        knn_M = torch.matmul(knn_features, knn_features.permute(0, 1, 3, 2))
        knn_M = torch.clamp(1 - (1 - knn_M) / self.sigma ** 2, min=0)
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
        knn_M = torch.clamp(1 - knn_M ** 2 / self.sigma_d ** 2, min=0)
        knn_M = knn_M.view([-1, k, k])
        spatial_knn_M = knn_M

        #################################
        # Power iteratation to get the inlier probability
        #################################
        total_knn_M = feature_knn_M * spatial_knn_M  # 有用
        total_knn_M[:, torch.arange(total_knn_M.shape[1]), torch.arange(total_knn_M.shape[1])] = 0
        total_weight = self.cal_leading_eigenvector(total_knn_M, method='power')
        total_weight = total_weight.view([bs, -1, k])
        total_weight = total_weight / (torch.sum(total_weight, dim=-1, keepdim=True) + 1e-6)
        total_weight = total_weight.view([-1, k])
        #################################
        # calculate the transformation by weighted least-squares for each subsets in parallel
        #################################
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
        fitness = self.cal_inliers_normal(seed_L2_dis < self.inlier_threshold, seedwise_trans, src_normal, tgt_normal)

        h = int(num_seeds * 0.1)
        hypo_inliers_idx = torch.topk(fitness, h, -1, largest=True)[1]  # [bs, h]
        #seeds = seeds.gather(1, hypo_inliers_idx)  # [bs, h]
        seed_mask = seed_mask.gather(dim=1,
                                     index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]
        L2_dis = L2_dis.gather(dim=1, index=hypo_inliers_idx[:, :, None].expand(-1, -1, num_corr))  # [bs, h, num_corr]

        # 计算最大滑动位置
        max_length = seed_mask.sum(dim=2).min()

        # prepare for sampling
        best_score, best_trans, best_labels = None, None, None
        L2_dis = L2_dis * seed_mask + (1 - seed_mask) * float(1e9)  # [bs, num_seeds, num_corr]
        # 3、滑动窗口采样
        s = 6  # 窗口长度
        m = 3  # 步长
        iters = 15 # 采样次数
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
            MAE_score = (self.inlier_threshold - sampled_L2_dis) / self.inlier_threshold
            fitness = torch.sum(MAE_score * (sampled_L2_dis < self.inlier_threshold), dim=-1)
            sampled_best_guess = fitness.argmax(dim=1)  # [bs, 1]
            sampled_best_score = fitness.gather(dim=1, index=sampled_best_guess[:, None]).squeeze(1)  # [bs, 1]

            sampled_best_trans = sampled_trans.gather(dim=1,
                                                      index=sampled_best_guess[:, None, None, None].expand(-1, -1, 4,
                                                                                                           4)).squeeze(
                1)  # [bs, 4, 4]

            # sampled_best_labels = sampled_L2_dis.gather(dim=1,
            #                                             index=sampled_best_guess[:, None, None].expand(-1, -1,
            #                                                                                            sampled_L2_dis.shape[
            #                                                                                                2])).squeeze(
            #     1)  # [bs, corr_num]
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
        #final_labels = (best_labels < self.inlier_threshold).float()
        return torch.stack(sampled_list), final_trans

    def cal_leading_eigenvector(self, M, method='power'):
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
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

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
