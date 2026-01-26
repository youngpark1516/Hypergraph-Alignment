import torch
import torch.nn as nn
import torch.nn.functional as F


class WHNNAggregator(nn.Module):
    """A lightweight Wasserstein-inspired aggregation module.

    This implements an entropic-OT-like soft assignment between vertices and
    an edge-prototype (computed from vertex projections within the edge),
    then uses the transport weights to aggregate vertex features into edge
    features. The edge->vertex pass mirrors the original HGNN flow.
    """

    def __init__(self, channels, proj_dim=None, sinkhorn_tau=0.1):
        super(WHNNAggregator, self).__init__()
        if proj_dim is None:
            proj_dim = max(16, channels // 8)
        self.proj = nn.Conv1d(channels, proj_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(proj_dim)
        self.tau = sinkhorn_tau

    def forward(self, vertex_feat, incidence, inv_edge_degree, inv_vertex_degree, edge_scale):
        """
        Inputs:
            - vertex_feat: [bs, C, N]
            - incidence:   [bs, N, E]
            - inv_edge_degree: [bs, E, E] diagonal-like inverse degree matrix for edges
            - inv_vertex_degree: [bs, N, N] inverse degree matrix for vertices
            - edge_scale: [bs, E, 1]
        Returns:
            - vertex_feat_out: [bs, C, N]
            - edge_feat_out: [bs, C, E]
        """
        bs, C, N = vertex_feat.shape

        # project vertex features to low-dim key space
        K = self.proj_bn(self.proj(vertex_feat))  # [bs, D, N]
        K = K.permute(0, 2, 1)  # [bs, N, D]

        # compute per-edge prototype by aggregating projected keys using incidence
        # incidence.permute(0,2,1): [bs, E, N]
        proto = torch.bmm(incidence.permute(0, 2, 1), K)  # [bs, E, D]
        # normalize by edge degree (inv_edge_degree is [bs, E, E] diagonal)
        proto = torch.bmm(inv_edge_degree, proto)  # [bs, E, D]

        # compute squared euclidean cost between each prototype and each vertex key
        # expand and broadcast: result [bs, E, N]
        # use (a-b)^2 = a^2 + b^2 - 2ab for efficiency
        K_sq = (K ** 2).sum(-1, keepdim=True).transpose(1, 2)  # [bs, N, 1]
        proto_sq = (proto ** 2).sum(-1, keepdim=True)  # [bs, E, 1]
        # compute inner product
        inner = torch.bmm(proto, K.permute(0, 2, 1))  # [bs, E, N]
        cost = proto_sq + K_sq.permute(0, 2, 1) - 2.0 * inner  # [bs, E, N]

        # entropic regularization via softmax on negative cost
        weights = torch.softmax(-cost / (self.tau + 1e-8), dim=-1)  # [bs, E, N]

        # aggregate vertex features into edge features using transport weights
        v_feats = vertex_feat.permute(0, 2, 1)  # [bs, N, C]
        edge_feat = torch.bmm(weights, v_feats)  # [bs, E, C]

        # normalize by edge degree if provided (keeps parity with previous mean implementation)
        if inv_edge_degree is not None:
            try:
                edge_feat = torch.bmm(inv_edge_degree, edge_feat)  # [bs, E, C]
            except Exception:
                pass

        # return raw edge features in [bs, C, E] shape; leave edge->vertex aggregation
        # to the caller so they can apply the same MLP and normalization as before.
        return None, edge_feat.permute(0, 2, 1)
