import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

WHNN_ROOT = Path(__file__).resolve().parents[3] / "whnn"
if str(WHNN_ROOT) not in sys.path:
    sys.path.insert(0, str(WHNN_ROOT))

from preprocessing import extend_edge_index
from . import SetHNN


class WHNN(nn.Module):
    """
    Wrapper around the external WHNN SetHNN model that keeps the current
    correspondence pipeline: build the dense compatibility hypergraph exactly
    as HyperGCT does, convert it to the sparse incidence format SetHNN expects,
    and return per-node logits for binary classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inlier_threshold = config.inlier_threshold
        self.num_iterations = 10
        self.num_channels = 128
        self.ratio = config.seed_ratio
        self.sigma = nn.Parameter(torch.Tensor([1.0]).float(), requires_grad=True)
        self.sigma_d = config.inlier_threshold
        self.proc_type = config.proc_type
        corr_feat_dim = 6 + (2 * config.feature_dim if config.use_features else 0)

        # SetHNN reads these directly from args.
        config.num_features = corr_feat_dim
        config.num_classes = 1

        self.model = SetHNN(config)

    def graph_matrix_reconstruct(self, H, thresh=1, seeds=None):
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
        H = self.graph_matrix_reconstruct(H, 1, None)
        bs, num_corr, _ = H.size()
        D = torch.sum(H, dim=-1)
        L = torch.diag_embed(D) - H
        xyz = torch.bmm(D[:, :, None].permute(0, 2, 1), L)
        Lscore = torch.norm(xyz, dim=1)
        low, _ = Lscore.min(dim=1, keepdim=True)
        up, _ = Lscore.max(dim=1, keepdim=True)
        Lscore = (Lscore - low) / (up - low) * (D > 0).float()

        score_relation = confidence.T >= confidence
        masked_score_relation = score_relation.masked_fill(H == 0, float("inf"))
        is_local_max = masked_score_relation.min(dim=-1)[0].float() * (D > 0).float()

        score = Lscore * is_local_max
        seed1 = torch.argsort(score, dim=1, descending=True)
        seed2 = torch.argsort(Lscore, dim=1, descending=True)
        sel_len1 = min(max_num, (score > 0).sum().item())
        set_seed1 = set(seed1[0][:sel_len1].tolist())
        unique_seed2 = [e for e in seed2[0].tolist() if e not in set_seed1][:max_num - sel_len1]

        appended_seed1 = list(set_seed1) + unique_seed2
        return torch.tensor([appended_seed1], device=H.device).detach()

    def _build_dense_hypergraph(self, src_pts, tgt_pts):
        bs, num_corr, _ = src_pts.shape
        fcg_k = max(1, int(num_corr * 0.1))

        with torch.no_grad():
            chunk_size = 128
            fcg = torch.zeros(bs, num_corr, num_corr, device=src_pts.device, dtype=src_pts.dtype)

            for start_idx in range(0, num_corr, chunk_size):
                end_idx = min(start_idx + chunk_size, num_corr)
                src_chunk = src_pts[:, start_idx:end_idx, None, :]
                src_dist_chunk = ((src_chunk - src_pts[:, None, :, :]) ** 2).sum(-1).sqrt()

                tgt_chunk = tgt_pts[:, start_idx:end_idx, None, :]
                tgt_dist_chunk = ((tgt_chunk - tgt_pts[:, None, :, :]) ** 2).sum(-1).sqrt()

                pairwise_dist_chunk = src_dist_chunk - tgt_dist_chunk
                fcg[:, start_idx:end_idx, :] = torch.clamp(
                    1 - pairwise_dist_chunk ** 2 / self.sigma_d ** 2,
                    min=0,
                )

            diag = torch.arange(num_corr, device=fcg.device)
            fcg[:, diag, diag] = 0

            sorted_value, _ = torch.topk(fcg, fcg_k, dim=2, largest=True, sorted=False)
            thresh = sorted_value.reshape(bs, -1).mean(dim=1, keepdim=True).unsqueeze(2)
            fcg = torch.where(fcg < thresh, torch.zeros((), device=fcg.device, dtype=fcg.dtype), fcg)
            hypergraph = torch.matmul(fcg, fcg) * fcg

        return hypergraph

    def _build_sample(self, x, dense_hypergraph):
        node_idx, hyperedge_idx = (dense_hypergraph > 0).nonzero(as_tuple=True)
        if node_idx.numel() == 0:
            node_idx = torch.arange(x.size(0), device=x.device)
            hyperedge_idx = torch.arange(x.size(0), device=x.device)

        edge_index = torch.stack((node_idx, hyperedge_idx), dim=0)
        data = SimpleNamespace(x=x, edge_index=edge_index)

        if self.proc_type in {"SAB", "ISAB"}:
            reversed_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)
            data.extended_index = extend_edge_index(edge_index).to(x.device)
            data.reversed_extended_index = extend_edge_index(reversed_edge_index).to(x.device)

        return data

    def forward(self, input_data):
        corr = input_data["corr_pos"]
        src_pts = input_data["src_keypts"]
        tgt_pts = input_data["tgt_keypts"]

        H = self._build_dense_hypergraph(src_pts, tgt_pts)

        logits = []
        corr_feats = []
        for batch_idx in range(corr.size(0)):
            sample = self._build_sample(corr[batch_idx], H[batch_idx])
            sample_logits, sample_embeddings, _ = self.model(sample)
            logits.append(sample_logits.reshape(-1))
            corr_feats.append(sample_embeddings.transpose(0, 1))

        confidence = torch.stack(logits, dim=0)
        corr_feats = torch.stack(corr_feats, dim=0)
        M = torch.matmul(corr_feats.permute(0, 2, 1), corr_feats)
        M = torch.clamp(1 - (1 - M) / self.sigma ** 2, min=0, max=1)
        M[:, torch.arange(M.shape[1]), torch.arange(M.shape[1])] = 0
        num_corr = confidence.shape[1]
        if not self.training and confidence.shape[0] == 1:
            seeds = self.graph_filter(H=H, confidence=confidence, max_num=int(num_corr * self.ratio))
        else:
            seeds = torch.argsort(confidence, dim=1, descending=True)[:, 0:int(num_corr * self.ratio)]

        return {
            "confidence": confidence,
            "final_labels": confidence,
            "corr_feats": corr_feats,
            "M": M,
            "hypergraph": H,
            "seeds": seeds,
        }
