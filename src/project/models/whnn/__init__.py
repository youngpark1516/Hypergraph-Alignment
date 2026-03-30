import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

WHNN_ROOT = Path(__file__).resolve().parents[3] / "whnn"
if str(WHNN_ROOT) not in sys.path:
    sys.path.insert(0, str(WHNN_ROOT))

from equiv_set import MLP
from layers_set import HyperLayer


class SetHNN(nn.Module):
    """
    Project-local variant of the WHNN SetHNN model.

    It mirrors src/whnn/models_set.py but also exposes the final node
    embeddings before the classifier so downstream code can construct the
    pairwise score matrix M.
    """

    def __init__(self, args, norm=None):
        super(SetHNN, self).__init__()
        self.All_num_layers = args.All_num_layers
        num_features = args.num_features
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.sharing = args.sharing

        self.layers = nn.ModuleList()
        self.dropout = args.dropout
        self.input_dropout = args.input_dropout

        self.proc_type_V2E = args.proc_type
        self.proc_type_E2V = args.proc_type
        self.pooling_type_V2E = args.pooling_type
        self.pooling_type_E2V = args.pooling_type

        self.lin = nn.Linear(num_features, args.MLP_hidden)
        self.layers.append(
            HyperLayer(
                proc_type_V2E=self.proc_type_V2E,
                pooling_type_V2E=self.pooling_type_V2E,
                proc_type_E2V=self.proc_type_E2V,
                pooling_type_E2V=self.pooling_type_E2V,
                args=args,
            )
        )

        for _ in range(self.All_num_layers - 1):
            self.layers.append(
                HyperLayer(
                    proc_type_V2E=self.proc_type_V2E,
                    pooling_type_V2E=self.pooling_type_V2E,
                    proc_type_E2V=self.proc_type_E2V,
                    pooling_type_E2V=self.pooling_type_E2V,
                    args=args,
                )
            )

        self.classifier = MLP(
            in_channels=args.MLP_hidden,
            hidden_channels=args.Classifier_hidden,
            out_channels=args.num_classes,
            num_layers=args.Classifier_num_layers,
            dropout=self.dropout,
            Normalization=self.NormLayer,
            InputNorm=False,
        )

    def reset_parameters(self):
        self.lin.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.classifier.reset_parameters()

    def _reset_pooling_caches(self):
        # The upstream FPSWE pooler caches tensors derived from the current
        # hypergraph structure. In this project we run many different
        # correspondence graphs through the same module, so those caches must
        # be cleared before each forward.
        for module in self.modules():
            if hasattr(module, "deg_helper"):
                module.deg_helper = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        self._reset_pooling_caches()
        edge_index = edge_index.clone()
        cidx = edge_index[1].min()
        edge_index[1] -= cidx
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin(x)
        x0 = x.clone()
        for i, _ in enumerate(self.layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            idx = 0 if self.sharing else i
            x, _ = self.layers[idx](x, x0, edge_index, reversed_edge_index, data)
            x = F.relu(x)

        node_embeddings = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(node_embeddings).squeeze(-1)
        split_idx_dict = None
        return logits, node_embeddings, split_idx_dict


from .model import WHNN

__all__ = [
    "SetHNN",
    "WHNN",
]
