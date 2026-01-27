import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch.nn import Linear


class APPNPNet(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 feat_dropout: float = 0.5, k: int = 10, alpha: float = 0.1, appnp_dropout: float = 0.0):
        super().__init__()
        self.lin1 = Linear(in_dim, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.feat_dropout = feat_dropout
        self.prop = APPNP(K=k, alpha=alpha, dropout=appnp_dropout, cached=False)  # cached=False (edge가 변함)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, edge_weight=edge_weight)
        return x