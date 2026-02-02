import torch.nn.functional as F
from torch import nn

from torch_geometric.nn import GCNConv, GATConv, APPNP
from torch.nn import Linear


class APPNPNet(nn.Module):
    """
    APPNP model for node classification.

    Architecture:
      - 2-layer MLP (Linear -> ReLU -> Linear) to produce node logits
      - APPNP propagation to diffuse logits over the graph
        (Personalized PageRank-style propagation)

    Notes:
      - We set cached=False because the graph structure (edge_index / edge_weight)
        may change across epochs (e.g., edge dropping, reweighting, masking).
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int,
                 feat_dropout: float = 0.5, k: int = 10, alpha: float = 0.1, appnp_dropout: float = 0.0):
        """
        Args:
            in_dim: Input feature dimension.
            hid_dim: Hidden dimension of the MLP.
            out_dim: Number of output classes.
            feat_dropout: Dropout probability applied to node features / hidden activations.
            k: Number of propagation steps in APPNP.
            alpha: Teleport (restart) probability in APPNP.
            appnp_dropout: Dropout inside APPNP propagation (on message passing).
        """
        super().__init__()
        self.lin1 = Linear(in_dim, hid_dim)
        self.lin2 = Linear(hid_dim, out_dim)
        self.feat_dropout = feat_dropout
        self.prop = APPNP(K=k, alpha=alpha, dropout=appnp_dropout, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass.

        Args:
            x: Node features of shape (N, in_dim).
            edge_index: Graph connectivity in COO format, shape (2, E).
            edge_weight: Optional edge weights of shape (E,).

        Returns:
            Logits of shape (N, out_dim).
        """
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.feat_dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index, edge_weight=edge_weight)
        return x
