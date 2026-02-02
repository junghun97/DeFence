"""
DeFence: Decoupled Feature Anchors for Robust Node Classification under Joint Label-Structure Noise
"""

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch import nn


def get_hidden(model: nn.Module, data) -> torch.Tensor:
    """Return hidden features h = ReLU(lin1(x)) for all nodes (N, D)."""
    model.eval()
    x, ei = data.x, data.edge_index

    h = F.relu(model.lin1(x))
    return h.detach().cpu()  # (N, D)

def _hungarian_perm_from_confusion(conf_mat: np.ndarray) -> np.ndarray:
    """Compute cluster->class permutation that maximizes conf_mat[cluster, class]."""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-conf_mat)
        return col_ind
    except Exception:
        C = conf_mat.shape[0]
        used_cls = set()
        perm = np.zeros(C, dtype=np.int64)
        for c in range(C):
            row = conf_mat[c].copy()
            for u in used_cls: row[u] = -1
            j = int(row.argmax())
            perm[c] = j
            used_cls.add(j)
        return perm

def compute_cluster_to_class_perm_by_train(
    cluster_hard: torch.Tensor,   # (N,)
    y: torch.Tensor,              # (N,)
    train_mask: torch.Tensor,     # (N,)
    num_classes: int
) -> torch.Tensor:
    """
    Build a (cluster, class) count matrix on training nodes and return the
    cluster->class permutation (length C), where perm[c] is the matched class.
    """

    device = cluster_hard.device
    m = train_mask.bool().to(device)
    ch = cluster_hard[m]
    yt = y[m]

    C = int(num_classes)
    conf = torch.zeros(C, C, dtype=torch.long, device=device)
    if ch.numel() > 0:
        for ci, yi in zip(ch.tolist(), yt.tolist()):
            if 0 <= ci < C and 0 <= yi < C:
                conf[ci, yi] += 1

    perm_np = _hungarian_perm_from_confusion(conf.detach().cpu().numpy())
    perm = torch.tensor(perm_np, dtype=torch.long, device=device)

    return perm

def _flatten_grads(grads, params):
    """Flatten a list of gradients into a single 1D vector (None -> zeros)."""
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs, dim=0)

def _pick_params_for_alignment(model, k_last_tensors: int = 2):
    """
    Pick the last k parameter tensors for alignment (cheap approximation).
    Typically captures the final classifier weight/bias.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    k = min(k_last_tensors, len(params))
    return params[-k:]

def _map_scores_to_weights(scores: torch.Tensor, idxes: torch.Tensor, N: int,
                           lo: float = 0.5, hi: float = 1.0) -> torch.Tensor:
    
    """
    Map per-item scores to weights in [lo, hi] for the indices in `idxes`.
    - Default weight is `lo` for all N items.
    - Negative scores are clamped to 0.
    - Scores are normalized by max score, then linearly scaled to [lo, hi].
    """                           
    w = torch.full((N,), lo, device=scores.device, dtype=scores.dtype)
    if idxes.numel() == 0:
        return w
    s = scores.clone()
    s = torch.relu(s)
    smax = s.max()
    if float(smax) == 0.0:
        w[idxes] = lo
        return w
    s_norm = s / (smax + 1e-12)            # [0,1]
    w[idxes] = lo + (hi - lo) * s_norm     # [0.5,1.0]
    return w
