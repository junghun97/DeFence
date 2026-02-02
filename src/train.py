import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .utils import *
from .utils import _pick_params_for_alignment, _flatten_grads, _map_scores_to_weights


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility (Python + PyTorch)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: nn.Module, data, split: str = "val", edge_weight=None, edge_index_override=None) -> Tuple[float, float]:
    """
    Evaluate classification performance on a given split.

    Args:
        model: GNN model producing logits of shape (N, C).
        data: PyG-style data object with x, edge_index, y, and masks.
        split: One of {"train", "val", "test"}.
        edge_weight: Optional edge weights for message passing.
        edge_index_override: If provided, use this edge_index instead of data.edge_index.

    Returns:
        (acc, loss): Accuracy and cross-entropy loss on the selected split.
    """
    
    model.eval()
    ei = edge_index_override if edge_index_override is not None else data.edge_index
    logits = model(data.x, ei, edge_weight=edge_weight)

    mask = data.train_mask if split == "train" else data.val_mask if split == "val" else data.test_mask
    pred = logits[mask].argmax(dim=-1)
    acc = (pred == data.y[mask]).float().mean().item()
    loss = F.cross_entropy(logits[mask], data.y[mask]).item()
    return acc, loss

def compute_meta_weights_by_grad_align(
    model: nn.Module,
    data,
    out: torch.Tensor,
    max_params: int = 2,
    max_train_samples: int = 2048,
) -> torch.Tensor:
    """
    Compute per-node training weights via gradient alignment.

    The idea:
      - Compute a validation gradient g_val (on val nodes).
      - For each (sampled) training node i, compute its gradient g_i.
      - Use dot(g_val, g_i) as an alignment score.
      - Map scores to weights in [0.5, 1.0] for training nodes; others remain 0.5.

    Args:
        model: Current model (used only to select alignment parameters).
        data: Data object with y and train/val masks.
        out: Logits tensor of shape (N, C) from a forward pass.
        max_params: Use only the last k parameter tensors to reduce cost.
        max_train_samples: Subsample training nodes for efficiency.

    Returns:
        w: Weight vector of shape (N,), where non-train nodes keep 0.5 by default.
    """
    
    device = out.device
    y = data.y
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).squeeze(1)
    val_idx   = torch.nonzero(data.val_mask,   as_tuple=False).squeeze(1)

    N = out.size(0)
    if val_idx.numel() == 0 or train_idx.numel() == 0:
        return torch.full((N,), 0.5, device=device, dtype=out.dtype)

    align_params = _pick_params_for_alignment(model, k_last_tensors=max_params)

    val_loss = F.cross_entropy(out[val_idx], y[val_idx])
    g_val = torch.autograd.grad(val_loss, align_params, retain_graph=True, allow_unused=True)
    g_val = _flatten_grads(g_val, align_params).detach()

    if max_train_samples is not None and train_idx.numel() > max_train_samples:
        perm = torch.randperm(train_idx.numel(), device=device)[:max_train_samples]
        train_idx_sampled = train_idx[perm]
    else:
        train_idx_sampled = train_idx

    scores = []
    for i in train_idx_sampled:
        li = F.cross_entropy(out[i].unsqueeze(0), y[i].unsqueeze(0))
        g_i = torch.autograd.grad(li, align_params, retain_graph=True, allow_unused=True)
        g_i = _flatten_grads(g_i, align_params)
        score = torch.dot(g_val, g_i)  # alignment
        scores.append(score)

    if len(scores) == 0:
        return torch.full((N,), 0.5, device=device, dtype=out.dtype)

    scores = torch.stack(scores, dim=0)  # [S]
    w = _map_scores_to_weights(scores, train_idx_sampled, N, lo=0.5, hi=1.0)
    
    return w

def get_hidden_for_pair(model: nn.Module, data) -> torch.Tensor:
    """
    Return hidden representations used for the pairwise cluster loss.
    Note: returns a device tensor (no detach/cpu), since it is used in training.
    """
    x, ei = data.x, data.edge_index
    h = F.relu(model.lin1(x))

    return h

def compute_pairwise_cluster_loss(
    hidden: torch.Tensor,
    data,
    cluster_labels: torch.Tensor,           # (N,) long
    num_pos: int = 1024,
    num_neg: int = 1024,
    margin: float = 0.3,
    pos_w: float = 1.0,
    neg_w: float = 1.0,
    normalize: bool = True,
    train_edges_only: bool = False,
) -> torch.Tensor:
    """
    Pairwise cluster consistency loss using cosine similarity.

    Positives (same cluster):
      - Sample same-cluster edges and encourage high cosine similarity:
        L_pos = mean(1 - cos)

    Negatives (different clusters):
      - Sample cross-cluster node pairs (not necessarily edges) and enforce separation:
        L_neg = mean(relu(cos - margin))

    Args:
        hidden: Node embeddings (N, D).
        data: Data object providing edge_index (and optionally train_mask).
        cluster_labels: Hard cluster assignment per node (N,).
        num_pos/num_neg: Number of positive/negative pairs to sample.
        margin: Cosine margin for negative separation.
        pos_w/neg_w: Weights for positive/negative terms.
        normalize: If True, L2-normalize hidden before cosine computation.
        train_edges_only: If True, restrict positives/negatives to training nodes.

    Returns:
        Scalar loss.
    """

    device = hidden.device
    row, col = data.edge_index
    cl = cluster_labels.to(device)

    # Positive: same-cluster edges
    if train_edges_only and hasattr(data, "train_mask"):
        edge_mask_all = (data.train_mask[row] & data.train_mask[col])
    else:
        edge_mask_all = torch.ones(row.size(0), dtype=torch.bool, device=device)

    same = (cl[row] == cl[col])
    pos_mask = same & edge_mask_all
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)

    # Negative: cross-cluster node pairs (not only edges)
    if train_edges_only and hasattr(data, "train_mask"):
        node_mask = data.train_mask.to(device)
    else:
        node_mask = torch.ones(cl.size(0), dtype=torch.bool, device=device)

    valid_nodes = torch.nonzero(node_mask, as_tuple=False).squeeze(1)
    cl_valid = cl[valid_nodes]
    uniq = torch.unique(cl_valid)
    can_sample_neg = uniq.numel() >= 2

    def _sample_pos(idx, k):
        """Sample positive edge indices with replacement if needed."""
        if k <= 0 or idx.numel() == 0:
            return idx[:0]
        if idx.numel() >= k:
            perm = torch.randperm(idx.numel(), device=device)[:k]
            return idx[perm]
        extra = idx[torch.randint(0, idx.numel(), (k,), device=device)]
        return extra

    def _sample_cross_cluster_pairs(cl_all, mask_nodes, num_pairs, max_tries=20):
        """
        Sample node pairs (i, j) such that cluster(i) != cluster(j), with replacement.
        Returns empty tensors if sampling fails.
        """
        if num_pairs <= 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        nodes = torch.nonzero(mask_nodes, as_tuple=False).squeeze(1)
        if nodes.numel() < 2:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

        i_list, j_list = [], []
        remain, tries = num_pairs, 0
        while remain > 0 and tries < max_tries:
            batch = max(remain * 2, 1024)
            i = nodes[torch.randint(0, nodes.numel(), (batch,), device=device)]
            j = nodes[torch.randint(0, nodes.numel(), (batch,), device=device)]
            valid = (cl_all[i] != cl_all[j])
            i, j = i[valid], j[valid]
            if i.numel() == 0:
                tries += 1
                continue
            take = min(remain, i.numel())
            i_list.append(i[:take])
            j_list.append(j[:take])
            remain -= take
        if remain > 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        return torch.cat(i_list, dim=0), torch.cat(j_list, dim=0)

    pos_s = _sample_pos(pos_idx, num_pos)
    if can_sample_neg:
        neg_i, neg_j = _sample_cross_cluster_pairs(cl, node_mask, num_neg)
    else:
        neg_i = neg_j = torch.empty(0, dtype=torch.long, device=device)

    # Cosine & Loss
    h = F.normalize(hidden, p=2, dim=1) if (normalize and hidden.size(1) > 0) else hidden
    loss = torch.tensor(0.0, device=device)

    if pos_s.numel() > 0:
        cos_pos = (h[row[pos_s]] * h[col[pos_s]]).sum(dim=1)
        loss = loss + pos_w * (1.0 - cos_pos).mean()
    if neg_i.numel() > 0:
        cos_neg = (h[neg_i] * h[neg_j]).sum(dim=1)
        loss = loss + neg_w * F.relu(cos_neg - margin).mean()

    return loss

def train(model, data, optimizer, edge_weight=None, cluster_labels=None, args=None,
        edge_index_override=None, cluster_probs=None):
    """
    Single training step.

    Components:
      (1) Standard CE loss on training nodes.
      (2) Optional gradient-alignment meta weights for reweighting CE.
      (3) Optional pairwise cluster loss on hidden features.
      (4) Optional KL alignment between model predictions and cluster probabilities.
    """
    
    model.train()
    optimizer.zero_grad()
    ei = edge_index_override if edge_index_override is not None else data.edge_index

    out = model(data.x, ei, edge_weight=edge_weight)

    use_meta = (args is not None) and getattr(args, "use_meta_align_weight", False)
    if use_meta:
        w_all = compute_meta_weights_by_grad_align(
            model, data, out,
            max_params=getattr(args, "meta_align_params_k", 2),
            max_train_samples=getattr(args, "meta_max_train_samples", 2048),
        ).detach()
    else:
        w_all = None

    out = model(data.x, ei, edge_weight=edge_weight)

    m = data.train_mask
    if w_all is not None:
        loss_per = F.cross_entropy(out[m], data.y[m], reduction='none')
        denom = w_all[m].sum().clamp_min(1e-12)
        loss = (loss_per * w_all[m]).sum() / denom
    else:
        loss = F.cross_entropy(out[m], data.y[m])
    
    if args is not None and getattr(args, "pair_loss_weight", 0.0) > 0 and cluster_labels is not None:
        h = get_hidden_for_pair(model, data)
        pair_loss = compute_pairwise_cluster_loss(
            hidden=h, data=data, cluster_labels=cluster_labels,
            num_pos=2048, num_neg=2048,
            margin=0.5, pos_w=1.0, neg_w=1.0,
            normalize=True, train_edges_only=False,
        )
        loss = loss + args.pair_loss_weight * pair_loss

    if (
        args is not None
        and getattr(args, "align_prob_weight", 0.0) > 0.0
        and cluster_probs is not None
    ):
        log_pred = F.log_softmax(out, dim=1)
        target_prob = cluster_probs

        if target_prob.size(1) == log_pred.size(1):
            m = data.train_mask
            align_loss = F.kl_div(log_pred[m], target_prob[m], reduction="batchmean")
            loss = loss + args.align_prob_weight * align_loss
        else:
            pass

    loss.backward()
    optimizer.step()
    return loss.item()

