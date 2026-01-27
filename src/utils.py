import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch import nn


def get_hidden(model: nn.Module, data) -> torch.Tensor:
    model.eval()
    x, ei = data.x, data.edge_index

    h = F.relu(model.lin1(x))
    return h.detach().cpu()  # (N, D)

def _hungarian_perm_from_confusion(conf_mat: np.ndarray) -> np.ndarray:
    """conf_mat[cluster, class]를 최대화하는 클러스터→클래스 순열을 반환."""
    try:
        from scipy.optimize import linear_sum_assignment
        # 최대화 -> 비용을 음수로
        row_ind, col_ind = linear_sum_assignment(-conf_mat)
        # row_ind는 [0..C-1]이 보장(모든 클러스터가 한 번씩), col_ind가 대응 클래스
        return col_ind
    except Exception:
        # scipy 없으면 간단 그리디 백업
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
    train 구간의 (클러스터, 클래스) 혼동행렬로 클러스터→클래스 매칭 순열 반환.
    반환: shape (C,), dtype long.  i번째 값은 '클러스터 i가 대응하는 클래스 인덱스'.
    """
    device = cluster_hard.device
    m = train_mask.bool().to(device)
    ch = cluster_hard[m]
    yt = y[m]

    C = int(num_classes)
    conf = torch.zeros(C, C, dtype=torch.long, device=device)
    if ch.numel() > 0:
        # 혼동행렬 집계
        for ci, yi in zip(ch.tolist(), yt.tolist()):
            if 0 <= ci < C and 0 <= yi < C:
                conf[ci, yi] += 1

    perm_np = _hungarian_perm_from_confusion(conf.detach().cpu().numpy())
    perm = torch.tensor(perm_np, dtype=torch.long, device=device)

    return perm

def _flatten_grads(grads, params):
    vecs = []
    for g, p in zip(grads, params):
        if g is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(g.reshape(-1))
    return torch.cat(vecs, dim=0)

def _pick_params_for_alignment(model, k_last_tensors: int = 2):
    # 비용을 낮추기 위해 마지막 파라미터 텐서 1~2개만 사용 (보통 최종 분류기 가중치/바이어스)
    params = [p for p in model.parameters() if p.requires_grad]
    k = min(k_last_tensors, len(params))
    return params[-k:]

def _map_scores_to_weights(scores: torch.Tensor, idxes: torch.Tensor, N: int,
                           lo: float = 0.5, hi: float = 1.0) -> torch.Tensor:
    w = torch.full((N,), lo, device=scores.device, dtype=scores.dtype)
    if idxes.numel() == 0:
        return w
    s = scores.clone()
    s = torch.relu(s)  # 음수 정렬은 0 처리
    smax = s.max()
    if float(smax) == 0.0:
        w[idxes] = lo
        return w
    s_norm = s / (smax + 1e-12)            # [0,1]
    w[idxes] = lo + (hi - lo) * s_norm     # [0.5,1.0]
    return w