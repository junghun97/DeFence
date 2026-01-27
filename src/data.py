import os.path as osp
import random
import re
import torch

from torch_geometric.datasets import (
    Planetoid, CitationFull, WebKB, WikipediaNetwork,
    Coauthor, Amazon, Flickr, WikiCS, FacebookPagePage,
    DeezerEurope, Actor, LastFMAsia, Twitch
)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected


def maybe_make_masks(data, num_classes: int, train_ratio=0.1, val_ratio=0.1):
    """Planetoid 등은 기본 마스크가 있지만, 없는 경우(일부 Coauthor/Amazon 등)를 대비해 생성."""
    if all(hasattr(data, k) for k in ["train_mask", "val_mask", "test_mask"]):
        N = data.num_nodes

        def _pick(mask):
            if mask is None:
                return None
            if mask.dim() == 1:
                return mask
            # 2D: (S, N) or (N, S) 모두 지원 (e.g., WikiCS 20 splits)
            if mask.size(0) == N and mask.size(1) != N:  # (N, S)
                S = mask.size(1)
                idx = 0  # 첫 split 사용
                return mask[:, idx]
            elif mask.size(1) == N:  # (S, N)
                S = mask.size(0)
                idx = 0
                return mask[idx]
            else:
                raise ValueError(f"Unexpected mask shape: {tuple(mask.size())}, N={N}")

        data.train_mask = _pick(getattr(data, "train_mask", None))
        data.val_mask   = _pick(getattr(data, "val_mask", None))
        data.test_mask  = _pick(getattr(data, "test_mask", None))
        return data


def add_edge_noise(data, noise_ratio):
    num_nodes = data.num_nodes
    num_existing_edges = data.edge_index.size(1)
    num_noisy_edges = int(noise_ratio * num_existing_edges)

    # 기존 edge set으로 중복 방지
    existing = set((int(data.edge_index[0, i]), int(data.edge_index[1, i]))
                   for i in range(num_existing_edges))

    noisy_edges = []
    # 너무 큰 그래프에서 무한루프 방지용 최대 시도 횟수
    max_trials = max(10 * num_noisy_edges, 10000)
    trials = 0
    while len(noisy_edges) < num_noisy_edges and trials < max_trials:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        trials += 1
        if u == v:
            continue
        if (u, v) in existing or (v, u) in existing:
            continue
        noisy_edges.append((u, v))
        existing.add((u, v))

    if noisy_edges:
        noisy_edges = torch.tensor(noisy_edges, dtype=torch.long).t()
        edge_index = torch.cat([data.edge_index, noisy_edges], dim=1)
        # 일관성 유지를 위해 undirected로 변환
        data.edge_index = to_undirected(edge_index)

    return data


def _infer_num_classes(data, dataset):
    if hasattr(dataset, "num_classes"):
        try:
            nc = int(dataset.num_classes)
            if nc > 0:
                return nc
        except Exception:
            pass
    # fallback
    return int(data.y.max().item() + 1)


def _parse_twitch_region(dataset_name_lower: str):
    """
    'twitch-de', 'twitch_de', 'twitch:DE', 'twitchDE' 등에서 지역 코드 추출.
    허용: DE, EN, ES, FR, PT, RU
    """
    m = re.match(r"twitch[-_: ]?([a-z]{2})$", dataset_name_lower)
    if m:
        return m.group(1).upper()
    # camelCase 'twitchDE' 형태
    m2 = re.match(r"twitch([A-Z]{2})$", dataset_name_lower, flags=0)
    if m2:
        return m2.group(1).upper()
    return None


def load_noisy_data(dataset_name="Cora", label_noise=0.0, edge_noise=0.0):
    name_raw = dataset_name
    name = dataset_name.strip().lower()
    transform = NormalizeFeatures()
    cur_path = osp.dirname(osp.abspath(__file__))

    # === Load dataset ===
    if name in ['cora', 'citeseer', 'pubmed']:
        path = osp.join(cur_path, 'data', name_raw)
        dataset = Planetoid(path, name=name_raw, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    data = dataset[0]

    # 마스크 정규화/생성 (없으면 생성)
    data = maybe_make_masks(data, num_classes=_infer_num_classes(data, dataset))

    # Inject label noise only in train set
    if label_noise > 0 and hasattr(data, 'train_mask'):
        if not hasattr(data, 'y_clean'):
            data.y_clean = data.y.clone()
        train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        num_noisy = int(label_noise * len(train_idx))
        if num_noisy > 0:
            noisy_idx = train_idx[torch.randperm(len(train_idx))[:num_noisy]]
            num_classes = _infer_num_classes(data, dataset)
            for i in noisy_idx:
                true_label = int(data.y[i])
                candidates = [c for c in range(num_classes) if c != true_label]
                data.y[i] = random.choice(candidates)

    # Inject structure noise (by adding random edges)
    if edge_noise > 0:
        data = add_edge_noise(data, edge_noise)

    return dataset, data


def drop_edge_random(data, drop_ratio: float, seed: int, min_keep: int = 1):
    """
    Randomly drop nodes by ratio and remove all incident edges.
    Reindexes node IDs and filters node/edge attributes accordingly.

    Args:
        data: PyG Data object (expects .edge_index [2, E], and optionally x, y, *_mask).
        drop_ratio: fraction of N nodes to drop (0.0 <= r < 1.0).
        seed: int random seed for reproducibility.
        min_keep: minimum number of nodes to keep.
    """

    # ---- figure out N (number of nodes) ----
    if hasattr(data, "num_nodes") and data.num_nodes is not None:
        N = int(data.num_nodes)
    elif hasattr(data, "x") and data.x is not None:
        N = int(data.x.size(0))
    elif hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
        N = int(data.edge_index.max().item()) + 1
    else:
        N = 0

    if N == 0 or drop_ratio <= 0.0:
        return data

    drop_ratio = float(max(0.0, min(drop_ratio, 0.999999)))
    keep_N = max(min_keep, int(round((1.0 - drop_ratio) * N)))
    if keep_N >= N:
        return data

    device = (
        data.edge_index.device
        if hasattr(data, "edge_index") and data.edge_index is not None
        else (data.x.device if hasattr(data, "x") and data.x is not None else "cpu")
    )

    # ---- sample nodes to KEEP ----
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    perm = torch.randperm(N, generator=g, device=device)
    keep_idx = perm[:keep_N]
    keep_idx, _ = torch.sort(keep_idx)  # stable ordering

    # ---- build old->new id mapping ----
    mapping = -torch.ones(N, dtype=torch.long, device=device)
    mapping[keep_idx] = torch.arange(keep_N, dtype=torch.long, device=device)

    # ---- reindex edges & drop incident ones ----
    if hasattr(data, "edge_index") and data.edge_index is not None:
        ei = data.edge_index
        if ei.numel() > 0:
            new_u = mapping[ei[0]]
            new_v = mapping[ei[1]]
            edge_keep = (new_u >= 0) & (new_v >= 0)
            data.edge_index = torch.stack([new_u[edge_keep], new_v[edge_keep]], dim=0)

            # edge-level attributes aligned by E
            E = ei.size(1)
            for attr in ("edge_weight", "edge_attr"):
                if hasattr(data, attr):
                    val = getattr(data, attr)
                    if isinstance(val, torch.Tensor) and val.size(0) == E:
                        setattr(data, attr, val[edge_keep])

    # ---- slice node-level tensors ----
    def _slice_node(name: str):
        if hasattr(data, name):
            t = getattr(data, name)
            if isinstance(t, torch.Tensor) and t.size(0) == N:
                setattr(data, name, t[keep_idx])

    for name in ("x", "y", "train_mask", "val_mask", "test_mask"):
        _slice_node(name)

    # ---- update num_nodes (optional; PyG can infer from x/edge_index) ----
    try:
        data.num_nodes = keep_N
    except Exception:
        pass

    return data