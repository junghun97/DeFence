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
    """
    Ensure that `data` has 1D boolean masks: train_mask / val_mask / test_mask.

    Motivation:
        Some PyG datasets (e.g., Planetoid) already provide split masks,
        but others may not, or may provide multiple splits in a 2D tensor
        (e.g., WikiCS provides 20 splits).

    Behavior:
        - If the masks exist, normalize them into a single 1D mask.
        - If the masks are 2D, pick the first split (index 0).
        - If the masks do not exist, this function currently does nothing
          (despite the docstring suggesting creation).

    Inputs:
        data: PyG Data object (expects .num_nodes and potentially masks)
        num_classes: number of classes (not used in current implementation)
        train_ratio, val_ratio: not used in current implementation (kept for future extension)

    Output:
        data with possibly modified train/val/test masks.
    """
    if all(hasattr(data, k) for k in ["train_mask", "val_mask", "test_mask"]):
        N = data.num_nodes

        def _pick(mask):
            """
            Convert various mask formats to a single 1D boolean mask.

            Supported:
                - 1D: (N,)
                - 2D: (N, S) or (S, N) where S is the number of splits
            Strategy:
                Use the first split (split index = 0).
            """
            if mask is None:
                return None
            if mask.dim() == 1:
                return mask
            if mask.size(0) == N and mask.size(1) != N:  # (N, S)
                S = mask.size(1)
                idx = 0
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
    """
    Inject structural noise by adding random edges.

    Definition:
        Add `noise_ratio * (#existing_edges)` random edges that do not already exist.
        Then convert the resulting graph into an undirected one.

    Inputs:
        data: PyG Data object with `edge_index`
        noise_ratio: fraction of existing edges to add as random edges

    Output:
        data with updated `edge_index` (undirected) if noise_ratio > 0.
    """
    num_nodes = data.num_nodes
    num_existing_edges = data.edge_index.size(1)
    num_noisy_edges = int(noise_ratio * num_existing_edges)

    # Use a set for duplicate checking
    existing = set((int(data.edge_index[0, i]), int(data.edge_index[1, i]))
                   for i in range(num_existing_edges))

    noisy_edges = []
    # Prevent infinite loops for very large graphs
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
        # Convert to undirected for consistency
        data.edge_index = to_undirected(edge_index)

    return data


def _infer_num_classes(data, dataset):
    """
    Infer the number of classes.

    Priority:
        1) dataset.num_classes (if available and valid)
        2) max label in data.y + 1 (fallback)

    Inputs:
        data: PyG Data object with `y`
        dataset: PyG dataset object (may have num_classes)

    Output:
        int number of classes
    """
    if hasattr(dataset, "num_classes"):
        try:
            nc = int(dataset.num_classes)
            if nc > 0:
                return nc
        except Exception:
            pass
    # fallback
    return int(data.y.max().item() + 1)


def load_noisy_data(dataset_name="Cora", label_noise=0.0, edge_noise=0.0):
    """
    Load a node classification dataset and optionally inject label/edge noise.

    Current supported datasets:
        - 'Cora', 'Citeseer', 'PubMed' (Planetoid)

    Label noise injection:
        - Only applied to the TRAIN split.
        - A fraction `label_noise` of train nodes are relabeled uniformly at random
          among classes excluding the true label.

    Edge noise injection:
        - Adds random edges via `add_edge_noise`.

    Inputs:
        dataset_name: dataset identifier (string)
        label_noise: fraction of training nodes to corrupt (0.0 to 1.0)
        edge_noise: fraction of edges to add as random edges (0.0 to ...)

    Outputs:
        dataset: PyG Dataset object
        data: PyG Data object (possibly modified)
    """
    
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
    Randomly DROP nodes (not edges), then remove all incident edges.
    Reindex remaining nodes to a compact range [0, keep_N-1].

    Important:
        Despite the function name, it performs node dropping, not edge dropping.

    Steps:
        1) Determine N (#nodes).
        2) Sample `keep_N = round((1-drop_ratio)*N)` nodes to keep.
        3) Build an old->new node ID mapping.
        4) Filter edges to those whose endpoints are both kept, then reindex.
        5) Slice node-level tensors (x, y, train/val/test masks).
        6) Update data.num_nodes.

    Inputs:
        data: PyG Data object (expects edge_index; optionally x, y, masks, edge_weight/edge_attr)
        drop_ratio: fraction of nodes to drop (0.0 <= r < 1.0)
        seed: random seed (deterministic selection of kept nodes)
        min_keep: minimum number of nodes to keep

    Output:
        Modified data with fewer nodes and reindexed edges.
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
