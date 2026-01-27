import argparse

import torch

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score as NMI,
    adjusted_rand_score as ARI,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
import numpy as np
import time

from src import *

def _now():
    # CUDA 커널 대기 없이 정확히 재려면 동기화 후 측정
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")

    # APPNP
    parser.add_argument("--appnp-k", type=int, default=10)
    parser.add_argument("--appnp-alpha", type=float, default=0.1)
    parser.add_argument("--appnp-dropout", type=float, default=0.0)

    # Clustering monitor
    parser.add_argument("--cluster-every", type=int, default=10)

    # Model/opt
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    # Iteration
    parser.add_argument("--inner-loops", type=int, default=None,
                        help="각 outer 반복에서 학습할 에폭 수 (미지정 시 epochs 사용)")

    # Noise
    parser.add_argument("--label-noise", type=float, default=0.1)
    parser.add_argument("--edge-noise", type=float, default=0.1)
    parser.add_argument("--node-drop", type=float, default=0.0)

    # Early stop
    parser.add_argument("--early-stop", type=int, default=50)

    # Pairwise cluster loss
    parser.add_argument("--pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--align-prob-weight", type=float, default=1.0)
    parser.add_argument("--use-meta-align-weight", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, data = load_noisy_data(args.dataset, args.label_noise, args.edge_noise)
    if args.node_drop > 0.0:
        before = int(data.edge_index.size(1))
        drop_edge_random(data, drop_ratio=args.node_drop, seed=args.seed, min_keep=1)
        after = int(data.edge_index.size(1))
        print(f"[edge-drop] edges: {before} -> {after} (drop={args.node_drop:.2f})")

    data = data.to(device)

    # ===== Iterative training with re-init per outer =====
    edge_weight = None
    cluster_labels_cache = None
    cluster_probs_cache = None
    MIN_EARLY_EPOCH = 100

    # (1) 새 모델/옵티마이저 선언
    model = APPNPNet(dataset.num_node_features, args.hidden, dataset.num_classes, feat_dropout=args.dropout,
             k=args.appnp_k, alpha=args.appnp_alpha, appnp_dropout=args.appnp_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # (2) 이번 iteration(outer)용 상태 초기화
    best_val_acc = 0
    best_state = None
    patience = 0

    # (3) inner loop 학습
    for inner in range(1, args.epochs + 1):
        inner += 1

        if args.cluster_every > 0 and (inner % args.cluster_every == 0 or inner == 1):
            hidden = get_hidden(model, data)  # (N, D) CPU

            # Soft KMeans
            X = hidden.numpy()
            km = KMeans(n_clusters=int(dataset.num_classes), n_init=10, random_state=0)
            km.fit(X)
            centers = torch.from_numpy(km.cluster_centers_).to(hidden)
            with torch.no_grad():
                H = torch.from_numpy(X).to(dtype=centers.dtype)
                diff = H.unsqueeze(1) - centers.unsqueeze(0)
                logits = -(diff * diff).sum(dim=2)
                probs_all = torch.softmax(logits, dim=1)

            cluster_hard = probs_all.argmax(dim=1)  # (N,)
            perm = compute_cluster_to_class_perm_by_train(
                cluster_hard=cluster_hard,
                y=data.y.to(cluster_hard.device),
                train_mask=data.train_mask.to(cluster_hard.device),
                num_classes=int(dataset.num_classes),
            )

            probs_all_aligned = probs_all[:, perm]
            cluster_labels_cache = cluster_hard.to(data.x.device)  # 하드 라벨은 그대로
            cluster_probs_cache = probs_all_aligned.to(data.x.device)


                # ---- Train/Eval ----
        loss_tr = train(model, data, optimizer,
                            edge_weight=edge_weight,
                            cluster_labels=cluster_labels_cache,
                            args=args,
                            edge_index_override=data.edge_index,
                            cluster_probs=cluster_probs_cache)

        val_acc, val_loss = evaluate(model, data, split="val",
                                     edge_weight=edge_weight,
                                     edge_index_override=data.edge_index)
        tst_acc, _ = evaluate(model, data, split="test",
                              edge_weight=edge_weight,
                              edge_index_override=data.edge_index)

        # Early stopping (이번 outer에만 적용)
        if args.early_stop > 0:
            improved = val_acc > best_val_acc - 1e-6
            if improved:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if (patience >= args.early_stop) and (inner >= MIN_EARLY_EPOCH):
                print(f"[EarlyStop] epoch={inner}, best_val_acc={best_val_acc:.2f} -> break inner")
                break  # inner만 종료

        if inner % 10 == 0 or inner == 1:
            print(f"[{inner:03d}] train_loss={loss_tr:.4f} | val_acc={val_acc*100:.2f}% | test_acc={tst_acc*100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)

    # 전체 종료 후 최종 성능 (마지막 outer의 active_edge_index = 마지막 학습에 사용된 그래프)
    tr_acc, _ = evaluate(model, data, split="train", edge_index_override=data.edge_index)
    val_acc, _ = evaluate(model, data, split="val", edge_index_override=data.edge_index)
    tst_acc, _ = evaluate(model, data, split="test", edge_index_override=data.edge_index)
    print(f"Final Acc | train={tr_acc*100:.2f}% | val={val_acc*100:.2f}% | test={tst_acc*100:.2f}%")

if __name__ == "__main__":
    main()
