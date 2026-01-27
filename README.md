# DeFence: Decoupled Feature Anchors for Robust Node Classification under Joint Label-Structure Noise

This is the official implementation of **DeFence** (*DEcoupled FEature aNChors for robust nodE classification*), submitted to KDD 2026.

---
## Abstract
How can we accurately classify nodes in graphs when both labels and edges are corrupted?
Such graphs are common in practice---fraud detection, citation, and bio-networks often exhibit both flipped node labels and spurious links.
For example, in fraud detection, delayed adjudication yields label noise while collusive rings inject spurious transaction edges.
However, most prior works tackle only one noise source, over-trusting structure for label noise and labels for edge noise; when both occur, these assumptions amplify each other's errors.
In this paper, we propose DeFence, a feature-anchored node classification model under joint label-structure noise.
DeFence first computes feature-only (pre-propagation) embeddings and clusters them to obtain soft anchors, explicitly decoupling their construction from message passing and supervision so that they remain in feature space and independent of noisy edges and labels.
During training, we i) apply an anchor-filtered contrastive loss, using only anchor-consistent structural signal and thereby countering spurious links,
ii) treat anchors as a soft prior to stabilize the posterior during propagation under joint noise,
and iii) use gradient-matched supervision that keeps updates aligned with a clean reference signal to suppress mislabeled instances.
Across standard benchmarks, DeFence consistently improves node classification accuracy over strong robust baselines.

---

## 📦 Requirements

We recommend using the following versions:

```bash
python==3.9
torch==2.7.0+cu118
torch-geometric==2.6.1
torchvision==0.22.0+cu118
torchaudio==2.7.0+cu118
scikit-learn==1.6.1
scipy==1.13.1
pandas==2.2.3
tqdm==4.67.1
```

---

## 📁 Code Structure

* `main.py`: Entry point for training and evaluation.
* `src/models.py`: Model definitions.
* `src/train.py`: Training/evaluation loops and losses supports instance-reweighting via `--use-meta-align-weight`.
* `src/data.py`: Dataset loaders; mask normalization; noise utilities `add_edge_noise`, `drop_edge_random`, and optional split helpers.
* `src/utils.py`: Utilities for seeding, metrics, hidden embedding extraction, and class–cluster alignment.

---

## 🧪 How to Run a Demo

You can run a demo on Cora dataset with the following command:

```bash
python main.py --dataset Cora --use-meta-align-weight
```

### Main Arguments

| Argument                  | Description                                        |
| ------------------------- |----------------------------------------------------|
| `--dataset`               | Dataset name (e.g., `Cora`, `CiteSeer`, `PubMed`.) |
| `--seed`                  | Random seed.                                       |
| `--label-noise`           | Fraction of training labels to randomly flip.      |
| `--edge-noise`            | Fraction of edges to add as random spurious edges  |
| `--pair-loss-weight`      | Weight of the pairwise/consistency loss.           |
| `--align-prob-weight`     | Weight of the prediction–anchor alignment loss.    |
| `--use-meta-align-weight` | Enable meta-reweighted alignment (boolean flag).   |
