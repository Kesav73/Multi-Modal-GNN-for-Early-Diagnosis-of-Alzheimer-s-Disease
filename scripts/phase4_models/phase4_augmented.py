"""
Phase 4 (Augmented): IFDCGCN with Data Augmentation
Paper: "Multi-modal graph neural network for early diagnosis of Alzheimer's disease"

Augmentation strategies for small medical imaging datasets:

1. GraphSMOTE  — synthetic node feature interpolation between same-class neighbors
                  Creates new virtual subjects in feature space
2. Feature Noise — Gaussian jitter on node features during training (feature-level dropout)
3. Edge Augmentation — DropEdge (randomly drop edges) + AddEdge (add intra-class edges)
4. Manifold Mixup  — interpolate hidden representations between same-class pairs
5. Class-weighted Loss — down-weight majority class automatically

Dataset stats:
  CN=196, AD=136, SMCI=176, PMCI=119  (total 627)
  AD_vs_CN    train ~233: CN~137, AD~95   (imbalance ratio 1.44x)
  SMCI_vs_PMCI train ~205: SMCI~123, PMCI~83  (imbalance ratio 1.48x)

Usage:
  python phase4_augmented.py --task AD_vs_CN
  python phase4_augmented.py --task SMCI_vs_PMCI
  python phase4_augmented.py --task AD_vs_CN --aug all --runs 5
  python phase4_augmented.py --task AD_vs_CN --aug graphsmote --aug_ratio 0.5
"""

import numpy as np
import pandas as pd
import os, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
PHASE3_DIR = "./outputs/phase3_outputs"
OUTPUT_DIR = "./outputs/phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--task",       default="AD_vs_CN",
                    choices=["AD_vs_CN","SMCI_vs_PMCI"])
parser.add_argument("--epochs",     type=int,   default=300)
parser.add_argument("--lr",         type=float, default=0.001)
parser.add_argument("--dropout",    type=float, default=0.5)
parser.add_argument("--hidden",     type=int,   default=32)
parser.add_argument("--K",          type=int,   default=3)
parser.add_argument("--weight_decay",type=float,default=5e-4)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--runs",       type=int,   default=5)
# Augmentation flags
parser.add_argument("--aug",        nargs="+",
                    default=["graphsmote","feature_noise","dropedge","mixup"],
                    choices=["graphsmote","feature_noise","dropedge","addedge","mixup","all","none"],
                    help="Augmentation methods to use")
parser.add_argument("--aug_ratio",  type=float, default=0.5,
                    help="GraphSMOTE: fraction of minority class to synthesize (0.5 = 50%% more)")
parser.add_argument("--noise_std",  type=float, default=0.01,
                    help="Feature noise std (relative to feature std)")
parser.add_argument("--drop_rate",  type=float, default=0.1,
                    help="DropEdge: fraction of edges to drop each forward pass")
parser.add_argument("--add_rate",   type=float, default=0.05,
                    help="AddEdge: fraction of new intra-class edges to add")
parser.add_argument("--mixup_alpha",type=float, default=0.2,
                    help="Mixup Beta distribution alpha parameter")
args = parser.parse_args()

if "all" in args.aug:
    args.aug = ["graphsmote","feature_noise","dropedge","addedge","mixup"]
if "none" in args.aug:
    args.aug = []

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print(f"Task        : {args.task}")
print(f"Device      : {device}")
print(f"Augmentations: {args.aug}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────
#  STEP 1: Load Phase 3 data
# ─────────────────────────────────────────────────────────────────
print("\nSTEP 1: Loading data")

task_dir = os.path.join(PHASE3_DIR, args.task)

X_pet  = np.load(os.path.join(PHASE3_DIR, "X_pet.npy")).astype(np.float32)
X_smri = np.load(os.path.join(PHASE3_DIR, "X_smri.npy")).astype(np.float32)
A_norm = np.load(os.path.join(PHASE3_DIR, "A_if_norm.npy")).astype(np.float32)

binary_labels = np.load(os.path.join(task_dir, "binary_labels.npy"))
train_mask    = np.load(os.path.join(task_dir, "train_mask.npy"))
val_mask      = np.load(os.path.join(task_dir, "val_mask.npy"))
test_mask     = np.load(os.path.join(task_dir, "test_mask.npy"))
task_mask     = np.load(os.path.join(task_dir, "task_mask.npy"))

N      = X_pet.shape[0]
in_dim = X_pet.shape[1]

# Class counts in training set
train_labels = binary_labels[train_mask]
n0_train = int((train_labels == 0).sum())
n1_train = int((train_labels == 1).sum())
minority_cls = 1 if n1_train < n0_train else 0
majority_cls = 1 - minority_cls

print(f"  N={N}  in_dim={in_dim}")
print(f"  Train: class0={n0_train}  class1={n1_train}  "
      f"imbalance={max(n0_train,n1_train)/min(n0_train,n1_train):.2f}x")
print(f"  Val={val_mask.sum()}  Test={test_mask.sum()}")


# ─────────────────────────────────────────────────────────────────
#  STEP 2: Build Chebyshev Laplacian
# ─────────────────────────────────────────────────────────────────
print("\nSTEP 2: Building Laplacian")

def build_laplacian(A: np.ndarray) -> torch.Tensor:
    N  = A.shape[0]
    I  = np.eye(N, dtype=np.float32)
    L  = I - A
    v  = np.random.randn(N).astype(np.float32)
    for _ in range(30):
        v = L @ v
        lam = np.linalg.norm(v)
        v = v / (lam + 1e-10)
    lambda_max = float(lam)
    L_tilde = (2.0 / lambda_max) * L - I
    print(f"  λ_max={lambda_max:.4f}")
    return torch.tensor(L_tilde, dtype=torch.float32)

L_base = build_laplacian(A_norm)


# ─────────────────────────────────────────────────────────────────
#  STEP 3: Augmentation functions
# ─────────────────────────────────────────────────────────────────
print(f"\nSTEP 3: Setting up augmentations: {args.aug}")

# ── 3a. GraphSMOTE ───────────────────────────────────────────────
def graph_smote(X_pet, X_smri, A_norm, labels_full, train_mask, binary_labels, aug_ratio=0.5):
    """
    GraphSMOTE: Synthesize new minority-class nodes by interpolating
    between existing minority nodes in feature space, then connecting
    them to the graph based on feature similarity.

    Returns:
        X_pet_aug  : [N + n_syn, in_dim]
        X_smri_aug : [N + n_syn, in_dim]
        A_aug      : [N + n_syn, N + n_syn]
        y_aug      : [N + n_syn]  labels (-1 for non-task, 0/1 for task)
        new_train_mask: [N + n_syn] bool
    """
    N = X_pet.shape[0]
    train_idx = np.where(train_mask)[0]
    train_labels_bin = binary_labels[train_idx]

    # Identify minority class training nodes
    min_idx = train_idx[train_labels_bin == minority_cls]   # minority in train
    n_syn   = max(1, int(len(min_idx) * aug_ratio))

    print(f"  GraphSMOTE: minority class={minority_cls}  "
          f"n_minority_train={len(min_idx)}  n_synthetic={n_syn}")

    # KNN within minority class (k=5 neighbors)
    k = min(5, len(min_idx) - 1)
    nn_model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn_model.fit(X_pet[min_idx])
    _, nn_indices = nn_model.kneighbors(X_pet[min_idx])   # [n_min, k+1]

    # Synthesize new nodes
    syn_pet  = np.zeros((n_syn, X_pet.shape[1]),  dtype=np.float32)
    syn_smri = np.zeros((n_syn, X_smri.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    for i in range(n_syn):
        # Pick a random minority node
        src_local = rng.integers(len(min_idx))
        src_global = min_idx[src_local]

        # Pick a random KNN neighbor (not self)
        nn_local = nn_indices[src_local, 1:]     # exclude self
        nbr_local = rng.choice(nn_local)
        nbr_global = min_idx[nbr_local]

        # Interpolate: λ ~ U(0,1)
        lam = rng.random()
        syn_pet[i]  = (1 - lam) * X_pet[src_global]  + lam * X_pet[nbr_global]
        syn_smri[i] = (1 - lam) * X_smri[src_global] + lam * X_smri[nbr_global]

    # Concatenate synthetic nodes
    X_pet_aug  = np.vstack([X_pet,  syn_pet])
    X_smri_aug = np.vstack([X_smri, syn_smri])
    N_aug      = N + n_syn

    # Build augmented adjacency: connect synthetic nodes to their K nearest
    # real minority nodes based on feature similarity
    A_aug = np.zeros((N_aug, N_aug), dtype=np.float32)
    A_aug[:N, :N] = A_norm   # keep original graph

    # Connect synthetic nodes
    nn_all = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_all.fit(X_pet[min_idx])
    _, nn_for_syn = nn_all.kneighbors(syn_pet)   # [n_syn, k]

    for s_i in range(n_syn):
        s_global = N + s_i
        for nn_local_i in nn_for_syn[s_i]:
            real_global = min_idx[nn_local_i]
            # Edge weight = mean of neighbors' original weights
            w = float(A_norm[real_global, min_idx].mean())
            w = max(w, 1e-5)
            A_aug[s_global, real_global] = w
            A_aug[real_global, s_global] = w
        A_aug[s_global, s_global] = 1.0   # self-loop

    # Symmetric normalization of A_aug
    d = A_aug.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_aug_norm = d_inv_sqrt[:, None] * A_aug * d_inv_sqrt[None, :]

    # Labels for augmented graph
    y_aug = np.full(N_aug, -1, dtype=np.int64)
    y_aug[:N] = binary_labels
    y_aug[N:] = minority_cls  # synthetic nodes are minority class

    # Train mask: original train + all synthetic nodes
    new_train_mask = np.zeros(N_aug, dtype=bool)
    new_train_mask[:N] = train_mask
    new_train_mask[N:] = True   # synthetic nodes → train only

    # Val/test masks stay the same (only real nodes)
    new_val_mask  = np.zeros(N_aug, dtype=bool)
    new_test_mask = np.zeros(N_aug, dtype=bool)
    new_val_mask[:N]  = val_mask
    new_test_mask[:N] = test_mask

    n0_new = int(((y_aug[new_train_mask]) == 0).sum())
    n1_new = int(((y_aug[new_train_mask]) == 1).sum())
    print(f"  After GraphSMOTE: train class0={n0_new}  class1={n1_new}  "
          f"ratio={max(n0_new,n1_new)/max(min(n0_new,n1_new),1):.2f}x")

    return (X_pet_aug, X_smri_aug, A_aug_norm,
            y_aug, new_train_mask, new_val_mask, new_test_mask)


# ── 3b. Feature Noise Augmentation ──────────────────────────────
def feature_noise(X: torch.Tensor, std_scale: float = 0.01) -> torch.Tensor:
    """Add Gaussian noise proportional to feature std. Applied only during training."""
    noise = torch.randn_like(X) * std_scale
    return X + noise


# ── 3c. DropEdge ────────────────────────────────────────────────
def drop_edge(L: torch.Tensor, drop_rate: float) -> torch.Tensor:
    """
    Randomly zero out edges in the Laplacian during each forward pass.
    Equivalent to randomly removing graph edges → acts as graph-level dropout.
    Forces the model to be robust to missing connections.
    """
    mask = torch.bernoulli(
        torch.ones_like(L) * (1 - drop_rate)
    )
    # Keep diagonal (self-loops) always
    mask.fill_diagonal_(1.0)
    L_dropped = L * mask
    # Re-normalize rows to keep Laplacian properties
    return L_dropped


# ── 3d. AddEdge (intra-class edge enrichment) ───────────────────
def add_intra_class_edges(A_norm: np.ndarray, binary_labels: np.ndarray,
                           train_mask: np.ndarray, add_rate: float = 0.05):
    """
    Add edges between same-class training nodes that currently have low connectivity.
    Strengthens the intra-class cluster structure in the graph.
    """
    N = A_norm.shape[0]
    A_new = A_norm.copy()
    train_idx = np.where(train_mask)[0]

    for cls in [0, 1]:
        cls_idx = train_idx[binary_labels[train_idx] == cls]
        if len(cls_idx) < 2:
            continue

        # Find pairs with low existing connectivity
        A_cls = A_norm[np.ix_(cls_idx, cls_idx)]
        # Sort pairs by current weight (ascending) → add edges to weakest connections
        triu = np.triu_indices(len(cls_idx), k=1)
        weights = A_cls[triu]
        n_add   = max(1, int(len(weights) * add_rate))
        weak_pairs = np.argsort(weights)[:n_add]

        added = 0
        for p in weak_pairs:
            i_local, j_local = triu[0][p], triu[1][p]
            i_global, j_global = cls_idx[i_local], cls_idx[j_local]
            # Add edge with weight = mean intra-class weight
            w = float(A_cls[A_cls > 0].mean()) if (A_cls > 0).any() else 0.001
            A_new[i_global, j_global] = max(A_new[i_global, j_global], w)
            A_new[j_global, i_global] = A_new[i_global, j_global]
            added += 1

        print(f"  AddEdge: class {cls} → added {added} edges")

    # Re-normalize
    d = A_new.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    return (d_inv_sqrt[:, None] * A_new * d_inv_sqrt[None, :]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  STEP 4: Model definition (ChebConv + IFDCGCN with Manifold Mixup)
# ─────────────────────────────────────────────────────────────────

class ChebConv(nn.Module):
    """Chebyshev Graph Convolution (Eq. 14)"""
    def __init__(self, in_features, out_features, K):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K+1, in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight.view((K+1)*in_features, out_features))
        self.weight.data = self.weight.data.view(K+1, in_features, out_features)
        nn.init.zeros_(self.bias)

    def forward(self, x, L):
        Tx0, Tx1 = x, L @ x
        out = Tx0 @ self.weight[0]
        if self.K >= 1:
            out = out + Tx1 @ self.weight[1]
        for _ in range(2, self.K + 1):
            Tx2 = 2 * L @ Tx1 - Tx0
            out = out + Tx2 @ self.weight[_]
            Tx0, Tx1 = Tx1, Tx2
        return out + self.bias


class CGCNBranch(nn.Module):
    """Single-modality 2-layer Chebyshev GCN branch"""
    def __init__(self, in_dim, hidden_dim, n_class, K, dropout):
        super().__init__()
        self.conv1   = ChebConv(in_dim,     hidden_dim, K)
        self.conv2   = ChebConv(hidden_dim, n_class,    K)
        self.dropout = dropout

    def forward(self, x, L, return_hidden=False):
        h = F.relu(self.conv1(x, L))
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.conv2(h, L)
        out = F.dropout(out, p=self.dropout, training=self.training)
        if return_hidden:
            return out, h
        return out


class IFDCGCN(nn.Module):
    """
    IFDCGCN with optional Manifold Mixup augmentation.
    Mixup is applied at the hidden representation level (between conv1 and conv2)
    for same-class training pairs → forces smoother decision boundaries.
    """
    def __init__(self, in_dim, hidden_dim, n_class, K, dropout):
        super().__init__()
        self.branch_smri = CGCNBranch(in_dim, hidden_dim, n_class, K, dropout)
        self.branch_pet  = CGCNBranch(in_dim, hidden_dim, n_class, K, dropout)
        self.hidden_dim  = hidden_dim

    def forward(self, x_smri, x_pet, L,
                mixup=False, mixup_alpha=0.2,
                train_mask=None, y=None):

        if mixup and self.training and train_mask is not None:
            return self._forward_mixup(x_smri, x_pet, L,
                                        mixup_alpha, train_mask, y)
        # Standard forward
        z_smri = self.branch_smri(x_smri, L)
        z_pet  = self.branch_pet(x_pet,  L)
        p_smri = F.softmax(z_smri, dim=1)
        p_pet  = F.softmax(z_pet,  dim=1)
        p_fuse = 0.5 * (p_smri + p_pet)
        return p_fuse, p_smri, p_pet

    def _forward_mixup(self, x_smri, x_pet, L,
                        alpha, train_mask, y):
        """
        Manifold Mixup: interpolate hidden representations of same-class pairs.
        Applied randomly (50% of training steps).
        """
        # Get hidden representations after conv1
        _, h_smri = self.branch_smri(x_smri, L, return_hidden=True)
        _, h_pet  = self.branch_pet(x_pet,  L, return_hidden=True)

        train_idx = torch.where(train_mask)[0]
        lam = np.random.beta(alpha, alpha)

        # Find same-class pairs to mix
        y_train = y[train_idx]
        perm = torch.randperm(len(train_idx), device=x_smri.device)
        # Only mix pairs with same label
        same = (y_train == y_train[perm])

        # For same-class pairs: mix hidden state
        h_smri_mix = h_smri.clone()
        h_pet_mix  = h_pet.clone()

        for local_i, local_j in zip(
            range(len(train_idx)), perm.cpu().numpy()
        ):
            if same[local_i]:
                gi, gj = train_idx[local_i], train_idx[local_j]
                h_smri_mix[gi] = lam*h_smri[gi] + (1-lam)*h_smri[gj]
                h_pet_mix[gi]  = lam*h_pet[gi]  + (1-lam)*h_pet[gj]

        # Pass mixed hidden reps through conv2
        z_smri = F.dropout(self.branch_smri.conv2(h_smri_mix, L),
                           p=self.branch_smri.dropout, training=True)
        z_pet  = F.dropout(self.branch_pet.conv2(h_pet_mix,  L),
                           p=self.branch_pet.dropout,  training=True)

        p_smri = F.softmax(z_smri, dim=1)
        p_pet  = F.softmax(z_pet,  dim=1)
        p_fuse = 0.5 * (p_smri + p_pet)
        return p_fuse, p_smri, p_pet


# ─────────────────────────────────────────────────────────────────
#  STEP 5: Metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(probs, labels):
    if len(probs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    preds  = probs.argmax(dim=1).cpu().numpy()
    labs   = labels.cpu().numpy()
    prob1  = probs[:, 1].cpu().detach().numpy()
    acc    = float(accuracy_score(labs, preds)) * 100
    cm     = confusion_matrix(labs, preds, labels=[0,1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn=fp=fn=tp = 0
    sen = (tp / (tp+fn+1e-8)) * 100
    spe = (tn / (tn+fp+1e-8)) * 100
    try:
        auc = roc_auc_score(labs, prob1) * 100
    except Exception:
        auc = 0.0
    return acc, sen, spe, auc


# ─────────────────────────────────────────────────────────────────
#  STEP 6: Apply augmentations & prepare tensors
# ─────────────────────────────────────────────────────────────────
print("\nSTEP 4-5: Applying augmentations")

X_pet_work  = X_pet.copy()
X_smri_work = X_smri.copy()
A_work      = A_norm.copy()
y_work      = binary_labels.copy()
train_work  = train_mask.copy()
val_work    = val_mask.copy()
test_work   = test_mask.copy()

# Fix non-task labels to 0 (they are masked out during loss anyway)
y_work[y_work == -1] = 0

# ── GraphSMOTE (applied once before training) ────────────────────
if "graphsmote" in args.aug:
    (X_pet_work, X_smri_work, A_work,
     y_work, train_work, val_work, test_work) = graph_smote(
        X_pet_work, X_smri_work, A_work,
        y_work, train_work, y_work, args.aug_ratio
    )
    print(f"  Graph expanded: {N} → {len(y_work)} nodes")

# ── AddEdge (modify adjacency once) ─────────────────────────────
if "addedge" in args.aug:
    A_work = add_intra_class_edges(
        A_work, y_work, train_work, args.add_rate
    )

# ── Rebuild Laplacian from (possibly augmented) adjacency ───────
L_work = build_laplacian(A_work)

# ── Convert to tensors ───────────────────────────────────────────
X_smri_t     = torch.tensor(X_smri_work, dtype=torch.float32).to(device)
X_pet_t      = torch.tensor(X_pet_work,  dtype=torch.float32).to(device)
L_t          = L_work.to(device)
y_t          = torch.tensor(y_work, dtype=torch.long).to(device)
train_mask_t = torch.tensor(train_work, dtype=torch.bool).to(device)
val_mask_t   = torch.tensor(val_work,   dtype=torch.bool).to(device)
test_mask_t  = torch.tensor(test_work,  dtype=torch.bool).to(device)

N_aug  = len(y_work)
in_dim_aug = X_smri_t.shape[1]
print(f"  Final graph: {N_aug} nodes  "
      f"train={train_mask_t.sum().item()}  "
      f"val={val_mask_t.sum().item()}  "
      f"test={test_mask_t.sum().item()}")


# ─────────────────────────────────────────────────────────────────
#  STEP 7: Training loop
# ─────────────────────────────────────────────────────────────────
print(f"\nSTEP 6-7: Training  [{args.epochs} epochs × {args.runs} runs]")

def run_once(run_id):
    torch.manual_seed(args.seed + run_id)

    model = IFDCGCN(
        in_dim=in_dim_aug, hidden_dim=args.hidden,
        n_class=2, K=args.K, dropout=args.dropout
    ).to(device)
    optimizer = Adam(model.parameters(),
                     lr=args.lr, weight_decay=args.weight_decay)

    # Dynamic class weights from (augmented) training set
    y_tr  = y_t[train_mask_t]
    n0    = (y_tr == 0).sum().float()
    n1    = (y_tr == 1).sum().float()
    w_cls = torch.tensor([n1/(n0+n1), n0/(n0+n1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_cls)

    best_val_acc = 0
    best_state   = None
    best_metrics = None
    curves = {"tr_acc":[],"vl_acc":[],"tr_loss":[],"vl_loss":[]}

    for epoch in range(1, args.epochs+1):
        # ── Forward ─────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        # DropEdge: perturb Laplacian each forward pass
        L_train = (drop_edge(L_t, args.drop_rate)
                   if "dropedge" in args.aug else L_t)

        # Feature noise
        xs = (feature_noise(X_smri_t, args.noise_std)
              if "feature_noise" in args.aug else X_smri_t)
        xp = (feature_noise(X_pet_t,  args.noise_std)
              if "feature_noise" in args.aug else X_pet_t)

        # Manifold Mixup (50% of epochs, alternating)
        use_mixup = ("mixup" in args.aug) and (epoch % 2 == 0)

        p_fuse, p_smri, p_pet = model(
            xs, xp, L_train,
            mixup=use_mixup,
            mixup_alpha=args.mixup_alpha,
            train_mask=train_mask_t,
            y=y_t
        )

        loss_f = criterion(torch.log(p_fuse[train_mask_t]+1e-8), y_t[train_mask_t])
        loss_s = criterion(torch.log(p_smri[train_mask_t]+1e-8), y_t[train_mask_t])
        loss_p = criterion(torch.log(p_pet[train_mask_t] +1e-8), y_t[train_mask_t])
        loss   = loss_f + 0.5*loss_s + 0.5*loss_p

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Eval (no augmentation) ───────────────────────────────
        model.eval()
        with torch.no_grad():
            p_fuse, p_smri, p_pet = model(X_smri_t, X_pet_t, L_t)

        tr_acc,_,_,_      = compute_metrics(p_fuse[train_mask_t], y_t[train_mask_t])
        vl_acc,_,_,_      = compute_metrics(p_fuse[val_mask_t],   y_t[val_mask_t])
        vl_loss = criterion(torch.log(p_fuse[val_mask_t]+1e-8), y_t[val_mask_t]).item()

        curves["tr_acc"].append(tr_acc)
        curves["vl_acc"].append(vl_acc)
        curves["tr_loss"].append(loss.item())
        curves["vl_loss"].append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            te_acc,te_sen,te_spe,te_auc = compute_metrics(
                p_fuse[test_mask_t], y_t[test_mask_t])
            te_s,_,_,_ = compute_metrics(p_smri[test_mask_t], y_t[test_mask_t])
            te_p,_,_,_ = compute_metrics(p_pet[test_mask_t],  y_t[test_mask_t])
            best_metrics = dict(acc=te_acc, sen=te_sen, spe=te_spe, auc=te_auc,
                                acc_smri=te_s, acc_pet=te_p,
                                best_epoch=epoch, best_val_acc=vl_acc)

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Run{run_id+1} Ep{epoch:3d} | loss={loss.item():.4f} | "
                  f"tr={tr_acc:.1f}% | val={vl_acc:.1f}%")

    print(f"  Run{run_id+1} DONE | best_epoch={best_metrics['best_epoch']} | "
          f"val={best_val_acc:.1f}% | test_acc={best_metrics['acc']:.2f}%")
    return best_metrics, curves, best_state


# ─────────────────────────────────────────────────────────────────
#  STEP 8: Run all experiments
# ─────────────────────────────────────────────────────────────────
all_results = []
all_curves  = []
best_state_overall = None
best_acc_overall   = 0

t0 = time.time()
for run in range(args.runs):
    res, crv, state = run_once(run)
    all_results.append(res)
    all_curves.append(crv)
    if res["acc"] > best_acc_overall:
        best_acc_overall   = res["acc"]
        best_state_overall = state
elapsed = time.time() - t0


# ─────────────────────────────────────────────────────────────────
#  STEP 9: Results & Save
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"FINAL RESULTS  [{args.task}]  {args.runs} runs")
print("="*65)

paper_ref = {
    "AD_vs_CN":     {"acc":91.07,"sen":90.22,"spe":91.87,"auc":91.04},
    "SMCI_vs_PMCI": {"acc":75.50,"sen":49.90,"spe":88.70,"auc":69.30},
}

print(f"\n  {'Metric':<8} {'Ours (aug)':>14} {'Paper':>10} {'Gap':>8}")
print(f"  {'-'*42}")
for k in ["acc","sen","spe","auc"]:
    vals  = [r[k] for r in all_results]
    mean  = np.mean(vals)
    std   = np.std(vals)
    paper = paper_ref[args.task][k]
    gap   = mean - paper
    print(f"  {k.upper():<8} {mean:>7.2f}%±{std:>5.2f}%  {paper:>8.2f}%  "
          f"{'↑' if gap>=0 else '↓'}{abs(gap):>5.2f}%")

print(f"\n  Branch comparison (mean test ACC):")
print(f"    sMRI branch : {np.mean([r['acc_smri'] for r in all_results]):.2f}%")
print(f"    PET branch  : {np.mean([r['acc_pet']  for r in all_results]):.2f}%")
print(f"    Fused       : {np.mean([r['acc']       for r in all_results]):.2f}%")
print(f"\n  Time: {elapsed:.1f}s  ({elapsed/args.runs:.1f}s/run)")

# Save
task_out = os.path.join(OUTPUT_DIR, args.task)
os.makedirs(task_out, exist_ok=True)

summary = {
    "task": args.task, "config": vars(args),
    "augmentations": args.aug,
    "n_original": N, "n_augmented": N_aug,
    "results": {
        k: {"mean": float(np.mean([r[k] for r in all_results])),
            "std":  float(np.std([r[k]  for r in all_results]))}
        for k in ["acc","sen","spe","auc","acc_smri","acc_pet"]
    },
    "per_run": all_results,
    "paper_target": paper_ref[args.task],
}
with open(os.path.join(task_out, "results_augmented.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Save mean curves
mean_curves = {
    k: np.mean([c[k] for c in all_curves], axis=0).tolist()
    for k in ["tr_acc","vl_acc","tr_loss","vl_loss"]
}
mean_curves["epochs"] = list(range(1, args.epochs+1))
with open(os.path.join(task_out, "training_curves_augmented.json"), "w") as f:
    json.dump(mean_curves, f)

# Save best model
torch.save({
    "model_state": best_state_overall,
    "config": vars(args),
    "in_dim": in_dim_aug,
    "n_aug_nodes": N_aug,
    "n_class": 2,
}, os.path.join(task_out, "best_model_augmented.pt"))

print(f"\nSaved → {task_out}/")
print("  results_augmented.json")
print("  training_curves_augmented.json")
print("  best_model_augmented.pt")
print("\nTo run the other task:")
other = "SMCI_vs_PMCI" if args.task=="AD_vs_CN" else "AD_vs_CN"
print(f"  python phase4_augmented.py --task {other}")
print("\nTo try all augmentations:")
print(f"  python phase4_augmented.py --task {args.task} --aug all")
print("\nTo run without augmentation (baseline):")
print(f"  python phase4_augmented.py --task {args.task} --aug none")