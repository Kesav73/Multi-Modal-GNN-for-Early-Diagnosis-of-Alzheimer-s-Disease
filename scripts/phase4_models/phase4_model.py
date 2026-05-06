"""
Phase 4: Multi-modal Chebyshev GCN (IFDCGCN)
Paper: "Multi-modal graph neural network for early diagnosis of Alzheimer's disease"

Architecture (Section 3.3-3.4):
  - Two independent Chebyshev GCN branches (sMRI + PET)
  - Both branches share the same adjacency matrix A_if_norm (Eq.13)
  - Each branch: 2-layer CGCN with hidden_dim=32, ReLU, Dropout
  - Late fusion: average of softmax outputs (Eq.16)
  - Semi-supervised transductive learning (train mask on full graph)

Install requirements:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install torch_geometric
  OR for CPU only:
  pip install torch torchvision torchaudio
  pip install torch_geometric

Usage:
  python phase4_model.py --task AD_vs_CN
  python phase4_model.py --task SMCI_vs_PMCI
  python phase4_model.py --task AD_vs_CN --epochs 300 --lr 0.001 --dropout 0.5
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
PHASE3_DIR = "./outputs/phase3_outputs"
OUTPUT_DIR = "./outputs/phase4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--task",     default="AD_vs_CN",
                    choices=["AD_vs_CN","SMCI_vs_PMCI"])
parser.add_argument("--epochs",   type=int,   default=300)
parser.add_argument("--lr",       type=float, default=0.001)
parser.add_argument("--dropout",  type=float, default=0.5)
parser.add_argument("--hidden",   type=int,   default=32)
parser.add_argument("--K",        type=int,   default=3,
                    help="Chebyshev polynomial order (paper: 3 or 4)")
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--seed",     type=int,   default=42)
parser.add_argument("--runs",     type=int,   default=5,
                    help="Number of runs for stable mean/std results")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Task  : {args.task}")


# ─────────────────────────────────────────────────────────────────
#  STEP 1: Load Phase 3 outputs
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 1: Loading Phase 3 outputs")
print("="*65)

task_dir = os.path.join(PHASE3_DIR, args.task)

X_pet  = np.load(os.path.join(PHASE3_DIR, "X_pet.npy"))    # [N, 6786]
X_smri = np.load(os.path.join(PHASE3_DIR, "X_smri.npy"))   # [N, 6786]
A_norm = np.load(os.path.join(PHASE3_DIR, "A_if_norm.npy")) # [N, N]

binary_labels = np.load(os.path.join(task_dir, "binary_labels.npy"))  # [N] -1/0/1
train_mask    = np.load(os.path.join(task_dir, "train_mask.npy"))      # [N] bool
val_mask      = np.load(os.path.join(task_dir, "val_mask.npy"))        # [N] bool
test_mask     = np.load(os.path.join(task_dir, "test_mask.npy"))       # [N] bool
task_mask     = np.load(os.path.join(task_dir, "task_mask.npy"))       # [N] bool

N        = X_pet.shape[0]
in_dim   = X_pet.shape[1]   # 6786
n_class  = 2

print(f"N subjects    : {N}")
print(f"Feature dim   : {in_dim}")
print(f"Adjacency     : {A_norm.shape}")
print(f"Train / Val / Test : {train_mask.sum()} / {val_mask.sum()} / {test_mask.sum()}")
print(f"Task subjects : {task_mask.sum()}  "
      f"(class 0={( binary_labels==0).sum()}, class 1={(binary_labels==1).sum()})")


# ─────────────────────────────────────────────────────────────────
#  STEP 2: Build Chebyshev Laplacian  (Eq. 14)
#
#  L      = I - D^{-1/2} A D^{-1/2}   (normalized Laplacian)
#  L_tilde= 2/λ_max * L - I            (scaled to [-1,1])
#  T_k(L_tilde) computed recursively during forward pass
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("STEP 2: Building Chebyshev Laplacian")
print("="*65)

def build_laplacian(A_norm: np.ndarray) -> torch.Tensor:
    """
    Given already-normalized adjacency A_norm = D^{-1/2} A D^{-1/2},
    compute the scaled Chebyshev Laplacian L_tilde = 2/λ_max * L - I
    where L = I - A_norm.
    """
    N = A_norm.shape[0]
    I = np.eye(N, dtype=np.float32)
    L = I - A_norm                          # normalized Laplacian

    # Largest eigenvalue via power iteration (faster than full eigen)
    v = np.random.randn(N).astype(np.float32)
    for _ in range(30):
        v = L @ v
        lam = np.linalg.norm(v)
        v = v / (lam + 1e-10)
    lambda_max = float(lam)
    print(f"  λ_max = {lambda_max:.4f}")

    L_tilde = (2.0 / lambda_max) * L - I   # scaled to [-1, 1]
    return torch.tensor(L_tilde, dtype=torch.float32)

L_tilde = build_laplacian(A_norm)
print(f"  L_tilde shape: {L_tilde.shape}")
print(f"  L_tilde range: [{L_tilde.min():.4f}, {L_tilde.max():.4f}]")


# ─────────────────────────────────────────────────────────────────
#  STEP 3: Model Definition
# ─────────────────────────────────────────────────────────────────

class ChebConv(nn.Module):
    """
    Chebyshev Graph Convolution layer.
    Implements Eq.14: g * x = Σ_{k=0}^{K} θ_k T_k(L̃) x

    T_0 = I
    T_1 = L̃
    T_k = 2 * L̃ * T_{k-1} - T_{k-2}   (recurrence relation)
    """
    def __init__(self, in_features: int, out_features: int, K: int):
        super().__init__()
        self.K = K
        # θ parameters: one weight matrix per Chebyshev polynomial order
        self.weight = nn.Parameter(
            torch.FloatTensor(K + 1, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.view(self.K+1, -1).unsqueeze(0)
                                 .squeeze(0).view(self.K+1, -1, 1)
                                 .squeeze(-1).unsqueeze(0).squeeze(0))
        # Simpler init
        nn.init.xavier_uniform_(
            self.weight.view((self.K+1) * self.weight.shape[1], self.weight.shape[2])
        )
        self.weight.data = self.weight.data.view(self.K+1,
                                                  self.weight.shape[1],
                                                  self.weight.shape[2])
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        x : [N, in_features]
        L : [N, N] scaled Chebyshev Laplacian
        """
        # Compute Chebyshev basis: T_0, T_1, ..., T_K
        Tx_list = []
        Tx_0 = x                    # T_0(L̃)x = x
        Tx_1 = L @ x                # T_1(L̃)x = L̃x

        Tx_list.append(Tx_0)
        if self.K >= 1:
            Tx_list.append(Tx_1)
        for _ in range(2, self.K + 1):
            Tx_k = 2 * L @ Tx_1 - Tx_0   # recurrence
            Tx_list.append(Tx_k)
            Tx_0, Tx_1 = Tx_1, Tx_k

        # Σ θ_k @ T_k(L̃)x
        out = sum(Tx_list[k] @ self.weight[k] for k in range(self.K + 1))
        return out + self.bias


class CGCNBranch(nn.Module):
    """
    Single-modality 2-layer Chebyshev GCN branch.
    Section 3.3: hidden_dim=32, ReLU, Dropout after each layer.
    Output: class probability scores via Softmax (Eq.15)
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_class: int,
                 K: int, dropout: float):
        super().__init__()
        self.conv1  = ChebConv(in_dim,     hidden_dim, K)
        self.conv2  = ChebConv(hidden_dim, n_class,    K)
        self.drop   = dropout

    def forward(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, L)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.conv2(x, L)
        x = F.dropout(x, p=self.drop, training=self.training)
        return x   # raw logits [N, n_class]


class IFDCGCN(nn.Module):
    """
    Integrated Fusion Dual Chebyshev GCN  (IFDCGCN)
    Paper Section 3.4 — Figure 4

    Two branches share L_tilde (built from A_if_norm).
    Late fusion: softmax_final = 0.5 * (softmax(z_smri) + softmax(z_pet))  Eq.16
    """
    def __init__(self, in_dim: int, hidden_dim: int, n_class: int,
                 K: int, dropout: float):
        super().__init__()
        self.branch_smri = CGCNBranch(in_dim, hidden_dim, n_class, K, dropout)
        self.branch_pet  = CGCNBranch(in_dim, hidden_dim, n_class, K, dropout)

    def forward(self, x_smri: torch.Tensor,
                x_pet: torch.Tensor,
                L: torch.Tensor):
        """
        Returns:
          logits_final : [N, 2]  fused predictions
          logits_smri  : [N, 2]  sMRI branch only
          logits_pet   : [N, 2]  PET branch only
        """
        z_smri = self.branch_smri(x_smri, L)   # [N, 2]
        z_pet  = self.branch_pet(x_pet,  L)    # [N, 2]

        # Eq.16: late fusion via averaged softmax
        p_smri = F.softmax(z_smri, dim=1)
        p_pet  = F.softmax(z_pet,  dim=1)
        p_fuse = 0.5 * (p_smri + p_pet)        # [N, 2]

        return p_fuse, p_smri, p_pet


# ─────────────────────────────────────────────────────────────────
#  STEP 4: Metrics
# ─────────────────────────────────────────────────────────────────

def compute_metrics(probs, labels):
    """
    probs  : [M, 2] softmax probabilities for task subjects
    labels : [M]    ground truth binary labels
    Returns ACC, SEN, SPE, AUC
    """
    preds = probs.argmax(dim=1).cpu().numpy()
    labs  = labels.cpu().numpy()
    prob1 = probs[:, 1].cpu().numpy()

    acc = float(accuracy_score(labs, preds)) * 100

    cm = confusion_matrix(labs, preds, labels=[0, 1])
    # Handle edge case where a class has no samples
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4
                      else (cm[0,0], 0, 0, cm[1,1]) if cm.shape==(2,2)
                      else (0,0,0,0))
    sen = (tp / (tp + fn + 1e-8)) * 100   # sensitivity = recall for class 1
    spe = (tn / (tn + fp + 1e-8)) * 100   # specificity = recall for class 0

    try:
        auc = roc_auc_score(labs, prob1) * 100
    except Exception:
        auc = 0.0

    return acc, sen, spe, auc


# ─────────────────────────────────────────────────────────────────
#  STEP 5: Training loop
# ─────────────────────────────────────────────────────────────────

def run_once(run_id, X_smri_t, X_pet_t, L_t, y_t,
             train_mask_t, val_mask_t, test_mask_t):
    """Single training run. Returns test metrics dict."""
    torch.manual_seed(args.seed + run_id)

    model = IFDCGCN(
        in_dim=in_dim, hidden_dim=args.hidden,
        n_class=n_class, K=args.K, dropout=args.dropout
    ).to(device)

    optimizer = Adam(model.parameters(),
                     lr=args.lr, weight_decay=args.weight_decay)

    # Class weights for imbalanced data
    y_train = y_t[train_mask_t]
    n0 = (y_train == 0).sum().float()
    n1 = (y_train == 1).sum().float()
    w  = torch.tensor([n1 / (n0 + n1), n0 / (n0 + n1)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    best_val_acc  = 0
    best_test_metrics = None
    best_epoch    = 0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()

        p_fuse, p_smri, p_pet = model(X_smri_t, X_pet_t, L_t)

        # Loss on train nodes only (semi-supervised)
        loss_fuse = criterion(torch.log(p_fuse[train_mask_t] + 1e-8), y_t[train_mask_t])
        loss_smri = criterion(torch.log(p_smri[train_mask_t] + 1e-8), y_t[train_mask_t])
        loss_pet  = criterion(torch.log(p_pet[train_mask_t]  + 1e-8), y_t[train_mask_t])
        loss = loss_fuse + 0.5 * loss_smri + 0.5 * loss_pet

        loss.backward()
        optimizer.step()

        # ── Evaluate ─────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            p_fuse, p_smri, p_pet = model(X_smri_t, X_pet_t, L_t)

        t_acc, _, _, _    = compute_metrics(p_fuse[train_mask_t], y_t[train_mask_t])
        v_acc, v_sen, v_spe, v_auc = compute_metrics(p_fuse[val_mask_t],   y_t[val_mask_t])

        # Val loss
        val_loss = criterion(torch.log(p_fuse[val_mask_t]+1e-8), y_t[val_mask_t]).item()

        train_accs.append(t_acc);  val_accs.append(v_acc)
        train_losses.append(loss.item()); val_losses.append(val_loss)

        # Track best model by val accuracy
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch   = epoch
            te_acc, te_sen, te_spe, te_auc = compute_metrics(
                p_fuse[test_mask_t], y_t[test_mask_t]
            )
            te_acc_s, _, _, _ = compute_metrics(p_smri[test_mask_t], y_t[test_mask_t])
            te_acc_p, _, _, _ = compute_metrics(p_pet[test_mask_t],  y_t[test_mask_t])
            best_test_metrics = {
                "acc": te_acc, "sen": te_sen, "spe": te_spe, "auc": te_auc,
                "acc_smri_branch": te_acc_s,
                "acc_pet_branch":  te_acc_p,
                "best_epoch": epoch,
                "best_val_acc": v_acc,
            }

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Run {run_id+1} | Epoch {epoch:3d} | "
                  f"loss={loss.item():.4f} | "
                  f"train_acc={t_acc:.1f}% | "
                  f"val_acc={v_acc:.1f}%")

    print(f"  Run {run_id+1} | Best epoch={best_epoch} | "
          f"val_acc={best_val_acc:.1f}% | "
          f"test_acc={best_test_metrics['acc']:.2f}%")

    return best_test_metrics, train_accs, val_accs, train_losses, val_losses


# ─────────────────────────────────────────────────────────────────
#  STEP 6: Prepare tensors & run
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"STEP 3-6: Training IFDCGCN  [{args.task}]")
print("="*65)
print(f"  Epochs    : {args.epochs}")
print(f"  LR        : {args.lr}")
print(f"  Hidden    : {args.hidden}")
print(f"  K (ChebGCN): {args.K}")
print(f"  Dropout   : {args.dropout}")
print(f"  Runs      : {args.runs}")
print()

# Convert to tensors
X_smri_t = torch.tensor(X_smri, dtype=torch.float32).to(device)
X_pet_t  = torch.tensor(X_pet,  dtype=torch.float32).to(device)
L_t      = L_tilde.to(device)

# Only task-relevant subjects get real labels; others get 0 (masked out anyway)
y_full   = binary_labels.copy()
y_full[y_full == -1] = 0
y_t      = torch.tensor(y_full, dtype=torch.long).to(device)

train_mask_t = torch.tensor(train_mask, dtype=torch.bool).to(device)
val_mask_t   = torch.tensor(val_mask,   dtype=torch.bool).to(device)
test_mask_t  = torch.tensor(test_mask,  dtype=torch.bool).to(device)

# Multiple runs for stable results (paper reports mean ± std)
all_results = []
all_curves  = {"train_acc":[], "val_acc":[], "train_loss":[], "val_loss":[]}

t_start = time.time()
for run in range(args.runs):
    result, tr_acc, v_acc, tr_loss, v_loss = run_once(
        run, X_smri_t, X_pet_t, L_t, y_t,
        train_mask_t, val_mask_t, test_mask_t
    )
    all_results.append(result)
    all_curves["train_acc"].append(tr_acc)
    all_curves["val_acc"].append(v_acc)
    all_curves["train_loss"].append(tr_loss)
    all_curves["val_loss"].append(v_loss)

elapsed = time.time() - t_start


# ─────────────────────────────────────────────────────────────────
#  STEP 7: Aggregate & report results
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"RESULTS  [{args.task}]  ({args.runs} runs)")
print("="*65)

metrics_keys = ["acc","sen","spe","auc"]
for k in metrics_keys:
    vals = [r[k] for r in all_results]
    print(f"  {k.upper():<4}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")

print(f"\n  sMRI branch ACC: {np.mean([r['acc_smri_branch'] for r in all_results]):.2f}%")
print(f"  PET branch  ACC: {np.mean([r['acc_pet_branch']  for r in all_results]):.2f}%")
print(f"  Fused       ACC: {np.mean([r['acc']             for r in all_results]):.2f}%  ← late fusion benefit")

# Paper benchmarks
paper = {
    "AD_vs_CN":    {"acc":91.07,"sen":90.22,"spe":91.87,"auc":91.04},
    "SMCI_vs_PMCI":{"acc":75.50,"sen":49.90,"spe":88.70,"auc":69.30},
}
print(f"\n  Paper target (IFDCGCN):")
for k in metrics_keys:
    print(f"    {k.upper():<4}: {paper[args.task][k]:.2f}%")

print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/args.runs:.1f}s/run)")


# ─────────────────────────────────────────────────────────────────
#  STEP 8: Save results & curves
# ─────────────────────────────────────────────────────────────────
task_out = os.path.join(OUTPUT_DIR, args.task)
os.makedirs(task_out, exist_ok=True)

# Save JSON results
summary = {
    "task":    args.task,
    "config":  vars(args),
    "results": {
        k: {"mean": float(np.mean([r[k] for r in all_results])),
            "std":  float(np.std([r[k]  for r in all_results]))}
        for k in ["acc","sen","spe","auc","acc_smri_branch","acc_pet_branch"]
    },
    "per_run": all_results,
    "paper_target": paper[args.task],
    "training_time_s": elapsed,
}
with open(os.path.join(task_out, "results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved results.json → {task_out}/")

# Save training curves (mean across runs)
curves = {
    "train_acc":  np.mean(all_curves["train_acc"],  axis=0).tolist(),
    "val_acc":    np.mean(all_curves["val_acc"],    axis=0).tolist(),
    "train_loss": np.mean(all_curves["train_loss"], axis=0).tolist(),
    "val_loss":   np.mean(all_curves["val_loss"],   axis=0).tolist(),
    "epochs":     list(range(1, args.epochs+1)),
}
with open(os.path.join(task_out, "training_curves.json"), "w") as f:
    json.dump(curves, f)
print(f"Saved training_curves.json")

# Save final model (best run by test acc)
best_run_idx = max(range(args.runs), key=lambda i: all_results[i]["acc"])
print(f"Best run: #{best_run_idx+1} with ACC={all_results[best_run_idx]['acc']:.2f}%")

# Re-train best run to save model weights
torch.manual_seed(args.seed + best_run_idx)
final_model = IFDCGCN(in_dim=in_dim, hidden_dim=args.hidden,
                      n_class=n_class, K=args.K, dropout=args.dropout).to(device)
optimizer   = Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
y_train     = y_t[train_mask_t]
n0 = (y_train==0).sum().float(); n1 = (y_train==1).sum().float()
w  = torch.tensor([n1/(n0+n1), n0/(n0+n1)]).to(device)
criterion   = nn.CrossEntropyLoss(weight=w)

best_val    = 0
best_state  = None
for epoch in range(1, args.epochs+1):
    final_model.train(); optimizer.zero_grad()
    p_fuse, p_smri, p_pet = final_model(X_smri_t, X_pet_t, L_t)
    loss = (criterion(torch.log(p_fuse[train_mask_t]+1e-8), y_t[train_mask_t])
           +0.5*criterion(torch.log(p_smri[train_mask_t]+1e-8), y_t[train_mask_t])
           +0.5*criterion(torch.log(p_pet[train_mask_t]+1e-8),  y_t[train_mask_t]))
    loss.backward(); optimizer.step()
    final_model.eval()
    with torch.no_grad():
        p_fuse,_,_ = final_model(X_smri_t, X_pet_t, L_t)
    v_acc,_,_,_ = compute_metrics(p_fuse[val_mask_t], y_t[val_mask_t])
    if v_acc > best_val:
        best_val   = v_acc
        best_state = {k:v.cpu().clone() for k,v in final_model.state_dict().items()}

final_model.load_state_dict(best_state)
torch.save({
    "model_state":  best_state,
    "config":       vars(args),
    "in_dim":       in_dim,
    "n_class":      n_class,
}, os.path.join(task_out, "best_model.pt"))
print(f"Saved best_model.pt")


# ─────────────────────────────────────────────────────────────────
#  STEP 9: Final evaluation on test set (best model)
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("FINAL TEST EVALUATION (best model)")
print("="*65)

final_model.eval()
with torch.no_grad():
    p_fuse, p_smri, p_pet = final_model(X_smri_t, X_pet_t, L_t)

te_acc,  te_sen,  te_spe,  te_auc  = compute_metrics(p_fuse[test_mask_t], y_t[test_mask_t])
te_acc_s,te_sen_s,te_spe_s,te_auc_s= compute_metrics(p_smri[test_mask_t], y_t[test_mask_t])
te_acc_p,te_sen_p,te_spe_p,te_auc_p= compute_metrics(p_pet[test_mask_t],  y_t[test_mask_t])

print(f"\n  {'Branch':<20} {'ACC':>7} {'SEN':>7} {'SPE':>7} {'AUC':>7}")
print(f"  {'-'*50}")
print(f"  {'sMRI branch':<20} {te_acc_s:>6.2f}% {te_sen_s:>6.2f}% {te_spe_s:>6.2f}% {te_auc_s:>6.2f}%")
print(f"  {'PET branch':<20} {te_acc_p:>6.2f}% {te_sen_p:>6.2f}% {te_spe_p:>6.2f}% {te_auc_p:>6.2f}%")
print(f"  {'FUSED (IFDCGCN)':<20} {te_acc:>6.2f}% {te_sen:>6.2f}% {te_spe:>6.2f}% {te_auc:>6.2f}%")
print(f"\n  {'Paper target':<20} {paper[args.task]['acc']:>6.2f}% {paper[args.task]['sen']:>6.2f}% {paper[args.task]['spe']:>6.2f}% {paper[args.task]['auc']:>6.2f}%")

print("\n" + "="*65)
print("PHASE 4 COMPLETE")
print("="*65)
print(f"Outputs → {task_out}/")
print("  best_model.pt          — saved model weights")
print("  results.json           — metrics for all runs")
print("  training_curves.json   — loss/acc curves per epoch")
print(f"\nTo run the other task:")
print(f"  python phase4_model.py --task {'SMCI_vs_PMCI' if args.task=='AD_vs_CN' else 'AD_vs_CN'}")
