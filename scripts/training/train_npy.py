"""
Phase 4: IFDCGCN Training  —  drop-in replacement for main_adj_dot_prod_ok1.py
Loads directly from Phase 3 .npy outputs instead of CSVs.

All hyperparameters, model architecture, training loop, voting, early-stop,
confusion matrix, and plotting are identical to the author's code.

Directory layout expected:
  PHASE3_DIR/
    labels_str.npy
    phenotypic_aligned.csv     ← columns: subject, label, label_int, age, gender, MMSCORE
    AD_vs_CN/
            X_smri.npy, X_pet.npy
            A_if_norm.npy            ← task-specific normalised adjacency
      train_mask.npy, val_mask.npy, test_mask.npy, binary_labels.npy, task_mask.npy
    SMCI_vs_PMCI/
            X_smri.npy, X_pet.npy
            A_if_norm.npy            ← task-specific normalised adjacency
      train_mask.npy, val_mask.npy, test_mask.npy, binary_labels.npy, task_mask.npy

Usage:
  python train_npy.py --task AD_vs_CN
  python train_npy.py --task SMCI_vs_PMCI
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear
from torch_geometric.nn import ChebConv
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, confusion_matrix
from matplotlib import pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  CONFIG  — change PHASE3_DIR to your actual path
# ─────────────────────────────────────────────────────────────────
PHASE3_DIR = "./outputs/phase3_outputs"

# ── Hyperparameters (identical to author's parameters.py) ────────
learning_rate        = 0.001
weight_decay         = 5e-4
epochs               = 300
minimum_epochs       = 30
epochlimit           = 50
gap                  = 0.02
Kco                  = 3          # Chebyshev order K
hidden_channels_Cheb = 32
model_weight         = 0.5        # each branch weight for late fusion (Eq.16)
early_stop           = False      # set True to enable early stopping

# ─────────────────────────────────────────────────────────────────
#  MODEL  (identical to author's models.py → class Cheb)
# ─────────────────────────────────────────────────────────────────
class Cheb(torch.nn.Module):
    def __init__(self, dim_nodes, hidden_channels):
        super(Cheb, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = ChebConv(dim_nodes, hidden_channels, K=Kco)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=Kco)
        self.lin1  = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        return x


# ─────────────────────────────────────────────────────────────────
#  HELPERS  (identical logic to author's util.py)
# ─────────────────────────────────────────────────────────────────
def normalization(adjacency):
    """D^{-0.5} (A+I) D^{-0.5}  —  same as author's normalization()"""
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat  = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()


def processData(features_np, adjacency_np, device):
    """
    Mirrors author's processData():
      normalise adjacency → COO → edge_index + edge_weight
    Returns (features_tensor, edge_index, edge_weight, None)
    """
    features  = torch.tensor(features_np, dtype=torch.float32).to(device)
    num_nodes = features.shape[0]

    coo       = coo_matrix(adjacency_np)
    norm_coo  = normalization(coo)

    edge_index = torch.from_numpy(
        np.asarray([norm_coo.row, norm_coo.col]).astype("int64")
    ).long().to(device)

    edge_weight = torch.from_numpy(
        norm_coo.data.astype(np.float32)
    ).to(device)

    return features, edge_index, edge_weight, None


def votingConference(logits_0, logits_1, w0, w1, labels):
    """
    Late fusion — Eq.16  (identical to author's votingConference)
    logits_0 / logits_1 : raw logits for masked nodes [n_masked, 2]
    """
    p0     = F.softmax(logits_0, dim=1)
    p1     = F.softmax(logits_1, dim=1)
    output = p0 * w0 + p1 * w1
    result = torch.max(output, 1)[1]
    acc    = torch.eq(result, labels).float().mean()
    return acc, result


def earlyStop(epoch, best_train_acc, best_val_acc,
              current_train_acc, current_val_acc):
    """Identical to author's earlyStop()"""
    if not early_stop:
        return False
    if best_train_acc > 0.99 and best_val_acc <= current_val_acc:
        return True
    if (epoch > epochlimit
            and (best_val_acc - gap) <= current_val_acc
            and current_train_acc >= (best_train_acc - gap)):
        return True
    return False


def confusionMatrix(true_label, data_pre):
    """Identical to author's confusionMatrix()"""
    TN, FP, FN, TP = confusion_matrix(true_label, data_pre).ravel()
    ACC = 100 * (TP + TN) / (TP + TN + FP + FN)
    SEN = 100 * TP / (TP + FN)
    SPE = 100 * TN / (TN + FP)
    AUC = 100 * roc_auc_score(true_label, data_pre)
    print("The result of test data for:")
    print(f"TP: {TP}  FP: {FP}  FN: {FN}  TN: {TN}")
    print(f"ACC: {ACC:.4f} %")
    print(f"SEN: {SEN:.4f} %")
    print(f"SPE: {SPE:.4f} %")
    print(f"AUC: {AUC:.4f} %")
    print()


def calculateAccuracy(logits, labels, mask):
    """Identical to author's calculateAccuracy()"""
    masked_logits = logits[mask]
    predict_y     = masked_logits.max(1)[1]
    accuracy      = torch.eq(predict_y, labels[mask]).float().mean()
    return accuracy, predict_y


def modelTest(model, mask, labels, features, edge_index, edge_weight,
              show_cm=False):
    """Identical to author's test()"""
    model.eval()
    true_label, data_pre = [], []
    with torch.no_grad():
        logits   = model(features, edge_index, edge_weight)
        acc, py  = calculateAccuracy(logits, labels, mask)
        data_pre.extend(py.cpu().numpy().tolist())
        true_label.extend(labels[mask].cpu().numpy().tolist())
    if show_cm:
        confusionMatrix(true_label, data_pre)
    return acc, logits[mask]


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main(task: str):
    print("=" * 65)
    print(f"IFDCGCN Training  —  Task: {task}")
    print("=" * 65)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss().to(device)
    task_dir  = os.path.join(PHASE3_DIR, task)

    # ── 1. Load node labels for logging ───────────────────────────
    labels_str  = np.load(os.path.join(PHASE3_DIR, "labels_str.npy"),
                          allow_pickle=True)

    # ── 2. Load task-specific graph/features (already filtered) ───
    A_sub  = np.load(os.path.join(task_dir, "A_if_norm.npy"))
    X_smri = np.load(os.path.join(task_dir, "X_smri.npy"))
    X_pet  = np.load(os.path.join(task_dir, "X_pet.npy"))

    # ── 3. Load task masks & labels ───────────────────────────────
    task_mask     = np.load(os.path.join(task_dir, "task_mask.npy"))     # [N] bool
    binary_labels = np.load(os.path.join(task_dir, "binary_labels.npy")) # [N] int, -1=other
    train_mask_full = np.load(os.path.join(task_dir, "train_mask.npy"))  # [N] bool
    val_mask_full   = np.load(os.path.join(task_dir, "val_mask.npy"))    # [N] bool
    test_mask_full  = np.load(os.path.join(task_dir, "test_mask.npy"))   # [N] bool

    # ── 4. Use task-specific graph directly; align masks if stored at full-N ─
    Nt = X_smri.shape[0]

    labels_task = binary_labels if len(binary_labels) == Nt else binary_labels[task_mask]

    if len(train_mask_full) == Nt:
        train_mask = train_mask_full
        val_mask = val_mask_full
        test_mask = test_mask_full
    else:
        train_mask = train_mask_full[task_mask]
        val_mask = val_mask_full[task_mask]
        test_mask = test_mask_full[task_mask]

    print(f"Task subjects  : {Nt}")
    if task == "AD_vs_CN":
        lnames = ["CN", "AD"]
    else:
        lnames = ["SMCI", "PMCI"]
    ls = labels_str[task_mask]
    print(f"Label counts   : { {k: int((ls==k).sum()) for k in lnames} }")
    print(f"Train / Val / Test : {train_mask.sum()} / {val_mask.sum()} / {test_mask.sum()}")

    labels = torch.LongTensor(labels_task).to(device)
    train_mask_t = torch.from_numpy(train_mask).to(device)
    val_mask_t   = torch.from_numpy(val_mask).to(device)
    test_mask_t  = torch.from_numpy(test_mask).to(device)

    # ── 5. Build adjacency dot-product & process both branches ────
    # Author's main: adjacency_dot_product = adjacency_0 * adjacency_1
    # Here A_sub is already the integrated-fusion adjacency (Eq.13),
    # so we use it for BOTH branches directly (same as author's dot-product path).
    feat_smri, ei_smri, ew_smri, ta_smri = processData(X_smri, A_sub, device)
    feat_pet,  ei_pet,  ew_pet,  ta_pet  = processData(X_pet,  A_sub, device)

    print(f"\nsMRI features  : {feat_smri.shape}")
    print(f"PET  features  : {feat_pet.shape}")
    print(f"edge_index     : {ei_smri.shape}  (shared adjacency, same for both branches)")

    # ── 6. Build models & optimisers (identical to author's getModels) ──
    model_smri = Cheb(dim_nodes=feat_smri.shape[1],
                      hidden_channels=hidden_channels_Cheb).to(device)
    model_pet  = Cheb(dim_nodes=feat_pet.shape[1],
                      hidden_channels=hidden_channels_Cheb).to(device)

    opt_smri = optim.Adam(model_smri.parameters(),
                          lr=learning_rate, weight_decay=weight_decay)
    opt_pet  = optim.Adam(model_pet.parameters(),
                          lr=learning_rate, weight_decay=weight_decay)

    # ── 7. History buffers (identical to author's main) ───────────
    loss_hist_smri, loss_hist_pet       = [], []
    train_acc_smri, train_acc_pet       = [], []
    val_acc_smri,   val_acc_pet         = [], []
    vote_train_hist, vote_val_hist      = [], []

    best_vote_train = best_vote_val = best_vote_test = 0.0
    ckpt_path = f"best_{task}.pt"

    # ── 8. Training loop (identical structure to author's main) ───
    print("\n" + "=" * 65)
    print("Training ...")
    print("=" * 65)

    for epoch in range(epochs):

        # ── sMRI branch ──────────────────────────────────────────
        model_smri.train()
        logits_smri     = model_smri(feat_smri, ei_smri, ew_smri)
        train_log_smri  = logits_smri[train_mask_t]
        loss_smri       = criterion(train_log_smri, labels[train_mask_t])
        opt_smri.zero_grad();  loss_smri.backward();  opt_smri.step()

        tr_acc_s, _  = modelTest(model_smri, train_mask_t, labels, feat_smri, ei_smri, ew_smri)
        val_acc_s, val_log_smri  = modelTest(model_smri, val_mask_t,   labels, feat_smri, ei_smri, ew_smri)
        tst_acc_s, tst_log_smri  = modelTest(model_smri, test_mask_t,  labels, feat_smri, ei_smri, ew_smri)

        # ── PET branch ───────────────────────────────────────────
        model_pet.train()
        logits_pet      = model_pet(feat_pet, ei_pet, ew_pet)
        train_log_pet   = logits_pet[train_mask_t]
        loss_pet        = criterion(train_log_pet, labels[train_mask_t])
        opt_pet.zero_grad();  loss_pet.backward();  opt_pet.step()

        tr_acc_p, _  = modelTest(model_pet,  train_mask_t, labels, feat_pet,  ei_pet,  ew_pet)
        val_acc_p, val_log_pet   = modelTest(model_pet,  val_mask_t,   labels, feat_pet,  ei_pet,  ew_pet)
        tst_acc_p, tst_log_pet   = modelTest(model_pet,  test_mask_t,  labels, feat_pet,  ei_pet,  ew_pet)

        # ── Late fusion / voting (Eq.16) ─────────────────────────
        vote_tr,  _          = votingConference(
            logits_smri[train_mask_t], logits_pet[train_mask_t],
            model_weight, model_weight, labels[train_mask_t])
        vote_val, _          = votingConference(
            val_log_smri, val_log_pet,
            model_weight, model_weight, labels[val_mask_t])
        vote_tst, tst_pred   = votingConference(
            tst_log_smri, tst_log_pet,
            model_weight, model_weight, labels[test_mask_t])

        # ── Logging ──────────────────────────────────────────────
        loss_hist_smri.append(loss_smri.item())
        loss_hist_pet.append(loss_pet.item())
        train_acc_smri.append(tr_acc_s.item())
        train_acc_pet.append(tr_acc_p.item())
        val_acc_smri.append(val_acc_s.item())
        val_acc_pet.append(val_acc_p.item())
        vote_train_hist.append(vote_tr.item())
        vote_val_hist.append(vote_val.item())

        print(
            f"Epoch {epoch:03d} | "
            f"sMRI  train={tr_acc_s:.4f} val={val_acc_s:.4f} test={tst_acc_s:.4f} loss={loss_smri.item():.4f} | "
            f"PET   train={tr_acc_p:.4f} val={val_acc_p:.4f} test={tst_acc_p:.4f} loss={loss_pet.item():.4f}"
        )
        print(
            f"         voting train={vote_tr:.4f}  val={vote_val:.4f}  test={vote_tst:.4f}"
        )

        # ── Confusion matrix every epoch (identical to author) ───
        true_label = labels[test_mask_t].cpu().numpy().tolist()
        data_pre   = tst_pred.cpu().numpy().tolist()
        confusionMatrix(true_label, data_pre)

        # ── Best tracker/checkpoint (after minimum_epochs) ───────
        if vote_tr.item() >= best_vote_train:
            best_vote_train = vote_tr.item()
        if epoch > minimum_epochs:
            if vote_val.item() >= best_vote_val:
                best_vote_val = vote_val.item()
                best_vote_test = vote_tst.item()
                torch.save({
                    'smri': model_smri.state_dict(),
                    'pet':  model_pet.state_dict(),
                    'epoch': epoch
                }, ckpt_path)

        # ── Early stop (identical to author's earlyStop) ─────────
        if earlyStop(epoch, best_vote_train, best_vote_val,
                     vote_tr.item(), vote_val.item()):
            print(f"Early stop at epoch {epoch}")
            break

    # ── 9. Final summary ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("FINAL RESULTS (best after minimum_epochs)")
    print("=" * 65)
    print(f"Voting fusion — best val={best_vote_val:.4f}  best test={best_vote_test:.4f}")

    # Final confusion matrix on test set (show_cm=True)
    print("\nFinal test confusion matrix (voting):")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_smri.load_state_dict(ckpt['smri'])
    model_pet.load_state_dict(ckpt['pet'])
    model_smri.eval(); model_pet.eval()
    with torch.no_grad():
        fl_smri = model_smri(feat_smri, ei_smri, ew_smri)[test_mask_t]
        fl_pet  = model_pet(feat_pet,   ei_pet,  ew_pet)[test_mask_t]
    _, final_pred = votingConference(fl_smri, fl_pet, model_weight, model_weight,
                                     labels[test_mask_t])
    confusionMatrix(labels[test_mask_t].cpu().numpy().tolist(),
                    final_pred.cpu().numpy().tolist())

    # ── 10. Plots (identical to author's plot section) ────────────
    plot_dir = f"./npy_training_plots_{task}"
    os.makedirs(plot_dir, exist_ok=True)
    colors = ["pink", "purple"]
    ep = range(len(loss_hist_smri))

    plt.figure(); plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("training losses")
    plt.plot(ep, loss_hist_smri, color=colors[0], label="sMRI", linewidth=1.2)
    plt.plot(ep, loss_hist_pet,  color=colors[1], label="PET",  linewidth=1.2)
    plt.legend(loc="upper left")
    plt.savefig(f"{plot_dir}/training_losses.png"); plt.show(block=False); plt.clf()

    plt.figure(); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("validation accuracies")
    plt.plot(ep, val_acc_smri, color=colors[0], label="sMRI",     linewidth=1.0)
    plt.plot(ep, val_acc_pet,  color=colors[1], label="PET",      linewidth=1.0)
    plt.plot(ep, vote_val_hist, color="green",  label="decision",  linewidth=1.0)
    plt.legend(loc="upper left")
    plt.savefig(f"{plot_dir}/validation_accuracies.png"); plt.show(block=False); plt.clf()

    plt.figure(); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("training accuracies")
    plt.plot(ep, train_acc_smri, color=colors[0], label="sMRI", linewidth=1.0)
    plt.plot(ep, train_acc_pet,  color=colors[1], label="PET",  linewidth=1.0)
    plt.legend(loc="upper left")
    plt.savefig(f"{plot_dir}/training_accuracies.png"); plt.show(block=False); plt.clf()

    print(f"\nPlots saved to {plot_dir}/")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["AD_vs_CN", "SMCI_vs_PMCI"],
        default="AD_vs_CN",
        help="Which classification task to run"
    )
    args = parser.parse_args()
    main(args.task)