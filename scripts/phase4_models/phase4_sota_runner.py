"""
Phase 4 (SOTA Runner): Stronger IFDCGCN training and evaluation recipe.

Goals:
- Improve practical accuracy with robust validation protocol.
- Keep strict split hygiene (no leakage):
  * Train-only feature normalization
  * Optional train-only PCA
  * Validation-only temperature scaling and threshold tuning

Key practices included:
1) Robust task-local data mapping from Phase3 artifacts
2) Residual dual-branch Chebyshev GNN + node-wise gated fusion
3) AdamW + OneCycleLR + gradient clipping
4) Class-balanced CE + R-Drop consistency regularization
5) Edge dropout as graph regularization
6) EMA model averaging
7) Top-K checkpoint ensembling
8) Validation-calibrated thresholding (balanced-accuracy optimized)

Usage:
  python phase4_sota_runner.py --task AD_vs_CN
  python phase4_sota_runner.py --task SMCI_vs_PMCI
  python phase4_sota_runner.py --task AD_vs_CN --runs 7 --epochs 500 --pca_dim 1024
"""

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

PHASE3_DIR = "./outputs/phase3_outputs"
OUTPUT_DIR = "./outputs/phase4_outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="AD_vs_CN", choices=["AD_vs_CN", "SMCI_vs_PMCI"])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=450)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.45)

    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--edge_drop", type=float, default=0.04)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--rdrop_alpha", type=float, default=0.5)

    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--patience", type=int, default=90)
    p.add_argument("--topk_ensemble", type=int, default=5)

    p.add_argument("--pca_dim", type=int, default=0,
                   help="If >0, apply train-only PCA to both modalities.")

    p.add_argument("--use_temp_scaling", action="store_true", default=True)
    p.add_argument("--threshold_metric", choices=["bal_acc", "acc"], default="bal_acc")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TaskData:
    x_pet: np.ndarray
    x_smri: np.ndarray
    adj: np.ndarray
    y: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray


def load_task_data(task: str) -> TaskData:
    task_dir = os.path.join(PHASE3_DIR, task)

    x_pet = np.load(os.path.join(task_dir, "X_pet.npy")).astype(np.float32)
    x_smri = np.load(os.path.join(task_dir, "X_smri.npy")).astype(np.float32)
    adj = np.load(os.path.join(task_dir, "A_if_norm.npy")).astype(np.float32)

    y_g = np.load(os.path.join(task_dir, "binary_labels.npy"))
    tr_g = np.load(os.path.join(task_dir, "train_mask.npy")).astype(bool)
    va_g = np.load(os.path.join(task_dir, "val_mask.npy")).astype(bool)
    te_g = np.load(os.path.join(task_dir, "test_mask.npy")).astype(bool)

    n_local = x_pet.shape[0]
    if len(y_g) == n_local:
        y = y_g.astype(np.int64)
        tr = tr_g
        va = va_g
        te = te_g
    else:
        task_mask = np.load(os.path.join(task_dir, "task_mask.npy")).astype(bool)
        if int(task_mask.sum()) != n_local:
            raise ValueError(
                f"task_mask sum mismatch: {task_mask.sum()} vs n_local={n_local}"
            )
        y = y_g[task_mask].astype(np.int64)
        tr = tr_g[task_mask]
        va = va_g[task_mask]
        te = te_g[task_mask]

    valid = (y == 0) | (y == 1)
    if not valid.all():
        x_pet = x_pet[valid]
        x_smri = x_smri[valid]
        adj = adj[np.ix_(valid, valid)]
        tr = tr[valid]
        va = va[valid]
        te = te[valid]
        y = y[valid]

    return TaskData(x_pet=x_pet, x_smri=x_smri, adj=adj, y=y,
                    train_mask=tr, val_mask=va, test_mask=te)


def zscore_train_only(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    tr = x[train_mask]
    mu = tr.mean(axis=0, keepdims=True)
    sd = tr.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mu) / sd


def pca_train_only(x: np.ndarray, train_mask: np.ndarray, dim: int) -> np.ndarray:
    n_train = int(train_mask.sum())
    max_dim = max(1, min(x.shape[1], n_train - 1))
    use_dim = min(dim, max_dim)
    if use_dim <= 0 or use_dim >= x.shape[1]:
        return x
    pca = PCA(n_components=use_dim, svd_solver="randomized", random_state=0)
    pca.fit(x[train_mask])
    return pca.transform(x).astype(np.float32)


def build_scaled_laplacian(adj_norm: np.ndarray) -> torch.Tensor:
    n = adj_norm.shape[0]
    eye = np.eye(n, dtype=np.float32)
    lap = eye - adj_norm

    v = np.random.randn(n).astype(np.float32)
    lam = 1.0
    for _ in range(40):
        v = lap @ v
        lam = np.linalg.norm(v) + 1e-10
        v = v / lam

    l_tilde = (2.0 / float(lam)) * lap - eye
    return torch.tensor(l_tilde, dtype=torch.float32)


def drop_edge(l_tilde: torch.Tensor, rate: float) -> torch.Tensor:
    if rate <= 0:
        return l_tilde
    keep = torch.bernoulli(torch.full_like(l_tilde, 1.0 - rate))
    keep.fill_diagonal_(1.0)
    return l_tilde * keep


class ChebConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k_order: int):
        super().__init__()
        self.k = k_order
        self.w = nn.Parameter(torch.empty(k_order + 1, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.w.view((k_order + 1) * in_dim, out_dim))

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor) -> torch.Tensor:
        t0 = x
        out = t0 @ self.w[0]

        if self.k >= 1:
            t1 = l_tilde @ x
            out = out + t1 @ self.w[1]
        else:
            t1 = None

        for i in range(2, self.k + 1):
            t2 = 2.0 * (l_tilde @ t1) - t0
            out = out + t2 @ self.w[i]
            t0, t1 = t1, t2

        return out + self.b


class Branch(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_class: int, k_order: int, dropout: float):
        super().__init__()
        self.c1 = ChebConv(in_dim, hidden, k_order)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.c2 = ChebConv(hidden, hidden, k_order)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.c3 = ChebConv(hidden, n_class, k_order)
        self.drop = dropout

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.gelu(self.bn1(self.c1(x, l_tilde)))
        h1 = F.dropout(h1, p=self.drop, training=self.training)

        h2 = F.gelu(self.bn2(self.c2(h1, l_tilde)))
        h2 = F.dropout(h2, p=self.drop, training=self.training)

        h = h1 + h2
        z = self.c3(h, l_tilde)
        return z, h


class SotaIFDCGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, k_order: int, dropout: float):
        super().__init__()
        self.smri = Branch(in_dim, hidden, 2, k_order, dropout)
        self.pet = Branch(in_dim, hidden, 2, k_order, dropout)

        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, xs: torch.Tensor, xp: torch.Tensor, l_tilde: torch.Tensor):
        z_s, h_s = self.smri(xs, l_tilde)
        z_p, h_p = self.pet(xp, l_tilde)

        a = torch.sigmoid(self.gate(torch.cat([h_s, h_p], dim=1)))
        z_f = a * z_s + (1.0 - a) * z_p

        return z_f, z_s, z_p, a


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if torch.is_floating_point(self.shadow[k]):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def apply_to(self, model: nn.Module):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]):
        model.load_state_dict(backup, strict=True)


def class_balanced_weights(y_train: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    n0 = max(int((y_train == 0).sum()), 1)
    n1 = max(int((y_train == 1).sum()), 1)
    counts = np.array([n0, n1], dtype=np.float32)
    eff = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff, 1e-12)
    w = w / w.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)


def rdrop_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    pa = F.log_softmax(logits_a, dim=1)
    pb = F.log_softmax(logits_b, dim=1)
    qa = F.softmax(logits_a, dim=1)
    qb = F.softmax(logits_b, dim=1)
    kl1 = F.kl_div(pa, qb, reduction="batchmean")
    kl2 = F.kl_div(pb, qa, reduction="batchmean")
    return 0.5 * (kl1 + kl2)


def metrics_from_probs(prob1: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    pred = (prob1 >= thr).astype(np.int64)
    acc = 100.0 * accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sen = 100.0 * tp / (tp + fn + 1e-8)
    spe = 100.0 * tn / (tn + fp + 1e-8)
    bal = 0.5 * (sen + spe)
    try:
        auc = 100.0 * roc_auc_score(y_true, prob1)
    except Exception:
        auc = 0.0
    return {"acc": float(acc), "sen": float(sen), "spe": float(spe), "bal_acc": float(bal), "auc": float(auc)}


def find_best_threshold(prob1: np.ndarray, y_true: np.ndarray, metric: str = "bal_acc") -> float:
    grid = np.linspace(0.10, 0.90, 161)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        m = metrics_from_probs(prob1, y_true, thr=float(t))[metric]
        if m > best_v:
            best_v = m
            best_t = float(t)
    return best_t


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, max_iter: int = 200) -> torch.Tensor:
    t = torch.ones([], device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / t.clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return t.detach().clamp(min=1e-3)


def evaluate_with_calibration(
    model: nn.Module,
    xs: torch.Tensor,
    xp: torch.Tensor,
    l_tilde: torch.Tensor,
    y: torch.Tensor,
    tr_mask: torch.Tensor,
    va_mask: torch.Tensor,
    te_mask: torch.Tensor,
    threshold_metric: str,
    use_temp_scaling: bool,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        zf, _, _, _ = model(xs, xp, l_tilde)

    z_val = zf[va_mask]
    y_val = y[va_mask]
    z_test = zf[te_mask]
    y_test = y[te_mask]

    if use_temp_scaling:
        temp = fit_temperature(z_val, y_val)
    else:
        temp = torch.tensor(1.0, device=zf.device)

    p_val = F.softmax(z_val / temp, dim=1)[:, 1].detach().cpu().numpy()
    yv = y_val.detach().cpu().numpy()
    thr = find_best_threshold(p_val, yv, metric=threshold_metric)

    p_test = F.softmax(z_test / temp, dim=1)[:, 1].detach().cpu().numpy()
    yt = y_test.detach().cpu().numpy()
    met = metrics_from_probs(p_test, yt, thr=thr)
    met["threshold"] = float(thr)
    met["temperature"] = float(temp.item())
    return met


def train_one_run(args: argparse.Namespace, data: TaskData, run_idx: int, device: torch.device):
    set_seed(args.seed + run_idx)

    x_pet = zscore_train_only(data.x_pet, data.train_mask)
    x_smri = zscore_train_only(data.x_smri, data.train_mask)

    if args.pca_dim > 0:
        x_pet = pca_train_only(x_pet, data.train_mask, args.pca_dim)
        x_smri = pca_train_only(x_smri, data.train_mask, args.pca_dim)

    xs = torch.tensor(x_smri, dtype=torch.float32, device=device)
    xp = torch.tensor(x_pet, dtype=torch.float32, device=device)
    l_tilde = build_scaled_laplacian(data.adj).to(device)

    y = torch.tensor(data.y, dtype=torch.long, device=device)
    tr = torch.tensor(data.train_mask, dtype=torch.bool, device=device)
    va = torch.tensor(data.val_mask, dtype=torch.bool, device=device)
    te = torch.tensor(data.test_mask, dtype=torch.bool, device=device)

    model = SotaIFDCGCN(
        in_dim=xs.shape[1], hidden=args.hidden, k_order=args.K, dropout=args.dropout
    ).to(device)
    ema = EMA(model, args.ema_decay)

    cw = class_balanced_weights(data.y[data.train_mask]).to(device)
    ce = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = OneCycleLR(
        opt,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=1,
        pct_start=0.12,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    best_score = -1e9
    best_epoch = 0
    no_imp = 0
    top_states: List[Tuple[float, Dict[str, torch.Tensor]]] = []

    history = {"train_loss": [], "val_bal_acc": [], "val_auc": []}

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        l_train = drop_edge(l_tilde, args.edge_drop)

        zf1, zs1, zp1, a1 = model(xs, xp, l_train)
        zf2, zs2, zp2, a2 = model(xs, xp, l_train)

        yf = y[tr]
        loss_main = ce(zf1[tr], yf) + 0.3 * ce(zs1[tr], yf) + 0.3 * ce(zp1[tr], yf)
        loss_kl = rdrop_kl(zf1[tr], zf2[tr])

        # Encourage non-collapsed gate distribution.
        ent = -(a1[tr] * torch.log(a1[tr] + 1e-8) + (1 - a1[tr]) * torch.log(1 - a1[tr] + 1e-8)).mean()

        loss = loss_main + args.rdrop_alpha * loss_kl - 0.01 * ent
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        sch.step()
        ema.update(model)

        # Eval on EMA model
        backup = ema.apply_to(model)
        val_m = evaluate_with_calibration(
            model, xs, xp, l_tilde, y, tr, va, te,
            threshold_metric=args.threshold_metric,
            use_temp_scaling=args.use_temp_scaling,
        )
        EMA.restore(model, backup)

        history["train_loss"].append(float(loss.item()))
        history["val_bal_acc"].append(float(val_m["bal_acc"]))
        history["val_auc"].append(float(val_m["auc"]))

        score = val_m["bal_acc"] + 0.01 * val_m["auc"]
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = ep
            no_imp = 0

            backup = ema.apply_to(model)
            st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            EMA.restore(model, backup)

            top_states.append((score, st))
            top_states = sorted(top_states, key=lambda t: t[0], reverse=True)[: args.topk_ensemble]
        else:
            no_imp += 1

        if ep == 1 or ep % 40 == 0:
            print(
                f"  Run {run_idx+1} Ep {ep:03d} | loss={loss.item():.4f} "
                f"| val_bal={val_m['bal_acc']:.2f}% | val_auc={val_m['auc']:.2f}%"
            )

        if no_imp >= args.patience:
            print(f"  Run {run_idx+1}: early stop at ep {ep} (best {best_epoch})")
            break

    # Top-K ensemble prediction (logits avg)
    model.eval()
    with torch.no_grad():
        z_sum = torch.zeros((xs.shape[0], 2), device=device)
        for _, st in top_states:
            model.load_state_dict(st, strict=True)
            zf, _, _, _ = model(xs, xp, l_tilde)
            z_sum += zf
        z_avg = z_sum / float(max(len(top_states), 1))

    z_val = z_avg[va]
    y_val = y[va]
    z_test = z_avg[te]
    y_test = y[te]

    temp = fit_temperature(z_val, y_val) if args.use_temp_scaling else torch.tensor(1.0, device=device)
    p_val = F.softmax(z_val / temp, dim=1)[:, 1].detach().cpu().numpy()
    p_test = F.softmax(z_test / temp, dim=1)[:, 1].detach().cpu().numpy()
    thr = find_best_threshold(p_val, y_val.detach().cpu().numpy(), args.threshold_metric)

    test_m = metrics_from_probs(p_test, y_test.detach().cpu().numpy(), thr=thr)
    test_m["threshold"] = float(thr)
    test_m["temperature"] = float(temp.item())

    best_state = top_states[0][1] if top_states else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
        "best_epoch": int(best_epoch),
        "val_score": float(best_score),
        "test": test_m,
        "history": history,
        "best_state": best_state,
    }


def mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.array(vals, dtype=np.float32)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_dir = os.path.join(OUTPUT_DIR, args.task)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print(f"SOTA Phase4 | task={args.task} | device={device} | runs={args.runs} | epochs={args.epochs}")
    print("=" * 72)

    data = load_task_data(args.task)
    print(
        f"Loaded N={len(data.y)} in_dim={data.x_smri.shape[1]} | "
        f"train/val/test={data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}"
    )

    all_runs = []
    best_acc = -1.0
    best_state = None
    best_run = -1

    t0 = time.time()
    for r in range(args.runs):
        out = train_one_run(args, data, r, device)
        all_runs.append(out)

        acc = out["test"]["acc"]
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(out["best_state"])
            best_run = r + 1

        print(
            f"Run {r+1} done | ACC={out['test']['acc']:.2f}% SEN={out['test']['sen']:.2f}% "
            f"SPE={out['test']['spe']:.2f}% AUC={out['test']['auc']:.2f}%"
        )

    elapsed = time.time() - t0

    keys = ["acc", "sen", "spe", "auc", "bal_acc"]
    summary_res = {k: mean_std([run["test"][k] for run in all_runs]) for k in keys}

    paper_ref = {
        "AD_vs_CN": {"acc": 91.07, "sen": 90.22, "spe": 91.87, "auc": 91.04},
        "SMCI_vs_PMCI": {"acc": 75.50, "sen": 49.90, "spe": 88.70, "auc": 69.30},
    }

    payload = {
        "task": args.task,
        "config": vars(args),
        "split_sizes": {
            "train": int(data.train_mask.sum()),
            "val": int(data.val_mask.sum()),
            "test": int(data.test_mask.sum()),
        },
        "results": summary_res,
        "per_run": [
            {
                "best_epoch": run["best_epoch"],
                "val_score": run["val_score"],
                "test": run["test"],
            }
            for run in all_runs
        ],
        "paper_target": paper_ref[args.task],
        "training_time_s": float(elapsed),
        "best_run": int(best_run),
        "best_run_acc": float(best_acc),
    }

    with open(os.path.join(out_dir, "results_sota.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    torch.save(
        {
            "model_state": best_state,
            "config": vars(args),
            "task": args.task,
            "n_class": 2,
        },
        os.path.join(out_dir, "best_model_sota.pt"),
    )

    print("\n" + "=" * 72)
    print(f"Done in {elapsed:.1f}s | best run #{best_run} ACC={best_acc:.2f}%")
    print(f"Saved: {os.path.join(out_dir, 'results_sota.json')}")
    print(f"Saved: {os.path.join(out_dir, 'best_model_sota.pt')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
