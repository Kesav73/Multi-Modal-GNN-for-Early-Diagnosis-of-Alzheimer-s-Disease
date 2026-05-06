"""
Phase 4 (Advanced): Stronger IFDCGCN training recipe for higher accuracy.

This script is intentionally separate from existing trainers.
It adds high-impact technical practices:
1) Split-safe feature normalization (train-only statistics)
2) Correct task-local mask/label projection when phase3 task files keep global masks
3) Learnable gated late-fusion instead of fixed 0.5 averaging
4) Class-balanced focal loss for imbalance robustness
5) AdamW + OneCycleLR + gradient clipping + early stopping
6) EMA (Exponential Moving Average) model evaluation
7) Top-K checkpoint ensembling on validation metric
8) Optional test-time augmentation (feature jitter averaging)

Usage:
  python phase4_advanced.py --task AD_vs_CN
  python phase4_advanced.py --task SMCI_vs_PMCI
  python phase4_advanced.py --task AD_vs_CN --runs 7 --epochs 500
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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


PHASE3_DIR = "./outputs/phase3_outputs"
OUTPUT_DIR = "./outputs/phase4_outputs"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", default="AD_vs_CN", choices=["AD_vs_CN", "SMCI_vs_PMCI"])
	parser.add_argument("--epochs", type=int, default=400)
	parser.add_argument("--runs", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--lr", type=float, default=1.2e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-4)
	parser.add_argument("--hidden", type=int, default=48)
	parser.add_argument("--K", type=int, default=3)
	parser.add_argument("--dropout", type=float, default=0.45)
	parser.add_argument("--patience", type=int, default=80)
	parser.add_argument("--grad_clip", type=float, default=1.0)
	parser.add_argument("--ema_decay", type=float, default=0.995)
	parser.add_argument("--focal_gamma", type=float, default=1.5)
	parser.add_argument("--topk_ensemble", type=int, default=5)
	parser.add_argument("--dropedge", type=float, default=0.05)
	parser.add_argument("--tta", type=int, default=5,
						help="Number of test-time augmentation passes (feature jitter).")
	parser.add_argument("--tta_noise", type=float, default=0.005)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def build_scaled_laplacian(adj_norm: np.ndarray) -> torch.Tensor:
	n = adj_norm.shape[0]
	eye = np.eye(n, dtype=np.float32)
	lap = eye - adj_norm

	# Power iteration for lambda_max
	v = np.random.randn(n).astype(np.float32)
	lam = 1.0
	for _ in range(40):
		v = lap @ v
		lam = np.linalg.norm(v) + 1e-10
		v = v / lam

	lap_tilde = (2.0 / float(lam)) * lap - eye
	return torch.tensor(lap_tilde, dtype=torch.float32)


def drop_edge_laplacian(l_tilde: torch.Tensor, drop_rate: float) -> torch.Tensor:
	if drop_rate <= 0.0:
		return l_tilde
	keep = torch.bernoulli(torch.full_like(l_tilde, 1.0 - drop_rate))
	keep.fill_diagonal_(1.0)
	return l_tilde * keep


def standardize_train_only(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
	tr = x[train_mask]
	mu = tr.mean(axis=0, keepdims=True)
	sigma = tr.std(axis=0, keepdims=True)
	sigma[sigma < 1e-6] = 1.0
	return (x - mu) / sigma


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
	"""
	Robust loader for phase3 task artifacts.

	Some phase3 variants store task-local X/A but keep global-length y/masks.
	This function projects labels/masks to task-local indexing using task_mask.
	"""
	task_dir = os.path.join(PHASE3_DIR, task)
	x_pet = np.load(os.path.join(task_dir, "X_pet.npy")).astype(np.float32)
	x_smri = np.load(os.path.join(task_dir, "X_smri.npy")).astype(np.float32)
	adj = np.load(os.path.join(task_dir, "A_if_norm.npy")).astype(np.float32)

	y_global = np.load(os.path.join(task_dir, "binary_labels.npy"))
	tr_global = np.load(os.path.join(task_dir, "train_mask.npy"))
	va_global = np.load(os.path.join(task_dir, "val_mask.npy"))
	te_global = np.load(os.path.join(task_dir, "test_mask.npy"))

	n_local = x_pet.shape[0]
	if len(y_global) == n_local:
		y = y_global.astype(np.int64)
		tr = tr_global.astype(bool)
		va = va_global.astype(bool)
		te = te_global.astype(bool)
	else:
		task_mask = np.load(os.path.join(task_dir, "task_mask.npy")).astype(bool)
		if int(task_mask.sum()) != n_local:
			raise ValueError(
				f"Task mask sum ({task_mask.sum()}) does not match local graph nodes ({n_local})"
			)

		# Project global masks/labels into the task-local ordering used by X_*.
		y = y_global[task_mask].astype(np.int64)
		tr = tr_global[task_mask].astype(bool)
		va = va_global[task_mask].astype(bool)
		te = te_global[task_mask].astype(bool)

	valid = (y == 0) | (y == 1)
	if not valid.all():
		# For safety, remove any non-binary entries from all tensors.
		x_pet = x_pet[valid]
		x_smri = x_smri[valid]
		adj = adj[np.ix_(valid, valid)]
		tr = tr[valid]
		va = va[valid]
		te = te[valid]
		y = y[valid]

	return TaskData(
		x_pet=x_pet,
		x_smri=x_smri,
		adj=adj,
		y=y,
		train_mask=tr,
		val_mask=va,
		test_mask=te,
	)


class ChebConv(nn.Module):
	def __init__(self, in_features: int, out_features: int, k_order: int):
		super().__init__()
		self.k_order = k_order
		self.weight = nn.Parameter(torch.empty(k_order + 1, in_features, out_features))
		self.bias = nn.Parameter(torch.zeros(out_features))
		nn.init.xavier_uniform_(self.weight.view((k_order + 1) * in_features, out_features))

	def forward(self, x: torch.Tensor, lap_tilde: torch.Tensor) -> torch.Tensor:
		tx0 = x
		out = tx0 @ self.weight[0]

		if self.k_order >= 1:
			tx1 = lap_tilde @ x
			out = out + tx1 @ self.weight[1]
		else:
			tx1 = None

		for k in range(2, self.k_order + 1):
			tx2 = 2.0 * (lap_tilde @ tx1) - tx0
			out = out + tx2 @ self.weight[k]
			tx0, tx1 = tx1, tx2

		return out + self.bias


class CGCNBranch(nn.Module):
	def __init__(self, in_dim: int, hidden: int, n_class: int, k_order: int, dropout: float):
		super().__init__()
		self.conv1 = ChebConv(in_dim, hidden, k_order)
		self.bn1 = nn.BatchNorm1d(hidden)
		self.conv2 = ChebConv(hidden, hidden, k_order)
		self.bn2 = nn.BatchNorm1d(hidden)
		self.conv3 = ChebConv(hidden, n_class, k_order)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, lap_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = self.conv1(x, lap_tilde)
		h = self.bn1(h)
		h = F.elu(h)
		h = F.dropout(h, p=self.dropout, training=self.training)

		h2 = self.conv2(h, lap_tilde)
		h2 = self.bn2(h2)
		h2 = F.elu(h2)
		h2 = F.dropout(h2, p=self.dropout, training=self.training)

		# Residual hidden representation
		h_final = h + h2
		logits = self.conv3(h_final, lap_tilde)
		return logits, h_final


class GatedIFDCGCN(nn.Module):
	def __init__(self, in_dim: int, hidden: int, n_class: int, k_order: int, dropout: float):
		super().__init__()
		self.smri = CGCNBranch(in_dim, hidden, n_class, k_order, dropout)
		self.pet = CGCNBranch(in_dim, hidden, n_class, k_order, dropout)

		# Node-wise fusion gate: learns modality confidence per subject.
		self.gate_mlp = nn.Sequential(
			nn.Linear(hidden * 2, hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden, 1),
		)

	def forward(self, x_smri: torch.Tensor, x_pet: torch.Tensor, lap_tilde: torch.Tensor):
		z_smri, h_smri = self.smri(x_smri, lap_tilde)
		z_pet, h_pet = self.pet(x_pet, lap_tilde)

		p_smri = F.softmax(z_smri, dim=1)
		p_pet = F.softmax(z_pet, dim=1)

		gate_in = torch.cat([h_smri, h_pet], dim=1)
		alpha = torch.sigmoid(self.gate_mlp(gate_in))
		p_fused = alpha * p_smri + (1.0 - alpha) * p_pet

		return p_fused, p_smri, p_pet, alpha


class ClassBalancedFocalLoss(nn.Module):
	def __init__(self, labels_train: np.ndarray, beta: float = 0.999, gamma: float = 1.5):
		super().__init__()
		self.gamma = gamma

		n0 = int((labels_train == 0).sum())
		n1 = int((labels_train == 1).sum())
		counts = np.array([max(n0, 1), max(n1, 1)], dtype=np.float32)

		eff_num = 1.0 - np.power(beta, counts)
		weights = (1.0 - beta) / np.maximum(eff_num, 1e-12)
		weights = weights / weights.sum() * 2.0
		self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

	def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		probs = probs.clamp(min=1e-8, max=1.0)
		pt = probs[torch.arange(target.shape[0], device=target.device), target]
		alpha = self.class_weights[target]
		focal = -alpha * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
		return focal.mean()


class EMA:
	def __init__(self, model: nn.Module, decay: float = 0.995):
		self.decay = decay
		self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

	@torch.no_grad()
	def update(self, model: nn.Module) -> None:
		for k, v in model.state_dict().items():
			# EMA is only valid for floating tensors; copy integer/bool buffers directly.
			if torch.is_floating_point(self.shadow[k]):
				self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
			else:
				self.shadow[k].copy_(v.detach())

	def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
		backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
		model.load_state_dict(self.shadow, strict=True)
		return backup

	@staticmethod
	def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
		model.load_state_dict(backup, strict=True)


def compute_metrics(prob: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
	pred = prob.argmax(dim=1).detach().cpu().numpy()
	y_np = y_true.detach().cpu().numpy()
	p1 = prob[:, 1].detach().cpu().numpy()

	acc = 100.0 * accuracy_score(y_np, pred)
	cm = confusion_matrix(y_np, pred, labels=[0, 1])
	if cm.size == 4:
		tn, fp, fn, tp = cm.ravel()
	else:
		tn = fp = fn = tp = 0
	sen = 100.0 * tp / (tp + fn + 1e-8)
	spe = 100.0 * tn / (tn + fp + 1e-8)
	try:
		auc = 100.0 * roc_auc_score(y_np, p1)
	except Exception:
		auc = 0.0

	return {"acc": float(acc), "sen": float(sen), "spe": float(spe), "auc": float(auc)}


def evaluate_model(
	model: nn.Module,
	x_smri: torch.Tensor,
	x_pet: torch.Tensor,
	lap_tilde: torch.Tensor,
	y: torch.Tensor,
	mask: torch.Tensor,
	tta: int = 0,
	tta_noise: float = 0.0,
) -> Dict[str, float]:
	model.eval()
	with torch.no_grad():
		if tta <= 0:
			p_fused, p_smri, p_pet, _ = model(x_smri, x_pet, lap_tilde)
			p = p_fused[mask]
		else:
			agg = 0.0
			for _ in range(tta):
				xs = x_smri + torch.randn_like(x_smri) * tta_noise
				xp = x_pet + torch.randn_like(x_pet) * tta_noise
				p_fused, _, _, _ = model(xs, xp, lap_tilde)
				agg = agg + p_fused
			p = (agg / float(tta))[mask]

	return compute_metrics(p, y[mask])


def train_one_run(args: argparse.Namespace, data: TaskData, run_idx: int, device: torch.device):
	set_seed(args.seed + run_idx)

	# Split-safe feature standardization.
	x_pet = standardize_train_only(data.x_pet, data.train_mask)
	x_smri = standardize_train_only(data.x_smri, data.train_mask)

	x_pet_t = torch.tensor(x_pet, dtype=torch.float32, device=device)
	x_smri_t = torch.tensor(x_smri, dtype=torch.float32, device=device)
	lap_tilde = build_scaled_laplacian(data.adj).to(device)

	y_t = torch.tensor(data.y, dtype=torch.long, device=device)
	train_mask = torch.tensor(data.train_mask, dtype=torch.bool, device=device)
	val_mask = torch.tensor(data.val_mask, dtype=torch.bool, device=device)
	test_mask = torch.tensor(data.test_mask, dtype=torch.bool, device=device)

	model = GatedIFDCGCN(
		in_dim=x_smri_t.shape[1],
		hidden=args.hidden,
		n_class=2,
		k_order=args.K,
		dropout=args.dropout,
	).to(device)
	ema = EMA(model, decay=args.ema_decay)

	cb_focal = ClassBalancedFocalLoss(data.y[data.train_mask], gamma=args.focal_gamma)
	cb_focal.to(device)

	optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = OneCycleLR(
		optimizer,
		max_lr=args.lr,
		epochs=args.epochs,
		steps_per_epoch=1,
		pct_start=0.15,
		anneal_strategy="cos",
		div_factor=20.0,
		final_div_factor=1000.0,
	)

	best_score = -1.0
	best_epoch = 0
	no_improve = 0
	top_states: List[Tuple[float, Dict[str, torch.Tensor]]] = []
	history = {"train_loss": [], "val_acc": [], "val_auc": []}

	for epoch in range(1, args.epochs + 1):
		model.train()
		optimizer.zero_grad(set_to_none=True)

		lap_train = drop_edge_laplacian(lap_tilde, args.dropedge)
		p_fused, p_smri, p_pet, alpha = model(x_smri_t, x_pet_t, lap_train)

		pf = p_fused[train_mask]
		ps = p_smri[train_mask]
		pp = p_pet[train_mask]
		yt = y_t[train_mask]

		# Main objective + branch supervision + gate entropy regularizer.
		loss_f = cb_focal(pf, yt)
		loss_s = cb_focal(ps, yt)
		loss_p = cb_focal(pp, yt)
		alpha_tr = alpha[train_mask]
		gate_entropy = -(alpha_tr * torch.log(alpha_tr + 1e-8) +
						 (1.0 - alpha_tr) * torch.log(1.0 - alpha_tr + 1e-8)).mean()

		loss = loss_f + 0.35 * loss_s + 0.35 * loss_p - 0.01 * gate_entropy
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
		optimizer.step()
		scheduler.step()
		ema.update(model)

		# Validate with EMA weights.
		backup = ema.apply_to(model)
		val_metrics = evaluate_model(model, x_smri_t, x_pet_t, lap_tilde, y_t, val_mask)
		EMA.restore(model, backup)

		history["train_loss"].append(float(loss.item()))
		history["val_acc"].append(val_metrics["acc"])
		history["val_auc"].append(val_metrics["auc"])

		# Prioritize AUC then ACC for model selection.
		score = val_metrics["auc"] + 0.01 * val_metrics["acc"]
		if score > best_score + 1e-6:
			best_score = score
			best_epoch = epoch
			no_improve = 0

			backup = ema.apply_to(model)
			state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			EMA.restore(model, backup)

			top_states.append((score, state))
			top_states = sorted(top_states, key=lambda x: x[0], reverse=True)[: args.topk_ensemble]
		else:
			no_improve += 1

		if epoch == 1 or epoch % 40 == 0:
			print(
				f"  Run {run_idx + 1} Ep {epoch:03d} | "
				f"loss={loss.item():.4f} | val_acc={val_metrics['acc']:.2f}% | val_auc={val_metrics['auc']:.2f}%"
			)

		if no_improve >= args.patience:
			print(f"  Run {run_idx + 1}: early stop at epoch {epoch} (best epoch {best_epoch})")
			break

	# Ensemble top validation checkpoints.
	model.eval()
	with torch.no_grad():
		fused_sum = torch.zeros((x_smri_t.shape[0], 2), device=device)
		for _, st in top_states:
			model.load_state_dict(st, strict=True)
			p_fused, _, _, _ = model(x_smri_t, x_pet_t, lap_tilde)
			fused_sum += p_fused
		fused_prob = fused_sum / float(max(len(top_states), 1))

	test_metrics = compute_metrics(fused_prob[test_mask], y_t[test_mask])

	# TTA on best single checkpoint for a robust final estimate.
	best_state = top_states[0][1] if top_states else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
	model.load_state_dict(best_state, strict=True)
	tta_metrics = evaluate_model(
		model, x_smri_t, x_pet_t, lap_tilde, y_t, test_mask,
		tta=args.tta, tta_noise=args.tta_noise,
	)

	return {
		"best_epoch": int(best_epoch),
		"val_score": float(best_score),
		"test_ensemble": test_metrics,
		"test_tta": tta_metrics,
		"history": history,
		"best_state": best_state,
	}


def mean_std(values: List[float]) -> Dict[str, float]:
	arr = np.array(values, dtype=np.float32)
	return {"mean": float(arr.mean()), "std": float(arr.std())}


def main() -> None:
	args = parse_args()
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	out_dir = os.path.join(OUTPUT_DIR, args.task)
	os.makedirs(out_dir, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("=" * 72)
	print(f"Task: {args.task} | Device: {device} | Runs: {args.runs} | Epochs: {args.epochs}")
	print("Advanced recipe: gated fusion + CB focal + AdamW/OneCycle + EMA + top-k ensemble")
	print("=" * 72)

	data = load_task_data(args.task)
	print(
		f"Loaded graph: N={len(data.y)}, in_dim={data.x_smri.shape[1]}, "
		f"train/val/test={data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}"
	)
	print(
		f"Train labels: class0={(data.y[data.train_mask] == 0).sum()} "
		f"class1={(data.y[data.train_mask] == 1).sum()}"
	)

	all_runs = []
	t0 = time.time()
	best_global_acc = -1.0
	best_global_state = None
	best_global_run = -1

	for r in range(args.runs):
		run_out = train_one_run(args, data, r, device)
		all_runs.append(run_out)

		acc = run_out["test_ensemble"]["acc"]
		if acc > best_global_acc:
			best_global_acc = acc
			best_global_state = copy.deepcopy(run_out["best_state"])
			best_global_run = r + 1

		print(
			f"Run {r + 1} done | Ensemble ACC={run_out['test_ensemble']['acc']:.2f}% "
			f"AUC={run_out['test_ensemble']['auc']:.2f}% | "
			f"TTA ACC={run_out['test_tta']['acc']:.2f}%"
		)

	elapsed = time.time() - t0

	# Aggregate summaries.
	keys = ["acc", "sen", "spe", "auc"]
	ensemble_summary = {
		k: mean_std([run["test_ensemble"][k] for run in all_runs]) for k in keys
	}
	tta_summary = {
		k: mean_std([run["test_tta"][k] for run in all_runs]) for k in keys
	}

	paper_ref = {
		"AD_vs_CN": {"acc": 91.07, "sen": 90.22, "spe": 91.87, "auc": 91.04},
		"SMCI_vs_PMCI": {"acc": 75.50, "sen": 49.90, "spe": 88.70, "auc": 69.30},
	}

	summary = {
		"task": args.task,
		"config": vars(args),
		"n_nodes": int(len(data.y)),
		"n_features": int(data.x_smri.shape[1]),
		"split_sizes": {
			"train": int(data.train_mask.sum()),
			"val": int(data.val_mask.sum()),
			"test": int(data.test_mask.sum()),
		},
		"results_ensemble": ensemble_summary,
		"results_tta": tta_summary,
		"per_run": [
			{
				"best_epoch": run["best_epoch"],
				"val_score": run["val_score"],
				"test_ensemble": run["test_ensemble"],
				"test_tta": run["test_tta"],
			}
			for run in all_runs
		],
		"paper_target": paper_ref[args.task],
		"training_time_s": float(elapsed),
		"best_run": int(best_global_run),
		"best_run_acc": float(best_global_acc),
	}

	with open(os.path.join(out_dir, "results_advanced.json"), "w", encoding="utf-8") as f:
		json.dump(summary, f, indent=2)

	torch.save(
		{
			"model_state": best_global_state,
			"config": vars(args),
			"in_dim": int(data.x_smri.shape[1]),
			"n_class": 2,
			"task": args.task,
		},
		os.path.join(out_dir, "best_model_advanced.pt"),
	)

	print("\n" + "=" * 72)
	print(f"Advanced training complete in {elapsed:.1f}s")
	print(f"Best run #{best_global_run} ensemble ACC: {best_global_acc:.2f}%")
	print("Saved:")
	print(f"  {os.path.join(out_dir, 'results_advanced.json')}")
	print(f"  {os.path.join(out_dir, 'best_model_advanced.pt')}")
	print("=" * 72)


if __name__ == "__main__":
	main()

