"""
Phase 3: Graph Construction  (updated with MMSE scores)
Paper: "Multi-modal graph neural network for early diagnosis of Alzheimer's disease"

Implements Equations 6-13:
  Eq.6  : S(v,u) = exp(-ρ(F_v,F_u)^2 / 2σ^2)   Gaussian kernel on correlation distance
  Eq.7  : A(v,u) = S(v,u) × (r_G + r_M)         Phenotypic weighting
  Eq.8  : r_G = 1 if same gender, else 0
  Eq.10 : r_M = 1 if |MMSE_v - MMSE_u| <= 1, else 0
  Eq.11 : A_im = A_s ⊙ A_f                       Hadamard of sMRI and PET adj
  Eq.12 : S(F_vc, F_uc) on concatenated features  Fusion adjacency
  Eq.13 : A_if = A_im ⊙ A_fm                     Final integrated adjacency

Key facts about your data:
  - 627 subjects with MMSE (2 subjects dropped — had missing PET SUVR in phase2)
  - 5 subjects have MMSCORE = -1 (missing) → imputed with per-label mean
  - Labels: CN=196, SMCI=176, AD=136, PMCI=119
  - Phenotypic: gender (Male/Female) + MMSCORE (0-30)
  - Subject IDs already in '002_S_4171' format (matches phase2)

Inputs  (outputs/phase2_outputs/):
  pet_individual_features.npy, smri_gm_individual_features.npy
  labels_int.npy, labels.npy, nc_mask.npy, subject_ids.npy

New input:
  phenotypic_with_MMSCORE.csv  (subject, label, label_int, age, gender, MMSCORE)

Outputs (outputs/phase3_outputs/):
  X_pet.npy, X_smri.npy             Node feature matrices (normalized)
  A_pet.npy, A_smri.npy             Per-modality similarity adjacency
  A_pet_pheno.npy, A_smri_pheno.npy Phenotypic-weighted adjacency
  A_im.npy                          Integrated adjacency (Eq.11)
  A_fm.npy                          Fusion adjacency (Eq.12)
  A_if.npy, A_if_norm.npy           Final adjacency raw + normalized (Eq.13) ← GNN input
  phenotypic_aligned.csv            Final aligned phenotypic info (N subjects)
  AD_vs_CN/   {train,val,test,binary_labels,task}_mask.npy
  SMCI_vs_PMCI/ {train,val,test,binary_labels,task}_mask.npy
  split_indices.json
"""

import numpy as np
import pandas as pd
import os
import json
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
PHASE2_DIR    = "./outputs/phase2_outputs"
PHENOTYPIC_CSV = "./data/phenotypic/phenotypic_with_MMSCORE.csv"
OUTPUT_DIR    = "./outputs/phase3_outputs"

SMRI_MODALITY = "gm"       # 'gm', 'wm', or 'gmwm'
SIGMA         = None        # None = auto (median heuristic)
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
RANDOM_SEED   = 42

TASK_CONFIGS = {
    "AD_vs_CN": {
        "keep_labels": ["AD", "CN"],
        "binary_map":  {"CN": 0, "AD": 1}
    },
    "SMCI_vs_PMCI": {
        "keep_labels": ["SMCI", "PMCI"],
        "binary_map":  {"SMCI": 0, "PMCI": 1}
    }
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  STEP 1: Load & Align all data sources
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading & Aligning Data")
print("=" * 65)

# Load phase2 features
X_pet_raw  = np.load(os.path.join(PHASE2_DIR, "pet_individual_features.npy"))
X_smri_raw = np.load(os.path.join(PHASE2_DIR, f"smri_{SMRI_MODALITY}_individual_features.npy"))
labels_str = np.load(os.path.join(PHASE2_DIR, "labels.npy"), allow_pickle=True)
subj_ids   = np.load(os.path.join(PHASE2_DIR, "subject_ids.npy"), allow_pickle=True)

# ──  CRITICAL: Trim to match actual feature array size ──
# Phase2 feature arrays are authoritative (602 subjects with complete features)
N_phase2_actual = X_pet_raw.shape[0]
if len(subj_ids) != N_phase2_actual:
    print(f"[WARNING] subject_ids.npy has {len(subj_ids)}, but features have {N_phase2_actual} subjects")
    print(f"  Trimming subject_ids to match features")
    subj_ids = subj_ids[:N_phase2_actual]
    labels_str = labels_str[:N_phase2_actual]

print(f"Phase2 subjects : {len(subj_ids)} (features: {X_pet_raw.shape})")
print(f"PET features    : {X_pet_raw.shape}")
print(f"sMRI features   : {X_smri_raw.shape}")

# Load new phenotypic CSV with MMSE
pheno = pd.read_csv(PHENOTYPIC_CSV)
print(f"\nPhenotypic CSV  : {len(pheno)} subjects")
print(f"Columns         : {pheno.columns.tolist()}")

# ── Handle MMSCORE = -1 (missing) ────────────────────────────────
# Impute with per-label mean MMSE (better than global mean)
bad_mmse = pheno["MMSCORE"] < 0
print(f"\nMMSCORE = -1 found in {bad_mmse.sum()} subjects → imputing with per-label mean")
label_mmse_means = pheno[pheno["MMSCORE"] >= 0].groupby("label")["MMSCORE"].mean()
print("  Per-label MMSE means:", label_mmse_means.to_dict())
for idx in pheno[bad_mmse].index:
    lbl = pheno.loc[idx, "label"]
    imputed = label_mmse_means[lbl]
    print(f"  {pheno.loc[idx,'subject']} ({lbl}): -1 → {imputed:.1f}")
    pheno.loc[idx, "MMSCORE"] = imputed

# ── Align: keep only subjects present in BOTH phase2 AND phenotypic ──
pheno_subjects = set(pheno["subject"].values)
phase2_subjects = set(subj_ids)
common_subjects = sorted(phase2_subjects & pheno_subjects)
dropped = phase2_subjects - pheno_subjects
print(f"\nPhase2 subjects : {len(phase2_subjects)}")
print(f"Pheno subjects  : {len(pheno_subjects)}")
print(f"Common          : {len(common_subjects)}")
if dropped:
    print(f"Dropped (no MMSE): {sorted(dropped)}")

# Build aligned index into phase2 arrays
phase2_idx = {s: i for i, s in enumerate(subj_ids)}
keep_idx   = np.array([phase2_idx[s] for s in common_subjects])

X_pet_raw  = X_pet_raw[keep_idx]
X_smri_raw = X_smri_raw[keep_idx]
labels_str = labels_str[keep_idx]
subj_ids   = np.array(common_subjects)
N          = len(subj_ids)

# Align phenotypic to same order
pheno = pheno.set_index("subject").loc[common_subjects].reset_index()
assert list(pheno["subject"]) == list(subj_ids), "Alignment error!"

genders    = np.array(pheno["gender"].values, dtype=str)   # force numpy string array
mmscores   = pheno["MMSCORE"].values.astype(np.float32)
labels_int = pheno["label_int"].values.astype(np.int64)

print(f"\nFinal N = {N}")
print(f"Labels  : { {k: int((labels_str==k).sum()) for k in ['CN','AD','SMCI','PMCI']} }")
print(f"MMSE    : min={mmscores.min():.1f}, max={mmscores.max():.1f}, "
      f"mean={mmscores.mean():.2f}")

# Save final aligned phenotypic
pheno.to_csv(os.path.join(OUTPUT_DIR, "phenotypic_aligned.csv"), index=False)
np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), subj_ids)
np.save(os.path.join(OUTPUT_DIR, "labels_str.npy"),  labels_str)
np.save(os.path.join(OUTPUT_DIR, "labels_int.npy"),  labels_int)
nc_mask = labels_str == "CN"
np.save(os.path.join(OUTPUT_DIR, "nc_mask.npy"), nc_mask)
print(f"\nSaved aligned phenotypic_aligned.csv, subject_ids, labels, nc_mask")


# ─────────────────────────────────────────────────────────────────
#  STEP 2: Normalize Node Features
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Normalizing Node Features")
print("=" * 65)

scaler_pet  = StandardScaler()
scaler_smri = StandardScaler()

X_pet  = scaler_pet.fit_transform(X_pet_raw).astype(np.float32)
X_smri = scaler_smri.fit_transform(X_smri_raw).astype(np.float32)

print(f"PET  normalized : mean={X_pet.mean():.4f}  std={X_pet.std():.4f}")
print(f"sMRI normalized : mean={X_smri.mean():.4f}  std={X_smri.std():.4f}")

np.save(os.path.join(OUTPUT_DIR, "X_pet.npy"),  X_pet)
np.save(os.path.join(OUTPUT_DIR, "X_smri.npy"), X_smri)
print("Saved X_pet.npy, X_smri.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 3: Similarity Adjacency (Eq. 6)
#  S(v,u) = exp(-ρ(F_v, F_u)^2 / 2σ^2)
#  ρ = correlation distance (1 - Pearson correlation)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Similarity Adjacency Matrices (Eq. 6)")
print("=" * 65)

def similarity_adjacency(X, sigma=None, name=""):
    """
    Gaussian kernel on pairwise correlation distances.
    sigma=None → median heuristic (standard RBF bandwidth selection).
    """
    print(f"  [{name}] Computing {X.shape[0]}x{X.shape[0]} pairwise distances ...")
    dist = squareform(pdist(X, metric="correlation")).astype(np.float32)
    dist = np.clip(dist, 0, None)   # numerical safety

    if sigma is None:
        upper = dist[np.triu_indices_from(dist, k=1)]
        sigma = float(np.sqrt(np.median(upper ** 2) / 2.0))
        print(f"  [{name}] Auto σ (median heuristic) = {sigma:.6f}")

    S = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    off = S[~np.eye(N, dtype=bool)]
    print(f"  [{name}] Weight range : [{S.min():.4f}, {S.max():.4f}]  "
          f"mean off-diag = {off.mean():.4f}")
    return S, sigma


A_pet,  sigma_pet  = similarity_adjacency(X_pet,  sigma=SIGMA, name="PET")
A_smri, sigma_smri = similarity_adjacency(X_smri, sigma=SIGMA, name="sMRI")

np.save(os.path.join(OUTPUT_DIR, "A_pet.npy"),  A_pet)
np.save(os.path.join(OUTPUT_DIR, "A_smri.npy"), A_smri)
print("Saved A_pet.npy, A_smri.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 4: Phenotypic Edge Weights (Eq. 7-10)
#
#  A(v,u) = S(v,u) × (r_G(v,u) + r_M(v,u))
#
#  r_G = 1 if same gender (Eq.8)
#  r_M = 1 if |MMSE_v - MMSE_u| ≤ 1 (Eq.10)
#
#  Matching on BOTH → weight × 2 (edge doubled, per paper)
#  Matching on ONE  → weight × 1
#  Matching on NONE → weight × 0 (edge suppressed)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Phenotypic Edge Weights — Gender + MMSE (Eq. 7-10)")
print("=" * 65)

# Eq.8: r_G — same gender indicator matrix [N, N]
r_G = (genders[:, None] == genders[None, :]).astype(np.float32)

# Eq.10: r_M — MMSE proximity indicator matrix [N, N]
mmse_diff = np.abs(mmscores[:, None] - mmscores[None, :])   # [N, N]
r_M = (mmse_diff <= 1).astype(np.float32)

# Combined phenotypic weight: values in {0, 1, 2}
pheno_weight = r_G + r_M   # [N, N]

# Stats
same_gender_pairs = int(r_G.sum()) // 2
close_mmse_pairs  = int(r_M.sum()) // 2
both_match_pairs  = int(((r_G == 1) & (r_M == 1)).sum()) // 2
print(f"  Same-gender pairs      : {same_gender_pairs}")
print(f"  Close-MMSE pairs (≤1)  : {close_mmse_pairs}")
print(f"  Both match (weight=2)  : {both_match_pairs}")
print(f"  No match (suppressed)  : {int(((r_G == 0) & (r_M == 0)).sum()) // 2}")

# Eq.7: Apply phenotypic weights to similarity matrices
A_pet_pheno  = A_pet  * pheno_weight   # [N, N]
A_smri_pheno = A_smri * pheno_weight   # [N, N]
np.fill_diagonal(A_pet_pheno,  1.0)
np.fill_diagonal(A_smri_pheno, 1.0)

print(f"\n  A_pet_pheno  : range=[{A_pet_pheno.min():.4f}, {A_pet_pheno.max():.4f}]  "
      f"density={(A_pet_pheno[~np.eye(N,dtype=bool)]>0).mean():.3f}")
print(f"  A_smri_pheno : range=[{A_smri_pheno.min():.4f}, {A_smri_pheno.max():.4f}]  "
      f"density={(A_smri_pheno[~np.eye(N,dtype=bool)]>0).mean():.3f}")

np.save(os.path.join(OUTPUT_DIR, "A_pet_pheno.npy"),  A_pet_pheno.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "A_smri_pheno.npy"), A_smri_pheno.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "r_G.npy"), r_G.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "r_M.npy"), r_M.astype(np.float32))
print("Saved A_pet_pheno.npy, A_smri_pheno.npy, r_G.npy, r_M.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 5: Integrated Adjacency A_im (Eq. 11)
#  A_im = A_s ⊙ A_f   (sMRI ⊙ PET)
#  Strong edge only where BOTH modalities agree on similarity
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Integrated Adjacency A_im (Eq. 11)")
print("=" * 65)

A_im = A_smri_pheno * A_pet_pheno    # Hadamard product [N, N]
np.fill_diagonal(A_im, 1.0)

print(f"  A_im range   : [{A_im.min():.4f}, {A_im.max():.4f}]")
print(f"  A_im density : {(A_im[~np.eye(N,dtype=bool)]>0).mean():.4f}")

np.save(os.path.join(OUTPUT_DIR, "A_im.npy"), A_im.astype(np.float32))
print("Saved A_im.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 6: Fusion Adjacency A_fm (Eq. 12)
#  Concatenate sMRI + PET features per subject → recompute similarity
#  Captures cross-modal correlations in a shared feature space
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Fusion Adjacency A_fm (Eq. 12)")
print("=" * 65)

X_concat = np.concatenate([X_smri, X_pet], axis=1)   # [N, 13572]
print(f"  Concatenated features : {X_concat.shape}")

sigma_fm = (sigma_pet + sigma_smri) / 2.0
print(f"  σ = (σ_PET + σ_sMRI)/2 = {sigma_fm:.6f}")

A_fm, _ = similarity_adjacency(X_concat, sigma=sigma_fm, name="Fusion")
A_fm    = A_fm * pheno_weight        # apply same phenotypic weights
np.fill_diagonal(A_fm, 1.0)

print(f"  A_fm range   : [{A_fm.min():.4f}, {A_fm.max():.4f}]")
print(f"  A_fm density : {(A_fm[~np.eye(N,dtype=bool)]>0).mean():.4f}")

np.save(os.path.join(OUTPUT_DIR, "A_fm.npy"), A_fm.astype(np.float32))
print("Saved A_fm.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 7: Final Integrated Fusion Adjacency A_if (Eq. 13)
#  A_if = A_im ⊙ A_fm
#  This is the SHARED adjacency matrix fed into BOTH GNN branches
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Final Adjacency A_if (Eq. 13)  ← GNN input")
print("=" * 65)

A_if = A_im * A_fm
np.fill_diagonal(A_if, 1.0)

print(f"  A_if range   : [{A_if.min():.6f}, {A_if.max():.6f}]")
print(f"  A_if density : {(A_if[~np.eye(N,dtype=bool)]>0).mean():.4f}")

# Symmetric normalization:  D^{-1/2} A D^{-1/2}
# Required for Chebyshev GCN (used to compute normalized Laplacian)
def sym_normalize(A):
    d         = A.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    A_norm    = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return A_norm.astype(np.float32)

A_if_norm = sym_normalize(A_if)

print(f"  A_if_norm range: [{A_if_norm.min():.6f}, {A_if_norm.max():.6f}]")
print(f"  Symmetric      : {np.allclose(A_if_norm, A_if_norm.T, atol=1e-5)}")

np.save(os.path.join(OUTPUT_DIR, "A_if.npy"),      A_if.astype(np.float32))
np.save(os.path.join(OUTPUT_DIR, "A_if_norm.npy"), A_if_norm)
print("Saved A_if.npy, A_if_norm.npy")


# ─────────────────────────────────────────────────────────────────
#  STEP 8: Train / Val / Test Splits  (70 / 15 / 15)
#  Stratified by class label, per task
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: Train / Val / Test Splits  (70/15/15, stratified)")
print("=" * 65)

all_splits = {}

for task_name, cfg in TASK_CONFIGS.items():
    print(f"\n  ── Task: {task_name} ──")
    keep = cfg["keep_labels"]
    bmap = cfg["binary_map"]

    task_mask     = np.array([l in keep for l in labels_str])
    task_idx      = np.where(task_mask)[0]
    task_labels_s = labels_str[task_idx]
    task_labels_b = np.array([bmap[l] for l in task_labels_s])

    print(f"  Subjects: {len(task_idx)}  "
          f"{ {k: int((task_labels_s==k).sum()) for k in keep} }")

    # Split 1: train  vs  val+test
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED
    )
    train_rel, valtest_rel = next(sss1.split(task_idx, task_labels_b))

    # Split 2: val  vs  test  (within the val+test pool)
    val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac,
        random_state=RANDOM_SEED
    )
    val_rel, test_rel = next(sss2.split(
        valtest_rel, task_labels_b[valtest_rel]
    ))

    train_idx = task_idx[train_rel]
    val_idx   = task_idx[valtest_rel[val_rel]]
    test_idx  = task_idx[valtest_rel[test_rel]]

    # Boolean masks over full N subjects
    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    # Binary labels (-1 for subjects not in this task)
    binary_labels = np.full(N, -1, dtype=np.int64)
    binary_labels[task_idx] = task_labels_b

    print(f"  Train : {train_mask.sum()}  "
          f"{ {k: int((labels_str[train_idx]==k).sum()) for k in keep} }")
    print(f"  Val   : {val_mask.sum()}  "
          f"{ {k: int((labels_str[val_idx]==k).sum()) for k in keep} }")
    print(f"  Test  : {test_mask.sum()}  "
          f"{ {k: int((labels_str[test_idx]==k).sum()) for k in keep} }")

    task_dir = os.path.join(OUTPUT_DIR, task_name)
    os.makedirs(task_dir, exist_ok=True)
    np.save(os.path.join(task_dir, "train_mask.npy"),    train_mask)
    np.save(os.path.join(task_dir, "val_mask.npy"),      val_mask)
    np.save(os.path.join(task_dir, "test_mask.npy"),     test_mask)
    np.save(os.path.join(task_dir, "binary_labels.npy"), binary_labels)
    np.save(os.path.join(task_dir, "task_mask.npy"),     task_mask)

    all_splits[task_name] = {
        "n_train": int(train_mask.sum()),
        "n_val":   int(val_mask.sum()),
        "n_test":  int(test_mask.sum()),
        "train_idx": train_idx.tolist(),
        "val_idx":   val_idx.tolist(),
        "test_idx":  test_idx.tolist(),
    }
    print(f"  Saved to {task_dir}/")

with open(os.path.join(OUTPUT_DIR, "split_indices.json"), "w") as f:
    json.dump(all_splits, f, indent=2)
print("\n  Saved split_indices.json")


# ─────────────────────────────────────────────────────────────────
#  STEP 9: Graph Statistics & Validation
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 9: Graph Statistics & Validation")
print("=" * 65)

def graph_stats(A, name):
    A_nd = A.copy()
    np.fill_diagonal(A_nd, 0)
    deg    = (A_nd > 0).sum(axis=1)
    n_comp, _ = connected_components(csr_matrix(A_nd > 0))
    sym    = np.allclose(A, A.T, atol=1e-5)
    has_nan = np.isnan(A).any()
    has_inf = np.isinf(A).any()
    status = "✓" if (n_comp == 1 and sym and not has_nan and not has_inf) else "⚠"
    print(f"\n  {status} [{name}]")
    print(f"    Density    : {(A_nd > 0).mean():.4f}  "
          f"({int((A_nd > 0).sum())//2} edges)")
    print(f"    Degree     : min={deg.min()}  max={deg.max()}  "
          f"mean={deg.mean():.1f}")
    print(f"    Weight     : [{A.min():.5f}, {A.max():.5f}]")
    print(f"    Symmetric  : {sym}")
    print(f"    Connected  : {n_comp == 1}  ({n_comp} component(s))")
    if has_nan or has_inf:
        print(f"    !! NaN={np.isnan(A).sum()}  Inf={np.isinf(A).sum()}")

graph_stats(A_pet,      "A_pet       (PET similarity)")
graph_stats(A_smri,     "A_smri      (sMRI similarity)")
graph_stats(A_im,       "A_im        (Eq.11  sMRI⊙PET)")
graph_stats(A_fm,       "A_fm        (Eq.12  fusion)")
graph_stats(A_if,       "A_if        (Eq.13  final)")
graph_stats(A_if_norm,  "A_if_norm   (normalized  ← GNN)")

# Per-class edge weight analysis — should see higher weights between same-class subjects
print("\n  Mean edge weight between same-class vs different-class pairs (A_if):")
for lbl in ["CN", "AD", "SMCI", "PMCI"]:
    m = labels_str == lbl
    if not m.any(): continue
    same_class  = A_if[np.ix_(m, m)]
    other_class = A_if[np.ix_(m, ~m)]
    np.fill_diagonal(same_class, 0)
    print(f"    {lbl:5s}: intra={same_class.mean():.5f}  "
          f"inter={other_class.mean():.5f}  "
          f"ratio={same_class.mean()/max(other_class.mean(),1e-10):.2f}x")


# ─────────────────────────────────────────────────────────────────
#  STEP 10: Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 3 COMPLETE — Output files")
print("=" * 65)

files = [
    ("X_pet.npy",                        f"[{N}, 6786]    PET node features (normalized)"),
    ("X_smri.npy",                       f"[{N}, 6786]    sMRI node features (normalized)"),
    ("A_pet.npy",                        f"[{N},{N}]  PET similarity adjacency (Eq.6)"),
    ("A_smri.npy",                       f"[{N},{N}]  sMRI similarity adjacency (Eq.6)"),
    ("A_pet_pheno.npy",                  f"[{N},{N}]  PET + gender + MMSE (Eq.7)"),
    ("A_smri_pheno.npy",                 f"[{N},{N}]  sMRI + gender + MMSE (Eq.7)"),
    ("r_G.npy",                          f"[{N},{N}]  Gender indicator matrix (Eq.8)"),
    ("r_M.npy",                          f"[{N},{N}]  MMSE indicator matrix (Eq.10)"),
    ("A_im.npy",                         f"[{N},{N}]  Integrated adjacency (Eq.11)"),
    ("A_fm.npy",                         f"[{N},{N}]  Fusion adjacency (Eq.12)"),
    ("A_if.npy",                         f"[{N},{N}]  Final adjacency (Eq.13)"),
    ("A_if_norm.npy",                    f"[{N},{N}]  Normalized A_if  ← GNN input"),
    ("phenotypic_aligned.csv",           f"[{N}]       Final phenotypic (aligned order)"),
    ("split_indices.json",               "Train/val/test indices for both tasks"),
    ("AD_vs_CN/binary_labels.npy",       f"[{N}]       0=CN, 1=AD, -1=other"),
    ("AD_vs_CN/train_mask.npy",          f"[{N}]       boolean"),
    ("AD_vs_CN/val_mask.npy",            f"[{N}]       boolean"),
    ("AD_vs_CN/test_mask.npy",           f"[{N}]       boolean"),
    ("SMCI_vs_PMCI/binary_labels.npy",   f"[{N}]       0=SMCI, 1=PMCI, -1=other"),
    ("SMCI_vs_PMCI/train_mask.npy",      f"[{N}]       boolean"),
    ("SMCI_vs_PMCI/val_mask.npy",        f"[{N}]       boolean"),
    ("SMCI_vs_PMCI/test_mask.npy",       f"[{N}]       boolean"),
]

for fname, desc in files:
    path = os.path.join(OUTPUT_DIR, fname)
    e = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {e}  {fname:<45} {desc}")

print(f"\nAll outputs → {OUTPUT_DIR}")
print("\n" + "=" * 65)
print("NEXT: Phase 4 — Chebyshev GCN Model (IFDCGCN)")
print("  GNN inputs:")
print("  • X_pet.npy + X_smri.npy    (node feature matrices)")
print("  • A_if_norm.npy             (shared normalized adjacency)")
print("  • AD_vs_CN/                 (labels + masks for task 1)")
print("  • SMCI_vs_PMCI/             (labels + masks for task 2)")
print("=" * 65)