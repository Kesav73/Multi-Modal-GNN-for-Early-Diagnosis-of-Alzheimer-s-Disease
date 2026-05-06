"""
Phase 2: Brain Network Construction & Individual Feature Extraction
Paper: "Multi-modal graph neural network for early diagnosis of Alzheimer's disease"
ADAPTED FOR MULTI-DIMENSIONAL ROI FEATURES (4 statistics per ROI)

Dataset layout (your structure):
  features/
  ├── sMRI/<subject_id>/gm_features.npy     [116, 4]  <- mean, std, p25, p75
  ├── sMRI/<subject_id>/wm_features.npy     [116, 4]  <- mean, std, p25, p75
  ├── sMRI/<subject_id>/csf_features.npy    [116, 4]  <- mean, std, p25, p75
  ├── PET/<subject_id>/suvr_stats.npy       [116, 4]  <- mean, std, p25, p75
  └── phenotypic.csv  (PTID, Label, AGE, PTGENDER)

Phenotypic labels: CN, AD, SMCI, PMCI
Subject ID mapping: features use '002_S_4171', phenotypic uses '002S4171'

Key change: Distances now computed in 4-dimensional feature space using Euclidean norms
E(i,j) = ||delta_i - delta_j||_2 / s_p(i,j)  instead of scalar differences

Outputs (saved to OUTPUT_DIR):
  pet_individual_features.npy       [N, 6786]
  smri_gm_individual_features.npy   [N, 6786]
  smri_wm_individual_features.npy   [N, 6786]
  smri_gmwm_individual_features.npy [N, 6786]
  pet_B_matrices.npy                [N, 116, 116]
  smri_gm_B_matrices.npy            [N, 116, 116]
  smri_wm_B_matrices.npy            [N, 116, 116]
  M_NC_pet.npy / M_NC_gm.npy / M_NC_wm.npy
  subject_ids.npy / labels.npy / labels_int.npy / nc_mask.npy
  phenotypic_aligned.csv
"""

import numpy as np
import pandas as pd
import os
import json

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
BASE_DIR       = "./features"
SMRI_DIR       = os.path.join(BASE_DIR, "sMRI")
PET_DIR        = os.path.join(BASE_DIR, "PET")
PHENOTYPIC_CSV = os.path.join(BASE_DIR, "phenotypic.csv")
OUTPUT_DIR     = os.path.join(BASE_DIR, "phase2_outputs")

N_ROI      = 116
LABEL_MAP  = {"CN": 0, "AD": 1, "SMCI": 2, "PMCI": 3}
NC_LABELS  = {"CN"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
#  STEP 1: Build aligned subject list
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Building aligned subject list")
print("=" * 65)

pheno = pd.read_csv(PHENOTYPIC_CSV)
print(f"Phenotypic CSV: {len(pheno)} subjects")
print("Label distribution:", pheno["Label"].value_counts().to_dict())


def pheno_to_feat_id(ptid: str) -> str:
    """002S4171  ->  002_S_4171"""
    return ptid[:3] + "_S_" + ptid[4:]


pheno_lookup = {pheno_to_feat_id(row["PTID"]): row for _, row in pheno.iterrows()}

smri_subjects  = set(os.listdir(SMRI_DIR))
pet_subjects   = set(os.listdir(PET_DIR))
both_subjects  = smri_subjects & pet_subjects
labeled_subjects = sorted([s for s in both_subjects if s in pheno_lookup])
N = len(labeled_subjects)

print(f"sMRI subjects: {len(smri_subjects)}")
print(f"PET subjects:  {len(pet_subjects)}")
print(f"Both modalities + labeled: {N}")

if N == 0:
    raise RuntimeError(
        "No subjects matched! Check ID format.\n"
        f"Sample sMRI: {sorted(smri_subjects)[:3]}\n"
        f"Sample pheno feat_id: {[pheno_to_feat_id(p) for p in pheno['PTID'].head(3)]}"
    )

subject_ids = np.array(labeled_subjects)
labels_str  = np.array([pheno_lookup[s]["Label"]    for s in labeled_subjects])
labels_int  = np.array([LABEL_MAP[pheno_lookup[s]["Label"]] for s in labeled_subjects])
ages        = np.array([pheno_lookup[s]["AGE"]       for s in labeled_subjects], dtype=np.float32)
genders     = np.array([pheno_lookup[s]["PTGENDER"]  for s in labeled_subjects])
nc_mask     = np.array([pheno_lookup[s]["Label"] in NC_LABELS for s in labeled_subjects])

print(f"\nLabel counts: { {k: int((labels_str==k).sum()) for k in LABEL_MAP} }")
print(f"NC subjects (for M_NC): {nc_mask.sum()}")

pd.DataFrame({
    "subject": subject_ids, "label": labels_str,
    "label_int": labels_int, "age": ages, "gender": genders
}).to_csv(os.path.join(OUTPUT_DIR, "phenotypic_aligned.csv"), index=False)

np.save(os.path.join(OUTPUT_DIR, "subject_ids.npy"), subject_ids)
np.save(os.path.join(OUTPUT_DIR, "labels.npy"),      labels_str)
np.save(os.path.join(OUTPUT_DIR, "labels_int.npy"),  labels_int)
np.save(os.path.join(OUTPUT_DIR, "nc_mask.npy"),     nc_mask)
print("Saved subject_ids, labels, nc_mask")


# ─────────────────────────────────────────────────────────────────
#  STEP 2: Load raw ROI features from per-subject npy files
#  NEW: Each ROI now has 4 feature dimensions [mean, std, p25, p75]
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Loading raw ROI features from disk")
print("FEATURE FORMAT: Each ROI has 4 dimensions [mean, std, p25, p75]")
print("=" * 65)

F_gm   = np.zeros((N, N_ROI, 4), dtype=np.float32)
F_wm   = np.zeros((N, N_ROI, 4), dtype=np.float32)
F_csf  = np.zeros((N, N_ROI, 4), dtype=np.float32)
F_suvr = np.zeros((N, N_ROI, 4), dtype=np.float32)

# Track which subjects are loadable
loadable_subjects = []
for idx, subj in enumerate(labeled_subjects):
    smri_d = os.path.join(SMRI_DIR, subj)
    pet_d  = os.path.join(PET_DIR,  subj)
    
    # Check if all required files exist
    gm_file   = os.path.join(smri_d, "gm_features.npy")
    wm_file   = os.path.join(smri_d, "wm_features.npy")
    csf_file  = os.path.join(smri_d, "csf_features.npy")
    suvr_file = os.path.join(pet_d,  "suvr_stats.npy")
    
    if not all(os.path.exists(f) for f in [gm_file, wm_file, csf_file, suvr_file]):
        print(f"  [SKIP] {subj} — missing feature files")
        continue
    
    # Load (116, 4) feature arrays
    F_gm[len(loadable_subjects)]   = np.load(gm_file)
    F_wm[len(loadable_subjects)]   = np.load(wm_file)
    F_csf[len(loadable_subjects)]  = np.load(csf_file)
    F_suvr[len(loadable_subjects)] = np.load(suvr_file)
    
    loadable_subjects.append(idx)
    
    if len(loadable_subjects) % 100 == 0:
        print(f"  Loaded {len(loadable_subjects)}/{N}")

# Trim arrays to actual number of loadable subjects
N_loaded = len(loadable_subjects)
F_gm   = F_gm[:N_loaded]
F_wm   = F_wm[:N_loaded]
F_csf  = F_csf[:N_loaded]
F_suvr = F_suvr[:N_loaded]

# Update labeled subjects and related arrays
labeled_subjects = [labeled_subjects[i] for i in loadable_subjects]
subject_ids = subject_ids[loadable_subjects]
labels_str  = labels_str[loadable_subjects]
labels_int  = labels_int[loadable_subjects]
ages        = ages[loadable_subjects]
genders     = genders[loadable_subjects]
nc_mask     = nc_mask[loadable_subjects]

N = N_loaded

print(f"\nLoaded {N}/{len(labeled_subjects)} subjects with complete features")

# Impute any NaNs with per-ROI per-dimension mean
for F, name in [(F_gm,"GM"),(F_wm,"WM"),(F_csf,"CSF"),(F_suvr,"SUVR")]:
    if np.isnan(F).any():
        print(f"  Imputing NaNs in {name}")
        # Reshape to [N*116, 4], compute means per dimension, then reshape back
        F_reshape = F.reshape(-1, 4)
        for dim in range(4):
            col_mean = np.nanmean(F_reshape[:, dim])
            F_reshape[np.isnan(F_reshape[:, dim]), dim] = col_mean
        F[:] = F_reshape.reshape(N, N_ROI, 4)

print(f"Loading complete. Shapes: GM={F_gm.shape}, SUVR={F_suvr.shape}")
print(f"SUVR range: [{F_suvr.min():.4f}, {F_suvr.max():.4f}]")
print(f"GM   range: [{F_gm.min():.4f},  {F_gm.max():.4f}]")
print(f"Feature dimensions per ROI: 4 (mean, std, p25, p75)")


# ─────────────────────────────────────────────────────────────────
#  CORE FUNCTION: Equations 1-4
# ─────────────────────────────────────────────────────────────────

def compute_brain_networks(F, nc_mask, modality_name):
    """
    Build individual brain network B for each subject (Equations 1-4).
    ADAPTED FOR MULTI-DIMENSIONAL ROI FEATURES: F shape [N, 116, 4]

    Each ROI is now a 4-dimensional feature vector. Distances are computed
    using Euclidean norm instead of scalar differences.

    Eq.1  E(i,j)   = ||delta_i - delta_j||_2 / s_p(i,j)
          where delta_i = f_i - f_NC_i are now 4-dim vectors
    Eq.2  R(i,j)   = (exp(2E)-1) / (exp(2E)+1)   [Fisher transform -> [0,1]]
    Eq.3  W(i,j)   = 1 - R(i,j)
    Eq.4  B(i,j)   = W(i,j) ⊙ M_NC(i,j)

    Returns:
        individual_features : [N, 6786]    upper triangle of B (node vectors)
        B_matrices          : [N, 116, 116]
        M_NC                : [116, 116]
    """
    N = F.shape[0]
    P = 116  # Fixed number of ROIs
    upper_tri = np.triu_indices(P, k=0)   # 116*117//2 = 6786 elements
    n_feat    = len(upper_tri[0])

    # NC statistics (now per feature dimension)
    F_nc    = F[nc_mask]  # [n_nc, 116, 4]
    nc_mean = F_nc.mean(axis=0)                  # [116, 4]
    nc_std  = np.maximum(F_nc.std(axis=0), 1e-8)  # [116, 4]

    # Pooled std matrix: compute L2 norm of std vectors per ROI
    # s_p(i,j) = sqrt( (||s_i||_2^2 + ||s_j||_2^2) / 2 )
    std_norms = np.linalg.norm(nc_std, axis=1)  # [116] - Euclidean norm of each ROI's std
    s_p = np.sqrt((std_norms[:, None]**2 + std_norms[None, :]**2) / 2.0)  # [116, 116]
    s_p = np.maximum(s_p, 1e-8)

    # M_NC: average correlation matrix of NC subjects
    print(f"  [{modality_name}] Computing M_NC from {nc_mask.sum()} NC subjects (4-dim features)...")
    M_NC = np.zeros((P, P), dtype=np.float32)
    for i in range(len(F_nc)):
        delta = F_nc[i] - nc_mean  # [116, 4] - feature space deltas per ROI
        
        # Vectorized pairwise Euclidean distances between ROI feature vectors
        # delta[:, None, :] - delta[None, :, :] gives [116, 116, 4]
        delta_diffs = delta[:, None, :] - delta[None, :, :]  # [116, 116, 4]
        E_mat = np.linalg.norm(delta_diffs, axis=2)  # [116, 116] - L2 norms
        
        E     = E_mat / s_p  # Eq.1 - normalize by pooled std
        e2E   = np.exp(np.clip(2 * E, -500, 500))
        R     = (e2E - 1) / (e2E + 1)  # Eq.2
        M_NC += R
    M_NC /= len(F_nc)
    print(f"  [{modality_name}] M_NC range: [{M_NC.min():.4f}, {M_NC.max():.4f}]")

    # Per-subject networks
    print(f"  [{modality_name}] Building {N} subject networks ...")
    individual_features = np.zeros((N, n_feat), dtype=np.float32)
    B_matrices          = np.zeros((N, P, P),   dtype=np.float32)

    for idx in range(N):
        delta = F[idx] - nc_mean  # [116, 4]
        
        # Vectorized pairwise Euclidean distances
        delta_diffs = delta[:, None, :] - delta[None, :, :]  # [116, 116, 4]
        E_mat = np.linalg.norm(delta_diffs, axis=2)  # [116, 116]
        
        E     = E_mat / s_p  # Eq.1
        e2E   = np.exp(np.clip(2 * E, -500, 500))
        R     = (e2E - 1) / (e2E + 1)  # Eq.2
        W     = 1.0 - R  # Eq.3
        B     = W * M_NC  # Eq.4

        B_matrices[idx]          = B.astype(np.float32)
        individual_features[idx] = B[upper_tri]

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{N}")

    print(f"  [{modality_name}] Done. features shape: {individual_features.shape}")
    return individual_features, B_matrices, M_NC


# ─────────────────────────────────────────────────────────────────
#  STEP 3: PET networks
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: PET Brain Networks")
print("=" * 65)

pet_features, pet_B, M_NC_pet = compute_brain_networks(F_suvr, nc_mask, "PET")
np.save(os.path.join(OUTPUT_DIR, "pet_individual_features.npy"), pet_features)
np.save(os.path.join(OUTPUT_DIR, "pet_B_matrices.npy"),          pet_B)
np.save(os.path.join(OUTPUT_DIR, "M_NC_pet.npy"),                M_NC_pet)

with open(os.path.join(OUTPUT_DIR, "pet_nc_stats.json"), "w") as f:
    json.dump({"nc_mean": F_suvr[nc_mask].mean(0).tolist(),
               "nc_std":  F_suvr[nc_mask].std(0).tolist(),
               "n_nc":    int(nc_mask.sum())}, f, indent=2)
print("Saved PET outputs.")


# ─────────────────────────────────────────────────────────────────
#  STEP 4: sMRI networks (GM, WM, GM+WM)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: sMRI Brain Networks (GM / WM / GM+WM)")
print("=" * 65)

print("\n  --- GM ---")
smri_gm_features, smri_gm_B, M_NC_gm = compute_brain_networks(F_gm, nc_mask, "sMRI-GM")
np.save(os.path.join(OUTPUT_DIR, "smri_gm_individual_features.npy"), smri_gm_features)
np.save(os.path.join(OUTPUT_DIR, "smri_gm_B_matrices.npy"),          smri_gm_B)
np.save(os.path.join(OUTPUT_DIR, "M_NC_gm.npy"),                     M_NC_gm)

print("\n  --- WM ---")
smri_wm_features, smri_wm_B, M_NC_wm = compute_brain_networks(F_wm, nc_mask, "sMRI-WM")
np.save(os.path.join(OUTPUT_DIR, "smri_wm_individual_features.npy"), smri_wm_features)
np.save(os.path.join(OUTPUT_DIR, "smri_wm_B_matrices.npy"),          smri_wm_B)
np.save(os.path.join(OUTPUT_DIR, "M_NC_wm.npy"),                     M_NC_wm)

print("\n  --- GM+WM (averaged brain networks) ---")
# Average the two [N,116,116] B matrices -> keeps 6786-dim node vectors
smri_gmwm_B        = (smri_gm_B + smri_wm_B) / 2.0
upper_tri          = np.triu_indices(N_ROI, k=0)
smri_gmwm_features = smri_gmwm_B[:, upper_tri[0], upper_tri[1]]
M_NC_gmwm          = (M_NC_gm + M_NC_wm) / 2.0
np.save(os.path.join(OUTPUT_DIR, "smri_gmwm_individual_features.npy"), smri_gmwm_features)
np.save(os.path.join(OUTPUT_DIR, "smri_gmwm_B_matrices.npy"),          smri_gmwm_B)
np.save(os.path.join(OUTPUT_DIR, "M_NC_gmwm.npy"),                     M_NC_gmwm)
print(f"  GM+WM features shape: {smri_gmwm_features.shape}")
print("Saved all sMRI outputs.")


# ─────────────────────────────────────────────────────────────────
#  STEP 5: Validation
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Validation")
print("=" * 65)

def validate(arr, name, exp_shape):
    ok   = arr.shape == exp_shape
    nans = np.isnan(arr).sum()
    infs = np.isinf(arr).sum()
    ok   = ok and nans == 0 and infs == 0
    sym  = "✓" if ok else "✗"
    warn = ""
    if arr.shape != exp_shape: warn += f" SHAPE:{arr.shape}!={exp_shape}"
    if nans: warn += f" NaN:{nans}"
    if infs: warn += f" Inf:{infs}"
    print(f"  {sym} {name:<42} {str(arr.shape):<18} "
          f"[{arr.min():.4f}, {arr.max():.4f}]{warn}")

print()
validate(pet_features,        "PET individual features",       (N, 6786))
validate(smri_gm_features,    "sMRI-GM individual features",   (N, 6786))
validate(smri_wm_features,    "sMRI-WM individual features",   (N, 6786))
validate(smri_gmwm_features,  "sMRI-GMWM individual features", (N, 6786))
validate(M_NC_pet,            "M_NC (PET)",                    (116, 116))
validate(M_NC_gm,             "M_NC (GM)",                     (116, 116))

print("\n  Per-class feature L2-norm (PET):")
for label in ["CN", "AD", "SMCI", "PMCI"]:
    m = labels_str == label
    if not m.any(): continue
    norms = np.linalg.norm(pet_features[m], axis=1)
    print(f"    {label:5s}: n={m.sum():3d}  "
          f"mean={norms.mean():.4f}  std={norms.std():.4f}")


# ─────────────────────────────────────────────────────────────────
#  STEP 6: Summary
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE 2 COMPLETE — Output files")
print("=" * 65)

files = [
    ("pet_individual_features.npy",       f"[{N}, 6786]       PET node vectors  ← Phase 3 input"),
    ("smri_gm_individual_features.npy",   f"[{N}, 6786]       sMRI-GM node vectors  ← Phase 3 input"),
    ("smri_wm_individual_features.npy",   f"[{N}, 6786]       sMRI-WM node vectors"),
    ("smri_gmwm_individual_features.npy", f"[{N}, 6786]       sMRI-GM+WM node vectors"),
    ("pet_B_matrices.npy",                f"[{N}, 116, 116]   PET brain matrices"),
    ("smri_gm_B_matrices.npy",            f"[{N}, 116, 116]   sMRI-GM brain matrices"),
    ("smri_wm_B_matrices.npy",            f"[{N}, 116, 116]   sMRI-WM brain matrices"),
    ("smri_gmwm_B_matrices.npy",          f"[{N}, 116, 116]   sMRI-GMWM brain matrices"),
    ("M_NC_pet.npy",                      "[116, 116]         NC reference matrix (PET)"),
    ("M_NC_gm.npy",                       "[116, 116]         NC reference matrix (GM)"),
    ("M_NC_wm.npy",                       "[116, 116]         NC reference matrix (WM)"),
    ("subject_ids.npy",                   f"[{N}]             Subject IDs"),
    ("labels.npy",                        f"[{N}]             Labels (CN/AD/SMCI/PMCI)"),
    ("labels_int.npy",                    f"[{N}]             Int labels (CN=0,AD=1,SMCI=2,PMCI=3)"),
    ("nc_mask.npy",                       f"[{N}]             Boolean NC mask"),
    ("phenotypic_aligned.csv",            f"[{N}]             Phenotypic aligned to subject order"),
    ("pet_nc_stats.json",                 "NC mean/std per ROI"),
]
for fname, desc in files:
    e = "✓" if os.path.exists(os.path.join(OUTPUT_DIR, fname)) else "✗ MISSING"
    print(f"  {e}  {fname:<45} {desc}")

print(f"\nAll outputs -> {OUTPUT_DIR}")
print("\n" + "=" * 65)
print("NEXT: Phase 3 — Graph Construction (Equations 6-13)")
print("  Key inputs:")
print("  • pet_individual_features.npy      (PET branch node features)")
print("  • smri_gm_individual_features.npy  (sMRI branch node features)")
print("  • phenotypic_aligned.csv           (gender for Eq.7 edge weights)")
print("  • labels_int.npy                   (train/val/test split)")
print("=" * 65)