"""
Step 2: ROI Feature Extraction using AAL Atlas (Linux)
=======================================================
Input:  Step 1 output (registered.nii.gz per subject)
Output: Per-subject feature vectors (116 ROIs × 4 statistics each) + combined CSV

For sMRI: GM, WM, CSF [mean, std, p25, p75] per ROI (116 ROIs × 4 stats × 3 tissues)
For PET:  [mean, std, p25, p75] SUVR per ROI (116 ROIs × 4 stats)

Total features per subject: 116 × 4 modalities × 4 statistics = 1856 features

Requirements:
    sudo apt install fsl
    pip install nibabel nilearn numpy pandas

Usage:
    python extract_features_linux.py \
        --processed /path/to/step1/output \
        --output    /path/to/features/output \
        --workers   8
"""

import os
import sys
import subprocess
import argparse
import shutil
import tarfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import nibabel as nib
import pandas as pd

FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")


# ── DEPENDENCY CHECK ──────────────────────────────────────────────────────────
def check_dependencies():
    errors = []

    if shutil.which("fast") is None:
        errors.append("FSL 'fast' not found — run: source $FSLDIR/etc/fslconf/fsl.sh")

    for pkg in ["nibabel", "nilearn", "numpy", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"Missing Python package: {pkg} — run: pip install {pkg}")

    if errors:
        print("[ERROR] Missing dependencies:")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)

    print("[OK] All dependencies found.")


# ── LOAD AAL ATLAS ────────────────────────────────────────────────────────────
def _safe_label(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(text)).strip("_") or "ROI"


def _labels_from_xml(labels_xml: Path):
    labels = ["Background"]
    indices = ["0"]
    root = ElementTree.parse(str(labels_xml)).getroot()
    for label in root.iter("label"):
        idx = label.find("index")
        name = label.find("name")
        if idx is None or name is None:
            continue
        indices.append(idx.text)
        labels.append(name.text)
    return labels, indices


def _labels_for_region_ids(region_ids, labels, indices=None):
    id_to_label = {}
    if indices is not None and len(indices) == len(labels):
        for idx, label in zip(indices, labels):
            try:
                rid = int(idx)
            except (TypeError, ValueError):
                continue
            if rid == 0:
                continue
            id_to_label[rid] = _safe_label(label)

    if not id_to_label:
        # Fallback for atlas variants where labels are ordered and include
        # a background entry at position 0.
        ordered = labels[1:] if len(labels) == len(region_ids) + 1 else labels
        for i, rid in enumerate(region_ids):
            if i < len(ordered):
                id_to_label[rid] = _safe_label(ordered[i])

    return [id_to_label.get(rid, f"ROI_{rid}") for rid in region_ids]


def _find_local_aal_spm12():
    candidates = [
        Path.home() / "nilearn_data" / "aal_SPM12" / "aal" / "atlas",
        Path("/home/user/nilearn_data/aal_SPM12/aal/atlas"),
    ]
    for base in candidates:
        nii = base / "AAL.nii"
        xml = base / "AAL.xml"
        if nii.exists() and xml.exists():
            return nii, xml
    return None, None


def _download_aal_spm12_insecure(cache_dir: Path):
    import requests

    url = "https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "aal_for_SPM12.tar.gz"
    extract_root = cache_dir

    print("[AAL] Attempting insecure atlas download (SSL verify disabled)...")
    requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    with requests.get(url, stream=True, timeout=120, verify=False) as resp:
        resp.raise_for_status()
        with archive.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with tarfile.open(archive, "r:gz") as tar:
        # Basic path traversal guard for extraction.
        for member in tar.getmembers():
            target = (extract_root / member.name).resolve()
            if not str(target).startswith(str(extract_root.resolve())):
                raise RuntimeError("Unsafe path found in atlas archive")
        tar.extractall(path=extract_root)

    nii = extract_root / "aal" / "atlas" / "AAL.nii"
    xml = extract_root / "aal" / "atlas" / "AAL.xml"
    if not (nii.exists() and xml.exists()):
        raise FileNotFoundError("Downloaded archive did not contain AAL.nii and AAL.xml")
    return nii, xml


def load_aal_atlas(atlas_map=None, atlas_labels=None, allow_insecure_download=False):
    from nilearn import datasets

    print("[AAL] Loading AAL atlas (116 ROIs)...")

    if atlas_map is not None and atlas_labels is not None:
        atlas_map = Path(atlas_map)
        atlas_labels = Path(atlas_labels)
        if not atlas_map.exists() or not atlas_labels.exists():
            raise FileNotFoundError("Provided atlas files do not exist")
        print(f"[AAL] Using custom atlas files: {atlas_map} and {atlas_labels}")
        labels, indices = _labels_from_xml(atlas_labels)
        atlas_img = nib.load(str(atlas_map))
    else:
        local_map, local_xml = _find_local_aal_spm12()
        if local_map is not None and local_xml is not None:
            print(f"[AAL] Using local cached atlas: {local_map}")
            labels, indices = _labels_from_xml(local_xml)
            atlas_img = nib.load(str(local_map))
        else:
            try:
                aal = datasets.fetch_atlas_aal(version="SPM12")
                atlas_img = nib.load(aal.maps)
                labels = aal.labels
                indices = getattr(aal, "indices", None)
            except Exception as e:
                err = str(e)
                is_ssl = "CERTIFICATE_VERIFY_FAILED" in err or "SSLError" in e.__class__.__name__
                if is_ssl and allow_insecure_download:
                    cache_dir = Path.home() / "nilearn_data" / "aal_SPM12"
                    local_map, local_xml = _download_aal_spm12_insecure(cache_dir)
                    labels, indices = _labels_from_xml(local_xml)
                    atlas_img = nib.load(str(local_map))
                else:
                    raise RuntimeError(
                        "AAL atlas download failed, likely due to SSL certificate verification. "
                        "Install/update CA certificates or rerun with "
                        "--allow-insecure-atlas-download (temporary workaround), or provide "
                        "--atlas-map and --atlas-labels."
                    ) from e

    atlas_data = atlas_img.get_fdata().astype(int)
    region_ids = sorted(set(atlas_data.flatten()) - {0})
    roi_labels = _labels_for_region_ids(region_ids, labels, indices)

    print(f"[AAL] Loaded {len(region_ids)} ROIs")
    return atlas_img, region_ids, roi_labels


# ── RESAMPLE ATLAS TO SUBJECT SPACE ──────────────────────────────────────────
def resample_atlas(atlas_img, subject_img):
    from nilearn import image
    resampled = image.resample_img(
        atlas_img,
        target_affine=subject_img.affine,
        target_shape=subject_img.shape[:3],
        interpolation="nearest"
    )
    return resampled.get_fdata().astype(int)


# ── SMRI TISSUE SEGMENTATION VIA FSL FAST ────────────────────────────────────
def segment_smri_fast(registered_smri: Path, out_dir: Path):
    """
    Run FSL FAST on registered sMRI to get GM/WM/CSF probability maps.

    FAST output naming:
      <base>_pve_0.nii.gz  -> CSF
      <base>_pve_1.nii.gz  -> GM
      <base>_pve_2.nii.gz  -> WM
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(out_dir / "fast")

    # Check if already done
    if (out_dir / "fast_pve_1.nii.gz").exists():
        print(f"    [FAST] Already segmented, loading existing maps")
    else:
        print(f"    [FAST] Running tissue segmentation...")
        cmd = [
            "fast",
            "-t", "1",       # T1 image type
            "-n", "3",       # 3 tissue classes (CSF, GM, WM)
            "-H", "0.1",     # MRF beta value
            "-I", "4",       # number of iterations
            "-l", "20.0",    # smoothing factor
            "-o", base,
            str(registered_smri)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    [ERROR] FAST failed:\n{result.stderr}")
            return None, None, None

    try:
        csf = nib.load(str(out_dir / "fast_pve_0.nii.gz")).get_fdata()
        gm  = nib.load(str(out_dir / "fast_pve_1.nii.gz")).get_fdata()
        wm  = nib.load(str(out_dir / "fast_pve_2.nii.gz")).get_fdata()
        print(f"    [OK] FAST segmentation loaded")
        return gm, wm, csf
    except Exception as e:
        print(f"    [ERROR] Could not load FAST output: {e}")
        return None, None, None


# ── EXTRACT SMRI ROI FEATURES ─────────────────────────────────────────────────
def extract_smri_features(registered_smri: Path, atlas_img, region_ids, out_dir: Path):
    """
    Extract rich statistics (mean, std, p25, p75) for GM, WM, CSF per ROI.
    Returns (gm_features, wm_features, csf_features) each shape (116, 4)
    where dim 1 = [mean, std, p25, p75]
    """
    # Check if already extracted
    if (out_dir / "gm_features.npy").exists():
        print(f"    [SKIP] sMRI features already extracted")
        return (
            np.load(str(out_dir / "gm_features.npy")),
            np.load(str(out_dir / "wm_features.npy")),
            np.load(str(out_dir / "csf_features.npy"))
        )

    subject_img = nib.load(str(registered_smri))
    gm, wm, csf = segment_smri_fast(registered_smri, out_dir)
    if gm is None:
        return None, None, None

    print(f"    [ROI] Extracting sMRI features from {len(region_ids)} ROIs (4 stats each)...")
    atlas_resampled = resample_atlas(atlas_img, subject_img)

    # Shape: (num_rois, 4) where 4 = [mean, std, p25, p75]
    gm_feat  = np.zeros((len(region_ids), 4))
    wm_feat  = np.zeros((len(region_ids), 4))
    csf_feat = np.zeros((len(region_ids), 4))

    for i, roi_id in enumerate(region_ids):
        mask = (atlas_resampled == roi_id)
        if mask.sum() == 0:
            # Leave as zeros (already initialized)
            continue
        
        gm_vals  = gm[mask]
        wm_vals  = wm[mask]
        csf_vals = csf[mask]
        
        gm_feat[i]  = [gm_vals.mean(),  gm_vals.std(),  np.percentile(gm_vals, 25),  np.percentile(gm_vals, 75)]
        wm_feat[i]  = [wm_vals.mean(),  wm_vals.std(),  np.percentile(wm_vals, 25),  np.percentile(wm_vals, 75)]
        csf_feat[i] = [csf_vals.mean(), csf_vals.std(), np.percentile(csf_vals, 25), np.percentile(csf_vals, 75)]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "gm_features.npy"),  gm_feat)
    np.save(str(out_dir / "wm_features.npy"),  wm_feat)
    np.save(str(out_dir / "csf_features.npy"), csf_feat)

    print(f"    [OK] sMRI features saved: shape {gm_feat.shape}")
    return gm_feat, wm_feat, csf_feat


# ── EXTRACT PET ROI FEATURES (SUVR) ──────────────────────────────────────────
def extract_pet_features(registered_pet: Path, atlas_img, region_ids, out_dir: Path):
    """
    Extract rich SUVR statistics (mean, std, p25, p75) per ROI.
    Returns suvr_stats shape (116, 4) where dim 1 = [mean, std, p25, p75]
    """
    # Check if already extracted
    if (out_dir / "suvr_stats.npy").exists():
        print(f"    [SKIP] PET features already extracted")
        return np.load(str(out_dir / "suvr_stats.npy"))

    print(f"    [ROI] Extracting PET SUVR features from {len(region_ids)} ROIs (4 stats each)...")
    pet_img  = nib.load(str(registered_pet))
    pet_data = pet_img.get_fdata()

    atlas_resampled = resample_atlas(atlas_img, pet_img)

    # Shape: (num_rois, 4) where 4 = [mean, std, p25, p75]
    suvr_stats = np.zeros((len(region_ids), 4))

    for i, roi_id in enumerate(region_ids):
        mask = (atlas_resampled == roi_id)
        if mask.sum() == 0:
            # Leave as zeros (already initialized)
            continue
        roi_vals = pet_data[mask]
        suvr_stats[i] = [roi_vals.mean(), roi_vals.std(), np.percentile(roi_vals, 25), np.percentile(roi_vals, 75)]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "suvr_stats.npy"), suvr_stats)

    print(f"    [OK] PET features saved: shape {suvr_stats.shape}")
    return suvr_stats


# ── SINGLE SUBJECT WORKER ─────────────────────────────────────────────────────
def process_subject(args):
    """
    Extract features for one subject. Runs in a separate process.
    Returns dict with subject id and all feature arrays.
    """
    subject, processed_dir_str, output_dir_str, atlas_img_path, region_ids = args

    processed_dir = Path(processed_dir_str)
    output_dir    = Path(output_dir_str)

    smri_registered = processed_dir / "sMRI" / subject / "registered.nii.gz"
    pet_registered  = processed_dir / "PET"  / subject / "registered.nii.gz"

    smri_feat_dir = output_dir / "sMRI" / subject
    pet_feat_dir  = output_dir / "PET"  / subject

    # Reload atlas in this process (can't pickle nibabel images)
    atlas_img = nib.load(atlas_img_path)

    result = {
        "subject": subject,
        "gm": None, "wm": None, "csf": None,
        "suvr_stats": None,
        "status": "failed"
    }

    # sMRI features
    if smri_registered.exists():
        smri_feat_dir.mkdir(parents=True, exist_ok=True)
        try:
            gm, wm, csf = extract_smri_features(
                smri_registered, atlas_img, region_ids, smri_feat_dir
            )
            result.update({"gm": gm, "wm": wm, "csf": csf})
        except Exception as e:
            print(f"  [ERROR] {subject} sMRI features failed: {e}")
    else:
        print(f"  [SKIP] {subject} — sMRI registered.nii.gz not found")

    # PET features
    if pet_registered.exists():
        pet_feat_dir.mkdir(parents=True, exist_ok=True)
        try:
            suvr_stats = extract_pet_features(
                pet_registered, atlas_img, region_ids, pet_feat_dir
            )
            result.update({"suvr_stats": suvr_stats})
        except Exception as e:
            print(f"  [ERROR] {subject} PET features failed: {e}")

    # Status
    has_smri = result["gm"] is not None
    has_pet  = result["suvr_stats"] is not None

    if has_smri and has_pet:
        result["status"] = "smri_and_pet"
    elif has_smri:
        result["status"] = "smri_only"
    elif has_pet:
        result["status"] = "pet_only"

    return result


# -- SAVE SUMMARY CSV --
def _flatten_features_to_dict(subject_id: str, labels: list, gm_feat, wm_feat, csf_feat, suvr_stats):
    """
    Flatten (116, 4) feature arrays into a flat dictionary for CSV export.
    
    Each ROI gets 4 columns per modality type:
      gm_ROI1_mean, gm_ROI1_std, gm_ROI1_p25, gm_ROI1_p75,
      wm_ROI1_mean, wm_ROI1_std, wm_ROI1_p25, wm_ROI1_p75,
      csf_ROI1_mean, csf_ROI1_std, csf_ROI1_p25, csf_ROI1_p75,
      suvr_ROI1_mean, suvr_ROI1_std, suvr_ROI1_p25, suvr_ROI1_p75,
    """
    stat_names = ['mean', 'std', 'p25', 'p75']
    row = {"subject": subject_id}
    
    for i, label in enumerate(labels):
        if gm_feat is not None:
            for j, stat in enumerate(stat_names):
                row[f"gm_{label}_{stat}"] = gm_feat[i, j] if gm_feat.ndim == 2 else gm_feat[i]
        if wm_feat is not None:
            for j, stat in enumerate(stat_names):
                row[f"wm_{label}_{stat}"] = wm_feat[i, j] if wm_feat.ndim == 2 else wm_feat[i]
        if csf_feat is not None:
            for j, stat in enumerate(stat_names):
                row[f"csf_{label}_{stat}"] = csf_feat[i, j] if csf_feat.ndim == 2 else csf_feat[i]
        if suvr_stats is not None:
            for j, stat in enumerate(stat_names):
                row[f"suvr_{label}_{stat}"] = suvr_stats[i, j] if suvr_stats.ndim == 2 else suvr_stats[i]
    
    return row


def save_summary(all_results: list, labels: list, out_path: Path):
    """
    Combine all subject features into a single CSV with flattened (116, 4) arrays.
    
    Each ROI now contributes 4 columns per modality (mean, std, p25, p75).
    """
    rows = []
    for r in all_results:
        row = _flatten_features_to_dict(
            r["subject"], labels,
            r["gm"], r["wm"], r["csf"],
            r["suvr_stats"]
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(str(out_path), index=False)
    print(f"\n[Summary] Saved CSV -> {out_path}")
    print(f"          {df.shape[0]} subjects x {df.shape[1]-1} features")
    print(f"          Feature breakdown: 116 ROIs x 4 modalities x 4 statistics = {116 * 4 * 4} total")
    return df


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run_extraction(
    processed_dir: str,
    output_dir: str,
    n_workers: int = None,
    atlas_map: str = None,
    atlas_labels: str = None,
    allow_insecure_atlas_download: bool = False,
):
    check_dependencies()

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"[Parallel] Using {n_workers} workers")

    processed = Path(processed_dir)
    output    = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Load atlas once in main process, save path for workers
    atlas_img, region_ids, labels = load_aal_atlas(
        atlas_map=atlas_map,
        atlas_labels=atlas_labels,
        allow_insecure_download=allow_insecure_atlas_download,
    )
    atlas_cache = str(output / "aal_atlas_cache.nii.gz")
    if not Path(atlas_cache).exists():
        nib.save(atlas_img, atlas_cache)

    # Find subjects that have at least sMRI registered output
    smri_out_dir = processed / "sMRI"
    pet_out_dir  = processed / "PET"

    if not smri_out_dir.exists():
        print(f"[ERROR] sMRI output folder not found: {smri_out_dir}")
        sys.exit(1)

    subjects = sorted([
        s.name for s in smri_out_dir.iterdir()
        if s.is_dir() and (s / "registered.nii.gz").exists()
    ])

    # Also include subjects that only have PET registered
    if pet_out_dir.exists():
        pet_subjects = set(
            s.name for s in pet_out_dir.iterdir()
            if s.is_dir() and (s / "registered.nii.gz").exists()
        )
        subjects = sorted(set(subjects) | pet_subjects)

    print(f"\nFound {len(subjects)} subject(s) to process")
    print("=" * 60)

    worker_args = [
        (subject, str(processed), str(output), atlas_cache, region_ids)
        for subject in subjects
    ]

    all_results = []
    completed   = 0
    total       = len(worker_args)

    status_counts = {
        "smri_and_pet": 0,
        "smri_only":    0,
        "pet_only":     0,
        "failed":       0
    }

    print(f"\n[Processing] Starting feature extraction for {total} subjects with {n_workers} workers")
    print(f"{'='*60}\n")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_subject, args): args[0]
            for args in worker_args
        }
        active_subjects = {future: args[0] for future, args in zip(futures, worker_args)}
        
        for future in as_completed(futures):
            subject_name = futures[future]
            completed += 1
            pending = total - completed
            try:
                result = future.result()
                all_results.append(result)
                status = result["status"]
                status_counts[status] += 1
                
                # Enhanced progress output
                pending = total - completed
                progress_pct = int((completed / total) * 100)
                status_label = f"{status.upper()}"
                print(f"  [{completed:3d}/{total}] {subject_name:20s} → {status_label:12s}  ({pending:3d} pending, {progress_pct:3d}%)")
            except Exception as e:
                status_counts["failed"] += 1
                all_results.append({
                    "subject": subject_name,
                    "gm": None, "wm": None, "csf": None,
                    "suvr_stats": None,
                    "status": "failed"
                })
                pending = total - completed
                progress_pct = int((completed / total) * 100)
                print(f"  [{completed:3d}/{total}] {subject_name:20s} → FAILED         ({pending:3d} pending, {progress_pct:3d}%)  |  {str(e)[:60]}")

    # Sort results by subject name for consistent CSV ordering
    all_results.sort(key=lambda x: x["subject"])

    # Save combined CSV
    save_summary(all_results, labels, output / "features_summary.csv")

    # Save individual numpy arrays per subject (already saved in workers,
    # but confirm counts here)
    successful = status_counts['smri_and_pet'] + status_counts['smri_only'] + status_counts['pet_only']
    
    print(f"\n{'='*60}")
    print(f"✓ Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Results summary:")
    print(f"  ✓ sMRI + PET : {status_counts['smri_and_pet']:4d} subjects")
    print(f"  ✓ sMRI only  : {status_counts['smri_only']:4d} subjects")
    print(f"  ✓ PET only   : {status_counts['pet_only']:4d} subjects")
    print(f"  ✗ Failed     : {status_counts['failed']:4d} subjects")
    print(f"  Failed     : {status_counts['failed']}")
    print(f"\nOutputs:")
    print(f"  Per-subject .npy files in: {output}/sMRI/<subject>/ and {output}/PET/<subject>/")
    print(f"  Combined CSV: {output}/features_summary.csv")
    print(f"\nFeature vector sizes per subject (116 ROIs × 4 statistics each):")
    print(f"  gm_features.npy  : (116, 4) - [mean, std, p25, p75] GM probability per ROI")
    print(f"  wm_features.npy  : (116, 4) - [mean, std, p25, p75] WM probability per ROI")
    print(f"  csf_features.npy : (116, 4) - [mean, std, p25, p75] CSF probability per ROI")
    print(f"  suvr_stats.npy   : (116, 4) - [mean, std, p25, p75] SUVR per ROI")
    print(f"\nCSV columns: 116 ROIs × 4 modalities (GM, WM, CSF, SUVR) × 4 stats = 1856 features")

    # Save log
    log = output / "extraction_summary.log"
    with open(log, "w") as f:
        f.write(f"sMRI + PET : {status_counts['smri_and_pet']}\n")
        f.write(f"sMRI only  : {status_counts['smri_only']}\n")
        f.write(f"PET only   : {status_counts['pet_only']}\n")
        f.write(f"Failed     : {status_counts['failed']}\n")
    print(f"  Log: {log}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hardcoded paths
    processed_dir = "./output"  # Step 1 preprocessing output
    output_dir = "./features"   # Feature extraction output (root folder)
    
    run_extraction(
        processed_dir,
        output_dir,
        n_workers=10,  # 6 parallel workers
        atlas_map=None,
        atlas_labels=None,
        allow_insecure_atlas_download=False,
    )