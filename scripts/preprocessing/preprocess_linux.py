"""
Preprocessing pipeline for structural MRI and PET data on Linux.

Per-subject workflow
--------------------
sMRI:
    1. DICOM to NIfTI with dcm2niix
    2. N4 bias field correction with SimpleITK
    3. Skull stripping with HD-BET
    4. WhiteStripe intensity normalization
    5. Registration to MNI152 with FSL flirt + fnirt
    6. Gaussian smoothing (5 mm FWHM)

PET:
    1. Convert DICOM, ECAT, or HRRT data to NIfTI
    2. Register PET to subject sMRI
    3. Apply the sMRI warp to MNI152 space
    4. SUVR normalization using a cerebellum reference region
    5. Gaussian smoothing (5 mm FWHM)
    6. Save QC overlay images

Requirements:
    sudo apt install fsl dcm2niix
    pip install SimpleITK nibabel nilearn numpy nipype
    pip install intensity-normalization matplotlib hd-bet

Expected dataset structure:
    dataset/
        sMRI/
            subject_001/
        PET/
            subject_001/

The default SUVR reference mask is:
    $FSLDIR/data/atlases/Cerebellum/
    Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz
"""

import os
import shutil
import subprocess
import sys
import time
import io
import contextlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path


# Configuration
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")
MNI_TEMPLATE = os.path.join(FSLDIR, "data/standard/MNI152_T1_1mm_brain.nii.gz")
MNI_HEAD = os.path.join(FSLDIR, "data/standard/MNI152_T1_1mm.nii.gz")
MNI_MASK = os.path.join(FSLDIR, "data/standard/MNI152_T1_1mm_brain_mask.nii.gz")

# Default cerebellum mask for SUVR, provided by FSL.
CEREBELLUM_MASK_DEFAULT = os.path.join(
    FSLDIR,
    "data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz",
)
DCM2NIIX_FALLBACK = "/home/user/fsl/pkgs/dcm2niix-1.0.20250506-hb700be7_1/bin/dcm2niix"

VERBOSE_SUBJECT_LOGS = False
LAST_ERROR_DETAIL = None


# Utilities
def describe_return_code(returncode: int, step_name: str) -> str:
    """Convert a subprocess return code into a more actionable error message."""
    if returncode < 0:
        signal_num = -returncode
        detail = f"{step_name} was terminated by signal {signal_num}"
        if signal_num == 9:
            detail += " (SIGKILL; often caused by GPU/CPU out-of-memory or an external kill)"
        return detail
    return f"{step_name} failed with exit code {returncode}"


def resolve_dcm2niix() -> str | None:
    """Resolve the dcm2niix executable from env, PATH, or a known local install."""
    env_path = os.environ.get("DCM2NIIX")
    candidates = [env_path, shutil.which("dcm2niix"), DCM2NIIX_FALLBACK]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def set_last_error(detail: str = None):
    """Store the most recent low-level error detail for the current process."""
    global LAST_ERROR_DETAIL
    LAST_ERROR_DETAIL = detail.strip() if isinstance(detail, str) else detail


def get_last_error() -> str:
    """Return the most recent low-level error detail for the current process."""
    return LAST_ERROR_DETAIL


def append_failure_log(
    failure_log_path: str,
    failure_lock,
    subject: str,
    status: str,
    step_label: str,
    reason: str,
    detail: str = None,
):
    """Append a failure record immediately so mid-run crashes are preserved."""
    if not failure_log_path:
        return

    detail_text = (detail or "").strip().replace("\n", " | ")
    line = f"{subject}\t{status}\t{step_label}\t{reason}"
    if detail_text:
        line += f"\t{detail_text}"
    line += "\n"

    if failure_lock is None:
        with open(failure_log_path, "a") as f:
            f.write(line)
        return

    with failure_lock:
        with open(failure_log_path, "a") as f:
            f.write(line)


def run_cmd(cmd, step_name):
    """Run a shell command, print output, return True on success."""
    set_last_error(None)
    print(f"    [{step_name}] Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if result.returncode != 0:
        set_last_error(
            result.stderr
            or result.stdout
            or describe_return_code(result.returncode, step_name)
        )
        print(f"    [ERROR] {step_name} failed:\n{result.stderr}")
        return False
    return True


def log_step(subject: str, step_num: int, total_steps: int, description: str):
    """Print a clearly formatted step progress line for a subject."""
    print(f"  [{subject}] Step {step_num}/{total_steps}: {description}", flush=True)


def update_subject_status(
    status_map,
    subject: str,
    step_num: int,
    total_steps: int,
    description: str,
):
    """Store the current live step for a subject in a shared status map."""
    if status_map is None:
        return
    status_map[subject] = f"Step {step_num}/{total_steps}: {description}"


def fail_subject(
    status_map,
    failure_log_path,
    failure_lock,
    subject: str,
    status: str,
    step_num: int,
    total_steps: int,
    description: str,
    reason: str,
    t_start: float,
):
    """Record and print a subject failure with step-level context."""
    step_label = f"Step {step_num}/{total_steps}: {description}"
    failure_message = f"{step_label} | {reason}"
    error_detail = get_last_error()
    if status_map is not None:
        status_map[subject] = f"FAILED at {step_label}"
    append_failure_log(
        failure_log_path,
        failure_lock,
        subject,
        status,
        step_label,
        reason,
        error_detail,
    )
    print(f"  [FAIL]  {subject} — {step_label} — {reason}", flush=True)
    if error_detail:
        print(f"           detail: {error_detail}", flush=True)
    return (subject, status, failure_message, time.time() - t_start)


def render_live_progress(
    completed: int,
    total: int,
    failed_count: int,
    eta_seconds: float,
    status_map,
    previous_lines: int = 0,
):
    """Render a live snapshot of active subjects and their current steps."""
    remaining = total - completed
    lines = [
        f"\n[Live Progress] Done: {completed} | Remaining: {remaining} | "
        f"Failed: {failed_count} | ETA: {eta_seconds/60:.1f} min"
    ]

    active_subjects = sorted(status_map.items()) if status_map is not None else []
    if not active_subjects:
        lines.append("  Active subjects: none")
    else:
        lines.append("  Active subjects:")
        for subject, status in active_subjects:
            lines.append(f"    {subject}: {status}")

    if sys.stdout.isatty():
        if previous_lines > 0:
            sys.stdout.write(f"\x1b[{previous_lines}F")
        sys.stdout.write("\x1b[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
    else:
        print("\n".join(lines), flush=True)

    return len(lines)


def detect_pet_format(pet_dir: Path) -> str:
    """
    Detect PET file format in a directory.

    Returns: 'dcm', 'ecat', 'hrrt', or 'unknown'
    """
    files = list(pet_dir.iterdir())
    extensions = {f.suffix.lower() for f in files if f.is_file()}

    has_dcm = bool(extensions & {".dcm"})
    has_ecat = bool(extensions & {".v", ".img"})
    # HRRT: check for .l64, .s, .sino, .hdr (standard Interfile), or .i, .i.hdr (ADNI HRRT variant)
    has_hrrt = bool(extensions & {".l64", ".s", ".sino", ".hdr", ".i"})

    detected = {k for k, v in {"dcm": has_dcm, "ecat": has_ecat, "hrrt": has_hrrt}.items() if v}
    if len(detected) > 1:
        return "unknown"

    if has_dcm:
        dcm_files = [f for f in files if f.suffix.lower() == ".dcm"]
        if dcm_files:
            return "dcm"

    if has_ecat:
        return "ecat"

    if has_hrrt:
        return "hrrt"

    return "unknown"


# Format conversion
def convert_dicom(dicom_dir: Path, out_dir: Path, prefix: str = "raw") -> Path:
    """Convert DICOM folder to NIfTI using dcm2niix."""
    set_last_error(None)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcm2niix_cmd = resolve_dcm2niix()
    if dcm2niix_cmd is None:
        set_last_error(
            "dcm2niix executable not found. Set DCM2NIIX or install dcm2niix."
        )
        print("    [ERROR] dcm2niix executable not found")
        return None

    cmd = [dcm2niix_cmd, "-z", "y", "-f", prefix, "-o", str(out_dir), str(dicom_dir)]
    print(f"    [dcm2niix] Converting DICOM: {dicom_dir.name}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        set_last_error(
            result.stderr
            or result.stdout
            or f"dcm2niix failed with exit code {result.returncode}"
        )
        print(f"    [ERROR] dcm2niix failed:\n{result.stderr}")
        return None

    nifti_files = sorted(out_dir.glob(f"{prefix}*.nii.gz"))
    if not nifti_files:
        set_last_error("No .nii.gz found after dcm2niix")
        print(f"    [ERROR] No .nii.gz found after dcm2niix")
        return None

    out_file = out_dir / f"{prefix}.nii.gz"
    if not out_file.exists():
        nifti_files[0].rename(out_file)

    print(f"    [OK] DICOM → {out_file}")
    return out_file


def convert_ecat(ecat_dir: Path, out_dir: Path, prefix: str = "raw") -> Path:
    """
    Convert ECAT7 (.v or .img) to NIfTI.
    Uses nibabel.ecat loader specifically for ECAT7 (MATRIX72v magic bytes).
    Falls back to nipype if nibabel fails.
    """
    import nibabel as nib
    import nibabel.ecat as ecat
    import numpy as np

    set_last_error(None)
    out_dir.mkdir(parents=True, exist_ok=True)

    ecat_files = list(ecat_dir.glob("*.v")) + list(ecat_dir.glob("*.img"))
    if not ecat_files:
        set_last_error(f"No .v or .img ECAT files found in {ecat_dir}")
        print(f"    [ERROR] No .v or .img ECAT files found in {ecat_dir}")
        return None

    ecat_file = ecat_files[0]
    out_file = out_dir / f"{prefix}.nii.gz"

    print(f"    [nibabel.ecat] Converting ECAT7: {ecat_file.name}")
    try:
        ecat_img = ecat.load(str(ecat_file))

        n_frames = ecat_img.shape[3] if len(ecat_img.shape) == 4 else 1
        print(f"    [INFO] ECAT7 frames: {n_frames}, shape: {ecat_img.shape}")

        data = np.array(ecat_img.get_fdata())
        if data.ndim == 4:
            print(f"    [INFO] 4D ECAT detected ({data.shape[3]} frames), averaging...")
            data = data.mean(axis=3)

        nifti_img = nib.Nifti1Image(data, ecat_img.affine)
        nib.save(nifti_img, str(out_file))
        print(f"    [OK] ECAT → {out_file}")
        return out_file

    except Exception as e:
        set_last_error(str(e))
        print(f"    [WARN] nibabel ECAT load failed: {e}")
        print(f"    [nipype] Trying nipype ecatconvert...")

        try:
            from nipype.interfaces.freesurfer import MRIConvert

            mc = MRIConvert()
            mc.inputs.in_file = str(ecat_file)
            mc.inputs.out_file = str(out_file)
            mc.inputs.out_type = "niigz"
            mc.run()
            print(f"    [OK] ECAT → {out_file} (via nipype)")
            return out_file
        except Exception as e2:
            set_last_error(str(e2))
            print(f"    [ERROR] nipype conversion also failed: {e2}")
            return None


def parse_interfile_header(hdr_path: Path) -> dict:
    """
    Parse ADNI HRRT Interfile header and return metadata dict.
    Returns: {'matrix_size': (x, y, z), 'voxel_size': (x, y, z), 'data_file': path, ...}
    """
    import re
    metadata = {}
    
    try:
        with open(hdr_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('!'):
                    continue
                
                if ':=' in line:
                    key, value = line.split(':=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key == 'name of data file':
                        if value:
                            data_file = hdr_path.parent / value
                            metadata['data_file'] = data_file
                    
                    elif 'matrix size' in key:
                        idx = re.search(r'\[(\d+)\]', key)
                        if idx and value:
                            dim = int(idx.group(1)) - 1
                            if 'matrix_size' not in metadata:
                                metadata['matrix_size'] = [1, 1, 1]
                            try:
                                metadata['matrix_size'][dim] = int(value)
                            except:
                                pass
                    
                    elif 'scaling factor' in key and 'pixel' in key:
                        idx = re.search(r'\[(\d+)\]', key)
                        if idx and value:
                            dim = int(idx.group(1)) - 1
                            if 'voxel_size' not in metadata:
                                metadata['voxel_size'] = [1.0, 1.0, 1.0]
                            try:
                                metadata['voxel_size'][dim] = float(value)
                            except:
                                pass
                    
                    elif key == 'number of bytes per pixel':
                        try:
                            metadata['bytes_per_pixel'] = int(value)
                        except:
                            pass
    except Exception as e:
        set_last_error(f"Error parsing Interfile header: {e}")
        return None
    
    return metadata if metadata else None


def load_interfile_data(metadata: dict, out_dir: Path, prefix: str) -> Path:
    """
    Load raw ADNI HRRT Interfile data and convert to NIfTI.
    Uses geometry from header metadata.
    """
    import nibabel as nib
    import numpy as np
    
    try:
        data_file = metadata.get('data_file')
        matrix_size = metadata.get('matrix_size', [1, 1, 1])
        voxel_size = metadata.get('voxel_size', [1.0, 1.0, 1.0])
        bytes_per_pixel = metadata.get('bytes_per_pixel', 4)
        
        # If the exact file from header doesn't exist, try to find a matching .i file
        if not data_file or not data_file.exists():
            parent_dir = data_file.parent if data_file else None
            if not parent_dir:
                set_last_error(f"Data file not found: {data_file}")
                return None
            
            i_files = list(parent_dir.glob("*.i"))
            if not i_files:
                set_last_error(f"No .i data files found in {parent_dir}")
                return None
            
            # Use the first .i file found
            data_file = i_files[0]
            print(f"    [INFO] Using data file: {data_file.name}")
        
        with open(data_file, 'rb') as f:
            raw_data = f.read()
        
        if bytes_per_pixel == 4:
            dtype = np.float32
        elif bytes_per_pixel == 2:
            dtype = np.int16
        else:
            dtype = np.float32
        
        data = np.frombuffer(raw_data, dtype=dtype)
        try:
            data = data.reshape(matrix_size)
        except ValueError as e:
            set_last_error(f"Cannot reshape data to {matrix_size}: {e}")
            return None
        
        affine = np.eye(4)
        np.fill_diagonal(affine[:3, :3], voxel_size)
        
        out_file = out_dir / f"{prefix}.nii.gz"
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, str(out_file))
        
        print(f"    [OK] HRRT (custom parser) → {out_file} (shape: {data.shape})")
        return out_file
    
    except Exception as e:
        set_last_error(str(e))
        print(f"    [ERROR] Failed to load Interfile data: {e}")
        return None


def convert_hrrt(hrrt_dir: Path, out_dir: Path, prefix: str = "raw") -> Path:
    """
    Convert HRRT format to NIfTI.
    Supports:
      - Standard Interfile: .hdr/.img pairs
      - ADNI HRRT variant: .i/.i.hdr pairs
      - HRRT listmode/sinogram: .l64, .s files (requires external reconstruction)
    """
    import nibabel as nib

    set_last_error(None)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try standard Interfile (.hdr/.img)
    hdr_files = list(hrrt_dir.glob("*.hdr"))
    img_files = list(hrrt_dir.glob("*.img"))

    if hdr_files and img_files:
        print(f"    [nibabel] Converting HRRT Interfile (.hdr/.img): {hdr_files[0].name}")
        try:
            hrrt_img = nib.load(str(hdr_files[0]))
            data = hrrt_img.get_fdata()
            if data.ndim == 4:
                print(f"    [INFO] 4D HRRT detected, averaging frames...")
                data = data.mean(axis=3)
            out_file = out_dir / f"{prefix}.nii.gz"
            nib.save(nib.Nifti1Image(data, hrrt_img.affine), str(out_file))
            print(f"    [OK] HRRT → {out_file}")
            return out_file
        except Exception as e:
            set_last_error(str(e))
            print(f"    [WARN] Standard Interfile load failed: {e}")

    # Try ADNI HRRT variant (.i/.i.hdr) with custom parser
    i_hdr_files = list(hrrt_dir.glob("*.i.hdr"))
    i_files = list(hrrt_dir.glob("*.i"))

    if i_hdr_files and i_files:
        print(f"    [custom parser] Converting HRRT Interfile (.i/.i.hdr): {i_hdr_files[0].name}")
        metadata = parse_interfile_header(i_hdr_files[0])
        if metadata:
            result = load_interfile_data(metadata, out_dir, prefix)
            if result:
                return result
        print(f"    [WARN] Custom parser failed, trying nibabel fallback...")
        try:
            hrrt_img = nib.load(str(i_hdr_files[0]))
            data = hrrt_img.get_fdata()
            if data.ndim == 4:
                print(f"    [INFO] 4D HRRT detected, averaging {data.shape[3]} frames...")
                data = data.mean(axis=3)
            out_file = out_dir / f"{prefix}.nii.gz"
            nib.save(nib.Nifti1Image(data, hrrt_img.affine), str(out_file))
            print(f"    [OK] HRRT → {out_file}")
            return out_file
        except Exception as e:
            set_last_error(str(e))
            print(f"    [WARN] ADNI HRRT Interfile load failed: {e}")

    s_files = list(hrrt_dir.glob("*.s"))
    l64_files = list(hrrt_dir.glob("*.l64"))

    if s_files or l64_files:
        print(f"    [INFO] Raw HRRT listmode/sinogram detected.")
        print(f"    [WARN] Raw HRRT reconstruction requires dedicated tools")
        print(f"           (e.g., lm-recon, e7tools). Attempting dcm2niix fallback...")
        return convert_dicom(hrrt_dir, out_dir, prefix)

    set_last_error(f"Could not identify HRRT file type in {hrrt_dir}")
    print(f"    [ERROR] Could not identify HRRT file type in {hrrt_dir}")
    return None


def convert_pet(pet_dir: Path, out_dir: Path) -> Path:
    """Auto-detect PET format and convert to NIfTI."""
    set_last_error(None)
    fmt = detect_pet_format(pet_dir)
    print(f"    [PET] Detected format: {fmt.upper()}")

    if fmt == "dcm":
        return convert_dicom(pet_dir, out_dir, prefix="raw")
    elif fmt == "ecat":
        return convert_ecat(pet_dir, out_dir, prefix="raw")
    elif fmt == "hrrt":
        return convert_hrrt(pet_dir, out_dir, prefix="raw")
    else:
        set_last_error(f"Unknown PET format in {pet_dir}")
        print(f"    [ERROR] Unknown PET format in {pet_dir}")
        return None


# MRI preprocessing
def n4_bias_correction(nifti_path: Path, out_dir: Path) -> Path:
    """
    Apply N4 bias field correction to sMRI using SimpleITK.
    Corrects intensity non-uniformity from magnetic field inhomogeneity.
    """
    import SimpleITK as sitk

    set_last_error(None)
    out_file = out_dir / "n4_corrected.nii.gz"
    print(f"    [N4] Applying N4 bias field correction...")

    try:
        img = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.SetConvergenceThreshold(0.001)

        corrected = corrector.Execute(img, mask)
        sitk.WriteImage(corrected, str(out_file))
        print(f"    [OK] N4 correction done → {out_file}")
        return out_file

    except Exception as e:
        set_last_error(str(e))
        print(f"    [ERROR] N4 correction failed: {e}")
        return None


def skull_strip(nifti_path: Path, out_dir: Path, hd_bet_semaphore=None) -> Path:
    """
    Strip skull from sMRI using HD-BET (deep-learning).

    HD-BET is invoked through its command-line interface.
    """
    set_last_error(None)
    out_file = out_dir / "brain.nii.gz"

    print(f"    [HD-BET] Running deep-learning skull stripping...")

    hd_bet_cmd = shutil.which("hd-bet")
    if hd_bet_cmd is None:
        set_last_error("hd-bet executable not found in PATH")
        raise RuntimeError("hd-bet executable not found in PATH")

    try:
        import torch

        if not torch.cuda.is_available():
            set_last_error("CUDA is not available in the current Python environment")
            raise RuntimeError("CUDA is not available in the current Python environment")
        device = "cuda"
    except Exception as e:
        set_last_error(str(e))
        raise RuntimeError(f"HD-BET requires a CUDA GPU, but GPU setup failed: {e}")

    cmd = [
        hd_bet_cmd,
        "-i",
        str(nifti_path),
        "-o",
        str(out_file),
        "-device",
        device,
    ]

    acquired_semaphore = False
    if hd_bet_semaphore is not None:
        print("    [HD-BET] Waiting for an available HD-BET slot...", flush=True)
        hd_bet_semaphore.acquire()
        acquired_semaphore = True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        set_last_error(str(e))
        raise
    finally:
        if acquired_semaphore:
            hd_bet_semaphore.release()

    if result.returncode != 0:
        error_detail = (
            result.stderr
            or result.stdout
            or describe_return_code(result.returncode, "hd-bet command")
        )
        set_last_error(
            error_detail
        )
        raise RuntimeError(describe_return_code(result.returncode, "hd-bet command"))

    if out_file.exists():
        print(f"    [OK] HD-BET skull stripping done → {out_file}")
        return out_file

    set_last_error(f"HD-BET output missing: expected {out_file.name} in {out_dir}")
    raise RuntimeError(
        f"HD-BET ran but no output found in {out_dir}. "
        f"Expected {out_file.name}."
    )


def normalize_intensity(nifti_path: Path, out_dir: Path) -> Path:
    """
    Standardize sMRI intensities across subjects using WhiteStripe
    normalization so that ROI means are on a comparable scale regardless
    of scanner or acquisition protocol.

    WhiteStripe identifies the white-matter signal peak and normalizes to it.
    Reference: Shinohara et al., 2014

    WhiteStripe is required — no fallback.
    Install: pip install intensity-normalization
    """
    set_last_error(None)
    out_file = out_dir / "normalized.nii.gz"
    print(f"    [Normalize] Applying WhiteStripe intensity normalization...")

    normalize_cmd = shutil.which("intensity-normalize")
    if normalize_cmd is None:
        set_last_error("intensity-normalize executable not found in PATH")
        raise RuntimeError("intensity-normalize executable not found in PATH")

    cmd = [
        normalize_cmd,
        "whitestripe",
        str(nifti_path),
        "-o",
        str(out_file),
        "--modality",
        "t1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        set_last_error(
            result.stderr
            or result.stdout
            or f"intensity-normalize failed with exit code {result.returncode}"
        )
        raise RuntimeError(
            f"intensity-normalize failed with exit code {result.returncode}"
        )
    if not out_file.exists():
        set_last_error(f"WhiteStripe finished but output was not created: {out_file}")
        raise RuntimeError(f"WhiteStripe finished but output was not created: {out_file}")

    print(f"    [OK] WhiteStripe normalization done → {out_file}")
    return out_file


# Registration
def register_smri(brain_path: Path, out_dir: Path) -> tuple:
    """
    Register skull-stripped sMRI to MNI152 using FSL flirt + fnirt.
    Returns (registered_path, warp_path) for use in PET registration.
    """
    affine_out = out_dir / "affine.nii.gz"
    affine_matrix = out_dir / "affine.mat"
    warp_out = out_dir / "warp.nii.gz"
    registered = out_dir / "registered.nii.gz"

    print(f"    [flirt] Affine registration to MNI152...")
    flirt_cmd = [
        "flirt",
        "-in",
        str(brain_path),
        "-ref",
        MNI_TEMPLATE,
        "-out",
        str(affine_out),
        "-omat",
        str(affine_matrix),
        "-dof",
        "12",
        "-interp", "trilinear",
        "-cost", "corratio",
    ]
    if not run_cmd(flirt_cmd, "flirt"):
        return None, None

    print(f"    [fnirt] Nonlinear registration (this takes a few minutes)...")
    fnirt_cmd = [
        "fnirt",
        f"--in={brain_path}",
        f"--aff={affine_matrix}",
        f"--ref={MNI_TEMPLATE}",
        f"--refmask={MNI_MASK}",
        f"--iout={registered}",
        f"--cout={warp_out}",
        "--subsamp=8,4,2,2",
        "--miter=5,5,5,5",
    ]
    if not run_cmd(fnirt_cmd, "fnirt"):
        return None, None

    print(f"    [OK] sMRI registered → {registered}")
    return registered, warp_out


def register_pet(
    pet_nifti: Path, smri_brain: Path, smri_warp: Path, out_dir: Path
) -> Path:
    """
    Register PET to MNI152:
    1. PET → subject sMRI (flirt, 6 DOF rigid)
    2. Apply sMRI warp to bring PET into MNI152 (applywarp)
    """
    pet_to_smri_out = out_dir / "pet_to_smri.nii.gz"
    pet_to_smri_matrix = out_dir / "pet_to_smri.mat"
    registered = out_dir / "registered.nii.gz"

    print(f"    [flirt] Registering PET → subject sMRI...")
    flirt_cmd = [
        "flirt",
        "-in",
        str(pet_nifti),
        "-ref",
        str(smri_brain),
        "-out",
        str(pet_to_smri_out),
        "-omat",
        str(pet_to_smri_matrix),
        "-dof",
        "6",
        "-interp", "trilinear",
        "-cost", "mutualinfo",
    ]
    if not run_cmd(flirt_cmd, "flirt PET→sMRI"):
        return None

    print(f"    [applywarp] Warping PET to MNI152...")
    applywarp_cmd = [
        "applywarp",
        f"--in={pet_to_smri_out}",
        f"--ref={MNI_TEMPLATE}",
        f"--warp={smri_warp}",
        f"--out={registered}",
        "--interp=trilinear"
    ]
    if not run_cmd(applywarp_cmd, "applywarp"):
        return None

    print(f"    [OK] PET registered to MNI152 → {registered}")
    return registered


def suvr_normalize(
    pet_mni_path: Path, out_dir: Path, cerebellum_mask_path: str = None
) -> Path:
    """
    Compute SUVR (Standardized Uptake Value Ratio) by dividing the
    PET image by the mean uptake in a reference region (cerebellum).

    Why this matters:
        Raw PET uptake values are scanner- and dose-dependent — completely
        non-comparable across subjects without this step. SUVR normalization
        produces dimensionless, biologically meaningful values.

    Reference region: cerebellum (grey matter)
        - Standard for amyloid, tau, and FDG PET studies
        - Assumed to have minimal specific binding / stable metabolism

    Args:
        pet_mni_path:        PET image already in MNI152 space
        out_dir:             Output directory
        cerebellum_mask_path: Path to binary cerebellum mask in MNI152 space.
                              Defaults to FSL's Cerebellum atlas (25% threshold).

    Returns:
        Path to SUVR-normalized PET image
    """
    import nibabel as nib
    import numpy as np

    set_last_error(None)
    out_file = out_dir / "suvr.nii.gz"
    print(f"    [SUVR] Computing SUVR normalization...")

    # Resolve cerebellum mask
    mask_path = cerebellum_mask_path or CEREBELLUM_MASK_DEFAULT
    if not os.path.exists(mask_path):
        set_last_error(f"Cerebellum mask not found: {mask_path}")
        print(f"    [ERROR] Cerebellum mask not found: {mask_path}")
        print(f"            Check FSLDIR or pass --cerebellum-mask explicitly")
        return None

    try:
        pet_img = nib.load(str(pet_mni_path))
        pet_data = pet_img.get_fdata().astype(float)

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()

        # Atlas masks may store probabilities rather than binary values.
        cerebellum_mask = mask_data > 0

        if pet_data.shape != cerebellum_mask.shape:
            print(
                f"    [WARN] Shape mismatch: PET {pet_data.shape} vs mask {cerebellum_mask.shape}"
            )
            print(f"           Resampling mask to PET space...")
            from nilearn.image import resample_to_img

            mask_resampled = resample_to_img(mask_img, pet_img, interpolation="nearest")
            cerebellum_mask = mask_resampled.get_fdata() > 0

        ref_voxels = pet_data[cerebellum_mask]

        if ref_voxels.size == 0:
            set_last_error("No voxels found in cerebellum mask; check mask alignment")
            print(f"    [ERROR] No voxels found in cerebellum mask — check mask alignment")
            return None

        ref_mean = ref_voxels.mean()
        print(f"    [SUVR] Cerebellum reference mean uptake: {ref_mean:.4f}")

        if ref_mean < 1e-6:
            set_last_error("Cerebellum reference mean is near zero; cannot normalize")
            print(f"    [ERROR] Cerebellum reference mean is near zero — cannot normalize")
            return None

        suvr_data = pet_data / ref_mean

        nib.save(
            nib.Nifti1Image(suvr_data, pet_img.affine, pet_img.header), str(out_file)
        )
        print(f"    [OK] SUVR normalization done (ref={ref_mean:.4f}) → {out_file}")
        return out_file

    except Exception as e:
        set_last_error(str(e))
        print(f"    [ERROR] SUVR normalization failed: {e}")
        return None



# Post-processing and QC
def smooth_image(
    nifti_path: Path,
    out_dir: Path,
    fwhm_mm: float = 5.0,
    label: str = "smoothed",
) -> Path:
    """
    Apply Gaussian spatial smoothing to a NIfTI image.

    Reduces voxel-level noise and improves ROI statistic stability.
    Applied after registration so smoothing is in MNI space.

    Args:
        nifti_path : input image (already in MNI space)
        out_dir    : output directory
        fwhm_mm    : smoothing kernel FWHM in mm (default 5mm — good for 1mm MNI)
        label      : output filename prefix ('smoothed_smri' or 'smoothed_pet')

    Returns:
        Path to smoothed image
    """
    from nilearn.image import smooth_img
    import nibabel as nib

    out_file = out_dir / f"{label}.nii.gz"
    print(f"    [Smooth] Applying {fwhm_mm}mm FWHM Gaussian smoothing...")

    img = nib.load(str(nifti_path))
    smoothed = smooth_img(img, fwhm=fwhm_mm)
    nib.save(smoothed, str(out_file))

    print(f"    [OK] Smoothing done → {out_file}")
    return out_file


def save_qc_overlay(
    bg_path: Path, overlay_path: Path, out_dir: Path, label: str
) -> Path:
    """
    Save a 3-plane (axial / coronal / sagittal) QC PNG showing
    the overlay image on top of the background image.

    Usage:
        sMRI QC : bg = MNI template,  overlay = registered sMRI
        PET QC  : bg = registered sMRI, overlay = registered/SUVR PET

    The overlay is shown at 40% alpha so both images are visible.
    Three representative slices are chosen automatically at 33/50/66%
    of each axis to avoid edge-of-brain slices.

    Args:
        bg_path      : background NIfTI (e.g. MNI template or sMRI)
        overlay_path : foreground NIfTI to check alignment
        out_dir      : directory to save PNG
        label        : filename label, e.g. 'smri_on_mni' or 'pet_on_smri'

    Returns:
        Path to saved PNG
    """
    import nibabel as nib
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn.image import resample_to_img

    out_file = out_dir / f"qc_{label}.png"
    print(f"    [QC] Saving overlay: {label}...")

    bg_img = nib.load(str(bg_path))
    ov_img = nib.load(str(overlay_path))

    if bg_img.shape != ov_img.shape:
        ov_img = resample_to_img(ov_img, bg_img, interpolation="continuous")

    bg_data = bg_img.get_fdata()
    ov_data = ov_img.get_fdata()

    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    bg_data = norm(bg_data)
    ov_data = norm(ov_data)

    def mid_slices(shape):
        return [int(shape * p) for p in (0.33, 0.50, 0.66)]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor="black")
    fig.suptitle(label.replace("_", " ").upper(), color="white", fontsize=13)

    planes = [
        ("Axial", 2, lambda d, i: d[:, :, i]),
        ("Coronal", 1, lambda d, i: d[:, i, :]),
        ("Sagittal", 0, lambda d, i: d[i, :, :]),
    ]

    for row, (plane_name, axis, slicer) in enumerate(planes):
        slices = mid_slices(bg_data.shape[axis])
        for col, sl in enumerate(slices):
            ax = axes[row][col]
            ax.imshow(
                np.rot90(slicer(bg_data, sl)),
                cmap="gray",
                origin="lower",
                interpolation="nearest",
            )
            ax.imshow(
                np.rot90(slicer(ov_data, sl)),
                cmap="hot",
                alpha=0.4,
                origin="lower",
                interpolation="nearest",
            )
            ax.set_title(f"{plane_name} sl={sl}", color="white", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(out_file), dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    print(f"    [OK] QC overlay saved → {out_file}")
    return out_file


# Main pipeline
def process_subject(args):
    """
    Process one subject - designed to run in a separate process.
    Returns (subject, status, message, elapsed_seconds)
    status: 'smri_and_pet' | 'smri_only' | 'pet_failed' | 'failed'

    sMRI pipeline:
        DICOM → NIfTI → N4 → HD-BET → intensity normalize → register MNI → smooth → QC

    PET pipeline:
        format → NIfTI → register MNI (via sMRI warp) → SUVR normalize → smooth → QC
    """
    (
        subject,
        has_pet,
        smri_dir_str,
        pet_dir_str,
        output_str,
        cerebellum_mask,
        status_map,
        failure_log_path,
        failure_lock,
        hd_bet_semaphore,
    ) = args

    try:
        if not VERBOSE_SUBJECT_LOGS:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                return _process_subject_impl(args)
        return _process_subject_impl(args)
    except Exception as e:
        if status_map is not None:
            status_map[subject] = f"FAILED: {e}"
        set_last_error(str(e))
        return fail_subject(
            status_map,
            failure_log_path,
            failure_lock,
            subject,
            "failed",
            1,
            1,
            "process_subject wrapper",
            str(e),
            0.0,
        )
    finally:
        if status_map is not None and subject in status_map:
            del status_map[subject]


def _process_subject_impl(args):
    (
        subject,
        has_pet,
        smri_dir_str,
        pet_dir_str,
        output_str,
        cerebellum_mask,
        status_map,
        failure_log_path,
        failure_lock,
        hd_bet_semaphore,
    ) = args

    smri_dir = Path(smri_dir_str)
    pet_dir = Path(pet_dir_str)
    output   = Path(output_str)

    smri_dcm = smri_dir / subject
    pet_input = pet_dir  / subject if has_pet else None
    smri_out = output / "sMRI" / subject
    pet_out = output / "PET" / subject

    smri_out.mkdir(parents=True, exist_ok=True)

    total_steps = 10 if has_pet else 6
    mode_label = "sMRI + PET" if has_pet else "sMRI only"
    current_step_num = 0
    current_step_desc = "initialization"

    t_start = time.time()
    print(f"\n{'─'*55}", flush=True)
    print(f"  [START] {subject}  ({mode_label})", flush=True)
    print(f"{'─'*55}", flush=True)

    try:
        current_step_num = 1
        current_step_desc = "DICOM -> NIfTI (dcm2niix)"
        update_subject_status(status_map, subject, 1, total_steps, current_step_desc)
        log_step(subject, 1, total_steps, "DICOM → NIfTI (dcm2niix)")
        smri_nifti = convert_dicom(smri_dcm, smri_out, prefix="raw")
        if smri_nifti is None:
            return fail_subject(
                status_map,
                failure_log_path,
                failure_lock,
                subject,
                "failed",
                current_step_num,
                total_steps,
                current_step_desc,
                "sMRI conversion failed",
                t_start,
            )

        current_step_num = 2
        current_step_desc = "N4 bias field correction"
        update_subject_status(status_map, subject, 2, total_steps, current_step_desc)
        log_step(subject, 2, total_steps, current_step_desc)
        smri_n4 = n4_bias_correction(smri_nifti, smri_out)
        if smri_n4 is None:
            smri_n4 = smri_nifti  # fallback to raw

        current_step_num = 3
        current_step_desc = "HD-BET skull stripping"
        update_subject_status(status_map, subject, 3, total_steps, current_step_desc)
        log_step(subject, 3, total_steps, current_step_desc)
        smri_brain = skull_strip(smri_n4, smri_out, hd_bet_semaphore=hd_bet_semaphore)

        current_step_num = 4
        current_step_desc = "WhiteStripe intensity normalization"
        update_subject_status(status_map, subject, 4, total_steps, current_step_desc)
        log_step(subject, 4, total_steps, current_step_desc)
        smri_normalized = normalize_intensity(smri_brain, smri_out)

        current_step_num = 5
        current_step_desc = "Registration to MNI152 (flirt + fnirt)"
        update_subject_status(status_map, subject, 5, total_steps, current_step_desc)
        log_step(subject, 5, total_steps, current_step_desc)
        smri_registered, smri_warp = register_smri(smri_normalized, smri_out)
        if smri_registered is None:
            return fail_subject(
                status_map,
                failure_log_path,
                failure_lock,
                subject,
                "failed",
                current_step_num,
                total_steps,
                current_step_desc,
                "sMRI registration failed",
                t_start,
            )

        current_step_num = 6
        current_step_desc = "Gaussian smoothing (5mm FWHM)"
        update_subject_status(status_map, subject, 6, total_steps, current_step_desc)
        log_step(subject, 6, total_steps, current_step_desc)
        smooth_image(smri_registered, smri_out, fwhm_mm=5.0, label="smoothed_smri")

        if not has_pet:
            elapsed = time.time() - t_start
            print(f"  [DONE]  {subject} — sMRI only — {elapsed/60:.1f} min", flush=True)
            return (subject, "smri_only", "OK", elapsed)

        pet_out.mkdir(parents=True, exist_ok=True)

        current_step_num = 7
        current_step_desc = "PET format detection + NIfTI conversion"
        update_subject_status(status_map, subject, 7, total_steps, current_step_desc)
        log_step(subject, 7, total_steps, current_step_desc)
        pet_nifti = convert_pet(pet_input, pet_out)
        if pet_nifti is None:
            return fail_subject(
                status_map,
                failure_log_path,
                failure_lock,
                subject,
                "pet_failed",
                current_step_num,
                total_steps,
                current_step_desc,
                "PET conversion failed",
                t_start,
            )

        current_step_num = 8
        current_step_desc = "PET registration to MNI152 (via sMRI warp)"
        update_subject_status(status_map, subject, 8, total_steps, current_step_desc)
        log_step(subject, 8, total_steps, current_step_desc)
        pet_registered = register_pet(pet_nifti, smri_brain, smri_warp, pet_out)
        if pet_registered is None:
            return fail_subject(
                status_map,
                failure_log_path,
                failure_lock,
                subject,
                "pet_failed",
                current_step_num,
                total_steps,
                current_step_desc,
                "PET registration failed",
                t_start,
            )

        current_step_num = 9
        current_step_desc = "SUVR normalization (cerebellum reference)"
        update_subject_status(status_map, subject, 9, total_steps, current_step_desc)
        log_step(subject, 9, total_steps, current_step_desc)
        pet_suvr = suvr_normalize(pet_registered, pet_out, cerebellum_mask)
        if pet_suvr is None:
            return fail_subject(
                status_map,
                failure_log_path,
                failure_lock,
                subject,
                "pet_failed",
                current_step_num,
                total_steps,
                current_step_desc,
                "SUVR normalization failed",
                t_start,
            )

        current_step_num = 10
        current_step_desc = "Gaussian smoothing + QC overlays"
        update_subject_status(status_map, subject, 10, total_steps, current_step_desc)
        log_step(subject, 10, total_steps, current_step_desc)
        smooth_image(pet_suvr, pet_out, fwhm_mm=5.0, label="smoothed_pet")
        save_qc_overlay(Path(MNI_TEMPLATE), smri_registered, smri_out, "smri_on_mni")
        save_qc_overlay(smri_registered, pet_suvr, pet_out, "pet_on_smri")

        elapsed = time.time() - t_start
        print(f"  [DONE]  {subject} — sMRI + PET — {elapsed/60:.1f} min", flush=True)
        return (subject, "smri_and_pet", "OK", elapsed)
    except Exception as e:
        return fail_subject(
            status_map,
            failure_log_path,
            failure_lock,
            subject,
            "pet_failed" if has_pet and current_step_num >= 7 else "failed",
            current_step_num if current_step_num else 1,
            total_steps,
            current_step_desc,
            f"Unhandled exception: {e}",
            t_start,
        )


def is_subject_complete(subject: str, has_pet: bool, output_dir: Path) -> bool:
    """
    Check if a subject has already been successfully processed by verifying
    the existence of final output files (smoothed/QC images).
    
    Returns True if the subject appears to be complete.
    """
    if has_pet:
        # For sMRI+PET: check that both smoothed images exist
        smri_final = output_dir / "sMRI" / subject / "smoothed_smri.nii.gz"
        pet_final = output_dir / "PET" / subject / "smoothed_pet.nii.gz"
        return smri_final.exists() and pet_final.exists()
    else:
        # For sMRI-only: check that smoothed sMRI exists
        smri_final = output_dir / "sMRI" / subject / "smoothed_smri.nii.gz"
        return smri_final.exists()


def run_pipeline(
    dataset_dir: str,
    output_dir: str,
    n_workers: int = None,
    cerebellum_mask: str = None,
    hd_bet_workers: int = None,
):
    if n_workers is None:
        n_workers = 3
    if hd_bet_workers is None:
        hd_bet_workers = max(1, int(os.environ.get("HD_BET_MAX_WORKERS", "3")))

    dataset = Path(dataset_dir)
    output = Path(output_dir)

    smri_dir = dataset / "sMRI"
    pet_dir = dataset / "PET"

    if not smri_dir.exists():
        print(f"[ERROR] sMRI folder not found: {smri_dir}")
        sys.exit(1)

    dcm2niix_cmd = resolve_dcm2niix()
    if dcm2niix_cmd is None:
        print("[ERROR] dcm2niix executable not found.")
        print(f"        Checked DCM2NIIX, PATH, and fallback: {DCM2NIIX_FALLBACK}")
        sys.exit(1)
    print(f"[Dependency] Using dcm2niix: {dcm2niix_cmd}")

    # Resolve cerebellum mask
    _cerebellum_mask = cerebellum_mask or CEREBELLUM_MASK_DEFAULT
    if not os.path.exists(_cerebellum_mask):
        print(f"[ERROR] Cerebellum mask not found at: {_cerebellum_mask}")
        print(f"        Set CEREBELLUM_MASK in the entry point or check your FSLDIR.")
        sys.exit(1)

    smri_subjects = set(s.name for s in smri_dir.iterdir() if s.is_dir())
    pet_subjects = (
        set(s.name for s in pet_dir.iterdir() if s.is_dir()) if pet_dir.exists() else set()
    )

    both_subjects = sorted(smri_subjects & pet_subjects)
    smri_only     = sorted(smri_subjects - pet_subjects)
    pet_only      = sorted(pet_subjects  - smri_subjects)

    # Check for already-completed subjects
    completed_both = [s for s in both_subjects if is_subject_complete(s, True, output)]
    completed_smri = [s for s in smri_only if is_subject_complete(s, False, output)]
    
    # Filter to only process incomplete subjects
    both_subjects_todo = [s for s in both_subjects if s not in completed_both]
    smri_only_todo = [s for s in smri_only if s not in completed_smri]

    print(f"\n[Subject Summary]")
    print(f"  sMRI + PET : {len(both_subjects)} total")
    print(f"    ├─ already done : {len(completed_both)}")
    print(f"    └─ to process   : {len(both_subjects_todo)}")
    print(f"  sMRI only  : {len(smri_only)} total")
    print(f"    ├─ already done : {len(completed_smri)}")
    print(f"    └─ to process   : {len(smri_only_todo)}")
    print(f"  PET only   : {len(pet_only)}   (skipped — no sMRI for registration)")
    print(f"  Total to process: {len(both_subjects_todo) + len(smri_only_todo)}")

    if pet_only:
        pet_only_log = output / "pet_only_subjects.txt"
        output.mkdir(parents=True, exist_ok=True)
        pet_only_log.write_text("\n".join(pet_only))
        print(f"  PET-only subjects logged → {pet_only_log}")

    print("=" * 60)

    all_subjects = [(s, True) for s in both_subjects_todo] + [(s, False) for s in smri_only_todo]

    results = {
        "smri_and_pet": [],
        "smri_only":    [],
        "pet_failed":   [],
        "failed":       [],
        "skipped_both": completed_both,
        "skipped_smri": completed_smri,
    }
    failure_details = []
    failure_log_path = output / "preprocessing_failures.log"
    output.mkdir(parents=True, exist_ok=True)
    failure_log_path.write_text("")

    worker_args = [
        (
            subject,
            has_pet,
            str(smri_dir),
            str(pet_dir),
            str(output),
            _cerebellum_mask,
            None,
            str(failure_log_path),
            None,
            None,
        )
        for subject, has_pet in all_subjects
    ]

    total = len(worker_args)
    completed = 0
    pipeline_start = time.time()
    failed_count = 0
    last_rendered_lines = 0

    print(f"\nRunning {total} subjects with {n_workers} workers...", flush=True)
    print(
        f"Limiting concurrent HD-BET jobs to {hd_bet_workers} to reduce GPU memory pressure.",
        flush=True,
    )

    with Manager() as manager:
        status_map = manager.dict()
        failure_lock = manager.Lock()
        hd_bet_semaphore = manager.BoundedSemaphore(hd_bet_workers)
        worker_args = [
            (
                subject,
                has_pet,
                str(smri_dir),
                str(pet_dir),
                str(output),
                _cerebellum_mask,
                status_map,
                str(failure_log_path),
                failure_lock,
                hd_bet_semaphore,
            )
            for subject, has_pet in all_subjects
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(process_subject, args): args[0]
                for args in worker_args
            }

            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                for future in done:
                    subject_name = futures[future]
                    try:
                        subject, status, message, elapsed = future.result()
                        results[status].append(subject)
                    except Exception as e:
                        subject = subject_name
                        status = "failed"
                        message = str(e)
                        elapsed = 0.0
                        results["failed"].append(subject_name)

                    if status in {"failed", "pet_failed"}:
                        failure_details.append((subject, status, message))

                    if status in {"failed", "pet_failed"}:
                        failed_count += 1

                    completed += 1

                total_elapsed = time.time() - pipeline_start
                avg_time = total_elapsed / completed if completed else 0.0
                eta_seconds = avg_time * (total - completed) if completed else 0.0
                last_rendered_lines = render_live_progress(
                    completed,
                    total,
                    failed_count,
                    eta_seconds,
                    status_map,
                    previous_lines=last_rendered_lines,
                )

    print(f"\n{'='*60}")
    print(f"Pipeline complete!")
    print(f"\n  New results (this run):")
    print(f"    sMRI + PET success : {len(results['smri_and_pet'])}")
    print(f"    sMRI only success  : {len(results['smri_only'])}")
    print(f"    PET failed         : {len(results['pet_failed'])} — {results['pet_failed']}")
    print(f"    sMRI failed        : {len(results['failed'])}     — {results['failed']}")
    print(f"\n  Already completed (skipped):")
    print(f"    sMRI + PET done    : {len(results['skipped_both'])}")
    print(f"    sMRI only done     : {len(results['skipped_smri'])}")
    print(f"\n  Not processed:")
    print(f"    PET-only           : {len(pet_only)}")
    if failure_details:
        print("\n  Failure details:")
        for subject_name, status, message in failure_details:
            print(f"    {subject_name} [{status}] {message}")
    print(f"\nOutputs saved to: {output}")

    log_path = output / "preprocessing_summary.log"
    with open(log_path, "w") as f:
        f.write(f"This run:\n")
        f.write(f"  sMRI + PET success : {len(results['smri_and_pet'])}\n")
        f.write(f"  sMRI only success  : {len(results['smri_only'])}\n")
        f.write(f"  PET failed         : {results['pet_failed']}\n")
        f.write(f"  sMRI failed        : {results['failed']}\n")
        f.write(f"\nAlready completed:\n")
        f.write(f"  sMRI + PET done    : {len(results['skipped_both'])}\n")
        f.write(f"  sMRI only done     : {len(results['skipped_smri'])}\n")
        f.write(f"\nNot processed:\n")
        f.write(f"  PET-only skipped   : {pet_only}\n")
    print(f"Summary log saved → {log_path}")

    if not failure_details:
        with open(failure_log_path, "w") as f:
            f.write("No subject failures.\n")
    print(f"Failure log saved → {failure_log_path}")


# Entry point
if __name__ == "__main__":
    DATASET_DIR = "/media/user/DATADRIVE0/mri_dataset/dataset"
    OUTPUT_DIR = "/media/user/DATADRIVE0/mri_dataset/kesav/output"
    CEREBELLUM_MASK = None
    N_WORKERS = None

    run_pipeline(DATASET_DIR, OUTPUT_DIR, N_WORKERS, CEREBELLUM_MASK)
