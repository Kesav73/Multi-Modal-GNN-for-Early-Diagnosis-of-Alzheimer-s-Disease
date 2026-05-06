# Project Configuration & Data Dictionary

## Configuration

### Model Hyperparameters

Edit these in each phase4 model script:

```python
# Common parameters
LEARNING_RATE = 0.001           # Adam optimizer learning rate
BATCH_SIZE = 64                 # Training batch size
NUM_EPOCHS = 100                # Maximum training epochs
HIDDEN_DIM = 128                # Hidden layer dimensions
DROPOUT = 0.5                   # Dropout rate
EARLY_STOPPING_PATIENCE = 20    # Stop if no improvement for N epochs
TRAIN_TEST_SPLIT = 0.8          # 80% train, 20% test
VALIDATION_SPLIT = 0.1          # 10% of training for validation
```

### Data Paths

All configured in the root directory:
```
data/raw/              # Original MRI/PET files
data/processed/        # Preprocessed outputs
data/phenotypic/       # Clinical metadata
models/                # Trained model checkpoints
outputs/               # Results and visualizations
logs/                  # Execution logs
```

---

## Data Dictionary

### Phenotypic CSV Columns

#### phenotypic.csv
Basic demographic and diagnostic information for each subject.

| Column | Type | Description |
|--------|------|-------------|
| Subject_ID | String | Unique identifier (e.g., `002_S_4171`) |
| Age | Integer | Age in years at baseline |
| Sex | Categorical | `M` (Male) or `F` (Female) |
| Group | Categorical | `AD` (Alzheimer's Disease) or `CN` (Cognitively Normal) |
| Education | Integer | Years of education |
| Visit | String | Visit code (baseline: `m00`, month 06: `m06`, etc.) |

#### phenotypic_with_MMSCORE.csv
Extends basic phenotypic data with cognitive assessment scores.

| Column | Type | Description |
|--------|------|-------------|
| Subject_ID | String | Unique subject identifier |
| ... | ... | All columns from phenotypic.csv |
| MMSE | Float | Mini-Mental State Examination score (0-30, higher = better) |
| MMSE_Date | Date | Date of MMSE assessment |
| MMSE_Status | Categorical | `Normal` (≥24), `Impaired` (<24) |

**MMSE Interpretation:**
- 24-30: Normal cognition
- 18-23: Mild cognitive impairment
- 0-17: Severe cognitive impairment

#### phenotypic_with_mmse_apoe4.csv
Most comprehensive phenotypic data including genetic risk factors.

| Column | Type | Description |
|--------|------|-------------|
| Subject_ID | String | Unique subject identifier |
| ... | ... | All columns from phenotypic_with_MMSCORE.csv |
| APOE4_Allele | Categorical | Number of ε4 alleles: `0`, `1`, `2` |
| APOE4_Status | Categorical | `Negative` (0 alleles), `Heterozygous` (1), `Homozygous` (2) |
| Genetic_Risk | Categorical | `Low` (APOE4=0), `Medium` (APOE4=1), `High` (APOE4=2) |

**APOE4 Significance:**
- ε4 is a risk factor for Alzheimer's Disease
- Each additional ε4 allele increases AD risk
- Homozygous (2 copies) = highest risk

---

## Imaging Data Dictionary

### PET (Positron Emission Tomography)
- **Modality**: Measures metabolic activity
- **Common Tracers**: FDG (Fluorodeoxyglucose) - measures glucose metabolism
- **Resolution**: ~4-6mm voxels
- **Key Findings**: Lower metabolism in AD in temporal-parietal regions
- **File Format**: NIfTI (.nii or .nii.gz)

### sMRI (Structural MRI)
- **Modality**: Anatomical brain structure imaging
- **Sequence**: T1-weighted 3D MPRAGE typically
- **Resolution**: ~1-2mm voxels
- **Key Findings**: Hippocampal atrophy in AD
- **File Format**: NIfTI (.nii or .nii.gz)

---

## AAL Atlas (Automated Anatomical Labeling)

The AAL atlas divides the brain into 116 regions (90 cortical + 26 subcortical).

### Major Regions Extracted:
- **Prefrontal Cortex**: Executive function, decision-making
- **Hippocampus**: Memory formation (typically atrophied in AD)
- **Temporal Lobe**: Language, memory
- **Parietal Lobe**: Sensory integration, spatial processing
- **Cerebellum**: Movement coordination
- **Basal Ganglia**: Motor control

### Feature Extraction:
For each subject and modality (PET/sMRI):
- Mean intensity per AAL region
- Standard deviation per region
- Other statistical measures
- **Output**: 116-dimensional feature vector per modality

---

## Processing Pipeline Details

### Preprocessing Steps (Phase 0)

1. **Skull Stripping**: Remove non-brain tissue
2. **Registration**: Align to MNI template (Montreal Neurological Institute)
3. **Normalization**: Standardize intensity values
4. **Smoothing**: Gaussian kernel (FWHM=4-8mm)
5. **Resampling**: Standardize voxel dimensions

### Network Construction (Phase 2)

1. **Feature Matrix**: 116 AAL regions × subjects
2. **Connectivity**: Compute correlations between regions
3. **Thresholding**: Keep top correlations (> 0.3 typically)
4. **Adjacency Matrix**: 116 × 116 connectivity matrix per subject

### Graph Construction (Phase 3)

1. **Node Features**: AAL region values from preprocessed data
2. **Edge Weights**: Network connectivity from Phase 2
3. **Graph Format**: PyTorch Geometric format for GNN
4. **Multimodal Fusion**: Combine PET + sMRI graphs

### Model Training (Phase 4)

1. **Input**: Graph structure + node features
2. **GNN Layers**: Learn hierarchical representations
3. **Classification**: Binary (AD vs CN) output
4. **Loss**: Cross-entropy or focal loss
5. **Metrics**: Accuracy, precision, recall, F1, AUC-ROC

---

## Output Files Description

### Features Summary (`outputs/features_summary.csv`)
Statistics about extracted features:
- Subject ID
- Mean/std of PET features
- Mean/std of sMRI features
- Data completeness flags
- Subject group (AD/CN)

### Training Plots (`outputs/training_plots/`)
- `loss_curve.png` - Training/validation loss over epochs
- `accuracy_curve.png` - Training/validation accuracy
- `confusion_matrix.png` - Prediction matrix (AD vs CN)
- `roc_curve.png` - Receiver Operating Characteristic curve
- `feature_importance.png` - Most important AAL regions

### Phase Outputs
- `phase2_outputs/` - Brain network adjacency matrices (.npy)
- `phase3_outputs/` - Graph PyTorch objects (.pt)
- `phase4_outputs/` - Model predictions and probabilities (.csv)

### Logs
- `preprocessing_summary.log` - Number of subjects processed
- `preprocessing_failures.log` - Subject IDs with missing data
- `extraction_summary.log` - Feature extraction statistics
- `output.log` - Complete execution log

---

## Environment Variables (Optional)

Create a `.env` file in the root directory:

```bash
# GPU Settings
CUDA_VISIBLE_DEVICES=0              # Use first GPU (or empty for CPU)
NUM_WORKERS=4                       # Parallel data loading

# Data Paths (optional, defaults work fine)
RAW_DATA_PATH=/media/user/DATADRIVE0/mri_dataset/kesav/data/raw
PROCESSED_DATA_PATH=/media/user/DATADRIVE0/mri_dataset/kesav/data/processed
PHENOTYPIC_PATH=/media/user/DATADRIVE0/mri_dataset/kesav/data/phenotypic

# Model Settings
BATCH_SIZE=64
LEARNING_RATE=0.001
MAX_EPOCHS=100

# Debugging
DEBUG=False
VERBOSE=True
```

Then load in Python scripts:
```python
from dotenv import load_dotenv
import os
load_dotenv()

batch_size = int(os.getenv('BATCH_SIZE', 64))
```

---

## Performance Targets

### Expected Metrics (from literature)
- **Accuracy**: 85-92% (AD vs CN classification)
- **Sensitivity**: 85-90% (correctly identify AD)
- **Specificity**: 80-90% (correctly identify CN)
- **AUC-ROC**: 0.90-0.95

### Factors Affecting Performance
- Sample size (more subjects = better)
- Image quality and preprocessing
- Model architecture complexity
- Hyperparameter tuning
- Feature engineering choices

---

## References & Resources

- **AAL Atlas**: Tzourio-Mazoyer et al. (2002)
- **Graph Neural Networks**: https://pytorch-geometric.readthedocs.io/
- **ADNI Dataset**: http://adni.loni.usc.edu/ (if applicable)
- **PyTorch**: https://pytorch.org/docs/stable/index.html

---

**Last Updated**: May 6, 2026
