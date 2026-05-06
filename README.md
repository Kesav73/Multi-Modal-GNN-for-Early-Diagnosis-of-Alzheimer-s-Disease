# Alzheimer's Disease Classification using Graph Neural Networks

## Project Overview

This project implements a comprehensive pipeline for **Alzheimer's Disease (AD) vs. Cognitively Normal (CN) classification** using multimodal neuroimaging data (PET and sMRI) and Graph Neural Networks (GNNs).

The pipeline processes raw neuroimaging data through multiple phases:
- **Phase 0**: Raw data preprocessing
- **Phase 2**: Brain network construction from imaging features
- **Phase 3**: Graph structure generation
- **Phase 4**: Deep learning model training with multiple architectures

## What This Project Does

### Objectives
- Classify subjects as Alzheimer's Disease (AD) or Cognitively Normal (CN)
- Leverage both structural MRI (sMRI) and PET imaging modalities
- Use Graph Neural Networks to capture brain connectivity patterns
- Provide multiple model architectures for performance comparison
- Enable reproducible analysis with standardized preprocessing

### Key Features
✓ Multimodal imaging support (PET + sMRI)  
✓ Automated preprocessing pipeline  
✓ Brain network construction with AAL atlas  
✓ Graph-based feature engineering  
✓ Multiple GNN architectures (standard, advanced, augmented, SOTA)  
✓ Comprehensive logging and feature extraction  
✓ Subject phenotypic data integration (MMSE, APOE4 status)  

## Dataset

### Data Structure
- **Subjects**: Multiple subjects identified by codes (e.g., `002_S_4171`, `005_S_0221`)
- **Modalities**: 
  - **PET**: Positron Emission Tomography scans
  - **sMRI**: Structural MRI scans
- **Phenotypic Data**: CSV files containing clinical information (MMSE scores, APOE4 status, diagnosis labels)

### Data Locations
```
data/
├── raw/                      # Original imaging data
│   ├── PET/                  # Positron Emission Tomography
│   └── sMRI/                 # Structural MRI
├── processed/                # Preprocessed imaging data
│   ├── PET/
│   └── sMRI/
└── phenotypic/               # Clinical and demographic data
    ├── phenotypic.csv
    ├── phenotypic_with_MMSCORE.csv
    └── phenotypic_with_mmse_apoe4.csv
```

## Pipeline Architecture

### Phase 0: Preprocessing
**Scripts**: `scripts/preprocessing/`
- `preprocess_linux.py`: Skull stripping, registration, normalization of raw MRI/PET
- `extract_features_linux.py`: Extract regional features using AAL (Automated Anatomical Labeling) atlas
- **Output**: Processed imaging data + feature matrices

### Phase 2: Brain Network Construction
**Script**: `scripts/phase2_brain_network/phase2_brain_network_construction.py`
- Constructs weighted brain networks from extracted features
- Uses anatomical connectivity based on AAL atlas
- Generates network matrices for each subject
- **Output**: `outputs/phase2_outputs/` - Network adjacency matrices

### Phase 3: Graph Construction
**Scripts**: `scripts/phase3_graph/`
- Converts brain networks to graph representations
- Handles multi-modal data fusion (PET + sMRI)
- Graph normalization and feature aggregation
- Multiple implementations available (phase3_graph_construction.py, phase3_graph_construction_ayush.py)
- **Output**: `outputs/phase3_outputs/` - Graph structures and features

### Phase 4: Model Training & Classification
**Scripts**: `scripts/phase4_models/`

**Available Models:**
1. **phase4_model.py** - Standard GNN baseline
2. **phase4_advanced.py** - Advanced GNN with enhanced feature learning
3. **phase4_augmented.py** - Data augmentation techniques + GNN
4. **phase4_sota_runner.py** - State-of-the-art model configurations

**Training Script**: `scripts/training/train_npy.py`
- Trains selected model architecture
- Handles train/validation/test splits
- Tracks loss, accuracy, and metrics
- Generates training plots
- **Output**: `models/best_AD_vs_CN.pt` - Best trained model

## Project Structure

```
kesav/
├── scripts/
│   ├── preprocessing/          # Data preprocessing & feature extraction
│   ├── phase2_brain_network/   # Brain network construction
│   ├── phase3_graph/           # Graph construction
│   ├── phase4_models/          # Model training & variants
│   └── training/               # Model training pipeline
├── data/
│   ├── raw/                    # Original imaging data (PET + sMRI)
│   ├── processed/              # Preprocessed data
│   ├── phenotypic/             # Clinical phenotype CSVs
│   └── cache/                  # Cached files (AAL atlas)
├── models/
│   └── best_AD_vs_CN.pt        # Trained model checkpoint
├── outputs/
│   ├── features_summary.csv    # Extracted feature statistics
│   ├── pet_only_subjects.txt   # Subjects with only PET data
│   ├── phase2_outputs/         # Network outputs
│   ├── phase3_outputs/         # Graph outputs
│   ├── phase4_outputs/         # Model predictions
│   └── training_plots/         # Training curves & visualizations
├── logs/
│   ├── output.log              # Main execution log
│   ├── preprocessing_summary.log
│   ├── preprocessing_failures.log
│   └── extraction_summary.log   # Feature extraction log
└── venv/                        # Python virtual environment
```

## Getting Started

### Requirements
- Python 3.8+
- PyTorch / PyTorch Geometric
- NumPy, SciPy, Pandas
- Scikit-learn
- Nibabel (for NIfTI image handling)
- Networkx (for graph operations)

### Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # (if available)
# OR manual install:
pip install torch torch-geometric numpy scipy pandas scikit-learn nibabel networkx
```

### Running the Pipeline

#### Step 1: Preprocess Raw Data
```bash
cd scripts/preprocessing/
python preprocess_linux.py
python extract_features_linux.py
```

#### Step 2: Build Brain Networks
```bash
cd ../phase2_brain_network/
python phase2_brain_network_construction.py
```

#### Step 3: Construct Graphs
```bash
cd ../phase3_graph/
python phase3_graph_construction.py
```

#### Step 4: Train Model
```bash
cd ../phase4_models/
# Choose a model architecture:
python phase4_model.py          # Standard baseline
python phase4_advanced.py       # Advanced model
python phase4_augmented.py      # With augmentation
python phase4_sota_runner.py    # State-of-the-art
```

#### Step 5: Train & Evaluate
```bash
cd ../training/
python train_npy.py
```

## Output Files

### Logs
- `logs/output.log` - Complete execution log
- `logs/preprocessing_summary.log` - Preprocessing statistics
- `logs/preprocessing_failures.log` - Failed subject IDs
- `logs/extraction_summary.log` - Feature extraction summary

### Results
- `outputs/features_summary.csv` - Feature statistics for all subjects
- `outputs/phase*_outputs/` - Intermediate results from each phase
- `outputs/training_plots/` - Loss curves, accuracy plots, confusion matrices
- `outputs/pet_only_subjects.txt` - Subjects with only PET modality

### Models
- `models/best_AD_vs_CN.pt` - Best performing model (PyTorch checkpoint)

## Phenotypic Data

The project uses clinical phenotypic data including:
- **MMSE Score**: Mini-Mental State Examination (cognitive assessment)
- **APOE4 Status**: Apolipoprotein E4 genotype (genetic risk factor)
- **Diagnosis**: Binary label (AD or CN)
- **Demographics**: Age, sex, education level

## Key Metrics Tracked

- **Classification Accuracy**: Percentage of correctly classified subjects
- **Sensitivity/Specificity**: AD detection rate / CN specificity
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Loss**: Cross-entropy or relevant loss metric

## Citation & References

If you use this project, cite the following papers:
- Graph Neural Networks for neuroimaging analysis
- AAL (Automated Anatomical Labeling) atlas
- Multimodal learning approaches

## Contact & Support

For questions or issues:
- Check logs in `logs/` directory
- Review preprocessing failures in `logs/preprocessing_failures.log`
- Verify data paths in `data/` directory

## License

[Specify your license here]

## Acknowledgments

- Dataset source: [ADNI or relevant imaging consortium]
- AAL Atlas: [Citation for atlas]
- Collaborators: [Team members]

---

**Last Updated**: May 6, 2026  
**Status**: Production Ready
