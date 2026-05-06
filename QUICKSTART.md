# Quick Start Guide

Get the AD vs CN classification pipeline running in 5 minutes.

## Prerequisites
- Python 3.8 or higher
- ~50GB disk space for raw/processed data
- GPU recommended (NVIDIA with CUDA) for faster training

## 1. Setup Environment

```bash
# Navigate to project directory
cd /media/user/DATADRIVE0/mri_dataset/kesav

# Activate virtual environment
source venv/bin/activate

# OR create new venv if needed
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Troubleshooting Installation:**
- If pip cache is full: `pip cache purge && pip install --no-cache-dir -r requirements.txt`
- If CUDA errors: Install CPU-only: `pip install torch torch-geometric`
- For GPU: Ensure CUDA 11.8+ is installed

## 2. Verify Data Structure

Check that your data is organized correctly:

```bash
# Verify preprocessing
ls -la data/raw/PET | head -5        # Should show subject directories
ls -la data/raw/sMRI | head -5       # Should show subject directories

# Verify phenotypic data
ls -la data/phenotypic/              # Should see 3 CSV files

# Verify preprocessing script
ls -la scripts/preprocessing/
```

## 3. Run Preprocessing (If Needed)

Skip this if preprocessing is already done. Check if processed data exists:

```bash
# Check if already processed
ls -la data/processed/PET | head -5

# If not processed yet, run:
cd scripts/preprocessing/
python preprocess_linux.py          # ~2-4 hours depending on dataset size
python extract_features_linux.py    # ~30-60 minutes
cd ../../
```

**Monitor Progress:**
```bash
# In another terminal:
tail -f logs/preprocessing_summary.log
```

## 4. Build Network & Graph (Sequential)

```bash
# Phase 2: Brain Network Construction
cd scripts/phase2_brain_network/
python phase2_brain_network_construction.py
cd ../../

# Phase 3: Graph Construction
cd scripts/phase3_graph/
python phase3_graph_construction.py
cd ../../
```

**Expected Output:**
- `outputs/phase2_outputs/` - Network matrices
- `outputs/phase3_outputs/` - Graph structures

## 5. Train Model

Choose one model architecture:

### Option A: Standard Baseline
```bash
cd scripts/phase4_models/
python phase4_model.py
```

### Option B: Advanced Model
```bash
python phase4_advanced.py
```

### Option C: Augmented (Recommended for smaller datasets)
```bash
python phase4_augmented.py
```

### Option D: State-of-the-Art
```bash
python phase4_sota_runner.py
```

## 6. Full Training Pipeline

```bash
cd scripts/training/
python train_npy.py
```

**Monitor Training:**
```bash
# In another terminal:
tail -f logs/output.log
watch -n 5 'ls -lh outputs/training_plots/ | tail -5'
```

## Expected Results

After training completes:

✓ **Model saved**: `models/best_AD_vs_CN.pt`  
✓ **Plots generated**: `outputs/training_plots/`
  - `loss_curve.png` - Training/validation loss
  - `accuracy_curve.png` - Accuracy over epochs
  - `confusion_matrix.png` - Final predictions
  - `roc_curve.png` - ROC-AUC plot

✓ **Logs updated**: `logs/output.log`

## 7. Evaluate Results

```bash
# View training logs
tail -50 logs/output.log

# Check feature summary
head -5 outputs/features_summary.csv

# View training plots
ls -lh outputs/training_plots/
```

## Common Issues & Solutions

### Issue: Out of Memory (OOM)
```bash
# Solution: Reduce batch size
# Edit the model script and change batch_size=256 to batch_size=32
```

### Issue: "CUDA out of memory"
```bash
# Solution: Use CPU instead
# Add CUDA_VISIBLE_DEVICES="" before running
CUDA_VISIBLE_DEVICES="" python train_npy.py
```

### Issue: Missing PET/sMRI data
```bash
# Check which subjects have complete data
cat outputs/pet_only_subjects.txt

# Run with available modalities:
# Edit scripts to handle missing modalities
```

### Issue: Long preprocessing times
```bash
# Enable parallel processing if available
# Edit preprocess_linux.py: change num_workers=1 to num_workers=4
```

## Next Steps

After successful training:

1. **Evaluate Model**: Check confusion matrix and ROC-AUC
2. **Compare Architectures**: Try different models from phase4_models/
3. **Hyperparameter Tuning**: Adjust learning rate, epochs, dropout
4. **External Validation**: Test on holdout subjects
5. **Feature Importance**: Analyze which brain regions matter most

## Useful Commands

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor disk space
df -h /media/user/DATADRIVE0/

# Monitor GPU during training
nvidia-smi -l 1

# Stop training (Ctrl+C)
# Resume from checkpoint:
# Edit train_npy.py to load best_AD_vs_CN.pt at start

# View all files created in outputs
find outputs/ -type f -mtime -1    # Files modified in last day
```

## Documentation

For detailed information, see:
- **Full Documentation**: [README.md](README.md)
- **Project Structure**: [README.md#project-structure](README.md#project-structure)
- **Pipeline Details**: [README.md#pipeline-architecture](README.md#pipeline-architecture)

## Support

Check logs for errors:
```bash
# Main log
tail -100 logs/output.log | grep -i error

# Preprocessing failures
cat logs/preprocessing_failures.log

# Feature extraction issues
cat logs/extraction_summary.log
```

---

**Estimated Time**: ~5-30 minutes (depending on data size and GPU availability)  
**Status**: Ready to run!
