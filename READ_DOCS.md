# 🚀 Seismic First Break Picking - Training Guide

## Quick Start

### Prerequisites

```bash
# 1. Activate your virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# 2. Install dependencies (if not already)
pip install -r requirements.txt

# 3. Verify data is downloaded
ls data/raw/
# Should show: Halfmile3D_add_geom_sorted.hdf5, Brunswick_orig_1500ms_V2.hdf5, etc.
```

---

## 1. Preprocess Data (One-Time Setup)

Before training, you need to preprocess the raw HDF5 data into chunks.

### Preprocess a Single Dataset

```bash
# Preprocess Halfmile
python scripts/preprocess.py --config configs/halfmile.yaml --force

# Preprocess Brunswick
python scripts/preprocess.py --config configs/brunswick.yaml --force

# Preprocess Lalor
python scripts/preprocess.py --config configs/lalor.yaml --force

# Preprocess Sudbury
python scripts/preprocess.py --config configs/sudbury.yaml --force
```

### Preprocess All Datasets

```bash
# Process all datasets
for dataset in halfmile brunswick lalor sudbury; do
    echo "Processing $dataset..."
    python scripts/preprocess.py --config configs/${dataset}.yaml --force
done
```

### Verify Preprocessing

```bash
# Inspect chunks to verify data quality
python scripts/inspect_chunks.py Halfmile
```

**Expected Output:**
```
Dataset: Halfmile
Total shots: 690
Total chunks: 10
Data shape: torch.Size([69, 1578, 751])
Mask classes: [0, 1, 2]
Has strip (class 2)? True
✅ Picks are in sample range (0-750)
```

---

## 2. Train a Model

### Option A: Resume Training (Recommended)

If you have a checkpoint from a previous run:

```bash
# Find your latest checkpoint
ls -lt models/registry/PicoUNet_Halfmile_*.pt | head -1

# Resume training
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model pico \
    --resume models/registry/PicoUNet_Halfmile_epoch_3_20260903_172644.pt \
    --epochs 30 \
    --verbose \
    --log-memory \
    --num-workers 4
```

### Option B: Start Fresh Training

```bash
# Train PicoUNet on Halfmile (fastest, good for testing)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model pico \
    --epochs 10 \
    --verbose \
    --log-memory

# Train MPSLightUNet on Halfmile (recommended for MPS)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --verbose \
    --log-memory \
    --num-workers 4

# Train LightUNet on Halfmile (balanced performance)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model light \
    --epochs 30 \
    --verbose \
    --log-memory \
    --num-workers 4
```

---

## 3. Available Models

| Model | Parameters | Speed | Best For |
|-------|-----------|-------|----------|
| `pico` | 532 | Fastest | Quick testing, sanity checks |
| `tiny` | 30K | Very Fast | Development iterations |
| `nano` | 91K | Fast | Edge devices |
| `mpslight` | 1.9M | Good | **Apple Silicon (MPS) - Recommended** |
| `light` | 774K | Good | Balanced performance |
| `mobile` | 3.7M | Medium | Mobile-friendly |
| `efficient` | 9.2M | Slower | High accuracy |
| `unet` | 7.8M | Slowest | Maximum accuracy |

**Recommendation:** Start with `mpslight` on MPS or `light` on CPU/GPU.

---

## 4. Common Training Commands

### Basic Training

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30
```

### Verbose Training (with detailed logging)

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --verbose \
    --log-memory \
    --log-level DEBUG
```

### Multi-Worker Training (faster)

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --verbose \
    --num-workers 4
```

### With Custom Batch Size (if memory allows)

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --batch-size 8 \
    --epochs 30 \
    --verbose
```

### With Different Loss Function

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --loss combo \
    --epochs 30 \
    --verbose
```

### Override Learning Rate

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --learning-rate 0.0005 \
    --epochs 30 \
    --verbose
```

---

## 5. Batch Training (Multiple Datasets)

### Auto-Config Mode (Recommended)

```bash
# Train all datasets with auto-detected optimal configs
python scripts/batch_train.py --auto-config

# Train specific datasets
python scripts/batch_train.py \
    --auto-config \
    --datasets Halfmile Brunswick
```

### Manual Mode

```bash
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --datasets Halfmile Brunswick
```

---

## 6. Monitor Training

### TensorBoard

```bash
# Start TensorBoard in a separate terminal
tensorboard --logdir runs

# Open in browser: http://localhost:6006
```

### MLflow

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Open in browser: http://localhost:5000
```

### View Logs

```bash
# Watch the latest log file
tail -f logs/$(date +%Y-%m-%d)/*_training_*.log

# Or use the symlink
tail -f logs/latest/latest.log
```

---

## 7. Evaluate Model

### Evaluate Best Model

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test \
    --detailed
```

### Evaluate Specific Checkpoint

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/mpslight_Halfmile_best.pt \
    --split test \
    --detailed
```

### Evaluate on All Splits

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split all \
    --detailed
```

---

## 8. Visualize Predictions

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/mpslight_Halfmile_best.pt \
    --n_samples 10 \
    --output visualizations
```

---

## 9. Export Model for Production

### Export to TorchScript

```bash
python scripts/export_model.py \
    --model models/registry/mpslight_Halfmile_best.pt \
    --model-type mpslight \
    --torchscript \
    --output exported_models
```

### Export to ONNX

```bash
python scripts/export_model.py \
    --model models/registry/mpslight_Halfmile_best.pt \
    --model-type mpslight \
    --onnx \
    --output exported_models
```

### Export Both Formats

```bash
python scripts/export_model.py \
    --model models/registry/mpslight_Halfmile_best.pt \
    --model-type mpslight \
    --onnx \
    --torchscript \
    --output exported_models
```

---

## 10. Command Reference

### `train.py` - Key Options

| Option | Description | Example |
|--------|-------------|---------|
| `--config` | Path to config file | `--config configs/halfmile.yaml` |
| `--model` | Model architecture | `--model mpslight` |
| `--epochs` | Number of epochs | `--epochs 30` |
| `--resume` | Resume from checkpoint | `--resume checkpoints/epoch_10.pt` |
| `--batch-size` | Override batch size | `--batch-size 8` |
| `--num-workers` | DataLoader workers | `--num-workers 4` |
| `--device` | Device (cpu/cuda/mps) | `--device mps` |
| `--loss` | Loss function | `--loss combo` |
| `--verbose` | Verbose logging | `--verbose` |
| `--log-memory` | Log memory usage | `--log-memory` |

### Available Models

```bash
python scripts/train.py --model [pico|nano|tiny|mpslight|light|mobile|efficient|unet]
```

### Available Loss Functions

```bash
python scripts/train.py --loss [cross_entropy|focal|dice|combo]
```

---

## 11. Quick Example: End-to-End Workflow

```bash
# 1. Preprocess data
python scripts/preprocess.py --config configs/halfmile.yaml --force

# 2. Train model (resume from existing checkpoint)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --resume models/registry/mpslight_Halfmile_best.pt \
    --epochs 50 \
    --verbose \
    --log-memory \
    --num-workers 4

# 3. Monitor training (in separate terminal)
tensorboard --logdir runs

# 4. Evaluate best model
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test \
    --detailed

# 5. Visualize predictions
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/mpslight_Halfmile_best.pt \
    --n_samples 10

# 6. Export model
python scripts/export_model.py \
    --model models/registry/mpslight_Halfmile_best.pt \
    --model-type mpslight \
    --torchscript \
    --onnx \
    --output exported_models
```

---

## 12. Troubleshooting

### Out of Memory (OOM) Errors

**Solution:** Reduce batch size or cache size

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --batch-size 2 \
    --cache-size 2 \
    --epochs 30
```

### Slow Training

**Solution:** Increase workers or reduce logging

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --num-workers 4 \
    --log-level INFO \
    --epochs 30
```

### MPS Issues

**Solution:** Fallback to CPU

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --device cpu \
    --epochs 30
```

### Checkpoint Not Found

**Solution:** List available checkpoints

```bash
ls -lt models/registry/
```

---

## 13. Where to Find Results

| Artifact | Location |
|----------|----------|
| **Checkpoints** | `models/registry/*.pt` |
| **Best Model** | `models/registry/{model}_{dataset}_best.pt` |
| **TensorBoard** | `runs/{dataset}/{model}/` |
| **MLflow** | `mlflow.db` (UI: `mlflow ui`) |
| **Logs** | `logs/YYYY-MM-DD/*.log` |
| **Evaluation** | `evaluation_results/` |
| **Visualizations** | `visualizations/` |
| **Exported Models** | `exported_models/` |

---

## 14. Quick Start for Your Current Checkpoint

Based on your current checkpoint:

```bash
# Resume training from epoch 3
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model pico \
    --resume models/registry/PicoUNet_Halfmile_epoch_3_20260903_172644.pt \
    --epochs 50 \
    --verbose \
    --log-memory \
    --num-workers 4
```

---

**Happy Training! 🚀**