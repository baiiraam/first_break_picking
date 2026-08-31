# 📚 Complete Documentation: Seismic FBP Pipeline with MLflow 3

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [MLflow Features](#mlflow-features)
5. [Scripts Reference](#scripts-reference)
6. [Configuration](#configuration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This is a **production-grade seismic first-break picking pipeline** using PyTorch with full MLflow 3 integration. It supports:

- **4 seismic datasets** (Halfmile, Brunswick, Lalor, Sudbury)
- **Multiple model architectures** (UNet, MPSLightUNet)
- **Full MLflow 3 features**: Autologging, System Metrics, Model Registry, Checkpoint Tracking, Search & Comparison
- **Apple Silicon (MPS) and NVIDIA GPU (CUDA) support**
- **TensorBoard visualization**
- **Wiggle plot visualization**

### Key Features

| Feature | Description |
|---------|-------------|
| **Autologging** | Automatic logging of loss, LR, gradients, model architecture |
| **System Metrics** | GPU/CPU utilization, memory, temperature monitoring |
| **Model Registry** | Versioned models with tags and aliases |
| **Model Aliases** | `champion`, `challenger`, `staging` for deployment |
| **Checkpoint Tracking** | Link checkpoints to metrics with `step` parameter |
| **Search & Comparison** | Programmatically find best models |
| **Multi-Dataset** | 4 datasets with per-dataset models |

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd seismic_fbp
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch (Choose One)

**For Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
```

**For NVIDIA GPU (CUDA):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Set Up Data
Place HDF5 files in `data/raw/`:
```
data/raw/
├── Halfmile3D_add_geom_sorted.hdf5
├── Brunswick_orig_1500ms_V2.hdf5
├── Lalor_raw_z_1500ms_norp_geom_v3.hdf5
└── preprocessed_Sudbury3D.hdf
```

### 6. Preprocess Datasets
```bash
# Preprocess all datasets
python scripts/preprocess.py --config configs/halfmile.yaml
python scripts/preprocess.py --config configs/brunswick.yaml
python scripts/preprocess.py --config configs/lalor.yaml
python scripts/preprocess.py --config configs/sudbury.yaml
```

---

## Quick Start

### Train a Model
```bash
# Basic training (1 epoch test)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 1

# Full training (30 epochs)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --log-memory

# Training with MLflow features
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --log-memory \
    --search-best
```

### Evaluate a Model
```bash
# Load champion model (from MLflow registry)
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best

# Load specific checkpoint
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt
```

### Search for Best Models
```bash
# Top 5 models for Halfmile
python scripts/search_models.py --dataset Halfmile --top 5 --compare

# Models with IoU > 0.6
python scripts/search_models.py --min-iou 0.6
```

### Visualize Results
```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 10 \
    --style wiggle
```

### View Monitoring
```bash
# TensorBoard
tensorboard --logdir runs/Halfmile/MPSLightUNet

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## MLflow Features

### 1. Autologging

Autologging is **enabled by default** in the `MLflowManager` class.

```python
# In your trainer, this happens automatically
mlflow.pytorch.autolog(
    log_models=True,
    log_every_n_epoch=1,
    log_every_n_step=10,
    log_gradients=False,
)
```

**What gets logged automatically:**
- Loss values (per batch & per epoch)
- Learning rate
- Model architecture
- Optimizer parameters
- Gradient norms
- Weight histograms

**Disable autologging:**
```bash
python scripts/train.py --config configs/halfmile.yaml --disable-autolog
```

### 2. System Metrics

System metrics are **enabled by default** on the GPU server.

```python
# In MLflowManager.__init__
mlflow.enable_system_metrics_logging()
```

**What gets logged:**
- GPU utilization (%)
- GPU memory used (MB)
- GPU temperature (°C)
- GPU power draw (W)
- CPU utilization (%)
- Memory usage

**Disable system metrics:**
```bash
python scripts/train.py --config configs/halfmile.yaml --disable-system-metrics
```

### 3. Model Registry

Models are automatically registered with the MLflow Model Registry.

**Registered model naming:**
```
SeismicUNet_{dataset_name}
Example: SeismicUNet_Halfmile
```

**Version tags:**
| Tag | Value |
|-----|-------|
| `dataset` | Halfmile, Brunswick, Lalor, Sudbury |
| `model_type` | MPSLightUNet, UNet, etc. |
| `step` | Epoch number |
| `timestamp` | ISO timestamp |

### 4. Model Aliases

The trainer automatically manages model aliases:

| Alias | Purpose | When Assigned |
|-------|---------|---------------|
| **champion** | Best performing model | First run, then when new model beats current |
| **challenger** | Candidate for promotion | Every new model version |
| **staging** | Latest model | Every new model version |

**Load champion model:**
```bash
# In evaluation
python scripts/evaluate.py --config configs/halfmile.yaml --model best

# In code
model = mlflow.pytorch.load_model("models:/SeismicUNet_Halfmile@champion")
```

### 5. Checkpoint Tracking

Checkpoints are logged with the `step` parameter linking them to metrics.

```python
# In trainer._log_model_checkpoint()
model_info = mlflow.pytorch.log_model(
    pytorch_model=model,
    name="MPSLightUNet_Halfmile_epoch_5",
    step=5,  # ← Links to epoch
    registered_model_name="SeismicUNet_Halfmile",
)

# Metrics linked to this checkpoint
mlflow.log_metrics(metrics, step=5, model_id=model_info.model_id)
```

### 6. Search & Comparison

Use `search_models.py` to search and compare models:

```bash
# Search by dataset
python scripts/search_models.py --dataset Halfmile

# Search by model type
python scripts/search_models.py --model-type MPSLightUNet

# Search by minimum IoU
python scripts/search_models.py --min-iou 0.6

# Compare top 5 models
python scripts/search_models.py --top 5 --compare
```

**Programmatic search:**
```python
from src.utils.mlflow_utils import get_mlflow_manager

manager = get_mlflow_manager()

# Find best models
best_models = manager.search_models(
    filter_string="tags.dataset = 'Halfmile'",
    order_by=[{"field_name": "metrics.val_iou", "ascending": False}],
    max_results=5,
)

# Load the best model
best_model = mlflow.pytorch.load_model(f"models:/{best_models[0].model_id}")
```

---

## Scripts Reference

### `scripts/train.py`

**Main training entry point.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | Choice | `unet` | Model architecture |
| `--epochs` | `-e` | Integer | From config | Override epochs |
| `--device` | `-d` | String | From config | `cpu`, `cuda`, `mps` |
| `--resume` | `-r` | String | None | Resume from checkpoint |
| `--verbose` | `-v` | Flag | False | Enable verbose logging |
| `--log-memory` | `-lm` | Flag | False | Log GPU memory usage |
| `--log-level` | `-ll` | Choice | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--disable-autolog` | - | Flag | False | Disable MLflow autologging |
| `--disable-system-metrics` | - | Flag | False | Disable system metrics |
| `--search-best` | - | Flag | False | Search for best model after training |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--dataset` | `-ds` | String | From config | Override dataset name |
| `--class-weights` | `-cw` | 3 Floats | From config | Override class weights |

**Examples:**
```bash
# Quick test
python scripts/train.py --config configs/halfmile.yaml --epochs 1

# Full training with MLflow
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --log-memory \
    --search-best

# Resume from checkpoint
python scripts/train.py \
    --config configs/halfmile.yaml \
    --resume models/registry/MPSLightUNet_Halfmile_epoch_10.pt
```

### `scripts/evaluate.py`

**Evaluation entry point.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | String | **Required** | Model path or `best` |
| `--output` | `-o` | String | `evaluation_results` | Output directory |
| `--device` | `-d` | String | `mps` | `cpu`, `cuda`, `mps` |
| `--batch_size` | `-b` | Integer | `4` | Batch size |
| `--dataset` | `-ds` | String | From config | Override dataset name |

**Examples:**
```bash
# Evaluate champion model
python scripts/evaluate.py --config configs/halfmile.yaml --model best

# Evaluate specific checkpoint
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt
```

### `scripts/visualize.py`

**Visualization entry point.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | String | **Required** | Model path |
| `--output` | `-o` | String | `visualization_results` | Output directory |
| `--n_samples` | `-n` | Integer | `10` | Number of samples |
| `--device` | `-d` | String | `mps` | `cpu`, `cuda`, `mps` |
| `--style` | `-s` | Choice | `wiggle` | `wiggle`, `mask` |
| `--dataset` | `-ds` | String | From config | Override dataset name |

### `scripts/search_models.py`

**Search and compare models.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dataset` | `-d` | String | None | Filter by dataset |
| `--model-type` | `-m` | String | None | Filter by model type |
| `--min-iou` | - | Float | None | Minimum IoU threshold |
| `--top` | `-n` | Integer | `10` | Number of results |
| `--compare` | `-c` | Flag | False | Compare models side by side |

### `scripts/export_model.py`

**Export model to ONNX/TorchScript.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model` | `-m` | String | **Required** | Model path |
| `--model-type` | `-t` | Choice | `unet` | `unet`, `mpslight` |
| `--output` | `-o` | String | `exported_models` | Output directory |
| `--device` | `-d` | String | `cpu` | `cpu`, `cuda`, `mps` |
| `--onnx` | - | Flag | False | Export to ONNX |
| `--torchscript` | - | Flag | False | Export to TorchScript |

### `scripts/preprocess.py`

**Preprocess HDF5 to chunks.**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--force` | `-f` | Flag | False | Force reprocessing |
| `--dataset` | `-d` | String | From config | Override dataset name |

---

## Configuration

### `configs/halfmile.yaml` (Example)

```yaml
# ============================================================
# HALFMILE DATASET CONFIGURATION
# ============================================================

# === Dataset ===
dataset_name: "Halfmile"
hdf5_path: "data/raw/Halfmile3D_add_geom_sorted.hdf5"
chunk_dir: "data/chunks"
preprocess: false
force_reprocess: false

# === Data ===
target_traces: 1578
n_samples: 751
strip_width: 8
chunk_size: 69
random_seed: 42
train_split: 0.8
val_split: 0.1
test_split: 0.1

# === Training ===
batch_size: 4
learning_rate: 0.001
n_epochs: 30
device: "mps"
num_workers: 2
multi_gpu: false
gpu_ids: null

# === Loss ===
class_weights: [0.1, 0.1, 0.8]

# === Model Registry ===
model_registry_dir: "models/registry"
checkpoint_dir: "models/registry"
checkpoint_every: 5

# === Cache ===
cache_size: 5

# === Scheduler ===
lr_scheduler: "plateau"
lr_patience: 3
lr_factor: 0.5
lr_step_size: 10
lr_gamma: 0.5
lr_T_max: 30

# === Regularization ===
gradient_clip_value: 1.0

# === Early Stopping ===
early_stopping_patience: 5
early_stopping_min_delta: 0.0001

# === Logging ===
tensorboard_log_dir: "runs"
mlflow_experiment_name: "seismic-fbp"
log_dir: "logs"
log_level: "INFO"

# === Debugging ===
verbose_training: false
log_batch_every: null
log_memory: false
log_predictions_every: 5
```

### Dataset-Specific Parameters

| Dataset | target_traces | n_samples | Chunks | Notes |
|---------|---------------|-----------|--------|-------|
| Halfmile | 1578 | 751 | 10 | ✅ Tested |
| Brunswick | 2582 | 751 | 24 | ✅ Tested |
| Lalor | 2685 | 1501 | 12 | Larger chunks |
| Sudbury | 1138 | 1001 | 13 | Sparse labels |

---

## Monitoring

### TensorBoard

```bash
# View specific dataset/model
tensorboard --logdir runs/Halfmile/MPSLightUNet

# View all
tensorboard --logdir runs/
```

**What you see:**
- Loss curves (train/val)
- IoU, F1, Accuracy curves
- Learning rate
- Memory usage
- Class-wise IoU
- Weight/gradient histograms

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**What you see:**
- **Experiments**: All runs
- **Model Registry**: Registered models with versions
- **Model Aliases**: champion, challenger, staging
- **System Metrics**: GPU/CPU utilization (if on CUDA)
- **Artifacts**: Models, checkpoints, predictions

### Logs

```bash
# Latest logs
tail -f logs/latest/latest.log

# Dataset-specific logs
tail -f logs/2026-08-31/10-30-45_training_Halfmile_mpslight.log

# Debug logs
tail -f logs/2026-08-31/10-30-45_training_Halfmile_mpslight_debug.log
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `MLflow autologging failed` | PyTorch version | Update PyTorch: `pip install --upgrade torch` |
| `System metrics logging failed` | Not on CUDA | Ignore on MPS/CPU |
| `Model registry not found` | No models registered | Train at least 1 epoch |
| `--model best not found` | No champion alias | Train at least 1 epoch |
| `MPS out of memory` | Batch size too large | Reduce `batch_size` in config |

### Common Commands

```bash
# Check MLflow database
sqlite3 mlflow.db "SELECT COUNT(*) FROM experiments;"
sqlite3 mlflow.db "SELECT COUNT(*) FROM runs;"

# Check registered models
sqlite3 mlflow.db "SELECT * FROM registered_models;"

# Check model aliases
sqlite3 mlflow.db "SELECT * FROM model_alias;"

# Delete MLflow data (start fresh)
rm -f mlflow.db
rm -rf mlruns/
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MLFLOW_ALLOW_FILE_STORE` | Allow file store | `true` |
| `MLFLOW_TRACKING_URI` | Tracking server URI | `sqlite:///mlflow.db` |
| `MLFLOW_S3_ENDPOINT_URL` | S3 endpoint | None |
| `AWS_ACCESS_KEY_ID` | S3 access key | None |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key | None |

---

## Next Steps

1. **Test MLflow Features:**
   ```bash
   python scripts/train.py --config configs/halfmile.yaml --epochs 1 --search-best
   ```

2. **Check MLflow UI:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

3. **Evaluate Champion Model:**
   ```bash
   python scripts/evaluate.py --config configs/halfmile.yaml --model best
   ```

4. **Train All Datasets:**
   ```bash
   for dataset in halfmile brunswick lalor sudbury; do
       python scripts/train.py --config configs/$dataset.yaml --epochs 30
   done
   ```

---

## Support

For issues, check:
1. **Logs:** `logs/latest/latest.log`
2. **TensorBoard:** `runs/{dataset}/{model}/`
3. **MLflow:** `mlflow ui`
4. **GitHub Issues:** [Repository URL]

---

**Happy training!** 🚀