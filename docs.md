# Complete Project Documentation

## Project Overview

**First Break Picking (FBP)** - A production-ready machine learning pipeline for automatic detection of seismic first breaks using deep learning (U-Net architectures). The system processes seismic trace data from multiple real-world seismic assets, handles varying signal-to-noise ratios, and includes robust memory error recovery for batch training.

---

## Project Structure

```
first_break_pick/
│
├── configs/                          # Configuration files
│   ├── batch_config.yaml             # Batch training configuration
│   ├── brunswick.yaml                # Brunswick dataset config
│   ├── default.yaml                  # Default configuration template
│   ├── experiment_001.yaml           # Experiment override example
│   ├── halfmile.yaml                 # Halfmile dataset config
│   ├── lalor.yaml                    # Lalor dataset config
│   ├── production.yaml               # Production mode config
│   └── sudbury.yaml                  # Sudbury dataset config
│
├── scripts/                          # Executable scripts
│   ├── batch_train.py                # Batch training orchestrator
│   ├── evaluate.py                   # Model evaluation script
│   ├── export_model.py               # Model export (ONNX/TorchScript)
│   ├── preprocess.py                 # Data preprocessing pipeline
│   ├── search_models.py              # MLflow model search
│   ├── train.py                      # Individual training script
│   └── visualize.py                  # Result visualization
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   │
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   ├── cache.py                  # LRU cache for chunked data
│   │   ├── chunked_dataset.py        # Memory-efficient dataset
│   │   └── hdf5_dataset.py           # HDF5 lazy loading
│   │
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── efficient_unet.py         # EfficientNet + U-Net
│   │   ├── light_unet.py             # Lightweight U-Net
│   │   ├── mobilenet.py              # MobileNet + U-Net
│   │   ├── mps_light_unet.py         # MPS-optimized U-Net
│   │   ├── nano_unet.py              # Ultra-lightweight U-Net
│   │   ├── pico_unet.py              # Minimal U-Net
│   │   ├── tiny_unet.py              # Tiny U-Net
│   │   └── unet.py                   # Standard U-Net
│   │
│   ├── preprocessing/                # Data preprocessing
│   │   ├── __init__.py
│   │   ├── chunker.py                # Chunk assignment logic
│   │   ├── manifest.py               # Manifest generation
│   │   ├── processor.py              # Shot processing
│   │   └── writer.py                 # Chunk writing
│   │
│   ├── training/                     # Training pipeline
│   │   ├── __init__.py
│   │   ├── callbacks.py              # Training callbacks
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── trainer.py                # Main trainer class
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── hdf5_utils.py             # HDF5 file operations
│       ├── logger.py                 # Loguru logging setup
│       ├── memory_utils.py           # Memory management
│       ├── mlflow_utils.py           # MLflow integration
│       └── tensorboard_utils.py      # TensorBoard logging
│
├── data/                             # Data directory (created at runtime)
│   ├── raw/                          # Raw HDF5 files
│   └── chunks/                       # Preprocessed chunks
│       ├── Brunswick/
│       ├── Halfmile/
│       ├── Lalor/
│       └── Sudbury/
│
├── models/                           # Model storage
│   └── registry/                     # Model checkpoints
│
├── logs/                             # Log files
│   └── batch/                        # Batch training logs
│
├── runs/                             # TensorBoard logs
│
├── mlflow.db                         # MLflow SQLite database
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Configuration Files

### `configs/batch_config.yaml`

The main batch training configuration file.

```yaml
# Global settings applied to all datasets
global:
  epochs: 30
  device: "mps"
  log_memory: true
  verbose: false
  log_level: "INFO"
  preprocess: false
  checkpoint_every: 5
  early_stopping: 5
  timeout_seconds: 7200
  skip_failed: true
  max_retries: 3
  clear_memory_between_datasets: true
  pause_between_datasets: 2

# Dataset-specific overrides
datasets:
  Brunswick:
    epochs: 40
    log_memory: true
  Halfmile:
    log_level: "DEBUG"
  Lalor:
    batch_size_override: 2
    model_override: "tiny"
    timeout_seconds: 10800
  Sudbury:
    epochs: 25
    device: "cpu"

# Memory error recovery variants (tried in order)
variants:
  - batch_size: 4
    model: "mpslight"
    cache_size: 3
    class_weights: "0.1,0.1,0.8"
    strip_width: 8
    memory_limit_gb: 8
  - batch_size: 2
    model: "mpslight"
    cache_size: 2
    class_weights: "0.1,0.1,0.8"
    strip_width: 8
    memory_limit_gb: 6
  - batch_size: 1
    model: "mpslight"
    cache_size: 1
    class_weights: "0.2,0.2,0.6"
    strip_width: 8
    memory_limit_gb: 4
  - batch_size: 1
    model: "tiny"
    cache_size: 1
    class_weights: "0.2,0.2,0.6"
    strip_width: 8
    memory_limit_gb: 4
  - batch_size: 1
    model: "nano"
    cache_size: 1
    class_weights: "0.2,0.2,0.6"
    strip_width: 8
    memory_limit_gb: 4
  - batch_size: 1
    model: "pico"
    cache_size: 1
    class_weights: "0.2,0.2,0.6"
    strip_width: 8
    memory_limit_gb: 2

# Monitoring and notifications
monitoring:
  memory_warning_threshold_gb: 16.0
  memory_critical_threshold_gb: 20.0
  system_memory_percent_warning: 80
  system_memory_percent_critical: 90
  email:
    enabled: false
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender: "your_email@gmail.com"
    recipient: "team@company.com"
    password_env_var: "EMAIL_PASSWORD"
  slack:
    enabled: false
    webhook_url_env_var: "SLACK_WEBHOOK_URL"
    channel: "#ml-training"
```

### Dataset Config Files

Each dataset has its own configuration:

```yaml
# configs/halfmile.yaml example
dataset_name: "Halfmile"
hdf5_path: "data/raw/Halfmile3D_add_geom_sorted.hdf5"
chunk_dir: "data/chunks"

target_traces: 1578      # Number of traces per shot
n_samples: 751           # Time samples per trace
strip_width: 8           # Width of first-break strip
chunk_size: 69           # Shots per chunk

train_split: 0.8
val_split: 0.1
test_split: 0.1

batch_size: 4
learning_rate: 0.001
n_epochs: 30
device: "mps"

class_weights: [0.1, 0.1, 0.8]
cache_size: 3
```

---

## Scripts Documentation

### 1. `scripts/batch_train.py`

**Purpose:** Orchestrates sequential training across multiple datasets with memory error recovery.

**Usage:**
```bash
python scripts/batch_train.py [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | `configs/batch_config.yaml` | Config file path |
| `--datasets` | `-d` | Multiple | All | Datasets to train |
| `--list-datasets` | - | Flag | False | List available datasets |
| `--epochs` | `-e` | Integer | None | Override epochs |
| `--device` | `-dev` | String | None | Override device |
| `--log-memory` | `-lm` | Flag | False | Enable memory logging |
| `--verbose` | `-v` | Flag | False | Enable verbose logging |
| `--log-level` | `-ll` | String | None | Override log level |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--timeout` | `-t` | Integer | None | Override timeout |

**Examples:**
```bash
# Train all datasets
python scripts/batch_train.py

# Train specific datasets
python scripts/batch_train.py --datasets Halfmile --datasets Brunswick

# Train with custom settings
python scripts/batch_train.py --epochs 50 --device cpu --log-memory --verbose

# Use custom config
python scripts/batch_train.py --config configs/my_config.yaml
```

---

### 2. `scripts/train.py`

**Purpose:** Train a single model on a single dataset.

**Usage:**
```bash
python scripts/train.py --config configs/halfmile.yaml [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--resume` | `-r` | String | None | Resume from checkpoint |
| `--device` | `-d` | String | None | Override device |
| `--epochs` | `-e` | Integer | None | Override epochs |
| `--model` | `-m` | String | `unet` | Model architecture |
| `--dataset` | `-ds` | String | None | Override dataset name |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--class-weights` | `-cw` | 3 Floats | None | Class weights |
| `--verbose` | `-v` | Flag | False | Enable verbose logging |
| `--log-memory` | `-lm` | Flag | False | Enable memory logging |
| `--log-level` | `-ll` | String | None | Override log level |
| `--search-best` | - | Flag | False | Search for best model |
| `--checkpoint-every` | `-ce` | Integer | 5 | Save checkpoint every N epochs |
| `--early-stopping` | `-es` | Integer | 5 | Early stopping patience |
| `--batch-size` | `-b` | Integer | None | Override batch size |
| `--cache-size` | - | Integer | None | Override cache size |
| `--learning-rate` | `-lr` | Float | None | Override learning rate |
| `--num-workers` | `-w` | Integer | None | Override workers |

**Examples:**
```bash
# Basic training
python scripts/train.py --config configs/halfmile.yaml

# Train with specific model and parameters
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 50

# Train with memory logging and verbose output
python scripts/train.py --config configs/halfmile.yaml --log-memory --verbose

# Resume training from checkpoint
python scripts/train.py --config configs/halfmile.yaml --resume checkpoints/epoch_10.pt
```

---

### 3. `scripts/evaluate.py`

**Purpose:** Evaluate trained models on test/validation sets.

**Usage:**
```bash
python scripts/evaluate.py --config configs/halfmile.yaml --model best [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model path or "best" |
| `--output` | `-o` | String | `evaluation_results` | Output directory |
| `--device` | `-d` | String | `mps` | Device to use |
| `--batch_size` | `-b` | Integer | 4 | Batch size |
| `--dataset` | `-ds` | String | None | Override dataset name |
| `--split` | `-s` | String | `test` | Split to evaluate |
| `--detailed` | - | Flag | False | Generate per-shot metrics |

**Examples:**
```bash
# Evaluate champion model
python scripts/evaluate.py --config configs/halfmile.yaml --model best

# Evaluate specific model on test set
python scripts/evaluate.py --config configs/halfmile.yaml --model models/registry/model.pt

# Generate detailed metrics
python scripts/evaluate.py --config configs/halfmile.yaml --model best --detailed

# Evaluate all splits
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split all
```

---

### 4. `scripts/preprocess.py`

**Purpose:** Preprocess raw HDF5 data into chunked format.

**Usage:**
```bash
python scripts/preprocess.py --config configs/halfmile.yaml [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--force` | `-f` | Flag | False | Force reprocessing |
| `--dataset` | `-d` | String | None | Override dataset name |

**Examples:**
```bash
# Preprocess dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Force reprocess even if chunks exist
python scripts/preprocess.py --config configs/halfmile.yaml --force
```

---

### 5. `scripts/visualize.py`

**Purpose:** Visualize model predictions on test samples.

**Usage:**
```bash
python scripts/visualize.py --config configs/halfmile.yaml --model models/registry/model.pt
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model checkpoint path |
| `--output` | `-o` | String | `visualization_results` | Output directory |
| `--n_samples` | `-n` | Integer | 10 | Number of samples |
| `--device` | `-d` | String | `mps` | Device to use |

**Examples:**
```bash
# Visualize 20 samples
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 20
```

---

### 6. `scripts/search_models.py`

**Purpose:** Search and compare models in MLflow registry.

**Usage:**
```bash
python scripts/search_models.py [OPTIONS]
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dataset` | `-d` | String | None | Filter by dataset |
| `--model-type` | `-m` | String | None | Filter by model type |
| `--min-iou` | - | Float | None | Minimum IoU threshold |
| `--top` | `-n` | Integer | 10 | Number of results |
| `--compare` | `-c` | Flag | False | Compare models side by side |

**Examples:**
```bash
# Search all models
python scripts/search_models.py

# Search Halfmile models with IoU > 0.6
python scripts/search_models.py --dataset Halfmile --min-iou 0.6

# Compare top 2 models
python scripts/search_models.py --compare --top 2
```

---

### 7. `scripts/export_model.py`

**Purpose:** Export trained models to ONNX and TorchScript formats.

**Usage:**
```bash
python scripts/export_model.py --model model.pt --onnx --torchscript
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model` | `-m` | Required | - | Model checkpoint path |
| `--output` | `-o` | String | `exported_models` | Output directory |
| `--onnx` | - | Flag | False | Export to ONNX |
| `--torchscript` | - | Flag | False | Export to TorchScript |
| `--device` | `-d` | String | `cpu` | Device for export |
| `--model-type` | `-t` | String | `unet` | Model architecture |
| `--config` | `-c` | String | None | Config file path |

**Examples:**
```bash
# Export to ONNX only
python scripts/export_model.py --model model.pt --onnx

# Export to both formats
python scripts/export_model.py --model model.pt --onnx --torchscript
```

---

## Model Architectures

| Model | Parameters | Memory | Speed | Description |
|-------|-----------|--------|-------|-------------|
| **PicoUNet** | ~2K | < 10MB | Fastest | Minimal U-Net, testing only |
| **NanoUNet** | ~10K | ~50MB | Very Fast | Ultra-lightweight testing |
| **TinyUNet** | ~50K | ~100MB | Fast | Lightweight testing |
| **MPSLightUNet** | ~1.7M | ~150MB | Fast | MPS-optimized, recommended |
| **LightUNet** | ~2.5M | ~200MB | Fast | Lightweight production |
| **MobileUNet** | ~3.5M | ~250MB | Fast | MobileNet + U-Net |
| **EfficientUNet** | ~5M | ~300MB | Medium | EfficientNet + U-Net |
| **UNet** | ~31M | ~500MB | Slow | Full U-Net, best accuracy |

---

## Datasets Information

| Dataset | Traces | Samples | File Size | Description |
|---------|--------|---------|-----------|-------------|
| **Brunswick** | 2582 | 751 | ~1.5GB | 3D seismic survey |
| **Halfmile** | 1578 | 751 | ~1.2GB | 3D seismic survey |
| **Lalor** | 2685 | 1501 | ~3.3GB | 3D seismic survey, larger |
| **Sudbury** | 1138 | 1001 | ~1.0GB | 3D seismic survey |

---

## MLflow Integration

The project uses MLflow for experiment tracking, model registry, and versioning.

### Key Features

| Feature | Description |
|---------|-------------|
| **Experiment Tracking** | All parameters, metrics, and artifacts logged |
| **Model Registry** | Versioned models with aliases |
| **Alias System** | Champion (production), Challenger, Staging |
| **Automatic Promotion** | Best model automatically becomes champion |
| **Run Comparison** | Compare models side by side |
| **System Metrics** | GPU/CPU usage monitoring |

### Model Registry Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Training Run                                              │
│       │                                                     │
│       ▼                                                     │
│  Model Version 1 (Candidate)                               │
│       │                                                     │
│       ▼                                                     │
│  Alias: staging  ←─ Latest model                           │
│       │                                                     │
│       ▼                                                     │
│  Is it better than champion?                               │
│       │                                                     │
│       ├── Yes ──► Alias: champion (new)                   │
│       │         Alias: challenger (old champion)          │
│       │                                                     │
│       └── No ───► Alias: challenger (this model)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### MLflow Commands

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# List registered models
mlflow models list

# Set model alias
mlflow models set-alias --name halfmile --alias champion --version 2

# Get model by alias
mlflow models get-alias --name halfmile --alias champion

# Load model from registry
python -c "
import mlflow
model = mlflow.pytorch.load_model('models:/halfmile@champion')
"
```

---

## Logging System

### Loguru Integration

The project uses Loguru for structured logging with:

| Feature | Description |
|---------|-------------|
| **Date-based directories** | `logs/YYYY-MM-DD/` |
| **Multiple log levels** | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| **Separate log files** | Main, Errors, Debug, JSON |
| **Log rotation** | 10MB per file, 7-day retention |
| **Compression** | .gz compression for old logs |
| **Symlink** | `logs/latest/latest.log` for quick access |

### Log Files

```
logs/
├── YYYY-MM-DD/
│   ├── HH-MM-SS_task_name.log           # Main log
│   ├── HH-MM-SS_task_name_errors.log    # Errors only
│   ├── HH-MM-SS_task_name_debug.log     # Debug only
│   └── HH-MM-SS_task_name.json          # Structured JSON logs
└── latest/
    └── latest.log                       # Symlink to latest log
```

---

## Memory Management

### LRU Cache System

The chunked dataset uses an LRU cache to manage memory:

```python
cache = LRUCache(max_size=3)  # Keep 3 chunks in memory

# Cache operations
cache.get(chunk_id)  # Retrieve chunk
cache.put(chunk_id, data)  # Store chunk
cache.clear()  # Clear all cached chunks
```

### Memory Monitoring

```python
from src.utils.memory_utils import check_memory_usage, clear_memory

# Check memory
usage = check_memory_usage()
print(f"Memory: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB")

# Clear memory between datasets
clear_memory()
```

---

## Quick Reference

### Common Commands

```bash
# 1. Preprocess a dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# 2. Train a single model
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 30

# 3. Batch train all datasets
python scripts/batch_train.py

# 4. Evaluate the champion model
python scripts/evaluate.py --config configs/halfmile.yaml --model best

# 5. Visualize predictions
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 10

# 6. Search MLflow models
python scripts/search_models.py --dataset Halfmile

# 7. Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 8. Start TensorBoard
tensorboard --logdir runs/
```

### Development Workflow

```bash
# 1. Test on a small dataset
python scripts/batch_train.py --datasets Halfmile --epochs 5 --verbose

# 2. Full training
python scripts/batch_train.py --epochs 30

# 3. Evaluate results
python scripts/evaluate.py --config configs/halfmile.yaml --model best --detailed

# 4. Export best model
python scripts/export_model.py --model models/registry/best_model.pt --onnx --torchscript

# 5. Visualize results
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 20
```

---

## Troubleshooting

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `No such option '--checkpoint-every'` | Missing option in train.py | Update train.py with missing options |
| Memory Error | Insufficient RAM | Reduce batch size or use smaller model |
| MPS Out of Memory | MPS memory limit | Set `PYTORCH_MPS_MEMORY_LIMIT` |
| HDF5 validation failed | Corrupt/missing file | Re-download dataset |
| MLflow model not found | Model not registered | Train with MLflow enabled |
| CUDA not available | No GPU or wrong CUDA version | Use `--device cpu` or `--device mps` |

### Debug Commands

```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Check MLflow runs
python scripts/search_models.py

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"

# Clear MPS cache
python -c "import torch; torch.mps.empty_cache()"
```

---

## Environment Setup

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_MPS_MEMORY_LIMIT=8000000000

# Optional: MLflow credentials
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

---

This documentation covers the complete project structure, all scripts, configuration files, and usage examples for the First Break Picking pipeline.