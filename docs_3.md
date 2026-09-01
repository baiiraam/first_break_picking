# First Break Picking - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Configuration Files](#configuration-files)
4. [Core Functionality](#core-functionality)
5. [CLI Commands](#cli-commands)
6. [Workflows](#workflows)
7. [MLflow Integration](#mlflow-integration)
8. [Testing](#testing)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is This Project?

This project provides an end-to-end machine learning pipeline for **automatic seismic first-break picking** using deep learning. It processes seismic trace data from multiple real-world assets, trains U-Net-based models to identify first arrivals, and includes robust memory error recovery for batch training.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Dataset Support** | Brunswick, Halfmile, Lalor, Sudbury |
| **Multiple Model Architectures** | 8 models from PicoUNet (~2K params) to UNet (~31M params) |
| **Multiple Loss Functions** | CrossEntropy, Focal Loss, Dice Loss, Combo Loss |
| **Smart Config Detection** | Auto-calculates batch_size, cache_size, memory_limit based on system resources |
| **Memory Error Recovery** | Automatically reduces configs on memory errors |
| **Sequential Training** | Trains models one after another with graceful failure |
| **MLflow Integration** | Full experiment tracking, model registry, versioning |
| **TensorBoard Support** | Real-time training visualization |
| **Comprehensive Testing** | Pytest test suite with 32+ tests |

### Supported Datasets

| Dataset | Traces | Samples | File Size | Description |
|---------|--------|---------|-----------|-------------|
| **Brunswick** | 2,582 | 751 | ~1.5 GB | 3D seismic survey |
| **Halfmile** | 1,578 | 751 | ~1.2 GB | 3D seismic survey |
| **Lalor** | 2,685 | 1,501 | ~3.3 GB | 3D seismic survey, larger |
| **Sudbury** | 1,138 | 1,001 | ~1.0 GB | 3D seismic survey |

### Supported Models

| Model | Parameters | Memory | Speed | Use Case |
|-------|-----------|--------|-------|----------|
| **PicoUNet** | ~2K | < 10MB | Fastest | Last resort, testing |
| **NanoUNet** | ~10K | ~50MB | Very Fast | Ultra-fast testing |
| **TinyUNet** | ~50K | ~100MB | Fast | Quick training, fallback |
| **MPSLightUNet** | ~1.7M | ~150MB | Fast | **Recommended for MPS** |
| **LightUNet** | ~2.5M | ~200MB | Fast | Balanced model |
| **MobileUNet** | ~3.5M | ~250MB | Medium | Good generalization |
| **EfficientUNet** | ~5M | ~300MB | Medium | Best lightweight accuracy |
| **UNet** | ~31M | ~500MB | Slow | Best accuracy |

### Supported Loss Functions

| Loss | Best For | Description |
|------|----------|-------------|
| **CrossEntropy** | Baseline | Standard classification loss |
| **Focal Loss** | Imbalanced classes | Focuses on hard-to-classify examples |
| **Dice Loss** | Segmentation | Directly optimizes IoU |
| **Combo Loss** | **Recommended** | CE + Focal + Dice combined |

---

## Project Structure

```
first_break_pick/
│
├── configs/                          # Configuration files
│   ├── batch_config.yaml             # Batch training configuration
│   ├── sweep_config.yaml             # Grid search sweep configuration
│   ├── brunswick.yaml                # Brunswick dataset config
│   ├── halfmile.yaml                 # Halfmile dataset config
│   ├── lalor.yaml                    # Lalor dataset config
│   ├── sudbury.yaml                  # Sudbury dataset config
│   ├── default.yaml                  # Default configuration template
│   └── production.yaml               # Production mode overrides
│
├── scripts/                          # Executable command-line scripts
│   ├── batch_train.py                # Batch training orchestrator
│   ├── sweep_mlflow.py               # Grid search with MLflow tracking
│   ├── train.py                      # Single model training
│   ├── evaluate.py                   # Model evaluation
│   ├── preprocess.py                 # Data preprocessing
│   ├── visualize.py                  # Result visualization
│   ├── export_model.py               # Model export (ONNX/TorchScript)
│   ├── check_device_memory.py        # Device memory detection
│   └── search_models.py              # MLflow model search
│
├── src/                              # Core source code
│   ├── config.py                     # Configuration management
│   ├── data/                         # Data handling
│   │   ├── cache.py                  # LRU cache for chunked data
│   │   ├── chunked_dataset.py        # Memory-efficient dataset
│   │   └── hdf5_dataset.py           # HDF5 lazy loading
│   ├── models/                       # Model architectures
│   │   ├── unet.py                   # Standard U-Net (31M params)
│   │   ├── mps_light_unet.py         # MPS-optimized (1.7M params)
│   │   ├── light_unet.py             # Lightweight (2.5M params)
│   │   ├── mobilenet.py              # MobileNet + U-Net
│   │   ├── efficient_unet.py         # EfficientNet + U-Net
│   │   ├── tiny_unet.py              # Tiny (50K params)
│   │   ├── nano_unet.py              # Nano (10K params)
│   │   └── pico_unet.py              # Pico (2K params)
│   ├── preprocessing/                # Data preprocessing
│   │   ├── processor.py              # Shot processing
│   │   ├── chunker.py                # Chunk assignment
│   │   ├── manifest.py               # Manifest generation
│   │   └── writer.py                 # Chunk writing
│   ├── training/                     # Training pipeline
│   │   ├── trainer.py                # Main trainer class
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── losses.py                 # Loss functions
│   │   └── callbacks.py              # Training callbacks
│   └── utils/                        # Utility functions
│       ├── logger.py                 # Loguru logging setup
│       ├── mlflow_utils.py           # MLflow integration
│       ├── tensorboard_utils.py      # TensorBoard logging
│       ├── hdf5_utils.py             # HDF5 file operations
│       └── memory_utils.py           # Memory management
│
├── tests/                            # Pytest test suite
│   ├── test_pipeline.py              # Main pipeline tests
│   ├── test_configuration.py         # Config tests
│   ├── test_metrics.py               # Metrics tests
│   └── ...                           # Other test files
│
├── data/                             # Data storage (created at runtime)
│   ├── raw/                          # Original HDF5 files
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
│   ├── YYYY-MM-DD/                   # Date-based logs
│   ├── batch/                        # Batch training logs
│   └── latest/                       # Symlink to latest log
│
├── runs/                             # TensorBoard logs
│
├── mlflow.db                         # MLflow SQLite database
│
├── pyproject.toml                    # Project configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview
```

### Folder Responsibilities

| Folder | Responsibility |
|--------|---------------|
| **configs/** | YAML configuration files for datasets and training modes |
| **scripts/** | Executable CLI tools for the entire pipeline |
| **src/** | Core library code organized by functionality |
| **tests/** | Pytest test suite for validation |
| **data/** | Raw and preprocessed seismic data |
| **models/** | Saved model checkpoints |
| **logs/** | Training logs with date-based organization |
| **runs/** | TensorBoard event logs |
| **mlflow.db** | MLflow metadata database |

---

## Configuration Files

### 1. `configs/batch_config.yaml`

**Purpose:** Main configuration for batch training orchestration.

```yaml
# ============================================================
# BATCH TRAINING CONFIGURATION
# ============================================================

global:
  # Training parameters
  epochs: 30
  device: "mps"                     # cpu, cuda, mps
  log_memory: true
  verbose: true
  log_level: "INFO"                 # DEBUG, INFO, WARNING, ERROR, CRITICAL
  preprocess: false
  checkpoint_every: 5
  early_stopping: 5
  
  # Pipeline settings
  skip_failed: true                 # Continue on failure
  clear_memory_between_datasets: true
  pause_between_datasets: 2

# Dataset-specific overrides
datasets:
  Lalor:
    batch_size_override: 2
    model_override: "tiny"
    preprocess: true

# Memory error recovery variants
variants:
  - batch_size: 4
    model: "mpslight"
    cache_size: 3
    class_weights: "0.1,0.1,0.8"
    strip_width: 8
    memory_limit_gb: 12

# Auto-config settings
auto:
  strategy: "smart"                 # smart, aggressive, nonlinear
  models:
    - "mpslight"
    - "light"
  skip_models:
    - "unet"
```

### 2. `configs/sweep_config.yaml`

**Purpose:** Grid search configuration for running multiple experiments.

```yaml
# ============================================================
# SWEEP CONFIGURATION
# ============================================================

global:
  epochs: 2
  device: "mps"
  log_memory: true
  verbose: true

sweep:
  datasets:
    - "Brunswick"
    - "Halfmile"
    - "Lalor"
    - "Sudbury"
  
  models:
    - "pico"
    - "nano"
    - "tiny"
    - "mpslight"
  
  losses:
    - "cross_entropy"
    - "focal"
    - "dice"
    - "combo"

tracking:
  enabled: true
  experiment_name: "model_loss_sweep"
```

### 3. Dataset Config Files

Each dataset has its own configuration:

```yaml
# configs/halfmile.yaml

dataset_name: "Halfmile"
hdf5_path: "data/raw/Halfmile3D_add_geom_sorted.hdf5"
chunk_dir: "data/chunks"

target_traces: 1578               # Traces per shot
n_samples: 751                    # Time samples per trace
strip_width: 8                    # First-break strip width
chunk_size: 69                    # Shots per chunk

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

## Core Functionality

### 1. Smart Configuration Detection

The system automatically detects your device and calculates optimal training parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SMART CONFIG DETECTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input:                                                                     │
│  • Device: MPS (Apple Silicon)                                            │
│  • Available Memory: 9.6 GB                                               │
│  • Dataset: Halfmile (1578 traces, 751 samples)                          │
│                                                                             │
│  Calculations:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Model: MPSLightUNet (1.7M params)                                 │    │
│  │  Base Memory: 600 MB                                               │    │
│  │  Available after base: 9.0 GB                                      │    │
│  │  Dataset factor: 1.0 (medium dataset)                             │    │
│  │  Optimal batch: 6 (max_by_memory=36, recommended=6)               │    │
│  │  Optimal cache: 4 (max_by_memory=59, recommended=4)               │    │
│  │  Final memory limit: 3.5 GB                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output:                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  batch_size: 6, cache_size: 4, memory_limit: 3.5GB               │    │
│  │  Fallback levels: 4 (if memory error occurs)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. Memory Error Recovery

When a memory error occurs during training:

```
Attempt 1: batch=8, cache=5, memory=8.6GB
  → ❌ MPS out of memory
  → Clear memory (torch.mps.empty_cache())
  → Try next variant

Attempt 2: batch=6, cache=4, memory=6.0GB
  → ✅ SUCCESS! Training continues
```

### 3. Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                     │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 1: Preprocess                                                │    │
│  │  • Load HDF5                                                        │    │
│  │  • Group traces by shot ID                                          │    │
│  │  • Create chunks (train/val/test)                                   │    │
│  │  • Generate manifest.json                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 2: Smart Config Detection                                    │    │
│  │  • Detect device memory                                             │    │
│  │  • Calculate optimal batch_size, cache_size, memory_limit          │    │
│  │  • Generate fallback variants                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 3: Training                                                  │    │
│  │  • Try each variant (sequential)                                    │    │
│  │  • On memory error: clear memory, try next                         │    │
│  │  • On success: save model, log metrics                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Phase 4: Logging & Registry                                        │    │
│  │  • Log to MLflow                                                    │    │
│  │  • Save to TensorBoard                                              │    │
│  │  • Register model with aliases (champion/challenger/staging)       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  END                                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Commands

### 1. Batch Training

```bash
# Train all datasets with auto-config
python scripts/batch_train.py --auto-config --epochs 30

# Train specific datasets
python scripts/batch_train.py --auto-config --datasets Halfmile --datasets Brunswick

# Train only specific models
python scripts/batch_train.py --auto-config --models pico --models mpslight

# Train with verbose logging
python scripts/batch_train.py --auto-config --verbose --log-level DEBUG --log-memory

# Train with custom epochs
python scripts/batch_train.py --auto-config --epochs 2 --models pico
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | `configs/batch_config.yaml` | Config file path |
| `--datasets` | `-d` | Multiple | All | Datasets to train |
| `--models` | `-m` | Multiple | All | Models to train |
| `--epochs` | `-e` | Integer | 30 | Number of epochs |
| `--device` | `-dev` | String | `mps` | Device to use |
| `--log-memory` | `-lm` | Flag | False | Log memory usage |
| `--verbose` | `-v` | Flag | False | Verbose output |
| `--log-level` | `-ll` | String | `INFO` | Log level |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--auto-config` | `-a` | Flag | False | Auto-detect optimal config |

### 2. Single Model Training

```bash
# Train a single model on a single dataset
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --verbose

# Train with specific loss function
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model pico \
    --epochs 2 \
    --loss focal \
    --class-weights 0.1 0.1 0.8
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | String | `unet` | Model architecture |
| `--epochs` | `-e` | Integer | 30 | Number of epochs |
| `--loss` | `-l` | String | `cross_entropy` | Loss function |
| `--class-weights` | `-cw` | 3 Floats | `0.2,0.2,0.6` | Class weights |
| `--device` | `-d` | String | `mps` | Device to use |
| `--batch-size` | `-b` | Integer | Config | Override batch size |
| `--cache-size` | - | Integer | Config | Override cache size |
| `--verbose` | `-v` | Flag | False | Verbose output |
| `--log-memory` | `-lm` | Flag | False | Log memory usage |
| `--resume` | `-r` | String | None | Resume from checkpoint |

### 3. Experiment Sweep

```bash
# Run grid search sweep
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml

# Run sweep with MLflow tracking
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | `configs/sweep_config.yaml` | Sweep config file |

### 4. Evaluation

```bash
# Evaluate champion model
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test

# Evaluate specific model
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/model.pt \
    --split test \
    --detailed
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model path or "best" |
| `--split` | `-s` | String | `test` | Split to evaluate |
| `--detailed` | - | Flag | False | Generate per-shot metrics |
| `--device` | `-d` | String | `mps` | Device to use |
| `--batch-size` | `-b` | Integer | 4 | Batch size |

### 5. Visualization

```bash
# Visualize model predictions
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model best \
    --n_samples 10
```

**Options:**

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model path |
| `--n_samples` | `-n` | Integer | 10 | Number of samples |
| `--output` | `-o` | String | `visualization_results` | Output directory |

### 6. Preprocessing

```bash
# Preprocess a dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Force reprocess
python scripts/preprocess.py --config configs/halfmile.yaml --force
```

### 7. Model Export

```bash
# Export to ONNX and TorchScript
python scripts/export_model.py \
    --model models/registry/model.pt \
    --onnx \
    --torchscript
```

### 8. Check Device Memory

```bash
# Check device memory and get recommendations
python scripts/check_device_memory.py
```

---

## Workflows

### Workflow 1: Train a Single Model

```bash
# 1. Preprocess dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# 2. Train model
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --verbose

# 3. Evaluate model
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test

# 4. Visualize results
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model best \
    --n_samples 10
```

### Workflow 2: Batch Train Multiple Datasets

```bash
# Train all datasets with auto-config
python scripts/batch_train.py \
    --auto-config \
    --epochs 30 \
    --models mpslight \
    --verbose \
    --log-memory
```

### Workflow 3: Experiment Sweep

```bash
# 1. Create sweep config
cat > configs/sweep_config.yaml << EOF
sweep:
  datasets: ["Halfmile"]
  models: ["pico", "nano", "tiny"]
  losses: ["cross_entropy", "combo"]
global:
  epochs: 2
  verbose: true
EOF

# 2. Run sweep
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml

# 3. View results
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Workflow 4: Production Deployment

```bash
# 1. Find champion model
python scripts/search_models.py --dataset Halfmile

# 2. Export champion
python scripts/export_model.py \
    --model models:/halfmile@champion \
    --onnx \
    --torchscript \
    --output production_models

# 3. Test exported model
python -c "
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('production_models/model.onnx')
input_data = np.random.randn(1, 1, 1578, 751).astype(np.float32)
output = session.run(None, {'input': input_data})
print(f'Inference successful! Output shape: {output[0].shape}')
"
```

---

## MLflow Integration

### Model Registry Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MLflow MODEL REGISTRY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Run                                                              │
│       │                                                                     │
│       ▼                                                                     │
│  Model Version 1 (Candidate)                                              │
│       │                                                                     │
│       ▼                                                                     │
│  Alias: staging  ←─ Latest model                                          │
│       │                                                                     │
│       ▼                                                                     │
│  Is it better than champion?                                              │
│       │                                                                     │
│       ├── Yes ──► Alias: champion (new)                                  │
│       │         Alias: challenger (old champion)                         │
│       │                                                                     │
│       └── No ───► Alias: challenger (this model)                         │
│                                                                             │
│  Champion = Production model                                               │
│  Staging = Latest model under review                                      │
│  Challenger = Previous champion                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
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

## Testing

### Run Tests

```bash
# Run all tests
python3.12 -m pytest tests/ -v

# Run specific test file
python3.12 -m pytest tests/test_pipeline.py -v

# Run specific test
python3.12 -m pytest tests/test_pipeline.py::TestConfiguration::test_config_loading -v

# Run with coverage
python3.12 -m pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Test Class | What It Tests |
|------------|---------------|
| **TestConfiguration** | Config loading, validation, dict conversion |
| **TestDeviceMemoryDetection** | Device detection, memory limits, model profiles |
| **TestSmartConfigDetection** | Config calculation, generation, reasoning |
| **TestLRUCache** | Cache operations, eviction, statistics |
| **TestChunker** | Split assignment, chunk creation |
| **TestManifest** | Validation, statistics |
| **TestProcessor** | Shot processing, mask creation |
| **TestMetrics** | Segmentation metrics, first-break metrics |
| **TestMemoryErrorDetection** | Error detection, memory monitoring |
| **TestMLflow** | Manager init, alias formatting |
| **TestBatchTrainingSequence** | Variant progression, dataset-aware skipping |
| **TestEndToEnd** | Dataset loading, training loop simulation |
| **TestLossFunctions** | Combo loss, cross-entropy with weights |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `No such option '--loss'` | Missing option in train.py | Add `@click.option('--loss')` to train.py |
| `MPS out of memory` | Memory limit too high | Reduce batch_size or use smaller model |
| `duration' is not defined` | Variable not initialized | Initialize `duration = 0.0` before try block |
| `HDF5 validation failed` | Corrupt/missing file | Re-download dataset |
| `MLflow model not found` | Model not registered | Train with MLflow enabled |
| `CUDA not available` | No GPU or wrong CUDA | Use `--device cpu` or `--device mps` |

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

# View latest logs
tail -f logs/latest/latest.log

# View errors
grep "ERROR" logs/latest/latest.log
```

### Linting with Ruff

```bash
# Check all files
python3.12 -m ruff check .

# Auto-fix issues
python3.12 -m ruff check --fix .

# Check specific rule
python3.12 -m ruff check --select F821 .
```

---

## Quick Reference

### Most Common Commands

```bash
# 1. Train all datasets (production)
python scripts/batch_train.py --auto-config --epochs 30

# 2. Quick test (2 epochs)
python scripts/batch_train.py --auto-config --epochs 2 --models pico --verbose

# 3. Train specific dataset
python scripts/batch_train.py --auto-config --datasets Halfmile --models mpslight

# 4. Evaluate champion
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split test

# 5. Run sweep
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml

# 6. Check device memory
python scripts/check_device_memory.py

# 7. Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 8. Start TensorBoard
tensorboard --logdir runs/
```

### Environment Setup

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTORCH_MPS_MEMORY_LIMIT=8000000000

# Optional: MLflow credentials
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

---

## Summary

This project provides a complete, production-ready pipeline for seismic first-break picking with:

| Feature | Status |
|---------|--------|
| **Multi-Dataset Support** | ✅ 4 datasets |
| **Multiple Models** | ✅ 8 architectures |
| **Multiple Losses** | ✅ 4 loss functions |
| **Smart Config Detection** | ✅ Auto-calculated |
| **Memory Error Recovery** | ✅ Automatic |
| **MLflow Integration** | ✅ Full tracking |
| **TensorBoard Support** | ✅ Real-time |
| **Comprehensive Testing** | ✅ 32+ tests |
| **Production Export** | ✅ ONNX/TorchScript |
| **Documentation** | ✅ Complete |

---

**Sleep well! 🚀**