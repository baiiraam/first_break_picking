# Seismic FBP - Command Line Interface Documentation

---

## 📋 **Overview**

This document covers all CLI scripts in the seismic first break picking pipeline. Each script is designed to be run independently with various configuration options.

---

## 🚀 **Main Training Script: `train.py`**

### **Purpose**
Train a seismic FBP model with U-Net or variants on a specific dataset.

### **Usage**
```bash
python scripts/train.py --config <config_file> [OPTIONS]
```

### **Required Arguments**

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to config YAML file (e.g., `configs/halfmile.yaml`) |

### **Training Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-m, --model` | Choice | `unet` | Model architecture: `unet`, `efficient`, `mobile`, `light`, `nano`, `nano-light`, `mpslight`, `tiny`, `pico` |
| `-e, --epochs` | Int | Config default | Number of training epochs |
| `-b, --batch-size` | Int | Config default | Batch size override |
| `--cache-size` | Int | Config default | Number of chunks to cache |
| `-d, --device` | String | Config default | Device: `cpu`, `cuda`, `mps` |
| `--num-workers` | Int | Config default | Number of data loading workers |
| `-r, --resume` | String | None | Path to checkpoint to resume from |
| `-l, --loss` | Choice | `cross_entropy` | Loss: `cross_entropy`, `focal`, `dice`, `combo` |
| `-cw, --class-weights` | 3 Floats | Config default | Class weights (e.g., `0.1 0.1 0.8`) |
| `--dice-weight` | Float | `0.5` | Dice weight for combo loss |
| `--focal-gamma` | Float | `2.0` | Focal gamma for focal/combo loss |
| `--lr-scheduler` | Choice | Config default | Scheduler: `step`, `plateau`, `cosine` |
| `--learning-rate` | Float | Config default | Learning rate override |
| `-ce, --checkpoint-every` | Int | `5` | Save checkpoint every N epochs |
| `-es, --early-stopping` | Int | `5` | Early stopping patience |

### **Logging Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-ds, --dataset` | String | Config default | Override dataset name for logging |
| `-p, --preprocess` | Flag | `False` | Force preprocessing even if chunks exist |
| `-v, --verbose` | Flag | `False` | Enable verbose logging (DEBUG level) |
| `--log-memory` | Flag | `False` | Enable memory logging |
| `-ll, --log-level` | Choice | Config default | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--search-best` | Flag | `False` | Search for best model after training |

### **MLflow Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--disable-autolog` | Flag | `False` | Disable MLflow autologging |
| `--disable-system-metrics` | Flag | `False` | Disable MLflow system metrics |

### **Examples**

```bash
# Basic training
python scripts/train.py --config configs/halfmile.yaml

# Quick test with PicoUNet (2 epochs)
python scripts/train.py --config configs/halfmile.yaml --model pico --epochs 2 --verbose

# Full training with MPSLightUNet and combo loss
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --class-weights 0.1 0.1 0.8 \
    --batch-size 6 \
    --verbose \
    --log-memory

# Resume from checkpoint
python scripts/train.py --config configs/halfmile.yaml --resume models/registry/checkpoint_epoch_10.pt

# Force preprocessing and train
python scripts/train.py --config configs/lalor.yaml --preprocess --model tiny
```

---

## 📊 **Preprocessing Script: `preprocess.py`**

### **Purpose**
Convert raw HDF5 seismic data into chunked PyTorch tensors with 3-class masks.

### **Usage**
```bash
python scripts/preprocess.py --config <config_file> [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | String | Required | Path to config YAML file |
| `-f, --force` | Flag | `False` | Force reprocessing even if chunks exist |
| `-d, --dataset` | String | Config default | Override dataset name for logging |

### **Examples**

```bash
# Preprocess dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Force reprocess
python scripts/preprocess.py --config configs/brunswick.yaml --force

# Preprocess with dataset override
python scripts/preprocess.py --config configs/default.yaml --dataset Halfmile
```

---

## 📈 **Evaluation Script: `evaluate.py`**

### **Purpose**
Evaluate a trained model on test/validation sets with comprehensive metrics.

### **Usage**
```bash
python scripts/evaluate.py --config <config_file> --model <model_path> [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | String | Required | Path to config YAML file |
| `-m, --model` | String | Required | Path to checkpoint (`.pt`) or `'best'` for MLflow champion |
| `-o, --output` | String | `evaluation_results` | Output directory for results |
| `-d, --device` | String | `mps` | Device: `cpu`, `cuda`, `mps` |
| `-b, --batch-size` | Int | `4` | Batch size for evaluation |
| `-ds, --dataset` | String | Config default | Override dataset name |
| `-s, --split` | Choice | `test` | Split: `train`, `val`, `test`, `all` |
| `--detailed` | Flag | `False` | Generate detailed per-shot metrics |

### **Examples**

```bash
# Evaluate best model on test set
python scripts/evaluate.py --config configs/halfmile.yaml --model best

# Evaluate specific checkpoint with detailed metrics
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/best_model.pt \
    --detailed \
    --split all

# Evaluate on validation set
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split val
```

---

## 🎨 **Visualization Script: `visualize.py`**

### **Purpose**
Generate visualizations of model predictions on test samples.

### **Usage**
```bash
python scripts/visualize.py --config <config_file> --model <model_path> [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | String | Required | Path to config YAML file |
| `-m, --model` | String | Required | Path to model checkpoint (`.pt`) |
| `-o, --output` | String | `visualization_results` | Output directory for plots |
| `-n, --n_samples` | Int | `10` | Number of samples to visualize |
| `-d, --device` | String | `mps` | Device: `cpu`, `cuda`, `mps` |

### **Examples**

```bash
# Visualize 10 samples
python scripts/visualize.py --config configs/halfmile.yaml --model models/registry/best_model.pt

# Visualize 20 samples to custom directory
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/best_model.pt \
    --n_samples 20 \
    --output my_viz_results
```

---

## 🔄 **Batch Training Script: `batch_train.py`**

### **Purpose**
Orchestrate training across multiple datasets with memory-aware auto-configuration.

### **Usage**
```bash
python scripts/batch_train.py [OPTIONS]
```

### **Configuration Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | String | `configs/batch_config.yaml` | Config file path |
| `-d, --datasets` | Multiple | All | Datasets to train (e.g., `-d Halfmile -d Brunswick`) |
| `--list-datasets` | Flag | `False` | List available datasets |

### **Mode Selection**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-a, --auto-config` | Flag | `False` | Auto-detect optimal config |
| `-m, --manual-config` | Flag | `True` | Use manual config from YAML |

### **Override Options** (Manual mode only)

| Option | Type | Description |
|--------|------|-------------|
| `-b, --batch-size` | Int | Override batch size |
| `--cache-size` | Int | Override cache size |
| `-ml, --memory-limit` | Float | Override memory limit in GB |
| `-e, --epochs` | Int | Override epochs |
| `-dev, --device` | String | Override device |
| `-lm, --log-memory` | Flag | Enable memory logging |
| `-v, --verbose` | Flag | Enable verbose logging |
| `-ll, --log-level` | String | Override log level |
| `-p, --preprocess` | Flag | Force preprocessing |

### **Examples**

```bash
# List available datasets
python scripts/batch_train.py --list-datasets

# Manual mode with default config
python scripts/batch_train.py --manual-config

# Auto-config mode (recommended)
python scripts/batch_train.py --auto-config

# Train specific datasets with auto-config
python scripts/batch_train.py --auto-config --datasets Halfmile Brunswick

# Manual mode with overrides
python scripts/batch_train.py \
    --manual-config \
    --epochs 20 \
    --device mps \
    --batch-size 8 \
    --log-memory
```

---

## 🧪 **MLflow Sweep Script: `sweep_mlflow.py`**

### **Purpose**
Run grid search over datasets, models, and loss functions with MLflow tracking.

### **Usage**
```bash
python scripts/sweep_mlflow.py [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | String | `configs/sweep_config.yaml` | Sweep config file |

### **Examples**

```bash
# Run default sweep
python scripts/sweep_mlflow.py

# Run custom sweep
python scripts/sweep_mlflow.py --config my_sweep_config.yaml
```

---

## 🔍 **Model Search Script: `search_models.py`**

### **Purpose**
Search and compare MLflow models by dataset, IoU threshold, and model type.

### **Usage**
```bash
python scripts/search_models.py [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-d, --dataset` | String | None | Filter by dataset name |
| `-m, --model-type` | String | None | Filter by model type |
| `--min-iou` | Float | None | Minimum IoU threshold |
| `-n, --top` | Int | `10` | Number of results to show |
| `-c, --compare` | Flag | `False` | Compare models side by side |

### **Examples**

```bash
# Show top 10 models for Halfmile
python scripts/search_models.py --dataset Halfmile

# Show models with IoU > 0.6
python scripts/search_models.py --min-iou 0.6

# Compare top 2 models
python scripts/search_models.py --dataset Halfmile --top 2 --compare

# Search by model type
python scripts/search_models.py --model-type MPSLightUNet
```

---

## 📦 **Model Export Script: `export_model.py`**

### **Purpose**
Export trained model to ONNX and TorchScript formats for deployment.

### **Usage**
```bash
python scripts/export_model.py --model <checkpoint_path> [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-m, --model` | String | Required | Path to model checkpoint (`.pt`) |
| `-o, --output` | String | `exported_models` | Output directory |
| `--onnx` | Flag | `False` | Export to ONNX format |
| `--torchscript` | Flag | `False` | Export to TorchScript format |
| `-d, --device` | String | `cpu` | Device for export |
| `-t, --model-type` | Choice | `unet` | Model type: `unet`, `mpslight` |
| `-c, --config` | String | None | Path to config YAML for logging |

### **Examples**

```bash
# Export to TorchScript only
python scripts/export_model.py --model models/registry/best_model.pt --torchscript

# Export to both formats
python scripts/export_model.py \
    --model models/registry/best_model.pt \
    --onnx \
    --torchscript \
    --model-type mpslight

# Export with config
python scripts/export_model.py \
    --model models/registry/best_model.pt \
    --onnx \
    --config configs/halfmile.yaml
```

---

## 🚀 **Quick Test Script: `run_pico_all.py`**

### **Purpose**
Run PicoUNet on all datasets with logging (quick validation test).

### **Usage**
```bash
python scripts/run_pico_all.py [--non-interactive]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--non-interactive` | Flag | `False` | Run without user input prompts |

### **Examples**

```bash
# Run interactively (press Enter between datasets)
python scripts/run_pico_all.py

# Run non-interactively (automated)
python scripts/run_pico_all.py --non-interactive
```

### **Logs**
- Output saved to: `logs/pico_runs/pico_{dataset}_{timestamp}.log`

---

## 🔗 **Model Pairs Script: `run_model_pairs.py`**

### **Purpose**
Train model pairs across all datasets (from smallest to largest).

### **Usage**
```bash
python scripts/run_model_pairs.py [OPTIONS]
```

### **Arguments**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-e, --epochs` | Int | `2` | Number of epochs |
| `-d, --device` | String | `mps` | Device to use |
| `--dry-run` | Flag | `False` | Print commands without running |
| `-v, --verbose` | Flag | `True` | Verbose output |
| `--no-log-memory` | Flag | `False` | Disable memory logging |

### **Model Pairs**
1. `pico` + `nano` (~2K + ~10K params)
2. `tiny` + `mpslight` (~50K + ~1.7M params)
3. `light` + `mobile` (~2.5M + ~3.5M params)
4. `efficient` + `unet` (~5M + ~31M params)

### **Examples**

```bash
# Run with default settings
python scripts/run_model_pairs.py

# Dry run (show commands without executing)
python scripts/run_model_pairs.py --dry-run

# Run with 5 epochs
python scripts/run_model_pairs.py --epochs 5

# Use CUDA
python scripts/run_model_pairs.py --device cuda
```

---

## 💻 **Device Check Script: `check_device_memory.py`**

### **Purpose**
Detect system memory and recommend optimal training configurations.

### **Usage**
```bash
python scripts/check_device_memory.py
```

### **Output**
- Device type detection (MPS, CUDA, CPU)
- Memory availability
- Optimal batch_size, cache_size, memory_limit for each model
- Auto-config saved to `auto_config.json`

### **Example**
```bash
python scripts/check_device_memory.py
```

---

## 📋 **Quick Reference Table**

| Script | Purpose | Key Options |
|--------|---------|-------------|
| `train.py` | Train a single model | `--model`, `--epochs`, `--loss`, `--class-weights` |
| `preprocess.py` | Preprocess data | `--force` |
| `evaluate.py` | Evaluate model | `--model`, `--detailed`, `--split` |
| `visualize.py` | Visualize predictions | `--model`, `--n_samples` |
| `batch_train.py` | Orchestrate batch training | `--auto-config`, `--datasets` |
| `sweep_mlflow.py` | Run hyperparameter sweep | `--config` |
| `search_models.py` | Search MLflow models | `--dataset`, `--min-iou` |
| `export_model.py` | Export to ONNX/TorchScript | `--onnx`, `--torchscript` |
| `run_pico_all.py` | Quick test on all datasets | `--non-interactive` |
| `run_model_pairs.py` | Train model pairs | `--dry-run`, `--epochs` |
| `check_device_memory.py` | Detect hardware | None |

---

## 🎯 **Common Workflows**

### **Full Pipeline**
```bash
# 1. Preprocess
python scripts/preprocess.py --config configs/halfmile.yaml --force

# 2. Train
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 30

# 3. Evaluate
python scripts/evaluate.py --config configs/halfmile.yaml --model best --detailed

# 4. Visualize
python scripts/visualize.py --config configs/halfmile.yaml --model best

# 5. Export
python scripts/export_model.py --model models/registry/best_model.pt --onnx --torchscript
```

### **Quick Test**
```bash
python scripts/train.py --config configs/halfmile.yaml --model pico --epochs 2 --verbose
```

### **Batch Training**
```bash
python scripts/batch_train.py --auto-config --datasets Halfmile Brunswick
```

### **Model Search**
```bash
python scripts/search_models.py --dataset Halfmile --top 5 --compare
```