# 🚀 Seismic FBP Pipeline: CLI Documentation for Team Members

---

## 📋 Project Overview

This is a **production-grade seismic first‑break picking pipeline** that automatically detects the first arrival of seismic waves using a U‑Net‑based segmentation model. The pipeline supports **multiple datasets** (Halfmile, Brunswick, Lalor, Sudbury) and runs on **Apple Silicon (MPS)**, NVIDIA GPUs (CUDA), or CPU.

---

## 📁 Repository Structure

```
seismic_fbp/
│
├── data/
│   ├── raw/                          # HDF5 files (source of truth)
│   └── chunks/                       # Preprocessed chunks (auto-generated)
│       ├── Halfmile/
│       ├── Brunswick/
│       ├── Lalor/
│       └── Sudbury/
│
├── models/
│   └── registry/                     # Trained models
│
├── src/                              # Source code
│   ├── config.py                     # Configuration
│   ├── data/                         # Dataset classes
│   ├── models/                       # U-Net architectures
│   ├── preprocessing/                # Preprocessing pipeline
│   ├── training/                     # Training loop
│   └── utils/                        # Logging, MLflow, TensorBoard
│
├── scripts/
│   ├── preprocess.py                 # Preprocess HDF5 → chunks
│   ├── train.py                      # Train model
│   ├── evaluate.py                   # Evaluate model
│   ├── visualize.py                  # Visualize predictions
│   └── export_model.py               # Export to ONNX/TorchScript
│
├── configs/                          # YAML configuration files
│   ├── halfmile.yaml
│   ├── brunswick.yaml
│   ├── lalor.yaml
│   └── sudbury.yaml
│
├── runs/                             # TensorBoard logs
├── mlruns/                           # MLflow logs
├── logs/                             # Application logs
├── checkpoints/                      # Legacy (deprecated)
├── evaluation_results/               # JSON evaluation results
├── visualization_results/            # PNG visualizations
│
├── requirements.txt
├── README.md
└── .env                              # Environment variables
```

---

## ⚙️ Installation

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ |
| **PyTorch** | 2.0+ |
| **MLflow** | 2.3+ |
| **HDF5** | 3.8+ |

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd seismic_fbp

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. For Mac MPS (Apple Silicon):
pip install torch torchvision torchaudio

# 5. For NVIDIA CUDA:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Data Setup

### 1. Download Dataset Files

Place HDF5 files in `data/raw/`:

```bash
data/raw/
├── Halfmile3D_add_geom_sorted.hdf5
├── Brunswick_orig_1500ms_V2.hdf5
├── Lalor_raw_z_1500ms_norp_geom_v3.hdf5
└── preprocessed_Sudbury3D.hdf
```

### 2. Preprocess Datasets

```bash
# Preprocess all datasets
python scripts/preprocess.py --config configs/halfmile.yaml
python scripts/preprocess.py --config configs/brunswick.yaml
python scripts/preprocess.py --config configs/lalor.yaml
python scripts/preprocess.py --config configs/sudbury.yaml

# Preprocess with force (overwrite existing chunks)
python scripts/preprocess.py --config configs/halfmile.yaml --force
```

**What it does:** Converts HDF5 files to chunked `.pt` files for memory-efficient training.

**Expected output:**

```
data/chunks/Halfmile/
├── manifest.json
├── chunk_001_train.pt
├── chunk_001_val.pt
├── chunk_001_test.pt
└── ...
```

---

## 🚀 Training

### Basic Training

```bash
python scripts/train.py --config configs/halfmile.yaml
```

### Training with Options

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --device mps \
    --log-memory \
    --verbose
```

### Training All Datasets

```bash
for dataset in halfmile brunswick lalor sudbury; do
    python scripts/train.py --config configs/$dataset.yaml --model mpslight --epochs 30 --log-memory
done
```

### Resume from Checkpoint

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --resume models/registry/MPSLightUNet_Halfmile_epoch_10_20260830_202148.pt
```

---

## 📊 CLI Reference: `scripts/train.py`

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | Choice | `unet` | Model: `unet`, `mpslight` |
| `--epochs` | `-e` | Integer | From config | Override epochs |
| `--device` | `-d` | String | From config | `cpu`, `cuda`, `mps` |
| `--resume` | `-r` | String | None | Resume from checkpoint |
| `--verbose` | `-v` | Flag | False | Enable verbose logging |
| `--log-memory` | `-lm` | Flag | False | Log GPU memory usage |
| `--log-level` | `-ll` | Choice | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--dataset` | `-ds` | String | From config | Override dataset name |
| `--class-weights` | `-cw` | 3 Floats | From config | Override class weights |

### Example: Training with Custom Weights

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --class-weights 0.1 0.1 0.8 \
    --log-memory
```

---

## 📊 Evaluation

### Basic Evaluation

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt
```

### Evaluation with Options

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --device cpu \
    --batch_size 2 \
    --output evaluation_results
```

---

## CLI Reference: `scripts/evaluate.py`

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | String | **Required** | Path to model checkpoint |
| `--output` | `-o` | String | `evaluation_results` | Output directory |
| `--device` | `-d` | String | `mps` | `cpu`, `cuda`, `mps` |
| `--batch_size` | `-b` | Integer | `4` | Batch size |
| `--dataset` | `-ds` | String | From config | Override dataset name |

---

## 📊 Visualization

### Basic Visualization

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt
```

### Visualization with Options

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 20 \
    --style wiggle \
    --output my_results
```

---

## CLI Reference: `scripts/visualize.py`

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--config` | `-c` | String | **Required** | Path to config YAML |
| `--model` | `-m` | String | **Required** | Path to model checkpoint |
| `--output` | `-o` | String | `visualization_results` | Output directory |
| `--n_samples` | `-n` | Integer | `10` | Number of samples |
| `--device` | `-d` | String | `mps` | `cpu`, `cuda`, `mps` |
| `--dataset` | `-ds` | String | From config | Override dataset name |
| `--style` | `-s` | Choice | `wiggle` | `wiggle`, `mask` |
| `--sample_start` | `-ss` | Integer | `0` | First sample to show |
| `--sample_end` | `-se` | Integer | None | Last sample to show |

---

## 📊 Model Export

```bash
# Export to TorchScript
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --torchscript

# Export to ONNX
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --onnx

# Export both formats
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --onnx --torchscript
```

---

## CLI Reference: `scripts/export_model.py`

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--model` | `-m` | String | **Required** | Path to model checkpoint |
| `--model-type` | `-t` | Choice | `unet` | `unet`, `mpslight` |
| `--output` | `-o` | String | `exported_models` | Output directory |
| `--device` | `-d` | String | `cpu` | `cpu`, `cuda`, `mps` |
| `--onnx` | — | Flag | False | Export to ONNX |
| `--torchscript` | — | Flag | False | Export to TorchScript |

---

## 🔧 Monitoring & Logging

### TensorBoard

```bash
# View training progress
tensorboard --logdir runs/Halfmile/MPSLightUNet

# View all datasets
tensorboard --logdir runs/
```

### MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Logs

```bash
# View latest logs
tail -f logs/latest/latest.log

# View dataset-specific logs
tail -f logs/2026-08-30/20-15-05_training_Halfmile_mpslight.log
```

---

## 📋 Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `MLFLOW_ALLOW_FILE_STORE` | Allow MLflow file store | `true` |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | MPS memory limit | `0.0` |
| `LOGURU_LEVEL` | Log level override | From config |

---

## 🔧 Quick Commands Reference

```bash
# ============================================================
# PREPROCESSING
# ============================================================
python scripts/preprocess.py --config configs/halfmile.yaml

# ============================================================
# TRAINING
# ============================================================
# Quick test (1 epoch)
python scripts/train.py --config configs/halfmile.yaml --epochs 1

# Full training (30 epochs)
python scripts/train.py --config configs/halfmile.yaml --epochs 30 --log-memory

# Resume from checkpoint
python scripts/train.py --config configs/halfmile.yaml --resume models/registry/MPSLightUNet_Halfmile_epoch_5.pt

# ============================================================
# EVALUATION
# ============================================================
python scripts/evaluate.py --config configs/halfmile.yaml --model models/registry/MPSLightUNet_Halfmile_best.pt

# ============================================================
# VISUALIZATION
# ============================================================
python scripts/visualize.py --config configs/halfmile.yaml --model models/registry/MPSLightUNet_Halfmile_best.pt --n_samples 10

# ============================================================
# EXPORT
# ============================================================
python scripts/export_model.py --model models/registry/MPSLightUNet_Halfmile_best.pt --model-type mpslight --onnx

# ============================================================
# MONITORING
# ============================================================
tensorboard --logdir runs/Halfmile/MPSLightUNet
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 📊 Output Files

| File | Purpose |
|---|---|
| `data/chunks/{dataset}/*.pt` | Preprocessed training data |
| `models/registry/*.pt` | Trained models |
| `runs/{dataset}/{model}/` | TensorBoard logs |
| `mlruns/` | MLflow logs |
| `logs/YYYY-MM-DD/*.log` | Application logs |
| `evaluation_results/*.json` | Evaluation metrics |
| `visualization_results/*.png` | Prediction visualizations |
| `exported_models/*.onnx` | ONNX models |
| `exported_models/*_scripted.pt` | TorchScript models |

---

## 🐛 Troubleshooting

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Missing dependencies | `pip install -r requirements.txt` |
| `FileNotFoundError` | HDF5 not in `data/raw/` | Download dataset files |
| `MPS backend out of memory` | Too much memory usage | Reduce `batch_size` or `cache_size` |
| `num_workers hang` | macOS multiprocessing issue | Set `num_workers: 0` |
| `MLflow file store deprecated` | MLflow version > 2.0 | Set `MLFLOW_ALLOW_FILE_STORE=true` |

---

## 📚 Support

For issues, check:
1. **Logs:** `logs/latest/latest.log`
2. **TensorBoard:** `runs/{dataset}/{model}/`
3. **MLflow:** `mlflow ui`
4. **Configuration:** `configs/{dataset}.yaml`

---

**Happy training!** 🚀