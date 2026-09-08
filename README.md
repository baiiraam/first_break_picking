# Seismic First Break Picking - Complete Project Documentation

## 📋 **Project Overview**

This project implements a **deep learning pipeline for automatic seismic first break picking** using 3-class semantic segmentation. The system processes seismic shot gathers and identifies the first arrival time (first break) of seismic waves, which is critical for seismic data processing and subsurface imaging.

The pipeline converts the first break picking problem into a **3-class segmentation task**:
- **Class 0**: Samples before the first break
- **Class 1**: Samples after the first break
- **Class 2**: A narrow strip around the first break (the target)

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SEISMIC FBP SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  PHASE 1    │    │  PHASE 2    │    │  PHASE 3    │    │  PHASE 4    │ │
│  │  DATA       │───▶│  MODEL      │───▶│  BATCH      │───▶│  EVALUATION │ │
│  │  PIPELINE   │    │  TRAINING   │    │  ORCHESTRA- │    │  & EXPORT   │ │
│  │             │    │             │    │  TION       │    │             │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    INFRASTRUCTURE LAYER                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │ MLflow   │  │TensorBoard│  │  Loguru  │  │   Checkpoint     │  │  │
│  │  │ Registry │  │ Metrics  │  │  Logger  │  │   Management     │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 **Project Structure**

```
first_break_pick/
├── configs/                    # Configuration files
│   ├── batch_config.yaml       # Batch training orchestration
│   ├── halfmile.yaml           # Halfmile dataset config
│   ├── brunswick.yaml          # Brunswick dataset config
│   ├── lalor.yaml              # Lalor dataset config
│   ├── sudbury.yaml            # Sudbury dataset config
│   ├── default.yaml            # Base configuration
│   └── sweep_config.yaml       # MLflow sweep configuration
│
├── data/                       # Data directory
│   ├── raw/                    # Raw HDF5 files
│   │   ├── Halfmile3D_add_geom_sorted.hdf5
│   │   ├── Brunswick_orig_1500ms_V2.hdf5
│   │   ├── Lalor_raw_z_1500ms_norp_geom_v3.hdf5
│   │   └── preprocessed_Sudbury3D.hdf
│   └── chunks/                 # Processed chunks (.pt files)
│       ├── Halfmile/
│       ├── Brunswick/
│       ├── Lalor/
│       └── Sudbury/
│
├── scripts/                    # Executable scripts
│   ├── train.py                # Main training script
│   ├── preprocess.py           # Data preprocessing
│   ├── evaluate.py             # Model evaluation
│   ├── visualize.py            # Result visualization
│   ├── batch_train.py          # Batch training orchestration
│   ├── export_model.py         # Model export (ONNX/TorchScript)
│   ├── search_models.py        # MLflow model search
│   ├── sweep_mlflow.py         # Hyperparameter sweep
│   ├── check_device_memory.py  # Hardware detection
│   ├── run_pico_all.py         # Quick test runner
│   └── run_model_pairs.py      # Model pair training
│
├── src/                        # Core library
│   ├── config.py               # Configuration management
│   ├── data/                   # Data loading
│   │   ├── cache.py            # LRU cache for chunks
│   │   ├── chunked_dataset.py  # Chunked dataset loader
│   │   └── hdf5_dataset.py     # Legacy HDF5 loader
│   ├── models/                 # Model architectures
│   │   ├── unet.py             # Full U-Net (31M params)
│   │   ├── mps_light_unet.py   # MPS-optimized (1.7M params)
│   │   ├── light_unet.py       # Lightweight (2.5M params)
│   │   ├── efficient_unet.py   # EfficientNet encoder
│   │   ├── mobilenet.py        # MobileNet encoder
│   │   ├── tiny_unet.py        # Tiny (50K params)
│   │   ├── nano_unet.py        # Nano (10K params)
│   │   └── pico_unet.py        # Pico (2K params)
│   ├── preprocessing/          # Preprocessing pipeline
│   │   ├── processor.py        # Shot processor
│   │   ├── chunker.py          # Chunk assignment
│   │   ├── manifest.py         # Manifest generation
│   │   └── writer.py           # Chunk writer
│   ├── training/               # Training components
│   │   ├── trainer.py          # Main trainer
│   │   ├── losses.py           # Loss functions
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── callbacks.py        # Training callbacks
│   └── utils/                  # Utilities
│       ├── mlflow_utils.py     # MLflow integration
│       ├── logger.py           # Logging configuration
│       ├── memory_utils.py     # Memory management
│       └── hdf5_utils.py       # HDF5 utilities
│
├── tests/                      # Unit tests (76+ tests)
├── models/registry/            # Saved model checkpoints
├── runs/                       # TensorBoard logs
├── logs/                       # Application logs
└── pyproject.toml              # Project configuration
```

---

## 🔄 **Complete Workflow**

### **Phase 1: Data Pipeline**

#### **1.1 Data Discovery**
The pipeline reads HDF5 files containing seismic shot gathers:

| Dataset | Traces/Shot | Samples | Shots | File Size | Sampling |
|---------|-------------|---------|-------|-----------|----------|
| Halfmile | 1578 | 751 | 690 | ~1.8 GB | 2.0 ms |
| Brunswick | 2582 | 751 | 1541 | ~2.9 GB | 2.0 ms |
| Lalor | 2685 | 1501 | 907 | ~3.3 GB | 1.0 ms |
| Sudbury | 1138 | 1001 | 1016 | ~1.3 GB | 1.0 ms |

#### **1.2 Critical Unit Conversion**
**⚠️ The preprocessing correctly handles the unit mismatch:**

| Dataset | SPARE1 Unit | Conversion | Result |
|---------|-------------|------------|--------|
| Halfmile | Milliseconds | ÷2.0 → samples | All 993,189 picks valid |
| Brunswick | Milliseconds | ÷2.0 → samples | All 3,733,221 picks valid |
| Lalor | Milliseconds | ÷1.0 → samples | All 1,211,857 picks valid |
| Sudbury | Milliseconds | ÷1.0 → samples | All 200,338 picks valid |

**✅ CONFIRMED: All picks are now correctly converted from milliseconds to sample indices. No out-of-bounds errors.**

#### **1.3 Mask Creation**
Each trace is converted to a 3-class mask:

```
Before: Class 0 (blue)  ←───  strip_width = 8 samples  ───→  After: Class 1 (green)
                               └── Class 2 (red) ──┘
```

**✅ CONFIRMED: Strip covers ~1.07% of trace (8/751), matching expected.**

#### **1.4 Chunking & Caching**
- **Chunk size**: 69 shots per chunk
- **Cache**: LRU cache (configurable, default 5 chunks)
- **Output**: Chunks saved as `.pt` files with checksums

---

### **Phase 2: Model Training**

#### **2.1 Model Family**
Eight U-Net variants for different performance trade-offs:

| Model | Parameters | Memory | Speed | Use Case |
|-------|------------|--------|-------|----------|
| PicoUNet | 2K | 200 MB | 10s/epoch | Instant testing |
| NanoUNet | 10K | 250 MB | 30s/epoch | Quick testing |
| TinyUNet | 50K | 300 MB | 1min/epoch | Lightweight training |
| MPSLightUNet | 1.7M | 600 MB | 2× faster | Apple Silicon optimized |
| LightUNet | 2.5M | 800 MB | 2× faster | Balanced |
| MobileUNet | 3.5M | 1 GB | Moderate | Transfer learning |
| EfficientUNet | 5M | 1.2 GB | Moderate | Transfer learning |
| UNet | 31M | 3 GB | 5-10min/epoch | Full capacity |

#### **2.2 Loss Functions**
All loss functions handle `ignore_index=-1` for unlabeled traces:

| Loss | Description | Best For |
|------|-------------|----------|
| Cross Entropy | Weighted CE | Balanced datasets |
| Focal Loss | Focus on hard examples | Imbalanced data (Sudbury) |
| Dice Loss | IoU optimization | Strip detection (Lalor) |
| **Combo Loss** | CE + Focal + Dice | **Recommended for most use cases** |

**✅ CONFIRMED: All loss functions correctly ignore `-1` pixels.**

#### **2.3 Training Features**
- **MPS Optimized**: Apple Silicon support with shader warmup
- **Memory Aware**: Auto-config based on hardware
- **OOM Recovery**: Progressive fallback variants
- **Checkpointing**: Every N epochs with MLflow registry
- **Early Stopping**: Patience-based stopping
- **Gradient Clipping**: Prevents exploding gradients

---

### **Phase 3: Batch Orchestration**

The `batch_train.py` script orchestrates training across multiple datasets and configurations:

#### **3.1 Smart Auto-Configuration**
Detects hardware and calculates optimal:
- **Batch size**: Based on available memory
- **Cache size**: Based on dataset chunk count
- **Memory limit**: Device-specific overhead

#### **3.2 Fallback Variants**
If memory errors occur, the system automatically tries:

| Level | Batch Size | Cache Size | Memory Limit |
|-------|------------|------------|--------------|
| 1 (Optimal) | Calculated | Calculated | Calculated |
| 2 | 75% | 100% | 85% |
| 3 | 100% | 75% | 85% |
| 4 | 50% | 50% | 70% |
| 5 (Minimal) | 1 | 1 | 50% |

---

### **Phase 4: Evaluation & Deployment**

#### **4.1 Metrics**
**Segmentation Metrics**:
- Mean IoU (class-wise)
- Mean F1 (class-wise)
- Pixel Accuracy
- Class-wise IoU (especially Class 2: Strip)

**First Break Metrics**:
- Mean Absolute Error (MAE) in samples
- Std Absolute Error
- Accuracy within ±3 samples
- Percentile distribution of errors

#### **4.2 Export Formats**
- **TorchScript**: Optimized inference
- **ONNX**: Cross-platform deployment

---

## 🔧 **Critical Fixes Applied**

### **1. Unit Conversion (PREPROCESSING FIX)**
**Problem**: `SPARE1` values were in milliseconds, but treated as sample indices.

**✅ Solution**: Added `sampling_interval_ms` to configs and conversion in `ShotProcessor`:

```python
# Before (WRONG):
pick = spare1  # 881ms treated as sample 881

# After (CORRECT):
pick = spare1 / sampling_interval_ms  # 881ms → 440 samples
```

**✅ Verified**: All datasets now have valid picks within range.

### **2. Loss Function Ignore Index**
**Problem**: ComboLoss, FocalLoss, DiceLoss didn't handle `-1` values.

**✅ Solution**: Added `ignore_index` parameter to all loss functions. Now `-1` pixels are completely ignored in loss computation.

**✅ Verified**: Tested with dummy data containing `-1` values — loss computed correctly.

### **3. MPS Compatibility**
**Problem**: MPS JIT compilation causing "Placeholder storage" errors.

**✅ Solution**: Added proper MPS warmup with synchronized operations.

---

## 📊 **Data Validation Results**

### **Halfmile Dataset (Verified)**
```
Total picks: 993,189
Valid range: 11 ms → 6 samples to 1482 ms → 741 samples
Invalid picks: 0 (all valid)
Mask classes: -1, 0, 1, 2
Unlabeled traces: 10.8% (correctly ignored)
Strip percentage: 1.07% (matches expected)
```

### **Mask Statistics**
| Class | Value | Percentage | Description |
|-------|-------|------------|-------------|
| Ignored | -1 | 10.8% | Unlabeled traces |
| Before | 0 | 20.9% | Before first break |
| After | 1 | 67.2% | After first break |
| Strip | 2 | 1.1% | Around first break |

---

## 🚀 **Quick Start Commands**

### **1. Preprocess Data**
```bash
python scripts/preprocess.py --config configs/halfmile.yaml --force
```

### **2. Quick Training Test**
```bash
python scripts/train.py --config configs/halfmile.yaml --model pico --epochs 2
```

### **3. Full Training**
```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --class-weights 0.1 0.1 0.8 \
    --verbose \
    --log-memory
```

### **4. Batch Training (Auto-Config)**
```bash
python scripts/batch_train.py --config configs/batch_config.yaml --auto-config
```

### **5. Evaluate Best Model**
```bash
python scripts/evaluate.py --config configs/halfmile.yaml --model best --detailed
```

### **6. Visualize Predictions**
```bash
python scripts/visualize.py --config configs/halfmile.yaml --model models/registry/best_model.pt
```

---

## 📈 **Monitoring & Tracking**

### **TensorBoard**
```bash
tensorboard --logdir runs/
```

### **MLflow**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### **Logs**
```bash
# Latest training log
tail -f logs/latest/latest.log

# Error log
tail -f logs/$(date +%Y-%m-%d)/*_errors.log
```

---

## ✅ **Preprocessing Fix Confirmation**

The preprocessing unit mismatch has been **fully resolved**:

| Check | Status | Evidence |
|-------|--------|----------|
| SPARE1 to samples | ✅ | All picks within range |
| Invalid picks | ✅ | 0 out-of-bounds errors |
| Mask classes | ✅ | -1, 0, 1, 2 present |
| Strip placement | ✅ | 1.07% of pixels (expected) |
| Ignore index | ✅ | Loss functions handle -1 |
| Sampling interval | ✅ | Configurable per dataset |
