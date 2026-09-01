# Complete Workflow Options for Running the Project

You have multiple options to run the project, from single dataset training to full batch processing. Here's a comprehensive guide to all available workflows.

---

## Option 1: Train a Single Dataset (Halfmile Only)

### Workflow Steps

```bash
# STEP 1: Preprocess Halfmile
python scripts/preprocess.py --config configs/halfmile.yaml

# STEP 2: Train Halfmile
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30

# STEP 3: Evaluate Halfmile
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test

# STEP 4: Visualize Results
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 10
```

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              WORKFLOW: SINGLE DATASET (Halfmile)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: PREPROCESS                                                 │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Input:  data/raw/Halfmile3D_add_geom_sorted.hdf5           │   │    │
│  │  │  Output: data/chunks/Halfmile/                              │   │    │
│  │  │          ├── manifest.json                                  │   │    │
│  │  │          ├── chunk_001_train.pt                             │   │    │
│  │  │          ├── chunk_002_train.pt                             │   │    │
│  │  │          ├── chunk_001_val.pt                               │   │    │
│  │  │          └── chunk_001_test.pt                              │   │    │
│  │  │  Time: ~3 minutes                                           │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: TRAIN                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Input:  data/chunks/Halfmile/                              │   │    │
│  │  │  Model:  MPSLightUNet                                       │   │    │
│  │  │  Output: models/registry/MPSLightUNet_Halfmile_best.pt     │   │    │
│  │  │  MLflow: Run logged with all metrics                       │   │    │
│  │  │  Time: ~25-30 minutes                                      │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: EVALUATE                                                   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Input:  models/registry/MPSLightUNet_Halfmile_best.pt      │   │    │
│  │  │  Output: eval_results/                                      │   │    │
│  │  │          ├── evaluation_results_Halfmile_*.json            │   │    │
│  │  │          ├── evaluation_summary_Halfmile_*.csv             │   │    │
│  │  │          └── detailed_errors_Halfmile_*.csv               │   │    │
│  │  │  Time: ~2 minutes                                          │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: VISUALIZE                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Output: visualization_results/                              │   │    │
│  │  │          ├── shot_123_comparison.png                         │   │    │
│  │  │          ├── shot_124_comparison.png                         │   │    │
│  │  │          └── ...                                             │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ✅ COMPLETE! (Total: ~30-35 minutes)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Commands for Single Dataset Options

| Dataset | Preprocess | Train | Evaluate |
|---------|------------|-------|----------|
| **Halfmile** | `--config configs/halfmile.yaml` | `--config configs/halfmile.yaml` | `--config configs/halfmile.yaml` |
| **Brunswick** | `--config configs/brunswick.yaml` | `--config configs/brunswick.yaml` | `--config configs/brunswick.yaml` |
| **Lalor** | `--config configs/lalor.yaml` | `--config configs/lalor.yaml` | `--config configs/lalor.yaml` |
| **Sudbury** | `--config configs/sudbury.yaml` | `--config configs/sudbury.yaml` | `--config configs/sudbury.yaml` |

---

## Option 2: Train Two Datasets (Halfmile + Brunswick)

### Using Batch Training

```bash
# Single command for both datasets
python scripts/batch_train.py \
    --datasets Halfmile \
    --datasets Brunswick \
    --epochs 30 \
    --log-memory

# The batch script handles:
# - Preprocessing (if chunks don't exist)
# - Sequential training
# - Memory error recovery
# - Automatic fallback configurations
```

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         WORKFLOW: TWO DATASETS (Halfmile + Brunswick)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: Preprocess Halfmile (if not already done)                 │    │
│  │  → data/chunks/Halfmile/                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: Preprocess Brunswick (if not already done)                │    │
│  │  → data/chunks/Brunswick/                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: Train Halfmile                                            │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                    │   │    │
│  │  │  ✅ SUCCESS → Save model                                    │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: Train Brunswick                                           │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                    │   │    │
│  │  │  ✅ SUCCESS → Save model                                    │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 5: Summary Report                                             │    │
│  │  ✅ Successful: 2/2 datasets                                        │    │
│  │  ⏱ Total time: ~55-60 minutes                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Manual Sequential Training

```bash
# Train Halfmile first
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 30

# Then train Brunswick
python scripts/train.py --config configs/brunswick.yaml --model mpslight --epochs 30

# Evaluate both
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split test
python scripts/evaluate.py --config configs/brunswick.yaml --model best --split test
```

---

## Option 3: Train All Four Datasets

### Using Batch Training (Recommended)

```bash
# Train all datasets with default config
python scripts/batch_train.py

# Train all datasets with custom settings
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --epochs 40 \
    --log-memory \
    --verbose

# Train all datasets with specific overrides
python scripts/batch_train.py \
    --epochs 50 \
    --device mps \
    --log-memory \
    --verbose \
    --log-level DEBUG
```

### Visual Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              WORKFLOW: ALL FOUR DATASETS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1: PREPROCESS ALL DATASETS (Sequential)                     │    │
│  │                                                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │    │
│  │  │  Brunswick   │  │  Halfmile    │  │   Lalor      │  │ Sudbury  │ │    │
│  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │ ┌──────┐ │ │    │
│  │  │  │Chunk 1 │  │  │  │Chunk 1 │  │  │  │Chunk 1 │  │  │ │Chunk1│ │ │    │
│  │  │  │Chunk 2 │  │  │  │Chunk 2 │  │  │  │Chunk 2 │  │  │ │Chunk2│ │ │    │
│  │  │  │Chunk 3 │  │  │  │Chunk 3 │  │  │  │Chunk 3 │  │  │ │Chunk3│ │ │    │
│  │  │  │Chunk 4 │  │  │  │Chunk 4 │  │  │  │Chunk 4 │  │  │ │Chunk4│ │ │    │
│  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │ └──────┘ │ │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘ │    │
│  │  Time: ~14 minutes                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2: TRAIN ALL DATASETS (Sequential with Memory Recovery)      │    │
│  │                                                                     │    │
│  │  Dataset 1: Brunswick                                              │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                 │    │    │
│  │  │  ✅ SUCCESS (25 minutes)                                  │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │  Dataset 2: Halfmile                                              │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                 │    │    │
│  │  │  ✅ SUCCESS (20 minutes)                                  │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │  Dataset 3: Lalor                                                 │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                 │    │    │
│  │  │  ❌ Out of Memory                                         │    │    │
│  │  │  Attempt 2: batch_size=2, model=mpslight                 │    │    │
│  │  │  ❌ Out of Memory                                         │    │    │
│  │  │  Attempt 3: batch_size=1, model=mpslight                 │    │    │
│  │  │  ❌ Out of Memory                                         │    │    │
│  │  │  Attempt 4: batch_size=1, model=tiny                    │    │    │
│  │  │  ✅ SUCCESS (15 minutes)                                 │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │  Dataset 4: Sudbury                                              │    │
│  │  ┌────────────────────────────────────────────────────────────┐    │    │
│  │  │  Attempt 1: batch_size=4, model=mpslight                 │    │    │
│  │  │  ✅ SUCCESS (18 minutes)                                  │    │    │
│  │  └────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3: SUMMARY & OUTPUT                                          │    │
│  │                                                                     │    │
│  │  ✅ Successful: 4/4 datasets                                       │    │
│  │  ⏱ Total time: ~80-95 minutes                                     │    │
│  │                                                                     │    │
│  │  Models Saved:                                                     │    │
│  │  • models/registry/MPSLightUNet_Brunswick_best.pt                 │    │
│  │  • models/registry/MPSLightUNet_Halfmile_best.pt                  │    │
│  │  • models/registry/TinyUNet_Lalor_best.pt                         │    │
│  │  • models/registry/MPSLightUNet_Sudbury_best.pt                   │    │
│  │                                                                     │    │
│  │  MLflow: 4 runs logged                                            │    │
│  │  TensorBoard: 4 runs                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Option 4: Preprocess + Train All Datasets with Custom Configuration

### Using `batch_config.yaml` with Overrides

```yaml
# configs/batch_config.yaml
datasets:
  Brunswick:
    epochs: 40
    log_memory: true
  
  Halfmile:
    log_level: "DEBUG"
    batch_size_override: 2  # Force smaller batch
  
  Lalor:
    batch_size_override: 2
    model_override: "tiny"
    timeout_seconds: 10800  # 3 hours for large dataset
  
  Sudbury:
    epochs: 25
    device: "cpu"  # Train on CPU to save GPU memory
```

```bash
# Run with custom config
python scripts/batch_train.py --config configs/batch_config.yaml --verbose
```

### Output Example

```
================================================================================
🚀 BATCH TRAINING PIPELINE
================================================================================
Config file: configs/batch_config.yaml
Datasets: ['Brunswick', 'Halfmile', 'Lalor', 'Sudbury']
Epochs: 30
Device: mps
Log memory: True
Verbose: True
Log level: INFO
Checkpoint every: 1
Early stopping: 5
Config variants: 6
Skip failed: True
Timeout: 7200s
================================================================================

================================================================================
📊 DATASET 1/4: Brunswick
================================================================================
💾 Memory before: 6.4GB / 17.2GB (43.7%)

  🔄 Attempt 1/6: {'batch_size': 4, 'model': 'mpslight', 'cache_size': 3}
  ✅ SUCCESS! Dataset Brunswick trained successfully
  ⏱ Duration: 245.3s
🧹 Clearing memory...

================================================================================
📊 DATASET 2/4: Halfmile
================================================================================
💾 Memory before: 6.4GB / 17.2GB (43.6%)

  🔄 Attempt 1/6: {'batch_size': 2, 'model': 'mpslight', 'cache_size': 2}  # ← Override applied
  ✅ SUCCESS! Dataset Halfmile trained successfully
  ⏱ Duration: 312.7s
🧹 Clearing memory...

================================================================================
📊 DATASET 3/4: Lalor
================================================================================
💾 Memory before: 6.5GB / 17.2GB (44.1%)

  🔄 Attempt 1/6: {'batch_size': 2, 'model': 'mpslight', 'cache_size': 2}  # ← Override applied
  ❌ Failed: RuntimeError: MPS out of memory
  🔄 Memory error detected, trying next variant

  🔄 Attempt 2/6: {'batch_size': 1, 'model': 'mpslight', 'cache_size': 1}
  ❌ Failed: RuntimeError: MPS out of memory
  🔄 Memory error detected, trying next variant

  🔄 Attempt 3/6: {'batch_size': 1, 'model': 'tiny', 'cache_size': 1}  # ← Override applied
  ✅ SUCCESS! Dataset Lalor trained successfully
  ⏱ Duration: 180.5s
🧹 Clearing memory...

================================================================================
📊 DATASET 4/4: Sudbury
================================================================================
💾 Memory before: 6.5GB / 17.2GB (43.9%)

  🔄 Attempt 1/6: {'batch_size': 4, 'model': 'mpslight', 'cache_size': 3}
  ✅ SUCCESS! Dataset Sudbury trained successfully
  ⏱ Duration: 198.2s
🧹 Clearing memory...

================================================================================
📊 BATCH TRAINING SUMMARY
================================================================================

✅ Successful: 4/4
  • Brunswick: mpslight (batch_size=4)
  • Halfmile: mpslight (batch_size=2)
  • Lalor: tiny (batch_size=1)
  • Sudbury: mpslight (batch_size=4)

⏱ Total time: 15.2 minutes

📁 Summary saved to: logs/batch/batch_summary_20260901_123456.json
================================================================================
```

---

## Option 5: Quick Test with Small Model (Development)

### For Quick Validation

```bash
# Test preprocessing on one dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Quick train with tiny model (only 5 epochs)
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model tiny \
    --epochs 5 \
    --verbose

# Quick evaluation
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/TinyUNet_Halfmile_best.pt \
    --split test
```

### For Debugging

```bash
# Debug mode with verbose logging
python scripts/batch_train.py \
    --datasets Halfmile \
    --epochs 5 \
    --model tiny \
    --verbose \
    --log-level DEBUG \
    --log-memory

# This will show:
# - Every batch processed
# - Memory usage at each step
# - Detailed error messages
# - Cache hit/miss rates
```

---

## Option 6: Run Only Preprocessing (No Training)

### Preprocess Specific Datasets

```bash
# Preprocess a single dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Preprocess multiple datasets
for dataset in brunswick halfmile lalor sudbury; do
    python scripts/preprocess.py --config configs/${dataset}.yaml
done

# Force reprocess (overwrites existing chunks)
python scripts/preprocess.py --config configs/halfmile.yaml --force
```

### Preprocess Output Structure

```
data/chunks/Halfmile/
├── manifest.json                    # Metadata
├── chunk_001_train.pt              # Training data (69 shots)
├── chunk_002_train.pt              # Training data (44 shots)
├── chunk_001_val.pt                # Validation data (14 shots)
└── chunk_001_test.pt               # Test data (15 shots)
```

---

## Option 7: Run Only Evaluation (Skip Training)

### Evaluate Existing Models

```bash
# Evaluate best model from MLflow
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test \
    --detailed

# Evaluate specific checkpoint
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test

# Evaluate all splits
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split all

# Compare multiple models
python scripts/search_models.py --dataset Halfmile --compare --top 5
```

---

## Option 8: Export Models for Production

### Export Trained Models

```bash
# Export best model to ONNX + TorchScript
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --onnx \
    --torchscript \
    --output production_models

# Export specific model type
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --onnx

# Export from MLflow champion
python scripts/export_model.py \
    --model models:/halfmile@champion \
    --onnx \
    --torchscript \
    --output production_models
```

---

## Summary: All Options Comparison

| Option | Command | Time | Use Case |
|--------|---------|------|----------|
| **Single Dataset** | `python scripts/train.py --config configs/halfmile.yaml` | ~30 min | Testing, development |
| **Two Datasets** | `python scripts/batch_train.py --datasets Halfmile --datasets Brunswick` | ~60 min | Parallel development |
| **All Datasets** | `python scripts/batch_train.py` | ~90 min | Production training |
| **Custom Config** | `python scripts/batch_train.py --config configs/batch_config.yaml` | ~90 min | Customized training |
| **Quick Test** | `python scripts/train.py --config configs/halfmile.yaml --model tiny --epochs 5` | ~5 min | Debugging, testing |
| **Preprocess Only** | `python scripts/preprocess.py --config configs/halfmile.yaml` | ~3 min | Data preparation |
| **Evaluate Only** | `python scripts/evaluate.py --config configs/halfmile.yaml --model best` | ~2 min | Model validation |
| **Export Only** | `python scripts/export_model.py --model model.pt --onnx` | ~1 min | Production deployment |

---

## Quick Reference: Most Common Commands

```bash
# 1. Train Halfmile only (most common for testing)
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 30

# 2. Train all datasets (production)
python scripts/batch_train.py --epochs 30 --log-memory

# 3. Train specific datasets
python scripts/batch_train.py --datasets Halfmile --datasets Brunswick

# 4. Evaluate the best model
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split test

# 5. Quick test with tiny model
python scripts/train.py --config configs/halfmile.yaml --model tiny --epochs 5 --verbose

# 6. Preprocess and train with verbose debug
python scripts/batch_train.py --verbose --log-level DEBUG --log-memory

# 7. Visualize predictions
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 10
```

---

## Decision Tree: Which Option to Choose?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECISION TREE FOR OPTIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                     │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  What is your goal?                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│    │                                                                        │
│    ├── "Test if the pipeline works"                                        │
│    │   └──➤ Use Quick Test: python scripts/train.py --model tiny --epochs 5│
│    │                                                                        │
│    ├── "Train a single dataset"                                            │
│    │   └──➤ Use Single Dataset: python scripts/train.py --config halfmile│
│    │                                                                        │
│    ├── "Train multiple datasets"                                           │
│    │   └──➤ Use Batch Training: python scripts/batch_train.py             │
│    │                                                                        │
│    ├── "Debug or investigate"                                              │
│    │   └──➤ Use Verbose Mode: --verbose --log-level DEBUG --log-memory   │
│    │                                                                        │
│    ├── "Prepare data only"                                                 │
│    │   └──➤ Use Preprocess Only: python scripts/preprocess.py             │
│    │                                                                        │
│    ├── "Validate trained model"                                            │
│    │   └──➤ Use Evaluate Only: python scripts/evaluate.py --model best   │
│    │                                                                        │
│    └── "Deploy to production"                                              │
│        └──➤ Use Export: python scripts/export_model.py --onnx --torchscript│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This gives you full flexibility to run exactly what you need, from quick tests to full production training!