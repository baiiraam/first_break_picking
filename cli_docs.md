# Deep Dive: `scripts/` Folder — Complete Documentation

The `scripts/` folder contains all **executable entry points** for the First Break Picking system. These are the CLI commands that users actually run to preprocess data, train models, evaluate results, and export models for production.

---

## Quick Reference: All Scripts

| Script | Purpose | Entry Point |
|--------|---------|-------------|
| `batch_train.py` | Multi-dataset, multi-model orchestration with auto-config | `python scripts/batch_train.py` |
| `train.py` | Single model training on one dataset | `python scripts/train.py` |
| `evaluate.py` | Evaluate trained models | `python scripts/evaluate.py` |
| `preprocess.py` | Convert HDF5 to chunks | `python scripts/preprocess.py` |
| `visualize.py` | Generate prediction visualizations | `python scripts/visualize.py` |
| `export_model.py` | Export to ONNX/TorchScript | `python scripts/export_model.py` |
| `sweep_mlflow.py` | Grid search with MLflow tracking | `python scripts/sweep_mlflow.py` |
| `check_device_memory.py` | Detect device memory and recommend configs | `python scripts/check_device_memory.py` |
| `search_models.py` | Search MLflow model registry | `python scripts/search_models.py` |
| `run_model_pairs.py` | Train model pairs across datasets | `python scripts/run_model_pairs.py` |

---

## 1. `batch_train.py` — The Main Orchestrator

### Purpose
The **master orchestrator** for batch training across multiple datasets and models. It handles:
- Sequential training of datasets
- Automatic memory error recovery
- Smart configuration detection
- Fallback variants (aggressive → minimal)
- Dataset-specific model overrides

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Sequential Processing** | Trains one dataset at a time, one model at a time |
| **Memory Recovery** | If a config causes memory error, tries next config |
| **Smart Config Detection** | Auto-calculates optimal batch_size, cache_size, memory_limit |
| **Fallback Variants** | 5+ config levels (aggressive → minimal) |
| **Graceful Failure** | If all fail for a dataset, moves to next dataset |

### CLI Options

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
| `--loss` | `-l` | String | `cross_entropy` | Loss function |
| `--class-weights` | `-cw` | 3 Floats | - | Class weights |

### Example Usage

```bash
# Train all datasets with auto-config
python scripts/batch_train.py --auto-config --epochs 30

# Train only specific datasets with specific models
python scripts/batch_train.py --auto-config --datasets Halfmile --models pico --epochs 2

# Train with custom loss and verbose logging
python scripts/batch_train.py --auto-config --loss combo --verbose --log-memory

# Quick test on one dataset
python scripts/batch_train.py --auto-config --datasets Halfmile --models pico --epochs 2 --verbose
```

### Internal Flow

```python
def run_auto_batch_training(...):
    # 1. Load configuration
    batch_config = load_batch_config(config_file)
    global_config = batch_config["global"]
    dataset_overrides = batch_config.get("datasets", {})
    auto_config = batch_config.get("auto", {})
    
    # 2. Detect device memory
    info = get_device_info()
    available_gb = get_recommended_memory_limits(info)["mps"]["recommended_gb"]
    device_type = "mps"  # or "cuda" or "cpu"
    
    # 3. Generate smart variants for each model
    all_variants = []
    for model_name in model_order:
        variants = generate_aggressive_variants(
            model_name, available_gb, device_type, max_batch_size, max_cache_size
        )
        all_variants.extend(variants)
    
    # 4. For each dataset
    for dataset_name in selected_datasets:
        logger.info(f"📊 DATASET: {dataset_name}")
        
        # 4a. Check memory before
        mem = check_memory_usage()
        
        # 4b. Apply dataset-specific model list
        ds_config = dataset_overrides.get(dataset_name, {})
        if "models" in ds_config:
            dataset_variants = [v for v in all_variants if v["model"] in ds_config["models"]]
        else:
            dataset_variants = all_variants
        
        # 4c. Try each variant (sequential)
        for variant in dataset_variants:
            result = train_dataset(dataset_name, variant, global_config, extra_args)
            
            if result["success"]:
                logger.info(f"✅ SUCCESS!")
                break
            elif is_memory_error(result["error"]):
                logger.info(f"🔄 Memory error, trying next variant")
                clear_memory()
                continue
            else:
                break
        
        # 4d. Cleanup between datasets
        clear_memory()
    
    # 5. Save summary
    save_summary(results)
```

### The `train_dataset()` Function

```python
def train_dataset(dataset_name, config_variant, global_config, extra_args):
    """Train a single dataset with a specific config variant."""
    
    # 1. Build command
    cmd = [
        "python3.12",
        "scripts/train.py",
        "--config", f"configs/{dataset_name.lower()}.yaml",
        "--model", config_variant["model"],
        "--epochs", str(global_config.get("epochs", 30)),
        "--batch-size", str(config_variant["batch_size"]),
        "--device", global_config.get("device", "mps"),
    ]
    
    # 2. Add class weights
    if config_variant.get("class_weights"):
        weights = config_variant["class_weights"].split(",")
        cmd.append("--class-weights")
        cmd.extend(weights)
    
    # 3. Set memory environment
    env = os.environ.copy()
    memory_limit = config_variant.get("memory_limit_gb", 8)
    env["PYTORCH_MPS_MEMORY_LIMIT"] = str(int(memory_limit * 1e9))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # 4. Run subprocess
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    duration = time.time() - start_time
    
    # 5. Check for real errors (filter MLflow messages)
    combined_output = result.stdout + result.stderr
    has_real_error = is_real_error(combined_output)
    success = not has_real_error
    
    return {
        "success": success,
        "duration": duration,
        "error": combined_output if not success else None,
        "return_code": result.returncode,
    }
```

### Memory Error Detection

```python
def is_real_error(output: str) -> bool:
    """Check if output contains a REAL error (not MLflow info)."""
    
    # 1. Filter out MLflow info messages
    mlflow_patterns = [
        "Skip logging GPU metrics",
        "Set logger level to DEBUG",
        "mlflow.system_metrics",
        "INFO mlflow",
    ]
    for pattern in mlflow_patterns:
        if pattern in str(output):
            return False
    
    # 2. Check for real errors
    error_patterns = [
        "RuntimeError",
        "ValueError",
        "MPS out of memory",
        "CUDA out of memory",
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
    ]
    for pattern in error_patterns:
        if pattern in str(output):
            return True
    
    return False
```

---

## 2. `train.py` — Single Model Training

### Purpose
Trains a **single model** on a **single dataset**. This is the workhorse that `batch_train.py` calls.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | String | `unet` | Model architecture |
| `--epochs` | `-e` | Integer | 30 | Number of epochs |
| `--loss` | `-l` | String | `cross_entropy` | Loss function |
| `--class-weights` | `-cw` | 3 Floats | - | Class weights |
| `--batch-size` | `-b` | Integer | Config | Override batch size |
| `--cache-size` | - | Integer | Config | Override cache size |
| `--device` | `-d` | String | `mps` | Device to use |
| `--verbose` | `-v` | Flag | False | Verbose output |
| `--log-memory` | `-lm` | Flag | False | Log memory usage |
| `--resume` | `-r` | String | None | Resume from checkpoint |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |

### Example Usage

```bash
# Train MPSLightUNet on Halfmile with Combo Loss
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --loss combo \
    --class-weights 0.1 0.1 0.8 \
    --verbose \
    --log-memory

# Quick test with PicoUNet
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model pico \
    --epochs 2 \
    --batch-size 8 \
    --verbose

# Resume from checkpoint
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --resume models/registry/checkpoint_epoch_10.pt
```

### Internal Flow

```python
def main(config, model, epochs, loss, class_weights, batch_size, ...):
    """Train a single model on a single dataset."""
    
    # 1. Load config from YAML
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Create SeismicConfig (applies defaults + YAML overrides)
    cfg = SeismicConfig(**config_dict)
    
    # 3. Apply CLI overrides
    if epochs: cfg.n_epochs = epochs
    if batch_size: cfg.batch_size = batch_size
    if class_weights: cfg.class_weights = list(class_weights)
    if device: cfg.device = device
    if loss: cfg.loss_function = loss
    
    # 4. Load data (manifest → chunks)
    manifest = load_manifest(f"data/chunks/{cfg.dataset_name}/manifest.json")
    data_manager = ChunkedDataManager(
        chunk_dir=f"data/chunks/{cfg.dataset_name}",
        manifest=manifest,
        cache_size=cfg.cache_size,
    )
    
    # 5. Create dataloaders
    train_loader = DataLoader(
        data_manager.get_dataset("train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        data_manager.get_dataset("val"),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    
    # 6. Create model
    model_obj = create_model(model, in_channels=1, out_channels=3)
    
    # 7. Create loss function
    criterion = create_loss_function(cfg)
    
    # 8. Create optimizer
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=cfg.learning_rate)
    
    # 9. Create trainer
    trainer = SeismicTrainer(
        model=model_obj,
        dataloaders={"train": train_loader, "val": val_loader},
        criterion=criterion,
        optimizer=optimizer,
        config=cfg,
        model_name=model,
    )
    
    # 10. Train!
    trainer.fit()
    
    logger.info("✅ TRAINING COMPLETE!")
```

### Supported Models

| Model | Class | File |
|-------|-------|------|
| `unet` | `UNet` | `src/models/unet.py` |
| `mpslight` | `MPSLightUNet` | `src/models/mps_light_unet.py` |
| `light` | `LightUNet` | `src/models/light_unet.py` |
| `nano` | `NanoUNet` | `src/models/light_unet.py` |
| `tiny` | `TinyUNet` | `src/models/tiny_unet.py` |
| `pico` | `PicoUNet` | `src/models/pico_unet.py` |
| `mobile` | `MobileUNet` | `src/models/mobilenet.py` |
| `efficient` | `EfficientUNet` | `src/models/efficient_unet.py` |

---

## 3. `evaluate.py` — Model Evaluation

### Purpose
Evaluates a trained model on test/validation data and computes comprehensive metrics.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model path or "best" |
| `--split` | `-s` | String | `test` | Split to evaluate |
| `--detailed` | - | Flag | False | Generate per-shot metrics |
| `--device` | `-d` | String | `mps` | Device to use |
| `--batch-size` | `-b` | Integer | 4 | Batch size |

### Example Usage

```bash
# Evaluate champion model
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test

# Evaluate specific model with detailed metrics
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test \
    --detailed

# Evaluate on all splits
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split all
```

### Metrics Computed

| Category | Metric | Description |
|----------|--------|-------------|
| **Segmentation** | Accuracy | Pixel-wise accuracy |
| | Mean IoU | Mean Intersection over Union |
| | Mean F1 | Mean F1 score |
| | IoU per class | IoU for Before, After, Strip |
| **First Break** | MAE | Mean Absolute Error (samples) |
| | Std AE | Standard deviation of error |
| | Median AE | Median error |
| | ±3 Accuracy | % within ±3 samples |

### Output Files

```
evaluation_results/
├── evaluation_results_Halfmile_20260902_123456.json
├── evaluation_summary_Halfmile_20260902_123456.csv
└── detailed_errors_Halfmile_test_20260902_123456.csv  (if --detailed)
```

---

## 4. `preprocess.py` — Data Preprocessing

### Purpose
Converts raw HDF5 files into chunked PyTorch tensors with 3-class segmentation masks.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--force` | `-f` | Flag | False | Force reprocessing |
| `--dataset` | `-d` | String | None | Override dataset name |

### Example Usage

```bash
# Preprocess a dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# Force reprocess (overwrite existing chunks)
python scripts/preprocess.py --config configs/halfmile.yaml --force

# Preprocess a specific dataset
python scripts/preprocess.py --config configs/halfmile.yaml --dataset Halfmile
```

### Internal Flow

```python
def main(config, force, dataset):
    """Run the preprocessing pipeline."""
    
    # 1. Phase 1: Data Discovery
    unique_shots, start_indices, end_indices = load_shot_indices(cfg.hdf5_path)
    logger.info(f"Total shots: {len(unique_shots)}")
    
    # 2. Filter valid shots (>= 10 traces)
    valid_shots = unique_shots[(end_indices - start_indices) >= 10]
    logger.info(f"Valid shots: {len(valid_shots)}")
    
    # 3. Phase 2: Chunk Assignment
    chunker = Chunker(
        chunk_size=cfg.chunk_size,
        train_split=cfg.train_split,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
    )
    splits = chunker.assign_splits(valid_shots)
    chunks = {}
    for split_name, shot_list in splits.items():
        chunks[split_name] = chunker.create_chunks(shot_list)
    
    # 4. Phase 3: Processing and Writing
    processor = ShotProcessor(
        target_traces=cfg.target_traces,
        n_samples=cfg.n_samples,
        strip_width=cfg.strip_width,
    )
    
    for split_name, chunk_list in chunks.items():
        for chunk in chunk_list:
            # Process each shot in the chunk
            for i, shot_id in enumerate(chunk["shot_ids"]):
                shot_data, shot_picks = load_shot_data(
                    cfg.hdf5_path,
                    start_indices[shot_id],
                    end_indices[shot_id],
                )
                processed_data, processed_mask = processor.process_shot(
                    shot_data, shot_picks
                )
                data_batch[i] = processed_data
                mask_batch[i] = processed_mask
            
            # Save chunk
            torch.save({
                'data': data_batch,
                'mask': mask_batch,
                'shot_ids': chunk["shot_ids"],
                'split': split_name,
            }, chunk_path)
    
    # 5. Phase 4: Generate Manifest
    manifest = generate_manifest(cfg.dataset_name, chunks, cfg.to_dict())
    save_manifest(manifest)
    
    logger.info("✅ PREPROCESSING COMPLETE!")
```

### Output Structure

```
data/chunks/Halfmile/
├── manifest.json
├── chunk_001_train.pt  (69 shots)
├── chunk_002_train.pt  (44 shots)
├── chunk_001_val.pt    (14 shots)
└── chunk_001_test.pt   (15 shots)
```

---

## 5. `visualize.py` — Result Visualization

### Purpose
Generates side-by-side visualizations of seismogram, ground truth mask, and model prediction.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | Required | - | Config file path |
| `--model` | `-m` | Required | - | Model path |
| `--output` | `-o` | String | `visualization_results` | Output directory |
| `--n_samples` | `-n` | Integer | 10 | Number of samples |
| `--device` | `-d` | String | `mps` | Device to use |

### Example Usage

```bash
# Visualize 10 predictions
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model best \
    --n_samples 10 \
    --output vis_results

# Visualize specific model
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 5
```

### Output

```
visualization_results/
├── shot_123_comparison.png
├── shot_124_comparison.png
├── shot_125_comparison.png
└── ...
```

Each image shows:
- Left: Original seismogram
- Middle: Ground truth mask
- Right: Model prediction

---

## 6. `export_model.py` — Model Export

### Purpose
Exports trained models to production formats: ONNX and TorchScript.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model` | `-m` | Required | - | Model checkpoint path |
| `--output` | `-o` | String | `exported_models` | Output directory |
| `--onnx` | - | Flag | False | Export to ONNX |
| `--torchscript` | - | Flag | False | Export to TorchScript |
| `--device` | `-d` | String | `cpu` | Device for export |
| `--model-type` | `-t` | String | `unet` | Model architecture |
| `--config` | `-c` | String | None | Config file path |

### Example Usage

```bash
# Export to ONNX only
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --onnx \
    --output production_models

# Export to both formats
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --onnx \
    --torchscript

# Export from MLflow champion
python scripts/export_model.py \
    --model models:/halfmile@champion \
    --onnx \
    --torchscript
```

### Internal Flow

```python
def main(model, output, onnx, torchscript, device, model_type, config):
    """Export model to ONNX and/or TorchScript."""
    
    # 1. Load model
    model_obj = load_model(model, model_type)
    model_obj.eval()
    model_obj = model_obj.to(device)
    
    # 2. Create example input
    example_input = torch.randn(1, 1, 1578, 751).to(device)
    
    # 3. Export to TorchScript
    if torchscript:
        scripted_model = torch.jit.trace(model_obj, example_input)
        torch.jit.save(scripted_model, output / "model_scripted.pt")
        logger.info(f"✅ Saved TorchScript: {output}/model_scripted.pt")
    
    # 4. Export to ONNX
    if onnx:
        torch.onnx.export(
            model_obj,
            example_input,
            output / "model.onnx",
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}},
            opset_version=11,
            do_constant_folding=True,
        )
        logger.info(f"✅ Saved ONNX: {output}/model.onnx")
        
        # Verify ONNX model
        import onnx
        onnx_model = onnx.load(output / "model.onnx")
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model verified")
```

---

## 7. `sweep_mlflow.py` — Grid Search

### Purpose
Runs a grid search over datasets, models, and loss functions with MLflow tracking.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | String | `configs/sweep_config.yaml` | Sweep config file |

### Example Config

```yaml
# configs/sweep_config.yaml
sweep:
  datasets:
    - "Halfmile"
    - "Brunswick"
  models:
    - "pico"
    - "mpslight"
  losses:
    - "cross_entropy"
    - "combo"

global:
  epochs: 2
  device: "mps"
  verbose: true
```

### Example Usage

```bash
# Run sweep
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml

# Run with specific config
python scripts/sweep_mlflow.py -c configs/my_sweep_config.yaml
```

### Internal Flow

```python
def run_sweep(config_file):
    """Run grid search over datasets, models, and losses."""
    
    # 1. Load sweep config
    config = load_sweep_config(config_file)
    datasets = config["sweep"]["datasets"]
    models = config["sweep"]["models"]
    losses = config["sweep"]["losses"]
    
    total = len(datasets) * len(models) * len(losses)
    logger.info(f"🧪 Running {total} experiments...")
    
    # 2. For each combination
    for dataset in datasets:
        for model in models:
            for loss in losses:
                # 3. Start MLflow run
                with mlflow.start_run(run_name=f"{dataset}_{model}_{loss}"):
                    mlflow.log_params({
                        "dataset": dataset,
                        "model": model,
                        "loss": loss,
                    })
                    
                    # 4. Run training
                    result = run_training(dataset, model, loss)
                    
                    # 5. Log metrics
                    mlflow.log_metrics(result["metrics"])
                    mlflow.log_artifact(result["model_path"])
    
    logger.info("✅ Sweep complete!")
```

---

## 8. `check_device_memory.py` — Device Detection

### Purpose
Detects device memory and recommends optimal training configurations.

### Example Usage

```bash
python scripts/check_device_memory.py
```

### Output Example

```
🔍 DEVICE & MEMORY INFORMATION
================================================================================

📱 System:
   OS: Darwin (Darwin Kernel Version 24.3.0)
   Machine: arm64

🔥 PyTorch:
   Version: 2.13.0
   MPS Available: True

💻 CPU Memory:
   Total: 16.0 GB
   Available: 10.2 GB

🍏 MPS Memory:
   System RAM: 16.0 GB
   Available: 10.2 GB
   MPS Limit: 12.0 GB
   💡 Recommended Limit: 9.6 GB
   💡 Max Safe: 10.8 GB

📊 OPTIMAL TRAINING CONFIGURATIONS
--------------------------------------------------------------------------------

🟢 PICO (2,000 params):
   Status: ✅ WILL FIT
   Batch Size: 8
   Cache Size: 5
   Memory Limit: 0.1 GB

🟢 MPSLIGHT (1,700,000 params):
   Status: ✅ WILL FIT
   Batch Size: 6
   Cache Size: 4
   Memory Limit: 0.6 GB

⚠️ UNET (31,000,000 params):
   Status: ⚠️ MAY NOT FIT
   Batch Size: 3
   Cache Size: 2
   Memory Limit: 10.5 GB
```

---

## 9. `search_models.py` — MLflow Model Search

### Purpose
Searches and compares models in the MLflow registry.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dataset` | `-d` | String | None | Filter by dataset |
| `--model-type` | `-m` | String | None | Filter by model type |
| `--min-iou` | - | Float | None | Minimum IoU threshold |
| `--top` | `-n` | Integer | 10 | Number of results |
| `--compare` | `-c` | Flag | False | Compare models side by side |

### Example Usage

```bash
# Search all models
python scripts/search_models.py

# Search Halfmile models with IoU > 0.5
python scripts/search_models.py --dataset Halfmile --min-iou 0.5

# Compare top 2 models
python scripts/search_models.py --dataset Halfmile --compare --top 2
```

### Output Example

```
🔍 TOP MODELS (Halfmile):
--------------------------------------------------------------------------------
#   Model                         Dataset      IoU     F1      Class2 IoU
--------------------------------------------------------------------------------
1   MPSLightUNet_Halfmile_epoch_30 Halfmile     0.65    0.72    0.85
2   UNet_Halfmile_epoch_30         Halfmile     0.68    0.70    0.82
3   LightUNet_Halfmile_epoch_30    Halfmile     0.60    0.68    0.78

📊 Model Comparison:
--------------------------------------------------------------------------------

Model 1: MPSLightUNet_Halfmile_epoch_30
  Dataset: Halfmile
  Type: MPSLightUNet
  IoU: 0.65
  URI: models:/halfmile/4
```

---

## 10. `run_model_pairs.py` — Model Pairs Training

### Purpose
Trains model pairs across all datasets in a controlled sequence.

### CLI Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--epochs` | `-e` | Integer | 2 | Number of epochs |
| `--device` | `-d` | String | `mps` | Device to use |
| `--dry-run` | - | Flag | False | Print commands without running |
| `--verbose` | `-v` | Flag | False | Verbose output |
| `--no-log-memory` | - | Flag | False | Disable memory logging |

### Example Usage

```bash
# Quick test with 2 epochs
python scripts/run_model_pairs.py --epochs 2 --verbose

# Full training with 30 epochs
python scripts/run_model_pairs.py --epochs 30 --verbose

# Dry run (see what would be executed)
python scripts/run_model_pairs.py --dry-run

# Run on CPU
python scripts/run_model_pairs.py --device cpu --epochs 2
```

### Internal Flow

```python
def run_model_pairs(model_pairs, datasets, epochs, device, verbose, log_memory, dry_run):
    """Run model pairs across all datasets."""
    
    # Outer loop: Model pairs
    for pair_idx, model_pair in enumerate(model_pairs, 1):
        logger.info(f"📊 PAIR {pair_idx}/{len(model_pairs)}: {model_pair}")
        
        # Inner loop: Datasets
        for dataset_idx, dataset in enumerate(datasets, 1):
            logger.info(f"📁 DATASET {dataset_idx}/{len(datasets)}: {dataset}")
            
            # Train each model in the pair on this dataset
            for model in model_pair:
                # Skip if model doesn't fit this dataset
                if model in SKIP_PER_DATASET.get(dataset, []):
                    logger.info(f"⚠️ Skipping {model} on {dataset} (won't fit)")
                    continue
                
                # Build command
                cmd = [
                    "python3.12",
                    "scripts/train.py",
                    "--config", f"configs/{dataset.lower()}.yaml",
                    "--model", model,
                    "--epochs", str(epochs),
                    "--device", device,
                    "--loss", "combo",
                ]
                
                if verbose: cmd.append("--verbose")
                if log_memory: cmd.append("--log-memory")
                
                if dry_run:
                    logger.info(f"🏃 DRY RUN: {' '.join(cmd)}")
                    continue
                
                # Run training
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ SUCCESS! {model} on {dataset}")
                else:
                    logger.error(f"❌ FAILED! {model} on {dataset}")
```

---

## Summary: Scripts Quick Reference

| Script | When to Use | Command |
|--------|-------------|---------|
| `batch_train.py` | Training multiple datasets/models | `python scripts/batch_train.py --auto-config` |
| `train.py` | Training one model on one dataset | `python scripts/train.py --config configs/halfmile.yaml --model mpslight` |
| `evaluate.py` | Evaluating a model | `python scripts/evaluate.py --config configs/halfmile.yaml --model best` |
| `preprocess.py` | Preprocessing raw data | `python scripts/preprocess.py --config configs/halfmile.yaml` |
| `visualize.py` | Visualizing predictions | `python scripts/visualize.py --config configs/halfmile.yaml --model best` |
| `export_model.py` | Exporting to production | `python scripts/export_model.py --model model.pt --onnx` |
| `sweep_mlflow.py` | Grid search experiments | `python scripts/sweep_mlflow.py --config configs/sweep_config.yaml` |
| `check_device_memory.py` | Checking device memory | `python scripts/check_device_memory.py` |
| `search_models.py` | Searching MLflow registry | `python scripts/search_models.py --dataset Halfmile` |
| `run_model_pairs.py` | Training model pairs | `python scripts/run_model_pairs.py --epochs 2` |

---

## Common Workflows

### Workflow 1: Quick Test

```bash
# 1. Check device memory
python scripts/check_device_memory.py

# 2. Preprocess a dataset
python scripts/preprocess.py --config configs/halfmile.yaml

# 3. Quick test with PicoUNet (2 epochs)
python scripts/train.py --config configs/halfmile.yaml --model pico --epochs 2 --verbose --log-memory

# 4. Evaluate
python scripts/evaluate.py --config configs/halfmile.yaml --model best --split test

# 5. Visualize
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 5
```

### Workflow 2: Full Training

```bash
# Train all datasets with auto-config
python scripts/batch_train.py --auto-config --epochs 30 --verbose --log-memory

# Evaluate all models
for dataset in Brunswick Halfmile Lalor Sudbury; do
    python scripts/evaluate.py --config configs/${dataset,,}.yaml --model best --split test
done

# Export best model
python scripts/export_model.py --model models:/halfmile@champion --onnx --torchscript
```

### Workflow 3: Model Pairs

```bash
# Quick test
python scripts/run_model_pairs.py --epochs 2 --verbose --dry-run

# Full run
python scripts/run_model_pairs.py --epochs 30 --verbose --log-memory
```
