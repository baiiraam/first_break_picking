# Deep Dive: Complete Documentation of `configs/`, `src/`, and `scripts/` Folders

---

## 1. `configs/` Folder — Configuration Management

The `configs/` folder contains all YAML configuration files that control the behavior of the First Break Picking system. These files act as the **control center** for the entire pipeline.

### Folder Structure

```
configs/
├── batch_config.yaml          # Master batch training configuration
├── brunswick.yaml             # Brunswick dataset parameters
├── halfmile.yaml              # Halfmile dataset parameters
├── lalor.yaml                 # Lalor dataset parameters
├── sudbury.yaml               # Sudbury dataset parameters
├── default.yaml               # Base configuration template
├── production.yaml            # Production mode overrides
└── sweep_config.yaml          # Grid search sweep configuration
```

---

### 1.1 `batch_config.yaml` — The Master Orchestrator

**Purpose:** This is the **most important configuration file**. It controls the entire batch training pipeline, orchestrating multiple datasets, models, and loss functions with automatic memory error recovery.

**Structure:**

```yaml
# ============================================================
# BATCH TRAINING CONFIGURATION (with Loss Functions)
# ============================================================

# --- SECTION 1: GLOBAL SETTINGS ---
global:
  # Training parameters
  epochs: 30
  device: "mps"                    # cpu, cuda, mps
  log_memory: true
  verbose: true
  log_level: "DEBUG"               # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
  # Loss Function Configuration
  loss_function: "combo"           # cross_entropy, focal, dice, combo
  class_weights: [0.1, 0.1, 0.8]  # Before, After, Strip
  dice_weight: 0.5                 # For combo loss
  focal_gamma: 2.0                 # For focal/combo loss
  
  # Pipeline settings
  preprocess: false
  checkpoint_every: 5
  early_stopping: 5
  skip_failed: true
  clear_memory_between_datasets: true
  pause_between_datasets: 2

# --- SECTION 2: DATASET-SPECIFIC OVERRIDES ---
datasets:
  Brunswick:
    epochs: 40
    log_memory: true
  
  Halfmile:
    log_level: "DEBUG"
  
  Lalor:
    batch_size_override: 2
    model_override: "tiny"
    preprocess: true
  
  Sudbury:
    epochs: 25
    device: "cpu"

# --- SECTION 3: AUTO-CONFIG SETTINGS ---
auto:
  strategy: "smart"
  memory_usage: 0.85
  
  # Loss function overrides per dataset
  loss_overrides:
    Lalor:
      loss_function: "dice"
      class_weights: [0.1, 0.1, 0.8]
    
    Sudbury:
      loss_function: "focal"
      focal_gamma: 3.0
    
    Halfmile:
      loss_function: "combo"
      dice_weight: 0.6
  
  model_order:
    - "pico"
    - "nano"
    - "tiny"
    - "mpslight"
    - "light"
    - "mobile"
    - "efficient"
    - "unet"
  
  skip_for_large:
    - "unet"
```

**Key Concepts:**

| Concept | Description |
|---------|-------------|
| **Global Settings** | Applied to all datasets unless overridden |
| **Dataset Overrides** | Per-dataset customizations (epochs, device, batch size) |
| **Loss Overrides** | Per-dataset loss function selection (useful for large datasets) |
| **Model Order** | The sequence in which models are tried (smallest to largest) |
| **Skip for Large** | Models automatically skipped for large datasets (prevents OOM) |

**How Loss Overrides Work:**

```python
# In batch_train.py
def get_loss_for_dataset(dataset_name):
    # Global default
    loss_function = global_config.get("loss_function", "combo")
    class_weights = global_config.get("class_weights", [0.1, 0.1, 0.8])
    dice_weight = global_config.get("dice_weight", 0.5)
    focal_gamma = global_config.get("focal_gamma", 2.0)
    
    # Dataset-specific override
    loss_overrides = auto_config.get("loss_overrides", {})
    if dataset_name in loss_overrides:
        override = loss_overrides[dataset_name]
        loss_function = override.get("loss_function", loss_function)
        class_weights = override.get("class_weights", class_weights)
        dice_weight = override.get("dice_weight", dice_weight)
        focal_gamma = override.get("focal_gamma", focal_gamma)
    
    return loss_function, class_weights, dice_weight, focal_gamma
```

**Example: What Happens for Each Dataset**

| Dataset | Loss Function | Class Weights | Why |
|---------|--------------|---------------|-----|
| **Brunswick** | Combo (global) | [0.1, 0.1, 0.8] | Default, balanced |
| **Halfmile** | Combo (override) | [0.1, 0.1, 0.8] | Higher dice weight (0.6) |
| **Lalor** | Dice (override) | [0.1, 0.1, 0.8] | Large dataset, Dice is more stable |
| **Sudbury** | Focal (override) | [0.1, 0.1, 0.8] | Focal with higher gamma (3.0) |

---

### 1.2 Dataset Config Files (`brunswick.yaml`, `halfmile.yaml`, etc.)

**Purpose:** Each dataset has its own configuration file that defines data paths, shape parameters, training settings, and caching behavior.

**Example: `halfmile.yaml`**

```yaml
# configs/halfmile.yaml

# === Dataset ===
dataset_name: "Halfmile"
hdf5_path: "data/raw/Halfmile3D_add_geom_sorted.hdf5"
chunk_dir: "data/chunks"
preprocess: false
force_reprocess: false

# === Data ===
target_traces: 1578              # Number of traces per shot
n_samples: 751                   # Time samples per trace
strip_width: 8                   # First-break strip width (must be even)
chunk_size: 69                   # Shots per chunk
random_seed: 42
train_split: 0.8
val_split: 0.1
test_split: 0.1

# === Training ===
batch_size: 4
learning_rate: 0.001
n_epochs: 30
device: "mps"
num_workers: 4

# === Loss ===
class_weights: [0.1, 0.1, 0.8]  # Before, After, Strip

# === Cache ===
cache_size: 3

# === Scheduler ===
lr_scheduler: "plateau"
lr_patience: 3
lr_factor: 0.5
lr_step_size: 10
lr_gamma: 0.5
lr_T_max: 30

# === Early Stopping ===
early_stopping_patience: 5
early_stopping_min_delta: 0.0001
```

**Dataset Comparison Table:**

| Parameter | Brunswick | Halfmile | Lalor | Sudbury |
|-----------|-----------|----------|-------|---------|
| **Traces** | 2582 | 1578 | 2685 | 1138 |
| **Samples** | 751 | 751 | 1501 | 1001 |
| **Chunk Size** | 69 | 69 | 69 | 69 |
| **Batch Size** | 4 | 4 | 4 | 4 |
| **Cache Size** | 5 | 3 | 3 | 5 |
| **Class Weights** | [0.1,0.1,0.8] | [0.1,0.1,0.8] | [0.1,0.1,0.8] | [0.1,0.1,0.8] |

---

### 1.3 `sweep_config.yaml` — Grid Search Configuration

**Purpose:** Configures the grid search sweep for running multiple experiments across datasets, models, and loss functions.

```yaml
# configs/sweep_config.yaml

global:
  epochs: 2
  device: "mps"
  log_memory: true
  verbose: true
  log_level: "INFO"

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
  
  loss_params:
    cross_entropy:
      class_weights: [0.1, 0.1, 0.8]
    focal:
      class_weights: [0.1, 0.1, 0.8]
      focal_gamma: 2.0
    dice:
      class_weights: [0.1, 0.1, 0.8]
    combo:
      class_weights: [0.1, 0.1, 0.8]
      dice_weight: 0.5
      focal_gamma: 2.0

tracking:
  enabled: true
  experiment_name: "model_loss_sweep"
  tags:
    project: "seismic_fbp"
    sweep_type: "grid_search"
```

---

### 1.4 `default.yaml` — Base Template

**Purpose:** Serves as a reference template for creating new dataset configuration files.

```yaml
# configs/default.yaml

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

dataset_name: "default"
hdf5_path: "data/raw/default.hdf5"
chunk_dir: "data/chunks"

target_traces: 1578
n_samples: 751
strip_width: 8
chunk_size: 69

train_split: 0.8
val_split: 0.1
test_split: 0.1

batch_size: 4
learning_rate: 0.001
n_epochs: 30
device: "mps"
num_workers: 0

class_weights: [0.2, 0.2, 0.6]
cache_size: 3

lr_scheduler: "plateau"
lr_patience: 3
lr_factor: 0.5
```

---

### 1.5 `production.yaml` — Production Mode

**Purpose:** Overrides for production deployment (lower memory usage, faster inference).

```yaml
# configs/production.yaml

global:
  batch_size: 1
  num_workers: 2
  checkpoint_every: 1
  device: "mps"
  log_memory: false
  verbose: false
```

---

### Configuration Precedence Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION PRECEDENCE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Default Values (SeismicConfig dataclass)                              │
│     ↓ (overwritten by)                                                     │
│  2. YAML Config File (dataset-specific .yaml)                             │
│     ↓ (overwritten by)                                                     │
│  3. Dataset Overrides (batch_config.yaml → datasets)                      │
│     ↓ (overwritten by)                                                     │
│  4. Auto-Config Variants (generated by smart detection)                   │
│     ↓ (overwritten by)                                                     │
│  5. Loss Overrides (batch_config.yaml → auto → loss_overrides)            │
│     ↓ (overwritten by)                                                     │
│  6. CLI Overrides (--epochs, --batch-size, --loss, etc.)                 │
│                                                                             │
│  FINAL CONFIGURATION = Highest precedence wins!                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Summary: `configs/` Quick Reference

| File | Purpose | Key Sections |
|------|---------|--------------|
| `batch_config.yaml` | Master batch training | global, datasets, auto, loss_overrides |
| `brunswick.yaml` | Brunswick dataset | data, training, loss, cache |
| `halfmile.yaml` | Halfmile dataset | data, training, loss, cache |
| `lalor.yaml` | Lalor dataset | data, training, loss, cache |
| `sudbury.yaml` | Sudbury dataset | data, training, loss, cache |
| `sweep_config.yaml` | Grid search | sweep, tracking, global |
| `default.yaml` | Template | All parameters with defaults |
| `production.yaml` | Production overrides | Reduced memory, inference |

---

## 2. `src/` Folder — Core Library

The `src/` folder contains the **core business logic** of the system. It's organized into subfolders by responsibility: data management, models, preprocessing, training, and utilities.

### 2.1 `src/config.py` — Configuration Management

**Purpose:** Centralized, validated configuration management using Python dataclasses.

**Key Class: `SeismicConfig`**

```python
@dataclass
class SeismicConfig:
    # Dataset Configuration
    dataset_name: str = "Halfmile"
    hdf5_path: str = "data/raw/Halfmile3D_add_geom_sorted.hdf5"
    chunk_dir: str = "data/chunks"
    
    # Data Shape
    target_traces: int = 1578
    n_samples: int = 751
    strip_width: int = 8
    chunk_size: int = 69
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Training
    batch_size: int = 4
    learning_rate: float = 1e-3
    n_epochs: int = 30
    device: str = "mps"
    num_workers: int = 0
    
    # Loss
    class_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.6])
    loss_function: str = "cross_entropy"
    dice_weight: float = 0.5
    focal_gamma: float = 2.0
    
    # Cache
    cache_size: int = 3
    
    # Logging
    log_level: str = "INFO"
    log_memory: bool = False
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__post_init__()` | Validates all parameters (positive values, split sums to 1, etc.) |
| `get_config_hash()` | Generates unique hash for experiment tracking |
| `to_dict()` | Converts to dictionary for logging |

---

### 2.2 `src/data/` — Data Management Layer

**Purpose:** Memory-efficient data loading with LRU caching.

**Files:**

| File | Purpose |
|------|---------|
| `cache.py` | LRU cache implementation with hit/miss tracking |
| `chunked_dataset.py` | PyTorch Dataset that loads chunks on-demand |
| `hdf5_dataset.py` | HDF5 lazy loader for raw data access |

**`src/data/cache.py` — LRU Cache**

```python
class LRUCache:
    def __init__(self, max_size: int = 3):
        self.cache: OrderedDict[int, Dict] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: int) -> Optional[Dict]:
        """Retrieve item, moves to end (most recent)."""
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: int, value: Dict):
        """Store item, evicts oldest if full."""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return
        
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._evict(oldest_key)
        
        self.cache[key] = value
    
    def get_stats(self) -> Dict:
        """Get cache statistics (hit rate, size, etc.)."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'active_keys': list(self.cache.keys())
        }
```

**`src/data/chunked_dataset.py` — Memory-Efficient Dataset**

```python
class ChunkedSeismicDataset(Dataset):
    def __init__(self, chunk_dir, manifest, split, cache_size=3):
        self.chunk_dir = Path(chunk_dir)
        self.manifest = manifest
        self.split = split
        self.cache = LRUCache(max_size=cache_size)
        
        # Build global index: sample_idx → (chunk_idx, local_idx)
        self.global_index = []
        self.chunk_indices = []
        self.chunk_offsets = []
        self.shot_ids = []
        
        for chunk_idx, chunk in enumerate(self.chunks):
            for local_idx in range(chunk['n_shots']):
                self.global_index.append(offset + local_idx)
                self.chunk_indices.append(chunk_idx)
                self.chunk_offsets.append(local_idx)
                self.shot_ids.append(chunk['shot_ids'][local_idx])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_idx = self.chunk_indices[idx]
        local_idx = self.chunk_offsets[idx]
        
        if chunk_idx not in self.cache:
            self._load_chunk(chunk_idx)
        
        cached_item = self.cache.get(chunk_idx)
        data = cached_item['data'][local_idx]
        mask = cached_item['mask'][local_idx]
        
        return data.unsqueeze(0).contiguous(), mask.contiguous()
```

---

### 2.3 `src/models/` — Model Architectures

**Purpose:** Eight U-Net variants with different parameter counts and memory footprints.

**Files:**

| File | Model | Params | Use Case |
|------|-------|--------|----------|
| `pico_unet.py` | PicoUNet | ~2K | Last resort, testing |
| `nano_unet.py` | NanoUNet | ~10K | Ultra-fast testing |
| `tiny_unet.py` | TinyUNet | ~50K | Quick training, fallback |
| `mps_light_unet.py` | MPSLightUNet | ~1.7M | **Recommended for MPS** |
| `light_unet.py` | LightUNet | ~2.5M | Balanced model |
| `mobilenet.py` | MobileUNet | ~3.5M | Good generalization |
| `efficient_unet.py` | EfficientUNet | ~5M | Best lightweight accuracy |
| `unet.py` | UNet | ~31M | Best accuracy |

**U-Net Architecture (All Models):**

```
Input (1, 1578, 751)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ENCODER (Downsampling)                                                   │
│  e1 ──▶ Pool ──▶ e2 ──▶ Pool ──▶ e3 ──▶ Pool ──▶ e4 ──▶ Pool          │
│  (channels increase)                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BOTTLENECK                                                               │
│  Deepest layer (highest channels)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DECODER (Upsampling with Skip Connections)                               │
│  Up ──▶ Concat(e4) ──▶ d4 ──▶ Up ──▶ Concat(e3) ──▶ d3 ──▶ ...        │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
Output (3, 1578, 751) - 3-class segmentation
```

---

### 2.4 `src/preprocessing/` — Data Preprocessing

**Purpose:** Transform raw HDF5 data into chunked PyTorch tensors with 3-class masks.

**Files:**

| File | Purpose |
|------|---------|
| `processor.py` | Shot processing & 3-class mask creation |
| `chunker.py` | Train/val/test chunking |
| `manifest.py` | Manifest generation/validation |
| `writer.py` | Chunk serialization with checksums |

**`src/preprocessing/processor.py` — Shot Processing**

```python
class ShotProcessor:
    def process_shot(self, shot_data, shot_picks):
        """Process a single shot."""
        # 1. Validate picks (clip to valid range)
        cleaned_picks, stats = self.validate_picks(shot_picks)
        
        # 2. Pad/crop to target_traces
        if actual_traces < self.target_traces:
            # Pad with zeros
        elif actual_traces > self.target_traces:
            # Crop to target_traces
        
        # 3. Create 3-class mask
        mask = self.create_mask_vectorized(shot_picks)
        
        return shot_data, mask, stats
    
    def create_mask_vectorized(self, picks):
        """Create 3-class mask using vectorized operations."""
        # Class 0: Before first break (default)
        # Class 2: Strip (within half_width of pick)
        # Class 1: After (beyond strip)
        mask = np.zeros((n_traces, self.n_samples), dtype=np.int64)
        
        strip_mask = (samples >= picks_expanded - self.half_width) & \
                     (samples <= picks_expanded + self.half_width)
        after_mask = samples > picks_expanded + self.half_width
        
        mask[strip_mask] = 2
        mask[after_mask] = 1
        mask[invalid, :] = 0  # Unlabeled traces → class 0
        
        return mask
```

---

### 2.5 `src/training/` — Training Pipeline

**Purpose:** Orchestrates training, validation, logging, and model registry.

**Files:**

| File | Purpose |
|------|---------|
| `trainer.py` | Main SeismicTrainer class |
| `metrics.py` | Segmentation & first-break metrics |
| `losses.py` | Loss functions (CrossEntropy, Focal, Dice, Combo) |
| `callbacks.py` | Training callbacks (early stopping, checkpointing) |

**`src/training/losses.py` — Loss Functions**

```python
# Factory Pattern
def create_loss_function(config) -> nn.Module:
    loss_type = getattr(config, "loss_function", "cross_entropy")
    class_weights = getattr(config, "class_weights", [0.2, 0.2, 0.6])
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    
    elif loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
    
    elif loss_type == "dice":
        return DiceLoss(num_classes=len(class_weights))
    
    elif loss_type == "combo":
        return ComboLoss(
            class_weights=class_weights,
            dice_weight=config.dice_weight,
            focal_gamma=config.focal_gamma
        )
    
    else:
        raise ValueError(f"Unknown loss function: {loss_type}")
```

**`src/training/trainer.py` — Main Trainer**

```python
class SeismicTrainer:
    def fit(self):
        """Main training loop."""
        # 1. Warmup MPS shaders (Apple Silicon)
        self._warmup_mps()
        
        # 2. Main epoch loop
        for epoch in range(self.config.n_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Checkpoint
            self._log_model_checkpoint(epoch, train_loss, val_loss, val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_loss):
                break
        
        # 3. Update model aliases (champion/challenger/staging)
        self._update_model_aliases(best_val_loss)
        
        # 4. Finalize
        self.writer.close()
        self.mlflow_manager.end_run()
    
    def _update_model_aliases(self, best_val_loss):
        """Promote champion/challenger/staging aliases."""
        champion = self.mlflow_manager.get_model_by_alias(registered_name, "champion")
        
        if champion:
            if current_val_loss < champion_val_loss:
                # New model is better → champion
                self.mlflow_manager.set_model_alias(registered_name, "champion", current_version)
                self.mlflow_manager.set_model_alias(registered_name, "challenger", champion.version)
            else:
                # New model is worse → challenger
                self.mlflow_manager.set_model_alias(registered_name, "challenger", current_version)
        else:
            # No champion yet → first model is champion
            self.mlflow_manager.set_model_alias(registered_name, "champion", current_version)
        
        # Always set staging to latest
        self.mlflow_manager.set_model_alias(registered_name, "staging", current_version)
```

---

### 2.6 `src/utils/` — Utilities

**Purpose:** Cross-cutting concerns: logging, MLflow, TensorBoard, HDF5 I/O, memory management.

**Files:**

| File | Purpose |
|------|---------|
| `logger.py` | Loguru logging with date-based rotation |
| `mlflow_utils.py` | MLflow experiment tracking and model registry |
| `tensorboard_utils.py` | TensorBoard logging with seismic visualizations |
| `hdf5_utils.py` | HDF5 file operations |
| `memory_utils.py` | Memory management and monitoring |

---

## 3. `scripts/` Folder — Executable Entry Points

The `scripts/` folder contains all executable CLI commands that users actually run.

### 3.1 `batch_train.py` — The Master Orchestrator

**Purpose:** Orchestrates training across multiple datasets, models, and configurations with automatic memory error recovery.

**Key Features:**
- Sequential training of datasets
- Automatic memory error recovery
- Smart configuration detection
- Fallback variants (aggressive → minimal)
- Dataset-specific model overrides
- Loss function overrides per dataset

**CLI Options:**

```bash
python scripts/batch_train.py [OPTIONS]

Options:
  -c, --config TEXT          Config file path [default: configs/batch_config.yaml]
  -d, --datasets TEXT        Datasets to train (can specify multiple)
  -m, --models TEXT          Models to train (can specify multiple)
  -e, --epochs INTEGER       Number of epochs
  --device TEXT              Device (cpu, cuda, mps)
  -lm, --log-memory          Enable memory logging
  -v, --verbose              Verbose output
  -ll, --log-level TEXT      Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  -p, --preprocess           Force preprocessing
  -a, --auto-config          Auto-detect optimal config
  -l, --loss TEXT            Loss function (cross_entropy, focal, dice, combo)
  -cw, --class-weights FLOAT...  Class weights (3 values)
```

**How Loss Overrides Work in `batch_train.py`:**

```python
# In batch_train.py
def get_loss_for_dataset(dataset_name, auto_config, global_config):
    # Start with global defaults
    loss_function = global_config.get("loss_function", "combo")
    class_weights = global_config.get("class_weights", [0.1, 0.1, 0.8])
    dice_weight = global_config.get("dice_weight", 0.5)
    focal_gamma = global_config.get("focal_gamma", 2.0)
    
    # Apply dataset-specific overrides from auto section
    loss_overrides = auto_config.get("loss_overrides", {})
    if dataset_name in loss_overrides:
        override = loss_overrides[dataset_name]
        loss_function = override.get("loss_function", loss_function)
        class_weights = override.get("class_weights", class_weights)
        dice_weight = override.get("dice_weight", dice_weight)
        focal_gamma = override.get("focal_gamma", focal_gamma)
    
    return loss_function, class_weights, dice_weight, focal_gamma
```

**Example Usage:**

```bash
# Quick test with PicoUNet on Halfmile
python scripts/batch_train.py --auto-config --epochs 2 --models pico --datasets Halfmile --verbose

# Full training with custom loss
python scripts/batch_train.py --auto-config --epochs 30 --loss focal --verbose --log-memory

# Train specific datasets with specific loss overrides
python scripts/batch_train.py --auto-config --datasets Lalor --loss dice --verbose
```

---

### 3.2 `train.py` — Single Model Training

**Purpose:** Trains a **single model** on a **single dataset**. This is the workhorse that `batch_train.py` calls.

**CLI Options:**

```bash
python scripts/train.py [OPTIONS]

Options:
  -c, --config TEXT          Config file path [required]
  -m, --model TEXT           Model architecture [default: unet]
  -e, --epochs INTEGER       Number of epochs
  -l, --loss TEXT            Loss function (cross_entropy, focal, dice, combo)
  -cw, --class-weights FLOAT...  Class weights (3 values)
  -b, --batch-size INTEGER   Batch size override
  --cache-size INTEGER       Cache size override
  -d, --device TEXT          Device override
  -v, --verbose              Verbose output
  -lm, --log-memory          Enable memory logging
  -r, --resume TEXT          Resume from checkpoint
```

**Internal Flow:**

```python
def main(config, model, epochs, loss, class_weights, batch_size, ...):
    # 1. Load config from YAML
    cfg = SeismicConfig(**yaml.safe_load(open(config, 'r')))
    
    # 2. Apply CLI overrides
    if epochs: cfg.n_epochs = epochs
    if batch_size: cfg.batch_size = batch_size
    if class_weights: cfg.class_weights = list(class_weights)
    if loss: cfg.loss_function = loss
    
    # 3. Load data (manifest → chunks)
    manifest = load_manifest(f"data/chunks/{cfg.dataset_name}/manifest.json")
    data_manager = ChunkedDataManager(...)
    
    # 4. Create model
    model_obj = create_model(model)
    
    # 5. Create loss function
    criterion = create_loss_function(cfg)
    
    # 6. Create trainer and train
    trainer = SeismicTrainer(...)
    trainer.fit()
```

---

### 3.3 `evaluate.py` — Model Evaluation

**Purpose:** Evaluates a trained model on test/validation data with comprehensive metrics.

**CLI Options:**

```bash
python scripts/evaluate.py [OPTIONS]

Options:
  -c, --config TEXT          Config file path [required]
  -m, --model TEXT           Model path or "best" [required]
  -s, --split TEXT           Split to evaluate (train, val, test, all)
  --detailed                 Generate per-shot metrics
  -d, --device TEXT          Device to use
  -b, --batch-size INTEGER   Batch size
```

**Metrics Computed:**

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

---

### 3.4 `preprocess.py` — Data Preprocessing

**Purpose:** Converts raw HDF5 files into chunked PyTorch tensors with 3-class masks.

**CLI Options:**

```bash
python scripts/preprocess.py [OPTIONS]

Options:
  -c, --config TEXT          Config file path [required]
  -f, --force                Force reprocessing
  --dataset TEXT             Override dataset name
```

---

### 3.5 `visualize.py` — Result Visualization

**Purpose:** Generates side-by-side visualizations of seismogram, ground truth, and prediction.

```bash
python scripts/visualize.py --config configs/halfmile.yaml --model best --n_samples 10
```

---

### 3.6 `export_model.py` — Model Export

**Purpose:** Exports trained models to ONNX and TorchScript for production.

```bash
python scripts/export_model.py --model model.pt --onnx --torchscript
```

---

### 3.7 `sweep_mlflow.py` — Grid Search

**Purpose:** Runs a grid search over datasets, models, and loss functions with MLflow tracking.

```bash
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml
```

---

### 3.8 `check_device_memory.py` — Device Detection

**Purpose:** Detects device memory and recommends optimal training configurations.

```bash
python scripts/check_device_memory.py
```

---

### 3.9 `search_models.py` — MLflow Model Search

**Purpose:** Searches and compares models in the MLflow registry.

```bash
python scripts/search_models.py --dataset Halfmile --min-iou 0.5
```

---

### 3.10 `run_model_pairs.py` — Model Pairs Training

**Purpose:** Trains model pairs across all datasets in a controlled sequence.

```bash
python scripts/run_model_pairs.py --epochs 2 --verbose
```

---

## Summary: Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  USER COMMANDS (scripts/)                                          │    │
│  │  batch_train.py  │  train.py  │  evaluate.py  │  visualize.py     │    │
│  │  preprocess.py   │  export_model.py  │  sweep_mlflow.py          │    │
│  │  check_device_memory.py  │  search_models.py  │  run_model_pairs │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CONFIGURATION (configs/)                                          │    │
│  │  batch_config.yaml │ dataset.yaml │ sweep_config.yaml              │    │
│  │  default.yaml │ production.yaml                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  CORE LIBRARY (src/)                                               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │  training/  │  │  models/    │  │  data/      │               │    │
│  │  │  trainer.py │  │  8 UNet     │  │  cache.py   │               │    │
│  │  │  metrics.py │  │  variants   │  │  chunked_   │               │    │
│  │  │  losses.py  │  │             │  │  dataset.py │               │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │  utils/     │  │  preprocessing/  │  config.py │               │    │
│  │  │  mlflow_    │  │  processor.py   │             │               │    │
│  │  │  utils.py   │  │  chunker.py     │             │               │    │
│  │  │  logger.py  │  │  manifest.py    │             │               │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  DATA (data/)                                                      │    │
│  │  raw/  ──preprocess──►  chunks/  ──train──►  models/registry/    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  OUTPUTS                                                           │    │
│  │  logs/  │  runs/ (TensorBoard)  │  mlflow.db  │  models/registry/ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: All Commands

| Task | Command |
|------|---------|
| **Batch Training** | `python scripts/batch_train.py --auto-config` |
| **Single Training** | `python scripts/train.py --config configs/halfmile.yaml --model mpslight` |
| **Evaluation** | `python scripts/evaluate.py --config configs/halfmile.yaml --model best` |
| **Preprocessing** | `python scripts/preprocess.py --config configs/halfmile.yaml` |
| **Visualization** | `python scripts/visualize.py --config configs/halfmile.yaml --model best` |
| **Model Export** | `python scripts/export_model.py --model model.pt --onnx` |
| **Grid Search** | `python scripts/sweep_mlflow.py --config configs/sweep_config.yaml` |
| **Device Check** | `python scripts/check_device_memory.py` |
| **Model Search** | `python scripts/search_models.py --dataset Halfmile` |
| **Model Pairs** | `python scripts/run_model_pairs.py --epochs 2` |
| **MLflow UI** | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| **Ruff Lint** | `python3.12 -m ruff check . --fix` |
