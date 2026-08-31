# Batch Training Pipeline Documentation

## Overview

The batch training pipeline is a robust, production-ready system for training multiple seismic datasets sequentially with automatic memory error recovery and fallback configurations. It's designed to handle failures gracefully without stopping the entire training process.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BATCH TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   1. CONFIGURATION LOADER                           │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  batch_config.yaml                                           │   │    │
│  │  │  ├── global:                                                  │   │    │
│  │  │  │   ├── epochs: 30                                         │   │    │
│  │  │  │   ├── device: mps                                        │   │    │
│  │  │  │   └── timeout_seconds: 7200                              │   │    │
│  │  │  ├── datasets:                                               │   │    │
│  │  │  │   ├── Halfmile: {epochs: 40}                             │   │    │
│  │  │  │   └── Lalor: {batch_size_override: 2}                    │   │    │
│  │  │  └── variants:                                               │   │    │
│  │  │      ├── Attempt 1: {batch_size: 4, model: mpslight}        │   │    │
│  │  │      ├── Attempt 2: {batch_size: 2, model: mpslight}        │   │    │
│  │  │      └── Attempt 3: {batch_size: 1, model: tiny}            │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   2. TRAINING ORCHESTRATOR                          │    │
│  │                                                                     │    │
│  │  For each dataset:                                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Dataset 1: Brunswick                                       │    │    │
│  │  │  ┌───────────────────────────────────────────────────────┐ │    │    │
│  │  │  │  Attempt 1: batch_size=4, model=mpslight              │ │    │    │
│  │  │  │  └─> Success ✅ → Save model, move to next dataset   │ │    │    │
│  │  │  │  Attempt 2: batch_size=2, model=mpslight              │ │    │    │
│  │  │  │  └─> Memory Error ❌ → Try next variant              │ │    │    │
│  │  │  │  Attempt 3: batch_size=1, model=tiny                 │ │    │    │
│  │  │  │  └─> Success ✅ → Save model, move to next dataset   │ │    │    │
│  │  │  └───────────────────────────────────────────────────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Dataset 2: Halfmile                                        │    │    │
│  │  │  └─> (same retry logic)                                    │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   3. SUMMARY & REPORTING                           │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ✅ Successful: 3/4 datasets                                 │   │    │
│  │  │  ❌ Failed: 1/4 datasets                                      │   │    │
│  │  │  ⏱ Total time: 15.2 minutes                                  │   │    │
│  │  │  📁 Summary saved to: logs/batch/batch_summary_*.json        │   │    │
│  │  │  📧 Optional: Email/Slack notifications                      │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files Structure

```
first_break_pick/
├── configs/
│   └── batch_config.yaml          # Main batch configuration file
├── scripts/
│   ├── batch_train.py             # Batch training orchestrator
│   ├── train.py                   # Individual training script
│   └── evaluate.py                # Evaluation script
├── logs/
│   └── batch/                     # Batch training logs
│       ├── batch_summary_*.json   # Training summary
│       └── batch_train_*.log      # Detailed logs
└── models/
    └── registry/                  # Saved model checkpoints
```

## Configuration File: `batch_config.yaml`

### Section 1: Global Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `epochs` | Integer | 30 | Number of training epochs |
| `device` | String | "mps" | Device: "cpu", "cuda", or "mps" |
| `log_memory` | Boolean | false | Enable memory logging |
| `verbose` | Boolean | false | Enable verbose logging |
| `log_level` | String | "INFO" | Log level: DEBUG/INFO/WARNING/ERROR/CRITICAL |
| `preprocess` | Boolean | false | Force preprocessing |
| `checkpoint_every` | Integer | 5 | Save checkpoint every N epochs |
| `early_stopping` | Integer | 5 | Early stopping patience |
| `timeout_seconds` | Integer | 7200 | Timeout per dataset (2 hours) |
| `skip_failed` | Boolean | true | Continue on failure |
| `max_retries` | Integer | 3 | Retries per config variant |
| `clear_memory_between_datasets` | Boolean | true | Clear memory between datasets |
| `pause_between_datasets` | Integer | 2 | Pause seconds between datasets |

### Section 2: Dataset Overrides

Each dataset can override global settings:

| Parameter | Type | Description |
|-----------|------|-------------|
| `epochs` | Integer | Override epochs for specific dataset |
| `device` | String | Override device |
| `log_memory` | Boolean | Override memory logging |
| `verbose` | Boolean | Override verbose logging |
| `log_level` | String | Override log level |
| `preprocess` | Boolean | Override preprocessing |
| `checkpoint_every` | Integer | Override checkpoint frequency |
| `early_stopping` | Integer | Override early stopping |
| `timeout_seconds` | Integer | Override timeout |
| `batch_size_override` | Integer | Force batch size for dataset |
| `model_override` | String | Force model type |

### Section 3: Fallback Variants

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch_size` | Integer | Batch size for training |
| `model` | String | Model architecture |
| `cache_size` | Integer | Cache size for data loading |
| `class_weights` | String | Class weights (e.g., "0.1,0.1,0.8") |
| `strip_width` | Integer | Strip width for first break |
| `memory_limit_gb` | Float | Memory limit in GB |

## CLI Commands

### Basic Usage

```bash
# Train all datasets with default config
python scripts/batch_train.py

# Use custom config file
python scripts/batch_train.py --config configs/my_config.yaml

# Train specific datasets
python scripts/batch_train.py --datasets Halfmile --datasets Brunswick

# List available datasets
python scripts/batch_train.py --list-datasets
```

### Override Options

```bash
# Override training parameters
python scripts/batch_train.py --epochs 50 --device cpu --log-memory

# Enable verbose debugging
python scripts/batch_train.py --verbose --log-level DEBUG

# Force preprocessing
python scripts/batch_train.py --preprocess

# Combine multiple options
python scripts/batch_train.py \
    --datasets Halfmile \
    --datasets Brunswick \
    --epochs 40 \
    --device mps \
    --log-memory \
    --verbose
```

### Complete Parameter Reference

| Parameter | Short | Type | Default | Description |
|-----------|-------|------|---------|-------------|
| `--config` | `-c` | String | `configs/batch_config.yaml` | Config file path |
| `--datasets` | `-d` | Multiple | All datasets | Datasets to train |
| `--list-datasets` | - | Flag | False | List available datasets |
| `--epochs` | `-e` | Integer | None | Override epochs |
| `--device` | `-dev` | String | None | Override device |
| `--log-memory` | `-lm` | Flag | False | Enable memory logging |
| `--verbose` | `-v` | Flag | False | Enable verbose logging |
| `--log-level` | `-ll` | String | None | Override log level |
| `--preprocess` | `-p` | Flag | False | Force preprocessing |
| `--timeout` | `-t` | Integer | None | Override timeout |

## Memory Error Recovery

The pipeline automatically detects and handles memory errors:

### Detection
```
Memory Error Keywords:
- "out of memory"
- "MPS"
- "CUDA out of memory"
- "cannot allocate"
- "memory exhausted"
- "OOM"
```

### Recovery Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 MEMORY ERROR RECOVERY FLOW                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Start Training                                             │
│       │                                                     │
│       ▼                                                     │
│  Attempt Variant 1 (batch_size=4, model=mpslight)          │
│       │                                                     │
│       ▼                                                     │
│  Memory Error? ──Yes──► Clear Memory                       │
│       │                    │                                │
│       No                   ▼                                │
│       │              Attempt Variant 2 (batch_size=2)       │
│       │                    │                                │
│       │                    ▼                                │
│       │              Memory Error? ──Yes──► Clear Memory   │
│       │                    │                    │           │
│       │                    No                   ▼           │
│       │                    │              Attempt Variant 3 │
│       │                    ▼              (batch_size=1)    │
│       │              Success!                    │           │
│       │                    │                    ▼           │
│       │                    │              ...              │
│       │                    │                    │           │
│       │                    ▼                    ▼           │
│       │              ✅ Save Model      ❌ Skip Dataset    │
│       │                    │                    │           │
│       ▼                    ▼                    ▼           │
│  Next Dataset        Next Dataset        Next Dataset      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Output Files

### Summary JSON
```json
{
  "timestamp": "20260831_194748",
  "config_file": "configs/batch_config.yaml",
  "global_config": {...},
  "total_datasets": 4,
  "successful_datasets": ["Halfmile", "Brunswick"],
  "failed_datasets": ["Lalor", "Sudbury"],
  "total_duration_seconds": 912.0,
  "results": {
    "Halfmile": {
      "success": true,
      "best_config": {"batch_size": 2, "model": "mpslight"},
      "duration": 245.3
    },
    "Brunswick": {
      "success": true,
      "best_config": {"batch_size": 4, "model": "mpslight"},
      "duration": 312.7
    }
  },
  "errors": [
    {"dataset": "Lalor", "error": "Memory error..."}
  ]
}
```

### Log Files
```
logs/batch/
├── batch_summary_20260831_194748.json
├── batch_train_20260831_194748.log
└── batch_train_20260831_194748_errors.log
```

## Notifications

### Email Notifications
```yaml
monitoring:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    sender: "your_email@gmail.com"
    recipient: "team@company.com"
    password_env_var: "EMAIL_PASSWORD"
```

### Slack Notifications
```yaml
monitoring:
  slack:
    enabled: true
    webhook_url_env_var: "SLACK_WEBHOOK_URL"
    channel: "#ml-training"
```

## Datasets

| Dataset | Config File | Description |
|---------|-------------|-------------|
| Brunswick | `configs/brunswick.yaml` | 2582 traces, 751 samples |
| Halfmile | `configs/halfmile.yaml` | 1578 traces, 751 samples |
| Lalor | `configs/lalor.yaml` | 2685 traces, 1501 samples |
| Sudbury | `configs/sudbury.yaml` | 1138 traces, 1001 samples |

## Model Architectures

| Model | Parameters | Description |
|-------|-----------|-------------|
| `mpslight` | ~1.7M | MPS-optimized lightweight U-Net |
| `light` | ~2.5M | Lightweight U-Net |
| `tiny` | ~50K | Tiny U-Net for quick testing |
| `nano` | ~10K | Nano U-Net for testing |
| `pico` | ~2K | Pico U-Net (last resort) |
| `unet` | ~31M | Full U-Net |
| `efficient` | ~5M | EfficientNet + U-Net |
| `mobile` | ~3.5M | MobileNet + U-Net |

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No such option '--checkpoint-every'` | Missing option in train.py | Add `--checkpoint-every` to train.py |
| `Memory error` | Insufficient RAM | Use smaller batch size or model |
| `Timeout` | Training takes too long | Increase `timeout_seconds` |
| `HDF5 validation failed` | Corrupt or missing data | Check dataset path |

### Quick Fixes

```bash
# Clear memory between runs
python -c "import torch; torch.mps.empty_cache()"

# Run with minimal configuration
python scripts/batch_train.py --datasets Halfmile --batch-size 1

# Increase timeout for large datasets
python scripts/batch_train.py --timeout 10800
```

## Best Practices

1. **Start with small datasets** (Halfmile, Brunswick) before large ones (Lalor)
2. **Monitor memory usage** with `--log-memory`
3. **Use verbose mode** for debugging: `--verbose --log-level DEBUG`
4. **Keep the summary JSON** for tracking performance over time
5. **Set appropriate timeouts** based on dataset size
6. **Use dataset-specific overrides** for different requirements

## Example Workflow

```bash
# 1. Test configuration on one dataset
python scripts/batch_train.py --datasets Halfmile --epochs 5 --verbose

# 2. Train all datasets with basic config
python scripts/batch_train.py --epochs 30

# 3. Train with custom settings
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --epochs 50 \
    --log-memory \
    --verbose

# 4. Check results
cat logs/batch/batch_summary_*.json

# 5. Evaluate best models
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model best \
    --split test
```

## Summary

The batch training pipeline provides:

1. ✅ **Sequential training** of multiple datasets
2. ✅ **Automatic memory error recovery** with fallback configurations
3. ✅ **Graceful failure handling** - one dataset failing doesn't stop others
4. ✅ **Comprehensive logging** - JSON summaries, detailed logs
5. ✅ **Flexible configuration** - YAML config with CLI overrides
6. ✅ **Notification support** - Email/Slack alerts
7. ✅ **Memory monitoring** - Track usage and prevent crashes

This pipeline is production-ready and can handle the entire training workflow with minimal manual intervention.