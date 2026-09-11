# CLI Reference

Complete command reference for all scripts in `scripts/`.

---

## Overview

| Script | Purpose |
|---|---|
| `preprocess.py` | Build chunks from raw HDF5 |
| `train.py` | Train a single model on a single dataset |
| `evaluate.py` | Compute metrics + diagnostic images |
| `visualize.py` | Sample shot predictions |
| `compare_baselines.py` | ML vs STA/LTA side-by-side |
| `diagnose.py` | Wavefront coherence + worst-N galleries |
| `explain.py` | Grad-CAM heatmaps |
| `analyze_features.py` | Weight/activation/kernel analysis |
| `export_model.py` | TorchScript + ONNX export |
| `benchmark_inference.py` | Latency measurements |
| `search_models.py` | Query MLflow for best runs |
| `sweep_mlflow.py` | Grid search over datasets × models × losses |
| `batch_train.py` | Multi-dataset orchestration |
| `run_model_pairs.py` | Train models in pairs across datasets |
| `run_pico_all.py` | Quick smoke test across all datasets |
| `verify_*.py` | Regression tests |

---

## Common Conventions

- All scripts accept `--help` for inline documentation.
- All scripts use `--config <yaml>` to specify the dataset config.
- MLflow logging uses a `--phase <tag>` flag for grouping.
- `--dry-run` typically means "compute but don't log to MLflow."
- Scripts assume the venv is active: `source .venv/bin/activate`.

---

## 1. `preprocess.py`

Build chunk files and a manifest from raw HDF5.

```bash
python scripts/preprocess.py --config configs/halfmile.yaml
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | YAML config file |
| `--force`, `-f` | flag | false | Force reprocessing even if chunks exist |
| `--dataset`, `-d` | string | from config | Override dataset name for logging |

**Outputs:**
- `data/chunks/<Dataset>/chunk_*.pt`
- `data/chunks/<Dataset>/manifest.json`

**Duration:** ~5-10 minutes per dataset.

---

## 2. `train.py`

Train a single model on a single dataset.

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --phase baseline-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--model`, `-m` | choice | `unet` | Architecture key |
| `--epochs`, `-e` | int | from config | Override `n_epochs` |
| `--phase` | string | `unset` | MLflow phase tag |
| `--device`, `-d` | string | from config | `cpu`/`cuda`/`mps` |
| `--dataset`, `-ds` | string | from config | Override dataset name |
| `--batch-size`, `-b` | int | from config | Batch size |
| `--cache-size` | int | from config | Chunk cache size |
| `--learning-rate`, `-lr` | float | from config | Learning rate |
| `--lr-scheduler` | choice | from config | `step`/`plateau`/`cosine` |
| `--num-workers`, `-w` | int | from config | DataLoader workers |
| `--loss`, `-l` | choice | `cross_entropy` | `cross_entropy`/`focal`/`dice`/`combo` |
| `--dice-weight` | float | 0.5 | Dice weight for combo loss |
| `--focal-gamma` | float | 2.0 | Gamma for focal/combo |
| `--class-weights`, `-cw` | 3 floats | from config | E.g., `0.05 0.05 0.9` |
| `--checkpoint-every`, `-ce` | int | 5 | Checkpoint interval |
| `--early-stopping`, `-es` | int | 5 | Patience (use 999 to disable) |
| `--preprocess`, `-p` | flag | false | Force re-chunking |
| `--resume`, `-r` | path | none | Checkpoint to resume from |
| `--verbose`, `-v` | flag | false | Enable DEBUG logging |
| `--log-memory`, `-lm` | flag | false | Memory tracking |
| `--log-level`, `-ll` | choice | from config | DEBUG/INFO/WARNING/ERROR |
| `--search-best` | flag | false | Post-training MLflow search |

**Model choices:** `pico`, `nano`, `tiny`, `nano-light`, `mpslight`, `light`, `mobile`, `efficient`, `unet`

**Outputs:**
- Checkpoint: `models/registry/<Model>_<Dataset>_best.pt` (+ timestamped archive)
- MLflow run in `seismic-fbp-training`
- TensorBoard: `runs/<Dataset>/<Model>/`
- Log: `logs/<Dataset>/<model_name>_<date>/`

**Duration:** depends on model + epochs. MPSLight @ 30 ep ≈ 90 min.

---

## 3. `evaluate.py`

Metrics + JSON/CSV + optional diagnostic images.

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split all \
    --detailed \
    --save-images \
    --phase baseline-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--model`, `-m` | path or `best` | required | Checkpoint or MLflow champion |
| `--split`, `-s` | choice | `test` | `train`/`val`/`test`/`all` |
| `--output`, `-o` | path | `evaluation_results` | Output directory |
| `--device`, `-d` | string | `mps` | Device |
| `--batch_size`, `-b` | int | 4 | Batch size |
| `--dataset`, `-ds` | string | from config | Override dataset name |
| `--detailed` | flag | false | Save per-trace CSV |
| `--save-images` | flag | false | Generate 5 diagnostic PNGs |
| `--phase` | string | `unset` | MLflow phase tag |

**Outputs:**
- `evaluation_results/evaluation_results_<Dataset>_<ts>.json`
- `evaluation_results/evaluation_summary_<Dataset>_<ts>.csv`
- `evaluation_results/detailed_errors_<Dataset>_<split>_<ts>.csv` (if `--detailed`)
- `evaluation_results/images_<Dataset>_<ts>/` with 5 PNGs (if `--save-images`)
- MLflow run in `seismic-fbp-evaluation`

**Duration:** ~30 sec to 2 min.

---

## 4. `visualize.py`

Sample shot predictions (3-panel PNGs).

```bash
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 10 \
    --output visualization_results
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--model`, `-m` | path | required | Checkpoint |
| `--n_samples`, `-n` | int | 10 | Shots to visualize |
| `--output`, `-o` | path | `visualization_results` | Output directory |
| `--device`, `-d` | string | `mps` | Device |

**Outputs:** `<output>/shot_<ID>_comparison.png` — one per shot.

**Duration:** ~1 min per 10 shots.

---

## 5. `compare_baselines.py`

ML vs STA/LTA side-by-side comparison.

```bash
python scripts/compare_baselines.py \
    --config configs/halfmile.yaml \
    --ml-model models/registry/MPSLightUNet_Halfmile_best.pt \
    --sta-window 15 --lta-window 150 --threshold 3.0 \
    --shots 5 \
    --phase baseline-comparison-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--ml-model`, `-m` | path | required | ML checkpoint |
| `--dataset`, `-ds` | string | from config | Override dataset |
| `--sta-window` | int | 15 | STA window (samples) |
| `--lta-window` | int | 150 | LTA window |
| `--threshold` | float | 3.0 | STA/LTA ratio threshold |
| `--shots` | int | 5 | Shots to compare |
| `--seed` | int | none | Shuffle seed |
| `--shots-from`, `--shots-to` | int | none | Explicit range |
| `--save-images` | flag | true | Generate images |
| `--no-images` | flag | false | Skip images |
| `--dry-run` | flag | false | No MLflow logging |
| `--phase` | string | `baseline-comparison-v1.0` | MLflow phase |

**Outputs:**
- `evaluation_results/comparison_<Dataset>_<ts>.csv`
- `evaluation_results/comparison_images_<Dataset>_<ts>/` with 6 PNGs:
  - `shot_best.png`, `shot_median.png`, `shot_worst.png`
  - `error_histogram.png`, `error_vs_position.png`, `summary_comparison.png`
- MLflow run in `seismic-fbp-comparison`

**Duration:** ~1 min.

---

## 6. `diagnose.py`

Wavefront coherence + worst-N galleries.

```bash
python scripts/diagnose.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test --n-worst 20 \
    --phase diagnose-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--model`, `-m` | path | required | ML checkpoint |
| `--split`, `-s` | choice | `test` | Which split |
| `--sta-window` | int | 15 | STA window |
| `--lta-window` | int | 150 | LTA window |
| `--threshold` | float | 3.0 | Ratio threshold |
| `--n-worst` | int | 20 | Gallery size |
| `--dry-run` | flag | false | No MLflow |
| `--phase` | string | `diagnose-v1.0` | MLflow phase |

**Outputs:**
- `evaluation_results/diagnostics_<Dataset>_<ts>/`:
  - `coherence_trajectories.png`
  - `coherence_distribution.png`
  - `worst_gallery.png`, `worst_gallery_sta.png`
  - `coherence_metrics.csv`
- MLflow run in `seismic-fbp-explainability`

**Duration:** ~2 min.

---

## 7. `explain.py`

Grad-CAM heatmaps for best/median/worst shots.

```bash
python scripts/explain.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test \
    --phase explain-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | required | Config YAML |
| `--model`, `-m` | path | required | ML checkpoint |
| `--split`, `-s` | choice | `test` | Which split |
| `--method` | choice | `gradcam` | `gradcam`/`hirescam` |
| `--target-class` | int | 2 | Class to explain |
| `--no-mlflow` | flag | false | Skip MLflow |
| `--dry-run` | flag | false | Skip MLflow |
| `--phase` | string | `explain-v1.0` | MLflow phase |

**Outputs:**
- `evaluation_results/explainability_images_<Dataset>_<ts>/` with 3 PNGs
- MLflow run in `seismic-fbp-explainability`

**Duration:** ~30 sec.

---

## 8. `analyze_features.py`

Weight histograms, activation statistics, first-layer filters, kernel similarity.

```bash
python scripts/analyze_features.py \
    --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
    --config configs/halfmile.yaml \
    --split test --n-shots 5 \
    --phase feature-analysis-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--checkpoint`, `-c` | path | required | Model checkpoint |
| `--config`, `-cfg` | path | required | Config YAML |
| `--split`, `-s` | choice | `test` | Split for activations |
| `--n-shots`, `-n` | int | 5 | Shots for activation statistics |
| `--no-mlflow` | flag | false | Skip MLflow |
| `--dry-run` | flag | false | Skip MLflow |
| `--phase` | string | `feature-analysis-v1.0` | MLflow phase |

**Outputs:**
- `evaluation_results/feature_analysis_<Dataset>_<ts>/`:
  - `weight_histograms.png`
  - `activation_statistics.png`
  - `first_layer_filters.png`
  - `kernel_similarity_<layer>.png`
  - `feature_analysis_summary.csv`
- MLflow run in `seismic-fbp-explainability`

**Duration:** ~1 min.

---

## 9. `export_model.py`

Export to TorchScript and ONNX with numeric equivalence verification.

```bash
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --config configs/halfmile.yaml \
    --output exported_models \
    --torchscript --onnx
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--model`, `-m` | path | required | Checkpoint |
| `--model-type`, `-t` | choice | auto-detect | Architecture key |
| `--config`, `-c` | path | none | Config YAML |
| `--output`, `-o` | path | `exported_models` | Output directory |
| `--device`, `-d` | string | `cpu` | Export device |
| `--torchscript` | flag | false | Export TorchScript |
| `--onnx` | flag | false | Export ONNX |
| `--verify/--no-verify` | flag | verify | Run numeric equivalence check |

**Outputs:**
- `<output>/<model-type>_model_scripted.pt`
- `<output>/<model-type>_model.onnx`
- Numeric equivalence output

**Duration:** ~30 sec.

---

## 10. `benchmark_inference.py`

Latency benchmark for PyTorch/TorchScript/ONNX.

```bash
python scripts/benchmark_inference.py \
    --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
    --device cpu \
    --n-iterations 10 \
    --torchscript exported_models/mpslight_model_scripted.pt \
    --onnx exported_models/mpslight_model.onnx
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--checkpoint`, `-c` | path | required | Model checkpoint |
| `--device`, `-d` | string | `cpu` | Device |
| `--n-iterations`, `-n` | int | 20 | Timing iterations |
| `--torchscript` | path | none | Optional TorchScript to compare |
| `--onnx` | path | none | Optional ONNX to compare |

**Outputs:** Console table with load time, median/min/p90 inference.

**Duration:** ~20 sec to 2 min (depends on iterations).

---

## 11. `search_models.py`

Query MLflow for models matching criteria.

```bash
python scripts/search_models.py --dataset Halfmile --top 10
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--dataset`, `-d` | string | none | Filter by dataset |
| `--model-type`, `-m` | string | none | Filter by architecture |
| `--min-iou` | float | none | Minimum val IoU |
| `--top`, `-n` | int | 10 | Number of results |
| `--compare`, `-c` | flag | false | Side-by-side comparison view |

**Duration:** seconds.

---

## 12. `sweep_mlflow.py`

Grid search over datasets × models × losses.

```bash
python scripts/sweep_mlflow.py --config configs/sweep_config.yaml
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | `configs/sweep_config.yaml` | Sweep config |

**Outputs:** one MLflow run per combination in `seismic-fbp-sweeps`.

**Duration:** hours.

---

## 13. `batch_train.py`

Multi-dataset/multi-model orchestration.

```bash
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --datasets Halfmile \
    --auto-config \
    --epochs 30 \
    --phase baseline-v1.0
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--config`, `-c` | path | `configs/batch_config.yaml` | Config |
| `--datasets`, `-d` | multiple | all | Datasets to train |
| `--list-datasets` | flag | false | List available datasets |
| `--auto-config`, `-a` | flag | false | Memory-aware config |
| `--manual-config`, `-m` | flag | false | Use YAML config |
| `--batch-size`, `-b` | int | none | Override |
| `--cache-size` | int | none | Override |
| `--memory-limit`, `-ml` | float | none | Override (GB) |
| `--epochs`, `-e` | int | none | Override |
| `--device`, `-dev` | string | none | Override |
| `--log-memory`, `-lm` | flag | false | Memory logging |
| `--verbose`, `-v` | flag | false | Verbose |
| `--log-level`, `-ll` | string | none | Override |
| `--preprocess`, `-p` | flag | false | Force preprocessing |
| `--concurrent` | flag | false | Parallel datasets |
| `--max-workers` | int | 2 | Parallelism (1-8) |

**Outputs:**
- MLflow runs per (dataset, model) pair
- `logs/batch/batch_summary_<ts>.json`
- Failed logs in `logs/batch/failed/`

**Duration:** hours to days.

---

## 14. `run_model_pairs.py`

Train models in pairs across datasets.

```bash
python scripts/run_model_pairs.py
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--epochs`, `-e` | int | 2 | Epochs per model |
| `--device`, `-d` | string | `mps` | Device |
| `--dry-run` | flag | false | Print commands only |
| `--verbose`, `-v` | flag | false | Verbose |
| `--no-log-memory` | flag | false | Disable memory logging |

**Pairs:** `[pico, nano]`, `[tiny, mpslight]`, `[light, mobile]`, `[efficient, unet]`

**Duration:** hours.

---

## 15. `run_pico_all.py`

Quick smoke test — PicoUNet on all datasets.

```bash
python scripts/run_pico_all.py
python scripts/run_pico_all.py --non-interactive
```

**Options:**

| Flag | Type | Default | Purpose |
|---|---|---|---|
| `--non-interactive` | flag | false | No pause between datasets |

**Duration:** ~30 sec per dataset.

---

## 16. Verify Scripts

Regression tests. Each is standalone.

```bash
python scripts/verify_tracking_conventions.py
python scripts/verify_mlflow_routing.py
python scripts/verify_baseline_run.py
python scripts/verify_baseline_run.py --phase baseline-v1.0-30ep
python scripts/verify_evaluation_images.py --split test --mlflow-latest
python scripts/verify_sta_lta_evaluation.py --mlflow-latest
python scripts/verify_baseline_comparison.py --mlflow-latest
python scripts/verify_extract_picks.py
python scripts/verify_gradcam.py
python scripts/verify_diagnose.py
python scripts/verify_deployment.py
python scripts/verify_feature_analysis.py
```

**Run all:**

```bash
for s in scripts/verify_*.py; do
    echo "=== $s ==="
    python "$s" || echo "❌ FAILED"
done
```

**Duration:** ~3 min total.

---

## Appendix A — Full Pipeline Example (Single Model)

Run these in order for a complete training → deployment cycle.

```bash
# 1. Preprocess (once per dataset)
python scripts/preprocess.py --config configs/halfmile.yaml

# 2. Train
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --phase baseline-v1.0

# 3. Evaluate
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split all --detailed --save-images \
    --phase baseline-v1.0

# 4. Visualize
python scripts/visualize.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --n_samples 10

# 5. Compare with STA/LTA
python scripts/compare_baselines.py \
    --config configs/halfmile.yaml \
    --ml-model models/registry/MPSLightUNet_Halfmile_best.pt \
    --shots 5 \
    --phase baseline-comparison-v1.0

# 6. Diagnose
python scripts/diagnose.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test --n-worst 20 \
    --phase diagnose-v1.0

# 7. Explain
python scripts/explain.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test \
    --phase explain-v1.0

# 8. Feature analysis
python scripts/analyze_features.py \
    --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
    --config configs/halfmile.yaml \
    --phase feature-analysis-v1.0

# 9. Export
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight \
    --config configs/halfmile.yaml \
    --torchscript --onnx

# 10. Benchmark
python scripts/benchmark_inference.py \
    --checkpoint models/registry/MPSLightUNet_Halfmile_best.pt \
    --device cpu \
    --n-iterations 10

# 11. Search
python scripts/search_models.py --dataset Halfmile --top 10

# 12. Verify
python scripts/verify_baseline_run.py --phase baseline-v1.0
```

---

## Appendix B — Multi-Model Sweep

Train and evaluate all models on a dataset.

**Train all models (batch orchestrator):**

```bash
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --datasets Halfmile \
    --auto-config \
    --epochs 30 \
    --phase all-models-v1.0
```

**Or via a shell loop (explicit control):**

```bash
for MODEL in pico nano tiny mpslight light mobile efficient unet; do
    python scripts/train.py \
        --config configs/halfmile.yaml \
        --model "$MODEL" \
        --epochs 30 \
        --phase "sweep-$MODEL"
done
```

**Then evaluate each trained model:**

```bash
for ENTRY in \
    "pico|models/registry/PicoUNet_Halfmile_best.pt" \
    "nano|models/registry/NanoUNet_Halfmile_best.pt" \
    "tiny|models/registry/TinyUNet_Halfmile_best.pt" \
    "mpslight|models/registry/MPSLightUNet_Halfmile_best.pt" \
    "light|models/registry/LightUNet_Halfmile_best.pt" \
    "mobile|models/registry/MobileUNet_Halfmile_best.pt" \
    "efficient|models/registry/EfficientUNet_Halfmile_best.pt" \
    "unet|models/registry/UNet_Halfmile_best.pt"
do
    MODEL="${ENTRY%%|*}"
    CKPT="${ENTRY##*|}"
    [ -f "$CKPT" ] || { echo "⚠️  Missing $CKPT"; continue; }

    python scripts/evaluate.py \
        --config configs/halfmile.yaml \
        --model "$CKPT" \
        --split test --detailed --save-images \
        --phase "sweep-$MODEL"

    python scripts/compare_baselines.py \
        --config configs/halfmile.yaml \
        --ml-model "$CKPT" \
        --shots 5 \
        --phase "sweep-$MODEL"
done
```

---

## Appendix C — Common Recipes

**Fast smoke test:**
```bash
python scripts/train.py --config configs/halfmile.yaml --model pico --epochs 3 --phase smoke-test
```

**Full baseline with images:**
```bash
python scripts/train.py --config configs/halfmile.yaml --model mpslight --epochs 30 --phase baseline-v1.0
python scripts/evaluate.py --config configs/halfmile.yaml --model models/registry/MPSLightUNet_Halfmile_best.pt --split all --detailed --save-images --phase baseline-v1.0
```

**Compare ML vs classical:**
```bash
python scripts/compare_baselines.py \
    --config configs/halfmile.yaml \
    --ml-model models/registry/MPSLightUNet_Halfmile_best.pt \
    --sta-window 15 --lta-window 150 --threshold 3.0 \
    --shots 5 --phase comparison-v1.0
```

**Explain a model:**
```bash
python scripts/explain.py --config configs/halfmile.yaml --model models/registry/MPSLightUNet_Halfmile_best.pt --phase explain-v1.0
```

**Export for production:**
```bash
python scripts/export_model.py \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --model-type mpslight --torchscript --onnx \
    --output exported_models
```

**Query MLflow for best runs:**
```bash
python scripts/search_models.py --dataset Halfmile --min-iou 0.7 --top 5
```

**Cross-dataset sweep:**
```bash
python scripts/batch_train.py \
    --config configs/batch_config.yaml \
    --datasets Halfmile Brunswick Lalor Sudbury \
    --auto-config --epochs 30 \
    --phase cross-dataset-v1.0
```

---

## Appendix D — Output Directory Reference

| Directory | Contents |
|---|---|
| `models/registry/` | Trained checkpoints |
| `data/chunks/` | Preprocessed chunk files + manifests |
| `data/raw/` | Original HDF5 files |
| `runs/` | TensorBoard logs |
| `logs/` | Text logs (date-organized) |
| `logs/batch/` | Batch training summaries + failed logs |
| `evaluation_results/` | All evaluation outputs |
| `evaluation_results/images_<Dataset>_<ts>/` | Evaluation diagnostic images |
| `evaluation_results/comparison_images_<Dataset>_<ts>/` | ML vs STA/LTA comparison |
| `evaluation_results/diagnostics_<Dataset>_<ts>/` | Coherence + galleries |
| `evaluation_results/explainability_images_<Dataset>_<ts>/` | Grad-CAM heatmaps |
| `evaluation_results/feature_analysis_<Dataset>_<ts>/` | Weight/activation analysis |
| `evaluation_results/sta_lta_sweep/` | STA/LTA sweep JSON |
| `exported_models/` | TorchScript + ONNX |
| `docs/` | Model cards + ADRs |
| `mlflow.db` | MLflow tracking database |

---

## Appendix E — MLflow Experiments

| Experiment | Purpose |
|---|---|
| `seismic-fbp-training` | Training runs |
| `seismic-fbp-evaluation` | Evaluation runs |
| `seismic-fbp-baselines` | Classical (STA/LTA) baselines |
| `seismic-fbp-sweeps` | Hyperparameter sweeps |
| `seismic-fbp-comparison` | ML vs STA/LTA comparisons |
| `seismic-fbp-explainability` | Grad-CAM, diagnostics, feature analysis |

Query examples:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# All baseline runs for Halfmile
mlflow.search_runs(
    experiment_ids=[...],
    filter_string="tags.dataset = 'Halfmile' AND tags.phase LIKE 'baseline%'",
    order_by=["metrics.val_iou DESC"],
)

# All training runs for MPSLight
mlflow.search_runs(
    experiment_ids=[...],
    filter_string="tags.model_type = 'MPSLightUNet'",
)

# The single baseline-v1.0 training run
mlflow.search_runs(
    experiment_ids=[...],
    filter_string="tags.phase = 'baseline-v1.0'",
)
```

## Commands to Save the Documentation

Save the content above to `docs/CLI_REFERENCE.md`:

```bash
# Make sure docs/ exists
mkdir -p docs

# Then save the content
```

**And commit:**

```bash
git add docs/CLI_REFERENCE.md
git commit -m "docs: complete CLI reference for all scripts

Comprehensive reference covering:
  - All 15+ scripts in scripts/
  - Every flag for every command
  - Input/output specs
  - Duration estimates
  - Appendices: full pipeline, multi-model sweep, common recipes,
    output directory reference, MLflow experiments"
```