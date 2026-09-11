# Seismic First-Break Picking

Deep learning pipeline for automated first-break picking on 2D seismic shot gathers.

**Status:** Research prototype. Complete training, evaluation, explainability, and deployment pipeline. Trained models beat the classical STA/LTA baseline on all metrics.

---

## What This Project Does

Given a seismic shot gather (a 2D array of traces vs. time samples), the model predicts the first-break time for each trace — the moment energy from a seismic source first arrives.

The model frames this as **3-class segmentation**:
- **Class 0:** samples before the first break
- **Class 1:** samples after the first break
- **Class 2:** a strip of ±4 samples around the first break

A pick is extracted as the center of the class-2 strip.

---

## Quick Start

### Install

```bash
git clone <repo>
cd first_break_pick
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare Data

```bash
python scripts/preprocess.py --config configs/halfmile.yaml
```

### Train

```bash
python scripts/train.py \
    --config configs/halfmile.yaml \
    --model mpslight \
    --epochs 30 \
    --phase baseline-v1.0
```

### Evaluate

```bash
python scripts/evaluate.py \
    --config configs/halfmile.yaml \
    --model models/registry/MPSLightUNet_Halfmile_best.pt \
    --split test --detailed --save-images
```

### Predict (Deployment)

```python
from src.deployment import Predictor

predictor = Predictor.load("models/registry/MPSLightUNet_Halfmile_best.pt")
picks = predictor.predict(shot_data)  # (n_traces,) int64
```

---

## Results

### ML Baseline vs STA/LTA (Halfmile)

| Metric | STA/LTA | MPSLightUNet | Winner |
|---|---|---|---|
| MAE (samples) | 38.34 | 20.19 | **ML** |
| Median (samples) | 3.00 | 1.00 | **ML** |
| ±3 accuracy | 51.9% | 74.96% | **ML** |

### Comparison Figures

- `scripts/compare_baselines.py` produces 6 side-by-side figures
- `scripts/diagnose.py` produces wavefront coherence and worst-N galleries
- `scripts/explain.py` produces Grad-CAM heatmaps

---

## Project Structure

```
src/
├── config.py              # SeismicConfig
├── preprocessing/         # HDF5 → chunks
├── data/                  # Chunked datasets, sampler, cache
├── models/                # 9 U-Net variants
├── training/              # Trainer, losses, metrics
├── evaluation/            # Evaluation runner, images, comparison
├── baselines/             # STA/LTA picker, evaluator
├── explainability/        # Grad-CAM, coherence, error galleries
├── deployment/            # Predictor, exporters, validators
├── batch/                 # Multi-dataset orchestration
└── utils/                 # MLflow, logging, memory, device

scripts/
├── preprocess.py
├── train.py
├── evaluate.py
├── compare_baselines.py
├── diagnose.py
├── explain.py
├── export_model.py
└── verify_*.py            # Regression tests

configs/                   # Dataset-specific YAML configs
data/                      # Raw HDF5 + chunks
models/registry/           # Trained checkpoints
evaluation_results/        # Metrics, CSVs, images
docs/                      # Model cards, design docs
```

---

## Methods

**Model zoo:** 9 U-Net variants (pico, nano, tiny, mpslight, light, mobile, efficient, unet, nano-light). Ranging from 2K to 31M parameters.

**Training:**
- Chunk-aware sampler minimizes cache churn
- LR scheduler: plateau / step / cosine
- Early stopping on val IoU
- Checkpoints, MLflow tracking, TensorBoard

**Evaluation:**
- Per-trace absolute error, median, percentiles
- Segmentation IoU (per-class and mean)
- Best/median/worst shot visualizations
- Error histograms and per-position scatter

**Baselines:**
- STA/LTA with parameter sweep (125 configs)
- Classical baseline established as reference

**Explainability:**
- Grad-CAM heatmaps showing which input regions drive the prediction
- Wavefront coherence analysis (are picks physically plausible?)
- Worst-N error galleries

**Deployment:**
- Single-call inference: `Predictor.load(...).predict(shot)`
- TorchScript and ONNX export with numeric equivalence verification
- InputContract documents the model's expected input

---

## Design Decisions

See `docs/decisions/` for the Architecture Decision Log.

---

## Limitations

- Heavy-tailed error distribution: median 1 sample, but ~20% of traces have >10-sample errors
- Trained on Halfmile; cross-dataset generalization untested
- Deterministic (no uncertainty estimate)
- No temporal context between shots

See `docs/model_cards/` for detailed model cards.

---

## Requirements

- Python 3.11+
- PyTorch 2.0+
- See `requirements.txt` for the full list

Tested on Apple Silicon (MPS). CUDA and CPU should also work.

---

## License

[CilginDonerci@sehran9mkr]

---

## Citation

```
@Bayram_Bayramov
```
