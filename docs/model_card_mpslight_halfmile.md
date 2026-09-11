# Model Card — MPSLightUNet on Halfmile

**Version:** 1.0.0
**Date:** 2026-09-11
**Author:** [Your name]
**Model type:** MPSLightUNet
**MLflow run:** `548d08c930c644a88105b122a07c1f5f` (5-epoch baseline)

---

## Overview

**What this model does:** Predicts first-break picks on 2D seismic shot gathers from the Halfmile dataset by segmenting each gather into three classes (before, strip, after the first break), then extracting a per-trace pick from the strip.

**Intended use:** Automated first-break picking to accelerate seismic processing workflows where picks will be reviewed or refined downstream.

**Out-of-scope use:** Direct production use without human verification. Deployment to seismic data with characteristics outside the training distribution.

---

## Training Data

| Field | Value |
|---|---|
| Dataset | Halfmile |
| Source | `data/raw/Halfmile3D_add_geom_sorted.hdf5` |
| Number of shots | 690 |
| Traces per shot | 1578 |
| Samples per trace | 751 |
| Sampling interval | 2.0 ms |
| Split | 80 / 10 / 10 (train / val / test) |
| Label source | Human picks in HDF5 SPARE1 field |
| Label quality | Reported as consistent; ~89.5% of traces have valid picks |

---

## Architecture

| Field | Value |
|---|---|
| Model | MPSLightUNet |
| Parameters | 1,943,795 |
| Input shape | (1, 1, 1578, 751) |
| Output shape | (1, 3, 1578, 751) |
| Loss | Cross-entropy (class weights 0.05, 0.05, 0.9) |
| Optimizer | Adam, lr=1e-3 |
| Epochs trained | 5 (smoke-test quality) |
| Batch size | 4 |
| Device | Apple Silicon MPS |

---

## Performance

**Test set metrics:**

| Metric | Value |
|---|---|
| Segmentation IoU | 0.7904 |
| Segmentation F1 | 0.8737 |
| Pick MAE | 20.19 samples |
| Pick median error | 1.00 samples |
| Pick ±3 accuracy | 74.96% |

**Baseline comparison (first 5 shots of the dataset):**

| Metric | MPSLightUNet | STA/LTA | Delta |
|---|---|---|---|
| MAE | 20.95 | 38.34 | −17.39 |
| Median | 1.00 | 3.00 | −2.00 |
| ±3 accuracy | 79.8% | 51.9% | +27.9 pp |

**Error distribution (test split):**

| Percentile | Error (samples) |
|---|---|
| 50th | 1.00 |
| 75th | 4.00 |
| 90th | 80.00 |
| 95th | 121.00 |
| 99th | 214.00 |

---

## Known Limitations

1. **Heavy-tailed errors.** Median 1 sample, but ~20% of traces have >10 sample errors. Model either nails the pick or misses badly.

2. **Halfmile-specific.** Trained on 690 shots from one survey. Generalization to other datasets untested.

3. **No uncertainty.** Deterministic output.

4. **No temporal context.** Each shot independently.

5. **Smoke-test quality.** Only 5 epochs. Longer training would likely improve metrics.

---

## Deployment

**Inference:**

```python
from src.deployment import Predictor

predictor = Predictor.load(
    "models/registry/MPSLightUNet_Halfmile_best.pt",
    device="cpu",
)
picks = predictor.predict(shot_data)
```