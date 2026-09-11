# Model Card — [MODEL NAME]

**Version:** [1.0.0]
**Date:** [YYYY-MM-DD]
**Author:** [Author]
**Model type:** [e.g., MPSLightUNet]
**MLflow run:** [run_id or URI]

---

## Overview

**What this model does:** Predicts first-break picks on 2D seismic shot gathers by segmenting each gather into three classes (before, strip, after the first break), then extracting a per-trace pick from the strip.

**Intended use:** Automated first-break picking to accelerate seismic processing workflows where the picks will be reviewed or refined downstream.

**Out-of-scope use:** Direct production use without human verification. Deployment to seismic data with characteristics outside the training distribution. Any application requiring regulatory compliance certification.

---

## Training Data

| Field | Value |
|---|---|
| Dataset | [e.g., Halfmile] |
| Source | [HDF5 file path or description] |
| Number of shots | [e.g., 690] |
| Traces per shot | [e.g., 1578] |
| Samples per trace | [e.g., 751] |
| Sampling interval | [e.g., 2.0 ms] |
| Split | [e.g., 80/10/10 train/val/test] |
| Label source | [e.g., human-picked SPARE1 from HDF5] |
| Label quality | [Notes on inter-picker variability if known] |

---

## Architecture

| Field | Value |
|---|---|
| Model | [e.g., MPSLightUNet] |
| Parameters | [e.g., 1,943,795] |
| Input shape | (1, 1, 1578, 751) |
| Output shape | (1, 3, 1578, 751) |
| Loss | [e.g., Combo: CE + Focal + Dice] |
| Optimizer | [e.g., Adam, lr=1e-3] |
| Epochs trained | [e.g., 5] |
| Batch size | [e.g., 4] |
| Device | [e.g., Apple Silicon MPS] |

---

## Performance

**Test set metrics:**

| Metric | Value |
|---|---|
| Segmentation IoU | [e.g., 0.7904] |
| Segmentation F1 | [e.g., 0.8737] |
| Pick MAE | [e.g., 20.19 samples] |
| Pick median error | [e.g., 1.00 samples] |
| Pick ±3 accuracy | [e.g., 74.96%] |

**Baseline comparison:**

| Metric | This model | STA/LTA | Delta |
|---|---|---|---|
| MAE | [e.g., 20.19] | [e.g., 38.34] | [e.g., −18.15] |
| Median | [e.g., 1.00] | [e.g., 3.00] | [e.g., −2.00] |
| ±3 accuracy | [e.g., 74.96%] | [e.g., 51.9%] | [e.g., +23.1 pp] |

**Error distribution:**

| Percentile | Error (samples) |
|---|---|
| 50th | [e.g., 1.00] |
| 75th | [e.g., 4.00] |
| 90th | [e.g., 80.00] |
| 95th | [e.g., 121.00] |
| 99th | [e.g., 214.00] |

---

## Known Limitations

1. **Heavy-tailed error distribution.** While the median error is 1 sample, ~20% of traces have errors >10 samples and ~5% have errors >100 samples. The model either nails a pick or misses badly; it rarely makes small errors.

2. **Domain-specific.** Trained on [dataset]. May not generalize to other acquisition geometries, sampling rates, or geological settings without retraining.

3. **No uncertainty estimate.** The model is deterministic. It does not report confidence.

4. **Requires preprocessing.** Input must be raw amplitudes shaped (1, 1, target_traces, n_samples) with zero-padding if fewer traces are available. See `src/deployment/contract.py`.

5. **No temporal context.** Each shot is processed independently; adjacent shots are not used.

---

## Ethical Considerations

**Environmental impact:** Training used Apple Silicon MPS. Estimated energy: [kWh] over [hours]. Approximate CO₂: [kg].

**Bias and fairness:** Not applicable — the model operates on physical measurements, not on humans or protected classes.

**Data provenance:** Training data is [proprietary/licensed/public]. Distribution of the trained model is subject to [license terms].

---

## Deployment

**Inference path:**

```python
from src.deployment import Predictor

predictor = Predictor.load("path/to/checkpoint.pt", device="cpu")
picks = predictor.predict(shot_data)  # (n_traces,) int64