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