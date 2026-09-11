# file location: src/utils/tracking_conventions.py

"""
Centralized MLflow naming and tagging conventions.

Single source of truth for experiment names and run tag keys, so every
script uses the same vocabulary. Provides helper functions that return
complete tag dicts, ready to pass to MLflow.

Usage:
    from src.utils.tracking_conventions import (
        EXPERIMENT_TRAINING,
        training_tags,
    )

    tags = training_tags(dataset="Halfmile", model_type="MPSLightUNet")
    mlflow.set_tags(tags)
"""

from typing import Any

# ============================================================
# EXPERIMENT NAMES
# ============================================================

EXPERIMENT_TRAINING = "seismic-fbp-training"
EXPERIMENT_EVALUATION = "seismic-fbp-evaluation"
EXPERIMENT_BASELINES = "seismic-fbp-baselines"
EXPERIMENT_SWEEPS = "seismic-fbp-sweeps"
EXPERIMENT_COMPARISON = "seismic-fbp-comparison"
EXPERIMENT_EXPLAINABILITY = "seismic-fbp-explainability"

ALL_EXPERIMENTS = (
    EXPERIMENT_TRAINING,
    EXPERIMENT_EVALUATION,
    EXPERIMENT_BASELINES,
    EXPERIMENT_SWEEPS,
    EXPERIMENT_COMPARISON,
    EXPERIMENT_EXPLAINABILITY,
)


# ============================================================
# TAG KEYS
# ============================================================

TAG_DATASET = "dataset"
TAG_MODEL_TYPE = "model_type"
TAG_PHASE = "phase"
TAG_ENV = "env"

# The four canonical tag keys every helper aims to produce.
CANONICAL_TAG_KEYS = (TAG_DATASET, TAG_MODEL_TYPE, TAG_PHASE, TAG_ENV)


# ============================================================
# HELPERS
# ============================================================


def _clean(tags: dict[str, Any]) -> dict[str, str]:
    """
    Coerce all values to strings and drop None entries.

    MLflow requires string tag values; this lets callers pass
    Path objects, None, etc. without each call site handling it.
    """
    cleaned: dict[str, str] = {}
    for k, v in tags.items():
        if v is None:
            continue
        cleaned[k] = str(v)
    return cleaned


def training_tags(
    dataset: str,
    model_type: str,
    phase: str = "unset",
    env: str = "research",
) -> dict[str, str]:
    """Canonical tags for a training run."""
    return _clean(
        {
            TAG_DATASET: dataset,
            TAG_MODEL_TYPE: model_type,
            TAG_PHASE: phase,
            TAG_ENV: env,
        }
    )


def evaluation_tags(
    dataset: str,
    model_type: str,
    phase: str = "unset",
    env: str = "research",
) -> dict[str, str]:
    """Canonical tags for an evaluation run."""
    return _clean(
        {
            TAG_DATASET: dataset,
            TAG_MODEL_TYPE: model_type,
            TAG_PHASE: phase,
            TAG_ENV: env,
        }
    )


def baseline_tags(
    dataset: str,
    method: str,
    phase: str = "baseline-v1.0",
    env: str = "research",
) -> dict[str, str]:
    """
    Canonical tags for a classical baseline run (e.g., STA/LTA).

    `method` is stored under the model_type tag for consistency with
    training runs — this way, "find all runs by model type" queries
    work for both ML and classical methods.
    """
    return _clean(
        {
            TAG_DATASET: dataset,
            TAG_MODEL_TYPE: method,
            TAG_PHASE: phase,
            TAG_ENV: env,
        }
    )


def sweep_tags(
    dataset: str,
    model: str,
    loss: str,
    sweep_id: str,
    phase: str = "unset",
    env: str = "research",
) -> dict[str, str]:
    """
    Canonical tags for a sweep run.

    Adds two sweep-specific tags (`model`, `loss`, `sweep_id`) on top
    of the canonical four.
    """
    base = _clean(
        {
            TAG_DATASET: dataset,
            TAG_MODEL_TYPE: model,
            TAG_PHASE: phase,
            TAG_ENV: env,
        }
    )
    base.update(
        _clean(
            {
                "loss": loss,
                "sweep_id": sweep_id,
            }
        )
    )
    return base

def comparison_tags(
    dataset: str,
    ml_model: str,
    sta_config: str,
    phase: str = "unset",
    env: str = "research",
) -> dict[str, str]:
    """
    Canonical tags for a baseline comparison run.

    ml_model: human-readable identifier for the ML model (e.g., the
        checkpoint filename stem, or an MLflow run ID).
    sta_config: human-readable identifier for the STA/LTA config
        (e.g., "sta15_lta150_thr3.0").
    """
    base = _clean({
        TAG_DATASET: dataset,
        TAG_MODEL_TYPE: "ML_vs_STALTA",
        TAG_PHASE: phase,
        TAG_ENV: env,
    })
    base.update(_clean({
        "ml_model": ml_model,
        "sta_config": sta_config,
    }))
    return base

def explainability_tags(
    dataset: str,
    model_type: str,
    split: str,
    phase: str = "unset",
    env: str = "research",
) -> dict[str, str]:
    """
    Canonical tags for an explainability run.
    """
    base = _clean({
        TAG_DATASET: dataset,
        TAG_MODEL_TYPE: model_type,
        TAG_PHASE: phase,
        TAG_ENV: env,
    })
    base.update(_clean({"split": split}))
    return base