#!/usr/bin/env python3
# file location: scripts/verify_baseline_run.py

"""
Verify that a baseline-v1.0 run exists in MLflow.

Checks:
    - seismic-fbp-training has exactly one run with tags.phase='baseline-v1.0'
    - seismic-fbp-evaluation has exactly one run with tags.phase='baseline-v1.0'
    - Both runs have non-empty metrics
    - Training run has val_iou and val_loss
    - Evaluation run has eval_test_mean_iou and eval_test_mae_samples

Prints a summary and exits 0 on success, 1 on failure.

Usage:
    python scripts/verify_baseline_run.py
    python scripts/verify_baseline_run.py --phase baseline-v1.0
"""

import os
import sys

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tracking_conventions import (
    EXPERIMENT_EVALUATION,
    EXPERIMENT_TRAINING,
)


BASELINE_PHASE = "baseline-v1.0"


# ============================================================
# HELPERS
# ============================================================

def get_run_metrics(client, run_id: str) -> dict[str, float]:
    """Fetch all metrics for a run as a flat dict."""
    try:
        run = client.get_run(run_id)
        return dict(run.data.metrics)
    except Exception as e:
        print(f"    ⚠️  Could not fetch metrics for {run_id[:8]}: {e}")
        return {}


def get_run_tags(client, run_id: str) -> dict[str, str]:
    """Fetch all tags for a run as a flat dict."""
    try:
        run = client.get_run(run_id)
        return dict(run.data.tags)
    except Exception as e:
        print(f"    ⚠️  Could not fetch tags for {run_id[:8]}: {e}")
        return {}


def find_runs_by_phase(
    client,
    experiment_name: str,
    phase: str,
) -> list:
    """Find runs in an experiment whose tags.phase matches."""
    import mlflow

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.phase = '{phase}'",
    )
    return list(runs.iterrows())


def print_run_summary(
    label: str,
    run_row,
    client,
    required_metrics: list[str],
) -> tuple[bool, dict[str, float]]:
    """
    Print a summary of one run and return (has_required, metrics).
    """
    _, row = run_row
    run_id = row["run_id"]
    run_name = row.get("tags.mlflow.runName", "N/A")

    print()
    print(f"  {label}:")
    print(f"    run_id:     {run_id}")
    print(f"    run_name:   {run_name}")

    metrics = get_run_metrics(client, run_id)
    tags = get_run_tags(client, run_id)

    print(f"    phase:      {tags.get('phase', 'N/A')}")
    print(f"    dataset:    {tags.get('dataset', 'N/A')}")
    print(f"    model_type: {tags.get('model_type', 'N/A')}")

    if not metrics:
        print("    ⚠️  No metrics found for this run.")
        return False, {}

    # Print a subset of interesting metrics
    interesting = ["val_loss", "val_iou", "val_f1", "val_accuracy"]
    for key in interesting:
        if key in metrics:
            print(f"    {key:<12} {metrics[key]:.4f}")

    # Evaluation-specific metrics
    eval_interesting = [
        "eval_test_mean_iou",
        "eval_test_mean_f1",
        "eval_test_mae_samples",
        "eval_test_accuracy_within_3",
        "eval_train_mean_iou",
        "eval_val_mean_iou",
    ]
    for key in eval_interesting:
        if key in metrics:
            print(f"    {key:<28} {metrics[key]:.4f}")

    # Check required metrics
    missing = [k for k in required_metrics if k not in metrics]
    if missing:
        print(f"    ⚠️  Missing required metrics: {missing}")
        return False, metrics

    return True, metrics


# ============================================================
# MAIN
# ============================================================

@click.command()
@click.option(
    "--phase",
    type=str,
    default=BASELINE_PHASE,
    help=f"Phase tag to look for. Default: {BASELINE_PHASE}",
)
def main(phase: str) -> None:
    print("=" * 80)
    print(f"VERIFY BASELINE RUN — phase='{phase}'")
    print("=" * 80)

    # Import MLflow and get a client
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("❌ MLflow is not installed.")
        sys.exit(1)

    # Ensure the tracking URI matches what the training script uses.
    # The MLflowManager uses sqlite:///mlflow.db by default.
    if not mlflow.get_tracking_uri() or "mlflow.db" not in mlflow.get_tracking_uri():
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"  Experiment (training):   {EXPERIMENT_TRAINING}")
    print(f"  Experiment (evaluation): {EXPERIMENT_EVALUATION}")

    client = MlflowClient()

    # ---- Find training run ----
    print()
    print("=" * 80)
    print("TRAINING RUN")
    print("=" * 80)

    training_runs = find_runs_by_phase(client, EXPERIMENT_TRAINING, phase)
    n_train = len(training_runs)

    if n_train == 0:
        print(f"  ❌ No training run found with phase='{phase}'")
        sys.exit(1)
    if n_train > 1:
        print(f"  ⚠️  Found {n_train} training runs with phase='{phase}'")
        print("     Expected exactly 1. Listing all:")
        for r in training_runs:
            _, row = r
            print(f"       - {row['run_id']} ({row.get('tags.mlflow.runName', 'N/A')})")
        sys.exit(1)

    train_ok, train_metrics = print_run_summary(
        "Training",
        training_runs[0],
        client,
        required_metrics=["val_loss", "val_iou"],
    )

    # ---- Find evaluation run ----
    print()
    print("=" * 80)
    print("EVALUATION RUN")
    print("=" * 80)

    eval_runs = find_runs_by_phase(client, EXPERIMENT_EVALUATION, phase)
    n_eval = len(eval_runs)

    if n_eval == 0:
        print(f"  ❌ No evaluation run found with phase='{phase}'")
        sys.exit(1)
    if n_eval > 1:
        print(f"  ⚠️  Found {n_eval} evaluation runs with phase='{phase}'")
        print("     Expected exactly 1. Listing all:")
        for r in eval_runs:
            _, row = r
            print(f"       - {row['run_id']} ({row.get('tags.mlflow.runName', 'N/A')})")
        sys.exit(1)

    eval_ok, eval_metrics = print_run_summary(
        "Evaluation",
        eval_runs[0],
        client,
        required_metrics=["eval_test_mean_iou", "eval_test_mae_samples"],
    )

    # ---- Consistency check (warning only) ----
    print()
    print("=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)

    val_iou = train_metrics.get("val_iou")
    test_iou = eval_metrics.get("eval_test_mean_iou")
    val_loss = train_metrics.get("val_loss")
    test_mae = eval_metrics.get("eval_test_mae_samples")

    if val_iou is not None and test_iou is not None:
        print(f"  Train val_iou:     {val_iou:.4f}")
        print(f"  Eval test_iou:     {test_iou:.4f}")
        if test_iou > val_iou + 0.05:
            print(
                f"  ⚠️  test_iou is more than 5 pp HIGHER than val_iou. "
                f"This is suspicious (val usually ≈ test or higher)."
            )
        else:
            print("  ✅ test_iou is consistent with val_iou.")
    else:
        print("  (skipping IoU consistency check — metric missing)")

    if test_mae is not None:
        print(f"  Eval test MAE:     {test_mae:.2f} samples")

    # ---- Summary ----
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_ok = train_ok and eval_ok

    print(f"  {'✅' if train_ok else '❌'} Training run has required metrics")
    print(f"  {'✅' if eval_ok else '❌'} Evaluation run has required metrics")

    print()
    if all_ok:
        print(f"🎉 BASELINE '{phase}' IS INTACT.")
        print()
        print("  Baseline metrics:")
        if val_iou is not None:
            print(f"    val_iou:            {val_iou:.4f}")
        if val_loss is not None:
            print(f"    val_loss:           {val_loss:.4f}")
        if test_iou is not None:
            print(f"    test_iou:           {test_iou:.4f}")
        if test_mae is not None:
            print(f"    test_mae_samples:   {test_mae:.2f}")
    else:
        print(f"❌ BASELINE '{phase}' HAS ISSUES — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()