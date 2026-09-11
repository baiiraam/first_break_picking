#!/usr/bin/env python3
# file location: scripts/verify_baseline_comparison.py

"""
Verify baseline comparison outputs (E.1).

Usage:
    python scripts/verify_baseline_comparison.py
    python scripts/verify_baseline_comparison.py --mlflow-latest
"""

import os
import sys
from pathlib import Path

import click
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REQUIRED_COLUMNS = [
    "shot_id",
    "trace_index",
    "gt_pick_sample",
    "sta_lta_pick",
    "sta_lta_error_samples",
    "ml_pick",
    "ml_error_samples",
]

REQUIRED_IMAGES = [
    "shot_best.png",
    "shot_median.png",
    "shot_worst.png",
    "error_histogram.png",
    "error_vs_position.png",
    "summary_comparison.png",
]


def find_latest_csv(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = sorted(
        base.glob("comparison_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_images_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        p for p in base.iterdir()
        if p.is_dir() and p.name.startswith("comparison_images_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_csv(csv_path: Path) -> bool:
    print(f"  CSV: {csv_path}")
    if not csv_path.exists():
        print("    ❌ not found")
        return False
    df = pd.read_csv(csv_path)
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {list(df.columns)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"    ❌ missing columns: {missing}")
        return False
    if len(df) == 0:
        print("    ⚠️  empty CSV")
        return False
    print("    ✅ columns OK")
    return True


def check_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < 1024:
        return False, f"too small ({size} bytes)"
    try:
        from PIL import Image
        Image.open(path).verify()
    except ImportError:
        return True, f"OK ({size} bytes, no PIL)"
    except Exception as e:  # noqa: BLE001
        return False, f"PIL error: {e}"
    return True, f"OK ({size} bytes)"


def check_images(images_dir: Path | None) -> tuple[int, int]:
    if images_dir is None:
        print("  Images: (no directory)")
        return 0, 0
    print(f"  Images dir: {images_dir}")
    ok = 0
    for name in REQUIRED_IMAGES:
        good, msg = check_image(images_dir / name)
        status = "✅" if good else "❌"
        print(f"    {status} {name:<32} {msg}")
        if good:
            ok += 1
    return ok, len(REQUIRED_IMAGES)


def check_mlflow() -> bool:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("  ⚠️  MLflow not installed — skipping")
        return True

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name("seismic-fbp-comparison")
    if exp is None:
        print("  ❌ Experiment seismic-fbp-comparison not found")
        return False
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        print("  ❌ No runs found")
        return False
    run = runs[0]
    print(f"  Latest run: {run.info.run_id}")
    print(f"    name:  {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"    phase: {run.data.tags.get('phase', 'N/A')}")
    metrics = run.data.metrics
    interesting = [
        "ml_mae_samples", "sta_lta_mae_samples",
        "ml_minus_sta_lta_mae",
        "ml_within_3_accuracy", "sta_lta_within_3_accuracy",
    ]
    for k in interesting:
        if k in metrics:
            print(f"    {k}: {metrics[k]:.4f}")

    artifacts = client.list_artifacts(run.info.run_id)
    paths = [a.path for a in artifacts]
    print(f"    artifact paths: {paths}")
    has_comp = any("comparison" in p for p in paths)
    has_pred = any("predictions" in p for p in paths)
    return has_comp or has_pred


@click.command()
@click.option("--mlflow-latest", is_flag=True, default=False)
def main(mlflow_latest: bool) -> None:
    print("=" * 80)
    print("VERIFY BASELINE COMPARISON (E.1)")
    print("=" * 80)

    base = Path("evaluation_results")
    csv_path = find_latest_csv(base)
    if csv_path is None:
        print("❌ No comparison_*.csv found")
        sys.exit(1)
    csv_ok = check_csv(csv_path)

    print()
    images_dir = find_latest_images_dir(base)
    req_ok, req_total = check_images(images_dir)

    mlflow_ok = True
    if mlflow_latest:
        print()
        print("-" * 60)
        print("  MLFLOW CHECK")
        print("-" * 60)
        mlflow_ok = check_mlflow()

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  CSV:    {'✅' if csv_ok else '❌'}")
    print(f"  Images: {req_ok}/{req_total}")
    all_ok = csv_ok and req_ok == req_total and mlflow_ok
    if all_ok:
        print("\n🎉 BASELINE COMPARISON VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("\n❌ BASELINE COMPARISON VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()