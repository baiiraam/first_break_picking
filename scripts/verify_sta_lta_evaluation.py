#!/usr/bin/env python3
# file location: scripts/verify_sta_lta_evaluation.py

"""
Verify STA/LTA evaluation outputs (D.2).

Checks:
    1. The per-trace CSV exists and has expected columns
    2. The images directory (if present) has the expected PNGs
    3. Each PNG is non-trivial and valid
    4. Optionally: check MLflow artifacts of the latest run

Usage:
    python scripts/verify_sta_lta_evaluation.py
    python scripts/verify_sta_lta_evaluation.py --mlflow-latest
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
    "pred_pick_sample",
    "error_samples",
]

REQUIRED_IMAGES = [
    "shot_best.png",
    "shot_median.png",
    "shot_worst.png",
    "error_histogram.png",
]

OPTIONAL_IMAGES = [
    "error_vs_position.png",
]


# ============================================================
# HELPERS
# ============================================================


def find_latest_csv(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = sorted(
        base.glob("sta_lta_eval_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_images_dir(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        p for p in base.iterdir() if p.is_dir() and p.name.startswith("sta_lta_images_")
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
        print("    ⚠️  CSV is empty")
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

        img = Image.open(path)
        img.verify()
    except ImportError:
        return True, f"OK ({size} bytes, no PIL)"
    except Exception as e:
        return False, f"PIL error: {e}"
    return True, f"OK ({size} bytes)"


def check_images(images_dir: Path | None) -> tuple[int, int]:
    """Returns (required_ok, required_total)."""
    if images_dir is None:
        print("  Images: (no images directory found)")
        return 0, 0

    print(f"  Images dir: {images_dir}")
    ok_count = 0
    for name in REQUIRED_IMAGES:
        ok, msg = check_image(images_dir / name)
        status = "✅" if ok else "❌"
        print(f"    {status} {name:<30} {msg}")
        if ok:
            ok_count += 1
    for name in OPTIONAL_IMAGES:
        ok, msg = check_image(images_dir / name)
        status = "✅" if ok else "⚠️ "
        print(f"    {status} {name:<30} {msg} (optional)")
    return ok_count, len(REQUIRED_IMAGES)


def check_mlflow() -> bool:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("  ⚠️  MLflow not installed — skipping MLflow check")
        return True

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name("seismic-fbp-baselines")
    if exp is None:
        print("  ❌ Experiment seismic-fbp-baselines not found")
        return False

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        print("  ❌ No runs in seismic-fbp-baselines")
        return False

    run = runs[0]
    print(f"  Latest baselines run: {run.info.run_id}")
    print(f"    name:  {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"    phase: {run.data.tags.get('phase', 'N/A')}")

    try:
        artifacts = client.list_artifacts(run.info.run_id)
    except Exception as e:
        print(f"    ❌ Could not list artifacts: {e}")
        return False

    paths = [a.path for a in artifacts]
    print(f"    Top-level artifact paths: {paths}")

    # Look for predictions and images subdirs
    has_predictions = any("predictions" in p for p in paths)
    has_images = any("images" in p for p in paths)

    if not has_predictions:
        print("    ⚠️  no 'predictions' artifact path found")
    if not has_images:
        print("    ⚠️  no 'images' artifact path found")

    return has_predictions or has_images


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--mlflow-latest", is_flag=True, default=False, help="Also verify MLflow artifacts"
)
def main(mlflow_latest: bool) -> None:
    print("=" * 80)
    print("VERIFY STA/LTA EVALUATION (D.2)")
    print("=" * 80)

    base = Path("evaluation_results")

    csv_path = find_latest_csv(base)
    if csv_path is None:
        print(f"❌ No sta_lta_eval_*.csv found under {base}")
        sys.exit(1)

    csv_ok = check_csv(csv_path)

    print()
    images_dir = find_latest_images_dir(base)
    req_ok, req_total = check_images(images_dir)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  CSV:            {'✅ PASS' if csv_ok else '❌ FAIL'}")
    if req_total > 0:
        print(f"  Images:         {req_ok}/{req_total} required present")
    else:
        print("  Images:         (none found)")

    mlflow_ok = True
    if mlflow_latest:
        print()
        print("-" * 60)
        print("  MLFLOW CHECK")
        print("-" * 60)
        mlflow_ok = check_mlflow()

    all_ok = csv_ok and (req_total == 0 or req_ok == req_total) and mlflow_ok

    print()
    if all_ok:
        print("🎉 STA/LTA EVALUATION VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("❌ STA/LTA EVALUATION VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
