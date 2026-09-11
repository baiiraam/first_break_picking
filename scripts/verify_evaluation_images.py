#!/usr/bin/env python3
# file location: scripts/verify_evaluation_images.py

"""
Verify evaluation image generation (C.2.1).

Checks:
    1. The expected image files exist on disk for the requested split(s)
    2. Each file is non-trivial (>1 KB)
    3. Each file opens as a valid PNG
    4. Each file has more than one unique color (not a blank canvas)
    5. If --mlflow-latest is set, that MLflow has the images as artifacts

error_vs_position.png is optional. It's skipped by the runner when the
detailed DataFrame lacks a 'pick_sample' column, which is currently the
case. Missing optional files are warnings, not failures.

Usage:
    # Check latest images folder for the test split
    python scripts/verify_evaluation_images.py

    # Check a specific split
    python scripts/verify_evaluation_images.py --split train
    python scripts/verify_evaluation_images.py --split all

    # Check a specific folder
    python scripts/verify_evaluation_images.py --dir evaluation_results/images_Halfmile_20260911_130644

    # Also verify MLflow artifacts
    python scripts/verify_evaluation_images.py --mlflow-latest
"""

import os
import sys
from pathlib import Path

import click

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Required images per split (must be present)
REQUIRED_SUFFIXES = [
    "shot_best.png",
    "shot_median.png",
    "shot_worst.png",
    "error_histogram.png",
]

# Optional images per split (warn if missing, don't fail)
OPTIONAL_SUFFIXES = [
    "error_vs_position.png",
]

ALL_SPLITS = ["train", "val", "test"]


# ============================================================
# HELPERS
# ============================================================


def find_latest_images_dir(base: Path) -> Path | None:
    """Return the most recent images_* directory under `base`."""
    if not base.exists():
        return None
    candidates = [
        p for p in base.iterdir() if p.is_dir() and p.name.startswith("images_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_file(path: Path) -> tuple[bool, str]:
    """Return (ok, message) for a single image file."""
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < 1024:
        return False, f"too small ({size} bytes)"
    try:
        from PIL import Image

        img = Image.open(path)
        img.verify()
        # verify() closes the file; reopen to inspect colors
        img = Image.open(path)
        import numpy as np

        arr = np.array(img.convert("RGB"))
        unique_colors = len(set(map(tuple, arr.reshape(-1, 3)[::200])))
        if unique_colors < 3:
            return False, f"too few colors ({unique_colors})"
    except ImportError:
        return True, f"OK ({size} bytes, no PIL for deeper check)"
    except Exception as e:  # noqa: BLE001
        return False, f"PIL error: {e}"
    return True, f"OK ({size} bytes)"


def check_mlflow_artifacts() -> bool:
    """Check that the latest evaluation run has image artifacts."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("  ⚠️  MLflow not installed — skipping artifact check")
        return True

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    exp = mlflow.get_experiment_by_name("seismic-fbp-evaluation")
    if exp is None:
        print("  ❌ Experiment 'seismic-fbp-evaluation' not found")
        return False

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        print("  ❌ No runs found in seismic-fbp-evaluation")
        return False

    run = runs[0]
    print(f"  Latest eval run: {run.info.run_id}")

    try:
        artifacts = client.list_artifacts(run.info.run_id, path="evaluation/images")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ Could not list artifacts: {e}")
        return False

    found = [a.path.split("/")[-1] for a in artifacts if a.path.endswith(".png")]
    print(f"  Found {len(found)} PNG artifacts in MLflow")
    for name in sorted(found):
        print(f"    - {name}")

    if not found:
        print("  ❌ No image artifacts logged to MLflow")
        return False

    return True


# ============================================================
# MAIN
# ============================================================


@click.command()
@click.option(
    "--dir",
    "images_dir_str",
    default=None,
    help="Specific images_* directory to check. "
    "Default: latest under evaluation_results/",
)
@click.option(
    "--split",
    "split_opt",
    default="test",
    type=click.Choice(["train", "val", "test", "all"]),
    help="Which split(s) to verify. Default: test.",
)
@click.option(
    "--mlflow-latest",
    is_flag=True,
    default=False,
    help="Also verify that the latest MLflow eval run has image artifacts.",
)
def main(images_dir_str: str | None, split_opt: str, mlflow_latest: bool) -> None:
    print("=" * 80)
    print("VERIFY EVALUATION IMAGES (C.2.1)")
    print("=" * 80)

    # Determine which splits to check
    if split_opt == "all":
        splits_to_check = ALL_SPLITS
    else:
        splits_to_check = [split_opt]

    # Locate images directory
    base = Path("evaluation_results")
    if images_dir_str:
        images_dir = Path(images_dir_str)
    else:
        images_dir = find_latest_images_dir(base)

    if images_dir is None:
        print(f"❌ No images_* directory found under {base}")
        print("   Run:  python scripts/evaluate.py ... --save-images")
        sys.exit(1)

    print(f"  Images directory: {images_dir}")
    print(f"  Splits to check:  {splits_to_check}")

    if not images_dir.exists():
        print(f"❌ Directory does not exist: {images_dir}")
        sys.exit(1)

    total_required = 0
    total_required_ok = 0
    total_optional = 0
    total_optional_ok = 0

    for split in splits_to_check:
        print()
        print("-" * 60)
        print(f"  SPLIT: {split}")
        print("-" * 60)

        # Required files
        for suffix in REQUIRED_SUFFIXES:
            fname = f"{split}_{suffix}"
            fpath = images_dir / fname
            ok, msg = check_file(fpath)
            status = "✅" if ok else "❌"
            print(f"    {status} {fname:<32} {msg}")
            total_required += 1
            if ok:
                total_required_ok += 1

        # Optional files (warn if missing)
        for suffix in OPTIONAL_SUFFIXES:
            fname = f"{split}_{suffix}"
            fpath = images_dir / fname
            if fpath.exists():
                ok, msg = check_file(fpath)
                status = "✅" if ok else "⚠️"
                print(f"    {status} {fname:<32} {msg} (optional)")
                total_optional += 1
                if ok:
                    total_optional_ok += 1
            else:
                print(
                    f"    ⚠️  {fname:<32} missing (optional — expected when"
                    f" 'pick_sample' column unavailable)"
                )
                total_optional += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Required: {total_required_ok}/{total_required} present and valid")
    print(f"  Optional: {total_optional_ok}/{total_optional} present and valid")

    required_ok = total_required_ok == total_required

    # Optional MLflow check
    mlflow_ok = True
    if mlflow_latest:
        print()
        print("-" * 60)
        print("  MLFLOW ARTIFACT CHECK")
        print("-" * 60)
        mlflow_ok = check_mlflow_artifacts()

    print()
    if required_ok and mlflow_ok:
        print("🎉 IMAGE VERIFICATION PASSED")
        sys.exit(0)
    else:
        if not required_ok:
            print(f"❌ Required images missing: {total_required - total_required_ok}")
        if not mlflow_ok:
            print("❌ MLflow artifact check failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
