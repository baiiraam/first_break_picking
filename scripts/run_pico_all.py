#!/usr/bin/env python3
"""
Run PicoUNet on all datasets with logging.
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Define datasets
DATASETS = ["Halfmile", "Sudbury", "Brunswick", "Lalor"]
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# Create logs directory
Path("logs/pico_runs").mkdir(parents=True, exist_ok=True)

print("=" * 40)
print("🚀 Running PicoUNet on All Datasets")
print(f"Timestamp: {TIMESTAMP}")
print("=" * 40)

for dataset in DATASETS:
    print(f"\n{'=' * 40}")
    print(f"📊 Dataset: {dataset}")
    print(f"{'=' * 40}")

    dataset_lower = dataset.lower()
    log_file = f"logs/pico_runs/pico_{dataset}_{TIMESTAMP}.log"

    cmd = [
        "python3.12",
        "scripts/train.py",
        "--config",
        f"configs/{dataset_lower}.yaml",
        "--model",
        "pico",
        "--epochs",
        "1",
        "--verbose",
        "--log-memory",
        "--log-level",
        "DEBUG",
    ]

    print(f"📝 Logging to: {log_file}")
    print(f"🚀 Running: {' '.join(cmd)}")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

        # Print to console and write to file simultaneously
        for line in process.stdout:
            print(line, end="")
            f.write(line)

        process.wait()

    if process.returncode == 0:
        print(f"✅ SUCCESS: {dataset} completed")
    else:
        print(f"❌ FAILED: {dataset} had errors (exit code: {process.returncode})")

    print("\n" + "-" * 40)
    print("Press Enter to continue to next dataset...")
    input()

print("\n" + "=" * 40)
print("✅ All datasets processed!")
print("📁 Logs saved in: logs/pico_runs/")
print("=" * 40)
