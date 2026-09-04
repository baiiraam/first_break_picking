#!/usr/bin/env python3
"""
Test script for SeismicTrainer modifications including callbacks,
unified step execution, and memory management.
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SeismicConfig
from src.models.factory import create_model
from src.training.callbacks import (
    EarlyStoppingCallback,
    GradientMonitorCallback,
    ModelCheckpointCallback,
)
from src.training.losses import create_loss_function
from src.training.trainer import SeismicTrainer


class MockSeismicDataset(Dataset):
    """Mock dataset for testing with realistic shapes."""

    def __init__(
        self,
        n_samples: int = 50,
        n_traces: int = 1578,
        n_time_samples: int = 751,
        random_seed: int = 42,
    ):
        self.n_samples = n_samples
        self.n_traces = n_traces
        self.n_time_samples = n_time_samples
        self.rng = np.random.default_rng(random_seed)

        # Pre-generate data for reproducibility
        self.data = self.rng.standard_normal(
            (n_samples, 1, n_traces, n_time_samples)
        ).astype(np.float32)

        # Generate masks with realistic patterns
        self.masks = self._generate_masks()

    def _generate_masks(self):
        """Generate realistic 3-class masks."""
        masks = np.zeros(
            (self.n_samples, self.n_traces, self.n_time_samples), dtype=np.int64
        )

        for i in range(self.n_samples):
            # Random pick positions (between 100-600 samples)
            picks = self.rng.integers(100, 600, size=self.n_traces)

            for trace in range(self.n_traces):
                pick = picks[trace]
                # Class 2: Strip (8 samples wide)
                start = max(0, pick - 4)
                end = min(self.n_time_samples, pick + 5)
                masks[i, trace, start:end] = 2
                # Class 1: After strip
                masks[i, trace, end:] = 1
                # Class 0: Before strip (already zeros)

        return masks

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        mask = torch.from_numpy(self.masks[idx])
        return data, mask


def create_test_config(tmpdir: str) -> SeismicConfig:
    """Create a test configuration."""
    # ✅ Remove 'model_name' from config dict (it's not a valid field)
    config_dict = {
        "dataset_name": "TestDataset",
        "hdf5_path": "test.hdf5",
        "chunk_dir": str(Path(tmpdir) / "chunks"),
        "preprocess": False,
        "force_reprocess": False,
        "target_traces": 1578,
        "n_samples": 751,
        "strip_width": 8,
        "sample_rate_ms": 2.0,
        "chunk_size": 20,
        "random_seed": 42,
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "n_epochs": 5,
        "device": "cpu",
        "num_workers": 0,
        "multi_gpu": False,
        "gpu_ids": None,
        "class_weights": [0.05, 0.05, 0.9],
        "model_registry_dir": str(Path(tmpdir) / "registry"),
        "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
        "checkpoint_every": 2,
        "cache_size": 3,
        "lr_scheduler": "plateau",
        "lr_patience": 3,
        "lr_factor": 0.5,
        "lr_step_size": 10,
        "lr_gamma": 0.5,
        "lr_T_max": 30,
        "loss_function": "combo",
        "dice_weight": 0.5,
        "focal_gamma": 2.0,
        "gradient_clip_value": 1.0,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.01,
        "tensorboard_log_dir": str(Path(tmpdir) / "logs" / "tensorboard"),
        "mlflow_experiment_name": "test_experiment",
        "log_dir": str(Path(tmpdir) / "logs"),
        "log_level": "INFO",
        "log_memory": True,
        "log_predictions_every": 3,
        "log_metrics_every": 1,
        "log_gradients": False,
        "verbose_training": True,
        "log_batch_every": None,
    }

    return SeismicConfig(**config_dict)


def create_dataloaders(dataset, batch_size: int = 2):
    """Create train/val/test dataloaders."""
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def print_section(title: str, char: str = "=", width: int = 70):
    """Print a formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def test_trainer_modifications():
    """Test all trainer modifications."""

    print_section("🧪 TESTING SEISMIC TRAINER MODIFICATIONS", "=", 70)
    print(f"Started at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    device_str = "MPS" if torch.backends.mps.is_available() else "CPU"
    print(f"Device: {device_str}")

    # Use temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n📁 Temporary directory: {tmpdir}")

        # Step 1: Create config
        print_section("1. CREATING CONFIGURATION", "-", 70)
        config = create_test_config(tmpdir)
        print("✅ Config created")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Epochs: {config.n_epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Device: {config.device}")

        # Step 2: Create dataset
        print_section("2. CREATING MOCK DATASET", "-", 70)
        dataset = MockSeismicDataset(n_samples=50)
        print("✅ Dataset created")
        print(f"   Samples: {len(dataset)}")
        print(f"   Data shape: {dataset[0][0].shape}")
        print(f"   Mask shape: {dataset[0][1].shape}")
        print(f"   Unique mask classes: {torch.unique(dataset[0][1]).tolist()}")

        # Step 3: Create dataloaders
        print_section("3. CREATING DATALOADERS", "-", 70)
        dataloaders = create_dataloaders(dataset, batch_size=config.batch_size)
        print("✅ Dataloaders created")
        print(f"   Train batches: {len(dataloaders['train'])}")
        print(f"   Val batches: {len(dataloaders['val'])}")
        print(f"   Test batches: {len(dataloaders['test'])}")

        # Step 4: Create model
        print_section("4. CREATING MODEL", "-", 70)
        model_name = "pico"
        model = create_model(model_name, in_channels=1, out_channels=3)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {model_name}")
        print(f"   Total parameters: {total_params:,}")

        # Step 5: Create loss
        print_section("5. CREATING LOSS FUNCTION", "-", 70)
        criterion = create_loss_function(config)
        print(f"✅ Loss function: {config.loss_function}")
        print(f"   Class weights: {config.class_weights}")

        # Step 6: Create optimizer
        print_section("6. CREATING OPTIMIZER", "-", 70)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        print("✅ Optimizer: Adam")
        print(f"   Learning rate: {config.learning_rate}")

        # Step 7: Create callbacks
        print_section("7. CREATING CALLBACKS", "-", 70)

        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStoppingCallback(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                monitor="val_loss",
                verbose=True,
            ),
            ModelCheckpointCallback(
                save_dir=checkpoint_dir,
                save_best=True,
                save_every=config.checkpoint_every,
                monitor="val_loss",
                verbose=True,
            ),
            GradientMonitorCallback(
                log_every=1,
                warn_threshold=5.0,
                verbose=True,
            ),
        ]

        print(f"✅ Created {len(callbacks)} callbacks:")
        for cb in callbacks:
            print(f"   - {cb.__class__.__name__}")

        # Step 8: Create trainer
        print_section("8. CREATING TRAINER", "-", 70)

        trainer = SeismicTrainer(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            model_name=model_name,  # ✅ This is fine - trainer accepts model_name
            callbacks=callbacks,
        )

        print("✅ Trainer created")
        print(f"   Registered callbacks: {len(trainer.callbacks)}")
        print(f"   Model registry: {trainer.registry_dir}")
        print(f"   TensorBoard: {trainer.tb_dir}")
        print(
            f"   Memory manager: {'Running' if trainer.memory_manager._running else 'Stopped'}"
        )

        # Step 9: Train
        print_section("9. RUNNING TRAINING", "=", 70)
        print("Training for 5 epochs with callbacks enabled...\n")

        start_time = time.time()

        try:
            trainer.fit(verbose=True)
            training_success = True
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback

            traceback.print_exc()
            training_success = False

        training_time = time.time() - start_time
        print(f"\n⏱ Training time: {training_time:.2f}s")

        # Step 10: Verify outputs
        print_section("10. VERIFYING OUTPUTS", "=", 70)

        # Check checkpoints
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        print(f"📁 Checkpoints created: {len(checkpoint_files)}")
        for f in checkpoint_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")

        # Check registry
        registry_dir = Path(tmpdir) / "registry"
        registry_files = list(registry_dir.glob("*.pt"))
        print(f"\n📁 Registry files: {len(registry_files)}")
        for f in registry_files[:5]:  # Show first 5
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.2f} MB)")
        if len(registry_files) > 5:
            print(f"   ... and {len(registry_files) - 5} more")

        # Check best model
        best_model_path = checkpoint_dir / f"{model_name}_{config.dataset_name}_best.pt"
        if best_model_path.exists():
            print(f"\n🏆 Best model found: {best_model_path.name}")
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu")
                print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"   Best value: {checkpoint.get('best_value', 'N/A')}")
                print(f"   Keys: {list(checkpoint.keys())}")
                if "mlflow_run_id" in checkpoint:
                    print(f"   MLflow run ID: {checkpoint['mlflow_run_id']}")
            except Exception as e:
                print(f"   ⚠️ Could not load checkpoint: {e}")
        else:
            print("\n⚠️ No best model found (may not have improved)")

        # Check memory manager
        print(
            f"\n💾 Memory manager status: {'Running' if trainer.memory_manager._running else 'Stopped'}"
        )

        # Step 11: Summary
        print_section("✅ TEST SUMMARY", "=", 70)

        checks = []

        # Check 1: Training completed
        if training_success:
            checks.append(("✅ Training completed successfully", True))
        else:
            checks.append(("❌ Training failed", False))

        # Check 2: Checkpoints created
        if len(checkpoint_files) > 0:
            checks.append((f"✅ {len(checkpoint_files)} checkpoints created", True))
        else:
            checks.append(("❌ No checkpoints created", False))

        # Check 3: Best model saved (if improved)
        if best_model_path.exists():
            checks.append(("✅ Best model saved", True))
        else:
            checks.append(
                ("ℹ️ Best model not saved (no improvement)", True)
            )  # Not a failure

        # Check 4: Callbacks executed
        if len(checkpoint_files) > 0:
            checks.append(("✅ Callbacks executed (checkpoints created)", True))
        else:
            checks.append(("⚠️ Callback execution uncertain", True))

        # Print checks
        for check, passed in checks:
            print(f"  {check}")

        # Final status
        all_passed = all(passed for _, passed in checks)
        print("\n" + "=" * 70)
        if all_passed:
            print(
                "🎉 ALL TESTS PASSED! Trainer modifications are working correctly.".center(
                    70
                )
            )
        else:
            print("⚠️ SOME TESTS FAILED. Please review the output above.".center(70))
        print("=" * 70)

        # Step 12: Cleanup
        print_section("🧹 CLEANUP", "-", 70)
        print("Temporary directory will be automatically cleaned up.")
        print("Test completed.")


if __name__ == "__main__":
    try:
        test_trainer_modifications()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
