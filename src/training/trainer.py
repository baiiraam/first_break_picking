"""
Training loop with MLflow, TensorBoard, Loguru integration, and modular callbacks.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import SeismicConfig
from src.training.callbacks import Callback  # 🔥 NEW: Import Callback base class
from src.training.losses import ComboLoss
from src.training.metrics import (
    SegmentationMetrics,
)
from src.utils.logger import get_logger
from src.utils.memory_utils import get_memory_manager  # 🔥 NEW: Memory manager
from src.utils.mlflow_utils import (
    get_mlflow_manager,
)

logger = get_logger()


class SeismicTrainer:
    """
    Integrated trainer with MLflow, TensorBoard, modular callbacks, and Loguru.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict[str, Any],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: SeismicConfig,
        model_name: str = "unet",
        mlflow_run_id: str | None = None,
        callbacks: list[Callback] | None = None,  # 🔥 NEW: Accept callbacks
    ):
        self.config = config
        self.model_name = model_name
        self.device = self._setup_device(config.device)
        self.mlflow_run_id = mlflow_run_id
        self.callbacks = callbacks or []  # 🔥 NEW: Store callbacks

        # Setup model
        if config.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(model, device_ids=config.gpu_ids)
            logger.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
        else:
            self.model = model
            logger.info(f"Single device: {self.device}")

        self.model = self.model.to(self.device)
        self.dataloaders = dataloaders
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = self._create_scheduler()

        # Model Registry Directory
        self.registry_dir = Path(config.model_registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb_dir = (
            Path(config.tensorboard_log_dir) / config.dataset_name / model_name
        )
        self.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.tb_dir))

        # Initialize MLflow manager
        self.mlflow_manager = get_mlflow_manager(
            experiment_name=config.mlflow_experiment_name,
            enable_system_metrics=True,
            enable_autolog=True,
        )

        # 🔥 NEW: Initialize Memory Manager
        self.memory_manager = get_memory_manager()
        self.memory_manager.start()

        self.registered_models = {}  # Track registered models for alias management

        # 🔥 NEW: Initialize callbacks with trainer reference
        self._setup_callbacks()

        logger.info(f"Model registry: {self.registry_dir}")
        logger.info(f"TensorBoard: {self.tb_dir}")
        logger.info(f"Callbacks: {len(self.callbacks)} registered")
        if self.mlflow_run_id:
            logger.info(f"MLflow run ID to continue: {self.mlflow_run_id}")

    # 🔥 NEW: Setup callbacks method
    def _setup_callbacks(self):
        """Setup and initialize callbacks with trainer reference."""
        for callback in self.callbacks:
            if hasattr(callback, "set_trainer"):
                callback.set_trainer(self)
            logger.debug(f"Registered callback: {callback.__class__.__name__}")

    def _setup_device(self, requested_device: str) -> torch.device:
        """Setup device with fallback."""
        if requested_device == "mps":
            if torch.backends.mps.is_available():
                try:
                    test = torch.ones(1, device="mps")
                    del test
                    logger.info("MPS device initialized successfully")
                    return torch.device("mps")
                except Exception as e:
                    logger.warning(
                        f"MPS initialization failed: {e}, falling back to CPU"
                    )
                    return torch.device("cpu")
            else:
                logger.warning("MPS not available, falling back to CPU")
                return torch.device("cpu")
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                logger.info(f"CUDA device initialized: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda")
            else:
                logger.warning("CUDA not available, falling back to CPU")
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma,
            )
        elif self.config.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
            )
        elif self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.lr_T_max,
            )
        return None

    def load_checkpoint(self, checkpoint_path: str) -> tuple[int, str | None]:
        """Load checkpoint with full state restoration."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        target_model = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )

        try:
            target_model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model state loaded successfully")
        except RuntimeError as e:
            logger.warning(f"Model state mismatch: {e}, loading with strict=False")
            target_model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded successfully")
        except ValueError as e:
            logger.warning(
                f"Optimizer state mismatch: {e}, using fresh optimizer state"
            )

        if (
            self.scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("Scheduler state loaded successfully")
            except ValueError as e:
                logger.warning(
                    f"Scheduler state mismatch: {e}, using fresh scheduler state"
                )

        mlflow_run_id = checkpoint.get("mlflow_run_id")
        if mlflow_run_id:
            logger.info(f"Found MLflow run ID in checkpoint: {mlflow_run_id}")
        else:
            logger.info("No MLflow run ID found in checkpoint (will create new run)")

        logger.info(
            f"Resumed from epoch {checkpoint['epoch']} with val_loss={checkpoint.get('val_loss', 'N/A')}"
        )
        return checkpoint["epoch"], mlflow_run_id

    # ============================================================
    # 🔥 NEW: UNIFIED STEP EXECUTION
    # ============================================================

    def _execute_step(
        self, x: torch.Tensor, y: torch.Tensor, is_train: bool
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Unified step execution for both training and validation.

        Returns:
            loss: The loss tensor
            components: Dictionary of loss components (if ComboLoss)
            outputs: Model outputs
        """
        x = x.to(self.device, non_blocking=True).contiguous()
        y = y.to(self.device, non_blocking=True).contiguous()

        if is_train:
            self.optimizer.zero_grad()

        outputs = self.model(x)

        # Handle loss and components cleanly
        if isinstance(self.criterion, ComboLoss):
            loss, components = self.criterion(outputs, y, return_components=True)
        else:
            loss = self.criterion(outputs, y)
            components = {}

        if is_train:
            loss.backward()
            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_value,
                )
            self.optimizer.step()

        return loss, components, outputs

    # ============================================================
    # TRAINING EPOCH (Uses unified step execution)
    # ============================================================

    def train_epoch(self, verbose: bool = False) -> tuple[float, dict]:
        """
        Run one training epoch using unified step execution.
        """
        self.model.train()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3, device=self.device)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        loss_components = {
            "total": 0.0,
            "ce": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "ce_focal_combined": 0.0,
            "per_class": {},
        }
        num_batches = 0
        pbar = tqdm(self.dataloaders["train"], desc="Training")

        for batch_idx, (x, y) in enumerate(pbar):
            # 🔥 Call batch start callback
            for callback in self.callbacks:
                callback.on_batch_start(batch_idx)

            # Execute step
            loss, components, outputs = self._execute_step(x, y, is_train=True)

            total_loss += loss.item()
            num_batches += 1

            # Accumulate component losses
            if components:
                loss_components["total"] += components.get("total", 0)
                loss_components["ce"] += components.get("ce", 0)
                loss_components["focal"] += components.get("focal", 0)
                loss_components["dice"] += components.get("dice", 0)
                loss_components["ce_focal_combined"] += components.get(
                    "ce_focal_combined", 0
                )
                for class_name, class_loss in components.get("per_class", {}).items():
                    if class_name not in loss_components["per_class"]:
                        loss_components["per_class"][class_name] = 0.0
                    loss_components["per_class"][class_name] += class_loss

            # Update segmentation metrics
            preds = torch.argmax(outputs, dim=1)
            seg_metrics.update(preds, y)

            # 🔥 Call batch end callback
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item())

            if verbose and batch_idx % 10 == 0:
                logger.debug(
                    f"Batch {batch_idx}/{len(self.dataloaders['train'])} - Loss: {loss.item():.4f}"
                )

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.dataloaders["train"])
        metrics = seg_metrics.compute()

        if num_batches > 0 and components:
            for key in ["total", "ce", "focal", "dice", "ce_focal_combined"]:
                metrics[f"loss_{key}"] = loss_components[key] / num_batches
            for class_name, class_loss in loss_components["per_class"].items():
                metrics[f"loss_{class_name}"] = class_loss / num_batches

        return avg_loss, metrics

    # ============================================================
    # VALIDATION (Uses unified step execution)
    # ============================================================

    @torch.no_grad()
    def validate(self, verbose: bool = False) -> tuple[float, dict]:
        """
        Run validation using unified step execution.
        """
        self.model.eval()
        total_loss = 0.0
        seg_metrics = SegmentationMetrics(num_classes=3, device=self.device)

        if self.device.type == "mps":
            torch.mps.empty_cache()

        loss_components = {
            "total": 0.0,
            "ce": 0.0,
            "focal": 0.0,
            "dice": 0.0,
            "ce_focal_combined": 0.0,
            "per_class": {},
        }
        num_batches = 0
        pbar = tqdm(self.dataloaders["val"], desc="Validation")

        for batch_idx, (x, y) in enumerate(pbar):
            # 🔥 Call batch start callback
            for callback in self.callbacks:
                callback.on_batch_start(batch_idx)

            # Execute step (validation)
            loss, components, outputs = self._execute_step(x, y, is_train=False)

            total_loss += loss.item()
            num_batches += 1

            # Accumulate component losses
            if components:
                loss_components["total"] += components.get("total", 0)
                loss_components["ce"] += components.get("ce", 0)
                loss_components["focal"] += components.get("focal", 0)
                loss_components["dice"] += components.get("dice", 0)
                loss_components["ce_focal_combined"] += components.get(
                    "ce_focal_combined", 0
                )
                for class_name, class_loss in components.get("per_class", {}).items():
                    if class_name not in loss_components["per_class"]:
                        loss_components["per_class"][class_name] = 0.0
                    loss_components["per_class"][class_name] += class_loss

            # Update segmentation metrics
            preds = torch.argmax(outputs, dim=1)
            seg_metrics.update(preds, y)

            # 🔥 Call batch end callback
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, loss.item())

        avg_loss = total_loss / len(self.dataloaders["val"])
        metrics = seg_metrics.compute()

        if num_batches > 0 and components:
            for key in ["total", "ce", "focal", "dice", "ce_focal_combined"]:
                metrics[f"val_loss_{key}"] = loss_components[key] / num_batches
            for class_name, class_loss in loss_components["per_class"].items():
                metrics[f"val_loss_{class_name}"] = class_loss / num_batches

        return avg_loss, metrics

    # ============================================================
    # REMAINING METHODS (warmup, logging, checkpoint, fit, etc.)
    # ============================================================

    def _warmup_mps(self):
        """Warm up MPS shaders to avoid JIT compilation delay during training."""
        if self.device.type != "mps":
            return

        logger.info("🔥 Warming up MPS shaders (first pass can take 2-10 minutes)...")
        dummy_x = torch.randn(1, 1, 1578, 751, device=self.device)
        dummy_y = torch.randint(0, 3, (1, 1578, 751), device=self.device)

        dummy_out = self.model(dummy_x)
        dummy_loss = self.criterion(dummy_out, dummy_y)
        dummy_loss.backward()
        torch.mps.synchronize()
        self.model.zero_grad()
        torch.mps.empty_cache()
        logger.info("✅ MPS warmup complete!")

    def _start_new_mlflow_run(self, config_dict):
        """Start a new MLflow run with descriptive name."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.model_name}-{self.config.dataset_name}-{timestamp}"

        self.mlflow_manager.start_run(
            config_dict=config_dict,
            run_name=run_name,
            tags={
                "dataset": self.config.dataset_name,
                "model_type": self.model_name,
                "device": str(self.device),
                "experiment_type": "training",
                "run_type": "new",
            },
        )

    # ============================================================
    # 🔥 UPDATED: fit() method with callback integration
    # ============================================================

    def fit(self, resume_from: str | None = None, verbose: bool = False):
        """Main training loop with integrated logging and callbacks."""
        # Setup
        start_epoch = 0
        mlflow_run_id_from_checkpoint = None

        if resume_from and os.path.exists(resume_from):
            start_epoch, mlflow_run_id_from_checkpoint = self.load_checkpoint(
                resume_from
            )
            if mlflow_run_id_from_checkpoint:
                self.mlflow_run_id = mlflow_run_id_from_checkpoint

        # Start/continue MLflow run
        config_dict = self.config.to_dict()
        config_dict["model_name"] = self.model_name

        if self.mlflow_run_id:
            try:
                self.mlflow_manager.set_run(self.mlflow_run_id)
                self.mlflow_manager.set_tag("resumed", "True")
                self.mlflow_manager.set_tag("resumed_from_epoch", str(start_epoch))
            except Exception as e:
                logger.warning(f"Could not continue MLflow run: {e}")
                self._start_new_mlflow_run(config_dict)
        else:
            self._start_new_mlflow_run(config_dict)

        run_id = self.mlflow_manager.run_id
        logger.info(f"Starting training from epoch {start_epoch}")
        logger.info(f"MLflow Run ID: {run_id}")

        # ✅ Wrap the entire training loop in try/finally for cleanup
        try:
            # Call epoch start callbacks for initial state
            for callback in self.callbacks:
                callback.on_epoch_start(start_epoch, model=self.model)

            # Warmup
            self._warmup_mps()

            # Training loop
            best_val_loss = float("inf")
            best_val_iou = 0.0
            patience_counter = 0
            epoch_times = []

            for epoch in range(start_epoch, self.config.n_epochs):
                epoch_start = datetime.now(timezone.utc)

                # Call epoch start callbacks
                for callback in self.callbacks:
                    callback.on_epoch_start(epoch, model=self.model)

                # Train
                train_loss, train_metrics = self.train_epoch(verbose=verbose)
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.n_epochs} - Train Loss: {train_loss:.4f}"
                )
                logger.info(
                    f"  Train IoU: {train_metrics['mean_iou']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
                )

                # Validate
                val_loss, val_metrics = self.validate(verbose=verbose)
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.n_epochs} - Val Loss: {val_loss:.4f}"
                )
                logger.info(
                    f"  Val IoU: {val_metrics['mean_iou']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                )

                # Update scheduler
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]["lr"]

                # Prepare metrics dict for callbacks
                epoch_duration = (
                    datetime.now(timezone.utc) - epoch_start
                ).total_seconds()
                epoch_times.append(epoch_duration)

                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_iou": train_metrics["mean_iou"],
                    "val_iou": val_metrics["mean_iou"],
                    "train_f1": train_metrics["mean_f1"],
                    "val_f1": val_metrics["mean_f1"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
                    "lr": current_lr,
                    "epoch_duration": epoch_duration,  # ✅ Add epoch duration to MLflow
                    **train_metrics,
                    **val_metrics,
                }

                # Call epoch end callbacks (handles checkpointing, early stopping, logging)
                should_stop = False
                for callback in self.callbacks:
                    callback.on_epoch_end(
                        epoch,
                        metrics,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                    )
                    # Check if any callback wants to stop
                    if hasattr(callback, "should_stop") and callback.should_stop:
                        should_stop = True

                # Check early stopping from callbacks
                if should_stop:
                    logger.info(f"Training stopped by callback at epoch {epoch + 1}")
                    break

                # Training time tracking
                logger.info(f"⏱ Epoch duration: {epoch_duration:.1f}s")

        finally:
            # ✅ Ensure cleanup always runs, even on exception
            logger.info("Performing final cleanup...")

            # Cleanup TensorBoard
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.close()
                logger.debug("TensorBoard writer closed")

            # Cleanup MLflow
            if hasattr(self, "mlflow_manager") and self.mlflow_manager is not None:
                self.mlflow_manager.end_run()
                logger.debug("MLflow run ended")

            # Cleanup Memory Manager
            if hasattr(self, "memory_manager") and self.memory_manager is not None:
                self.memory_manager.stop()
                logger.debug("Memory manager stopped")

        # Log final summary after cleanup
        total_time = sum(epoch_times) if epoch_times else 0
        avg_time = total_time / len(epoch_times) if epoch_times else 0

        logger.info(f"Total training time: {total_time:.1f}s")
        logger.info(f"Average epoch time: {avg_time:.1f}s")
        logger.info("=" * 60)
