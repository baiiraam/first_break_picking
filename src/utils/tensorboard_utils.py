"""
TensorBoard utilities with bounded thread pool for asynchronous visualization.
"""

import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """
    ThreadPoolExecutor with a bounded work queue to prevent memory overflow.

    IMPORTANT: The queue must be reassigned AFTER super().__init__() because
    ThreadPoolExecutor unconditionally creates its own unbounded queue.
    """

    def __init__(self, max_workers: int = 1, max_queue_size: int = 5, *args, **kwargs):
        # ✅ Call parent first - it creates self._work_queue
        super().__init__(max_workers=max_workers, *args, **kwargs)

        # ✅ Reassign AFTER super().__init__() so it isn't overwritten
        self._work_queue = queue.Queue(max_queue_size)
        self._max_queue_size = max_queue_size

        logger.debug(
            f"[TensorBoard] BoundedThreadPool: max_workers={max_workers}, "
            f"max_queue_size={max_queue_size}"
        )

    def submit(self, fn, *args, **kwargs):
        """
        Submit a task with bounded queue.
        Blocks if queue is full to prevent memory overflow.
        """
        try:
            return super().submit(fn, *args, **kwargs)
        except queue.Full:
            logger.warning(
                "[TensorBoard] Visualization queue full. Dropping frame to protect memory."
            )
            return None


class TensorBoardManager:
    """
    Manages TensorBoard logging with asynchronous seismogram visualizations.
    Uses bounded thread pool to prevent memory overflow.
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        flush_secs: int = 10,
        async_viz: bool = True,
        max_viz_workers: int = 1,
        max_queue_size: int = 5,
    ):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)
        self.step = 0
        self.async_viz = async_viz

        # ✅ Bounded thread pool for non-blocking visualizations
        if async_viz:
            self._viz_executor = BoundedThreadPoolExecutor(
                max_workers=max_viz_workers,
                max_queue_size=max_queue_size,
                thread_name_prefix="tb_viz",
            )
            logger.info(
                f"TensorBoard logging to: {self.log_dir} "
                f"(async visualization, max_queue={max_queue_size})"
            )
        else:
            self._viz_executor = None
            logger.info(f"TensorBoard logging to: {self.log_dir} (sync visualization)")

    def set_step(self, step: int):
        """Set current step for logging."""
        self.step = step

    def log_scalar(self, tag: str, value: float, step: int | None = None):
        """Log a scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict[str, float], step: int | None = None):
        """Log multiple scalars."""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: int | None = None,
        dataformats: str = "HWC",
    ):
        """Log an image."""
        if step is None:
            step = self.step
        self.writer.add_image(tag, image, step, dataformats=dataformats)

    def log_figure(self, tag: str, figure: plt.Figure, step: int | None = None):
        """Log a matplotlib figure."""
        if step is None:
            step = self.step
        self.writer.add_figure(tag, figure, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int | None = None):
        """Log a histogram."""
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)

    def log_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log the model graph."""
        self.writer.add_graph(model, input_tensor)

    def log_seismogram(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        predictions: np.ndarray | None,
        shot_id: int,
        step: int | None = None,
    ):
        """
        Log a seismogram with mask and predictions (synchronous version).
        """
        if step is None:
            step = self.step

        if predictions is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        # Original seismogram
        ax = axes[0]
        ax.imshow(
            data.T,
            cmap="seismic",
            aspect="auto",
            vmin=-np.percentile(np.abs(data), 95),
            vmax=np.percentile(np.abs(data), 95),
        )
        ax.set_title(f"Seismogram (Shot {shot_id})")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Ground truth mask
        ax = axes[1]
        ax.imshow(mask.T, cmap="tab10", aspect="auto", vmin=0, vmax=2)
        ax.set_title("Ground Truth")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Predictions (if available)
        if predictions is not None:
            ax = axes[2]
            ax.imshow(predictions.T, cmap="tab10", aspect="auto", vmin=0, vmax=2)
            ax.set_title("Predictions")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")

            # Overlay: prediction vs ground truth
            ax = axes[3]
            overlay = np.zeros_like(data)
            overlay[mask == 2] = 1
            overlay[predictions == 2] += 2
            ax.imshow(overlay.T, cmap="viridis", aspect="auto", vmin=0, vmax=3)
            ax.set_title("Overlay (GT=1, Pred=2, Both=3)")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")
        else:
            # Overlay mask on seismogram
            ax = axes[2]
            ax.imshow(data.T, cmap="gray", aspect="auto")
            strip_mask = (mask == 2).T
            ax.imshow(strip_mask, cmap="Reds", aspect="auto", alpha=0.4)
            ax.set_title("Seismogram + Strip")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Sample")

        plt.tight_layout()
        self.writer.add_figure(f"Seismogram/Shot_{shot_id}", fig, step)
        plt.close(fig)

    def log_seismogram_async(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        predictions: np.ndarray | None,
        shot_id: int,
        step: int | None = None,
    ):
        """
        Asynchronously submit seismogram figure rendering with bounded queue.
        """
        if not self.async_viz or self._viz_executor is None:
            self.log_seismogram(data, mask, predictions, shot_id, step)
            return

        if step is None:
            step = self.step

        # Copy arrays to avoid race conditions
        data_copy = data.copy()
        mask_copy = mask.copy()
        preds_copy = predictions.copy() if predictions is not None else None

        self._viz_executor.submit(
            self._render_and_log_seismogram,
            data_copy,
            mask_copy,
            preds_copy,
            shot_id,
            step,
        )

    def _render_and_log_seismogram(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        predictions: np.ndarray | None,
        shot_id: int,
        step: int,
    ):
        """Internal worker method executing the heavy Matplotlib plotting."""
        try:
            self.log_seismogram(data, mask, predictions, shot_id, step)
        except Exception as e:
            logger.error(f"Failed to render async seismogram for shot {shot_id}: {e}")

    def log_learning_rate(self, lr: float, step: int | None = None):
        """Log learning rate."""
        if step is None:
            step = self.step
        self.writer.add_scalar("Metrics/lr", lr, step)

    def log_loss_curves(
        self, train_losses: list[float], val_losses: list[float], step: int
    ):
        """Log loss curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_losses, label="Training Loss", color="blue")
        ax.plot(val_losses, label="Validation Loss", color="red")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self.writer.add_figure("Loss/Curves", fig, step)
        plt.close(fig)

    def log_weights_histograms(self, model: torch.nn.Module, step: int | None = None):
        """Log weight histograms for model layers."""
        if step is None:
            step = self.step

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"Weights/{name}", param.data.cpu().numpy(), step
                )

    def flush(self):
        """Flush all pending logs."""
        self.writer.flush()

    def close(self):
        """Close the writer and shutdown background threads."""
        if self._viz_executor is not None:
            logger.info("Shutting down TensorBoard visualization thread pool...")
            self._viz_executor.shutdown(wait=True)
            logger.info("Visualization thread pool shutdown complete")
        self.writer.close()
        logger.info("TensorBoard writer closed")
