# file location: src/evaluation/image_generator.py

"""
Evaluation image generation for seismic FBP.

Produces 5 PNGs per split:
    - shot_best.png             4-panel (seismogram, GT, prediction, overlay)
    - shot_median.png           same layout
    - shot_worst.png            same layout
    - error_histogram.png       log-scale histogram of per-trace absolute error
    - error_vs_position.png     scatter of error vs. ground-truth pick position

All figures are matplotlib. No TensorBoard dependency.

The generator re-runs the model on the 3 selected representative shots
rather than retaining predictions for every shot in memory.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless backend — required for script use

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.config import SeismicConfig
from src.types import LoggerType

# ============================================================
# COLORMAPS
# ============================================================

# Mask classes: 0 = before, 1 = after, 2 = strip
MASK_CMAP = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])  # blue, green, red
MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

# Overlay classes: 0 = none, 1 = GT strip only (missed), 2 = pred strip only
# (false), 3 = both agree on strip
OVERLAY_CMAP = ListedColormap(["#000000", "#d62728", "#1f77b4", "#2ca02c"])
#                                                red=missed  blue=false  green=agree
OVERLAY_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)


# ============================================================
# GENERATOR
# ============================================================


class EvaluationImageGenerator:
    """
    Generates the 5 evaluation PNGs for a single split.

    Usage:
        gen = EvaluationImageGenerator(Path("out"), logger)
        paths = gen.generate_for_split(
            split_name="test",
            shot_errors=df,           # DataFrame with shot_id, error_samples
            model=model,
            device=device,
            data_manager=data_manager,
            cfg=cfg,
        )
    """

    def __init__(self, output_dir: Path, logger: LoggerType):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def generate_for_split(
        self,
        split_name: str,
        shot_errors: pd.DataFrame,
        model: torch.nn.Module,
        device: torch.device,
        data_manager: Any,
        cfg: SeismicConfig,
    ) -> list[Path]:
        """
        Generate the 5 PNGs for a split.

        Args:
            split_name: "train" | "val" | "test"
            shot_errors: DataFrame with at least columns shot_id, error_samples.
                If empty or missing shot_id, only the histogram figure is
                produced using whatever error_samples are available.
            model: PyTorch model (already loaded, eval mode not required)
            device: torch device
            data_manager: ChunkedDataManager (has get_dataset(split))
            cfg: SeismicConfig

        Returns:
            List of paths to the PNGs written.
        """
        paths: list[Path] = []

        # Normalize input: we expect a DataFrame with shot_id and error_samples
        if shot_errors is None or len(shot_errors) == 0:
            self.logger.warning(
                f"[ImageGen] No per-trace errors for split '{split_name}' — "
                f"skipping shot/error figures."
            )
            return paths

        # Filter to finite, non-negative errors
        df = shot_errors.copy()
        if "error_samples" not in df.columns:
            self.logger.warning(
                f"[ImageGen] DataFrame for '{split_name}' has no "
                f"'error_samples' column — skipping."
            )
            return paths

        df = df[np.isfinite(df["error_samples"])]
        df = df[df["error_samples"] >= 0]
        if len(df) == 0:
            self.logger.warning(
                f"[ImageGen] No finite errors for split '{split_name}' — skipping."
            )
            return paths

        # Error histogram
        hist_path = self._plot_error_histogram(
            df["error_samples"].to_numpy(), split_name
        )
        paths.append(hist_path)

        # Error vs. position (needs shot_id and a way to look up pick position)
        pos_path = self._plot_error_vs_position(df, split_name, data_manager, cfg)
        if pos_path is not None:
            paths.append(pos_path)

        # Shot comparisons require shot_id column
        if "shot_id" not in df.columns:
            self.logger.warning(
                f"[ImageGen] No 'shot_id' column for split '{split_name}' — "
                f"skipping shot comparison figures."
            )
            return paths

        # Select 3 representative shots (by mean error per shot)
        per_shot = df.groupby("shot_id")["error_samples"].mean().sort_values()
        if len(per_shot) == 0:
            return paths

        n = len(per_shot)
        best_id = int(per_shot.index[0])
        median_id = int(per_shot.index[n // 2])
        worst_id = int(per_shot.index[-1])

        selections = [
            ("best", best_id),
            ("median", median_id),
            ("worst", worst_id),
        ]

        for label, shot_id in selections:
            path = self._plot_shot_comparison(
                split_name=split_name,
                label=label,
                shot_id=shot_id,
                mean_error=float(per_shot.loc[shot_id]),
                model=model,
                device=device,
                data_manager=data_manager,
                cfg=cfg,
            )
            if path is not None:
                paths.append(path)

        self.logger.info(
            f"[ImageGen] Generated {len(paths)} images for split '{split_name}'"
        )
        return paths

    # ---------------------------------------------------------
    # SHOT COMPARISON (4-panel)
    # ---------------------------------------------------------

    def _plot_shot_comparison(
        self,
        split_name: str,
        label: str,
        shot_id: int,
        mean_error: float,
        model: torch.nn.Module,
        device: torch.device,
        data_manager: Any,
        cfg: SeismicConfig,
    ) -> Path | None:
        """Generate a 4-panel comparison figure for a single shot."""
        try:
            seismogram, gt_mask, pred_mask = self._load_and_infer_shot(
                shot_id, model, device, data_manager, split_name
            )
        except (KeyError, IndexError, RuntimeError, ValueError) as e:
            self.logger.warning(
                f"[ImageGen] Could not generate shot comparison for "
                f"split='{split_name}' shot_id={shot_id}: {e}"
            )
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: seismogram
        ax = axes[0, 0]
        v = np.percentile(np.abs(seismogram), 95)
        if v == 0:
            v = 1.0
        ax.imshow(seismogram.T, cmap="seismic", aspect="auto", vmin=-v, vmax=v)
        ax.set_title(f"Seismogram — shot {shot_id}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 2: ground-truth mask
        ax = axes[0, 1]
        ax.imshow(gt_mask.T, cmap=MASK_CMAP, norm=MASK_NORM, aspect="auto")
        ax.set_title("Ground truth")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 3: prediction
        ax = axes[1, 0]
        ax.imshow(pred_mask.T, cmap=MASK_CMAP, norm=MASK_NORM, aspect="auto")
        ax.set_title("Prediction")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 4: overlay diff
        ax = axes[1, 1]
        overlay = self._build_overlay(gt_mask, pred_mask)
        ax.imshow(overlay.T, cmap=OVERLAY_CMAP, norm=OVERLAY_NORM, aspect="auto")
        ax.set_title("Overlay: green=agree, red=missed, blue=false")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        fig.suptitle(
            f"[{split_name}] {label.upper()} shot — id={shot_id}, "
            f"mean error={mean_error:.2f} samples",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_path = self.output_dir / f"{split_name}_shot_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------
    # ERROR HISTOGRAM
    # ---------------------------------------------------------

    def _plot_error_histogram(self, errors: np.ndarray, split_name: str) -> Path:
        """Log-scale histogram of per-trace absolute error."""
        p99 = float(np.percentile(errors, 99))
        upper = max(50.0, min(p99 * 1.1, 500.0))

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, upper, 50)
        ax.hist(errors, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="black")
        ax.set_yscale("log")
        ax.set_xlabel("Absolute error (samples)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(f"[{split_name}] Per-trace absolute error distribution")
        ax.axvline(
            3.0, color="red", linestyle="--", linewidth=1.5, label="±3 sample tolerance"
        )

        # Annotation box
        mae = float(errors.mean())
        med = float(np.median(errors))
        p90 = float(np.percentile(errors, 90))
        p95 = float(np.percentile(errors, 95))
        within3 = float((errors <= 3).mean()) * 100.0

        text = (
            f"MAE:     {mae:.2f}\n"
            f"Median:  {med:.2f}\n"
            f"p90:     {p90:.2f}\n"
            f"p95:     {p95:.2f}\n"
            f"±3 acc:  {within3:.1f}%"
        )
        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = self.output_dir / f"{split_name}_error_histogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------
    # ERROR VS POSITION
    # ---------------------------------------------------------

    def _plot_error_vs_position(
        self,
        df: pd.DataFrame,
        split_name: str,
        data_manager: Any,
        cfg: SeismicConfig,
    ) -> Path | None:
        """
        Scatter of error vs. ground-truth pick position.

        Needs to know the pick position for each trace. If df already
        has a 'pick_sample' column, use it. Otherwise, we skip (we don't
        want to re-read the whole manifest just for this plot).
        """
        if "pick_sample" not in df.columns:
            self.logger.info(
                f"[ImageGen] No 'pick_sample' column for split '{split_name}'; "
                f"skipping error-vs-position scatter."
            )
            return None

        pick_pos = df["pick_sample"].to_numpy()
        errors = df["error_samples"].to_numpy()

        # Clip lower bound for log scale
        y = np.clip(errors, 1.0, None)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pick_pos, y, s=2, alpha=0.15, color="#1f77b4", rasterized=True)
        ax.set_yscale("log")
        ax.set_xlabel("Ground-truth pick position (samples)")
        ax.set_ylabel("Absolute error (samples, log scale)")
        ax.set_title(f"[{split_name}] Error vs. pick position")
        ax.axhline(
            3.0, color="red", linestyle="--", linewidth=1.5, label="±3 sample tolerance"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        out_path = self.output_dir / f"{split_name}_error_vs_position.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------

    def _load_and_infer_shot(
        self,
        shot_id: int,
        model: torch.nn.Module,
        device: torch.device,
        data_manager: Any,
        split_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single shot from the split's dataset, run inference,
        return (seismogram, gt_mask, pred_mask) as numpy arrays.
        """
        dataset = data_manager.get_dataset(split_name)

        # Find the dataset index for this shot_id
        idx = None
        for i in range(len(dataset)):
            if dataset.get_shot_id(i) == shot_id:
                idx = i
                break

        if idx is None:
            raise KeyError(f"shot_id {shot_id} not found in split '{split_name}'")

        x, y = dataset[idx]  # x: (1, H, W) float; y: (H, W) int

        seismogram = x.squeeze(0).cpu().numpy()  # (H, W)
        gt_mask = y.cpu().numpy()  # (H, W)

        with torch.no_grad():
            model.eval()
            xb = x.unsqueeze(0).to(device)  # (1, 1, H, W)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H, W)

        return seismogram, gt_mask, pred

    def _build_overlay(self, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        """
        Build overlay:
            0 = neither GT nor pred strip
            1 = GT strip only (missed pick — red)
            2 = pred strip only (false pick — blue)
            3 = both GT and pred strip (agree — green)
        """
        gt_strip = gt_mask == 2
        pred_strip = pred_mask == 2

        overlay = np.zeros_like(gt_mask, dtype=np.int8)
        overlay[gt_strip & ~pred_strip] = 1
        overlay[~gt_strip & pred_strip] = 2
        overlay[gt_strip & pred_strip] = 3
        return overlay
