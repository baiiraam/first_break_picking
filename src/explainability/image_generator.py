# file location: src/explainability/image_generator.py

"""
Image generation for explainability figures.

Produces 4-panel PNGs:
    - Panel 1: seismogram
    - Panel 2: ground-truth mask
    - Panel 3: predicted mask
    - Panel 4: seismogram + Grad-CAM heatmap overlay
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# Mask colormap: 0 = before (blue), 1 = after (green), 2 = strip (red)
MASK_CMAP = ListedColormap(["#1f77b4", "#2ca02c", "#d62728"])
MASK_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


class ExplainabilityImageGenerator:
    """
    Generates 4-panel explainability PNGs.
    """

    def __init__(self, output_dir: Path, logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def generate_for_shot(
        self,
        shot_id: int,
        shot_data: np.ndarray,     # (n_traces, n_samples) float
        gt_mask: np.ndarray,       # (n_traces, n_samples) int
        pred_mask: np.ndarray,     # (n_traces, n_samples) int
        heatmap: np.ndarray,       # (n_traces, n_samples) float in [0,1]
        label: str,                # "best" | "median" | "worst"
        mean_error: float,
        dataset_name: str,
    ) -> Path:
        """Generate the 4-panel figure for one shot. Returns the path."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: seismogram
        ax = axes[0, 0]
        v = np.percentile(np.abs(shot_data), 95)
        if v == 0:
            v = 1.0
        ax.imshow(
            shot_data.T, cmap="seismic", aspect="auto",
            vmin=-v, vmax=v,
        )
        ax.set_title(f"Seismogram — shot {shot_id}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 2: ground-truth mask
        ax = axes[0, 1]
        ax.imshow(
            gt_mask.T, cmap=MASK_CMAP, norm=MASK_NORM, aspect="auto"
        )
        ax.set_title("Ground truth mask")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 3: predicted mask
        ax = axes[1, 0]
        ax.imshow(
            pred_mask.T, cmap=MASK_CMAP, norm=MASK_NORM, aspect="auto"
        )
        ax.set_title("Predicted mask")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 4: seismogram + Grad-CAM overlay
        ax = axes[1, 1]
        v = np.percentile(np.abs(shot_data), 95)
        if v == 0:
            v = 1.0
        ax.imshow(
            shot_data.T, cmap="gray", aspect="auto",
            vmin=-v, vmax=v,
        )
        # Overlay heatmap. Use 'jet' colormap.
        ax.imshow(
            heatmap.T, cmap="jet", aspect="auto",
            alpha=0.5, vmin=0.0, vmax=1.0,
        )
        ax.set_title("Seismogram + Grad-CAM (class 2 / strip)")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        fig.suptitle(
            f"[{dataset_name}] {label.upper()} shot — id={shot_id}, "
            f"mean error={mean_error:.2f} samples",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_path = self.output_dir / f"shot_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"  Generated: {out_path.name}")
        return out_path