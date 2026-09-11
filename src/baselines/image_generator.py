# file location: src/baselines/image_generator.py

"""
Evaluation image generation for classical baseline pickers.

Produces 5 PNGs per evaluation:
    - shot_best.png             3-panel (seismogram, picks overlay, error bars)
    - shot_median.png           same layout
    - shot_worst.png            same layout
    - error_histogram.png       log-scale histogram of per-trace absolute error
    - error_vs_position.png     scatter of error vs. ground-truth pick position

Pure matplotlib. Does not log to MLflow.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

# ============================================================
# CONSTANTS
# ============================================================

# Error magnitude colormap for the per-trace bar chart
ERROR_CMAP = ListedColormap(["#2ca02c", "#ff7f0e", "#d62728"])
#                                     green    orange     red
ERROR_NORM = BoundaryNorm([-0.5, 2.5, 19.5, 10000.0], 3)


# ============================================================
# GENERATOR
# ============================================================


class BaselineImageGenerator:
    """
    Generates the 5 evaluation PNGs for a baseline picker.

    Usage:
        gen = BaselineImageGenerator(Path("out"), logger)
        paths = gen.generate_all(result=result, shots=shots,
                                 dataset_name="Halfmile")
    """

    def __init__(self, output_dir: Path, logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def generate_all(
        self,
        result: dict[str, Any],
        shots: list[tuple[int, np.ndarray, np.ndarray]],
        dataset_name: str,
    ) -> list[Path]:
        """
        Generate all 5 images. Returns list of paths written.
        """
        paths: list[Path] = []
        per_trace: pd.DataFrame = result["per_trace"]

        if len(per_trace) == 0:
            self.logger.warning(
                "[ImageGen] No per-trace data — cannot generate any images."
            )
            return paths

        # --- Histogram ---
        hist_path = self._plot_error_histogram(per_trace["error_samples"].to_numpy())
        paths.append(hist_path)

        # --- Error vs. position ---
        pos_path = self._plot_error_vs_position(per_trace)
        if pos_path is not None:
            paths.append(pos_path)

        # --- Best / median / worst shots ---
        per_shot = per_trace.groupby("shot_id")["error_samples"].mean().sort_values()
        if len(per_shot) == 0:
            return paths

        n = len(per_shot)
        selections = [
            ("best", int(per_shot.index[0])),
            ("median", int(per_shot.index[n // 2])),
            ("worst", int(per_shot.index[-1])),
        ]

        # Build lookup from shot_id → (shot_data, gt_picks, pred_picks)
        # We need predictions, which aren't in the result dict.
        # Re-run the picker here — cheap for 3 shots.
        shots_by_id = {sid: (data, gt) for sid, data, gt in shots}

        for label, shot_id in selections:
            if shot_id not in shots_by_id:
                self.logger.warning(
                    f"[ImageGen] Shot {shot_id} not found for label '{label}'"
                )
                continue

            shot_data, gt_picks = shots_by_id[shot_id]
            mean_err = float(per_shot.loc[shot_id])

            path = self._plot_shot_comparison(
                shot_id=shot_id,
                shot_data=shot_data,
                gt_picks=gt_picks,
                per_trace=per_trace,
                label=label,
                mean_error=mean_err,
                dataset_name=dataset_name,
            )
            if path is not None:
                paths.append(path)

        self.logger.info(f"[ImageGen] Generated {len(paths)} images")
        return paths

    # ---------------------------------------------------------
    # SHOT COMPARISON (3-panel)
    # ---------------------------------------------------------

    def _plot_shot_comparison(
        self,
        shot_id: int,
        shot_data: np.ndarray,
        gt_picks: np.ndarray,
        per_trace: pd.DataFrame,
        label: str,
        mean_error: float,
        dataset_name: str,
    ) -> Path | None:
        """Generate a 3-panel comparison figure for a single shot."""
        # Filter per_trace to this shot
        sub = per_trace[per_trace["shot_id"] == shot_id]
        if len(sub) == 0:
            self.logger.warning(f"[ImageGen] No per-trace data for shot {shot_id}")
            return None

        # We need the predicted picks. Reconstruct from the per_trace df,
        # but note: only traces with both GT and pred valid are in the df.
        # Traces where pred=0 are missing.
        # To show ALL predictions, we need to re-run the picker. That
        # would require the picker instance, which we don't have here.
        # Instead, we plot what we have: valid comparisons only.
        # Traces without a valid comparison are shown without markers.
        #
        # Note: this means "worst shot" may have many traces with no
        # markers, which is itself informative.

        trace_idx = sub["trace_index"].to_numpy()
        gt_at = sub["gt_pick_sample"].to_numpy()
        pred_at = sub["pred_pick_sample"].to_numpy()
        errors = sub["error_samples"].to_numpy()

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # Panel 1: seismogram
        ax = axes[0]
        v = np.percentile(np.abs(shot_data), 95)
        if v == 0:
            v = 1.0
        ax.imshow(
            shot_data.T,
            cmap="seismic",
            aspect="auto",
            vmin=-v,
            vmax=v,
            extent=(0, shot_data.shape[0], shot_data.shape[1], 0),
        )
        ax.set_title(f"Seismogram — shot {shot_id}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 2: seismogram + picks overlay
        ax = axes[1]
        ax.imshow(
            shot_data.T,
            cmap="gray",
            aspect="auto",
            extent=(0, shot_data.shape[0], shot_data.shape[1], 0),
            alpha=0.5,
        )
        ax.scatter(
            trace_idx,
            gt_at,
            marker="o",
            s=8,
            color="#2ca02c",
            label="Ground truth",
            alpha=0.7,
        )
        ax.scatter(
            trace_idx,
            pred_at,
            marker="x",
            s=20,
            color="#d62728",
            label="STA/LTA",
            alpha=0.9,
        )
        ax.set_title(f"Picks: GT (green o) vs STA/LTA (red x) — shot {shot_id}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        ax.legend(loc="upper right")
        # Match the seismogram's y-direction
        ax.set_ylim(shot_data.shape[1], 0)

        # Panel 3: per-trace error bars
        ax = axes[2]
        ax.bar(
            trace_idx,
            errors,
            color=[ERROR_CMAP(ERROR_NORM(e)) for e in errors],
            width=1.0,
        )
        ax.set_title("Per-trace absolute error (green <3, orange 3-20, red >20)")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Error (samples)")
        ax.set_xlim(0, shot_data.shape[0])
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"[{dataset_name}] {label.upper()} shot — id={shot_id}, "
            f"mean error={mean_error:.2f} samples, "
            f"valid traces={len(sub)}/{shot_data.shape[0]}",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_path = self.output_dir / f"shot_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------
    # ERROR HISTOGRAM
    # ---------------------------------------------------------

    def _plot_error_histogram(self, errors: np.ndarray) -> Path:
        p99 = float(np.percentile(errors, 99))
        upper = max(50.0, min(p99 * 1.1, 500.0))

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, upper, 50)
        ax.hist(errors, bins=bins, color="#2ca02c", alpha=0.75, edgecolor="black")
        ax.set_yscale("log")
        ax.set_xlabel("Absolute error (samples)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title("Per-trace absolute error distribution")
        ax.axvline(
            3.0, color="red", linestyle="--", linewidth=1.5, label="±3 sample tolerance"
        )

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
        out_path = self.output_dir / "error_histogram.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------
    # ERROR VS POSITION
    # ---------------------------------------------------------

    def _plot_error_vs_position(self, per_trace: pd.DataFrame) -> Path | None:
        if len(per_trace) == 0:
            return None

        pick_pos = per_trace["gt_pick_sample"].to_numpy()
        errors = per_trace["error_samples"].to_numpy()
        y = np.clip(errors, 1.0, None)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(pick_pos, y, s=2, alpha=0.15, color="#1f77b4", rasterized=True)
        ax.set_yscale("log")
        ax.set_xlabel("Ground-truth pick position (samples)")
        ax.set_ylabel("Absolute error (samples, log scale)")
        ax.set_title("Error vs. pick position")
        ax.axhline(
            3.0, color="red", linestyle="--", linewidth=1.5, label="±3 sample tolerance"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        out_path = self.output_dir / "error_vs_position.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
