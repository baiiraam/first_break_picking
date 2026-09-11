# file location: src/evaluation/comparison_image_generator.py

"""
Comparison image generation: ML picks vs STA/LTA picks.

Produces 6 PNGs:
    - shot_best.png              3-panel (seismogram, picks overlay, error bars)
    - shot_median.png            same layout
    - shot_worst.png             same layout
    - error_histogram.png        overlaid histograms, both methods
    - error_vs_position.png      overlaid scatters, both methods
    - summary_comparison.png     grouped bar chart of key metrics

All figures are matplotlib. No TensorBoard dependency.

Joint shot selection: best/median/worst by mean of (ML error, STA/LTA error).
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# COLORS
# ============================================================

COLOR_STA = "#d62728"   # red
COLOR_ML = "#1f77b4"    # blue
COLOR_GT = "#2ca02c"    # green


# ============================================================
# GENERATOR
# ============================================================

class ComparisonImageGenerator:
    """
    Generates the 6 comparison PNGs for ML vs STA/LTA.
    """

    def __init__(self, output_dir: Path, logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def generate_all(
        self,
        result: dict[str, Any],
        shots: list[tuple[int, np.ndarray, np.ndarray]],
        dataset_name: str,
    ) -> list[Path]:
        """Generate all 6 images. Returns list of paths written."""
        paths: list[Path] = []
        per_trace: pd.DataFrame = result["per_trace"]

        if len(per_trace) == 0:
            self.logger.warning(
                "[CompareImageGen] No per-trace data — skipping all figures."
            )
            return paths

        # --- Histogram ---
        paths.append(self._plot_error_histogram(per_trace))

        # --- Error vs. position ---
        paths.append(self._plot_error_vs_position(per_trace))

        # --- Summary bar chart ---
        paths.append(
            self._plot_summary_comparison(
                result["ml_metrics"], result["sta_lta_metrics"]
            )
        )

        # --- Best / median / worst shots (joint) ---
        per_shot = self._joint_error_per_shot(per_trace).sort_values()
        if len(per_shot) == 0:
            return paths

        n = len(per_shot)
        selections = [
            ("best", int(per_shot.index[0])),
            ("median", int(per_shot.index[n // 2])),
            ("worst", int(per_shot.index[-1])),
        ]

        shots_by_id = {sid: (data, gt) for sid, data, gt in shots}

        for label, shot_id in selections:
            if shot_id not in shots_by_id:
                self.logger.warning(
                    f"[CompareImageGen] Shot {shot_id} not in loaded shots"
                )
                continue
            shot_data, gt_picks = shots_by_id[shot_id]
            mean_err = float(per_shot.loc[shot_id])

            p = self._plot_shot_comparison(
                shot_id=shot_id,
                shot_data=shot_data,
                gt_picks=gt_picks,
                per_trace=per_trace,
                label=label,
                joint_mean_error=mean_err,
                dataset_name=dataset_name,
            )
            if p is not None:
                paths.append(p)

        self.logger.info(
            f"[CompareImageGen] Generated {len(paths)} images"
        )
        return paths

    # ---------------------------------------------------------------
    # JOINT SHOT SELECTION
    # ---------------------------------------------------------------

    def _joint_error_per_shot(self, per_trace: pd.DataFrame) -> pd.Series:
        """
        For each shot, compute the mean of (ML mean error, STA/LTA mean error).
        Only counts traces where BOTH methods produced a valid pick (>0).
        """
        valid = per_trace[
            (per_trace["sta_lta_error_samples"] >= 0)
            & (per_trace["ml_error_samples"] >= 0)
        ].copy()

        if len(valid) == 0:
            # Fallback: use whichever is available
            valid = per_trace.copy()

        ml_mean = valid.groupby("shot_id")["ml_error_samples"].mean()
        sta_mean = valid.groupby("shot_id")["sta_lta_error_samples"].mean()
        joint = (ml_mean + sta_mean) / 2.0
        return joint

    # ---------------------------------------------------------------
    # SHOT COMPARISON (3-panel)
    # ---------------------------------------------------------------

    def _plot_shot_comparison(
        self,
        shot_id: int,
        shot_data: np.ndarray,
        gt_picks: np.ndarray,
        per_trace: pd.DataFrame,
        label: str,
        joint_mean_error: float,
        dataset_name: str,
    ) -> Path | None:
        sub = per_trace[per_trace["shot_id"] == shot_id]
        if len(sub) == 0:
            return None

        trace_idx = sub["trace_index"].to_numpy()
        gt_at = sub["gt_pick_sample"].to_numpy()
        sta_at = sub["sta_lta_pick"].to_numpy()
        ml_at = sub["ml_pick"].to_numpy()
        sta_err = sub["sta_lta_error_samples"].to_numpy()
        ml_err = sub["ml_error_samples"].to_numpy()

        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # Panel 1 — seismogram
        ax = axes[0]
        v = np.percentile(np.abs(shot_data), 95)
        if v == 0:
            v = 1.0
        ax.imshow(
            shot_data.T, cmap="seismic", aspect="auto",
            vmin=-v, vmax=v,
            extent=(0, shot_data.shape[0], shot_data.shape[1], 0),
        )
        ax.set_title(f"Seismogram — shot {shot_id}")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")

        # Panel 2 — picks overlay
        ax = axes[1]
        ax.imshow(
            shot_data.T, cmap="gray", aspect="auto",
            extent=(0, shot_data.shape[0], shot_data.shape[1], 0),
            alpha=0.5,
        )
        # Filter to valid picks for markers
        gt_mask = gt_at > 0
        sta_mask = sta_at > 0
        ml_mask = ml_at > 0

        ax.scatter(
            trace_idx[gt_mask], gt_at[gt_mask],
            marker="o", s=10, color=COLOR_GT,
            label="Ground truth", alpha=0.8,
        )
        ax.scatter(
            trace_idx[sta_mask], sta_at[sta_mask],
            marker="x", s=30, color=COLOR_STA,
            label="STA/LTA", alpha=0.9,
        )
        ax.scatter(
            trace_idx[ml_mask], ml_at[ml_mask],
            marker="+", s=40, color=COLOR_ML,
            label="ML model", alpha=0.9,
        )
        ax.set_title(
            f"Picks: GT (green o), STA/LTA (red x), ML (blue +) — shot {shot_id}"
        )
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(shot_data.shape[1], 0)

        # Panel 3 — grouped error bars
        ax = axes[2]
        # Only plot traces where at least one method has a pick
        both = np.arange(len(trace_idx))
        width = 0.4
        ax.bar(
            both - width / 2, np.clip(sta_err, 0.5, None), width,
            color=COLOR_STA, alpha=0.75, label="STA/LTA error",
        )
        ax.bar(
            both + width / 2, np.clip(ml_err, 0.5, None), width,
            color=COLOR_ML, alpha=0.75, label="ML error",
        )
        ax.set_yscale("log")
        ax.set_title("Per-trace absolute error (log scale)")
        ax.set_xlabel("Trace (display order)")
        ax.set_ylabel("Error (samples)")
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1.0)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        # Per-shot summary in the suptitle
        gt_valid = gt_mask.sum()
        sta_ok = (sta_err[sta_mask] <= 3).sum() if sta_mask.any() else 0
        ml_ok = (ml_err[ml_mask] <= 3).sum() if ml_mask.any() else 0
        sta_mae = float(sta_err[sta_mask].mean()) if sta_mask.any() else float("nan")
        ml_mae = float(ml_err[ml_mask].mean()) if ml_mask.any() else float("nan")
        sta_pct = 100.0 * sta_ok / gt_valid if gt_valid > 0 else 0.0
        ml_pct = 100.0 * ml_ok / gt_valid if gt_valid > 0 else 0.0

        title = (
            f"[{dataset_name}] {label.upper()} shot — id={shot_id}, "
            f"joint mean err={joint_mean_error:.2f}\n"
            f"STA/LTA: MAE={sta_mae:.2f}, ±3={sta_pct:.1f}%  |  "
            f"ML: MAE={ml_mae:.2f}, ±3={ml_pct:.1f}%"
        )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))

        out_path = self.output_dir / f"shot_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # ---------------------------------------------------------------
    # HISTOGRAM
    # ---------------------------------------------------------------

    def _plot_error_histogram(self, per_trace: pd.DataFrame) -> Path:
        sta = per_trace[per_trace["sta_lta_error_samples"] >= 0][
            "sta_lta_error_samples"
        ].to_numpy()
        ml = per_trace[per_trace["ml_error_samples"] >= 0][
            "ml_error_samples"
        ].to_numpy()

        p99 = float(np.percentile(np.concatenate([sta, ml]), 99))
        upper = max(50.0, min(p99 * 1.1, 500.0))
        bins = np.linspace(0, upper, 50)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sta, bins=bins, color=COLOR_STA, alpha=0.5,
                label="STA/LTA", edgecolor="black")
        ax.hist(ml, bins=bins, color=COLOR_ML, alpha=0.5,
                label="ML", edgecolor="black")
        ax.set_yscale("log")
        ax.set_xlabel("Absolute error (samples)")
        ax.set_ylabel("Count (log scale)")
        ax.set_title("Per-trace error distribution — ML vs STA/LTA")
        ax.axvline(3.0, color="black", linestyle="--", linewidth=1.5,
                   label="±3 tolerance")

        sta_mae = float(sta.mean()) if len(sta) > 0 else 0.0
        ml_mae = float(ml.mean()) if len(ml) > 0 else 0.0
        sta_med = float(np.median(sta)) if len(sta) > 0 else 0.0
        ml_med = float(np.median(ml)) if len(ml) > 0 else 0.0
        sta_pct = float((sta <= 3).mean() * 100) if len(sta) > 0 else 0.0
        ml_pct = float((ml <= 3).mean() * 100) if len(ml) > 0 else 0.0

        text = (
            f"{'':<12}{'STA/LTA':>10}{'ML':>10}\n"
            f"MAE:         {sta_mae:>10.2f}{ml_mae:>10.2f}\n"
            f"Median:      {sta_med:>10.2f}{ml_med:>10.2f}\n"
            f"±3 acc:      {sta_pct:>9.1f}%{ml_pct:>9.1f}%"
        )
        ax.text(
            0.97, 0.97, text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = self.output_dir / "error_histogram.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ---------------------------------------------------------------
    # ERROR VS POSITION
    # ---------------------------------------------------------------

    def _plot_error_vs_position(self, per_trace: pd.DataFrame) -> Path:
        gt_pos = per_trace["gt_pick_sample"].to_numpy()
        sta_err = per_trace["sta_lta_error_samples"].to_numpy()
        ml_err = per_trace["ml_error_samples"].to_numpy()

        sta_valid = sta_err >= 0
        ml_valid = ml_err >= 0

        fig, ax = plt.subplots(figsize=(10, 6))
        if sta_valid.any():
            ax.scatter(
                gt_pos[sta_valid], np.clip(sta_err[sta_valid], 1, None),
                s=2, alpha=0.15, color=COLOR_STA, rasterized=True,
                label="STA/LTA",
            )
        if ml_valid.any():
            ax.scatter(
                gt_pos[ml_valid], np.clip(ml_err[ml_valid], 1, None),
                s=2, alpha=0.15, color=COLOR_ML, rasterized=True,
                label="ML",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Ground-truth pick position (samples)")
        ax.set_ylabel("Absolute error (samples, log scale)")
        ax.set_title("Error vs. pick position — ML vs STA/LTA")
        ax.axhline(3.0, color="black", linestyle="--", linewidth=1.5)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()

        out = self.output_dir / "error_vs_position.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ---------------------------------------------------------------
    # SUMMARY COMPARISON
    # ---------------------------------------------------------------

    def _plot_summary_comparison(
        self,
        ml_metrics: dict,
        sta_metrics: dict,
    ) -> Path:
        labels = ["MAE\n(samples)", "Median\n(samples)", "±3 accuracy\n(%)"]
        sta_vals = [
            float(sta_metrics["mean_absolute_error"]),
            float(sta_metrics["median_absolute_error"]),
            float(sta_metrics["accuracy_within_tolerance"]) * 100.0,
        ]
        ml_vals = [
            float(ml_metrics["mean_absolute_error"]),
            float(ml_metrics["median_absolute_error"]),
            float(ml_metrics["accuracy_within_tolerance"]) * 100.0,
        ]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars_sta = ax.bar(x - width / 2, sta_vals, width,
                          color=COLOR_STA, label="STA/LTA",
                          edgecolor="black")
        bars_ml = ax.bar(x + width / 2, ml_vals, width,
                         color=COLOR_ML, label="ML",
                         edgecolor="black")

        # Annotate bar heights
        for bars in (bars_sta, bars_ml):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.2f}" if h < 100 else f"{h:.0f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Summary comparison — ML vs STA/LTA")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        out = self.output_dir / "summary_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out