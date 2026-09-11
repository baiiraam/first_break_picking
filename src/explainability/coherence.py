# file location: src/explainability/coherence.py

"""
Wavefront coherence analysis.

A seismic first break is a wavefront: across traces, the pick
positions should form a smooth, slowly-varying curve. This module
computes per-shot coherence metrics (how smooth the picks are) and
generates trajectory / distribution figures.

Coherence metric: median absolute second-order finite difference of
the valid picks along the trace axis. Lower = smoother.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_GT = "#2ca02c"    # green
COLOR_ML = "#1f77b4"    # blue
COLOR_STA = "#d62728"   # red


class CoherenceAnalyzer:
    """
    Computes per-shot wavefront coherence and generates figures.
    """

    def __init__(self, logger):
        self.logger = logger

    # ---------------------------------------------------------------
    # METRIC
    # ---------------------------------------------------------------

    @staticmethod
    def _coherence(picks: np.ndarray) -> float:
        """
        Median absolute second-order difference of valid picks.

        Args:
            picks: (n_traces,) int array; 0 = no pick.

        Returns:
            coherence score (float); inf if fewer than 3 valid picks.
        """
        valid = picks[picks > 0].astype(np.int64)
        if len(valid) < 3:
            return float("inf")

        d2 = np.abs(np.diff(valid, n=2))
        return float(np.median(d2))

    def compute_metrics(self, per_trace: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-shot coherence metrics from a per-trace DataFrame.

        Args:
            per_trace: DataFrame with columns:
                shot_id, trace_index, gt_pick_sample,
                sta_lta_pick, ml_pick

        Returns:
            DataFrame with one row per shot:
                shot_id, ml_coherence, gt_coherence, sta_coherence,
                ml_over_gt_ratio, n_valid
        """
        required = {
            "shot_id", "trace_index", "gt_pick_sample",
            "sta_lta_pick", "ml_pick",
        }
        missing = required - set(per_trace.columns)
        if missing:
            raise ValueError(
                f"per_trace DataFrame missing columns: {sorted(missing)}"
            )

        rows: list[dict] = []

        for shot_id, group in per_trace.groupby("shot_id"):
            g = group.sort_values("trace_index")

            ml = g["ml_pick"].to_numpy()
            gt = g["gt_pick_sample"].to_numpy()
            sta = g["sta_lta_pick"].to_numpy()

            ml_coh = self._coherence(ml)
            gt_coh = self._coherence(gt)
            sta_coh = self._coherence(sta)

            # Ratio, guarding against div-by-zero
            denom = gt_coh if np.isfinite(gt_coh) and gt_coh > 0.5 else 0.5
            ratio = ml_coh / denom if np.isfinite(ml_coh) else float("inf")

            n_valid = int((ml > 0).sum())

            rows.append({
                "shot_id": int(shot_id),
                "ml_coherence": ml_coh,
                "gt_coherence": gt_coh,
                "sta_coherence": sta_coh,
                "ml_over_gt_ratio": ratio,
                "n_valid": n_valid,
            })

        df = pd.DataFrame(rows)
        self.logger.info(
            f"[Coherence] Computed metrics for {len(df)} shots; "
            f"median ML coherence = "
            f"{df['ml_coherence'].median():.2f}"
        )
        return df

    # ---------------------------------------------------------------
    # TRAJECTORIES FIGURE
    # ---------------------------------------------------------------

    def plot_trajectories(
        self,
        per_trace: pd.DataFrame,
        selections: list[tuple[int, str]],
        output_dir: Path,
        n_samples: int,
    ) -> Path:
        """
        Generate coherence_trajectories.png.

        Args:
            per_trace: the per-trace DataFrame
            selections: list of (shot_id, label) tuples
            output_dir: where to save
            n_samples: total samples per trace (for y-axis limits)

        Returns:
            Path to saved figure.
        """
        n = len(selections)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n))
        if n == 1:
            axes = [axes]

        for ax, (shot_id, label) in zip(axes, selections):
            sub = per_trace[per_trace["shot_id"] == shot_id].sort_values(
                "trace_index"
            )
            trace_idx = sub["trace_index"].to_numpy()
            ml = sub["ml_pick"].to_numpy()
            gt = sub["gt_pick_sample"].to_numpy()
            sta = sub["sta_lta_pick"].to_numpy()

            # Plot only valid picks as points+lines
            for arr, color, name in [
                (gt, COLOR_GT, "GT"),
                (ml, COLOR_ML, "ML"),
                (sta, COLOR_STA, "STA/LTA"),
            ]:
                valid = arr > 0
                ax.plot(
                    trace_idx[valid], arr[valid],
                    marker=".", linestyle="-", linewidth=0.8,
                    color=color, label=name, markersize=4,
                )

            ax.set_ylim(n_samples, 0)
            ax.set_xlabel("Trace index")
            ax.set_ylabel("Pick sample")
            ax.set_title(f"Shot {shot_id} — {label}")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Wavefront coherence — pick trajectories",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        out = output_dir / "coherence_trajectories.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out

    # ---------------------------------------------------------------
    # DISTRIBUTION FIGURE
    # ---------------------------------------------------------------

    def plot_distribution(
        self,
        metrics: pd.DataFrame,
        output_dir: Path,
    ) -> Path:
        """
        Generate coherence_distribution.png.
        """
        ml = metrics["ml_coherence"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        gt = metrics["gt_coherence"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        sta = metrics["sta_coherence"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Use a shared bin range
        p99 = float(np.percentile(
            np.concatenate([ml.values, gt.values, sta.values]), 99
        ))
        upper = max(20.0, min(p99 * 1.1, 200.0))
        bins = np.linspace(0, upper, 40)

        ax.hist(
            gt.values, bins=bins, color=COLOR_GT, alpha=0.5,
            label="GT", edgecolor="black",
        )
        ax.hist(
            ml.values, bins=bins, color=COLOR_ML, alpha=0.5,
            label="ML", edgecolor="black",
        )
        ax.hist(
            sta.values, bins=bins, color=COLOR_STA, alpha=0.5,
            label="STA/LTA", edgecolor="black",
        )

        ax.set_xlabel("Coherence score (median absolute curvature)")
        ax.set_ylabel("Count")
        ax.set_title(
            "Wavefront coherence distribution across test shots "
            "(lower = smoother)"
        )

        text = (
            f"Median coherence\n"
            f"GT:      {gt.median():.2f}\n"
            f"ML:      {ml.median():.2f}\n"
            f"STA/LTA: {sta.median():.2f}\n"
            f"\n"
            f"ML / GT ratio (median): "
            f"{metrics['ml_over_gt_ratio'].replace([np.inf,-np.inf], np.nan).median():.2f}"
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
        out = output_dir / "coherence_distribution.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out