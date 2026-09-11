# file location: src/explainability/error_gallery.py

"""
Worst-N error gallery.

Produces a compact figure showing the N traces with the largest
errors for a given method (ML or STA/LTA). Each row shows one trace's
seismogram with GT, ML, and STA/LTA picks marked.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_GT = "#2ca02c"
COLOR_ML = "#1f77b4"
COLOR_STA = "#d62728"


class ErrorGalleryGenerator:
    """
    Builds worst-N galleries for ML and STA/LTA.
    """

    def __init__(self, output_dir: Path, logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def generate_gallery(
        self,
        per_trace: pd.DataFrame,
        shots: dict[int, np.ndarray],
        method: str,
        n_worst: int = 20,
        output_filename: str = "worst_gallery.png",
    ) -> Path:
        """
        Generate the worst-N gallery for a method.

        Args:
            per_trace: DataFrame with columns:
                shot_id, trace_index, gt_pick_sample,
                sta_lta_pick, sta_lta_error_samples,
                ml_pick, ml_error_samples
            shots: dict shot_id -> (n_traces, n_samples) seismogram
            method: "ml" or "sta_lta"
            n_worst: how many traces to show
            output_filename: output filename

        Returns:
            Path to saved figure.
        """
        if method not in ("ml", "sta_lta"):
            raise ValueError(
                f"method must be 'ml' or 'sta_lta', got {method!r}"
            )

        err_col = f"{method}_error_samples"
        pred_col = f"{method}_pick"

        if err_col not in per_trace.columns:
            raise ValueError(f"Column {err_col} not found")

        # Sort by error, filter out -1 (no pick)
        df = per_trace[per_trace[err_col] >= 0].copy()
        df = df.sort_values(err_col, ascending=False)
        df = df.head(n_worst)

        if len(df) == 0:
            self.logger.warning(
                f"[Gallery] No valid traces for method '{method}'"
            )
            # Write a placeholder figure
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, f"No valid picks for {method}",
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            out = self.output_dir / output_filename
            fig.savefig(out, dpi=120)
            plt.close(fig)
            return out

        n = len(df)
        fig, axes = plt.subplots(n, 1, figsize=(14, 0.9 * n))
        if n == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, df.iterrows()):
            shot_id = int(row["shot_id"])
            trace_idx = int(row["trace_index"])

            if shot_id not in shots:
                ax.text(0.5, 0.5, f"shot {shot_id} not loaded",
                        ha="center", va="center")
                ax.axis("off")
                continue

            seismogram = shots[shot_id]
            if trace_idx >= seismogram.shape[0]:
                ax.text(0.5, 0.5, f"trace {trace_idx} out of range",
                        ha="center", va="center")
                ax.axis("off")
                continue

            trace = seismogram[trace_idx]

            # Plot trace
            ax.plot(trace, color="black", linewidth=0.6, alpha=0.7)

            # Markers
            gt = int(row["gt_pick_sample"])
            sta = int(row["sta_lta_pick"])
            ml = int(row["ml_pick"])

            ymin, ymax = float(trace.min()), float(trace.max())
            pad = 0.1 * (ymax - ymin + 1e-9)

            ax.set_xlim(0, len(trace))
            ax.set_ylim(ymin - pad, ymax + pad)

            if gt > 0:
                ax.axvline(gt, color=COLOR_GT, linewidth=1.2, alpha=0.8)
            if sta > 0:
                ax.axvline(sta, color=COLOR_STA, linewidth=1.2,
                           linestyle="--", alpha=0.8)
            if ml > 0:
                ax.axvline(ml, color=COLOR_ML, linewidth=1.2,
                           linestyle=":", alpha=0.8)

            # Row annotation
            err_val = float(row[err_col])
            other = "sta_lta" if method == "ml" else "ml"
            other_err = float(row[f"{other}_error_samples"])
            other_name = "STA" if method == "ml" else "ML"

            ax.set_ylabel(
                f"shot {shot_id}\ntrace {trace_idx}",
                fontsize=7, rotation=0, labelpad=45, va="center",
            )
            ax.set_title(
                f"{method.upper()} err={err_val:.0f}  |  "
                f"{other_name} err={other_err:.0f}",
                fontsize=7, loc="right",
            )
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=6)

        # Legend for the whole figure
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=COLOR_GT, lw=2, label="GT pick"),
            Line2D([0], [0], color=COLOR_STA, lw=2, linestyle="--",
                   label="STA/LTA pick"),
            Line2D([0], [0], color=COLOR_ML, lw=2, linestyle=":",
                   label="ML pick"),
        ]
        fig.legend(
            handles=legend_elements, loc="upper center",
            ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.995),
        )

        fig.suptitle(
            f"Worst {n} traces by {method.upper()} error",
            fontsize=12, y=0.999,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))

        out = self.output_dir / output_filename
        fig.savefig(out, dpi=120)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out