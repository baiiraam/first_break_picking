# file location: src/explainability/feature_analysis.py

"""
Feature analysis for convolutional models.

Three analyses:
    1. Weight histograms per conv layer
    2. Activation statistics per channel (mean, dead-channel fraction)
    3. First-layer filter bank visualization
    4. Kernel similarity matrices

All operate on a loaded PyTorch model. No training, no gradients.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# ============================================================
# LAYER DISCOVERY
# ============================================================

def find_conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    """Return [(name, layer), ...] for all nn.Conv2d in the model."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]


# ============================================================
# ANALYSIS 1 — WEIGHT HISTOGRAMS
# ============================================================

class WeightHistogramAnalyzer:
    """Per-layer histograms of conv weights."""

    def __init__(self, logger):
        self.logger = logger

    def analyze(self, model: nn.Module) -> dict[str, dict]:
        """Returns per-layer weight statistics."""
        stats: dict[str, dict] = {}
        layers = find_conv_layers(model)

        for name, layer in layers:
            w = layer.weight.detach().cpu().numpy().flatten()
            stats[name] = {
                "n_params": int(w.size),
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
                "abs_mean": float(np.abs(w).mean()),
                "frac_near_zero": float((np.abs(w) < 1e-3).mean()),
                "shape": tuple(layer.weight.shape),
            }

        self.logger.info(
            f"[FeatureAnalysis] Weight stats for {len(layers)} conv layers"
        )
        return stats

    def plot(self, model: nn.Module, output_dir: Path) -> Path:
        """Generate weight_histograms.png."""
        layers = find_conv_layers(model)
        n = len(layers)
        if n == 0:
            raise ValueError("No conv layers found in model")

        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if n == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, (name, layer) in enumerate(layers):
            ax = axes_flat[i]
            w = layer.weight.detach().cpu().numpy().flatten()
            ax.hist(w, bins=50, color="#1f77b4", alpha=0.75, edgecolor="black")
            ax.set_title(f"{name}\nshape={tuple(layer.weight.shape)}", fontsize=8)
            ax.set_yscale("log")
            ax.tick_params(labelsize=7)
            ax.axvline(0, color="red", linestyle="--", linewidth=0.5)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle("Conv layer weight distributions", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out = output_dir / "weight_histograms.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out


# ============================================================
# ANALYSIS 2 — ACTIVATION STATISTICS
# ============================================================

class ActivationStatisticsAnalyzer:
    """Per-channel activation statistics over a set of shots."""

    def __init__(self, logger, max_shots: int = 10):
        self.logger = logger
        self.max_shots = max_shots

    def analyze(
        self,
        model: nn.Module,
        shots: list[np.ndarray],
        device: torch.device,
    ) -> dict[str, dict]:
        """
        Run model over shots, collect per-channel activation stats.

        Args:
            model: PyTorch model
            shots: list of (n_traces, n_samples) or (1, H, W) arrays
            device: torch device

        Returns:
            {
              layer_name: {
                "n_channels": int,
                "channel_mean": list[float],       # per channel
                "channel_std": list[float],
                "channel_dead_frac": list[float],  # fraction of zeros
                "overall_dead_frac": float,
              }, ...
            }
        """
        layers = find_conv_layers(model)
        if not layers:
            return {}

        # Per-layer, per-channel accumulators
        accum: dict[str, dict[str, np.ndarray]] = {}
        for name, layer in layers:
            n_ch = layer.out_channels
            accum[name] = {
                "sum": np.zeros(n_ch, dtype=np.float64),
                "sumsq": np.zeros(n_ch, dtype=np.float64),
                "zero_count": np.zeros(n_ch, dtype=np.float64),
                "total_count": 0.0,   # per-channel count (same for all channels)
            }

        handles = []

        def make_hook(layer_name):
            def hook(module, inp, out):
                act = out.detach().cpu().numpy()   # (B, C, H, W)
                B, C, H, W = act.shape
                per_channel_count = B * H * W
                for c in range(C):
                    ch = act[:, c, :, :].flatten()
                    accum[layer_name]["sum"][c] += ch.sum()
                    accum[layer_name]["sumsq"][c] += (ch ** 2).sum()
                    accum[layer_name]["zero_count"][c] += (ch == 0).sum()
                accum[layer_name]["total_count"] += per_channel_count
            return hook

        for name, layer in layers:
            handles.append(layer.register_forward_hook(make_hook(name)))

        model.eval()
        n_shots = min(len(shots), self.max_shots)

        with torch.no_grad():
            for i in range(n_shots):
                shot = shots[i]
                if shot.ndim == 2:
                    x = (
                        torch.from_numpy(shot)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                else:
                    x = torch.from_numpy(shot).float().to(device)
                model(x)

        for h in handles:
            h.remove()

        # Aggregate
        result: dict[str, dict] = {}
        for name in accum:
            a = accum[name]
            count = max(a["total_count"], 1.0)
            mean = a["sum"] / count
            var = a["sumsq"] / count - mean ** 2
            std = np.sqrt(np.maximum(var, 0.0))
            dead_frac = a["zero_count"] / count

            result[name] = {
                "n_channels": int(a["sum"].shape[0]),
                "channel_mean": mean.tolist(),
                "channel_std": std.tolist(),
                "channel_dead_frac": dead_frac.tolist(),
                "overall_dead_frac": float(dead_frac.mean()),
            }

        self.logger.info(
            f"[FeatureAnalysis] Activation stats for {len(layers)} layers "
            f"over {n_shots} shots"
        )
        return result

    def plot(self, stats: dict[str, dict], output_dir: Path) -> Path:
        """Generate activation_statistics.png."""
        if not stats:
            raise ValueError("No activation statistics provided")

        layer_names = list(stats.keys())
        n = len(layer_names)
        cols = 2
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        if n == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, name in enumerate(layer_names):
            ax = axes_flat[i]
            s = stats[name]
            mean = np.array(s["channel_mean"])
            dead = np.array(s["channel_dead_frac"])
            x = np.arange(len(mean))

            color = np.where(dead > 0.9, "#d62728", "#1f77b4")
            ax.bar(x, mean, color=color, alpha=0.7, width=1.0)
            ax.set_title(
                f"{name} — {s['n_channels']} ch, "
                f"dead={s['overall_dead_frac']*100:.1f}%",
                fontsize=9,
            )
            ax.set_xlabel("Channel", fontsize=7)
            ax.set_ylabel("Mean activation", fontsize=7)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            "Per-channel activation statistics (red = >90% dead)",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out = output_dir / "activation_statistics.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out


# ============================================================
# ANALYSIS 3 — FIRST-LAYER FILTER BANK
# ============================================================

class FirstLayerFilterAnalyzer:
    """Renders the first conv layer's kernels as a grid of images."""

    def __init__(self, logger):
        self.logger = logger

    def plot(self, model: nn.Module, output_dir: Path) -> Path:
        """Generate first_layer_filters.png."""
        layers = find_conv_layers(model)
        if not layers:
            raise ValueError("No conv layers found in model")

        name, layer = layers[0]
        weights = layer.weight.detach().cpu().numpy()   # (out_c, in_c, kh, kw)
        out_c, in_c, kh, kw = weights.shape

        cols = min(8, out_c)
        rows = (out_c + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(1.5 * cols, 1.5 * rows))
        if out_c == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        vmax = float(np.abs(weights).max())
        if vmax == 0:
            vmax = 1.0

        for i in range(out_c):
            ax = axes_flat[i]
            if in_c == 1:
                kernel = weights[i, 0]
            else:
                kernel = weights[i].mean(axis=0)
            ax.imshow(kernel, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"ch {i}", fontsize=7)
            ax.axis("off")

        for j in range(out_c, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(
            f"First conv layer filters — {name} ({out_c} channels, "
            f"kernel {kh}×{kw})",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out = output_dir / "first_layer_filters.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self.logger.info(f"  Generated: {out.name}")
        return out


# ============================================================
# ANALYSIS 4 — KERNEL SIMILARITY
# ============================================================

class KernelSimilarityAnalyzer:
    """Pairwise cosine similarity between kernels in each conv layer."""

    def __init__(self, logger):
        self.logger = logger

    def analyze(self, model: nn.Module) -> dict[str, dict]:
        """
        Returns per-layer:
            {
              "n_channels": int,
              "kernel_size": tuple,
              "high_similarity_pairs": int,
              "frac_high_similarity": float,
              "mean_similarity": float,
              "max_offdiag_similarity": float,
            }
        """
        layers = find_conv_layers(model)
        result: dict[str, dict] = {}

        for name, layer in layers:
            w = layer.weight.detach().cpu().numpy()
            n_out = w.shape[0]
            flat = w.reshape(n_out, -1)

            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            unit = flat / norms

            sim = unit @ unit.T   # (out_c, out_c)

            mask = ~np.eye(n_out, dtype=bool)
            off_diag = sim[mask]

            upper = np.triu(sim, k=1)
            high = int((upper > 0.95).sum())

            result[name] = {
                "n_channels": int(n_out),
                "kernel_size": tuple(layer.weight.shape[1:]),
                "high_similarity_pairs": high,
                "frac_high_similarity": float(
                    high / max(n_out * (n_out - 1) / 2, 1)
                ),
                "mean_similarity": float(off_diag.mean()),
                "max_offdiag_similarity": float(off_diag.max()),
            }

        return result

    def plot(
        self,
        model: nn.Module,
        output_dir: Path,
        max_layers: int = 4,
    ) -> list[Path]:
        """Generate kernel_similarity_<layer>.png for the first N layers."""
        layers = find_conv_layers(model)
        paths = []

        for name, layer in layers[:max_layers]:
            w = layer.weight.detach().cpu().numpy()
            n_out = w.shape[0]
            if n_out < 2:
                continue

            flat = w.reshape(n_out, -1)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            unit = flat / norms
            sim = unit @ unit.T

            fig, ax = plt.subplots(
                figsize=(0.35 * n_out + 2, 0.35 * n_out + 2)
            )
            im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
            ax.set_title(
                f"Kernel similarity — {name} ({n_out} channels)",
                fontsize=10,
            )
            ax.set_xlabel("Channel", fontsize=8)
            ax.set_ylabel("Channel", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            safe = name.replace(".", "_")
            out = output_dir / f"kernel_similarity_{safe}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            self.logger.info(f"  Generated: {out.name}")
            paths.append(out)

        return paths