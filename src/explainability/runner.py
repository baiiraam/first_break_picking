# file location: src/explainability/runner.py

"""
Orchestrates explainability: runs the model, computes the heatmap,
generates the figure, for a set of shots.
"""

from pathlib import Path

import numpy as np
import torch

from src.explainability.base import Explainer
from src.explainability.image_generator import ExplainabilityImageGenerator
from src.types import LoggerType


class ExplainabilityRunner:
    """
    Runs an explainability method on a set of shots and produces figures.

    Usage:
        runner = ExplainabilityRunner(model, explainer, device, logger)
        paths = runner.explain_shots(
            shot_ids=[...],
            shots=[(shot_data, gt_mask), ...],
            labels=["best", "median", "worst"],
            mean_errors=[...],
            image_generator=gen,
            dataset_name="Halfmile",
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        explainer: Explainer,
        device: torch.device,
        logger: LoggerType,
        target_class: int = 2,
    ):
        self.model = model
        self.explainer = explainer
        self.device = device
        self.logger = logger
        self.target_class = target_class

    def explain_shots(
        self,
        shot_ids: list[int],
        shots: list[tuple[np.ndarray, np.ndarray]],
        labels: list[str],
        mean_errors: list[float],
        image_generator: ExplainabilityImageGenerator,
        dataset_name: str,
    ) -> list[Path]:
        """
        Run the explainer on each shot and generate a 4-panel figure.

        Args:
            shot_ids: shot IDs
            shots: list of (shot_data, gt_mask) tuples
            labels: "best" / "median" / "worst"
            mean_errors: per-shot mean error (for the title)
            image_generator: for figure output
            dataset_name: for figure title

        Returns:
            List of paths to generated figures.
        """
        if not (
            len(shot_ids) == len(shots) == len(labels) == len(mean_errors)
        ):
            raise ValueError(
                "shot_ids, shots, labels, and mean_errors must have "
                "the same length"
            )

        paths: list[Path] = []
        self.model.eval()

        for shot_id, (shot_data, gt_mask), label, mean_err in zip(
            shot_ids, shots, labels, mean_errors
        ):
            self.logger.info(f"  Explaining shot {shot_id} ({label})...")

            x = (
                torch.from_numpy(shot_data)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

            # Forward pass for prediction
            with torch.no_grad():
                logits = self.model(x)
                pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

            # Explainability (needs grad enabled)
            heatmap = self.explainer.explain(
                self.model, x, target_class=self.target_class
            )

            # Safety net: match heatmap to shot shape
            if heatmap.shape != shot_data.shape:
                import torch.nn.functional as F
                h_t = torch.from_numpy(heatmap).float()[None, None]
                h_t = F.interpolate(
                    h_t, size=shot_data.shape, mode="bilinear",
                    align_corners=False,
                )
                heatmap = h_t[0, 0].numpy()

            path = image_generator.generate_for_shot(
                shot_id=shot_id,
                shot_data=shot_data,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                heatmap=heatmap,
                label=label,
                mean_error=mean_err,
                dataset_name=dataset_name,
            )
            paths.append(path)

        return paths