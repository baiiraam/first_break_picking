# file location: src/deployment/predictor.py

"""
Single-call inference interface for trained models.

Usage:
    predictor = Predictor.load(
        checkpoint="models/registry/MPSLightUNet_Halfmile_best.pt",
        device="mps",
    )

    # From an in-memory shot
    picks = predictor.predict(shot_data)      # (n_traces,) int64

    # From HDF5 directly
    picks = predictor.predict_from_hdf5(
        "data/raw/Halfmile3D_add_geom_sorted.hdf5",
        shot_index=0,
    )
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import SeismicConfig
from src.deployment.contract import InputContract, read_contract_from_checkpoint
from src.models.loader import load_evaluation_model
from src.training.metrics import extract_picks_from_mask
from src.types import LoggerType
from src.utils.hdf5_utils import load_shot_data, load_shot_indices
from src.utils.logger import get_logger


class Predictor:
    """
    Loads a model once and provides a simple predict() interface.

    All shape manipulation, dtype conversion, and pick extraction
    happen inside this class. Users interact with numpy arrays only.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        contract: InputContract,
        device: torch.device,
        model_name: str,
        logger: LoggerType,
    ):
        self.model = model
        self.contract = contract
        self.device = device
        self.model_name = model_name
        self.logger = logger
        self.model.eval()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        config: SeismicConfig | None = None,
        device: str | torch.device = "cpu",
        logger: LoggerType | None = None,
    ) -> "Predictor":
        """
        Load a model from a checkpoint.

        Args:
            checkpoint: path to .pt checkpoint
            config: optional SeismicConfig (used for the loader).
                    If None, one is built from the checkpoint.
            device: torch device or string ("cpu", "cuda", "mps")
            logger: optional logger; defaults to global

        Returns:
            A ready-to-use Predictor.
        """
        if logger is None:
            logger = get_logger()

        if isinstance(device, str):
            device = torch.device(device)

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Read the contract
        contract = read_contract_from_checkpoint(checkpoint_path)

        # Build a minimal config if none was provided
        if config is None:
            config = SeismicConfig(
                target_traces=contract.target_traces,
                n_samples=contract.n_samples,
                device=str(device),
            )

        # Load the model
        model = load_evaluation_model(str(checkpoint_path), config, device, logger)

        # Determine a model name for display
        model_name = checkpoint_path.stem

        return cls(
            model=model,
            contract=contract,
            device=device,
            model_name=model_name,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, shot_data: np.ndarray) -> np.ndarray:
        """
        Predict picks for a shot.

        Args:
            shot_data: (n_traces, n_samples) numpy array of raw amplitudes.
                       Will be padded or cropped to match the contract.

        Returns:
            (target_traces,) int64 picks. Value 0 = no pick.
        """
        # 1. Validate input
        if not isinstance(shot_data, np.ndarray):
            shot_data = np.asarray(shot_data)
        if shot_data.ndim != 2:
            raise ValueError(
                f"shot_data must be 2D (n_traces, n_samples), "
                f"got shape {shot_data.shape}"
            )

        # 2. Pad/crop to (target_traces, n_samples)
        prepared = self._pad_or_crop(shot_data)

        # 3. Build tensor (1, 1, H, W)
        x = (
            torch.from_numpy(prepared)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # 4. Run model
        with torch.no_grad():
            logits = self.model(x)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

        # 5. Extract picks
        picks = extract_picks_from_mask(pred_mask)
        return picks

    def predict_from_hdf5(
        self,
        hdf5_path: str | Path,
        shot_index: int,
    ) -> np.ndarray:
        """
        Load a shot from HDF5 and predict its picks.

        Args:
            hdf5_path: path to the HDF5 file
            shot_index: which shot (0-indexed) to predict

        Returns:
            (target_traces,) int64 picks.
        """
        unique_shots, start_indices, end_indices = load_shot_indices(
            str(hdf5_path)
        )
        if shot_index < 0 or shot_index >= len(unique_shots):
            raise IndexError(
                f"shot_index {shot_index} out of range "
                f"(file has {len(unique_shots)} shots)"
            )

        start = int(start_indices[shot_index])
        end = int(end_indices[shot_index])

        shot_data, _ = load_shot_data(
            str(hdf5_path),
            start,
            end,
            target_traces=self.contract.target_traces,
            n_samples=self.contract.n_samples,
        )
        return self.predict(shot_data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _pad_or_crop(self, shot_data: np.ndarray) -> np.ndarray:
        """
        Pad with zeros or crop so the shot matches the contract's shape.
        """
        actual_traces, actual_samples = shot_data.shape
        target_traces = self.contract.target_traces
        n_samples = self.contract.n_samples

        # Crop in the sample dimension if needed
        if actual_samples > n_samples:
            shot_data = shot_data[:, :n_samples]
            actual_samples = n_samples
        elif actual_samples < n_samples:
            # Pad sample dimension with zeros
            padded = np.zeros((actual_traces, n_samples), dtype=shot_data.dtype)
            padded[:, :actual_samples] = shot_data
            shot_data = padded

        # Pad or crop trace dimension
        if actual_traces < target_traces:
            padded = np.zeros((target_traces, n_samples), dtype=shot_data.dtype)
            padded[:actual_traces, :] = shot_data
            return padded.astype(np.float32)
        elif actual_traces > target_traces:
            return shot_data[:target_traces, :].astype(np.float32)

        return shot_data.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"Predictor(model={self.model_name!r}, "
            f"device={self.device}, "
            f"traces={self.contract.target_traces}, "
            f"samples={self.contract.n_samples})"
        )