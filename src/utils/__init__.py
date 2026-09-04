# src/utils/__init__.py

# Import from hdf5_utils
from src.utils.hdf5_utils import (
    HDF5SeismicReader,
    load_shot_data,
    load_shot_indices,
    # validate_hdf5 is now a method on HDF5SeismicReader, not a standalone function
)

# Import from other utils
from src.utils.logger import create_task_name, get_logger, setup_logger
from src.utils.memory_utils import clear_memory, get_memory_manager, get_memory_usage
from src.utils.mlflow_utils import get_mlflow_manager
from src.utils.tensorboard_utils import TensorBoardManager


# If you need validate_hdf5 as a convenience function:
def validate_hdf5(hdf5_path: str) -> bool:
    """Convenience function for HDF5 validation."""
    with HDF5SeismicReader(hdf5_path) as reader:
        return reader.validate_hdf5()


__all__ = [
    "HDF5SeismicReader",
    "TensorBoardManager",
    "clear_memory",
    "create_task_name",
    "get_logger",
    "get_memory_manager",
    "get_memory_usage",
    "get_mlflow_manager",
    "load_shot_data",
    "load_shot_indices",
    "setup_logger",
    "validate_hdf5",  # Now available
]
