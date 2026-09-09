# src/utils/__init__.py
"""
Utilities module for Seismic FBP.
"""

from src.utils.device_utils import (
    get_cpu_memory_info,
    get_cuda_memory_info,
    get_device_info,
    get_mps_memory_info,
    get_recommended_memory_limits,
)
from src.utils.error_patterns import is_memory_error, is_real_error
from src.utils.hdf5_utils import load_shot_data, load_shot_indices, validate_hdf5
from src.utils.logger import create_task_name, get_logger, setup_logger
from src.utils.memory import (
    clear_memory,
    get_memory_usage,
    is_memory_low,
    log_memory_stats,
)
from src.utils.mlflow_utils import MLflowManager, get_mlflow_manager
from src.utils.model_registry import (
    get_all_model_profiles,
    get_model_profile,
    get_model_registry,
)
from src.utils.tensorboard_utils import TensorBoardManager

# Sorted alphabetically
__all__ = [
    "MLflowManager",
    "TensorBoardManager",
    "clear_memory",
    "create_task_name",
    "get_all_model_profiles",
    "get_cpu_memory_info",
    "get_cuda_memory_info",
    "get_device_info",
    "get_logger",
    "get_memory_usage",
    "get_mlflow_manager",
    "get_model_profile",
    "get_model_registry",
    "get_mps_memory_info",
    "get_recommended_memory_limits",
    "is_memory_error",
    "is_memory_low",
    "is_real_error",
    "load_shot_data",
    "load_shot_indices",
    "log_memory_stats",
    "setup_logger",
    "validate_hdf5",
]
