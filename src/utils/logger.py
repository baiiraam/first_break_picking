"""
Centralized logging configuration using Loguru with date-based folders and configurable log level.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


class SeismicLogger:
    """
    Centralized logging manager with date-based organization and dynamic task naming.
    
    Features:
        - Date-based subdirectories (logs/YYYY-MM-DD/)
        - Task name in filename (e.g., preprocess_Halfmile)
        - Timestamp in filename (HH-MM-SS)
        - Symlink to latest log
        - Log rotation, compression, and retention
        - Configurable log level
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        task_name: str = "general",
        level: str = "INFO",
        create_latest_symlink: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.task_name = task_name
        self.level = level.upper()
        self.create_latest_symlink = create_latest_symlink
        
        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            print(f"Warning: Invalid log level '{level}', defaulting to INFO")
            self.level = "INFO"
        
        # Create base log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create date-based subdirectory
        self.date_dir = self.log_dir / datetime.now().strftime("%Y-%m-%d")
        self.date_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        self.timestamp = datetime.now().strftime("%H-%M-%S")
        
        # Setup logging
        self._setup_logger()
        
        # Create symlink to latest log
        if self.create_latest_symlink:
            self._create_symlink()
    
    def _setup_logger(self):
        """Setup all logger handlers with configurable level."""
        # Remove default handlers
        logger.remove()
        
        # Console handler (colorized, human-readable)
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            level=self.level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # --- File Handlers ---
        
        # 1. Main log file (configurable level)
        self.main_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}.log"
        logger.add(
            self.main_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level=self.level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True
        )
        
        # 2. Error file handler (ERROR and above, always on)
        self.error_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}_errors.log"
        logger.add(
            self.error_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enqueue=True
        )
        
        # 3. Debug file handler (DEBUG only, always on)
        self.debug_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}_debug.log"
        logger.add(
            self.debug_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="50 MB",
            retention="3 days",
            compression="gz",
            enqueue=True,
            filter=lambda record: record["level"].name == "DEBUG"
        )
        
        # 4. JSON logs for monitoring (INFO and above)
        self.json_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}.json"
        logger.add(
            self.json_log_path,
            format="{message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
            serialize=True
        )
        
        # Store log paths for reference
        self.main_log_path = self.main_log_path
        self.error_log_path = self.error_log_path
        self.debug_log_path = self.debug_log_path
        self.json_log_path = self.json_log_path
    
    def _create_symlink(self):
        """Create a symlink to the latest log file."""
        # Skip in worker processes
        from multiprocessing import current_process
        if current_process().name != 'MainProcess':
            return
        
        latest_dir = self.log_dir / "latest"
        latest_dir.mkdir(exist_ok=True)
        
        latest_link = latest_dir / "latest.log"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        
        try:
            rel_path = os.path.relpath(self.main_log_path, latest_dir)
            os.symlink(rel_path, latest_link)
        except Exception as e:
            logger.debug(f"Could not create symlink: {e}")
    
    def get_log_paths(self) -> dict:
        """Return all log file paths."""
        return {
            "main": self.main_log_path,
            "errors": self.error_log_path,
            "debug": self.debug_log_path,
            "json": self.json_log_path,
        }
    
    def get_logger(self):
        """Return the logger instance."""
        return logger


# ============================================================
# GLOBAL LOGGER INSTANCE
# ============================================================

_global_logger = None


def setup_logger(
    task_name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    create_latest_symlink: bool = True
):
    """
    Setup a logger with a specific task name and log level.
    
    Args:
        task_name: Name of the task (e.g., 'preprocess_Halfmile', 'training_Halfmile_unet')
        log_dir: Directory to store logs
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        create_latest_symlink: Whether to create a symlink to the latest log
    
    Returns:
        loguru.Logger: Configured logger instance
    """
    global _global_logger
    
    seismic_logger = SeismicLogger(
        log_dir=log_dir,
        task_name=task_name,
        level=level,
        create_latest_symlink=create_latest_symlink
    )
    
    _global_logger = seismic_logger
    return seismic_logger.get_logger()


def get_logger():
    """
    Get the global logger instance.
    
    If the logger hasn't been set up, it creates a default one.
    """
    global _global_logger
    
    if _global_logger is None:
        seismic_logger = SeismicLogger(task_name="general", level="INFO")
        _global_logger = seismic_logger
    
    return _global_logger.get_logger()


def create_task_name(config, task_type: str, model_name: Optional[str] = None) -> str:
    """
    Create a task name from config and task type.
    
    Args:
        config: SeismicConfig object
        task_type: 'preprocess', 'train', 'evaluate', 'visualize', 'export'
        model_name: Optional model name for training runs
    
    Returns:
        str: Task name (e.g., 'preprocess_Halfmile', 'training_Halfmile_unet')
    """
    parts = [task_type, config.dataset_name]
    if model_name:
        parts.append(model_name)
    return "_".join(parts)