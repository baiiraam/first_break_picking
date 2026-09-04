"""
Centralized logging configuration using Loguru with context injection.
"""

import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from multiprocessing import current_process
from pathlib import Path

from loguru import logger

# ✅ Context variables for automatic metadata injection
_task_context: ContextVar[str] = ContextVar("task_context", default="general")
_run_id: ContextVar[str] = ContextVar("run_id", default="")
_epoch: ContextVar[int] = ContextVar("epoch", default=-1)
_batch: ContextVar[int] = ContextVar("batch", default=-1)


def set_task_context(task_name: str):
    """Set the current task context for all log messages."""
    _task_context.set(task_name)


def set_run_id(run_id: str):
    """Set the current MLflow run ID for all log messages."""
    _run_id.set(run_id)


def set_epoch(epoch: int):
    """Set the current epoch for all log messages."""
    _epoch.set(epoch)


def set_batch(batch: int):
    """Set the current batch for all log messages."""
    _batch.set(batch)


def clear_context():
    """Clear all context variables."""
    _task_context.set("general")
    _run_id.set("")
    _epoch.set(-1)
    _batch.set(-1)


class SeismicLogger:
    """
    Centralized logging manager with date-based organization and context injection.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        task_name: str = "general",
        level: str = "INFO",
        create_latest_symlink: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.task_name = task_name
        self.level = level.upper()
        self.create_latest_symlink = create_latest_symlink

        # Set initial context
        _task_context.set(task_name)

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            print(f"Warning: Invalid log level '{level}', defaulting to INFO")
            self.level = "INFO"

        # Create base log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create date-based subdirectory
        self.date_dir = self.log_dir / datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.date_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp
        self.timestamp = datetime.now(timezone.utc).strftime("%H-%M-%S")

        # Setup logging
        self._setup_logger()

        # Create symlink to latest log
        if self.create_latest_symlink:
            self._create_symlink()

    def _setup_logger(self):
        """Setup all logger handlers with configurable level and context injection."""
        # Remove default handlers
        logger.remove()

        # ✅ Custom format with context injection
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<yellow>[{extra[task]}]</yellow> "
            "<blue>[run:{extra[run_id]}]</blue> "
            "<magenta>[epoch:{extra[epoch]}]</magenta> "
            "<level>{message}</level>"
        )

        def inject_context(record):
            record["extra"]["task"] = _task_context.get()
            record["extra"]["run_id"] = _run_id.get()
            record["extra"]["epoch"] = _epoch.get()
            record["extra"]["batch"] = _batch.get()
            return True

        # Console handler (colorized, human-readable)
        logger.add(
            sys.stdout,
            format=log_format,
            level=self.level,
            colorize=True,
            filter=inject_context,
            backtrace=True,
            diagnose=True,
        )

        # --- File Handlers ---

        # 1. Main log file (configurable level)
        self.main_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}.log"
        logger.add(
            self.main_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | "
            "[{extra[task]}] [run:{extra[run_id]}] [epoch:{extra[epoch]}] | {message}",
            filter=inject_context,
            level=self.level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
        )

        # 2. Error file handler (ERROR and above)
        self.error_log_path = (
            self.date_dir / f"{self.timestamp}_{self.task_name}_errors.log"
        )
        logger.add(
            self.error_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | "
            "[{extra[task]}] [run:{extra[run_id]}] [epoch:{extra[epoch]}] | {message}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enqueue=True,
            filter=lambda record: self._inject_context(record),
        )

        # 3. JSON logs for monitoring
        self.json_log_path = self.date_dir / f"{self.timestamp}_{self.task_name}.json"
        logger.add(
            self.json_log_path,
            format="{message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
            serialize=True,
            filter=lambda record: self._inject_context(record),
        )

        # Store log paths for reference
        self.main_log_path = self.main_log_path
        self.error_log_path = self.error_log_path
        self.json_log_path = self.json_log_path

    def _inject_context(self, record):
        """Inject context variables into log record."""
        record["extra"]["task"] = _task_context.get()
        record["extra"]["run_id"] = _run_id.get()
        record["extra"]["epoch"] = _epoch.get()
        record["extra"]["batch"] = _batch.get()
        return True

    def _create_symlink(self):
        """Create a symlink to the latest log file."""
        # Skip in worker processes
        if current_process().name != "MainProcess":
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
    create_latest_symlink: bool = True,
):
    """Setup a logger with a specific task name and log level."""
    global _global_logger

    seismic_logger = SeismicLogger(
        log_dir=log_dir,
        task_name=task_name,
        level=level,
        create_latest_symlink=create_latest_symlink,
    )

    _global_logger = seismic_logger
    return seismic_logger.get_logger()


def get_logger():
    """Get the global logger instance."""
    global _global_logger

    if _global_logger is None:
        seismic_logger = SeismicLogger(task_name="general", level="INFO")
        _global_logger = seismic_logger

    return _global_logger.get_logger()


def create_task_name(config, task_type: str, model_name: str | None = None) -> str:
    """Create a task name from config and task type."""
    parts = [task_type, config.dataset_name]
    if model_name:
        parts.append(model_name)
    return "_".join(parts)
