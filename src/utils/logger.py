"""
Centralized logging configuration using Loguru.
"""

import sys
import json
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any


# --- Global logger instance ---
_logger = None


def setup_logger(log_dir: str = "logs", level: str = "INFO") -> logger:
    """
    Setup and return the global logger instance.
    
    Returns:
        loguru.Logger: Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
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
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Main log file
    logger.add(
        log_dir_path / "{time:YYYY-MM-DD_HH-MM-SS}.log",
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True
    )
    
    # Error file handler
    logger.add(
        log_dir_path / "{time:YYYY-MM-DD_HH-MM-SS}_errors.log",
        format="{time} | {level} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True
    )
    
    # JSON logs
    logger.add(
        log_dir_path / "{time:YYYY-MM-DD_HH-MM-SS}.json",
        format="{message}",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        enqueue=True,
        serialize=True
    )
    
    _logger = logger
    return _logger


def get_logger() -> logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        return setup_logger()
    return _logger