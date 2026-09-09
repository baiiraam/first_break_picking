# src/evaluation/__init__.py
"""
Evaluation package for seismic models.

Provides:
- EvaluationRunner: Core evaluation loop with metric collection
- ResultExporter: Export results to JSON, CSV, and summary formats
"""

from src.evaluation.exporter import ResultExporter
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "ResultExporter",
]
