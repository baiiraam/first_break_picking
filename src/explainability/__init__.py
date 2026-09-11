# file location: src/explainability/__init__.py

"""
Explainability and diagnostics for seismic FBP models.
"""

from src.explainability.base import Explainer, find_last_conv_layer
from src.explainability.coherence import CoherenceAnalyzer
from src.explainability.error_gallery import ErrorGalleryGenerator
from src.explainability.feature_analysis import (
    ActivationStatisticsAnalyzer,
    FirstLayerFilterAnalyzer,
    KernelSimilarityAnalyzer,
    WeightHistogramAnalyzer,
    find_conv_layers,
)
from src.explainability.gradcam import GradCAM
from src.explainability.image_generator import ExplainabilityImageGenerator
from src.explainability.runner import ExplainabilityRunner

__all__ = [
    "Explainer",
    "find_last_conv_layer",
    "find_conv_layers",
    "GradCAM",
    "ExplainabilityImageGenerator",
    "ExplainabilityRunner",
    "CoherenceAnalyzer",
    "ErrorGalleryGenerator",
    "WeightHistogramAnalyzer",
    "ActivationStatisticsAnalyzer",
    "FirstLayerFilterAnalyzer",
    "KernelSimilarityAnalyzer",
]