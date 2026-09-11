# file location: src/baselines/__init__.py

"""
Classical baseline methods for seismic first-break picking.

These pickers are non-learned algorithms that serve as reference
points for the ML models in src/models/. They operate on raw shot
data and produce 1D picks per trace.
"""

from src.baselines.base import BaselinePicker
from src.baselines.sta_lta import STALTAPicker

__all__ = ["BaselinePicker", "STALTAPicker"]