"""
================================================================================
SNN COMPARISON MODULE - ANTIGRAVITY STROKE AI
================================================================================
Dedicated benchmarking and comparative evaluation package for:
  - Baseline Old SNN (.old/best_model.pt)
  - Proposed Multi-Scale Spatial SNN (best_model.pt)
================================================================================
"""

from .old_snn import OldSiameseNet, load_old_snn_model, predict_old_snn

__all__ = [
    "OldSiameseNet",
    "load_old_snn_model",
    "predict_old_snn",
]
