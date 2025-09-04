#!/usr/bin/env python3
"""Model Interpretability Package for Trading Pipeline.

This package provides comprehensive model interpretability using SHAP and LIME:
- SHAP (SHapley Additive exPlanations) for global and local feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- Feature importance analysis and visualization
- Model explanation reporting and insights
- Integration with the training pipeline
"""

from .model_explainer import ModelExplainer
from .shap_analyzer import SHAPAnalyzer
from .lime_analyzer import LIMEAnalyzer
from .interpretability_visualizer import InterpretabilityVisualizer
from .interpretability_reporter import InterpretabilityReporter

__all__ = [
    'ModelExplainer',
    'SHAPAnalyzer', 
    'LIMEAnalyzer',
    'InterpretabilityVisualizer',
    'InterpretabilityReporter'
]