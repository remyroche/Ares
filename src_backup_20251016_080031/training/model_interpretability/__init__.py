#!/usr/bin/env python3
"""Model Interpretability Package for Trading Pipeline.

This package provides comprehensive model interpretability using SHAP and LIME:
- SHAP (SHapley Additive exPlanations) for global and local feature importance
- LIME (Local Interpretable Model-agnostic Explanations) for local explanations
- Feature importance analysis and visualization
- Model explanation reporting and insights
- Integration with the training pipeline
"""


__all__ = [
    'ModelExplainer',
    'SHAPAnalyzer', 
    'LIMEAnalyzer',
    'InterpretabilityVisualizer',
    'InterpretabilityReporter'
]