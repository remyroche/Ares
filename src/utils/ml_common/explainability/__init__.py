"""
ML Common - Explainability Module

This module contains all explainability and interpretability functionality including:
- SHAP integration
- LIME integration
- Model explanations
- Feature importance analysis
"""

from .model_explainability import ModelExplainer, SHAPExplainer, LIMEExplainer
from .model_explanations import ModelExplanations, ExplanationResult
from .model_interpretability import ModelInterpreter, InterpretabilityResult
from .shap_lime_integration import (
    SHAPLIMEExplainer, ExplanationConfig, ExplanationResult as SHAPLIMEExplanationResult,
    create_explainer, explain_model, explain_stacking_ensemble
)

__all__ = [
    # Model Explainability
    'ModelExplainer', 'SHAPExplainer', 'LIMEExplainer',
    
    # Model Explanations
    'ModelExplanations', 'ExplanationResult',
    
    # Model Interpretability
    'ModelInterpreter', 'InterpretabilityResult',
    
    # SHAP/LIME Integration
    'SHAPLIMEExplainer', 'ExplanationConfig', 'SHAPLIMEExplanationResult',
    'create_explainer', 'explain_model', 'explain_stacking_ensemble'
]