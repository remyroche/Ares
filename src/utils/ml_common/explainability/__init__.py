"""
ML Common - Explainability Module

This module contains all explainability and interpretability functionality including:
- SHAP integration
- LIME integration
- Model explanations
- Feature importance analysis
"""

from src.utils.tprint import tprint_data_format, LogLevel

from .model_explainability import ModelExplainer
# from .model_explanations import ModelExplanations, ExplanationResult  # Classes don't exist
from .model_interpretability import ModelInterpretabilityEngine, ExplanationResult
from .shap_lime_integration import (
    SHAPLIMEExplainer, ExplanationConfig, ExplanationResult as SHAPLIMEExplanationResult,
    create_explainer, explain_model, explain_stacking_ensemble
)
from .model_registry import (
    ExplainabilityRegistry, ExplainabilityConfig, ModelType,
    get_explainability_registry, register_model_for_explainability,
    get_model_explainability_config
)

__all__ = [
    # Model Explainability
    'ModelExplainer',

    # Model Interpretability
    'ModelInterpretabilityEngine', 'ExplanationResult',

    # SHAP/LIME Integration
    'SHAPLIMEExplainer', 'ExplanationConfig', 'SHAPLIMEExplanationResult',
    'create_explainer', 'explain_model', 'explain_stacking_ensemble',

    # Model Registry
    'ExplainabilityRegistry', 'ExplainabilityConfig', 'ModelType',
    'get_explainability_registry', 'register_model_for_explainability',
    'get_model_explainability_config'
]
