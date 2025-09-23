"""
Attention Mechanisms for Tree-Based Models

This module provides attention mechanisms for CatBoost, LightGBM, XGBoost, and other tree-based models.
"""

from .tree_attention import TreeAttentionMechanism
from .catboost_attention import CatBoostAttentionWrapper
from .lightgbm_attention import LightGBMAttentionWrapper
from .xgboost_attention import XGBoostAttentionWrapper
from .ensemble_attention import EnsembleAttentionWrapper

__all__ = [
    'TreeAttentionMechanism',
    'CatBoostAttentionWrapper',
    'LightGBMAttentionWrapper', 
    'XGBoostAttentionWrapper',
    'EnsembleAttentionWrapper'
]