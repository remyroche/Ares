"""
Feature Selection Module.

Provides LGBM-based feature selection with:
1. EWMA Spearman IC + Spearman IC stability analysis pre-filters
2. Hierarchical clustering (avoids collinearity, keeps best in class)
3. 2-step RFE with light then strong LGBM
"""

from .feature_selection_with_lgbm import FeatureSelector

__all__ = ['FeatureSelector']
