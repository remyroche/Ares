"""Step 10 Feature Engineering Module.

This module handles all feature engineering tasks including:
- Cross-timeframe correlations
- Regime transition features
- Sequence creation and preprocessing
- Intensity-based feature processing
"""

from .engineer import FeatureEngineer

__all__ = ['FeatureEngineer']
