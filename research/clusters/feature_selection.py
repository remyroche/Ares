"""
Feature selection utilities: mRMR screening aligned with the training framework.

This module delegates mRMR to src/training/utils/feature_selection/selection_methods
for consistency and robustness. If the training selector is unavailable, it
fails fast rather than providing a degraded local fallback.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Use the richer selection framework and do not provide local fallbacks
from src.training.utils.feature_selection.selection_methods import MRMRSelector  # type: ignore


@dataclass
class MRMRConfig:
    """Configuration for mRMR selection."""
    max_features: int = 50
    redundancy_penalty: float = 0.5
    normalize_scores: bool = True
    random_state: int = 42


def mrmr_select(X: pd.DataFrame, y: np.ndarray, config: Optional[MRMRConfig] = None) -> List[str]:
    """Select features using the training MRMR selector. Fails fast if unavailable."""
    cfg = config or MRMRConfig()
    feature_names = list(X.columns)
    if len(feature_names) == 0:
        return []

    try:
        selector = MRMRSelector(config={
            'relevance_method': 'mutual_info',
            'redundancy_method': 'mutual_info',
            'n_neighbors': 3,
        })
    except Exception as e:
        raise RuntimeError(f"MRMRSelector unavailable: {e}")

    res = selector.select_features(X.values, y, feature_names, n_features=cfg.max_features)
    sel = res.get('selected_features', [])
    if not sel:
        raise RuntimeError("MRMRSelector returned no features")
    return sel

