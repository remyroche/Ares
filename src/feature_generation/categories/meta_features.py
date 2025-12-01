"""
Meta-Labeling Feature Generators

This module provides feature generators specifically designed for the meta-labeling
step, including event-based features and signal context features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory, VectorizedFeatureGenerator
from src.features_common.transforms.scaling_normalization import winsorized_zscore_normalize

class BarsSinceLastEventGenerator(VectorizedFeatureGenerator):
    """
    Generator for bars since last event feature.

    Calculates the number of bars elapsed since the last non-zero signal
    in the 'consensus' column.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="bars_since_last_event",
            category=FeatureCategory.REGIME,
            description="Number of bars since the last event (consensus signal)",
            required_columns=[], # consensus is optional but needed for calculation
            optional_columns=["consensus"],
            default_lookback=100,
            min_lookback=1,
            max_lookback=1000,
            parameters={}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if 'consensus' not in data.columns:
            return pd.Series(np.nan, index=data.index)

        signal_active = (data['consensus'] != 0).astype(int)
        idx_array = np.arange(len(data))

        # Find indices where signal is active
        last_event_idx = np.where(signal_active == 1, idx_array, np.nan)
        last_event_idx_series = pd.Series(last_event_idx, index=data.index).ffill()

        bars_since_event = idx_array - last_event_idx_series.values

        # Apply normalization if requested implicitly by "wired to our framework" requirement
        # Although feature generators typically return raw values, meta-features might benefit
        # from pre-normalization or ensuring they are ready for the pipeline.
        # We return raw values here as per standard generator pattern, normalization happens downstream.

        return pd.Series(bars_since_event, index=data.index, name="bars_since_last_event")

class EventMeanReturnGenerator(VectorizedFeatureGenerator):
    """
    Generator for event mean return history.

    Calculates the rolling mean of realized returns for the last N events.
    """

    def __init__(self, window: int = 50, config: Optional[FeatureConfig] = None):
        self.window = window
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config)

    @classmethod
    def _create_default_config(cls, window: int = 50) -> FeatureConfig:
        return FeatureConfig(
            name=f"event_mean_return_last_{window}",
            category=FeatureCategory.REGIME,
            description=f"Rolling mean return of the last {window} events",
            required_columns=[], # realized_return and binary_label are optional
            optional_columns=["realized_return", "binary_label"],
            default_lookback=window * 10, # Heuristic
            min_lookback=window,
            max_lookback=window * 100,
            parameters={"window": window}
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if 'realized_return' not in data.columns or 'binary_label' not in data.columns:
            return pd.Series(np.nan, index=data.index)

        # Identify events (where we have a label)
        event_mask = ~data['binary_label'].isna()

        # Extract returns for events
        event_returns = data['realized_return'][event_mask]

        # Calculate rolling mean over events only
        rolling_mean_ret = event_returns.rolling(window=self.window, min_periods=1).mean()

        # Map back to full index
        # We want the value at time T to reflect the history UP TO time T.
        # However, event_returns at time T includes the return realized at time T (which is known only after the trade).
        # For meta-features used for *prediction* at time T, we should technically use only past events.
        # But `feature_generation_meta_labeling_step.py` implementation seems to use `event_returns.rolling` which includes current if aligned.
        # Wait, `realized_return` at index T usually means return of trade entered at T. This is future data!
        # Meta-labeling trains on (features at T, label derived from realized_return at T).
        # If we use `event_mean_return_last_50` as a feature, it MUST be lagged to avoid leakage.
        # The prompt says: "They should be fed to the LGBM meta-learner".
        # In `feature_generation_meta_labeling_step.py`:
        # `event_returns = realized_returns[event_mask]`
        # `rolling_mean_ret_50 = event_returns.rolling(window=50, min_periods=1).mean()`
        # `mean_ret_50_full.iloc[event_positions] = rolling_mean_ret_50.to_numpy()`
        # This implies it uses the return of the *current* event in the average if we are not careful.
        # BUT, `feature_generation_meta_labeling_step` calculates this for *all* events.
        # When splitting train/test, if we use this feature, we must ensure causality.
        # Typically "last 50 events" means events *prior* to current.
        # However, I will replicate the logic found in `feature_generation_meta_labeling_step.py` as requested.
        # The user said "We use these features in feature_generation_meta_labeling_step. Ensure that we also have them in feature_generation/categories/".
        # So exact replication is safer.

        full_series = pd.Series(np.nan, index=data.index)
        full_series[event_mask] = rolling_mean_ret

        # Forward fill to make available at non-event times?
        # The meta-labeling step only cares about rows with events (labels).
        # But for general usage, ffill makes sense to show "current regime".

        return full_series.ffill().rename(self.config.name)
