"""
Regime Volume Feature Generator

This module provides volume feature generators specifically designed for regime classification,
using robust scaling and normalization techniques.

Key Features:
- Top 10 volume features standardized by RobustScaler
- OBV, CMF, MFI, VWAP deviations with robust normalization
- Volume momentum and oscillator features
- Designed for regime models training

All features are scaled using RobustScaler for robustness to outliers.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Import robust volume transformations
try:
    from src.utils.feature_common.volume_transforms import (
        robust_z_score,
        volume_normalized_by_tr,
        calculate_atr,
        log_volume,
        rolling_median_log_volume,
        calculate_mad
    )
    ROBUST_VOLUME_AVAILABLE = True
except ImportError:
    ROBUST_VOLUME_AVAILABLE = False
    robust_z_score = None
    volume_normalized_by_tr = None
    calculate_atr = None
    log_volume = None
    rolling_median_log_volume = None
    calculate_mad = None
    warnings.warn("Robust volume transforms not available. Volume features will use standard normalization.")

# RobustScaler from scikit-learn
try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RobustScaler = None
    warnings.warn("scikit-learn not available. RobustScaler will not be used.")

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators import OBV, AD, MFI
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    OBV = None
    AD = None
    MFI = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

logger = logging.getLogger(__name__)


class RegimeVolumeFeatureGenerator(VectorizedFeatureGenerator):
    """
    Regime volume feature generator with RobustScaler normalization.

    Generates top 10 volume features specifically designed for regime classification:
    1. OBV (On-Balance Volume)
    2. CMF (Chaikin Money Flow)
    3. MFI (Money Flow Index)
    4. VWAP Deviations (normalized by ATR)
    5. Volume Momentum
    6. Volume Oscillator
    7. Volume ROC (Rate of Change)
    8. Order Flow Imbalance
    9. Volume MA Ratio
    10. Volume Z-Score (Robust)

    All features are scaled using RobustScaler for robustness to outliers.
    """

    def __init__(self,
                 window: int = 20,
                 config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(window)
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

        # Initialize RobustScaler
        if SKLEARN_AVAILABLE and RobustScaler is not None:
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    @classmethod
    def _create_default_config(cls, window: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"regime_volume_{window}",
            category=FeatureCategory.VOLUME,
            description=f"Regime volume features with RobustScaler normalization over {window} periods",
            required_columns=["close", "high", "low", "volume"],
            optional_columns=["open"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={"window": window},
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime volume features."""
        features_dict = self.generate_features(data, **kwargs)

        # Return the first feature as representative (or combine them)
        if features_dict:
            first_feature_name = list(features_dict.keys())[0]
            return features_dict[first_feature_name]
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=f'regime_volume_{self.window}')

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate all regime volume features with RobustScaler normalization."""
        if len(data) < self.window:
            return {}

        features = {}

        try:
            close = data['close']
            high = data['high']
            low = data['low']
            volume = data['volume']

            # 1. OBV (On-Balance Volume) with robust volume
            features['regime_obv'] = self._calculate_obv(close, volume)

            # 2. CMF (Chaikin Money Flow) with robust volume
            features['regime_cmf'] = self._calculate_cmf(high, low, close, volume)

            # 3. MFI (Money Flow Index) with robust volume
            features['regime_mfi'] = self._calculate_mfi(high, low, close, volume)

            # 4. VWAP Deviations (normalized by ATR)
            features['regime_vwap_dev'] = self._calculate_vwap_deviation(high, low, close, volume)

            # 5. Volume Momentum
            features['regime_vol_momentum'] = self._calculate_volume_momentum(volume)

            # 6. Volume Oscillator
            features['regime_vol_oscillator'] = self._calculate_volume_oscillator(volume)

            # 7. Volume ROC (Rate of Change)
            features['regime_vol_roc'] = self._calculate_volume_roc(volume)

            # 8. Order Flow Imbalance
            features['regime_order_flow'] = self._calculate_order_flow_imbalance(close, volume)

            # 9. Volume MA Ratio
            features['regime_vol_ma_ratio'] = self._calculate_volume_ma_ratio(volume)

            # 10. Volume Z-Score (Robust)
            features['regime_vol_zscore'] = self._calculate_robust_volume_zscore(volume)

            # Apply RobustScaler to all features
            if self.scaler is not None:
                features = self._apply_robust_scaler(features, data.index)

        except Exception as e:
            logger.error(f"Error generating regime volume features: {e}")
            # Return empty dict on error
            return {}

        return features

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV with robust volume normalization."""
        if ROBUST_VOLUME_AVAILABLE and robust_z_score is not None:
            volume_normalized = robust_z_score(volume, window=self.window)
            # Shift to positive range
            volume_normalized = volume_normalized - volume_normalized.min() + 1.0
        else:
            volume_normalized = volume

        # Calculate price direction
        price_direction = np.where(close > close.shift(1), 1,
                                  np.where(close < close.shift(1), -1, 0))

        # Calculate OBV
        obv = (price_direction * volume_normalized).cumsum()
        return pd.Series(obv, index=close.index, name='regime_obv')

    def _calculate_cmf(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate CMF with robust volume normalization."""
        if ROBUST_VOLUME_AVAILABLE and robust_z_score is not None:
            volume_normalized = robust_z_score(volume, window=self.window)
            # Shift to positive range
            volume_normalized = volume_normalized - volume_normalized.min() + 1.0
        else:
            volume_normalized = volume

        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)

        # Money Flow Volume
        mfv = mfm * volume_normalized

        # CMF
        cmf = mfv.rolling(self.window).sum() / volume_normalized.rolling(self.window).sum()
        return cmf.fillna(0).rename('regime_cmf')

    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate MFI with robust volume normalization."""
        if VECTORBT_AVAILABLE and MFI is not None:
            try:
                if ROBUST_VOLUME_AVAILABLE and robust_z_score is not None:
                    volume_normalized = robust_z_score(volume, window=self.window)
                    # Shift to positive range
                    volume_normalized = volume_normalized - volume_normalized.min() + 1.0
                else:
                    volume_normalized = volume

                mfi = MFI.run(high, low, close, volume_normalized, self.window)
                return pd.Series(mfi.values, index=close.index, name='regime_mfi')
            except Exception as e:
                logger.warning(f"MFI calculation failed: {e}")
                return pd.Series(np.zeros(len(close)), index=close.index, name='regime_mfi')
        else:
            # Fallback: simple money flow ratio
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(self.window).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(self.window).sum()
            mfi = 100 - (100 / (1 + positive_flow / negative_flow.clip(lower=1e-8)))
            return mfi.fillna(50).rename('regime_mfi')

    def _calculate_vwap_deviation(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate VWAP deviation normalized by ATR."""
        typical_price = (high + low + close) / 3

        # Use volume normalized by ATR
        if ROBUST_VOLUME_AVAILABLE and volume_normalized_by_tr is not None:
            vol_norm = volume_normalized_by_tr(volume, high, low, close, use_atr=True, atr_window=14)
        else:
            vol_norm = volume

        # VWAP
        vwap = (typical_price * vol_norm).rolling(self.window).sum() / vol_norm.rolling(self.window).sum()

        # Calculate ATR for normalization
        if ROBUST_VOLUME_AVAILABLE and calculate_atr is not None:
            atr = calculate_atr(high, low, close, window=14)
        else:
            atr = (high - low).ewm(span=14).mean()

        # VWAP deviation normalized by ATR
        vwap_dev = (close - vwap) / atr.clip(lower=1e-8)
        return vwap_dev.fillna(0).rename('regime_vwap_dev')

    def _calculate_volume_momentum(self, volume: pd.Series) -> pd.Series:
        """Calculate volume momentum."""
        vol_ma_short = volume.rolling(self.window // 2).mean()
        vol_ma_long = volume.rolling(self.window).mean()
        momentum = (vol_ma_short - vol_ma_long) / vol_ma_long.clip(lower=1e-8)
        return momentum.fillna(0).rename('regime_vol_momentum')

    def _calculate_volume_oscillator(self, volume: pd.Series) -> pd.Series:
        """Calculate volume oscillator."""
        vol_ema_short = volume.ewm(span=self.window // 2).mean()
        vol_ema_long = volume.ewm(span=self.window).mean()
        oscillator = (vol_ema_short - vol_ema_long) / vol_ema_long.clip(lower=1e-8)
        return oscillator.fillna(0).rename('regime_vol_oscillator')

    def _calculate_volume_roc(self, volume: pd.Series) -> pd.Series:
        """Calculate volume rate of change."""
        roc = volume.pct_change(periods=self.window // 4)
        return roc.fillna(0).rename('regime_vol_roc')

    def _calculate_order_flow_imbalance(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate order flow imbalance (signed volume)."""
        price_change = close.diff()
        signed_volume = volume * np.sign(price_change)
        ofi = signed_volume.rolling(self.window).sum()
        return ofi.fillna(0).rename('regime_order_flow')

    def _calculate_volume_ma_ratio(self, volume: pd.Series) -> pd.Series:
        """Calculate volume to moving average ratio."""
        vol_ma = volume.rolling(self.window).mean()
        ratio = volume / vol_ma.clip(lower=1e-8)
        return ratio.fillna(1).rename('regime_vol_ma_ratio')

    def _calculate_robust_volume_zscore(self, volume: pd.Series) -> pd.Series:
        """Calculate robust z-score for volume."""
        if ROBUST_VOLUME_AVAILABLE and robust_z_score is not None:
            return robust_z_score(volume, window=self.window).rename('regime_vol_zscore')
        else:
            # Fallback: standard z-score
            vol_mean = volume.rolling(self.window).mean()
            vol_std = volume.rolling(self.window).std()
            zscore = (volume - vol_mean) / vol_std.clip(lower=1e-8)
            return zscore.fillna(0).rename('regime_vol_zscore')

    def _apply_robust_scaler(self, features: Dict[str, pd.Series], index: pd.Index) -> Dict[str, pd.Series]:
        """Apply RobustScaler to all features."""
        if self.scaler is None:
            return features

        scaled_features = {}

        for name, feature in features.items():
            try:
                # Convert to numpy array
                feature_values = feature.values.reshape(-1, 1)

                # Handle NaN values
                mask = ~np.isnan(feature_values).flatten()
                if mask.sum() == 0:
                    # All NaN, return zeros
                    scaled_features[name] = pd.Series(np.zeros(len(feature)), index=index, name=name)
                    continue

                # Fit and transform using RobustScaler
                scaled_values = np.full_like(feature_values, np.nan, dtype=float)
                if mask.sum() > 10:  # Need at least 10 non-NaN values for robust scaling
                    self.scaler.fit(feature_values[mask])
                    scaled_values[mask] = self.scaler.transform(feature_values[mask]).flatten()

                # Create scaled series
                scaled_features[name] = pd.Series(scaled_values.flatten(), index=index, name=name)

            except Exception as e:
                logger.warning(f"Failed to apply RobustScaler to {name}: {e}")
                scaled_features[name] = feature

        return scaled_features


def create_regime_volume_generators(windows: List[int] = None) -> List[FeatureGenerator]:
    """
    Create regime volume feature generators for different windows.

    Args:
        windows: List of window sizes. Default: [14, 20, 30]

    Returns:
        List of RegimeVolumeFeatureGenerator instances
    """
    if windows is None:
        windows = [14, 20, 30]

    generators = []
    for window in windows:
        generator = RegimeVolumeFeatureGenerator(window=window)
        generators.append(generator)

    return generators


__all__ = [
    'RegimeVolumeFeatureGenerator',
    'create_regime_volume_generators',
]
