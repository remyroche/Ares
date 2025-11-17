"""Signal-Boosting Interaction Features

Creates explicit interactions between the top-performing features identified
in multivariate baseline analysis to amplify predictive signal.

Based on multivariate LGBM baseline showing:
- Best triplet: candlestick_doji_pattern + vectorbt_jerk_10_price_returns + sma_10_returns_vwap
- Test R² = 0.154 (vs 0.084 for best single feature)
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def add_signal_boosting_interactions(features: pd.DataFrame) -> pd.DataFrame:
    """
    Add signal-boosting interaction features to an existing feature DataFrame.
    
    This function creates explicit interactions between the top-performing features
    identified in multivariate baseline analysis. Call this after initial feature
    generation to amplify predictive signal.
    
    Args:
        features: DataFrame with base features already generated
        
    Returns:
        DataFrame with original features plus new interaction features
    """
    # Check if required base features exist
    required = ['candlestick_doji_pattern', 'vectorbt_jerk_10_price_returns', 'sma_10_returns_vwap']
    missing = [f for f in required if f not in features.columns]
    
    if missing:
        logger.warning(f"Missing base features for interactions: {missing}. Skipping interaction generation.")
        return features
    
    # Extract base features
    doji = features['candlestick_doji_pattern']
    jerk = features['vectorbt_jerk_10_price_returns']
    sma_vwap = features['sma_10_returns_vwap']
    
    # === Multiplicative Interactions ===
    # Amplify signal when both pattern and momentum agree
    features['doji_x_jerk'] = doji * jerk
    features['doji_x_sma_vwap'] = doji * sma_vwap
    features['jerk_x_sma_vwap'] = jerk * sma_vwap
    
    # Three-way interaction
    features['doji_x_jerk_x_sma_vwap'] = doji * jerk * sma_vwap
    
    # === Ratio Features ===
    # Relative strength of momentum vs trend
    features['jerk_div_sma_vwap'] = jerk / (sma_vwap.abs() + 1e-8)
    
    # === Conditional Features ===
    # Signal only when pattern is active
    features['jerk_when_doji'] = jerk * (doji > 0).astype(float)
    features['sma_vwap_when_doji'] = sma_vwap * (doji > 0).astype(float)
    
    # Signal only when momentum is strong
    jerk_strong = (jerk.abs() > jerk.abs().rolling(20).mean()).astype(float)
    features['doji_when_strong_jerk'] = doji * jerk_strong
    features['sma_vwap_when_strong_jerk'] = sma_vwap * jerk_strong
    
    # === Regime-Aware Interactions ===
    # Compute local volatility regime
    if 'close' in features.columns:
        returns = features['close'].pct_change()
        vol_ma = returns.rolling(20).std()
        vol_median = vol_ma.rolling(100).median()
        is_high_vol = (vol_ma > vol_median).astype(float)
        
        # Features that work differently in high/low vol
        features['jerk_high_vol'] = jerk * is_high_vol
        features['jerk_low_vol'] = jerk * (1 - is_high_vol)
        features['doji_high_vol'] = doji * is_high_vol
        features['doji_low_vol'] = doji * (1 - is_high_vol)
    
    # === Temporal Interactions ===
    # Hour-of-day effects (crypto has strong intraday patterns)
    try:
        hour = features.index.hour
        is_asia = hour.isin(range(0, 8)).astype(float)
        is_europe = hour.isin(range(8, 16)).astype(float)
        is_us = hour.isin(range(16, 24)).astype(float)
        
        features['doji_x_asia'] = doji * is_asia
        features['doji_x_europe'] = doji * is_europe
        features['doji_x_us'] = doji * is_us
        
        features['jerk_x_asia'] = jerk * is_asia
        features['jerk_x_europe'] = jerk * is_europe
        features['jerk_x_us'] = jerk * is_us
    except AttributeError:
        logger.warning("Index does not have hour attribute, skipping temporal interactions")
    
    # === Lagged Interactions ===
    # Capture temporal dependencies
    features['doji_x_jerk_lag1'] = (doji * jerk).shift(1)
    features['doji_x_jerk_lag2'] = (doji * jerk).shift(2)
    
    # === Smoothed Interactions ===
    # Reduce noise via exponential smoothing
    features['doji_x_jerk_ema3'] = (doji * jerk).ewm(span=3, adjust=False).mean()
    features['doji_x_jerk_ema5'] = (doji * jerk).ewm(span=5, adjust=False).mean()
    
    # Clean up NaNs and infinities
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0.0)
    
    logger.info(f"Added {len([c for c in features.columns if c not in required])} interaction features")
    
    return features
