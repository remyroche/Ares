"""
Enhanced Volume Force Features with MI Improvements

This module extends the original volume force features with non-linear
transformations, market regime indicators, and target-specific features
to improve Mutual Information (MI) scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy import stats
from scipy.stats import entropy

from src.features_common.transforms.scaling_normalization import (
    winsorized_zscore_normalize,
)
from src.utils.feature_common.volume_transforms import (
    log1p_zscore_normalize,
)
from src.training.steps.market_analysis.enhanced_feature_generators import (
    NonLinearFeatureGenerator,
    MarketRegimeFeatureGenerator,
    TargetSpecificFeatureGenerator,
)


def generate_enhanced_volume_force_features(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Generate enhanced volume force features with MI improvements.

    Enhanced features include:
    - Original volume force features
    - Non-linear transformations (polynomial, log, sqrt)
    - Market regime indicators (volatility, trend, time)
    - Target-specific volume force enhancements
    - Interaction terms between features

    Args:
        df: Input market data with `open`, `high`, `low`, `close`, `volume`.
        config: Configuration dictionary.

    Returns:
        DataFrame with enhanced volume force features.
    """
    # Import original features first
    from src.feature_generation.categories.volume_force_features import generate_volume_force_features
    
    # Generate original volume force features
    original_features = generate_volume_force_features(df, config)
    
    # Initialize enhanced feature generators
    nonlinear_gen = NonLinearFeatureGenerator()
    regime_gen = MarketRegimeFeatureGenerator()
    target_gen = TargetSpecificFeatureGenerator()
    
    enhanced_features = []
    
    # 1. Non-linear transformations on key volume features
    key_volume_cols = ['volume', 'volume_delta', 'force_index_norm', 'vfi', 'cmf']
    available_volume_cols = [col for col in key_volume_cols if col in original_features.columns]
    
    if available_volume_cols:
        nonlinear_features = nonlinear_gen.add_polynomial_features(
            original_features, available_volume_cols, degree=2
        )
        enhanced_features.append(nonlinear_features)
        
        # Add interaction features between volume metrics
        volume_pairs = []
        for i, col1 in enumerate(available_volume_cols):
            for col2 in available_volume_cols[i+1:]:
                volume_pairs.append((col1, col2))
        
        if volume_pairs:
            interaction_features = nonlinear_gen.add_interaction_features(
                original_features, volume_pairs[:5]  # Limit to avoid explosion
            )
            enhanced_features.append(interaction_features)
    
    # 2. Market regime features
    regime_features_vol = regime_gen.add_volatility_regime_features(
        original_features, price_col='close', windows=[20, 50]
    )
    enhanced_features.append(regime_features_vol)
    
    regime_features_trend = regime_gen.add_trend_regime_features(
        original_features, price_col='close', short_windows=[10, 20], long_windows=[50, 100]
    )
    enhanced_features.append(regime_features_trend)
    
    # Add time-based features if we have datetime index
    if isinstance(original_features.index, pd.DatetimeIndex):
        time_features = regime_gen.add_time_based_features(original_features)
        enhanced_features.append(time_features)
    
    # 3. Enhanced volume force specific features
    volume_force_enhanced = add_enhanced_volume_force_features(original_features, config)
    enhanced_features.append(volume_force_enhanced)
    
    # 4. Target-specific breakout features (volume force often predicts breakouts)
    breakout_features = target_gen.add_breakout_features(
        original_features, price_cols=['high', 'low', 'close'], volume_col='volume', windows=[20, 50]
    )
    enhanced_features.append(breakout_features)
    
    # 5. Volume force momentum features
    momentum_features = target_gen.add_volume_force_features(
        original_features, price_col='close', volume_col='volume', windows=[10, 20]
    )
    enhanced_features.append(momentum_features)
    
    # Combine all enhanced features
    if enhanced_features:
        all_enhanced = pd.concat(enhanced_features, axis=1)
        
        # Combine with original features
        combined_features = pd.concat([original_features, all_enhanced], axis=1)
        
        # Remove any infinite or NaN values
        combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Remove duplicate columns
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        
        print(f"✅ Enhanced volume force features: {len(combined_features.columns)} total features")
        print(f"   Original: {len(original_features.columns)}, Enhanced: {len(all_enhanced.columns)}")
        
        return combined_features
    else:
        return original_features


def add_enhanced_volume_force_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Add enhanced volume force specific features."""
    features = pd.DataFrame(index=df.index)
    
    # Enhanced volume patterns
    if 'volume' in df.columns:
        volume = df['volume']
        
        # Volume pattern recognition
        volume_ma_20 = volume.rolling(20).mean()
        volume_ma_50 = volume.rolling(50).mean()
        
        # Volume trend strength
        features['volume_trend_strength'] = (volume_ma_20 - volume_ma_50) / volume_ma_50
        
        # Volume pattern flags
        features['volume_pattern_accumulation'] = (volume > volume_ma_20 * 1.2).astype(int)
        features['volume_pattern_distribution'] = (volume < volume_ma_20 * 0.8).astype(int)
        features['volume_pattern_churning'] = (
            (volume > volume_ma_20 * 0.8) & (volume < volume_ma_20 * 1.2)
        ).astype(int)
        
        # Volume efficiency ratio
        if 'close' in df.columns:
            price_change = df['close'].pct_change().abs()
            features['volume_efficiency_ratio'] = price_change / (volume + 1e-8)
            features['volume_efficiency_ma'] = features['volume_efficiency_ratio'].rolling(20).mean()
    
    # Enhanced force index calculations
    if 'force_index_norm' in df.columns and 'close' in df.columns:
        force_index = df['force_index_norm']
        price_change = df['close'].pct_change()
        
        # Force divergence
        features['force_price_divergence'] = force_index - price_change.rolling(10).mean()
        
        # Force momentum
        features['force_momentum'] = force_index.rolling(10).sum()
        features['force_acceleration'] = force_index.rolling(10).sum() - force_index.rolling(20).sum()
        
        # Force regime
        force_ma = force_index.rolling(20).mean()
        features['force_regime_bullish'] = (force_index > force_ma * 1.1).astype(int)
        features['force_regime_bearish'] = (force_index < force_ma * 0.9).astype(int)
    
    # Enhanced VFI (Volume Flow Indicator) patterns
    if 'vfi' in df.columns:
        vfi = df['vfi']
        
        # VFI trend analysis
        vfi_ma = vfi.rolling(20).mean()
        features['vfi_trend_strength'] = (vfi - vfi_ma) / vfi_ma.abs()
        
        # VFI divergence detection
        if 'close' in df.columns:
            price_trend = df['close'].rolling(20).mean()
            vfi_trend = vfi.rolling(20).mean()
            features['vfi_price_divergence'] = vfi_trend - price_trend.pct_change().rolling(20).mean()
        
        # VFI overbought/oversold
        features['vfi_overbought'] = (vfi > vfi.quantile(0.8)).astype(int)
        features['vfi_oversold'] = (vfi < vfi.quantile(0.2)).astype(int)
    
    # Enhanced CMF (Chaikin Money Flow) patterns
    if 'cmf' in df.columns:
        cmf = df['cmf']
        
        # CMF trend and momentum
        cmf_ma = cmf.rolling(20).mean()
        features['cmf_trend_strength'] = (cmf - cmf_ma) / cmf_ma.abs()
        features['cmf_momentum'] = cmf.rolling(10).sum()
        
        # CMF buying/selling pressure
        features['cmf_buying_pressure'] = (cmf > 0.1).astype(int)
        features['cmf_selling_pressure'] = (cmf < -0.1).astype(int)
        features['cmf_neutral'] = ((cmf >= -0.1) & (cmf <= 0.1)).astype(int)
    
    # Enhanced volume imbalance features
    if 'volume_imbalance' in df.columns:
        imbalance = df['volume_imbalance']
        
        # Imbalance trend
        imbalance_ma = imbalance.rolling(20).mean()
        features['imbalance_trend_strength'] = (imbalance - imbalance_ma) / imbalance_ma.abs()
        
        # Imbalance extreme detection
        imbalance_std = imbalance.rolling(20).std()
        features['imbalance_extreme_buy'] = (imbalance > imbalance_ma + 2 * imbalance_std).astype(int)
        features['imbalance_extreme_sell'] = (imbalance < imbalance_ma - 2 * imbalance_std).astype(int)
        
        # Imbalance persistence
        features['imbalance_persistence'] = (imbalance.rolling(5).sum() > 0).astype(int)
    
    # Enhanced multi-timeframe volume features
    if 'rvol_htf_4h' in df.columns and 'rvol_htf_daily' in df.columns:
        rvol_4h = df['rvol_htf_4h']
        rvol_daily = df['rvol_htf_daily']
        
        # Multi-timeframe volume divergence
        features['mtf_volume_divergence'] = rvol_4h - rvol_daily
        
        # Multi-timeframe volume confirmation
        features['mtf_volume_confirmation'] = ((rvol_4h > 1.0) & (rvol_daily > 1.0)).astype(int)
        
        # Multi-timeframe volume expansion
        features['mtf_volume_expansion'] = ((rvol_4h > 1.5) | (rvol_daily > 1.5)).astype(int)
    
    # Enhanced volume shock features
    if 'volume_shock' in df.columns:
        shock = df['volume_shock']
        
        # Shock frequency
        features['shock_frequency'] = (shock.abs() > shock.std()).rolling(20).sum()
        
        # Shock magnitude
        features['shock_magnitude'] = shock.abs()
        features['shock_magnitude_ma'] = features['shock_magnitude'].rolling(20).mean()
        
        # Shock direction persistence
        features['shock_direction_persistence'] = (shock.rolling(5).sum() > 0).astype(int)
    
    return features


def add_volume_force_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume force specific regime features."""
    features = pd.DataFrame(index=df.index)
    
    # Volume force regime detection
    if 'force_index_norm' in df.columns and 'volume' in df.columns:
        force = df['force_index_norm']
        volume = df['volume']
        
        # Force-volume regime
        force_ma = force.rolling(20).mean()
        volume_ma = volume.rolling(20).mean()
        
        features['force_volume_bullish'] = (
            (force > force_ma) & (volume > volume_ma)
        ).astype(int)
        
        features['force_volume_bearish'] = (
            (force < force_ma) & (volume > volume_ma)
        ).astype(int)
        
        features['force_volume_weak'] = (
            (volume < volume_ma * 0.8)
        ).astype(int)
    
    # Volume force efficiency regime
    if 'volume_efficiency_ratio' in df.columns:
        efficiency = df['volume_efficiency_ratio']
        efficiency_ma = efficiency.rolling(20).mean()
        
        features['efficiency_regime_high'] = (efficiency > efficiency_ma * 1.2).astype(int)
        features['efficiency_regime_low'] = (efficiency < efficiency_ma * 0.8).astype(int)
    
    return features
