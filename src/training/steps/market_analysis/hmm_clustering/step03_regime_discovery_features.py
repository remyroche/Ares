
#!/usr/bin/env python3
"""Enhanced Regime Discovery Feature Engineering for Step 3.

This module creates regime-aware features specifically designed for regime discovery,
focusing on features that help distinguish between different market regimes.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from ...utils.regime_feature_utils import RegimeFeatureUtils

warnings.filterwarnings('ignore')

class RegimeDiscoveryFeatureEngineer:
    """Enhanced feature engineering specifically for regime discovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_cache = {}
        
    def create_regime_discovery_features(self, df: pd.DataFrame, existing_regimes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create comprehensive regime discovery features with optimized computation.
        
        Args:
            df: Market data with OHLCV columns
            existing_regimes: Optional existing regime labels for iterative improvement
            
        Returns:
            DataFrame with regime discovery features
        """
        # Pre-compute common calculations for efficiency
        self._precompute_common_features(df)
        
        # Use vectorized operations for all feature creation
        feature_dict = {}
        
        # 1. Regime Transition Prediction Features (vectorized)
        feature_dict.update(self._create_regime_transition_features_vectorized(df))
        
        # 2. Market Microstructure Features (vectorized)
        feature_dict.update(self._create_microstructure_features_vectorized(df))
        
        # 3. Temporal Regime Features (vectorized)
        feature_dict.update(self._create_temporal_regime_features_vectorized(df))
        
        # 4. Volatility Regime Features (vectorized)
        feature_dict.update(self._create_volatility_regime_features_vectorized(df))
        
        # 5. Volume Regime Features (vectorized)
        feature_dict.update(self._create_volume_regime_features_vectorized(df))
        
        # 6. Price Action Regime Features (vectorized)
        feature_dict.update(self._create_price_action_regime_features_vectorized(df))
        
        # 7. Regime Persistence Features (if available)
        if existing_regimes is not None:
            feature_dict.update(self._create_regime_persistence_features_vectorized(df, existing_regimes))
        
        # 8. Regime Strength Features (vectorized)
        feature_dict.update(self._create_regime_strength_features_vectorized(df))
        
        # 9. Regime Change Early Warning Features (vectorized)
        feature_dict.update(self._create_regime_change_warning_features_vectorized(df))
        
        # Convert to DataFrame efficiently
        features = pd.DataFrame(feature_dict, index=df.index)
        
        return features.fillna(0)
    
    def _precompute_common_features(self, df: pd.DataFrame) -> None:
        """Pre-compute common features for efficiency."""
        # Pre-compute price changes
        self.price_changes = df['close'].pct_change()
        self.volume_changes = df['volume'].pct_change()
        
        # Pre-compute rolling statistics
        self.volatility_5 = self.price_changes.rolling(5).std()
        self.volatility_10 = self.price_changes.rolling(10).std()
        self.volatility_20 = self.price_changes.rolling(20).std()
        
        self.volume_mean_5 = df['volume'].rolling(5).mean()
        self.volume_mean_10 = df['volume'].rolling(10).mean()
        self.volume_mean_20 = df['volume'].rolling(20).mean()
        
        # Pre-compute price position
        self.high_20 = df['high'].rolling(20).max()
        self.low_20 = df['low'].rolling(20).min()
        self.price_position = (df['close'] - self.low_20) / (self.high_20 - self.low_20)
        
        # Pre-compute momentum
        self.momentum_5 = df['close'].pct_change(5)
        self.momentum_10 = df['close'].pct_change(10)
        self.momentum_20 = df['close'].pct_change(20)
    
    def _create_regime_transition_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime transition features using vectorized operations."""
        features = {}
        
        # Regime change probability indicators (vectorized)
        features['regime_change_prob_volatility'] = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(
            self.volatility_20, window=10
        )
        features['regime_change_prob_volume'] = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(
            self.volume_mean_20, window=10
        )
        features['regime_change_prob_momentum'] = RegimeFeatureUtils.calculate_regime_change_probability_vectorized(
            self.momentum_10, window=10
        )
        
        # Regime persistence indicators (vectorized)
        features['regime_persistence_volatility'] = RegimeFeatureUtils.calculate_regime_persistence_vectorized(
            self.volatility_20, min_duration=5
        )
        features['regime_persistence_volume'] = RegimeFeatureUtils.calculate_regime_persistence_vectorized(
            self.volume_mean_20, min_duration=5
        )
        
        # Regime transition timing (vectorized)
        features['regime_transition_timing'] = RegimeFeatureUtils.calculate_regime_transition_timing_vectorized(
            df, self.volatility_20, self.volume_mean_20, self.momentum_10
        )
        
        # Regime stability indicators (vectorized)
        features['regime_stability_volatility'] = RegimeFeatureUtils.calculate_regime_stability_vectorized(
            self.volatility_20, window=20
        )
        features['regime_stability_volume'] = RegimeFeatureUtils.calculate_regime_stability_vectorized(
            self.volume_mean_20, window=20
        )
        
        return features
    
    def _create_microstructure_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create microstructure features using vectorized operations."""
        features = {}
        
        # Order flow imbalance (vectorized)
        features['order_flow_imbalance'] = RegimeFeatureUtils.calculate_order_flow_imbalance(df)
        
        # Volume profile analysis (vectorized)
        volume_profile_features = RegimeFeatureUtils.calculate_volume_profile_features(df)
        features.update(volume_profile_features)
        
        # Price impact features (vectorized)
        price_impact_features = RegimeFeatureUtils.calculate_price_impact_features(
            self.price_changes, self.volume_changes
        )
        features.update(price_impact_features)
        
        # Liquidity features (vectorized)
        liquidity_features = RegimeFeatureUtils.calculate_liquidity_features(df)
        features.update(liquidity_features)
        
        return features
    
    def _create_temporal_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create temporal regime features using vectorized operations."""
        features = {}
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return features
        
        # Get temporal features from utility class
        temporal_features = RegimeFeatureUtils.calculate_temporal_features(df)
        features.update(temporal_features)
        
        # Regime duration forecast (vectorized)
        vol_factor = 1 / (1 + self.volatility_20 / self.volatility_20.rolling(50).mean())
        vol_factor_vol = 1 / (1 + self.volume_mean_20 / self.volume_mean_20.rolling(50).mean())
        features['regime_duration_forecast'] = (vol_factor * vol_factor_vol * 20).fillna(20).values
        
        return features
    
    def _create_volatility_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create volatility regime features using vectorized operations."""
        return RegimeFeatureUtils.calculate_volatility_features(self.price_changes)
    
    def _create_volume_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create volume regime features using vectorized operations."""
        return RegimeFeatureUtils.calculate_volume_features(df, self.volume_changes, self.momentum_5)
    
    def _create_price_action_regime_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create price action regime features using vectorized operations."""
        return RegimeFeatureUtils.calculate_price_action_features(df, self.momentum_10, self.volatility_20)
    
    def _create_regime_persistence_features_vectorized(self, df: pd.DataFrame, existing_regimes: np.ndarray) -> Dict[str, np.ndarray]:
        """Create regime persistence features using vectorized operations."""
        features = {}
        
        # Regime duration (vectorized)
        regime_changes = np.diff(existing_regimes) != 0
        regime_duration = np.zeros(len(existing_regimes))
        current_duration = 0
        current_regime = existing_regimes[0]
        
        for i in range(len(existing_regimes)):
            if existing_regimes[i] == current_regime:
                current_duration += 1
            else:
                current_duration = 1
                current_regime = existing_regimes[i]
            regime_duration[i] = current_duration
        
        features['regime_duration'] = regime_duration
        
        # Regime stability score (vectorized)
        stability = np.zeros(len(existing_regimes))
        for i in range(len(existing_regimes)):
            start_idx = max(0, i - 19)
            recent_changes = np.sum(regime_changes[start_idx:i])
            stability[i] = 1 / (1 + recent_changes)
        
        features['regime_stability_score'] = stability
        
        return features
    
    def _create_regime_strength_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime strength features using vectorized operations."""
        features = {}
        
        # Regime strength indicators (vectorized)
        volatility_trend = self.volatility_20.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        vol_of_vol = self.volatility_20.rolling(20).std()
        features['regime_strength_volatility'] = 1 / (1 + vol_of_vol)
        
        volume_consistency = 1 / (1 + df['volume'].rolling(20).std() / df['volume'].rolling(20).mean())
        features['regime_strength_volume'] = volume_consistency
        
        momentum_consistency = 1 / (1 + self.momentum_10.rolling(20).std())
        features['regime_strength_momentum'] = momentum_consistency
        
        # Regime confidence score (vectorized)
        confidence = (features['regime_strength_volatility'] + 
                    features['regime_strength_volume'] +
                    features['regime_strength_momentum']) / 3
        features['regime_confidence_score'] = confidence
        
        return features

    
    def _create_regime_change_warning_features_vectorized(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create regime change warning features using vectorized operations."""
        # Get regime strength features first
        strength_features = self._create_regime_strength_features_vectorized(df)
        regime_confidence_score = strength_features['regime_confidence_score']
        
        return RegimeFeatureUtils.calculate_regime_change_warning_features(
            self.volatility_20, self.volume_mean_20, self.momentum_10, regime_confidence_score
        )
    

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    
    # Generate sample OHLCV data
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
    volumes = np.random.lognormal(10, 1, 1000)
    
    df = pd.DataFrame({
        'open': prices + np.random.randn(1000) * 0.001,
        'high': prices + np.abs(np.random.randn(1000)) * 0.002,
        'low': prices - np.abs(np.random.randn(1000)) * 0.002,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Initialize feature engineer
    engineer = RegimeDiscoveryFeatureEngineer()
    
    # Create regime discovery features
    features = engineer.create_regime_discovery_features(df)
    
    print(f"Created {len(features.columns)} regime discovery features")
    print(f"Feature shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Display feature statistics
    print("\nFeature Statistics:")
    print(features.describe())