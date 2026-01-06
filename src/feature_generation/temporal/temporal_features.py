"""
Temporal Feature Engineering for Specialist Models

This module provides temporal feature engineering capabilities to add
time-dependent patterns and persistence to specialist model predictions.

Key Features:
- Lagged specialist outputs
- Rolling statistics on predictions
- Momentum features from specialist time series
- Cross-specialist interaction terms
- Time-decay weighted features
- Regime-aware temporal patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler
import warnings

from src.utils.logger import system_logger

logger = system_logger.getChild("TemporalFeatures")


class TemporalFeatureEngineer:
    """
    Temporal feature engineer for enhancing specialist models with time-dependent patterns.
    
    Creates features that capture:
    - Temporal persistence and momentum
    - Cross-specialist temporal interactions
    - Regime-dependent patterns
    - Adaptive time windows
    """
    
    def __init__(
        self,
        lag_periods: List[int] = [1, 2, 3, 5, 10],
        rolling_windows: List[int] = [5, 10, 20, 50],
        momentum_windows: List[int] = [3, 5, 10],
        interaction_lags: List[int] = [1, 2],
        decay_factor: float = 0.95,
        min_periods: int = 3,
    ):
        """
        Initialize temporal feature engineer.
        
        Args:
            lag_periods: List of lag periods for lagged features
            rolling_windows: List of windows for rolling statistics
            momentum_windows: List of windows for momentum calculations
            interaction_lags: Lags for cross-specialist interactions
            decay_factor: Time decay factor for weighted features
            min_periods: Minimum periods for rolling calculations
        """
        self.lag_periods = lag_periods
        self.rolling_windows = rolling_windows
        self.momentum_windows = momentum_windows
        self.interaction_lags = interaction_lags
        self.decay_factor = decay_factor
        self.min_periods = min_periods
        
        # Feature tracking
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def create_temporal_features(
        self,
        specialist_predictions: Dict[str, pd.Series],
        price_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Create comprehensive temporal features from specialist predictions.
        
        Args:
            specialist_predictions: Dict of specialist prediction series
            price_data: Optional OHLCV price data
            regime_data: Optional regime labels
        
        Returns:
            DataFrame with temporal features
        """
        logger.info(f"🕐 Creating temporal features from {len(specialist_predictions)} specialists")
        
        # Align all predictions to common index
        aligned_predictions = self._align_predictions(specialist_predictions)
        if aligned_predictions is None:
            return pd.DataFrame()
        
        temporal_features = pd.DataFrame(index=aligned_predictions.index)
        
        # 1. Lagged features
        lagged_features = self._create_lagged_features(aligned_predictions)
        temporal_features = pd.concat([temporal_features, lagged_features], axis=1)
        
        # 2. Rolling statistics
        rolling_features = self._create_rolling_features(aligned_predictions)
        temporal_features = pd.concat([temporal_features, rolling_features], axis=1)
        
        # 3. Momentum features
        momentum_features = self._create_momentum_features(aligned_predictions)
        temporal_features = pd.concat([temporal_features, momentum_features], axis=1)
        
        # 4. Cross-specialist interactions
        interaction_features = self._create_interaction_features(aligned_predictions)
        temporal_features = pd.concat([temporal_features, interaction_features], axis=1)
        
        # 5. Time-decay weighted features
        decay_features = self._create_decay_features(aligned_predictions)
        temporal_features = pd.concat([temporal_features, decay_features], axis=1)
        
        # 6. Price-based temporal features (if price data available)
        if price_data is not None:
            price_temporal_features = self._create_price_temporal_features(
                aligned_predictions, price_data
            )
            temporal_features = pd.concat([temporal_features, price_temporal_features], axis=1)
        
        # 7. Regime-aware features (if regime data available)
        if regime_data is not None:
            regime_features = self._create_regime_temporal_features(
                aligned_predictions, regime_data
            )
            temporal_features = pd.concat([temporal_features, regime_features], axis=1)
        
        # Clean and validate features
        temporal_features = self._clean_features(temporal_features)
        
        self.feature_names = list(temporal_features.columns)
        
        logger.info(f"✅ Created {len(temporal_features.columns)} temporal features")
        
        return temporal_features
    
    def _align_predictions(self, specialist_predictions: Dict[str, pd.Series]) -> Optional[pd.DataFrame]:
        """Align all specialist predictions to common index."""
        try:
            # Find common index
            all_indices = [pred.index for pred in specialist_predictions.values()]
            common_index = all_indices[0]
            
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            if len(common_index) < 50:
                logger.error(f"Insufficient overlapping samples: {len(common_index)}")
                return None
            
            # Align all predictions
            aligned = {}
            for name, pred in specialist_predictions.items():
                aligned[name] = pred.loc[common_index]
            
            return pd.DataFrame(aligned)
            
        except Exception as e:
            logger.error(f"Failed to align predictions: {e}")
            return None
    
    def _create_lagged_features(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for each specialist."""
        lagged_features = pd.DataFrame(index=predictions.index)
        
        for specialist in predictions.columns:
            for lag in self.lag_periods:
                lag_col = f"{specialist}_lag_{lag}"
                lagged_features[lag_col] = predictions[specialist].shift(lag)
        
        return lagged_features
    
    def _create_rolling_features(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistical features."""
        rolling_features = pd.DataFrame(index=predictions.index)
        
        for specialist in predictions.columns:
            for window in self.rolling_windows:
                # Rolling mean
                mean_col = f"{specialist}_rolling_mean_{window}"
                rolling_features[mean_col] = predictions[specialist].rolling(
                    window=window, min_periods=self.min_periods
                ).mean()
                
                # Rolling standard deviation
                std_col = f"{specialist}_rolling_std_{window}"
                rolling_features[std_col] = predictions[specialist].rolling(
                    window=window, min_periods=self.min_periods
                ).std()
                
                # Rolling min/max
                min_col = f"{specialist}_rolling_min_{window}"
                max_col = f"{specialist}_rolling_max_{window}"
                rolling_features[min_col] = predictions[specialist].rolling(
                    window=window, min_periods=self.min_periods
                ).min()
                rolling_features[max_col] = predictions[specialist].rolling(
                    window=window, min_periods=self.min_periods
                ).max()
                
                # Rolling range
                range_col = f"{specialist}_rolling_range_{window}"
                rolling_features[range_col] = (
                    rolling_features[max_col] - rolling_features[min_col]
                )
                
                # Z-score (rolling normalization)
                zscore_col = f"{specialist}_rolling_zscore_{window}"
                rolling_features[zscore_col] = (
                    (predictions[specialist] - rolling_features[mean_col]) /
                    (rolling_features[std_col] + 1e-8)
                )
        
        return rolling_features
    
    def _create_momentum_features(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create momentum features from specialist predictions."""
        momentum_features = pd.DataFrame(index=predictions.index)
        
        for specialist in predictions.columns:
            for window in self.momentum_windows:
                # Price momentum (rate of change)
                momentum_col = f"{specialist}_momentum_{window}"
                momentum_features[momentum_col] = predictions[specialist].pct_change(window)
                
                # Acceleration (second derivative)
                accel_col = f"{specialist}_acceleration_{window}"
                momentum_features[accel_col_col] = momentum_features[momentum_col].pct_change(window)
                
                # Momentum vs mean (deviation from rolling mean)
                mean_deviation_col = f"{specialist}_momentum_vs_mean_{window}"
                rolling_mean = predictions[specialist].rolling(window=window).mean()
                momentum_features[mean_deviation_col] = (
                    predictions[specialist] - rolling_mean
                ) / (rolling_mean + 1e-8)
        
        return momentum_features
    
    def _create_interaction_features(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create cross-specialist interaction features."""
        interaction_features = pd.DataFrame(index=predictions.index)
        
        specialists = list(predictions.columns)
        
        # Pairwise interactions with lags
        for i, specialist1 in enumerate(specialists):
            for j, specialist2 in enumerate(specialists):
                if i >= j:  # Avoid duplicates and self-interactions
                    continue
                
                for lag in self.interaction_lags:
                    # Correlation-based interaction
                    corr_col = f"{specialist1}_x_{specialist2}_corr_{lag}"
                    interaction_features[corr_col] = (
                        predictions[specialist1].rolling(window=20).corr(
                            predictions[specialist2].shift(lag)
                        )
                    )
                    
                    # Product interaction
                    prod_col = f"{specialist1}_x_{specialist2}_prod_{lag}"
                    interaction_features[prod_col] = (
                        predictions[specialist1] * predictions[specialist2].shift(lag)
                    )
                    
                    # Ratio interaction
                    ratio_col = f"{specialist1}_x_{specialist2}_ratio_{lag}"
                    interaction_features[ratio_col] = (
                        predictions[specialist1] / (predictions[specialist2].shift(lag) + 1e-8)
                    )
        
        return interaction_features
    
    def _create_decay_features(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Create time-decay weighted features."""
        decay_features = pd.DataFrame(index=predictions.index)
        
        for specialist in predictions.columns:
            # Exponential weighted moving average
            ewma_col = f"{specialist}_ewma"
            decay_features[ewma_col] = predictions[specialist].ewm(
                alpha=1 - self.decay_factor, adjust=False
            ).mean()
            
            # Decay-weighted variance
            decay_var_col = f"{specialist}_ewm_var"
            decay_features[decay_var_col] = predictions[specialist].ewm(
                alpha=1 - self.decay_factor, adjust=False
            ).var()
            
            # Recent vs historical ratio
            recent_col = f"{specialist}_recent_vs_historical"
            recent_mean = predictions[specialist].rolling(window=5).mean()
            historical_mean = predictions[specialist].rolling(window=50).mean()
            decay_features[recent_col] = recent_mean / (historical_mean + 1e-8)
        
        return decay_features
    
    def _create_price_temporal_features(
        self, predictions: pd.DataFrame, price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Create price-based temporal features."""
        price_temporal_features = pd.DataFrame(index=predictions.index)
        
        try:
            # Align price data
            common_index = predictions.index.intersection(price_data.index)
            if len(common_index) < 50:
                return price_temporal_features
            
            aligned_prices = price_data.loc[common_index]
            aligned_preds = predictions.loc[common_index]
            
            # Price returns
            if 'close' in aligned_prices.columns:
                returns = aligned_prices['close'].pct_change()
                
                for specialist in predictions.columns:
                    # Specialist vs price returns correlation
                    corr_col = f"{specialist}_price_return_corr"
                    price_temporal_features[corr_col] = (
                        aligned_preds[specialist].rolling(window=20).corr(returns)
                    )
                    
                    # Specialist performance during high volatility
                    vol_threshold = returns.rolling(window=20).std().quantile(0.8)
                    high_vol_mask = returns.rolling(window=20).std() > vol_threshold
                    
                    high_vol_perf_col = f"{specialist}_high_vol_performance"
                    price_temporal_features[high_vol_perf_col] = (
                        aligned_preds[specialist].where(high_vol_mask, np.nan).rolling(window=10).mean()
                    )
            
        except Exception as e:
            logger.warning(f"Failed to create price temporal features: {e}")
        
        return price_temporal_features
    
    def _create_regime_temporal_features(
        self, predictions: pd.DataFrame, regime_data: pd.Series
    ) -> pd.DataFrame:
        """Create regime-aware temporal features."""
        regime_features = pd.DataFrame(index=predictions.index)
        
        try:
            # Align regime data
            common_index = predictions.index.intersection(regime_data.index)
            if len(common_index) < 50:
                return regime_features
            
            aligned_regimes = regime_data.loc[common_index]
            aligned_preds = predictions.loc[common_index]
            
            # Regime-specific statistics
            for specialist in predictions.columns:
                for regime in aligned_regimes.unique():
                    if pd.isna(regime):
                        continue
                    
                    regime_mask = aligned_regimes == regime
                    regime_col = f"{specialist}_regime_{regime}_mean"
                    
                    # Rolling mean within regime
                    regime_features[regime_col] = (
                        aligned_preds[specialist].where(regime_mask, np.nan)
                        .rolling(window=20, min_periods=5)
                        .mean()
                    )
            
        except Exception as e:
            logger.warning(f"Failed to create regime temporal features: {e}")
        
        return regime_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate temporal features."""
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with too many missing values
        missing_threshold = 0.5  # Remove features with >50% missing
        missing_ratios = features.isna().mean()
        valid_features = missing_ratios[missing_ratios <= missing_threshold].index
        features = features[valid_features]
        
        # Forward fill remaining missing values
        features = features.fillna(method='ffill', limit=3)
        
        # Backward fill remaining missing values
        features = features.fillna(method='bfill', limit=3)
        
        # Fill any remaining NaN with 0
        features = features.fillna(0)
        
        return features
    
    def get_feature_importance(
        self, features: pd.DataFrame, target: pd.Series, method: str = "correlation"
    ) -> Dict[str, float]:
        """
        Calculate feature importance for temporal features.
        
        Args:
            features: Temporal features DataFrame
            target: Target series
            method: Importance calculation method
        
        Returns:
            Feature importance scores
        """
        importance_scores = {}
        
        if method == "correlation":
            for feature in features.columns:
                try:
                    corr = features[feature].corr(target)
                    importance_scores[feature] = abs(corr) if not np.isnan(corr) else 0.0
                except:
                    importance_scores[feature] = 0.0
        
        elif method == "mutual_info":
            try:
                from sklearn.feature_selection import mutual_info_regression
                
                # Align features and target
                common_idx = features.index.intersection(target.index)
                X_aligned = features.loc[common_idx]
                y_aligned = target.loc[common_idx]
                
                mi_scores = mutual_info_regression(X_aligned, y_aligned)
                importance_scores = dict(zip(features.columns, mi_scores))
                
            except Exception as e:
                logger.warning(f"Mutual information calculation failed: {e}")
                importance_scores = {feat: 0.0 for feat in features.columns}
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance
    
    def select_top_features(
        self, features: pd.DataFrame, target: pd.Series, n_features: int = 50
    ) -> pd.DataFrame:
        """
        Select top temporal features by importance.
        
        Args:
            features: Temporal features DataFrame
            target: Target series
            n_features: Number of top features to select
        
        Returns:
            Selected features DataFrame
        """
        importance = self.get_feature_importance(features, target)
        
        top_features = list(importance.keys())[:n_features]
        
        logger.info(f"Selected top {len(top_features)} temporal features")
        
        return features[top_features]


def create_temporal_features_batch(
    specialist_predictions: Dict[str, pd.Series],
    price_data: Optional[pd.DataFrame] = None,
    regime_data: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Convenience function for batch temporal feature creation.
    
    Args:
        specialist_predictions: Dict of specialist predictions
        price_data: Optional price data
        regime_data: Optional regime data
        config: Configuration dictionary
    
    Returns:
        Temporal features DataFrame
    """
    if config is None:
        config = {
            "lag_periods": [1, 2, 3, 5],
            "rolling_windows": [5, 10, 20],
            "momentum_windows": [3, 5, 10],
            "interaction_lags": [1, 2],
            "decay_factor": 0.95,
        }
    
    engineer = TemporalFeatureEngineer(**config)
    
    return engineer.create_temporal_features(
        specialist_predictions=specialist_predictions,
        price_data=price_data,
        regime_data=regime_data,
    )
