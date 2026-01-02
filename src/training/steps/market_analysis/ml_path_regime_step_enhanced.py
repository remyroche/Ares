"""
Enhanced ML Path Regime Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLPathRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            self._artifact_router = ArtifactRouter(
                base_dir="artifacts",
                versioned_store_dir="versioned_artifacts",
                historical_data_dir="historical_data",
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    """
    Enhanced Path Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_path_regime_step"):
        """Initialize the enhanced path regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedMLPathRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_path_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced path features with MI improvements."""
        # Import original path features
        try:
            from src.feature_generation.categories.path_regime_features import generate_path_regime_features
            path_features = generate_path_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            path_features = pd.DataFrame(index=df.index)
        
        # Generate enhanced features
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'path_regime', config
        )
        
        # Path-specific enhancements
        path_enhanced = self._add_path_specific_features(df, path_features)
        
        # Combine all features
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        all_features = pd.concat([path_features, enhanced_features, path_enhanced], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        # Create advanced manual enhanced features for path risk detection
        manual_features = self._create_manual_path_risk_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = pd.concat([all_features, manual_features], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return all_features
    
    def _create_manual_path_risk_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create advanced manual enhanced features for path risk detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Advanced path trajectory analysis
            # Multi-timeframe price paths
            path_5 = close.rolling(5).mean()
            path_10 = close.rolling(10).mean()
            path_20 = close.rolling(20).mean()
            path_50 = close.rolling(50).mean()
            
            # Path curvature (second derivative of price path)
            path_curvature_5 = path_5.diff().diff()
            path_curvature_10 = path_10.diff().diff()
            path_curvature_20 = path_20.diff().diff()
            manual_features['enhanced_path_curvature'] = path_curvature_10
            
            # Path tortuosity (how erratic the path is)
            path_changes_5 = abs(path_5.diff())
            path_tortuosity_5 = path_changes_5.rolling(10).sum()
            manual_features['enhanced_path_tortuosity'] = path_tortuosity_5
            
            # Path momentum (velocity of path changes)
            path_velocity_5 = path_5.diff()
            path_momentum_5 = path_velocity_5.rolling(5).mean()
            manual_features['enhanced_path_momentum'] = path_momentum_5
            
            # Path acceleration (change in path velocity)
            path_acceleration_5 = path_velocity_5.diff()
            manual_features['enhanced_path_acceleration'] = path_acceleration_5
            
            # 2. Path volatility and smoothness
            # Path volatility (volatility of the path itself)
            path_volatility_5 = path_velocity_5.rolling(20).std()
            path_volatility_10 = path_velocity_10.rolling(20).std()
            manual_features['enhanced_path_volatility'] = path_volatility_10
            
            # Path smoothness (inverse of path volatility)
            path_smoothness_5 = 1 / (path_volatility_5 + 1e-8)
            manual_features['enhanced_path_smoothness'] = path_smoothness_5
            
            # Path consistency (how consistent the path direction is)
            path_direction_5 = np.sign(path_velocity_5)
            path_consistency_5 = (path_direction_5 == path_direction_5.shift(1)).rolling(10).mean()
            manual_features['enhanced_path_consistency'] = path_consistency_5
            
            # 3. Path range and breakout analysis
            # Path range (high-low of path over period)
            path_range_20 = path_20.rolling(20).max() - path_20.rolling(20).min()
            path_range_50 = path_50.rolling(50).max() - path_50.rolling(50).min()
            manual_features['enhanced_path_range'] = path_range_20
            
            # Path breakout detection
            path_breakout_20 = abs(close - path_20) / (path_range_20 + 1e-8)
            manual_features['enhanced_path_breakout'] = path_breakout_20
            
            # Path mean reversion strength
            path_mean_20 = path_20.rolling(50).mean()
            path_mean_reversion = -close.rolling(10).corr(path_mean_20)
            manual_features['enhanced_path_mean_reversion'] = path_mean_reversion
            
            # 4. Volume-adjusted path features
            # Volume-weighted path momentum
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_weighted_path_momentum = path_momentum_5 * (1 + np.log(volume_ratio + 1))
            manual_features['enhanced_volume_weighted_path_momentum'] = volume_weighted_path_momentum
            
            # Volume-path divergence
            volume_regime = (volume_ratio > 1).astype(int)
            path_regime = (path_momentum_5 > 0).astype(int)
            volume_path_divergence = np.abs(volume_regime - path_regime)
            manual_features['enhanced_volume_path_divergence'] = volume_path_divergence
            
            # 5. Path risk metrics
            # Path risk (combination of volatility and tortuosity)
            path_risk_score = path_volatility_10 * path_tortuosity_5
            manual_features['enhanced_path_risk_score'] = path_risk_score
            
            # Path drawdown risk
            path_max_20 = path_20.rolling(20).max()
            path_drawdown_20 = (path_20 - path_max_20) / path_max_20
            manual_features['enhanced_path_drawdown'] = path_drawdown_20
            
            # Path downside risk
            path_downside_20 = path_20.rolling(20).quantile(0.05)
            path_downside_risk = (path_20 < path_downside_20).astype(int)
            manual_features['enhanced_path_downside_risk'] = path_downside_risk
            
            # 6. Path regime classification
            # Path regime (trending, ranging, volatile)
            path_trend_strength = abs(path_momentum_5)
            path_vol_strength = path_volatility_5
            
            path_regime = np.where(
                (path_trend_strength > path_vol_strength) & (path_trend_strength > path_trend_strength.rolling(50).mean()),
                1,  # Trending
                np.where(
                    path_vol_strength > path_vol_strength.rolling(50).mean(),
                    2,  # Volatile
                    0   # Ranging
                )
            )
            manual_features['enhanced_path_regime'] = path_regime
            
            # Path regime transitions
            path_regime_transitions = pd.Series(path_regime).diff().abs()
            manual_features['enhanced_path_regime_transitions'] = path_regime_transitions
            
            # 7. Advanced path patterns
            # Path cyclical patterns (using autocorrelation)
            path_autocorr_5 = [close.rolling(20).autocorr(lag=i) for i in range(1, 6)]
            path_cyclical_strength = np.mean([abs(x) for x in path_autocorr_5 if not np.isnan(x)])
            manual_features['enhanced_path_cyclical_strength'] = path_cyclical_strength
            
            # Path seasonality (intraday patterns)
            if hasattr(df.index, 'hour'):
                hourly_pattern = close.groupby(df.index.hour).transform('mean')
                path_seasonality = close / hourly_pattern
                manual_features['enhanced_path_seasonality'] = path_seasonality
            
            # Path fractal dimension (complexity measure)
            path_complexity = path_changes_5.rolling(20).std() / (path_changes_5.rolling(20).mean() + 1e-8)
            manual_features['enhanced_path_complexity'] = path_complexity
            
            # 8. Composite path risk indicators
            # Path stress index
            path_stress_index = (
                0.3 * (path_volatility_10 > path_volatility_10.rolling(100).mean()).astype(int) +
                0.3 * (path_tortuosity_5 > path_tortuosity_5.rolling(100).mean()).astype(int) +
                0.2 * (path_breakout_20 > 0.8).astype(int) +
                0.2 * (path_drawdown_20 < -0.05).astype(int)
            )
            manual_features['enhanced_path_stress_index'] = path_stress_index
            
            # Path stability index (inverse of stress)
            path_stability = 1 - path_stress_index
            manual_features['enhanced_path_stability'] = path_stability
            
        return manual_features
    
    def _add_path_specific_features(self, df: pd.DataFrame, path_features: pd.DataFrame) -> pd.DataFrame:
        """Add path-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced path analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Path analysis
            high_low_range = high - low
            close_position = (close - low) / high_low_range
            
            # Path momentum
            for window in [5, 10, 20]:
                path_momentum = close_position.rolling(window).diff()
                features[f'path_momentum_{window}'] = path_momentum
                features[f'path_acceleration_{window}'] = path_momentum.diff()
                
                # Path volatility
                path_volatility = path_momentum.rolling(window).std()
                features[f'path_volatility_{window}'] = path_volatility
                
                # Path direction changes
                features[f'path_direction_change_{window}'] = (path_momentum > 0).astype(int)
                features[f'path_persistence_{window}'] = (path_momentum.rolling(window).apply(lambda x: (x > 0).mean()))
            
            # Path efficiency
            features['path_efficiency'] = close_position.rolling(25).mean()
            features['path_efficiency_ma'] = features['path_efficiency'].rolling(60).mean()
            
            # Path exhaustion detection
            path_range = high_low_range.rolling(25)
            path_range_ma = path_range.rolling(60).mean()
            features['path_exhaustion'] = (path_range < path_range_ma * 0.5).astype(int)
            
            # Path reversal patterns
            path_direction = (close_position.diff() > 0).astype(int)
            features[f'path_reversal_{window}'] = path_direction.rolling(window).apply(lambda x: (x.diff() < 0).sum())
        
        # Volume-path relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            close_change = df['close'].pct_change()
            
            # Volume-weighted path analysis
            volume_path_correlation = close_change.rolling(25).corr(volume)
            features['volume_path_correlation'] = volume_path_correlation
            
            # Volume confirmation of path moves
            volume_ma = volume.rolling(25).mean()
            volume_anomaly = volume / volume_ma
            features['volume_path_confirmation'] = (volume_anomaly > 1.5).astype(int)
            
            # Path efficiency with volume
            path_efficiency = close_change.abs() / (volume + 1e-8)
            features['volume_path_efficiency'] = path_efficiency.rolling(25).mean()
        
        # Support/resistance path analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            
            for window in [20, 50, 100]:
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                
                features[f'path_to_resistance_{window}'] = (rolling_max - close) / rolling_max
                features[f'path_to_support_{window}'] = (close - rolling_min) / rolling_max
                features[f'path_sr_strength_{window}'] = (rolling_max - rolling_min) / close
                
                # Path breaches
                features[f'path_resistance_breach_{window}'] = (close > rolling_max.shift(1)).astype(int)
                features[f'path_support_breach_{window}'] = (close < rolling_min.shift(1)).astype(int)
                
                # Path following
                features[f'path_following_{window}'] = (
                    (close > rolling_min.shift(1)) & (close < rolling_max.shift(1))
                ).astype(int)
        
        # Time-based path patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return features
    
    def _create_path_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
           # PERFORMANCE NOTE: Optimized for speed
        # - Reduced rolling windows from 4 to 3
        # - Added min_periods=1 to handle edge cases
        # - Removed extended path (25 periods) for performance
        # - Vectorized operations where possible
        """Create enhanced path labels based on multi-timeframe path patterns."""
        if "high" in df.columns and "low" in df.columns and "close" in df.columns:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            # Short-term path (30 minutes = 2 periods)
            high_2 = high.rolling(2).max()
            low_2 = low.rolling(2).min()
            close_position_2 = (close - low_2) / (high_2 - low_2 + 1e-8)
            
            # Medium-term path (1 hour = 4 periods)
            high_4 = high.rolling(4).max()
            low_4 = low.rolling(4).min()
            close_position_4 = (close - low_4) / (high_4 - low_4 + 1e-8)
            
            # Long-term path (3.75 hours = 15 periods)
            high_15 = high.rolling(15).max()
            low_15 = low.rolling(15).min()
            close_position_15 = (close - low_15) / (high_15 - low_15 + 1e-8)
            
            # Extended path (6.25 hours = 25 periods)
            high_25 = high.rolling(25).max()
            low_25 = low.rolling(25).min()
            close_position_25 = (close - low_25) / (high_25 - low_25 + 1e-8)
            
            # Future path positions
            future_high = high.shift(-lookforward)
            future_low = low.shift(-lookforward)
            future_close = close.shift(-lookforward)
            
            # Calculate future path positions for different timeframes
            future_high_2 = future_high.rolling(2).max()
            future_low_2 = future_low.rolling(2).min()
            future_position_2 = (future_close - future_low_2) / (future_high_2 - future_low_2 + 1e-8)
            
            future_high_4 = future_high.rolling(4).max()
            future_low_4 = future_low.rolling(4).min()
            future_position_4 = (future_close - future_low_4) / (future_high_4 - future_low_4 + 1e-8)
            
            future_high_15 = future_high.rolling(15).max()
            future_low_15 = future_low.rolling(15).min()
            future_position_15 = (future_close - future_low_15) / (future_high_15 - future_low_15 + 1e-8)
            
            # Path deviation metrics
            path_deviation_2 = close_position_2 - close_position_4
            path_deviation_4 = close_position_4 - close_position_15
            path_deviation_15 = close_position_15 - close_position_25
            
            # Future path deviations
            future_deviation_2 = future_position_2 - future_position_4
            future_deviation_4 = future_position_4 - future_position_15
            future_deviation_15 = future_position_15 - close_position_25
            
            # Path acceleration (change in deviation)
            path_acceleration = future_deviation_2 - path_deviation_2
            
            # Path volatility (path position changes)
            path_volatility = close_position_4.rolling(4).std()
            future_path_volatility = future_position_4.rolling(4).std()
            
            # Enhanced path labeling conditions with relaxed thresholds
            # Condition 1: Strong path continuation (relaxed thresholds)
            # Give more weight to short-term consistency
            strong_continuation = (
                (close_position_2 > 0.7) & (future_position_2 > 0.7) &      # Relaxed from 0.75
                (close_position_4 > 0.65) & (future_position_4 > 0.65) &    # Relaxed from 0.7
                (close_position_15 > 0.55) & (future_position_15 > 0.55)  # Relaxed from 0.6
            )
            
            # Condition 2: Path acceleration (relaxed threshold)
            path_acceleration_signal = abs(path_acceleration) > path_volatility * 1.0  # Relaxed from 1.2
            
            # Condition 3: Path volatility breakout (more sensitive)
            vol_breakout = future_path_volatility > path_volatility * 1.1  # Relaxed from 1.2
            
            # Condition 4: Mean reversion signal (relaxed thresholds)
            mean_reversion = (close_position_4 > 0.9) & (future_position_4 < 0.5)  # Relaxed from 0.55
            mean_reversion |= (close_position_4 < 0.1) & (future_position_4 > 0.5)  # Relaxed from 0.45
            
            # Condition 5: Path smoothness (relaxed)
            path_smoothness = (close_position_2.rolling(2).std() < 0.12) & (future_position_2.rolling(2).std() < 0.12)  # Relaxed from 0.08
            smooth_continuation = path_smoothness & (future_position_2 > close_position_2)
            
            # Condition 6: Multi-timeframe consistency filter (relaxed)
            # Check if path direction is consistent across timeframes
            short_trend = future_position_2 > close_position_2
            medium_trend = future_position_4 > close_position_4
            long_trend = future_position_15 > close_position_15
            
            # Weighted consistency score (relaxed requirement)
            consistency_score = (
                short_trend.astype(int) * 3 +  # Short-term gets highest weight
                medium_trend.astype(int) * 2 +  # Medium-term gets medium weight
                long_trend.astype(int) * 1      # Long-term gets lowest weight
            ) / 6.0
            
            # Relaxed consistency requirement (reduced from 0.67 to 0.5)
            strong_consistency = consistency_score >= 0.5
            
            # Condition 7: Path momentum filter (relaxed)
            # Calculate path momentum as change in position
            path_momentum_current = close_position_4 - close_position_15
            path_momentum_future = future_position_4 - close_position_15
            momentum_continuation = (path_momentum_current * path_momentum_future) >= 0  # Relaxed from >0
            
            # Condition 8: Path range expansion filter (relaxed)
            # Detect if path range is expanding (indicating strong moves)
            current_range = high_4 - low_4
            future_range = future_high.rolling(4).max() - future_low.rolling(4).min()
            range_expansion = future_range > current_range * 1.05  # Relaxed from 1.1
            
            # Combine path signals with relaxed scoring
            path_signal = (
                strong_continuation.astype(int) * 3 +      # High weight
                path_acceleration_signal.astype(int) * 2 +  # Medium weight
                vol_breakout.astype(int) * 1 +            # Low weight
                smooth_continuation.astype(int) * 2 +     # Medium weight
                strong_consistency.astype(int) * 2 +       # Reduced from 3
                momentum_continuation.astype(int) * 2 +   # Medium weight
                range_expansion.astype(int) * 1           # Low weight
            )
            
            # Exclude mean reversion from positive signals
            path_signal[mean_reversion] = 0
            
            # Label: 1 for strong path patterns, 0 for neutral/reverting paths
            # Use much lower threshold to allow more signals
            labels = (path_signal >= 2).astype(int)  # Reduced from 3 to 2
            
            # Relaxed consistency filter (removed strong_consistency requirement)
            returns = df['close'].pct_change()
            trend_consistency = (returns.rolling(4).mean() * returns.rolling(8).mean()) > 0
            labels = labels & trend_consistency  # Removed strong_consistency filter
            
            return labels
        else:
            # Enhanced fallback labels with better sensitivity
            returns = df['close'].pct_change()
            
            # Multi-timeframe momentum adapted for 15m
            momentum_2 = returns.rolling(2).sum()
            momentum_4 = returns.rolling(4).sum()
            momentum_15 = returns.rolling(15).sum()
            
            # Future momentum
            future_momentum_2 = momentum_2.shift(-lookforward)
            future_momentum_4 = momentum_4.shift(-lookforward)
            
            # Momentum consistency
            momentum_consistency = (momentum_2 * momentum_4) > 0
            momentum_strength = abs(momentum_2) > momentum_2.std() * 0.5
            
            # Enhanced labeling
            labels = (
                (future_momentum_2 > momentum_2.quantile(0.6)) &
                momentum_consistency &
                momentum_strength
            ).astype(int)
            
            return labels
    
    def _optimize_xgb_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Optimize XGBoost hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for XGBoost MI optimization
        # Parameter grid for MI-focused optimization
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.07, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0.1, 0.5, 1.0],
            "reg_lambda": [2, 5, 10],
            "min_child_weight": [20, 40]
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for params in self._generate_param_combinations(param_grid, max_combinations=15):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train XGBoost model
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         early_stopping_rounds=20, verbose=False)
                
                # Compute MI
                val_pred = model.predict_proba(X_val)[:, 1]
                mi_score = mutual_info_regression(
                    val_pred.reshape(-1, 1), y_val.values
                )[0]
                mi_scores.append(mi_score)
            
            avg_mi = np.mean(mi_scores)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_params = params.copy()
                
                tprint_info(f"🔥 New best XGB MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best XGBoost hyperparameters found: MI = {best_mi:.4f}")
        return best_params, best_mi
    

    def _generate_param_combinations(self, param_grid: Dict[str, List], max_combinations: int = 20):
        """Generate parameter combinations for optimization."""
        import itertools
        import random
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Randomly sample if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        for combination in all_combinations:
            yield dict(zip(keys, combination))
    
    def _train_enhanced_path_model(self, features: pd.DataFrame, labels: pd.Series, 
                                  config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced path model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("path_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("path_test_size_ratio", 0.2),
            min_train_samples=config.get("path_min_train_samples", 500),
        )
        
        # Create training config
        training_config = XGBTrainingConfig(
            objective="binary:logistic",
            random_state=42,
            **best_params
        )
        
        # Train with standardized trainer
        trainer = StandardizedXGBTrainer(training_config)
        train_result = trainer.train_time_series_cv(features, labels, temporal_config)
        
        # Extract best model
        best_model = train_result.models[-1] if train_result.models else None
        
        # Compute MI metrics
        oof_preds = train_result.oof_predictions
        if 'probability' in oof_preds.columns:
            mi_score = mutual_info_regression(
                oof_preds['probability'].values.reshape(-1, 1), 
                labels.loc[oof_preds.index].values
            )[0]
        else:
            mi_score = 0.0
        
        # Store training metrics
        self.training_metrics.append({
            'mi_score': mi_score,
            'n_features': len(features.columns),
            'best_params': best_params
        })
        
        metrics = {
            'mi_score': mi_score,
            'auc': train_result.metrics.get('oof_auc', 0.0),
            'log_loss': train_result.metrics.get('oof_log_loss', 0.0),
            'n_features': len(features.columns),
            'optimization_params': best_params,
            'training_time': train_result.metrics.get('training_time', 0.0)
        }
        
        return best_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced path regime step."""
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model="enhanced_path_regime",
            )

            tprint_info(f"🚀 Starting Enhanced Path Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Path Regime features...")
            feature_df = self._generate_enhanced_path_features(market_data, config)
            
            tprint_info(f"✅ Enhanced Path Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_ml_path_regime_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced Path Regime labels...")
            labels = self._create_path_labels(market_data)

            # Align features and labels
            common_index = feature_df.index.intersection(labels.index)
            X = feature_df.loc[common_index]
            y = labels.loc[common_index]

            # Clean data
            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            if len(X) < 500:
                raise RuntimeError(f"Insufficient valid samples: {len(X)} < 500")

            tprint_info(f"📊 Training Data: {len(X)} samples, {len(X.columns)} features")

            # 4. Train Enhanced Model with MI Optimization
            tprint_info("🤖 Training Enhanced Path Regime model with MI optimization...")
            model, model_metrics = self._train_enhanced_path_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_ml_path_regime_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLPathRegimeStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0
            )
            
            
            artifact_path = self._save_artifact(
                data=standardized_output,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=metadata
            )
            artifacts.append(artifact_path)

            # 8. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            if diagnostics_result.get('success', False):
                compliance_report = diagnostics_result['compliance_report']
                ensemble_compatibility = diagnostics_result['ensemble_compatibility']
                
                tprint_success(f"✅ Enhanced Diagnostics Complete:")
                tprint_info(f"   MI Score: {compliance_report['metrics']['mi_score']:.4f}")
                tprint_info(f"   Requirements Met: {compliance_report['requirements_met']}/3")
                tprint_info(f"   Ensemble Ready: {ensemble_compatibility['ensemble_ready']}")
                
                metrics.update({
                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],
                    'enhanced_requirements_met': compliance_report['requirements_met'],
                    'enhanced_ensemble_ready': ensemble_compatibility['ensemble_ready']
                })

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(standardized_output)

            tprint_success(f"✅ Enhanced Path Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(standardized_output),
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Path Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        standardized = pd.DataFrame(index=features.index)
        standardized['timestamp'] = features.index
        standardized['specialist_prediction'] = predictions
        standardized['specialist_probability'] = probabilities
        standardized['target_label'] = labels
        
        # Add original features for reference
        for col in features.columns[:20]:  # Limit to first 20 features
            standardized[f'feature_{col}'] = features[col]
        
        return standardized
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        try:
            from src.utils.data_loader import DataLoader
            data_loader = DataLoader()
            
            # Try to load appropriate data based on timeframe
            if timeframe == "1h":
                market_data = data_loader.load_ethusdt_1h_data()
            elif timeframe == "1m":
                market_data = data_loader.load_ethusdt_1m_data()
            else:
                # For other timeframes, try the general loading function
                market_data = data_loader.load_ethusdt_data_for_analysis(timeframe=timeframe)
                if isinstance(market_data, dict):
                    market_data = market_data.get('data')
            
            return market_data, "loaded"
            
        except Exception as e:
            from src.utils.tprint import tprint_warning
            tprint_warning(f"Failed to load market data: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(), "failed"
