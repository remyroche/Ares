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
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, compute_efficiency_ratio, get_sample_weights
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLPathRegimeStep(AFMLSpecialistMixin, SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLPathRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _get_path_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine path features and manual enhancements."""
        # 1. Pipeline Features + Efficiency Ratio (Base Path Features)
        features = pd.DataFrame(index=df.index)
        
        # Geometry focus: Efficiency Ratio across multiple windows
        for window in [10, 20, 40, 60]:
            features[f'efficiency_ratio_{window}'] = compute_efficiency_ratio(df['close'], window=window)
            # Change in efficiency (acceleration/deceleration of 'travel')
            features[f'efficiency_delta_{window}'] = features[f'efficiency_ratio_{window}'].diff()
            
        # 2. Manual Features
        manual_features = self._create_manual_path_risk_enhanced_features(df, pipeline_features)
        
        # Combine
        return pd.concat([features, manual_features], axis=1)

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
            path_velocity_10 = path_10.diff()
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
            
            # 4. Volume-adjusted path features
            # Volume-weighted path momentum
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_weighted_path_momentum = path_momentum_5 * (1 + np.log(volume_ratio + 1))
            manual_features['enhanced_vol_scaled_path_momentum'] = volume_weighted_path_momentum
            
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
            # Path cyclical patterns (using rolling autocorrelation) - FIXED: removed global leak
            # Optimization: Only compute every 5th row to reduce CPU load
            try:
                # Rolling autocorrelation over 50 bars with lag 5
                step_autocorr = 5
                autocorr_vals = np.full(len(df), np.nan)
                rets_vals = returns.values
                
                for i in range(50, len(df), step_autocorr):
                    window_data = rets_vals[i-50:i]
                    # Manual autocorr calculation for lag 5
                    if len(window_data) > 5:
                        s1 = window_data[5:]
                        s2 = window_data[:-5]
                        if np.std(s1) > 0 and np.std(s2) > 0:
                            autocorr_vals[i] = np.corrcoef(s1, s2)[0, 1]
                
                manual_features['enhanced_path_cyclical_strength'] = pd.Series(autocorr_vals, index=df.index).ffill().fillna(0.0)
            except Exception:
                manual_features['enhanced_path_cyclical_strength'] = 0.0
            
            # Path seasonality (intraday patterns) - FIXED: use expanding mean or lagged mean to avoid leakage
            if hasattr(df.index, 'hour'):
                # Use a lagged hourly mean (e.g., from previous days) or expanding mean
                # For simplicity and robustness, we'll use a 20-day lagged hourly mean if we had enough data,
                # but here we'll just use the hour as a feature and let the model learn the pattern
                manual_features['enhanced_path_hour'] = df.index.hour
            
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
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced path regime step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.PATH_REGIME, # Assuming this exists
            manual_feature_func=self._get_path_combined_manual_features,
            filter_type='volatility',
            pt_sl_config_key='path_regime_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_ml_path_regime_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
