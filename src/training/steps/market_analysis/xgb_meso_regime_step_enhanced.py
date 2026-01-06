"""
Enhanced XGB Meso Regime Step with MI Improvements

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
from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights
)
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)


class EnhancedXGBMesoRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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

    def __init__(self, step_name: str = "enhanced_xgb_meso_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedXGBMesoRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _get_meso_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine meso features, enhanced features, and specific meso enhancements."""
        # Import original meso features
        try:
            from src.feature_generation.categories.meso_regime_features import generate_meso_regime_features
            # Need config for generate_meso_regime_features.
            config = {
                'symbol': self._current_context.get('symbol'),
                'exchange': self._current_context.get('exchange'),
                'timeframe': self._current_context.get('timeframe'),
                'direction': self._current_context.get('direction')
            }
            meso_features = generate_meso_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            meso_features = pd.DataFrame(index=df.index)
        
        # Meso-specific enhancements
        meso_enhanced = self._add_meso_specific_features(df, meso_features)
        
        # Combine all features
        all_features = pd.concat([meso_features, meso_enhanced], axis=1)
        
        return all_features
    
    def _add_meso_specific_features(self, df: pd.DataFrame, meso_features: pd.DataFrame) -> pd.DataFrame:
        """Add meso-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced meso analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe meso analysis
            for window in [40,60,80]:
                # Meso trend
                meso_trend = returns.rolling(window).mean()
                features[f'meso_trend_{window}'] = meso_trend
                
                # Meso momentum
                meso_momentum = returns.rolling(window).sum()
                features[f'meso_momentum_{window}'] = meso_momentum
                
                # Meso acceleration
                meso_acceleration = meso_momentum.diff()
                features[f'meso_acceleration_{window}'] = meso_acceleration
                
                # Meso volatility
                meso_volatility = returns.rolling(window).std()
                features[f'meso_volatility_{window}'] = meso_volatility
                
                # Meso risk-adjusted returns
                risk_adjusted = meso_trend / meso_volatility
                features[f'meso_risk_adjusted_{window}'] = risk_adjusted
                
                # Meso regime strength
                regime_strength = abs(meso_trend) / meso_volatility
                features[f'meso_regime_strength_{window}'] = regime_strength
                
                # Meso persistence
                meso_persistence = (meso_trend > 0).rolling(window).mean()
                features[f'meso_persistence_{window}'] = meso_persistence
                
                # Meso regime transitions
                regime_transition = meso_persistence.diff()
                features[f'meso_regime_transition_{window}'] = regime_transition
            
            # Cross-timeframe meso analysis
            for short_window in [5, 10]:
                for long_window in [20, 50]:
                    short_trend = returns.rolling(short_window).mean()
                    long_trend = returns.rolling(long_window).mean()
                    
                    # Trend alignment
                    trend_alignment = (short_trend * long_trend)
                    features[f'meso_trend_alignment_{short_window}_{long_window}'] = trend_alignment
                    
                    # Trend divergence
                    trend_divergence = abs(short_trend - long_trend)
                    features[f'meso_trend_divergence_{short_window}_{long_window}'] = trend_divergence
                    
                    # Momentum convergence
                    momentum_convergence = (short_trend > 0) == (long_trend > 0)
                    features[f'meso_momentum_convergence_{short_window}_{long_window}'] = momentum_convergence.astype(int)
            
            # Meso cycle analysis
            for window in [10, 20, 50]:
                # Cycle detection using autocorrelation
                cycle_strength = returns.rolling(window).apply(lambda x: x.autocorr())
                features[f'meso_cycle_strength_{window}'] = cycle_strength
                
                # Cycle phase
                cycle_phase = np.arctan2(returns.rolling(window).mean(), returns.rolling(window).std())
                features[f'meso_cycle_phase_{window}'] = cycle_phase
                
                # Cycle amplitude
                cycle_amplitude = returns.rolling(window).std()
                features[f'meso_cycle_amplitude_{window}'] = cycle_amplitude
            
            # Meso extreme analysis
            for window in [40,60,80]:
                # Extreme returns
                extreme_returns = returns.rolling(window).apply(lambda x: (x.abs() > x.std() * 1.5).sum())
                features[f'meso_extreme_returns_{window}'] = extreme_returns
                
                # Tail risk
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.1)).mean())
                features[f'meso_tail_risk_{window}'] = tail_risk
                
                # Volatility clustering
                volatility_clustering = returns.rolling(window).std().rolling(window).corr(returns.rolling(window).std())
                features[f'meso_volatility_clustering_{window}'] = volatility_clustering
            
            # Meso regime classification
            for window in [20, 50]:
                # Bull regime
                bull_regime = (returns.rolling(window).mean() > 0) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_bull_regime_{window}'] = bull_regime.astype(int)
                
                # Bear regime
                bear_regime = (returns.rolling(window).mean() < 0) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_bear_regime_{window}'] = bear_regime.astype(int)
                
                # Volatile regime
                volatile_regime = returns.rolling(window).std() > returns.rolling(window*2).std() * 1.2
                features[f'meso_volatile_regime_{window}'] = volatile_regime.astype(int)
                
                # Range-bound regime
                range_bound = (abs(returns.rolling(window).mean()) < returns.rolling(window).std() * 0.3) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_range_bound_{window}'] = range_bound.astype(int)
        
        # Volume-meso relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted meso analysis
            volume_ma = volume.rolling(25).mean()
            volume_anomaly = volume / volume_ma
            
            for window in [5, 10, 20]:
                # Volume-scaled meso trend
                vol_scaled_trend = (returns * volume).rolling(window).sum()
                features[f'meso_vol_scaled_trend_{window}'] = vol_scaled_trend
                
                # Volume-meso correlation
                volume_meso_corr = returns.rolling(window).corr(volume)
                features[f'meso_volume_meso_corr_{window}'] = volume_meso_corr
                
                # Volume confirmation of meso moves
                volume_confirmation = (volume_anomaly > 1.5) & (abs(returns.rolling(window).mean()) > returns.rolling(window*2).std() * 0.3)
                features[f'meso_volume_confirmation_{window}'] = volume_confirmation.astype(int)
                
                # Volume-meso divergence
                volume_divergence = abs(volume_meso_corr) < 0.3
                features[f'meso_volume_divergence_{window}'] = volume_divergence.astype(int)
                
                # Volume-meso efficiency
                volume_efficiency = returns.abs() / (volume + 1e-8)
                features[f'meso_volume_efficiency_{window}'] = volume_efficiency.rolling(window).mean()
        
        # Support/resistance meso analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            for window in [10, 20, 50]:
                # Meso support/resistance
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                
                # Distance to meso levels
                features[f'meso_distance_to_resistance_{window}'] = (rolling_max - close) / rolling_max
                features[f'meso_distance_to_support_{window}'] = (close - rolling_min) / rolling_max
                
                # Meso SR strength
                features[f'meso_sr_strength_{window}'] = (rolling_max - rolling_min) / close
                
                # Meso level breaches
                features[f'meso_resistance_breach_{window}'] = (close > rolling_max.shift(1)).astype(int)
                features[f'meso_support_breach_{window}'] = (close < rolling_min.shift(1)).astype(int)
                
                # Meso range expansion
                range_expansion = (rolling_max - rolling_min) / (rolling_max - rolling_min).rolling(window*2).mean()
                features[f'meso_range_expansion_{window}'] = range_expansion
                
                # Meso range contraction
                range_contraction = range_expansion < 0.8
                features[f'meso_range_contraction_{window}'] = range_contraction.astype(int)
                
                # Meso position
                meso_position = (close - rolling_min) / (rolling_max - rolling_min)
                features[f'meso_position_{window}'] = meso_position
                
                # Meso position momentum
                meso_position_momentum = meso_position.diff()
                features[f'meso_position_momentum_{window}'] = meso_position_momentum
        
        # Time-based meso patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on meso
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Time-based meso transitions
            features['is_end_of_day'] = (df.index.hour >= 20).astype(int)
            features['is_start_of_day'] = (df.index.hour <= 8).astype(int)
            
            # Weekly patterns
            features['is_monday'] = (df.index.dayofweek == 0).astype(int)
            features['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        return features

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced XGB meso regime step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.MESO_REGIME, # Assuming this exists
            manual_feature_func=self._get_meso_combined_manual_features,
            filter_type='price',
            pt_sl_config_key='meso_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_xgb_meso_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
