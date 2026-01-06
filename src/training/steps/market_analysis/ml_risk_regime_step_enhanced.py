"""
Enhanced ML Risk Regime Step with MI Improvements

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
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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
    frac_diff_fixed, get_sample_weights
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
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLRiskRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced Risk Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Risk-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_risk_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLRiskRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _get_risk_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine risk features, enhanced features, and specific risk enhancements."""
        # Import original risk features
        try:
            # Reconstruct basic config from context
            config = {
                'symbol': self._current_context.get('symbol'),
                'exchange': self._current_context.get('exchange'),
                'timeframe': self._current_context.get('timeframe'),
                'direction': self._current_context.get('direction')
            }
            from src.feature_generation.categories.risk_regime_features import generate_risk_regime_features
            base_risk_features = generate_risk_regime_features(df, config)
        except ImportError:
            base_risk_features = pd.DataFrame(index=df.index)
        
        # Manual feature engineering for risk regime
        manual_features = self._create_manual_risk_enhanced_features(df, pipeline_features)
        
        # Combine all features
        all_features = pd.concat([base_risk_features, manual_features], axis=1)
        
        return all_features
    
    def _create_manual_risk_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create advanced manual enhanced features for risk regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Tail Risk Measures: VaR and CVaR at multiple confidence intervals
            for window in [20, 50]:
                # VaR at 95% and 99% confidence intervals
                var_95 = returns.rolling(window).quantile(0.05)
                var_99 = returns.rolling(window).quantile(0.01)
                manual_features[f'var_95_{window}d'] = var_95
                manual_features[f'var_99_{window}d'] = var_99
                
                # CVaR (Expected Shortfall) at 95% and 99%
                cvar_95 = returns[returns <= var_95].rolling(window).mean()
                cvar_99 = returns[returns <= var_99].rolling(window).mean()
                manual_features[f'cvar_95_{window}d'] = cvar_95
                manual_features[f'cvar_99_{window}d'] = cvar_99
                
                # Tail risk ratios
                manual_features[f'tail_risk_ratio_95_{window}d'] = cvar_95 / (var_95 + 1e-8)
                manual_features[f'tail_risk_ratio_99_{window}d'] = cvar_99 / (var_99 + 1e-8)
            
            # 2. Enhanced Drawdown Risk Features
            cum_returns = (1 + returns).cumprod()
            for window in [20, 50, 100]:
                rolling_max = cum_returns.rolling(window).max()
                drawdown = (cum_returns - rolling_max) / rolling_max
                
                # Maximum drawdown
                max_drawdown = drawdown.rolling(window).min()
                manual_features[f'max_drawdown_{window}d'] = max_drawdown
                
                # Current drawdown percentage
                manual_features[f'current_drawdown_{window}d'] = drawdown
                
                # Drawdown velocity (speed of decline)
                drawdown_velocity = drawdown.diff().rolling(5).mean()
                manual_features[f'drawdown_velocity_{window}d'] = drawdown_velocity
                
                # Drawdown duration (consecutive periods in drawdown)
                drawdown_duration = (drawdown < 0).astype(int).rolling(window).sum()
                manual_features[f'drawdown_duration_{window}d'] = drawdown_duration
                
                # Drawdown recovery time
                drawdown_recovery = (drawdown >= 0).astype(int).rolling(window).sum()
                manual_features[f'drawdown_recovery_{window}d'] = drawdown_recovery
            
            # 3. Volatility-Adjusted Risk Features
            for window in [20, 50]:
                volatility = returns.rolling(window).std()
                
                # Risk-adjusted returns
                risk_adj_returns = returns.rolling(window).mean() / (volatility + 1e-8)
                manual_features[f'risk_adj_returns_{window}d'] = risk_adj_returns
                
                # Sharpe ratio volatility
                sharpe_vol = volatility * np.sqrt(252)
                manual_features[f'sharpe_volatility_{window}d'] = sharpe_vol
                
                # Sortino ratio (downside risk adjusted)
                downside_returns = returns.copy()
                downside_returns[downside_returns > 0] = 0
                downside_vol = downside_returns.rolling(window).std()
                sortino_ratio = returns.rolling(window).mean() / (downside_vol + 1e-8)
                manual_features[f'sortino_ratio_{window}d'] = sortino_ratio
                
                # Volatility-adjusted VaR
                var_95_adj = var_95 / (volatility + 1e-8)
                manual_features[f'var_95_adj_{window}d'] = var_95_adj
            
            # 4. Risk Regime Classification
            for window in [20, 50]:
                volatility = returns.rolling(window).std()
                vol_ma = volatility.rolling(window*2).mean()
                vol_std = volatility.rolling(window*2).std()
                
                # High/low volatility regimes
                high_vol_regime = (volatility > vol_ma * 1.5).astype(int)
                low_vol_regime = (volatility < vol_ma * 0.5).astype(int)
                normal_vol_regime = ((volatility >= vol_ma * 0.5) & (volatility <= vol_ma * 1.5)).astype(int)
                
                manual_features[f'high_vol_regime_{window}d'] = high_vol_regime
                manual_features[f'low_vol_regime_{window}d'] = low_vol_regime
                manual_features[f'normal_vol_regime_{window}d'] = normal_vol_regime
                
                # Risk regime transitions
                vol_regime = np.where(volatility > vol_ma * 1.5, 2, np.where(volatility < vol_ma * 0.5, 0, 1))
                regime_changes = np.diff(vol_regime, prepend=vol_regime[0])
                manual_features[f'risk_regime_changes_{window}d'] = np.abs(regime_changes)
                
                # Risk regime persistence
                vol_regime_series = pd.Series(vol_regime, index=df.index)
                regime_persistence = (vol_regime_series == 1).rolling(10).sum()
                manual_features[f'risk_regime_persistence_{window}d'] = regime_persistence
            
            # 5. Enhanced multi-dimensional volatility features
            # Realized volatility across multiple timeframes
            vol_5 = returns.rolling(5).std()
            vol_10 = returns.rolling(10).std()
            vol_20 = returns.rolling(20).std()
            vol_50 = returns.rolling(50).std()
            
            # Volatility term structure
            vol_term_structure = vol_20 / (vol_50 + 1e-8)
            manual_features['vol_term_structure'] = vol_term_structure
            
            # Volatility momentum
            vol_momentum = vol_20.diff().rolling(5).mean()
            manual_features['vol_momentum'] = vol_momentum
            
            # Volatility acceleration
            vol_acceleration = vol_20.diff().diff()
            manual_features['vol_acceleration'] = vol_acceleration
            
            # 6. Advanced tail risk features
            manual_features['skewness_20'] = returns.rolling(20).skew()
            manual_features['kurtosis_20'] = returns.rolling(20).kurt()
            
            # Downside risk features
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            manual_features['downside_deviation_20'] = downside_returns.rolling(20).std()
            
            # Semi-variance (downside risk focus)
            semi_variance = ((downside_returns) ** 2).rolling(20).mean()
            manual_features['semi_variance_20'] = semi_variance
            
            # 7. Volume-adjusted risk features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_weighted_vol = vol_20 * (1 + np.log(volume_ratio + 1))
            manual_features['volume_weighted_volatility'] = volume_weighted_vol
            
            # Volume-volatility divergence
            vol_regime = (vol_20 > vol_20.rolling(100).mean()).astype(int)
            volume_regime = (volume_ratio > 1).astype(int)
            volume_vol_divergence = np.abs(vol_regime - volume_regime)
            manual_features['volume_vol_divergence'] = volume_vol_divergence
            
            # 8. Price-based risk features
            range_ratio = (high - low) / close
            range_vol = range_ratio.rolling(20).std()
            manual_features['range_volatility'] = range_vol
            
            # Price efficiency
            price_efficiency = abs(returns.rolling(10).mean()) / (vol_10 + 1e-8)
            manual_features['price_efficiency'] = price_efficiency
            
            # Trend strength
            trend_strength = abs(returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)
            manual_features['trend_strength'] = trend_strength
            
            # 9. Market microstructure risk features
            spread_proxy = range_ratio
            spread_volatility = spread_proxy.rolling(20).std()
            manual_features['spread_volatility'] = spread_volatility
            
            # Order flow imbalance proxy
            price_volume_imbalance = (returns * volume).rolling(10).sum()
            manual_features['order_flow_imbalance'] = price_volume_imbalance
            
            # Market depth proxy
            market_depth = volume / (range_ratio + 1e-8)
            manual_features['market_depth'] = market_depth
            
            # 10. Composite risk indicators
            # Risk stress index
            vol_zscore_20 = (vol_20 - vol_20.rolling(100).mean()) / (vol_20.rolling(100).std() + 1e-8)
            risk_stress_index = (
                0.3 * (vol_zscore_20 > 1).astype(int) +
                0.3 * (max_drawdown < -0.05).astype(int) +
                0.2 * (volume_vol_divergence > 0).astype(int) +
                0.2 * (drawdown_velocity < -0.01).astype(int)
            )
            manual_features['risk_stress_index'] = risk_stress_index
            
            # Risk appetite indicator
            risk_appetite = 1 - risk_stress_index
            manual_features['risk_appetite'] = risk_appetite
            
        return manual_features

    def _add_risk_specific_features(self, df: pd.DataFrame, risk_features: pd.DataFrame) -> pd.DataFrame:
        """Add risk-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced risk analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe risk analysis
            for window in [60,80,100]:
                # Volatility risk
                volatility = returns.rolling(window).std()
                features[f'volatility_risk_{window}'] = volatility
                
                # Downside risk
                downside_returns = returns[returns < 0]
                downside_volatility = downside_returns.rolling(window).std()
                features[f'downside_volatility_{window}'] = downside_volatility
                
                # Upside volatility
                upside_returns = returns[returns > 0]
                upside_volatility = upside_returns.rolling(window).std()
                features[f'upside_volatility_{window}'] = upside_volatility
                
                # Volatility skewness
                volatility_skew = upside_volatility / (downside_volatility + 1e-8)
                features[f'volatility_skew_{window}'] = volatility_skew
                
                # Maximum drawdown risk
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.rolling(window).max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdown.rolling(window).min()
                features[f'max_drawdown_{window}'] = max_drawdown
                
                # Drawdown duration
                drawdown_duration = (drawdown < 0).rolling(window).sum()
                features[f'drawdown_duration_{window}'] = drawdown_duration
                
                # Risk-adjusted returns
                risk_adjusted_returns = returns.rolling(window).mean() / volatility
                features[f'risk_adjusted_returns_{window}'] = risk_adjusted_returns
                
                # Value at Risk (VaR)
                var_95 = returns.rolling(window).quantile(0.05)
                var_99 = returns.rolling(window).quantile(0.01)
                features[f'var_95_{window}'] = var_95
                features[f'var_99_{window}'] = var_99
                
                # Conditional Value at Risk (CVaR)
                cvar_95 = returns[returns <= var_95].rolling(window).mean()
                features[f'cvar_95_{window}'] = cvar_95
            
            # Risk regime indicators
            for window in [20, 50]:
                # High volatility regime
                volatility_ma = returns.rolling(window).std().rolling(window*2).mean()
                high_volatility = (returns.rolling(window).std() > volatility_ma * 1.5)
                features[f'high_volatility_regime_{window}'] = high_volatility.astype(int)
                
                # Low volatility regime
                low_volatility = (returns.rolling(window).std() < volatility_ma * 0.5)
                features[f'low_volatility_regime_{window}'] = low_volatility.astype(int)
                
                # Risk escalation
                risk_escalation = volatility.rolling(window).diff()
                features[f'risk_escalation_{window}'] = risk_escalation
                
                # Risk persistence
                risk_persistence = (high_volatility.rolling(window).mean())
                features[f'risk_persistence_{window}'] = risk_persistence
            
            # Tail risk analysis
            for window in [20, 50]:
                # Tail risk indicator
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.05)).mean())
                features[f'tail_risk_{window}'] = tail_risk
                
                # Extreme events
                extreme_events = (returns.abs() > returns.rolling(window).std() * 2)
                features[f'extreme_events_{window}'] = extreme_events.rolling(window).sum()
                
                # Tail risk persistence
                tail_risk_persistence = tail_risk.rolling(window).mean()
                features[f'tail_risk_persistence_{window}'] = tail_risk_persistence
        
        # Volume-risk relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted risk
            volume_ma = volume.rolling(25).mean()
            volume_anomaly = volume / volume_ma
            volume_adjusted_volatility = returns.rolling(25).std() * volume_anomaly
            features['volume_adjusted_volatility'] = volume_adjusted_volatility
            
            # Volume-risk correlation
            for window in [10, 20, 50]:
                volume_risk_corr = returns.rolling(window).corr(volume)
                features[f'volume_risk_corr_{window}'] = volume_risk_corr
                
                # Volume confirmation of risk
                features[f'volume_risk_confirmation_{window}'] = (
                    (volume_anomaly > 1.5) & (returns.rolling(window).std() > returns.rolling(window*2).std() * 1.2)
                ).astype(int)
                
                # Volume-weighted risk
                volume_weighted_risk = (returns.abs() * volume).rolling(window).sum()
                features[f'volume_weighted_risk_{window}'] = volume_weighted_risk
        
        # Support/resistance risk analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            # Range-based risk
            for window in [20, 50]:
                range_volatility = (high - low).rolling(window).std()
                features[f'range_volatility_{window}'] = range_volatility
                
                # Range expansion risk
                range_ma = (high - low).rolling(window).mean()
                range_expansion = (high - low) / range_ma
                features[f'range_expansion_{window}'] = range_expansion
                
                # Range contraction risk
                range_contraction = (range_expansion < 0.7).astype(int)
                features[f'range_contraction_{window}'] = range_contraction
                
                # Breakout risk
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                breakout_risk = (close > rolling_max.shift(1)) | (close < rolling_min.shift(1))
                features[f'breakout_risk_{window}'] = breakout_risk.astype(int)
        
        # Time-based risk patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on risk
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Time-based risk escalation
            features['is_end_of_day'] = (df.index.hour >= 20).astype(int)
            features['is_start_of_day'] = (df.index.hour <= 8).astype(int)
            
            # Month-end risk
            features['is_month_end'] = (df.index.day >= 28).astype(int)
        
        return features
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced risk regime step."""
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.RISK_REGIME, # Assuming this exists or falls back
            manual_feature_func=self._get_risk_combined_manual_features,
            filter_type='volatility',
            pt_sl_config_key='risk_regime_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_risk_regime_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching using BaseStep method."""
        symbol = config.get("symbol", "ETHUSDT")
        exchange = config.get("exchange", "binance")
        
        # Generate cache key
        cache_key = (symbol, exchange, timeframe)
        
        # Check cache first
        if cache_key in self._market_data_cache:
            self.logger.info(f"📦 Using cached market data for {symbol}")
            return self._market_data_cache[cache_key], "cache"
        
        # Use standard BaseStep method
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        
        # Cache the data
        self._market_data_cache[cache_key] = market_data
        
        self.logger.info(f"✅ Loaded {len(market_data)} rows of market data for {symbol} from {market_source}")
        return market_data, market_source
