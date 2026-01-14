"""
Enhanced ML SMC Regime Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
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
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLSMCRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced SMC Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_smc_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLSMCRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _get_smc_combined_manual_features(self, df: pd.DataFrame, pipeline_features: pd.DataFrame) -> pd.DataFrame:
        """Combine SMC features, enhanced features, and specific SMC enhancements."""
        # Import original SMC features
        try:
            from src.feature_generation.categories.smc_regime_features import generate_smc_regime_features
            
            # Safely extract and convert config values
            symbol = self._current_context.get('symbol', 'ETHUSDT')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            
            # Ensure all values are strings, not floats
            config = {
                'symbol': str(symbol) if not isinstance(symbol, str) else symbol,
                'exchange': str(exchange) if not isinstance(exchange, str) else exchange,
                'timeframe': str(timeframe) if not isinstance(timeframe, str) else timeframe,
                'direction': str(direction) if not isinstance(direction, str) else direction
            }
            
            # Additional safety check for symbol
            if config['symbol'].replace('.', '').replace('-', '').isdigit():
                # If it looks like a number, convert to proper symbol
                config['symbol'] = 'ETHUSDT'
                
            tprint_info(f"🔧 SMC Config: {config}")
            base_smc_features = generate_smc_regime_features(df, config)
        except ImportError:
            base_smc_features = pd.DataFrame(index=df.index)
        except Exception as e:
            tprint_error(f"❌ SMC feature generation failed: {e}")
            base_smc_features = pd.DataFrame(index=df.index)
        
        # Manual feature engineering for SMC regime
        manual_features = self._create_manual_smc_enhanced_features(df, pipeline_features)
        
        # Combine all features
        all_features = pd.concat([base_smc_features, manual_features], axis=1)
        
        return all_features
    
    def _create_manual_smc_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create advanced manual enhanced features for SMC regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Advanced Market Structure Detection
            for window in [20, 50]:
                # Higher highs identification
                rolling_highs = high.rolling(window).max()
                higher_highs = (high > rolling_highs.shift(1)).astype(int)
                manual_features[f'higher_highs_{window}d'] = higher_highs
                
                # Lower lows identification
                rolling_lows = low.rolling(window).min()
                lower_lows = (low < rolling_lows.shift(1)).astype(int)
                manual_features[f'lower_lows_{window}d'] = lower_lows
                
                # Market structure validation
                # Bullish structure: higher highs and higher lows
                bullish_structure = higher_highs & (low > rolling_lows.shift(1)).astype(int)
                manual_features[f'bullish_structure_{window}d'] = bullish_structure
                
                # Bearish structure: lower highs and lower lows
                rolling_highs_prev = high.rolling(window).max().shift(1)
                lower_highs = (high < rolling_highs_prev).astype(int)
                bearish_structure = lower_highs & lower_lows
                manual_features[f'bearish_structure_{window}d'] = bearish_structure
                
                # Structure strength
                structure_strength = (higher_highs.astype(int) - lower_lows.astype(int)).abs()
                manual_features[f'structure_strength_{window}d'] = structure_strength
                
                # Structure breaks
                structure_break_up = (close > rolling_highs).astype(int)
                structure_break_down = (close < rolling_lows).astype(int)
                manual_features[f'structure_break_up_{window}d'] = structure_break_up
                manual_features[f'structure_break_down_{window}d'] = structure_break_down
            
            # 2. Support and Resistance Strength Analysis
            for window in [20, 50]:
                # Support levels
                support_levels = low.rolling(window).min()
                resistance_levels = high.rolling(window).max()
                
                # Support/Resistance touch counts
                support_touches = (low <= support_levels + 0.001 * close).astype(int).rolling(window).sum()
                resistance_touches = (high >= resistance_levels - 0.001 * close).astype(int).rolling(window).sum()
                
                manual_features[f'support_touches_{window}d'] = support_touches
                manual_features[f'resistance_touches_{window}d'] = resistance_touches
                
                # Support/Resistance holding periods
                support_holding = (low > support_levels * 0.99).astype(int).rolling(window).sum()
                resistance_holding = (high < resistance_levels * 1.01).astype(int).rolling(window).sum()
                
                manual_features[f'support_holding_{window}d'] = support_holding
                manual_features[f'resistance_holding_{window}d'] = resistance_holding
                
                # Level strength (touches * holding)
                support_strength = support_touches * support_holding
                resistance_strength = resistance_touches * resistance_holding
                
                manual_features[f'support_strength_{window}d'] = support_strength
                manual_features[f'resistance_strength_{window}d'] = resistance_strength
                
                # Distance from levels
                support_distance = (close - support_levels) / (support_levels + 1e-8)
                resistance_distance = (resistance_levels - close) / (resistance_levels + 1e-8)
                
                manual_features[f'support_distance_{window}d'] = support_distance
                manual_features[f'resistance_distance_{window}d'] = resistance_distance
            
            # 3. Breakout Confirmation with Volume and Volatility
            for window in [20, 50]:
                # Breakout detection
                resistance_level = high.rolling(window).max()
                support_level = low.rolling(window).min()
                
                breakout_up = (close > resistance_level).astype(int)
                breakout_down = (close < support_level).astype(int)
                
                manual_features[f'breakout_up_{window}d'] = breakout_up
                manual_features[f'breakout_down_{window}d'] = breakout_down
                
                # Volume confirmation
                volume_ma = volume.rolling(window).mean()
                volume_spike = (volume > volume_ma * 1.5).astype(int)
                
                volume_confirmed_up = breakout_up & volume_spike
                volume_confirmed_down = breakout_down & volume_spike
                
                manual_features[f'volume_confirmed_up_{window}d'] = volume_confirmed_up
                manual_features[f'volume_confirmed_down_{window}d'] = volume_confirmed_down
                
                # Volatility confirmation
                volatility = returns.rolling(window).std()
                vol_ma = volatility.rolling(window*2).mean()
                vol_spike = (volatility > vol_ma * 1.5).astype(int)
                
                vol_confirmed_up = breakout_up & vol_spike
                vol_confirmed_down = breakout_down & vol_spike
                
                manual_features[f'vol_confirmed_up_{window}d'] = vol_confirmed_up
                manual_features[f'vol_confirmed_down_{window}d'] = vol_confirmed_down
                
                # Full confirmation (volume + volatility)
                full_confirmed_up = breakout_up & volume_spike & vol_spike
                full_confirmed_down = breakout_down & volume_spike & vol_spike
                
                manual_features[f'full_confirmed_up_{window}d'] = full_confirmed_up
                manual_features[f'full_confirmed_down_{window}d'] = full_confirmed_down
            
            # 4. Multi-Timeframe SMC Alignment (15m, 1h, 4h simulation)
            # Optimization: Only resample if enough data and use faster resampling
            def resample_to_timeframe(df, timeframe_minutes):
                """Resample OHLCV data to higher timeframe efficiently."""
                if len(df) < timeframe_minutes:
                    return pd.DataFrame()
                
                resampled = df.resample(f'{timeframe_minutes}T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                return resampled
            
            # Create 1h (60min) and 4h (240min) data
            df_1h = resample_to_timeframe(df, 60)
            df_4h = resample_to_timeframe(df, 240)
            
            # Reindex to match original timeframe
            df_1h = df_1h.reindex(df.index, method='ffill')
            df_4h = df_4h.reindex(df.index, method='ffill')
            
            # 1h SMC signals
            for window in [20, 50]:
                if not df_1h.empty:
                    h1_support = df_1h['low'].rolling(window).min()
                    h1_resistance = df_1h['high'].rolling(window).max()
                    h1_close = df_1h['close']
                    
                    # 1h position relative to levels
                    h1_support_distance = (h1_close - h1_support) / (h1_support + 1e-8)
                    h1_resistance_distance = (h1_resistance - h1_close) / (h1_resistance + 1e-8)
                    
                    manual_features[f'h1_support_distance_{window}d'] = h1_support_distance
                    manual_features[f'h1_resistance_distance_{window}d'] = h1_resistance_distance
                    
                    # 1h trend direction
                    h1_ma_short = h1_close.rolling(window//2).mean()
                    h1_ma_long = h1_close.rolling(window).mean()
                    h1_trend = (h1_ma_short > h1_ma_long).astype(int)
                    
                    manual_features[f'h1_trend_{window}d'] = h1_trend
            
            # 4h SMC signals
            for window in [20, 50]:
                if not df_4h.empty:
                    h4_support = df_4h['low'].rolling(window).min()
                    h4_resistance = df_4h['high'].rolling(window).max()
                    h4_close = df_4h['close']
                    
                    # 4h position relative to levels
                    h4_support_distance = (h4_close - h4_support) / (h4_support + 1e-8)
                    h4_resistance_distance = (h4_resistance - h4_close) / (h4_resistance + 1e-8)
                    
                    manual_features[f'h4_support_distance_{window}d'] = h4_support_distance
                    manual_features[f'h4_resistance_distance_{window}d'] = h4_resistance_distance
                    
                    # 4h trend direction
                    h4_ma_short = h4_close.rolling(window//2).mean()
                    h4_ma_long = h4_close.rolling(window).mean()
                    h4_trend = (h4_ma_short > h4_ma_long).astype(int)
                    
                    manual_features[f'h4_trend_{window}d'] = h4_trend
            
            # 5. Timeframe Confluence Analysis
            # Check alignment between 15m, 1h, and 4h signals
            for window in [20]:
                # Trend confluence
                trend_15m = (close.rolling(window//2).mean() > close.rolling(window).mean()).astype(int)
                trend_1h = manual_features.get(f'h1_trend_{window}d', pd.Series(0, index=df.index))
                trend_4h = manual_features.get(f'h4_trend_{window}d', pd.Series(0, index=df.index))
                
                # Confluence score (0-3)
                trend_confluence = trend_15m + trend_1h + trend_4h
                manual_features[f'trend_confluence_{window}d'] = trend_confluence
                
                # Strong confluence (all timeframes aligned)
                strong_bullish_confluence = (trend_confluence == 3).astype(int)
                strong_bearish_confluence = (trend_confluence == 0).astype(int)
                
                manual_features[f'strong_bullish_confluence_{window}d'] = strong_bullish_confluence
                manual_features[f'strong_bearish_confluence_{window}d'] = strong_bearish_confluence
                
                # Support/Resistance confluence
                support_15m = (close > low.rolling(window).min()).astype(int)
                support_1h = (manual_features.get(f'h1_support_distance_{window}d', pd.Series(0, index=df.index)) > 0).astype(int)
                support_4h = (manual_features.get(f'h4_support_distance_{window}d', pd.Series(0, index=df.index)) > 0).astype(int)
                
                support_confluence = support_15m + support_1h + support_4h
                manual_features[f'support_confluence_{window}d'] = support_confluence
            
            # 6. Liquidity Zones Detection (OHLCV only)
            for window in [20, 50]:
                # Order block detection (strong momentum candles)
                momentum_candles = abs(returns) > returns.rolling(window).std() * 2
                order_blocks = momentum_candles.astype(int)
                
                manual_features[f'order_blocks_{window}d'] = order_blocks
                
                # Liquidity pool zones (price levels with high volume)
                volume_profile = volume.rolling(window).sum()
                high_volume_zones = (volume_profile > volume_profile.rolling(window*2).quantile(0.8)).astype(int)
                
                manual_features[f'high_volume_zones_{window}d'] = high_volume_zones
                
                # Price rejection zones (failed breakouts)
                failed_breakout_up = (high > high.rolling(window).max().shift(1)) & (close < high.rolling(window).max().shift(1))
                failed_breakout_down = (low < low.rolling(window).min().shift(1)) & (close > low.rolling(window).min().shift(1))
                
                rejection_zones = (failed_breakout_up | failed_breakout_down).astype(int)
                manual_features[f'rejection_zones_{window}d'] = rejection_zones
                
                # Liquidity sweep detection
                liquidity_sweep_up = (low < low.rolling(window).min().shift(1)) & (close > low.rolling(window).min().shift(1))
                liquidity_sweep_down = (high > high.rolling(window).max().shift(1)) & (close < high.rolling(window).max().shift(1))
                
                liquidity_sweeps = (liquidity_sweep_up | liquidity_sweep_down).astype(int)
                manual_features[f'liquidity_sweeps_{window}d'] = liquidity_sweeps
            
            # 7. Advanced SMC Pattern Recognition
            for window in [20, 50]:
                # Wyckoff accumulation/distribution patterns
                price_range = high - low
                range_ma = price_range.rolling(window).mean()
                vol_ma = volume.rolling(window).mean()
                
                # Accumulation (narrowing range, volume accumulation)
                accumulation_signal = (price_range < range_ma * 0.7) & (volume > vol_ma * 1.2)
                manual_features[f'accumulation_signal_{window}d'] = accumulation_signal.astype(int)
                
                # Distribution (widening range, volume distribution)
                distribution_signal = (price_range > range_ma * 1.3) & (volume < vol_ma * 0.8)
                manual_features[f'distribution_signal_{window}d'] = distribution_signal.astype(int)
                
                # Smart money entry/exit patterns
                close_ma = close.rolling(window).mean()
                smart_entry = (close > close_ma) & (volume > vol_ma * 1.5)
                smart_exit = (close < close_ma) & (volume > vol_ma * 1.5)
                
                manual_features[f'smart_entry_{window}d'] = smart_entry.astype(int)
                manual_features[f'smart_exit_{window}d'] = smart_exit.astype(int)
            
            # 8. Composite SMC Signals
            # Multi-indicator SMC score
            bullish_structure_20 = manual_features.get('bullish_structure_20d', pd.Series(0, index=df.index))
            trend_confluence_20 = manual_features.get('trend_confluence_20d', pd.Series(0, index=df.index))
            support_confluence_20 = manual_features.get('support_confluence_20d', pd.Series(0, index=df.index))
            smart_entry_20 = manual_features.get('smart_entry_20d', pd.Series(0, index=df.index))
            
            # Composite SMC strength
            composite_smc = (
                0.3 * bullish_structure_20 +
                0.25 * (trend_confluence_20 / 3) +
                0.25 * (support_confluence_20 / 3) +
                0.2 * smart_entry_20
            )
            manual_features['composite_smc_strength'] = composite_smc
            
            # SMC regime classification
            smc_regime = np.where(
                composite_smc > 0.6, 2,  # Strong SMC bullish
                np.where(composite_smc < 0.2, 0, 1)  # Weak/neutral SMC
            )
            manual_features['smc_regime'] = smc_regime
            
            # SMC persistence
            smc_persistence = (composite_smc > 0.4).rolling(10).sum()
            manual_features['smc_persistence'] = smc_persistence
            
        return manual_features
    
    def generate_pipeline_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate base pipeline features."""
        # For SMC regime, we use the combined manual features as pipeline features
        # when called from external consumers (like GMM pipeline)
        return self._get_smc_combined_manual_features(market_data, pd.DataFrame(index=market_data.index))

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced momentum persistence step."""
        # Ensure full data coverage by using 'price' filter instead of 'volatility'
        # This prevents data range restrictions that cause limited coverage
        return await self.execute_standard_specialist_logic(
            config=config,
            specialist_type=SpecialistType.SMC_REGIME, # Assuming exist or fallback
            manual_feature_func=self._get_smc_combined_manual_features,
            filter_type='price',  # Changed from 'volatility' to 'price' for full coverage
            pt_sl_config_key='smc_pt_sl',
            default_pt_sl=[2.0, 1.0],
            suffix="enhanced_smc_features"
        )
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
