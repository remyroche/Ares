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
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

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
    
    def _generate_enhanced_smc_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced SMC features with manual feature engineering."""
        # Import original SMC features
        from src.feature_generation.categories.smc_regime_features import generate_smc_regime_features
        base_smc_features = generate_smc_regime_features(df, config)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'smc_regime', {'enhanced_features': True}
        )
        
        # Manual feature engineering for SMC regime
        manual_features = self._create_manual_smc_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [base_smc_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_smc_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
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
            # Simulate higher timeframes by resampling
            def resample_to_timeframe(df, timeframe_minutes):
                """Resample OHLCV data to higher timeframe."""
                resampled = pd.DataFrame()
                resampled['open'] = df['close'].resample(f'{timeframe_minutes}T').first()
                resampled['high'] = df['high'].resample(f'{timeframe_minutes}T').max()
                resampled['low'] = df['low'].resample(f'{timeframe_minutes}T').min()
                resampled['close'] = df['close'].resample(f'{timeframe_minutes}T').last()
                resampled['volume'] = df['volume'].resample(f'{timeframe_minutes}T').sum()
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
                
                # Accumulation (narrowing range, volume accumulation)
                accumulation_signal = (price_range < range_ma * 0.7) & (volume > volume.rolling(window).mean() * 1.2)
                manual_features[f'accumulation_signal_{window}d'] = accumulation_signal.astype(int)
                
                # Distribution (widening range, volume distribution)
                distribution_signal = (price_range > range_ma * 1.3) & (volume < volume.rolling(window).mean() * 0.8)
                manual_features[f'distribution_signal_{window}d'] = distribution_signal.astype(int)
                
                # Smart money entry/exit patterns
                smart_entry = (close > close.rolling(window).mean()) & (volume > volume.rolling(window).mean() * 1.5)
                smart_exit = (close < close.rolling(window).mean()) & (volume > volume.rolling(window).mean() * 1.5)
                
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
    def _apply_manual_smc_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for SMC regime features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant SMC features")
        
        # Manual redundancy reduction - remove highly correlated features
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find highly correlated pairs (>0.9)
        to_drop = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > 0.9]
            if not correlated_features.empty:
                # Keep the feature that comes first alphabetically (deterministic)
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:  # Drop the later feature alphabetically
                        to_drop.append(correlated_feature)
        
        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
            self.logger.info(f"Removed {len(set(to_drop))} redundant SMC features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited SMC features to top 30 by variance")
        
        return features
    
    def _add_smc_specific_features(self, df: pd.DataFrame, smc_features: pd.DataFrame) -> pd.DataFrame:
        """Add SMC-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced SMC analysis
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # SMC-specific enhancements
            features['smc_volatility_ratio'] = returns.rolling(20).std() / returns.rolling(50).std()
            features['smc_trend_consistency'] = (returns.rolling(10).mean() > 0).rolling(30).mean()
            
        return features
    
    def _train_enhanced_smc_model(self, features: pd.DataFrame, labels: pd.Series, 
                                 config: Dict[str, Any], sample_weight: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced SMC model using centralized specialist trainer."""
        training_result = train_specialist_xgb_with_oof(
            features.fillna(0.0),
            labels.fillna(0.0),
            sample_weight=sample_weight,
            n_splits=5,
        )
        return training_result.model, training_result.metrics
    
    def _create_smc_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create SMC regime labels based on price efficiency and range utilization."""
        if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            # Fallback to simple return-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns > returns.rolling(25).std()).astype(int)
            return labels
        
        # SMC-specific labeling
        # Calculate price efficiency
        mid_price = (df['high'] + df['low']) / 2
        price_efficiency = (df['close'] - mid_price) / mid_price
        
        # Calculate future efficiency
        future_mid_price = (df['high'].shift(-lookforward) + df['low'].shift(-lookforward)) / 2
        future_efficiency = (df['close'].shift(-lookforward) - future_mid_price) / future_mid_price
        
        # Label: positive if efficiency improves and price moves in direction of efficiency
        efficiency_improvement = future_efficiency > price_efficiency
        price_direction = (df['close'].shift(-lookforward) > df['close']) == (price_efficiency > 0)
        
        labels = (efficiency_improvement & price_direction).astype(int)
        
        return labels
    
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
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced SMC regime step."""
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
                model="enhanced_smc_regime",
            )

            tprint_info(f"🚀 Starting Enhanced SMC Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced SMC features...")
            feature_df = self._generate_enhanced_smc_features(market_data, config)
            
            # 3-5. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='volume',
                pt_sl_config_key='smc_pt_sl',
                default_pt_sl=[2.0, 1.0]
            )

            # 6. Centralized purged-CV training
            tprint_info("🤖 Training Enhanced SMC model with centralized XGB helper (purged CV & AFML weights)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
            )

            oof_probs = training_result.oof_predictions
            last_model = training_result.model

            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                y_full_pred = (y_full_pred_prob >= 0.5).astype(int)
                
                metrics = training_result.metrics.copy()
                if 'auc' not in metrics:
                    try:
                        metrics['auc'] = float(roc_auc_score(y_full_true, y_full_pred_prob))
                    except Exception:
                        metrics['auc'] = 0.5
                if 'mi_score' not in metrics:
                    try:
                        metrics['mi_score'] = float(self.compute_binned_mi(y_full_pred_prob, y_full_true.values))
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate full OOF metrics: {e}")
                        metrics['mi_score'] = 0.0
            else:
                metrics = {'auc': 0.5, 'mi_score': 0.0}
                y_full_pred_prob = np.array([])
                y_full_pred = np.array([])

            metrics.update({
                'n_features': len(X.columns),
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            # AFML FIX: Initialize with NaN instead of 0.5 to allow proper ffilling downstream
            final_probs = pd.Series(np.nan, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            full_labels.loc[y.index] = y

            result = self.save_specialist_results(
                config=config,
                feature_df=feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X),
                labels=full_labels,
                predictions=final_preds.values,
                probabilities=final_probs.values,
                model=last_model,
                metrics=metrics,
                specialist_name="EnhancedMLSMCRegimeStep"
            )

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(X)

            result["execution_time"] = execution_time
            result["mi_history"] = self.mi_history
            result["training_metrics"] = self.training_metrics

            tprint_success(f"✅ Enhanced SMC Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return result

        except Exception as e:
            self.logger.exception(f"❌ Enhanced SMC Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
            # market_data = self._load_alternative_market_data(config, timeframe)
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
