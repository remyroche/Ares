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
    
    def _generate_enhanced_meso_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced meso features with MI improvements."""
        # Import original meso features
        try:
            from src.feature_generation.categories.meso_regime_features import generate_meso_regime_features
            meso_features = generate_meso_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            meso_features = pd.DataFrame(index=df.index)
        
        # Generate enhanced features
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'meso_regime', config
        )
        
        # Meso-specific enhancements
        meso_enhanced = self._add_meso_specific_features(df, meso_features)
        
        # Combine all features
        all_features = pd.concat([meso_features, enhanced_features, meso_enhanced], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Remove duplicates and clean
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
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
    

    def _create_meso_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create meso labels based on meso regime patterns."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe meso analysis
            meso_trend_10 = returns.rolling(15).mean()
            meso_trend_20 = returns.rolling(25).mean()
            
            # Meso regime strength
            regime_strength = abs(meso_trend_10) / returns.rolling(15).std()
            
            # Future meso trend
            future_meso_trend = returns.shift(-lookforward).rolling(15).mean()
            
            # Meso regime change detection
            regime_change = abs(future_meso_trend - meso_trend_10)
            regime_change_threshold = returns.rolling(15).std() * 0.3
            
            # Label: 1 for significant meso regime change
            labels = (regime_change > regime_change_threshold).astype(int)
            
            return labels
        else:
            # Fallback to simple trend-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns.abs() > returns.rolling(15).std()).astype(int)
            return labels

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced XGB meso regime step."""
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
                model="enhanced_xgb_meso_regime",
            )

            tprint_info(f"🚀 Starting Enhanced XGB Meso Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced XGB Meso Regime features...")
            feature_df = self._generate_enhanced_meso_features(market_data, config)
            
            tprint_info(f"✅ Enhanced XGB Meso Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_xgb_meso_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3-5. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            tprint_info("🎯 Applying AFML hardening (CUSUM, TBM, Hardened Weights)...")
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='price',
                pt_sl_config_key='meso_pt_sl',
                default_pt_sl=[2.0, 1.0]
            )

            # 4. Centralized purged-CV training
            tprint_info("🤖 Training Enhanced XGB Meso Regime model with centralized XGB helper (purged CV & AFML weights)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
            )

            oof_probs = training_result.oof_predictions
            last_model = training_result.model
            metrics = training_result.metrics
            
            # AFML Audit: Update metrics using full OOF set
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                y_full_true = y.loc[valid_oof.index]
                y_full_pred_prob = valid_oof.values
                y_full_pred = (y_full_pred_prob >= 0.5).astype(int)
                
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
                'n_samples': len(X),
            })

            # 5. Generate Final Standardized Output (Aligned to full market_data index)
            # AFML FIX: Initialize with NaN instead of 0.5 to allow proper ffilling downstream
            final_probs = pd.Series(np.nan, index=market_data.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index)
            full_labels.loc[y.index] = y

            result = self.save_specialist_results(
                config=config,
                feature_df=feature_df,
                labels=full_labels,
                predictions=final_preds.values,
                probabilities=final_probs.values,
                model=last_model,
                metrics=metrics,
                specialist_name="EnhancedXGBMesoRegimeStep"
            )

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(X)

            result["execution_time"] = execution_time
            result["mi_history"] = self.mi_history
            result["training_metrics"] = self.training_metrics

            tprint_success(f"✅ Enhanced XGB Meso Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return result

        except Exception as e:
            self.logger.exception(f"❌ Enhanced XGB Meso Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
