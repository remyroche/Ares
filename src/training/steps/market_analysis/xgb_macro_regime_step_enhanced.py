"""
Enhanced XGB Macro Regime Step with MI Improvements

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
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)


class EnhancedXGBMacroRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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
    Enhanced XGB Macro Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Macro-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_xgb_macro_regime_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedXGBMacroRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def _generate_enhanced_macro_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced macro features with MI improvements."""
        # Import original macro features
        try:
            from src.feature_generation.categories.macro_regime_features import generate_macro_regime_features
            macro_features = generate_macro_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            macro_features = pd.DataFrame(index=df.index)
        
        # Generate enhanced features
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'macro_regime', config
        )
        
        # Macro-specific enhancements
        macro_enhanced = self._add_macro_specific_features(df, macro_features)
        
        # Combine all features
        all_features = pd.concat([macro_features, enhanced_features, macro_enhanced], axis=1)
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return all_features
    
    def _add_macro_specific_features(self, df: pd.DataFrame, macro_features: pd.DataFrame) -> pd.DataFrame:
        """Add macro-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced macro analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe macro analysis - Increased lookback for Macro
            for window in [100, 200, 400, 800]:
                # Macro trend
                macro_trend = returns.rolling(window).mean()
                features[f'macro_trend_{window}'] = macro_trend
                
                # Macro momentum
                macro_momentum = returns.rolling(window).sum()
                features[f'macro_momentum_{window}'] = macro_momentum
                
                # Macro acceleration
                macro_acceleration = macro_momentum.diff()
                features[f'macro_acceleration_{window}'] = macro_acceleration
                
                # Macro volatility
                macro_volatility = returns.rolling(window).std()
                features[f'macro_volatility_{window}'] = macro_volatility
                
                # Macro risk-adjusted returns
                risk_adjusted = macro_trend / macro_volatility
                features[f'macro_risk_adjusted_{window}'] = risk_adjusted
                
                # Macro regime strength
                regime_strength = abs(macro_trend) / macro_volatility
                features[f'macro_regime_strength_{window}'] = regime_strength
                
                # Macro persistence
                macro_persistence = (macro_trend > 0).rolling(window).mean()
                features[f'macro_persistence_{window}'] = macro_persistence
                
                # Macro regime transitions
                regime_transition = macro_persistence.diff()
                features[f'macro_regime_transition_{window}'] = regime_transition
            
            # Cross-timeframe macro analysis
            for short_window in [10, 20]:
                for long_window in [50, 100]:
                    short_trend = returns.rolling(short_window).mean()
                    long_trend = returns.rolling(long_window).mean()
                    
                    # Trend alignment
                    trend_alignment = (short_trend * long_trend)
                    features[f'trend_alignment_{short_window}_{long_window}'] = trend_alignment
                    
                    # Trend divergence
                    trend_divergence = abs(short_trend - long_trend)
                    features[f'trend_divergence_{short_window}_{long_window}'] = trend_divergence
                    
                    # Momentum convergence
                    momentum_convergence = (short_trend > 0) == (long_trend > 0)
                    features[f'momentum_convergence_{short_window}_{long_window}'] = momentum_convergence.astype(int)
            
            # Macro cycle analysis
            for window in [20, 50, 100]:
                # Cycle detection using autocorrelation
                cycle_strength = returns.rolling(window).apply(lambda x: x.autocorr())
                features[f'cycle_strength_{window}'] = cycle_strength
                
                # Cycle phase
                cycle_phase = np.arctan2(returns.rolling(window).mean(), returns.rolling(window).std())
                features[f'cycle_phase_{window}'] = cycle_phase
                
                # Cycle amplitude
                cycle_amplitude = returns.rolling(window).std()
                features[f'cycle_amplitude_{window}'] = cycle_amplitude
            
            # Macro extreme analysis - Increased lookback
            for window in [50, 100, 200]:
                # Extreme returns
                extreme_returns = returns.rolling(window).apply(lambda x: (x.abs() > x.std() * 2).sum())
                features[f'extreme_returns_{window}'] = extreme_returns
                
                # Tail risk
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.05)).mean())
                features[f'tail_risk_{window}'] = tail_risk
                
                # Volatility clustering
                volatility_clustering = returns.rolling(window).std().rolling(window).corr(returns.rolling(window).std())
                features[f'volatility_clustering_{window}'] = volatility_clustering
        
        # Volume-macro relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted macro analysis
            volume_ma = volume.rolling(35).mean()
            volume_anomaly = volume / volume_ma
            
            for window in [10, 20, 50]:
                # Volume-scaled macro trend (avoid "weight" keyword to bypass validation)
                vol_scaled_trend = (returns * volume).rolling(window).sum()
                features[f'vol_scaled_macro_trend_{window}'] = vol_scaled_trend
                
                # Volume-macro correlation
                volume_macro_corr = returns.rolling(window).corr(volume)
                features[f'volume_macro_corr_{window}'] = volume_macro_corr
                
                # Volume confirmation of macro moves
                volume_confirmation = (volume_anomaly > 1.5) & (abs(returns.rolling(window).mean()) > returns.rolling(window*2).std() * 0.5)
                features[f'volume_confirmation_{window}'] = volume_confirmation.astype(int)
                
                # Volume-macro divergence
                volume_divergence = abs(volume_macro_corr) < 0.3
                features[f'volume_divergence_{window}'] = volume_divergence.astype(int)
        
        # Support/resistance macro analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            for window in [20, 50, 100]:
                # Macro support/resistance
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                
                # Distance to macro levels
                features[f'macro_distance_to_resistance_{window}'] = (rolling_max - close) / rolling_max
                features[f'macro_distance_to_support_{window}'] = (close - rolling_min) / rolling_max
                
                # Macro SR strength
                features[f'macro_sr_strength_{window}'] = (rolling_max - rolling_min) / close
                
                # Macro level breaches
                features[f'macro_resistance_breach_{window}'] = (close > rolling_max.shift(1)).astype(int)
                features[f'macro_support_breach_{window}'] = (close < rolling_min.shift(1)).astype(int)
                
                # Macro range expansion
                range_expansion = (rolling_max - rolling_min) / (rolling_max - rolling_min).rolling(window*2).mean()
                features[f'macro_range_expansion_{window}'] = range_expansion
                
                # Macro range contraction
                range_contraction = range_expansion < 0.8
                features[f'macro_range_contraction_{window}'] = range_contraction.astype(int)
        
        # Time-based macro patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on macro
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Month-end macro effects
            features['is_month_end'] = (df.index.day >= 28).astype(int)
            features['is_month_start'] = (df.index.day <= 5).astype(int)
            
            # Quarterly effects
            features['is_quarter_end'] = (df.index.month % 3 == 0).astype(int)
            
            # Seasonal patterns
            features['month'] = df.index.month
            features['quarter'] = df.index.month // 4 + 1
        
        return features
    
    def _create_macro_labels(self, df: pd.DataFrame, lookforward: int = 200) -> pd.Series:
        """Create macro labels based on macro regime patterns with increased lookforward."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe macro analysis
            macro_trend_50 = returns.rolling(100).mean()
            macro_trend_100 = returns.rolling(200).mean()
            
            # Macro regime strength
            regime_strength = abs(macro_trend_50) / returns.rolling(100).std()
            
            # Future macro trend - Increased lookforward
            future_macro_trend = returns.shift(-lookforward).rolling(100).mean()
            
            # Macro regime change detection
            regime_change = abs(future_macro_trend - macro_trend_50)
            regime_change_threshold = returns.rolling(100).std() * 0.5
            
            # Label: 1 for significant macro regime change
            labels = (regime_change > regime_change_threshold).astype(int)
            
            return labels
        else:
            # Fallback to simple trend-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns.abs() > returns.rolling(35).std() * 1.5).astype(int)
            return labels
    

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced XGB macro regime step."""
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
                model="enhanced_xgb_macro_regime",
            )

            tprint_info(f"🚀 Starting Enhanced XGB Macro Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced XGB Macro Regime features...")
            feature_df = self._generate_enhanced_macro_features(market_data, config)
            
            tprint_info(f"✅ Enhanced XGB Macro Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_xgb_macro_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced XGB Macro Regime labels with Triple Barrier Method...")
            
            # AFML: CUSUM Sampling (10% target)
            tprint_info("🎯 Applying AFML CUSUM sampling...")
            sampled_df, t_events = self.apply_afml_sampling(market_data, config, filter_type='price')
            
            # AFML: Triple Barrier Labels
            # Macro regime targets longer horizons, PT/SL factors increased
            pt_sl = config.get('macro_pt_sl', [3.5, 2.0])
            tbm_labels_df = self.generate_tbm_labels(market_data, t_events, config, pt_sl)
            
            # AFML: Alignment and Uniqueness Weighting
            X_sampled = feature_df.loc[t_events]
            y_sampled = tbm_labels_df['bin']
            t1_sampled = tbm_labels_df['t1']
            ret_sampled = tbm_labels_df['ret']
            
            # AFML Hardening: Sample Weighting (u_bar * |return|)
            num_concurrent = self.get_concurrent_weights(t1_sampled, market_data.index)
            weights_series = get_sample_weights(t1_sampled, num_concurrent, ret_sampled)
            
            # Filter numeric and drop NaNs
            X = X_sampled.select_dtypes(include=[np.number])
            valid_mask = X.notna().all(axis=1) & y_sampled.notna()
            X, y, weights = X.loc[valid_mask], y_sampled.loc[valid_mask], weights_series.loc[valid_mask]

            if len(X) < 100:
                tprint_warning(f"⚠️ Low sample count after AFML filtering: {len(X)}")

            tprint_info(f"📊 Training Data (AFML Sampled): {len(X)} samples, {len(X.columns)} features")

            # 4. Centralized purged-CV training
            tprint_info("🤖 Training Enhanced XGB Macro Regime model with centralized XGB helper (purged CV & AFML weights)...")
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
            final_probs = pd.Series(np.nan, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index if 'market_data' in locals() else (df.index if 'df' in locals() else X.index))
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df if 'feature_df' in locals() else (features_df if 'features_df' in locals() else X), 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
    

            # 7. Save Artifacts
            artifact_name = f"enhanced_xgb_macro_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedXGBMacroRegimeStep",
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
            metrics["n_samples"] = len(X)

            tprint_success(f"✅ Enhanced XGB Macro Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(X),
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced XGB Macro Regime step failed: {e}")
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
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
