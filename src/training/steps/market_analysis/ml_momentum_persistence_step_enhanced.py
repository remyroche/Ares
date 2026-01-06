"""
Enhanced ML Momentum Persistence Step with MI Improvements

This enhanced version implements:
- AFML hardening: CUSUM filtering (Price), Triple Barrier Method, Uniqueness weighting
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import logging
import time
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, roc_auc_score

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
    frac_diff_fixed, compute_structural_inertia, get_sample_weights
)
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.base_step import BaseStep
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)


class EnhancedMLMomentumPersistenceStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            # Discover repo root dynamically
            repo_root = Path(__file__).resolve().parents[4]
            self._artifact_router = ArtifactRouter(
                base_dir=str(repo_root / "artifacts"),
                versioned_store_dir=str(repo_root / "versioned_artifacts"),
                historical_data_dir=str(repo_root / "historical_data"),
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    def __init__(self, step_name: str = "enhanced_ml_momentum_persistence_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLMomentumPersistenceStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    def versioned_store(self):
        """Use a specialist-specific versioned store path."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            symbol = self._current_context.get('symbol', 'UNKNOWN')
            exchange = self._current_context.get('exchange', 'binance')
            timeframe = self._current_context.get('timeframe', '15m')
            direction = self._current_context.get('direction', 'long')
            model = self.step_name

            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            # Discover repo root dynamically
            repo_root = Path(__file__).resolve().parents[4]
            store_root = repo_root / "versioned_artifacts"
            store_root.mkdir(parents=True, exist_ok=True)
            store_path = store_root / store_name

            self._versioned_store = VersionedArtifactStore(
                store_path=str(store_path),
                auto_version=True,
                enable_row_versioning=True
            )

            if hasattr(self._versioned_store, '_metadata'):
                self._versioned_store._metadata['context'] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'model': model
                }
                self._versioned_store._save_metadata()

        return self._versioned_store
        
    def _compute_structural_inertia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Structural Inertia features:
        1. Fractional Differentiation (d=0.3 to 0.7)
        2. Normalized Regression Slope
        """
        features = pd.DataFrame(index=df.index)
        
        # Apply Fractional Differentiation to preserve memory
        d_val = 0.4
        close_fd = frac_diff_fixed(df['close'], d=d_val)
        # Re-index to match original df
        close_fd = close_fd.reindex(df.index).fillna(method='ffill').fillna(0)
        
        # Calculate Structural Inertia (Slope / SE) on FD series
        for window in [20, 40, 60]:
            features[f'structural_inertia_{window}'] = compute_structural_inertia(close_fd, window=window)
            # Acceleration of inertia
            features[f'inertia_accel_{window}'] = features[f'structural_inertia_{window}'].diff()
            
        return features

    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type: SpecialistType = None) -> pd.DataFrame:
        """Generate Structural Inertia features for Momentum Persistence."""
        # 1. Structural Inertia focus
        inertia_features = self._compute_structural_inertia_features(df)
        
        # 2. Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'momentum', {'enhanced_features': True}
        )
        
        # Combine all features
        all_features = [inertia_features, enhanced_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        return pd.DataFrame(index=df.index)
    
    def _create_manual_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features to address redundancy and improve poor performers."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Address smc|volume_force_breakout redundancy (0.632)
            # Create orthogonal regime features
            if 'smc_predicted' in enhanced_features.columns and 'vol_force_breakout' in enhanced_features.columns:
                smc = enhanced_features['smc_predicted']
                vol = enhanced_features['vol_force_breakout']
                
                # Standardize for orthogonal decomposition
                smc_std = (smc - smc.mean()) / (smc.std() + 1e-8)
                vol_std = (vol - vol.mean()) / (vol.std() + 1e-8)
                
                # Create orthogonal volume signal (remove smc component)
                if len(smc_std) > 1:
                    cov_matrix = np.cov(vol_std, smc_std)
                    if cov_matrix.shape == (2, 2):
                        orthogonal_vol = vol_std - (cov_matrix[0,1] / (np.var(smc_std) + 1e-8)) * smc_std
                        orthogonal_vol = orthogonal_vol / (orthogonal_vol.std() + 1e-8)
                        manual_features['orthogonal_volume_regime'] = orthogonal_vol
                
                # Regime divergence (captures disagreement)
                regime_divergence = np.abs(smc_std - vol_std)
                manual_features['regime_divergence'] = regime_divergence
                
                # Regime consensus (captures agreement)
                regime_consensus = (smc_std + vol_std) / 2
                manual_features['regime_consensus'] = regime_consensus
            
            # 2. Improve risk_score
            medium_vol = returns.rolling(20).std()
            
            # Volume-adjusted risk
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            volume_adjusted_risk = medium_vol * (1 + np.log(volume_ratio + 1))
            manual_features['enhanced_volume_adjusted_risk'] = volume_adjusted_risk
            
            # Range-based risk
            range_ratio = (high - low) / (close + 1e-8)
            range_vol = range_ratio.rolling(20).std()
            manual_features['enhanced_range_based_risk'] = range_vol
            
            # Downside risk
            downside_returns = returns.copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(20).std()
            manual_features['enhanced_downside_risk'] = downside_vol
            
            # Risk regime classification
            risk_zscore = (medium_vol - medium_vol.rolling(100).mean()) / (medium_vol.rolling(100).std() + 1e-8)
            manual_features['enhanced_risk_regime'] = np.where(risk_zscore > 1, 2, np.where(risk_zscore < -1, 0, 1))
            
            # 3. Improve path_risk_score
            price_path = close.rolling(10).mean()
            path_smoothness = np.abs(price_path.diff().diff())
            manual_features['enhanced_path_smoothness'] = path_smoothness
            
            path_velocity = close.rolling(5).mean().diff()
            path_acceleration = path_velocity.diff()
            manual_features['enhanced_path_acceleration'] = path_acceleration
            
            path_vol = path_velocity.rolling(20).std()
            manual_features['enhanced_path_volatility'] = path_vol
            
            path_range = price_path.rolling(20).max() - price_path.rolling(20).min()
            path_breakout = np.abs(close - price_path) / (path_range + 1e-8)
            manual_features['enhanced_path_breakout'] = path_breakout
            
            # 4. Additional orthogonal momentum features
            momentum_regime = (returns.rolling(20).mean() > 0).astype(int)
            manual_features['momentum_regime'] = momentum_regime
            
            vol_adjusted_momentum = returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-8)
            manual_features['vol_adjusted_momentum'] = vol_adjusted_momentum
            
            momentum_persistence = (returns.rolling(5).mean() * returns.rolling(10).mean()).rolling(5).sum()
            manual_features['momentum_persistence'] = momentum_persistence
            
        return manual_features
    
    def _apply_manual_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection to reduce redundancy and keep high-quality features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant features")
        
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
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:
                        to_drop.append(correlated_feature)
        
        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
            self.logger.info(f"Removed {len(set(to_drop))} redundant features")
        
        # Keep only the most informative features
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited to top 30 features by variance")
        
        return features
    
    def _train_enhanced_momentum_model(self, features: pd.DataFrame, labels: pd.Series, 
                                       sample_weight: Optional[pd.Series] = None) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced momentum model with MI optimization and AFML weights."""
        if len(features) < 100:
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
            dummy_model.fit(features, labels)
            return dummy_model, {'auc': 0.5, 'accuracy': 0.5, 'model_type': 'dummy_fallback'}
        
        training_result = train_specialist_xgb_with_oof(
            features.fillna(0.0),
            labels.fillna(0.0),
            sample_weight=sample_weight,
            n_splits=5,
        )
        return training_result.model, training_result.metrics

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced momentum persistence specialist training with AFML hardening."""
        start_time = time.time()
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model=self.step_name
            )
            
            self._versioned_store = None
            _ = self.versioned_store
            
            df = self._load_market_data(symbol, exchange, timeframe)
            if df is None or len(df) < 1000:
                return {"success": False, "error": "Insufficient data"}
            
            # 1. Feature Generation
            tprint_info("🛠️ Generating enhanced momentum features...")
            feature_df = self._generate_enhanced_features(df, SpecialistType.MOMENTUM_PERSISTENCE)
            
            # 2. AFML: CUSUM Sampling (Price-based for Momentum)
            tprint_info("🎯 Applying AFML CUSUM sampling...")
            sampled_df, t_events = self.apply_afml_sampling(df, config, filter_type='price')
            
            # 3. AFML: Triple Barrier Labels
            # Momentum refactored: Success = 2.5 sigma, Failure = Trendline break
            # We use pt_sl = [2.5, 1.0] where PT is 2.5 sigma and SL is 1.0 sigma (proxy for trendline break)
            pt_sl = config.get('momentum_pt_sl', [2.5, 1.0])
            tbm_labels_df = self.generate_tbm_labels(df, t_events, config, pt_sl)
            
            # 4. AFML: Alignment and Uniqueness Weighting
            X_sampled = feature_df.loc[t_events]
            y_sampled = tbm_labels_df['bin']
            t1_sampled = tbm_labels_df['t1']
            ret_sampled = tbm_labels_df['ret']
            
            # AFML Hardening: Sample Weighting (u_bar * |return|)
            num_concurrent = self.get_concurrent_weights(t1_sampled, df.index)
            # Note: afml_specialist_mixin.get_concurrent_weights currently returns uniqueness weights
            # We want the combined weighting: uniqueness * |return|
            weights_sampled = get_sample_weights(t1_sampled, num_concurrent, ret_sampled)
            
            # Filter numeric and drop NaNs
            X = X_sampled.select_dtypes(include=[np.number])
            valid_mask = X.notna().all(axis=1) & y_sampled.notna()
            X, y, weights = X.loc[valid_mask], y_sampled.loc[valid_mask], weights_sampled.loc[valid_mask]
            
            if len(X) < 100:
                tprint_warning(f"⚠️ Low sample count after AFML filtering: {len(X)}")
            
            tprint_info(f"📊 Training Data (AFML Sampled): {len(X)} samples, {len(X.columns)} features")
            
            # 5. Centralized purged-CV training
            tprint_info("🤖 Training enhanced momentum model with centralized XGB helper (purged CV & AFML weights)...")
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
            
            # 6. Align results back to full market index
            # AFML FIX: Initialize with NaN instead of 0.5 to allow proper ffilling downstream
            final_probs = pd.Series(np.nan, index=df.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            # Ffill probabilities so the signal is persistent between events
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=df.index)
            full_labels.loc[y.index] = y
            
            # 7. Standardized Output and Artifacts
            output_df = self._create_standardized_output(
                feature_df, full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
            
            artifact_name = f"enhanced_momentum_persistence_prediction_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLMomentumPersistenceStep",
                config=config,
                metrics=metrics,
                mi_score=metrics.get('mi_score', 0.0),
                hsic_score=0.0
            )
            
            self._save_artifact(data=output_df, artifact_name=artifact_name, artifact_type="data", data_category="predictions", metadata=metadata)
            
            try:
                if self.versioned_store:
                    self.versioned_store.add_data(output_df, version_name=artifact_name)
                    tprint_success(f"💾 Saved predictions to versioned store as '{artifact_name}'")
            except Exception as ve:
                tprint_warning(f"Versioned store save failed: {ve}")
            
            self._save_artifact(data=last_model, artifact_name=f"enhanced_momentum_model_{timeframe}", artifact_type="model", data_category="models", metadata=metadata)
            
            # 7. Diagnostics
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            tprint_success(f"✅ Enhanced Momentum Persistence completed in {time.time()-start_time:.2f}s")
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(X),
                "artifact_name": artifact_name,
                "diagnostics": diagnostics_result
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced momentum persistence failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        output_df = pd.DataFrame(index=features.index)
        output_df['timestamp'] = features.index
        output_df['specialist_prediction'] = predictions
        output_df['specialist_probability'] = probabilities
        output_df['target_label'] = labels
        for col in features.columns[:20]:
            output_df[f'feature_{col}'] = features[col]
        return output_df

    def _load_market_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load market data using BaseStep method."""
        market_data, _ = self.load_market_data_or_fail(
            {"symbol": symbol, "exchange": exchange, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data
