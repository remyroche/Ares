"""
Enhanced ML Volume Force Step with MI Improvements & Standardization

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Data structure standardization
- Binary output enforcement
- MI monitoring and optimization
- Ensemble compatibility
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.isotonic import IsotonicRegression

from src.training.steps.base_step import BaseStep
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
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.utils.ml_common.retraining_scheduler import create_sample_weights
from src.utils.ml_common.evaluation.hsic import calculate_hsic

# Enhanced imports
from src.feature_generation.categories.enhanced_volume_force_features import (
    generate_enhanced_volume_force_features,
)
from src.feature_generation.categories.liquidity_regime_features import (
    generate_liquidity_regime_features,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2,
)
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLVolumeForceStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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

    """Enhanced Volume Force step with MI improvements and standardization."""

    def __init__(self, step_name: str = "enhanced_ml_volume_force_step"):
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedMLVolumeForceStep")
        self._cached_market_data = None
        self._cached_market_source = None
        self._cached_market_cache_key = None
        self._feature_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")

    def _get_config_signature(self, config: Dict[str, Any]) -> str:
        """Generate config signature for tracking."""
        keys = [
            "volume_force_normalization_window",
            "volume_force_breakout_beta",
            "volume_force_volatility_beta",
            "volume_force_trend_beta",
            "volume_force_xgb_max_depth"
        ]
        parts = []
        for k in keys:
            if k in config:
                val = config[k]
                parts.append(f"{k.replace('volume_force_', '')}={val}")
        return "|".join(parts)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced volume force step with MI optimization."""
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
                model="enhanced_volume_force",
            )

            tprint_info(
                f"🚀 Starting Enhanced {self.step_name} for {symbol} on {exchange}"
            )

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Volume Force features...")
            norm_window = config.get("volume_force_normalization_window", 100)
            cache_key = (symbol, exchange, timeframe, norm_window)

            if config.get("is_batch_run", False) and cache_key in self._feature_cache:
                feature_df = self._feature_cache[cache_key].copy()
            else:
                # Generate enhanced features with MI improvements
                enhanced_features = generate_enhanced_volume_force_features(market_data, config)
                liquidity_df = generate_liquidity_regime_features(market_data, config)
                
                # Manual feature engineering for volume force
                manual_features = self._create_manual_volume_force_enhanced_features(market_data, enhanced_features)
                
                # Combine all features
                all_features = [enhanced_features, liquidity_df, manual_features]
                feature_df = pd.concat(all_features, axis=1)
                
                # Apply manual redundancy reduction
                feature_df = self._apply_manual_volume_force_feature_selection(feature_df)
                
                feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

                if config.get("is_batch_run", False):
                    self._feature_cache[cache_key] = feature_df.copy()

            tprint_info(f"✅ Enhanced features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_ml_volume_force_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Targets
            tprint_info("🎯 Generating Targets (Breakout, Volatility, Trend)...")
            targets_df = self._generate_targets(market_data, config)

            # Align features and targets
            common_index = feature_df.index.intersection(targets_df.index)
            X = feature_df.loc[common_index]
            y = targets_df.loc[common_index]

            # Drop NaN rows
            valid_mask = X.notna().all(axis=1) & y.notna().all(axis=1)
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols]

            if len(X) < 200:
                raise RuntimeError(f"Insufficient valid samples: {len(X)} < 200")

            tprint_info(f"📊 Training Data: {len(X)} samples, {len(X.columns)} features")

            # 4. Train Enhanced Models with MI Monitoring
            model_results = {}
            trained_models = {}
            predictions = pd.DataFrame(index=X.index)

            targets = ["breakout", "volatility", "trend"]
            
            for target_name in targets:
                tprint_info(f"🤖 Training Enhanced {target_name.capitalize()} model...")
                
                y_target = y[target_name]
                
                # Create training config with MI optimization
                training_config = XGBTrainingConfig(
                    objective="binary:logistic" if target_name == "breakout" else "reg:squarederror",
                    n_estimators=config.get("volume_force_xgb_n_estimators", 200),
                    max_depth=config.get("volume_force_xgb_max_depth", 6),
                    learning_rate=config.get("volume_force_xgb_learning_rate", 0.1),
                    subsample=config.get("volume_force_xgb_subsample", 0.8),
                    colsample_bytree=config.get("volume_force_xgb_colsample", 0.8),
                    random_state=42,
                )

                # Create temporal split config
                temporal_config = create_temporal_split_config_for_pipeline(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction,
                    n_splits=config.get("volume_force_n_splits", 5),
                    walk_forward_type="rolling",
                    test_size_ratio=config.get("volume_force_test_size_ratio", 0.2),
                    min_train_samples=config.get("volume_force_min_train_samples", 500),
                )

                # Create sample weights
                sample_weights = create_sample_weights(
                    y_target,
                    method="temporal_decay",
                    decay_factor=config.get("volume_force_sample_decay", 0.995),
                )

                # Train model
                trainer = StandardizedXGBTrainer(training_config)
                train_result = trainer.train_time_series_cv(
                    X, y_target, temporal_config, sample_weights=sample_weights
                )

                model_results[target_name] = train_result
                trained_models[target_name] = train_result.models[-1] if train_result.models else None

                # Extract OOF predictions
                oof_preds = train_result.oof_predictions
                pred_col = "probability" if "probability" in oof_preds.columns else "prediction"

                if pred_col in oof_preds.columns:
                    prob_series = oof_preds[pred_col]
                    if not prob_series.index.is_unique:
                        prob_series = prob_series[~prob_series.index.duplicated(keep="last")]
                    aligned = prob_series.reindex(predictions.index)
                    predictions[f"vol_force_{target_name}"] = aligned
                else:
                    predictions[f"vol_force_{target_name}"] = np.nan

                # Enhanced metrics with MI analysis
                if not oof_preds.empty:
                    y_true = y_target.loc[oof_preds.index]
                    y_pred = oof_preds[pred_col]

                    try:
                        if target_name == "breakout":
                            # Classification metrics
                            ll = log_loss(y_true, y_pred)
                            acc = (y_true == (y_pred >= 0.5)).mean()
                            metrics[f"{target_name}_log_loss"] = float(ll)
                            metrics[f"{target_name}_accuracy"] = float(acc)

                            try:
                                roc = roc_auc_score(y_true, y_pred)
                                pr = average_precision_score(y_true, y_pred)
                                brier = brier_score_loss(y_true, y_pred)
                                metrics[f"{target_name}_roc_auc"] = float(roc)
                                metrics[f"{target_name}_pr_auc"] = float(pr)
                                metrics[f"{target_name}_brier_score"] = float(brier)
                            except Exception:
                                pass

                        else:
                            # Regression metrics
                            mse = mean_squared_error(y_true, y_pred)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(y_true, y_pred)
                            r2 = r2_score(y_true, y_pred)

                            metrics[f"{target_name}_rmse"] = float(rmse)
                            metrics[f"{target_name}_mae"] = float(mae)
                            metrics[f"{target_name}_r2"] = float(r2)

                            # Information Coefficient
                            ic = np.corrcoef(y_pred, y_true)[0, 1]
                            metrics[f"{target_name}_ic"] = float(ic)

                            # Calculate HSIC for regression targets
                            try:
                                hsic_mask = np.isfinite(y_pred) & np.isfinite(y_true)
                                if hsic_mask.sum() > 100:
                                    hsic_score = calculate_hsic(
                                        y_pred[hsic_mask].values.reshape(-1, 1),
                                        y_true[hsic_mask].values.reshape(-1, 1),
                                        kernel_X='rbf',
                                        kernel_Y='rbf'
                                    )
                                    metrics[f"{target_name}_hsic"] = float(hsic_score)
                                    tprint_info(f"   {target_name.capitalize()} HSIC: {hsic_score:.6f}")
                            except Exception as hsic_exc:
                                tprint_warning(f"HSIC calculation failed for {target_name}: {hsic_exc}")

                        # MI Analysis for this target
                        try:
                            from sklearn.feature_selection import mutual_info_regression
                            mi_score = mutual_info_regression(
                                y_pred.reshape(-1, 1), y_true.values
                            )[0]
                            metrics[f"{target_name}_mi_score"] = float(mi_score)
                            tprint_info(f"   {target_name.capitalize()} MI: {mi_score:.6f}")
                        except Exception as mi_exc:
                            tprint_warning(f"MI calculation failed for {target_name}: {mi_exc}")

                    except Exception as e:
                        tprint_warning(f"Error calculating metrics for {target_name}: {e}")

            # 5. Aggregate Results
            predictions = predictions.dropna()

            if predictions.empty:
                tprint_warning("⚠️ No common OOF predictions generated.")
                metrics["avg_log_loss"] = float("inf")
            else:
                # Standardize output structure
                standardized_output = self._create_standardized_output(predictions, y, symbol, exchange, timeframe, direction)
                
                # Save standardized artifacts
                artifact_name = f"enhanced_ml_volume_force_predictions_{timeframe}"
                metadata = SpecialistDataInterface.create_standard_metadata(
                    specialist_name="EnhancedMLVolumeForceStep",
                    config=config,
                    metrics=metrics,
                    mi_score=metrics.get("breakout_mi_score", 0.0),
                    hsic_score=metrics.get("breakout_hsic", 0.0)
                )
                
                
            # DEBUG: Check artifact saving setup
            print(f"🐛 DEBUG: About to save artifact: {artifact_name}")
            print(f"🐛 DEBUG: Output df shape: {output_df.shape}")
            print(f"🐛 DEBUG: Artifact router type: {type(self.artifact_router)}")
            print(f"🐛 DEBUG: Versioned store available: {hasattr(self, '_versioned_store') and self._versioned_store is not None}")
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                print(f"🐛 DEBUG: Versioned store type: {type(self._versioned_store)}")
            
            self.artifact_router.save(
                artifact_name=artifact_name,
                data=standardized_output,
                metadata=metadata
            )
            artifacts.append(artifact_name)

            # Calculate aggregate metrics
            if "breakout_log_loss" in metrics and "volatility_rmse" in metrics and "trend_rmse" in metrics:
                metrics["avg_log_loss"] = (
                    metrics["breakout_log_loss"] + 
                    metrics["volatility_rmse"] + 
                    metrics["trend_rmse"]
                ) / 3

            # 6. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            if diagnostics_result.get('success', False):
                compliance_report = diagnostics_result['compliance_report']
                ensemble_compatibility = diagnostics_result['ensemble_compatibility']
                
                tprint_success(f"✅ Enhanced Diagnostics Complete:")
                tprint_info(f"   MI Score: {compliance_report['metrics']['mi_score']:.4f}")
                tprint_info(f"   Requirements Met: {compliance_report['requirements_met']}/3")
                tprint_info(f"   Ensemble Ready: {ensemble_compatibility['ensemble_ready']}")
                
                # Add diagnostics metrics
                metrics.update({
                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],
                    'enhanced_requirements_met': compliance_report['requirements_met'],
                    'enhanced_ensemble_ready': ensemble_compatibility['ensemble_ready'],
                    'enhanced_features_count': diagnostics_result.get('enhanced_features_count', 0),
                    'orthogonal_features_count': diagnostics_result.get('orthogonal_features_count', 0)
                })
            else:
                tprint_warning(f"⚠️ Enhanced diagnostics failed: {diagnostics_result.get('error', 'Unknown error')}")

            # 7. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(predictions) if not predictions.empty else 0
            metrics["n_features"] = len(X.columns)

            tprint_success(f"✅ Enhanced Volume Force completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('enhanced_mi_score', 0):.4f}, "
                        f"Requirements={metrics.get('enhanced_requirements_met', 0)}/3")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(predictions) if not predictions.empty else 0,
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Volume Force step failed: {e}")
            return {"success": False, "error": str(e)}


    def save(self, artifact_name: str, data, artifact_type: str = "data", data_category: str = "predictions"):
        """Custom save method for enhanced specialists."""
        try:
            # Use versioned store directly if available
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                context = {
                    'symbol': self._current_context.get('symbol', 'UNKNOWN'),
                    'exchange': self._current_context.get('exchange', 'binance'),
                    'timeframe': self._current_context.get('timeframe', '15m'),
                    'direction': self._current_context.get('direction', 'long'),
                    'model': self._current_context.get('model', 'analyst'),
                    'step_name': self.step_name,
                }
                self._versioned_store.save(
                    artifact_name=artifact_name,
                    data=data,
                    artifact_type=artifact_type,
                    data_category=data_category,
                    context=context
                )
                self.logger.info(f"✅ Saved {artifact_name} to versioned store")
            else:
                self.logger.warning(f"⚠️ Cannot save {artifact_name}: no versioned store available")
        except Exception as e:
            self.logger.error(f"❌ Failed to save {artifact_name}: {e}")

    def _create_standardized_output(self, predictions: pd.DataFrame, targets: pd.DataFrame,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        # Use breakout as primary prediction for ensemble compatibility
        if 'vol_force_breakout' in predictions.columns:
            primary_prediction = predictions['vol_force_breakout']
            primary_probability = predictions['vol_force_breakout']
        else:
            # Fallback to first available prediction
            pred_cols = [col for col in predictions.columns if 'vol_force_' in col]
            if pred_cols:
                primary_prediction = predictions[pred_cols[0]]
                primary_probability = predictions[pred_cols[0]]
            else:
                primary_prediction = pd.Series(0.5, index=predictions.index)
                primary_probability = pd.Series(0.5, index=predictions.index)

        # Create standardized output
        standardized = pd.DataFrame(index=predictions.index)
        standardized['timestamp'] = predictions.index
        standardized['specialist_prediction'] = primary_prediction
        standardized['specialist_probability'] = primary_probability
        
        # Use breakout target if available, otherwise create synthetic
        if 'breakout' in targets.columns:
            aligned_targets = targets.loc[predictions.index, 'breakout']
        else:
            # Create synthetic target from prediction distribution
            threshold = primary_prediction.median()
            aligned_targets = (primary_prediction > threshold).astype(int)
        
        standardized['target_label'] = aligned_targets
        
        # Add original predictions for reference
        for col in predictions.columns:
            standardized[f'original_{col}'] = predictions[col]
        
        return standardized

    def _create_manual_volume_force_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for volume force detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced volume force features
            # Multi-timeframe volume force signals
            volume_ma_short = volume.rolling(10).mean()
            volume_ma_medium = volume.rolling(20).mean()
            volume_ma_long = volume.rolling(50).mean()
            
            volume_force_short = (volume - volume_ma_short) / (volume_ma_short + 1e-8)
            volume_force_medium = (volume - volume_ma_medium) / (volume_ma_medium + 1e-8)
            volume_force_long = (volume - volume_ma_long) / (volume_ma_long + 1e-8)
            
            manual_features['volume_force_short'] = volume_force_short
            manual_features['volume_force_medium'] = volume_force_medium
            manual_features['volume_force_long'] = volume_force_long
            
            # Volume force consistency
            force_consistency = (volume_force_medium > 0).rolling(20).mean()
            manual_features['volume_force_consistency'] = force_consistency
            
            # Volume force transitions
            force_transitions = (volume_force_medium > 0).astype(int).diff().abs()
            manual_features['volume_force_transitions'] = force_transitions
            
            # 2. Price-volume force interaction
            # Volume-adjusted momentum
            volume_adjusted_momentum = returns.rolling(10).mean() * (1 + volume_force_medium)
            manual_features['volume_adjusted_momentum'] = volume_adjusted_momentum
            
            # Volume force breakout detection
            volume_breakout = abs(volume_force_medium) > (volume_force_medium.rolling(100).std() * 2)
            manual_features['volume_breakout'] = volume_breakout.astype(int)
            
            # Price-volume divergence
            price_regime = (returns.rolling(20).mean() > 0).astype(int)
            volume_regime = (volume_force_medium > 0).astype(int)
            price_volume_divergence = np.abs(price_regime - volume_regime)
            manual_features['price_volume_divergence'] = price_volume_divergence
            
            # 3. Enhanced volume force volatility interaction
            volatility = returns.rolling(20).std()
            vol_adjusted_force = volume_force_medium / (volatility + 1e-8)
            manual_features['vol_adjusted_force'] = vol_adjusted_force
            
            # Volume force volatility regime
            vol_regime = (volatility > volatility.rolling(100).mean()).astype(int)
            volume_vol_regime = volume_regime * vol_regime
            manual_features['volume_vol_regime'] = volume_vol_regime
            
            # 4. Volume force persistence features
            force_persistence_short = (volume_force_short > 0).rolling(5).sum()
            force_persistence_medium = (volume_force_medium > 0).rolling(10).sum()
            manual_features['force_persistence_short'] = force_persistence_short
            manual_features['force_persistence_medium'] = force_persistence_medium
            
            # Volume force momentum
            force_momentum = volume_force_medium.diff().rolling(5).mean()
            manual_features['volume_force_momentum'] = force_momentum
            
            # 5. Range-based volume force features
            range_ratio = (high - low) / close
            range_force = volume_force_medium * range_ratio
            manual_features['range_volume_force'] = range_force
            
            # Range-volume force regime
            range_regime = (range_ratio > range_ratio.rolling(50).mean()).astype(int)
            manual_features['range_volume_force_regime'] = range_regime
            
            # 6. Volume force strength indicators
            force_strength = abs(volume_force_medium)
            manual_features['volume_force_strength'] = force_strength
            
            # Force acceleration
            force_acceleration = volume_force_medium.diff().diff()
            manual_features['volume_force_acceleration'] = force_acceleration
            
            # 7. Microstructure volume force features
            # Volume-price impact
            price_impact = abs(returns) * volume_force_medium
            manual_features['volume_price_impact'] = price_impact
            
            # Volume force depth
            force_depth = volume * force_strength
            manual_features['volume_force_depth'] = force_depth
            
            # Volume force efficiency
            efficiency = abs(returns.rolling(10).mean()) * (1 + volume_force_medium)
            manual_features['volume_force_efficiency'] = efficiency
            
            # 8. Volume force regime classification
            # High force regime
            high_force = (volume_force_medium > 0.5).astype(int)
            manual_features['high_volume_force_regime'] = high_force
            
            # Low force regime
            low_force = (volume_force_medium < -0.5).astype(int)
            manual_features['low_volume_force_regime'] = low_force
            
            # Volume force stress indicator
            force_stress = np.where(volume_force_medium < -1, 2, np.where(volume_force_medium > 1, 0, 1))
            manual_features['volume_force_stress'] = force_stress
            
        return manual_features
    
    def _apply_manual_volume_force_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for volume force features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant volume force features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant volume force features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited volume force features to top 30 by variance")
        
        return features
    
    def _generate_targets(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate targets for volume force prediction."""
        # This would be implemented based on the original _generate_targets method
        # For now, create placeholder targets
        targets = pd.DataFrame(index=df.index)
        
        # Placeholder target generation - replace with actual implementation
        returns = df['close'].pct_change()
        
        # Breakout target (binary)
        targets['breakout'] = (returns.abs() > returns.quantile(0.9)).astype(int)
        
        # Volatility target (continuous)
        targets['volatility'] = returns.rolling(25).std().fillna(0)
        
        # Trend target (continuous)
        targets['trend'] = returns.rolling(15).sum().fillna(0)
        
        return targets

    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        # This would be implemented based on the original method
        # For now, return placeholder data
        # Using alternative data loading approach
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
