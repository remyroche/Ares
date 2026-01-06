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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
)

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
from src.feature_generation.categories.enhanced_volume_force_features import (
    generate_enhanced_volume_force_features,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2,
)
from src.training.steps.market_analysis.afml_specialist_mixin import AFMLSpecialistMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof
from src.utils.versioned_artifacts import VersionedArtifactStore

logger = logging.getLogger(__name__)


class EnhancedMLVolumeForceStep(SpecialistDiagnosticsMixinEnhancedV2, AFMLSpecialistMixin, BaseStep):

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

    """Enhanced Volume Force step with MI improvements and standardization."""

    def __init__(self, step_name: str = "enhanced_ml_volume_force_step"):
        """Initialize the enhanced specialist."""
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        SpecialistDiagnosticsMixinEnhancedV2.__init__(self, step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLVolumeForceStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        self._market_data_cache = {}
        self._feature_cache = {}
        tprint(f"✅ Initialized Enhanced {step_name} (MI-Optimized)", "SUCCESS")
    
    @property
    def versioned_store(self):
        """Specialized versioned store aligned to repo-root versioned_artifacts."""
        if self._versioned_store is None and self.use_versioned_artifacts:
            symbol = self._current_context.get("symbol", "UNKNOWN")
            exchange = self._current_context.get("exchange", "binance")
            timeframe = self._current_context.get("timeframe", "15m")
            direction = self._current_context.get("direction", "long")
            model = self.step_name

            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
            repo_root = Path(__file__).resolve().parents[4]
            store_root = repo_root / "versioned_artifacts"
            store_root.mkdir(parents=True, exist_ok=True)
            store_path = store_root / store_name

            self._versioned_store = VersionedArtifactStore(
                store_path=str(store_path),
                auto_version=True,
                enable_row_versioning=True,
            )

            if hasattr(self._versioned_store, "_metadata"):
                self._versioned_store._metadata["context"] = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "direction": direction,
                    "model": model,
                }
                self._versioned_store._save_metadata()

        return self._versioned_store

    def _generate_enhanced_features(self, df: pd.DataFrame, specialist_type: SpecialistType) -> pd.DataFrame:
        """Generate enhanced features for Volume Force specialist."""
        # 1. Base Volume Force Features
        base_features = generate_enhanced_volume_force_features(df, {})
        
        # 2. Pipeline Features
        pipeline_features = self.feature_pipeline.generate_enhanced_features(
            df, 'volume_force', {'enhanced_features': True}
        )
        
        # 3. Manual Features
        manual_features = self._create_manual_volume_force_enhanced_features(df, pipeline_features)
        
        # Combine
        combined = pd.concat([base_features, pipeline_features, manual_features], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        # Selection
        selected = self._apply_manual_volume_force_feature_selection(combined)
        
        return selected.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
                model=self.step_name,
            )
            self._versioned_store = None
            _ = self.versioned_store

            tprint_info(f"🚀 Starting Enhanced {self.step_name} for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Volume Force features...")
            feature_df = self._generate_enhanced_features(market_data, SpecialistType.VOLUME_FORCE)
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

            # 3. AFML: Sampling, Labeling, Weighting, Alignment via Helper
            X, y, weights = self.prepare_specialist_data(
                market_data=market_data,
                feature_df=feature_df,
                config=config,
                filter_type='volume',
                pt_sl_config_key='volume_force_pt_sl',
                default_pt_sl=[1.5, 1.5]
            )

            # 4. Centralized purged-CV training
            tprint_info("🤖 Training Enhanced Volume Force model with centralized XGB helper (purged CV & AFML weights)...")
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

            metrics.update({
                'n_features': len(X.columns),
                'n_samples': len(X)
            })

            # 5. Generate Final Standardized Output
            final_probs = pd.Series(np.nan, index=market_data.index)
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = y_full_pred_prob
            
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            
            full_labels = pd.Series(0, index=market_data.index)
            full_labels.loc[y.index] = y

            standardized_output = self._create_standardized_output(
                feature_df, 
                full_labels, final_preds.values, final_probs.values, symbol, exchange, timeframe, direction
            )
    
            artifact_name = f"enhanced_ml_volume_force_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLVolumeForceStep",
                config=config,
                metrics=metrics,
                mi_score=metrics.get('mi_score', 0.0),
                hsic_score=0.0
            )
            
            self._save_artifact(data=standardized_output, artifact_name=artifact_name, artifact_type="data", data_category="predictions", metadata=metadata)
            
            try:
                if self.versioned_store:
                    self.versioned_store.add_data(standardized_output, version_name=artifact_name)
                    tprint_success(f"💾 Saved predictions to versioned store as '{artifact_name}'")
            except Exception as ve:
                tprint_warning(f"Versioned store save failed: {ve}")
            
            self._save_artifact(data=last_model, artifact_name=f"enhanced_volume_force_model_{timeframe}", artifact_type="model", data_category="models", metadata=metadata)
            
            # 6. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            if diagnostics_result.get('success', False):
                compliance_report = diagnostics_result['compliance_report']
                metrics.update({
                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],
                    'enhanced_requirements_met': compliance_report['requirements_met'],
                })

            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            tprint_success(f"✅ Enhanced Volume Force completed in {execution_time:.2f}s")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(X),
                "artifact_name": artifact_name,
                "diagnostics": diagnostics_result,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced Volume Force step failed: {e}")
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

    def _create_manual_volume_force_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for volume force detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            log_volume = np.log1p(volume)
            try:
                fd_vol = self.apply_fractional_diff(log_volume, d=0.45)
                manual_features['frac_diff_log_volume_norm'] = (fd_vol - fd_vol.rolling(100).mean()) / (fd_vol.rolling(100).std() + 1e-8)
            except Exception as e:
                self.logger.warning(f"Fractional differentiation failed for log_volume: {e}")
                manual_features['frac_diff_log_volume_norm'] = log_volume.diff().rolling(20).mean()
            
            volume_ma_long = volume.rolling(200).mean()
            manual_features['volume_impulse_ratio'] = (volume - volume.rolling(5).mean()) / (volume_ma_long + 1e-8)
            manual_features['order_flow_pressure'] = (returns * volume).rolling(10).sum() / (volume.rolling(10).sum() + 1e-8)
            
        return manual_features

    def _apply_manual_volume_force_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection."""
        if features.empty:
            return features
        
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
        
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
        
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
        
        return features

    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
