"""
Analyst Ensemble Training Step.

This step trains ensemble analyst models using outputs from base models,
regime probabilities, and feature generation outputs.
"""

import asyncio
import logging
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = logging.getLogger(__name__)


class AnalystEnsembleTrainingStep(BaseStep):
    """
    Analyst Ensemble Training Step.

    Trains ensemble analyst models using outputs from:
    - feature_generation_labeling_integration_step (features and labels)
    - regime_ensemble_training (regime probabilities)
    - analyst_base_training (base model predictions/confidence)
    - disagreement features (generated from base model outputs)
    """

    def __init__(self, step_name: str = "analyst_ensemble_training"):
        """Initialize the analyst ensemble training step."""
        super().__init__(step_name, use_versioned_artifacts=True)  # Enable HDF5 storage
        self.logger = system_logger.getChild('AnalystEnsembleTraining')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analyst ensemble model training.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')
                - execution_mode: 'full', 'light', or 'blank'

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        tprint(f"🧠 Starting analyst ensemble training for {config.get('symbol', 'UNKNOWN')}", "INFO")

        try:
            # Set context for artifact loading
            analyst_timeframe = config.get('timeframe', '15m')
            regime_timeframe = config.get('regime_timeframe', '1h')
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            direction = config.get('direction', 'long')

            tprint(f"📊 Configuration: {symbol}/{exchange} [{analyst_timeframe}] {direction}", "INFO")

            # Step 1: Load feature generation data (features and labels)
            tprint("📥 Step 1/4: Loading feature generation data from HDF5...", "INFO")
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=analyst_timeframe,
                direction=direction,
                model='analyst'
            )

            features_data = self._get_artifact(
                'feature_generation_labeling_integration',
                artifact_type='data',
                data_category='features'
            )

            if features_data is None:
                error_msg = (
                    f"❌ No feature generation data found in versioned artifacts!\n"
                    f"   Please run feature_generation_labeling_integration_step first:\n"
                    f"   python3 src/launcher/ares_launcher.py feature_generation_labeling_integration_step --symbol {symbol} --timeframe {analyst_timeframe}"
                )
                tprint(error_msg, "ERROR")
                return {'success': False, 'artifacts': {}, 'metrics': {}, 'error': error_msg}

            tprint(f"✅ Loaded features: {features_data.shape}", "SUCCESS")

            # Step 2: Load regime probabilities (prefer HMM Alpha regimes)
            tprint("📥 Step 2/4: Loading regime probabilities (preferring HMM Alpha)...", "INFO")

            regime_probs = None

            # 2a. Try HMM Alpha regime features first (preferred)
            try:
                # Try loading from ML Risk Regime step first (newer), then fall back to HMM Alpha step
                alpha_training = None
                source_type = None

                # Try ML Risk Regime first
                self.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=regime_timeframe,
                    direction=direction,
                    model='regime_risk'
                )

                alpha_training = self._get_artifact(
                    'ml_risk_training_data_1h',
                    artifact_type='data',
                    data_category='features'
                )

                if alpha_training is not None and not alpha_training.empty:
                    source_type = 'ml_risk'
                    tprint(f"✅ Retrieved ml_risk_training_data_1h for regimes: {alpha_training.shape}", "SUCCESS")
                else:
                    # Fall back to HMM Alpha
                    self.set_context(
                        symbol=symbol,
                        exchange=exchange,
                        timeframe=regime_timeframe,
                        direction=direction,
                        model='regime_alpha'
                    )

                    alpha_training = self._get_artifact(
                        'hmm_alpha_training_data_1h',
                        artifact_type='data',
                        data_category='features'
                    )

                    if alpha_training is not None and not alpha_training.empty:
                        source_type = 'hmm_alpha'
                        tprint(f"✅ Retrieved hmm_alpha_training_data_1h for regimes: {alpha_training.shape}", "SUCCESS")

                if alpha_training is not None:
                    if not isinstance(alpha_training, pd.DataFrame):
                        alpha_training = pd.DataFrame(alpha_training)

                    # Standardize to DatetimeIndex using timestamp column if present
                    if 'timestamp' in alpha_training.columns:
                        alpha_training = alpha_training.copy()
                        alpha_training['timestamp'] = pd.to_datetime(alpha_training['timestamp'])
                        alpha_training.set_index('timestamp', inplace=True)
                    elif not isinstance(alpha_training.index, pd.DatetimeIndex):
                        tprint(f"⚠️ {source_type} training data has no DatetimeIndex; skipping regime features", "WARNING")
                        alpha_training = None

                if alpha_training is not None and not alpha_training.empty:

                    # Select regime feature columns (supports both alpha and risk regimes)
                    expectation_cols = [
                        c for c in alpha_training.columns
                        if c.startswith('alpha_expectation_')
                    ]
                    risk_cols = [
                        c for c in alpha_training.columns
                        if (c.startswith('risk_regime') or c.startswith('risk_pred_') or c.startswith('risk_score'))
                    ]
                    if expectation_cols:
                        alpha_cols = expectation_cols + risk_cols
                    else:
                        alpha_cols = [
                            c for c in alpha_training.columns
                            if (c.startswith('alpha_regime_bucket_') or c.startswith('alpha_pred_') or
                                c.startswith('risk_regime') or c.startswith('risk_pred_') or
                                c.startswith('risk_score'))
                        ]

                    if alpha_cols:
                        alpha_features = alpha_training[alpha_cols].copy()
                        tprint(
                            f"   ↪ Selected {len(alpha_cols)} HMM Alpha regime/score columns: "
                            f"{alpha_cols[:5]}{'...' if len(alpha_cols) > 5 else ''}",
                            "INFO",
                        )

                        # Create an OOS-style proxy by shifting one step forward and ffill
                        tprint(
                            "⚠️ HMM Alpha regime features are in-sample; "
                            "creating OOS proxy via 1-step shift (ffill)",
                            "WARNING",
                        )
                        alpha_features = alpha_features.shift(1).fillna(method='ffill')

                        # Align to analyst feature index (typically 15m)
                        if not alpha_features.index.equals(features_data.index):
                            tprint(
                                "   Aligning HMM Alpha regime features to analyst timeframe via reindex+ffill",
                                "INFO",
                            )
                            alpha_features = alpha_features.reindex(features_data.index, method='ffill')

                        regime_probs = alpha_features
                        tprint(f"✅ Using HMM Alpha regime probabilities: {regime_probs.shape}", "SUCCESS")
                    else:
                        tprint(
                            "⚠️ HMM Alpha training data contains no alpha_regime_bucket_* or "
                            "alpha_pred_* columns; skipping alpha regime features",
                            "WARNING",
                        )

            except Exception as e:
                tprint(
                    f"⚠️ Failed to load HMM Alpha regime features, will try legacy regime artifacts: {e}",
                    "WARNING",
                )

            # 2b. Legacy regime ensemble / rolling HMM fallback if Alpha not available
            if regime_probs is None:
                self.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=regime_timeframe,
                    direction=direction,
                    model='regime'
                )

                regime_probs = self._get_artifact(
                    'regime_ensemble_predictions',
                    artifact_type='data',
                    data_category='predictions'
                )

                if regime_probs is None:
                    tprint(
                        "⚠️ No regime probabilities from HMM Alpha or regime_ensemble_predictions; "
                        "trying rolling_hmm_regime_probabilities...",
                        "WARNING",
                    )
                    regime_probs = self._get_artifact(
                        'rolling_hmm_regime_probabilities',
                        artifact_type='data',
                        data_category='features'
                    )

                if regime_probs is not None:
                    tprint(f"✅ Loaded regime probabilities from legacy artifacts: {regime_probs.shape}", "SUCCESS")
                else:
                    tprint(
                        "⚠️ No regime probabilities found from HMM Alpha or legacy artifacts; "
                        "will continue without regime features",
                        "WARNING",
                    )

            # Step 3: Load analyst base model outputs
            tprint("📥 Step 3/4: Loading analyst base model outputs from HDF5...", "INFO")
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=analyst_timeframe,
                direction=direction,
                model='analyst'
            )

            # Prefer OOF base predictions if available
            base_predictions = self._get_artifact(
                'analyst_base_predictions_oof',
                artifact_type='data',
                data_category='predictions'
            )
            used_oof = base_predictions is not None
            tprint(f"   Base predictions source: OOF={used_oof}", "INFO")
            if not used_oof:
                error_msg = (
                    "❌ OOF base predictions not found for analyst ensemble stacking.\n"
                    "   Please rerun analyst_base_training to generate 'analyst_base_predictions_oof'."
                )
                tprint(error_msg, "ERROR")
                return {'success': False, 'artifacts': {}, 'metrics': {}, 'error': error_msg}

            base_confidence = self._get_artifact(
                'analyst_base_confidence',
                artifact_type='data',
                data_category='predictions'
            )

            if base_predictions is None:
                error_msg = (
                    f"❌ No analyst base model outputs found in versioned artifacts!\n"
                    f"   Please run analyst_base_training first:\n"
                    f"   python3 src/launcher/ares_launcher.py --train-analyst-base --symbol {symbol} --timeframe {analyst_timeframe} --direction {direction}"
                )
                tprint(error_msg, "ERROR")
                return {'success': False, 'artifacts': {}, 'metrics': {}, 'error': error_msg}

            tprint(f"✅ Loaded base predictions: {base_predictions.shape}", "SUCCESS")
            if base_confidence is not None:
                tprint(f"✅ Loaded base confidence: {base_confidence.shape}", "SUCCESS")

            # Step 4: Generate disagreement features
            tprint("🔧 Step 4/4: Generating disagreement features from base model outputs...", "INFO")
            disagreement_features = self._generate_disagreement_features(base_predictions)
            tprint(f"✅ Generated disagreement features: {disagreement_features.shape}", "SUCCESS")

            # Log disagreement feature generation with comprehensive preview
            from src.utils.tprint import tprint_data_preview
            tprint("=" * 80, "INFO")
            tprint("🎲 DISAGREEMENT FEATURES: Generated from Base Model Outputs", "INFO")
            tprint("=" * 80, "INFO")
            tprint_data_preview(
                disagreement_features,
                name="Disagreement Features",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint("=" * 80, "INFO")

            # Combine all features
            tprint("🔗 Combining all features for ensemble training...", "INFO")

            # Track feature counts before combination
            original_feature_count = features_data.shape[1]
            regime_feature_count = regime_probs.shape[1] if regime_probs is not None else 0
            base_pred_count = base_predictions.shape[1]
            base_conf_count = base_confidence.shape[1] if base_confidence is not None else 0
            disagreement_count = disagreement_features.shape[1] if not disagreement_features.empty else 0

            ensemble_features = self._combine_features(
                features_data,
                regime_probs,
                base_predictions,
                base_confidence,
                disagreement_features,
                config
            )
            tprint(f"✅ Combined features shape: {ensemble_features.shape}", "SUCCESS")

            # Log feature combination with comprehensive tracking
            tprint("=" * 80, "INFO")
            tprint("🔗 FEATURE COMBINATION: Merging All Feature Sets", "INFO")
            tprint("=" * 80, "INFO")
            execution_mode = config.get('execution_mode', 'full')
            tprint(f"   Execution Mode: {execution_mode.upper()}", "INFO")

            if execution_mode == 'blank':
                # In blank mode, we intentionally use regime features along with
                # base predictions and disagreement features (no raw FG features).
                label_count = len([col for col in ensemble_features.columns
                                 if ('label' in col.lower() or 'target' in col.lower())
                                 and 'regime' not in col.lower()])
                base_pred_actual = len([col for col in ensemble_features.columns
                                      if any(term in col.lower() for term in ['_prob', '_pred', '_confidence'])])
                disagreement_actual = len([col for col in ensemble_features.columns
                                          if 'disagreement' in col.lower()])
                regime_in_final = len([col for col in ensemble_features.columns
                                     if 'regime' in col.lower()])

                tprint(f"   Label columns: {label_count}", "INFO")
                tprint(f"   Base predictions: {base_pred_actual}", "INFO")
                tprint(f"   Disagreement features: {disagreement_actual}", "INFO")
                tprint(f"   Regime features: {regime_in_final}", "INFO")
                tprint(f"   Total combined: {ensemble_features.shape[1]}", "INFO")
            else:
                # Full/light mode - show original breakdown
                tprint(f"   Original features: {original_feature_count}", "INFO")
                tprint(f"   Regime features: {regime_feature_count}", "INFO")
                tprint(f"   Base predictions: {base_pred_count}", "INFO")
                tprint(f"   Base confidence: {base_conf_count}", "INFO")
                tprint(f"   Disagreement features: {disagreement_count}", "INFO")
                tprint(f"   Total combined: {ensemble_features.shape[1]}", "INFO")

            tprint_data_preview(
                ensemble_features,
                name="Combined Ensemble Features",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint("=" * 80, "INFO")

            # Train ensemble model
            tprint("🏋️ Training analyst ensemble model...", "INFO")
            ensemble_result = await self._train_ensemble_model(
                ensemble_features,
                features_data,
                config
            )

            if not ensemble_result['success']:
                return ensemble_result

            # Save metrics to .md and JSON
            tprint("💾 Saving metrics to .md and JSON...", "INFO")
            metrics_saved = self._save_metrics(
                ensemble_result['metrics'],
                symbol,
                analyst_timeframe,
                direction
            )

            # Verify model is saved in Pickle format
            tprint("✅ Verifying model saved in Pickle format...", "INFO")
            model_path = ensemble_result.get('model_path')

            # Fallback: derive model path from artifacts if not explicitly provided
            if not model_path:
                artifacts = ensemble_result.get('artifacts', {})
                if isinstance(artifacts, dict):
                    candidate = artifacts.get('analyst_ensemble_model') or artifacts.get('ensemble_model')
                    if candidate:
                        model_path = candidate

            if model_path and Path(model_path).exists():
                tprint(f"✅ Model saved at: {model_path}", "SUCCESS")
            else:
                artifacts = ensemble_result.get('artifacts', {})
                if isinstance(artifacts, dict) and artifacts:
                    tprint(f"⚠️ Model path not found; available artifacts: {list(artifacts.keys())}", "WARNING")
                else:
                    tprint("⚠️ Model path not found in result", "WARNING")

            return {
                'success': True,
                'artifacts': ensemble_result.get('artifacts', {}),
                'metrics': ensemble_result.get('metrics', {}),
                'model_path': model_path,
                'metrics_files': metrics_saved
            }

        except Exception as e:
            error_msg = f"Analyst ensemble training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }

    def _generate_disagreement_features(self, base_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Generate disagreement features from base model predictions.

        Args:
            base_predictions: DataFrame with base model predictions

        Returns:
            DataFrame with disagreement features
        """
        try:
            from src.training.steps.market_analysis.components.ensemble_meta_features import (
                EnsembleMetaFeaturesGenerator
            )

            # Extract model columns (assuming format: model_name_prob or model_name_pred)
            model_cols = [col for col in base_predictions.columns if any(
                suffix in col for suffix in ['_prob', '_pred', '_confidence']
            )]

            if not model_cols:
                tprint("⚠️ No model prediction columns found, using all columns", "WARNING")
                model_cols = list(base_predictions.columns)

            # Group by model name
            model_predictions = {}
            for col in model_cols:
                # Extract model name (before _prob, _pred, etc.)
                model_name = col.split('_prob')[0].split('_pred')[0].split('_confidence')[0]
                if model_name not in model_predictions:
                    model_predictions[model_name] = []
                model_predictions[model_name].append(col)

            tprint(f"📊 Found {len(model_predictions)} base models: {list(model_predictions.keys())}", "INFO")

            # Create pseudo-models for meta-feature generation
            # Since we only have predictions, we'll compute disagreement metrics directly
            predictions_array = base_predictions[model_cols].values

            # Log base predictions before computing disagreement
            from src.utils.tprint import tprint_data_preview
            tprint("=" * 80, "INFO")
            tprint("📊 COMPUTING DISAGREEMENT: Base Model Predictions", "INFO")
            tprint("=" * 80, "INFO")
            tprint_data_preview(
                base_predictions[model_cols],
                name="Base Model Predictions",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint(f"📊 Using {len(model_cols)} model prediction columns from {len(model_predictions)} models", "INFO")
            tprint("=" * 80, "INFO")

            # Compute disagreement metrics
            disagreement_df = pd.DataFrame(index=base_predictions.index)

            # Variance across models
            disagreement_df['disagreement_variance'] = np.var(predictions_array, axis=1)

            # Standard deviation
            disagreement_df['disagreement_std'] = np.std(predictions_array, axis=1)

            # Range (max - min)
            disagreement_df['disagreement_range'] = np.max(predictions_array, axis=1) - np.min(predictions_array, axis=1)

            # Mean absolute deviation from mean
            mean_pred = np.mean(predictions_array, axis=1, keepdims=True)
            disagreement_df['disagreement_mad'] = np.mean(np.abs(predictions_array - mean_pred), axis=1)

            # Coefficient of variation
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = disagreement_df['disagreement_std'] / mean_pred.flatten()
                disagreement_df['disagreement_cv'] = np.where(np.isfinite(cv), cv, 0)

            tprint(f"✅ Generated {len(disagreement_df.columns)} disagreement features", "SUCCESS")

            # Log computed disagreement features
            tprint("=" * 80, "INFO")
            tprint("📊 DISAGREEMENT FEATURES COMPUTED: Statistical Metrics", "INFO")
            tprint("=" * 80, "INFO")
            tprint_data_preview(
                disagreement_df,
                name="Disagreement Features",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint(f"📊 Disagreement statistics:", "INFO")
            tprint(f"   Avg variance: {disagreement_df['disagreement_variance'].mean():.6f}", "INFO")
            tprint(f"   Avg std: {disagreement_df['disagreement_std'].mean():.6f}", "INFO")
            tprint(f"   Avg range: {disagreement_df['disagreement_range'].mean():.6f}", "INFO")
            tprint("=" * 80, "INFO")

            return disagreement_df

        except Exception as e:
            tprint(f"⚠️ Failed to generate disagreement features: {e}, returning empty DataFrame", "WARNING")
            return pd.DataFrame(index=base_predictions.index)

    def _combine_features(
        self,
        features_data: pd.DataFrame,
        regime_probs: Optional[pd.DataFrame],
        base_predictions: pd.DataFrame,
        base_confidence: Optional[pd.DataFrame],
        disagreement_features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Combine all features for ensemble training."""
        execution_mode = config.get('execution_mode', 'full')
        
        if execution_mode == 'blank':
            # In blank mode, avoid feeding raw feature-generation columns directly,
            # but still allow regime probabilities, base predictions, and disagreement
            # features to be used as inputs to the ensemble.
            tprint("🎯 BLANK MODE: Using regime probabilities, base predictions and disagreement features (no raw FG features)", "INFO")
            combined = pd.DataFrame(index=features_data.index)
        else:
            combined = features_data.copy()
            leak_cols = [col for col in combined.columns if ('label' in col.lower() or 'target' in col.lower())]
            if leak_cols:
                tprint(f"⚠️ Removing {len(leak_cols)} target/label columns from features", "WARNING")
                combined = combined.drop(columns=leak_cols)

        # Align all dataframes to the same index
        common_index = combined.index if not combined.empty else features_data.index

        # Add regime probabilities if available (for all execution modes)
        if regime_probs is not None:
            # Ensure regime_probs has datetime index if common_index is datetime
            if hasattr(common_index, 'dtype') and 'datetime' in str(common_index.dtype):
                if not hasattr(regime_probs.index, 'dtype') or 'datetime' not in str(regime_probs.index.dtype):
                    tprint("🔄 Converting regime probabilities index to datetime...", "INFO")
                    # If regime_probs has int index, use common_index directly
                    if len(regime_probs) == len(common_index):
                        regime_probs.index = common_index
                    else:
                        tprint(f"⚠️ Regime probs length ({len(regime_probs)}) != features length ({len(common_index)}), using reindex+ffill alignment (possible leakage if labels/features lookahead-dependent)", "WARNING")
                        # Create a mapping - this is a fallback
                        regime_probs = regime_probs.reindex(range(len(common_index)), method='ffill')
                        regime_probs.index = common_index
            
            # Align regime probs to analyst timeframe if needed
            if not regime_probs.index.equals(common_index):
                tprint("🔄 Aligning regime probabilities to analyst timeframe via reindex+ffill", "INFO")
                regime_probs_aligned = regime_probs.reindex(common_index, method='ffill')
                combined = combined.join(regime_probs_aligned, how='left', rsuffix='_regime')
            else:
                combined = combined.join(regime_probs, how='left', rsuffix='_regime')

        # Add base predictions (remove duplicates first)
        if base_predictions.index.duplicated().any():
            self.logger.warning(f"⚠️ Removing {base_predictions.index.duplicated().sum()} duplicate indices from base_predictions")
            base_predictions = base_predictions[~base_predictions.index.duplicated(keep='first')]
        
        if not base_predictions.index.equals(common_index):
            tprint("   Aligning base_predictions to common index via reindex (no ffill)", "INFO")
            base_predictions = base_predictions.reindex(common_index)
            tprint(f"   Base predictions alignment: {base_predictions.isna().sum().sum()} nulls", "INFO")
            if base_predictions.isna().any().any():
                tprint("⚠️ Base predictions leakage detected", "WARNING")
            else:
                tprint("✅ No base predictions leakage detected", "SUCCESS")
        combined = combined.join(base_predictions, how='left', rsuffix='_base')

        # Add base confidence if available (remove duplicates first)
        if base_confidence is not None:
            if base_confidence.index.duplicated().any():
                self.logger.warning(f"⚠️ Removing {base_confidence.index.duplicated().sum()} duplicate indices from base_confidence")
                base_confidence = base_confidence[~base_confidence.index.duplicated(keep='first')]
            
            if not base_confidence.index.equals(common_index):
                tprint("   Aligning base_confidence to common index via reindex (no ffill)", "INFO")
                base_confidence = base_confidence.reindex(common_index)
            combined = combined.join(base_confidence, how='left', rsuffix='_conf')

        # Add disagreement features (remove duplicates first)
        if not disagreement_features.empty:
            if disagreement_features.index.duplicated().any():
                self.logger.warning(f"⚠️ Removing {disagreement_features.index.duplicated().sum()} duplicate indices from disagreement_features")
                disagreement_features = disagreement_features[~disagreement_features.index.duplicated(keep='first')]
            
            if not disagreement_features.index.equals(common_index):
                tprint("   Aligning disagreement_features to common index via reindex (no ffill)", "INFO")
                disagreement_features = disagreement_features.reindex(common_index)
            combined = combined.join(disagreement_features, how='left')

        # Final NA handling (may hide alignment issues if excessive)
        pre_rows = len(combined)
        na_before = int(combined.isna().sum().sum())
        combined = combined.fillna(method='ffill')
        combined = combined.dropna()
        na_after = int(combined.isna().sum().sum())
        tprint(f"   NA handling: before={na_before} nulls, after={na_after} nulls; rows before={pre_rows}, after={len(combined)}", "INFO")

        return combined

    async def _train_ensemble_model(
        self,
        ensemble_features: pd.DataFrame,
        target_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train the ensemble model and save it in Pickle format."""
        try:
            # Import and call unified training step for actual model training
            from .unified_models_training_step import UnifiedModelsTrainingStep

            # Set training type for unified step
            config['training_type'] = 'analyst_ensemble'
            config['execution_context'] = 'analyst'
            config['ensemble_features'] = ensemble_features
            config['target_data'] = target_data

            # Create and execute unified training step
            unified_step = UnifiedModelsTrainingStep()
            result = await unified_step.execute(config)

            return result

        except Exception as e:
            error_msg = f"Ensemble model training failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            return {'success': False, 'error': error_msg}

    def _save_metrics(
        self,
        metrics: Dict[str, Any],
        symbol: str,
        timeframe: str,
        direction: str
    ) -> Dict[str, str]:
        """
        Save metrics to .md, JSON, and CSV formats with calibration metrics.

        Args:
            metrics: Metrics dictionary
            symbol: Trading symbol
            timeframe: Timeframe
            direction: Trading direction

        Returns:
            Dictionary with paths to saved files
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"analyst_ensemble_metrics_{symbol}_{timeframe}_{direction}_{timestamp}"

            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(parents=True, exist_ok=True)

            # Save as Markdown
            md_path = outcomes_dir / f"{base_filename}.md"
            with open(md_path, 'w') as f:
                f.write(f"# Analyst Ensemble Training Metrics\n\n")
                f.write(f"**Symbol**: {symbol}\n")
                f.write(f"**Timeframe**: {timeframe}\n")
                f.write(f"**Direction**: {direction}\n")
                f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")

                f.write("## Performance Metrics\n\n")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- **{key}**: {value:.6f}\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")

                # Add calibration metrics section if available
                if 'calibration' in metrics:
                    f.write("\n## Calibration Metrics\n\n")
                    calibration = metrics['calibration']

                    # Brier Score Decomposition
                    if 'brier_score_decomposition' in calibration:
                        brier = calibration['brier_score_decomposition']
                        if 'error' not in brier:
                            f.write("### Brier Score Decomposition\n\n")
                            f.write(f"- **Brier Score**: {brier.get('brier_score', 0):.4f}\n")
                            f.write(f"- **Reliability (Calibration Error)**: {brier.get('reliability', 0):.4f}\n")
                            f.write(f"- **Resolution (Discrimination)**: {brier.get('resolution', 0):.4f}\n")
                            f.write(f"- **Uncertainty**: {brier.get('uncertainty', 0):.4f}\n")
                            f.write(f"- **Quality**: {brier.get('quality_assessment', 'N/A')}\n")
                            f.write(f"- **Interpretation**: {brier.get('interpretation', 'N/A')}\n\n")

                    # Rolling Calibration Error
                    if 'rolling_calibration_error' in calibration:
                        rolling = calibration['rolling_calibration_error']
                        if 'error' not in rolling:
                            f.write("### Rolling Calibration Error\n\n")
                            f.write(f"- **Overall Mean ECE**: {rolling.get('overall_mean_ece', 0):.4f}\n")
                            f.write(f"- **ECE Trend**: {rolling.get('overall_ece_trend', 0):.6f}\n")
                            f.write(f"- **Degradation Status**: {rolling.get('degradation_status', 'N/A')}\n")
                            f.write(f"- **Interpretation**: {rolling.get('interpretation', 'N/A')}\n\n")

                    # Threshold-Weighted ECE
                    if 'threshold_weighted_ece' in calibration:
                        tw_ece = calibration['threshold_weighted_ece']
                        if 'error' not in tw_ece:
                            f.write("### Threshold-Weighted ECE (High-Confidence Focus)\n\n")
                            f.write(f"- **Standard ECE**: {tw_ece.get('standard_ece', 0):.4f}\n")
                            f.write(f"- **Threshold-Weighted ECE**: {tw_ece.get('threshold_weighted_ece', 0):.4f}\n")
                            f.write(f"- **High-Confidence MAE**: {tw_ece.get('high_confidence_mae', 0):.4f}\n")
                            f.write(f"- **High-Confidence %**: {tw_ece.get('high_confidence_pct', 0):.1f}%\n")
                            f.write(f"- **Quality**: {tw_ece.get('quality_assessment', 'N/A')}\n")
                            f.write(f"- **Interpretation**: {tw_ece.get('interpretation', 'N/A')}\n\n")

                    # Standard calibration metrics
                    if 'calibration_curve' in calibration:
                        curve = calibration['calibration_curve']
                        if 'error' not in curve:
                            f.write("### Calibration Curve\n\n")
                            f.write(f"- **Expected Calibration Error (ECE)**: {curve.get('expected_calibration_error', 0):.4f}\n")
                            f.write(f"- **Max Calibration Error (MCE)**: {curve.get('max_calibration_error', 0):.4f}\n")
                            f.write(f"- **Avg Calibration Error (ACE)**: {curve.get('avg_calibration_error', 0):.4f}\n")
                            f.write(f"- **RMSCE**: {curve.get('rmsce', 0):.4f}\n")
                            f.write(f"- **Quality**: {curve.get('calibration_quality', 'N/A')}\n\n")

                    # Conditional Calibration Analysis
                    if 'conditional_calibration' in calibration:
                        cond_cal = calibration['conditional_calibration']
                        if 'error' not in cond_cal:
                            f.write("### Conditional Calibration (Market Regime Adaptation)\n\n")
                            f.write(f"- **Interpretation**: {cond_cal.get('interpretation', 'N/A')}\n\n")

                            # Top offender features
                            top_offenders = cond_cal.get('top_offender_features', [])
                            if top_offenders:
                                f.write("#### Top Offender Features (Affecting Calibration)\n\n")
                                for i, feat in enumerate(top_offenders, 1):
                                    f.write(f"{i}. **{feat}**\n")
                                f.write("\n")

                            # Decile Analysis
                            if 'decile_analysis' in cond_cal:
                                f.write("#### Decile Binning Analysis\n\n")
                                f.write("| Feature | Min Brier | Max Brier | Brier Range | Worst Deciles |\n")
                                f.write("|---------|-----------|-----------|-------------|---------------|\n")

                                for feat_name, analysis in cond_cal['decile_analysis'].items():
                                    if 'error' not in analysis:
                                        min_b = analysis.get('min_brier', 0)
                                        max_b = analysis.get('max_brier', 0)
                                        range_b = analysis.get('brier_range', 0)
                                        worst = analysis.get('worst_deciles', [])
                                        worst_str = ', '.join([f"D{w['decile']+1}" for w in worst[:2]])

                                        f.write(f"| {feat_name} | {min_b:.4f} | {max_b:.4f} | {range_b:.4f} | {worst_str} |\n")
                                f.write("\n")

                            # Lasso Conditional Fix
                            if 'lasso_conditional_fix' in cond_cal:
                                lasso = cond_cal['lasso_conditional_fix']
                                if 'error' not in lasso:
                                    f.write("#### Lasso Conditional Fix (Before vs After)\n\n")

                                    metrics_l = lasso.get('metrics', {})
                                    f.write("**Performance Comparison:**\n\n")
                                    f.write(f"- **Brier Score**: {metrics_l.get('raw_brier', 0):.4f} → {metrics_l.get('calibrated_brier', 0):.4f} "
                                           f"({metrics_l.get('brier_improvement', 0)*100:.1f}% change)\n")
                                    f.write(f"- **Resolution (Sharpness)**: {metrics_l.get('raw_resolution', 0):.4f} → {metrics_l.get('calibrated_resolution', 0):.4f} "
                                           f"({metrics_l.get('resolution_change', 0)*100:.1f}% change)\n")
                                    f.write(f"- **ROC-AUC**: {metrics_l.get('raw_auc', 0):.4f} → {metrics_l.get('calibrated_auc', 0):.4f} "
                                           f"({metrics_l.get('auc_change', 0)*100:.1f}% change)\n")
                                    f.write(f"- **Regularization Strength (C)**: {lasso.get('best_C', 0):.4f}\n")
                                    f.write(f"- **Status**: {lasso.get('status', 'unknown').upper()}\n\n")

                                    # Coefficients (survivors)
                                    coeffs = lasso.get('coefficients', {})
                                    if coeffs:
                                        f.write("**Lasso Coefficients (Survivors):**\n\n")
                                        for feat, coeff in sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True):
                                            direction = "reduces" if coeff < 0 else "increases"
                                            f.write(f"- **{feat}**: {coeff:.4f} ({direction} confidence)\n")
                                        f.write("\n")

                                    # Warnings
                                    warnings_l = lasso.get('warnings', [])
                                    if warnings_l:
                                        f.write("**⚠️ Warnings:**\n\n")
                                        for warning in warnings_l:
                                            f.write(f"- {warning}\n")
                                        f.write("\n")

                f.write("\n## Detailed Metrics\n\n")
                f.write("```json\n")
                f.write(json.dumps(metrics, indent=2, default=str))
                f.write("\n```\n")

            tprint(f"✅ Saved metrics to: {md_path}", "SUCCESS")

            # Save as JSON
            json_path = outcomes_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics
                }, f, indent=2, default=str)

            tprint(f"✅ Saved metrics to: {json_path}", "SUCCESS")

            # Save as CSV (flattened metrics)
            csv_path = outcomes_dir / f"{base_filename}.csv"
            try:
                import csv

                # Flatten metrics dictionary
                flattened = self._flatten_metrics(metrics, prefix='')

                # Add metadata
                flattened_with_meta = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    **flattened
                }

                # Write CSV
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=flattened_with_meta.keys())
                    writer.writeheader()
                    writer.writerow(flattened_with_meta)

                tprint(f"✅ Saved metrics to: {csv_path}", "SUCCESS")
            except Exception as csv_error:
                tprint(f"⚠️ Failed to save CSV metrics: {csv_error}", "WARNING")
                csv_path = None

            return {
                'markdown': str(md_path),
                'json': str(json_path),
                'csv': str(csv_path) if csv_path else None
            }

        except Exception as e:
            tprint(f"⚠️ Failed to save metrics: {e}", "WARNING")
            return {}

    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = '', separator: str = '.') -> Dict[str, Any]:
        """
        Flatten nested metrics dictionary for CSV export.

        Args:
            metrics: Nested metrics dictionary
            prefix: Prefix for nested keys
            separator: Separator for nested keys

        Returns:
            Flattened dictionary with string keys and numeric/string values
        """
        flattened = {}

        for key, value in metrics.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self._flatten_metrics(value, new_key, separator))
            elif isinstance(value, (list, tuple)):
                # For lists, only include if they're simple numeric lists
                if value and all(isinstance(v, (int, float)) for v in value):
                    # Store list stats instead of full list
                    flattened[f"{new_key}_mean"] = np.mean(value) if value else 0
                    flattened[f"{new_key}_std"] = np.std(value) if len(value) > 1 else 0
                    flattened[f"{new_key}_min"] = np.min(value) if value else 0
                    flattened[f"{new_key}_max"] = np.max(value) if value else 0
                # Skip complex lists for CSV
            elif isinstance(value, (int, float, str, bool)) or value is None:
                # Store simple values directly
                flattened[new_key] = value
            else:
                # Convert other types to string
                flattened[new_key] = str(value)

        return flattened

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_analyst_ensemble_training_step():
    """Register the analyst ensemble training step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("analyst_ensemble_training", AnalystEnsembleTrainingStep)
    tprint("✅ Analyst ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_analyst_ensemble_training_step()
