# src/training/steps/step7_analyst_ensemble_creation.py

import json
import os
from typing import Any

import joblib
import pandas as pd

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger

logger, system_logger


class AnalystEnsembleCreationStep:
    """Step 7: Analyst Ensemble Creation - Combines multiple models into ensemble predictions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config, config
        self.logger, logger
        self.ensemble_models = {}
        self.ensemble_weights = {}

    @handle_errors
    def execute(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        training_input: dict[str, Any],
    ) -> bool:
        """Execute Step 7: Create analyst ensemble models.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory
            training_input: Training input data

        Returns:
            bool: True if successful

        """
        logger.info("🚀 Starting Step 7: Analyst Ensemble Creation")

        try:
        # Check if enhanced HMM models exist from Step 6
            enhanced_models_dir, os.path.join(data_dir, "enhanced_hmm_models")
        if not os.path.exists(enhanced_models_dir):
                logger.warning(
                    f"⚠️ Enhanced HMM models directory not found: {enhanced_models_dir}",
                )
                logger.info("📝 Creating placeholder ensemble for Step 7")
        return self._create_placeholder_ensemble(
                    symbol, exchange, data_dir, training_input,
                )

        # Load enhanced models from Step 6
            ensemble_models, self._load_enhanced_models(enhanced_models_dir)

        if not ensemble_models:
                logger.warning(
                    "⚠️ No enhanced models found, creating placeholder ensemble",
                )
        return self._create_placeholder_ensemble(
                    symbol, exchange, data_dir, training_input,
                )

        # Create ensemble
            ensemble_result, self._create_ensemble(
                ensemble_models, symbol, exchange, data_dir,
            )

        # Save ensemble summary
        self._save_ensemble_summary(ensemble_result, symbol, exchange, data_dir)

            logger.info("✅ Step 7: Analyst Ensemble Creation completed successfully")
        return True

        except Exception as e:
            logger.exception(f"❌ Error in Step 7: {e}")
        return False

    def _load_enhanced_models(self, enhanced_models_dir: str) -> dict[str, Any]:
        """Load enhanced models from Step 6."""
        try:
            ensemble_models = {}

        if not os.path.exists(enhanced_models_dir):
        return ensemble_models

        # Look for model files in the enhanced models directory
        for regime_dir in os.listdir(enhanced_models_dir):
                regime_path, os.path.join(enhanced_models_dir, regime_dir)
        if os.path.isdir(regime_path):
                    ensemble_models[regime_dir] = {}

        for model_file in os.listdir(regime_path):
        if model_file.endswith(".joblib"):
                            model_path, os.path.join(regime_path, model_file)
        try:
                                model, joblib.load(model_path)
                                model_name, model_file.replace(".joblib", "")
                                ensemble_models[regime_dir][model_name] = model
                                logger.info(
                                    f"📦 Loaded model: {regime_dir}/{model_name}",
                                )
        except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to load model {model_path}: {e}",
                                )

        return ensemble_models

        except Exception as e:
            logger.exception(f"❌ Error loading enhanced models: {e}")
        return {}

    def _create_ensemble(
        self, ensemble_models: dict[str, Any], symbol: str, exchange: str, data_dir: str,
    ) -> dict[str, Any]:
        """Create ensemble from loaded models."""
        try:
        # Apply optimized feature selection for ensemble creation
        try:
                from src.training.optimized_feature_selection_manager import (
                    OptimizedFeatureSelectionManager,
                )
                optimized_feature_selection, OptimizedFeatureSelectionManager(self.config)

        # Get sample data for feature selection (if available)
                sample_data, self._get_sample_data_for_feature_selection(data_dir, symbol, exchange)
        if sample_data is not None:
                    features_df, target, sample_data

                    optimized_features, selection_metadata, optimized_feature_selection.select_features_optimized(
                        features_df, target, model_type="ensemble_models", step_name="step7_ensemble",
                    )

                    logger.info(f"✅ Applied optimized feature selection for ensemble: {features_df.shape[1]} -> {optimized_features.shape[1]} features")

        # Log performance metrics
        if "performance_metrics" in selection_metadata:
                        perf_metrics, selection_metadata["performance_metrics"]
                        logger.info("📊 Ensemble feature selection performance:")
                        logger.info(f"   - VIF calculation: {perf_metrics.get('vif_calculation_time', 0):.2f}s")
                        logger.info(f"   - SHAP analysis: {perf_metrics.get('shap_calculation_time', 0):.2f}s")
                        logger.info(f"   - Total time: {selection_metadata.get('total_time', 0):.2f}s")

        # Store selection metadata
                    ensemble_result = {
                        "ensemble_models": ensemble_models,
                        "ensemble_weights": {},
                        "ensemble_metadata": {
                            "symbol": symbol,
                            "exchange": exchange,
                            "created_at": pd.Timestamp.now().isoformat(),
                            "model_count": sum(
                                len(models) for models in ensemble_models.values()
                            ),
                            "feature_selection_metadata": selection_metadata,
                        },
                    }
                else:
                    ensemble_result = {
                        "ensemble_models": ensemble_models,
                        "ensemble_weights": {},
                        "ensemble_metadata": {
                            "symbol": symbol,
                            "exchange": exchange,
                            "created_at": pd.Timestamp.now().isoformat(),
                            "model_count": sum(
                                len(models) for models in ensemble_models.values()
                            ),
                        },
                    }

        except Exception as e:
                logger.warning(f"⚠️ Optimized feature selection failed: {e}")
                ensemble_result = {
                    "ensemble_models": ensemble_models,
                    "ensemble_weights": {},
                    "ensemble_metadata": {
                        "symbol": symbol,
                        "exchange": exchange,
                        "created_at": pd.Timestamp.now().isoformat(),
                        "model_count": sum(
                            len(models) for models in ensemble_models.values()
                        ),
                    },
                }

        # Assign equal weights to all models for now
        for regime, models in ensemble_models.items():
                ensemble_result["ensemble_weights"][regime] = {
                    model_name: 1.0 / len(models) for model_name in models
                }

            logger.info(
                f"🎯 Created ensemble with {ensemble_result['ensemble_metadata']['model_count']} models",
            )
        return ensemble_result

        except Exception as e:
            logger.exception(f"❌ Error creating ensemble: {e}")
        return {}

    def _get_sample_data_for_feature_selection(self, data_dir: str, symbol: str, exchange: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Get sample data for feature selection from existing features."""
        try:
        # Try to load sample features and labels from Step 2 artifacts
            features_file, f"{data_dir}/{exchange}_{symbol}_features_train.parquet"
            labels_file, f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet"

        if os.path.exists(features_file) and os.path.exists(labels_file):
                features_df, pd.read_parquet(features_file)
                labels_df, pd.read_parquet(labels_file)

        # Align and extract target series
        # This assumes 'target' is the target column and they share an index (e.g., timestamp)
        if "target" in labels_df.columns:
        # Ensure indices are aligned before extracting the target
        if not features_df.index.equals(labels_df.index):
        if "timestamp" in labels_df.columns:
                            labels_df, labels_df.set_index("timestamp")
                        labels_df, labels_df.reindex(features_df.index)

                    target, labels_df["target"].dropna()
                    features_df, features_df.loc[target.index]  # Ensure features and target align after dropping NaNs
        return features_df, target
                logger.warning(f"⚠️ Target 'target' column not found in {labels_file}")

        return None

        except Exception as e:
            logger.warning(f"⚠️ Failed to get sample data for feature selection: {e}")
        return None

    def _create_placeholder_ensemble(
        self, symbol: str, exchange: str, data_dir: str, training_input: dict[str, Any],
    ) -> bool:
        """Create a placeholder ensemble when no enhanced models are available."""
        try:
            logger.info("📝 Creating placeholder ensemble for Step 7")

        # Create placeholder ensemble structure
            placeholder_ensemble = {
                "ensemble_models": {"placeholder_regime": {"placeholder_model": None}},
                "ensemble_weights": {"placeholder_regime": {"placeholder_model": 1.0}},
                "ensemble_metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "created_at": pd.Timestamp.now().isoformat(),
                    "model_count": 1,
                    "is_placeholder": True,
                },
            }

        # Save placeholder ensemble
        self._save_ensemble_summary(
                placeholder_ensemble, symbol, exchange, data_dir,
            )

            logger.info("✅ Placeholder ensemble created successfully")
        return True

        except Exception as e:
            logger.exception(f"❌ Error creating placeholder ensemble: {e}")
        return False

    def _save_ensemble_summary(
        self, ensemble_result: dict[str, Any], symbol: str, exchange: str, data_dir: str,
    ) -> None:
        """Save ensemble summary to file."""
        try:
        # Create ensemble directory
            ensemble_dir, os.path.join(data_dir, "analyst_ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)

        # Save ensemble summary
            summary_file, os.path.join(
                ensemble_dir, f"{exchange}_{symbol}_analyst_ensemble_summary.json",
            )

        # Convert to serializable format
            serializable_result, ensemble_result.copy()
            serializable_result["ensemble_models"] = {
                regime: list(models.keys())
        for regime, models in ensemble_result["ensemble_models"].items()
            }

        with open(summary_file, "w") as f:
                json.dump(serializable_result, f, indent=2, default=str)

            logger.info(f"💾 Saved ensemble summary to {summary_file}")

        except Exception as e:
            logger.exception(f"❌ Error saving ensemble summary: {e}")


def step7_analyst_ensemble_creation(
    symbol: str,
    exchange: str,
    data_dir: str,
    training_input: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Step 7: Analyst Ensemble Creation.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if successful

    """
    step, AnalystEnsembleCreationStep(config)
    return step.execute(symbol, exchange, data_dir, training_input)
