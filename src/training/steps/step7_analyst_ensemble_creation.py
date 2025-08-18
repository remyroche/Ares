# src/training/steps/step7_analyst_ensemble_creation.py

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import joblib

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, success, warning
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span

logger = system_logger


class AnalystEnsembleCreationStep:
    """Step 7: Analyst Ensemble Creation - Combines multiple models into ensemble predictions."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.ensemble_models = {}
        self.ensemble_weights = {}

    @handle_errors
    def execute(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        training_input: Dict[str, Any],
    ) -> bool:
        """
        Execute Step 7: Create analyst ensemble models.

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
            enhanced_models_dir = os.path.join(data_dir, "enhanced_hmm_models")
            if not os.path.exists(enhanced_models_dir):
                logger.warning(
                    f"⚠️ Enhanced HMM models directory not found: {enhanced_models_dir}"
                )
                logger.info("📝 Creating placeholder ensemble for Step 7")
                return self._create_placeholder_ensemble(
                    symbol, exchange, data_dir, training_input
                )

            # Load enhanced models from Step 6
            ensemble_models = self._load_enhanced_models(enhanced_models_dir)

            if not ensemble_models:
                logger.warning(
                    "⚠️ No enhanced models found, creating placeholder ensemble"
                )
                return self._create_placeholder_ensemble(
                    symbol, exchange, data_dir, training_input
                )

            # Create ensemble
            ensemble_result = self._create_ensemble(
                ensemble_models, symbol, exchange, data_dir
            )

            # Save ensemble summary
            self._save_ensemble_summary(ensemble_result, symbol, exchange, data_dir)

            logger.info("✅ Step 7: Analyst Ensemble Creation completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error in Step 7: {e}")
            return False

    def _load_enhanced_models(self, enhanced_models_dir: str) -> Dict[str, Any]:
        """Load enhanced models from Step 6."""
        try:
            ensemble_models = {}

            if not os.path.exists(enhanced_models_dir):
                return ensemble_models

            # Look for model files in the enhanced models directory
            for regime_dir in os.listdir(enhanced_models_dir):
                regime_path = os.path.join(enhanced_models_dir, regime_dir)
                if os.path.isdir(regime_path):
                    ensemble_models[regime_dir] = {}

                    for model_file in os.listdir(regime_path):
                        if model_file.endswith(".joblib"):
                            model_path = os.path.join(regime_path, model_file)
                            try:
                                model = joblib.load(model_path)
                                model_name = model_file.replace(".joblib", "")
                                ensemble_models[regime_dir][model_name] = model
                                logger.info(
                                    f"📦 Loaded model: {regime_dir}/{model_name}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Failed to load model {model_path}: {e}"
                                )

            return ensemble_models

        except Exception as e:
            logger.error(f"❌ Error loading enhanced models: {e}")
            return {}

    def _create_ensemble(
        self, ensemble_models: Dict[str, Any], symbol: str, exchange: str, data_dir: str
    ) -> Dict[str, Any]:
        """Create ensemble from loaded models."""
        try:
            # Apply model-specific pruning for ensemble creation
            try:
                from src.training.model_specific_pruning import ModelSpecificPruning
                pruning_manager = ModelSpecificPruning(self.config)
                
                # Get sample data for pruning (if available)
                sample_data = self._get_sample_data_for_pruning(data_dir, symbol, exchange)
                if sample_data is not None:
                    features_df, target = sample_data
                    
                    pruned_features, pruning_metadata = pruning_manager.prune_for_step7_ensemble(
                        features_df, target
                    )
                    
                    logger.info(f"✅ Applied ensemble-specific pruning: {features_df.shape[1]} -> {pruned_features.shape[1]} features")
                    
                    # Store pruning metadata
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
                            "pruning_metadata": pruning_metadata,
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
                logger.warning(f"⚠️ Model-specific pruning failed: {e}")
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
                    model_name: 1.0 / len(models) for model_name in models.keys()
                }

            logger.info(
                f"🎯 Created ensemble with {ensemble_result['ensemble_metadata']['model_count']} models"
            )
            return ensemble_result

        except Exception as e:
            logger.error(f"❌ Error creating ensemble: {e}")
            return {}
    
    def _get_sample_data_for_pruning(self, data_dir: str, symbol: str, exchange: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Get sample data for pruning from existing features."""
        try:
            # Try to load sample features and labels from Step 2 artifacts
            features_file = f"{data_dir}/{exchange}_{symbol}_features_train.parquet"
            labels_file = f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet"
            
            if os.path.exists(features_file) and os.path.exists(labels_file):
                features_df = pd.read_parquet(features_file)
                labels_df = pd.read_parquet(labels_file)
                
                # Align and extract target series
                # This assumes 'target' is the target column and they share an index (e.g., timestamp)
                if 'target' in labels_df.columns:
                    # Ensure indices are aligned before extracting the target
                    if not features_df.index.equals(labels_df.index):
                        if 'timestamp' in labels_df.columns:
                            labels_df = labels_df.set_index('timestamp')
                        labels_df = labels_df.reindex(features_df.index)
                    
                    target = labels_df['target'].dropna()
                    features_df = features_df.loc[target.index]  # Ensure features and target align after dropping NaNs
                    return features_df, target
                else:
                    logger.warning(f"⚠️ Target 'target' column not found in {labels_file}")
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get sample data for pruning: {e}")
            return None

    def _create_placeholder_ensemble(
        self, symbol: str, exchange: str, data_dir: str, training_input: Dict[str, Any]
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
                placeholder_ensemble, symbol, exchange, data_dir
            )

            logger.info("✅ Placeholder ensemble created successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error creating placeholder ensemble: {e}")
            return False

    def _save_ensemble_summary(
        self, ensemble_result: Dict[str, Any], symbol: str, exchange: str, data_dir: str
    ) -> None:
        """Save ensemble summary to file."""
        try:
            # Create ensemble directory
            ensemble_dir = os.path.join(data_dir, "analyst_ensemble")
            os.makedirs(ensemble_dir, exist_ok=True)

            # Save ensemble summary
            summary_file = os.path.join(
                ensemble_dir, f"{exchange}_{symbol}_analyst_ensemble_summary.json"
            )

            # Convert to serializable format
            serializable_result = ensemble_result.copy()
            serializable_result["ensemble_models"] = {
                regime: list(models.keys())
                for regime, models in ensemble_result["ensemble_models"].items()
            }

            with open(summary_file, "w") as f:
                json.dump(serializable_result, f, indent=2, default=str)

            logger.info(f"💾 Saved ensemble summary to {summary_file}")

        except Exception as e:
            logger.error(f"❌ Error saving ensemble summary: {e}")


def step7_analyst_ensemble_creation(
    symbol: str,
    exchange: str,
    data_dir: str,
    training_input: Dict[str, Any],
    config: Dict[str, Any],
) -> bool:
    """
    Step 7: Analyst Ensemble Creation

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        training_input: Training input data
        config: Configuration dictionary

    Returns:
        bool: True if successful
    """
    step = AnalystEnsembleCreationStep(config)
    return step.execute(symbol, exchange, data_dir, training_input)
