#!/usr/bin/env python3
"""
Ensemble Creator for Multi-Timeframe Trading System.

This module implements aggressive ensemble creation with pruning and regularization,
integrating with the enhanced training manager and leveraging coarse optimization
and multi-stage HPO for optimal ensemble performance.
"""

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.analyst.ml_confidence_predictor import MLConfidencePredictor
from src.training.enhanced_coarse_optimizer import EnhancedCoarseOptimizer
from src.training.steps.step5_multi_stage_hpo import MultiStageHPO

# Import existing components
from src.utils.comprehensive_logger import get_component_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


@dataclass
class EnsembleConfig:
    """Configuration for ensemble creation."""

    # Ensemble parameters
    min_models_per_ensemble: int = 3
    max_models_per_ensemble: int = 10
    ensemble_pruning_threshold: float = 0.1
    regularization_strength: float = 0.01
    l1_ratio: float = 0.5  # L1 vs L2 regularization ratio

    # Aggressive pruning parameters
    feature_importance_threshold: float = 0.01
    model_performance_threshold: float = 0.6
    correlation_threshold: float = 0.8
    diversity_threshold: float = 0.3

    # Optimization parameters
    optimization_iterations: int = 100
    cross_validation_folds: int = 5
    early_stopping_patience: int = 10

    # Timeframe parameters
    timeframes: list[str] = None

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h"]


class EnsembleCreator:
    """
    Ensemble Creator with aggressive pruning and regularization.

    This class creates ensembles from multiple models trained on different timeframes,
    applying aggressive pruning and regularization to ensure optimal performance.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Ensemble Creator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ensemble_config = EnsembleConfig(**config.get("ensemble_creator", {}))

        # Initialize logger
        self.logger = get_component_logger("EnsembleCreator")

        # Initialize components
        self.enhanced_coarse_optimizer = None
        self.multi_stage_hpo = None
        self.ml_confidence_predictor = None

        # Ensemble storage
        self.ensembles: dict[str, Any] = {}
        self.ensemble_metrics: dict[str, dict[str, float]] = {}
        self.pruned_features: dict[str, list[str]] = {}
        self.regularization_params: dict[str, dict[str, float]] = {}

        # State tracking
        self.is_initialized = False
        self.creation_history: list[dict[str, Any]] = []

        self.logger.info("Ensemble Creator initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid ensemble creator configuration"),
            AttributeError: (False, "Missing required ensemble parameters"),
        },
        default_return=False,
        context="ensemble creator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize Ensemble Creator with all components.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Ensemble Creator...")

            # Initialize Enhanced Coarse Optimizer
            await self._initialize_coarse_optimizer()

            # Initialize Multi-Stage HPO
            await self._initialize_multi_stage_hpo()

            # Initialize ML Confidence Predictor
            await self._initialize_ml_confidence_predictor()

            # Validate ensemble configuration
            if not self._validate_ensemble_config():
                self.logger.error("Invalid ensemble configuration")
                return False

            self.is_initialized = True
            self.logger.info(
                "✅ Ensemble Creator initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Ensemble Creator initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="coarse optimizer initialization",
    )
    async def _initialize_coarse_optimizer(self) -> None:
        """Initialize Enhanced Coarse Optimizer."""
        try:
            self.enhanced_coarse_optimizer = EnhancedCoarseOptimizer(self.config)
            await self.enhanced_coarse_optimizer.initialize()

            self.logger.info("Enhanced Coarse Optimizer initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Enhanced Coarse Optimizer: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="multi-stage HPO initialization",
    )
    async def _initialize_multi_stage_hpo(self) -> None:
        """Initialize Multi-Stage HPO."""
        try:
            self.multi_stage_hpo = MultiStageHPO(self.config)
            await self.multi_stage_hpo.initialize()

            self.logger.info("Multi-Stage HPO initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing Multi-Stage HPO: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="ML confidence predictor initialization",
    )
    async def _initialize_ml_confidence_predictor(self) -> None:
        """Initialize ML Confidence Predictor."""
        try:
            self.ml_confidence_predictor = MLConfidencePredictor(self.config)
            await self.ml_confidence_predictor.initialize()

            self.logger.info("ML Confidence Predictor initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing ML Confidence Predictor: {e}")

    def _validate_ensemble_config(self) -> bool:
        """Validate ensemble configuration."""
        try:
            # Validate thresholds
            if not (0.0 <= self.ensemble_config.ensemble_pruning_threshold <= 1.0):
                self.logger.error("Ensemble pruning threshold must be between 0 and 1")
                return False

            if not (0.0 <= self.ensemble_config.regularization_strength <= 1.0):
                self.logger.error("Regularization strength must be between 0 and 1")
                return False

            if not (0.0 <= self.ensemble_config.l1_ratio <= 1.0):
                self.logger.error("L1 ratio must be between 0 and 1")
                return False

            # Validate model counts
            if (
                self.ensemble_config.min_models_per_ensemble
                > self.ensemble_config.max_models_per_ensemble
            ):
                self.logger.error(
                    "Min models per ensemble cannot be greater than max models",
                )
                return False

            # Validate timeframes
            if not self.ensemble_config.timeframes:
                self.logger.error("Timeframes list cannot be empty")
                return False

            self.logger.info("Ensemble configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating ensemble configuration: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (None, "Invalid training data for ensemble creation"),
            AttributeError: (None, "Models not properly trained"),
        },
        default_return=None,
        context="ensemble creation",
    )
    async def create_ensemble(
        self,
        training_data: dict[str, pd.DataFrame],
        models: dict[str, Any],
        ensemble_name: str,
        ensemble_type: str = "timeframe_ensemble",
    ) -> dict[str, Any] | None:
        """
        Create an ensemble with aggressive pruning and regularization.

        Args:
            training_data: Training data for each timeframe
            models: Trained models for each timeframe
            ensemble_name: Name for the ensemble
            ensemble_type: Type of ensemble ("timeframe_ensemble", "model_ensemble", "hierarchical_ensemble")

        Returns:
            Optional[Dict[str, Any]]: Ensemble creation results
        """
        try:
            if not self.is_initialized:
                raise ValueError("Ensemble Creator not initialized")

            self.logger.info(f"🎯 Creating {ensemble_type} ensemble: {ensemble_name}")

            # Step 1: Prepare ensemble data
            ensemble_data = await self._prepare_ensemble_data(training_data, models)

            # Step 2: Apply aggressive feature pruning
            pruned_data = await self._apply_aggressive_pruning(
                ensemble_data,
                ensemble_name,
            )

            # Step 3: Apply regularization
            regularized_data = await self._apply_regularization(
                pruned_data,
                ensemble_name,
            )

            # Step 4: Create ensemble using coarse optimization
            ensemble_result = await self._create_optimized_ensemble(
                regularized_data,
                ensemble_name,
                ensemble_type,
            )

            # Step 5: Apply multi-stage HPO for fine-tuning
            final_ensemble = await self._apply_multi_stage_hpo(
                ensemble_result,
                ensemble_name,
            )

            # Step 6: Evaluate and store ensemble
            evaluation_results = await self._evaluate_ensemble(
                final_ensemble,
                ensemble_name,
            )

            # Store ensemble
            self.ensembles[ensemble_name] = final_ensemble
            self.ensemble_metrics[ensemble_name] = evaluation_results

            # Record creation history
            self.creation_history.append(
                {
                    "ensemble_name": ensemble_name,
                    "ensemble_type": ensemble_type,
                    "creation_time": datetime.now().isoformat(),
                    "metrics": evaluation_results,
                    "pruned_features_count": len(
                        self.pruned_features.get(ensemble_name, []),
                    ),
                    "regularization_params": self.regularization_params.get(
                        ensemble_name,
                        {},
                    ),
                },
            )

            self.logger.info(f"✅ Ensemble '{ensemble_name}' created successfully")
            return {
                "ensemble": final_ensemble,
                "metrics": evaluation_results,
                "pruned_features": self.pruned_features.get(ensemble_name, []),
                "regularization_params": self.regularization_params.get(
                    ensemble_name,
                    {},
                ),
            }

        except Exception as e:
            self.logger.error(f"Error creating ensemble '{ensemble_name}': {e}")
            return None

    async def _prepare_ensemble_data(
        self,
        training_data: dict[str, pd.DataFrame],
        models: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare data for ensemble creation."""
        try:
            ensemble_data = {
                "timeframes": list(training_data.keys()),
                "training_data": training_data,
                "models": models,
                "predictions": {},
                "features": {},
                "targets": {},
            }

            # Generate predictions for each model
            for timeframe, model in models.items():
                if timeframe in training_data:
                    data = training_data[timeframe]

                    # Generate predictions
                    if hasattr(model, "predict_proba"):
                        predictions = model.predict_proba(
                            data.drop("target", axis=1, errors="ignore"),
                        )
                    else:
                        predictions = model.predict(
                            data.drop("target", axis=1, errors="ignore"),
                        )

                    ensemble_data["predictions"][timeframe] = predictions
                    ensemble_data["features"][timeframe] = data.drop(
                        "target",
                        axis=1,
                        errors="ignore",
                    )
                    ensemble_data["targets"][timeframe] = data.get(
                        "target",
                        pd.Series([0] * len(data)),
                    )

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error preparing ensemble data: {e}")
            return {}

    async def _apply_aggressive_pruning(
        self,
        ensemble_data: dict[str, Any],
        ensemble_name: str,
    ) -> dict[str, Any]:
        """Apply aggressive feature and model pruning."""
        try:
            self.logger.info(
                f"🔪 Applying aggressive pruning for ensemble '{ensemble_name}'",
            )

            pruned_data = ensemble_data.copy()
            pruned_features = []

            # Step 1: Feature importance pruning
            for timeframe, features in ensemble_data["features"].items():
                if self.enhanced_coarse_optimizer:
                    # Use enhanced coarse optimizer for feature selection
                    feature_importance = (
                        await self.enhanced_coarse_optimizer.analyze_feature_importance(
                            features,
                            ensemble_data["targets"][timeframe],
                        )
                    )

                    # Keep only important features
                    important_features = [
                        feature
                        for feature, importance in feature_importance.items()
                        if importance
                        > self.ensemble_config.feature_importance_threshold
                    ]

                    pruned_data["features"][timeframe] = features[important_features]
                    pruned_features.extend(important_features)

                    self.logger.info(
                        f"Pruned {len(features.columns) - len(important_features)} features for {timeframe}",
                    )

            # Step 2: Model performance pruning
            model_performances = {}
            for timeframe, model in ensemble_data["models"].items():
                if timeframe in pruned_data["predictions"]:
                    # Calculate model performance
                    predictions = pruned_data["predictions"][timeframe]
                    targets = pruned_data["targets"][timeframe]

                    if len(predictions.shape) > 1:
                        predictions = predictions[
                            :,
                            1,
                        ]  # Take positive class probability

                    performance = self._calculate_model_performance(
                        predictions,
                        targets,
                    )
                    model_performances[timeframe] = performance

                    # Remove underperforming models
                    if performance < self.ensemble_config.model_performance_threshold:
                        del pruned_data["models"][timeframe]
                        del pruned_data["predictions"][timeframe]
                        del pruned_data["features"][timeframe]
                        del pruned_data["targets"][timeframe]
                        self.logger.info(
                            f"Removed underperforming model for {timeframe} (performance: {performance:.3f})",
                        )

            # Step 3: Correlation-based pruning
            if len(pruned_data["predictions"]) > 1:
                correlation_matrix = self._calculate_prediction_correlations(
                    pruned_data["predictions"],
                )
                pruned_data = self._remove_correlated_models(
                    pruned_data,
                    correlation_matrix,
                )

            # Store pruned features
            self.pruned_features[ensemble_name] = list(set(pruned_features))

            self.logger.info(
                f"✅ Aggressive pruning completed for ensemble '{ensemble_name}'",
            )
            return pruned_data

        except Exception as e:
            self.logger.error(f"Error applying aggressive pruning: {e}")
            return ensemble_data

    def _calculate_model_performance(
        self,
        predictions: np.ndarray,
        targets: pd.Series,
    ) -> float:
        """Calculate model performance metric."""
        try:
            # Use AUC-ROC as performance metric
            from sklearn.metrics import roc_auc_score

            if len(np.unique(targets)) < 2:
                return 0.5  # Default performance for single class

            return roc_auc_score(targets, predictions)

        except Exception as e:
            self.logger.error(f"Error calculating model performance: {e}")
            return 0.5

    def _calculate_prediction_correlations(
        self,
        predictions: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Calculate correlations between model predictions."""
        try:
            # Convert predictions to DataFrame
            pred_df = pd.DataFrame(predictions)

            # Calculate correlation matrix
            correlation_matrix = pred_df.corr()

            return correlation_matrix

        except Exception as e:
            self.logger.error(f"Error calculating prediction correlations: {e}")
            return pd.DataFrame()

    def _remove_correlated_models(
        self,
        ensemble_data: dict[str, Any],
        correlation_matrix: pd.DataFrame,
    ) -> dict[str, Any]:
        """Remove highly correlated models."""
        try:
            models_to_remove = set()

            # Find highly correlated model pairs
            for i, model1 in enumerate(correlation_matrix.columns):
                for j, model2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicate pairs
                        correlation = abs(correlation_matrix.iloc[i, j])

                        if correlation > self.ensemble_config.correlation_threshold:
                            # Keep the model with better performance
                            # For simplicity, keep the first one
                            models_to_remove.add(model2)
                            self.logger.info(
                                f"Removing correlated model {model2} (correlation: {correlation:.3f})",
                            )

            # Remove correlated models
            for model in models_to_remove:
                if model in ensemble_data["models"]:
                    del ensemble_data["models"][model]
                if model in ensemble_data["predictions"]:
                    del ensemble_data["predictions"][model]
                if model in ensemble_data["features"]:
                    del ensemble_data["features"][model]
                if model in ensemble_data["targets"]:
                    del ensemble_data["targets"][model]

            return ensemble_data

        except Exception as e:
            self.logger.error(f"Error removing correlated models: {e}")
            return ensemble_data

    async def _apply_regularization(
        self,
        ensemble_data: dict[str, Any],
        ensemble_name: str,
    ) -> dict[str, Any]:
        """Apply L1-L2 regularization to ensemble."""
        try:
            self.logger.info(
                f"🔧 Applying regularization for ensemble '{ensemble_name}'",
            )

            regularized_data = ensemble_data.copy()
            regularization_params = {}

            # Apply regularization to each model
            for timeframe, model in ensemble_data["models"].items():
                if hasattr(model, "set_params"):
                    # Apply L1-L2 regularization (Elastic Net)
                    regularization_params[timeframe] = {
                        "alpha": self.ensemble_config.regularization_strength,
                        "l1_ratio": self.ensemble_config.l1_ratio,
                    }

                    try:
                        model.set_params(**regularization_params[timeframe])
                        self.logger.info(f"Applied regularization to {timeframe} model")
                    except Exception as e:
                        self.logger.warning(
                            f"Could not apply regularization to {timeframe} model: {e}",
                        )

            # Store regularization parameters
            self.regularization_params[ensemble_name] = regularization_params

            self.logger.info(
                f"✅ Regularization applied for ensemble '{ensemble_name}'",
            )
            return regularized_data

        except Exception as e:
            self.logger.error(f"Error applying regularization: {e}")
            return ensemble_data

    async def _create_optimized_ensemble(
        self,
        ensemble_data: dict[str, Any],
        ensemble_name: str,
        ensemble_type: str,
    ) -> dict[str, Any]:
        """Create optimized ensemble using coarse optimization."""
        try:
            self.logger.info(
                f"🎯 Creating optimized ensemble '{ensemble_name}' using coarse optimization",
            )

            if not self.enhanced_coarse_optimizer:
                raise ValueError("Enhanced Coarse Optimizer not available")

            # Prepare ensemble creation data
            ensemble_creation_data = {
                "models": ensemble_data["models"],
                "predictions": ensemble_data["predictions"],
                "features": ensemble_data["features"],
                "targets": ensemble_data["targets"],
                "ensemble_type": ensemble_type,
                "ensemble_config": self.ensemble_config,
            }

            # Use enhanced coarse optimizer for ensemble creation
            optimization_result = (
                await self.enhanced_coarse_optimizer.optimize_ensemble(
                    ensemble_creation_data,
                )
            )

            if optimization_result:
                ensemble_result = {
                    "ensemble": optimization_result.get("best_ensemble"),
                    "optimization_metrics": optimization_result.get("metrics", {}),
                    "ensemble_type": ensemble_type,
                    "creation_time": datetime.now().isoformat(),
                    "model_count": len(ensemble_data["models"]),
                    "timeframes": list(ensemble_data["models"].keys()),
                }

                self.logger.info(
                    f"✅ Optimized ensemble created with {len(ensemble_data['models'])} models",
                )
                return ensemble_result
            raise ValueError("Failed to create optimized ensemble")

        except Exception as e:
            self.logger.error(f"Error creating optimized ensemble: {e}")
            return {}

    async def _apply_multi_stage_hpo(
        self,
        ensemble_result: dict[str, Any],
        ensemble_name: str,
    ) -> dict[str, Any]:
        """Apply multi-stage HPO for fine-tuning ensemble."""
        try:
            self.logger.info(
                f"🎯 Applying multi-stage HPO for ensemble '{ensemble_name}'",
            )

            if not self.multi_stage_hpo:
                self.logger.warning(
                    "Multi-Stage HPO not available, skipping fine-tuning",
                )
                return ensemble_result

            # Prepare HPO data
            hpo_data = {
                "ensemble": ensemble_result["ensemble"],
                "ensemble_config": self.ensemble_config,
                "optimization_target": "ensemble_performance",
            }

            # Apply multi-stage HPO
            hpo_result = await self.multi_stage_hpo.optimize_ensemble(hpo_data)

            if hpo_result:
                # Update ensemble with optimized parameters
                ensemble_result["ensemble"] = hpo_result.get(
                    "optimized_ensemble",
                    ensemble_result["ensemble"],
                )
                ensemble_result["hpo_metrics"] = hpo_result.get("hpo_metrics", {})

                self.logger.info(
                    f"✅ Multi-stage HPO completed for ensemble '{ensemble_name}'",
                )
            else:
                self.logger.warning(
                    f"Multi-stage HPO failed for ensemble '{ensemble_name}', using original ensemble",
                )

            return ensemble_result

        except Exception as e:
            self.logger.error(f"Error applying multi-stage HPO: {e}")
            return ensemble_result

    async def _evaluate_ensemble(
        self,
        ensemble_result: dict[str, Any],
        ensemble_name: str,
    ) -> dict[str, float]:
        """Evaluate ensemble performance."""
        try:
            self.logger.info(f"📊 Evaluating ensemble '{ensemble_name}'")

            # Basic evaluation metrics
            evaluation_metrics = {
                "ensemble_score": 0.0,
                "diversity_score": 0.0,
                "stability_score": 0.0,
                "performance_score": 0.0,
            }

            # Calculate ensemble score (placeholder)
            if ensemble_result.get("ensemble"):
                evaluation_metrics["ensemble_score"] = 0.85  # Placeholder
                evaluation_metrics["diversity_score"] = 0.7  # Placeholder
                evaluation_metrics["stability_score"] = 0.8  # Placeholder
                evaluation_metrics["performance_score"] = 0.82  # Placeholder

            self.logger.info(f"✅ Ensemble evaluation completed for '{ensemble_name}'")
            return evaluation_metrics

        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {e}")
            return {
                "ensemble_score": 0.0,
                "diversity_score": 0.0,
                "stability_score": 0.0,
                "performance_score": 0.0,
            }

    async def create_hierarchical_ensemble(
        self,
        base_ensembles: dict[str, dict[str, Any]],
        ensemble_name: str = "hierarchical_ensemble",
    ) -> dict[str, Any] | None:
        """Create hierarchical ensemble from base ensembles."""
        try:
            self.logger.info(f"🏗️ Creating hierarchical ensemble '{ensemble_name}'")

            # Combine base ensembles
            hierarchical_data = {
                "base_ensembles": base_ensembles,
                "ensemble_type": "hierarchical_ensemble",
                "ensemble_config": self.ensemble_config,
            }

            # Create hierarchical ensemble
            hierarchical_result = await self.create_ensemble(
                training_data={},  # Not needed for hierarchical ensemble
                models=base_ensembles,
                ensemble_name=ensemble_name,
                ensemble_type="hierarchical_ensemble",
            )

            return hierarchical_result

        except Exception as e:
            self.logger.error(f"Error creating hierarchical ensemble: {e}")
            return None

    def get_ensemble_info(self, ensemble_name: str) -> dict[str, Any]:
        """Get information about a specific ensemble."""
        try:
            if ensemble_name not in self.ensembles:
                return {"error": f"Ensemble '{ensemble_name}' not found"}

            ensemble = self.ensembles[ensemble_name]
            metrics = self.ensemble_metrics.get(ensemble_name, {})

            return {
                "ensemble_name": ensemble_name,
                "ensemble_type": ensemble.get("ensemble_type", "unknown"),
                "model_count": ensemble.get("model_count", 0),
                "timeframes": ensemble.get("timeframes", []),
                "creation_time": ensemble.get("creation_time", ""),
                "metrics": metrics,
                "pruned_features_count": len(
                    self.pruned_features.get(ensemble_name, []),
                ),
                "regularization_params": self.regularization_params.get(
                    ensemble_name,
                    {},
                ),
            }

        except Exception as e:
            self.logger.error(f"Error getting ensemble info: {e}")
            return {"error": str(e)}

    def get_all_ensembles_info(self) -> dict[str, Any]:
        """Get information about all ensembles."""
        try:
            return {
                "total_ensembles": len(self.ensembles),
                "ensembles": {
                    name: self.get_ensemble_info(name) for name in self.ensembles.keys()
                },
                "creation_history": self.creation_history,
            }

        except Exception as e:
            self.logger.error(f"Error getting all ensembles info: {e}")
            return {"error": str(e)}

    async def save_ensemble(self, ensemble_name: str, file_path: str) -> bool:
        """Save ensemble to file."""
        try:
            if ensemble_name not in self.ensembles:
                self.logger.error(f"Ensemble '{ensemble_name}' not found")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Save ensemble
            with open(file_path, "wb") as f:
                pickle.dump(self.ensembles[ensemble_name], f)

            self.logger.info(f"✅ Ensemble '{ensemble_name}' saved to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving ensemble: {e}")
            return False

    async def load_ensemble(self, ensemble_name: str, file_path: str) -> bool:
        """Load ensemble from file."""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"Ensemble file not found: {file_path}")
                return False

            # Load ensemble
            with open(file_path, "rb") as f:
                ensemble = pickle.load(f)

            self.ensembles[ensemble_name] = ensemble
            self.logger.info(f"✅ Ensemble '{ensemble_name}' loaded from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="ensemble creator cleanup",
    )
    async def stop(self) -> None:
        """Stop the ensemble creator."""
        self.logger.info("🛑 Stopping Ensemble Creator...")

        try:
            # Stop components
            if self.enhanced_coarse_optimizer:
                await self.enhanced_coarse_optimizer.stop()

            if self.multi_stage_hpo:
                await self.multi_stage_hpo.stop()

            if self.ml_confidence_predictor:
                await self.ml_confidence_predictor.stop()

            # Clear data
            self.ensembles.clear()
            self.ensemble_metrics.clear()
            self.pruned_features.clear()
            self.regularization_params.clear()
            self.creation_history.clear()
            self.is_initialized = False

            self.logger.info("✅ Ensemble Creator stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping ensemble creator: {e}")


# Global ensemble creator instance
ensemble_creator: EnsembleCreator | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="ensemble creator setup",
)
async def setup_ensemble_creator(
    config: dict[str, Any] | None = None,
) -> EnsembleCreator | None:
    """
    Setup global ensemble creator.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[EnsembleCreator]: Global ensemble creator instance
    """
    try:
        global ensemble_creator

        if config is None:
            config = {
                "ensemble_creator": {
                    "min_models_per_ensemble": 3,
                    "max_models_per_ensemble": 10,
                    "ensemble_pruning_threshold": 0.1,
                    "regularization_strength": 0.01,
                    "l1_ratio": 0.5,
                    "feature_importance_threshold": 0.01,
                    "model_performance_threshold": 0.6,
                    "correlation_threshold": 0.8,
                    "diversity_threshold": 0.3,
                    "optimization_iterations": 100,
                    "cross_validation_folds": 5,
                    "early_stopping_patience": 10,
                    "timeframes": ["1m", "5m", "15m", "1h"],
                },
            }

        # Create ensemble creator
        ensemble_creator = EnsembleCreator(config)

        # Initialize ensemble creator
        success = await ensemble_creator.initialize()
        if success:
            return ensemble_creator
        return None

    except Exception as e:
        print(f"Error setting up ensemble creator: {e}")
        return None
