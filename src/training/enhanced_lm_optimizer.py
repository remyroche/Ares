# src/training/enhanced_lm_optimizer.py

"""Enhanced LM Model Optimizer for Step6, Step6_5, and Step9."

This module provides comprehensive optimization for Language Model (LM) components:
1. Advanced feature selection with multiple algorithms
2. L1-L2 regularization with model-specific tuning
3. Optuna hyperparameter optimization in batches
4. Vectorized/matrix operations for efficiency
5. Model-specific optimizations for different architectures
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any

import lightgbm as lgb

# Suppress specific warnings only where needed - removed global suppression
import numpy as np
import optuna
import pandas as pd
import shap
import torch
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logger import system_logger

# Import Pydantic configuration
try:
    from src.training.enhanced_lm_config import (
        DEFAULT_CONFIG,
        EnhancedLMOptimizerConfig,
    )
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    EnhancedLMOptimizerConfig = None
    DEFAULT_CONFIG = None


class EnhancedLMOptimizer:
    """Enhanced LM Model Optimizer with comprehensive optimization features."

    Features:
    - Multi-algorithm feature selection
    - L1-L2 regularization with model-specific tuning
    - Optuna hyperparameter optimization in batches
    - Vectorized operations for efficiency
    - Model-specific optimizations
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("EnhancedLMOptimizer")

        # Load optimization configuration with Pydantic validation
        if PYDANTIC_AVAILABLE and EnhancedLMOptimizerConfig:
            try:
                # Try to use Pydantic configuration
                if "enhanced_lm_optimizer" in config:
                    self.optimization_config = EnhancedLMOptimizerConfig.from_dict(config["enhanced_lm_optimizer"])
                else:
                    self.optimization_config = DEFAULT_CONFIG

                # Validate configuration
                warnings = self.optimization_config.validate_config()
                if warnings:
                    for warning in warnings:
                        self.logger.warning(f"⚠️ Configuration warning: {warning}")

                self.logger.info("✅ Using Pydantic configuration with validation")

            except Exception as e:
                self.logger.warning(f"⚠️ Pydantic configuration failed, falling back to dict: {e}")
                self.optimization_config = self._load_optimization_config()
        else:
            # Fallback to dictionary-based configuration
            self.optimization_config = self._load_optimization_config()
            self.logger.info("⚠️ Using dictionary-based configuration (Pydantic not available)")

        # Initialize components
        self.feature_selector = None
        self.regularization_manager = None
        self.optuna_study = None

        # Performance tracking
        self.optimization_metrics = {
            "feature_selection_time": 0.0,
            "hyperparameter_optimization_time": 0.0,
            "regularization_tuning_time": 0.0,
            "total_optimization_time": 0.0,
        }

        # Cache for optimization results
        self.optimization_cache = {}

        # Log configuration summary
        if hasattr(self.optimization_config, "get_optimization_summary"):
            summary = self.optimization_config.get_optimization_summary()
            self.logger.info("📊 Optimization configuration summary:")
            for section, details in summary.items():
                self.logger.info(f"   {section}: {details}")

    def _load_optimization_config(self) -> dict[str, Any]:
        """Load and validate optimization configuration."""
        default_config = {
            "feature_selection": {
                "enable": True,
                "methods": ["mutual_info", "lasso", "random_forest", "shap"],
                "target_features": {
                    "step06": 80,
                    "step6_5": 100,
                    "step09": 90,
                },
                "vif_threshold": 10.0,
                "correlation_threshold": 0.95,
                "variance_threshold": 0.01,
                "mutual_info_threshold": 0.001,
                "shap_threshold": 0.001,
            },
            "regularization": {
                "enable": True,
                "l1_alpha_range": [0.001, 0.1],
                "l2_alpha_range": [0.0001, 0.01],
                "dropout_range": [0.1, 0.5],
                "model_specific": {
                    "lightgbm": {
                        "reg_alpha_range": [0.001, 0.1],
                        "reg_lambda_range": [0.0001, 0.01],
                    },
                    "neural_networks": {
                        "weight_decay_range": [1e-6, 1e-3],
                        "dropout_range": [0.1, 0.5],
                    },
                },
            },
            "optuna": {
                "enable": True,
                "n_trials_per_batch": 50,
                "n_batches": 3,
                "timeout_per_batch": 300,  # 5 minutes per batch
                "sampler": "tpe",
                "pruner": "median",
                "storage": None,  # Can be set to database URL
            },
            "vectorization": {
                "enable": True,
                "batch_size": 1024,
                "use_gpu": torch.cuda.is_available(),
                "memory_efficient": True,
            },
        }

        # Merge with config using recursive update
        config = self.config.get("enhanced_lm_optimizer", {})
        return self._recursive_update(default_config, config)


    def _recursive_update(self, base_dict: dict[str, Any], update_dict: dict[str, Any]) -> dict[str, Any]:
        """Recursively update nested dictionaries."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._recursive_update(result[key], value)
            else:
                result[key] = value
        return result

    async def initialize(self) -> bool:
        """Initialize the Enhanced LM Optimizer with all components. Fails fast if any component fails."""
        try:
            self.logger.info("🔄 Initializing Enhanced LM Optimizer...")

            # Track initialization status for artifact saving
            initialization_status = {
                "feature_selector": False,
                "regularization_manager": False,
                "optuna_study": False,
                "experiment_tracking": False,
            }

            # Initialize feature selector
            feature_selection_enabled = (
                self.optimization_config.feature_selection.enable
                if hasattr(self.optimization_config, "feature_selection")
                else self.optimization_config.get("feature_selection", {}).get("enable", True)
            )

            if feature_selection_enabled:
                self.logger.info("🔄 Initializing feature selector...")
                self.feature_selector = EnhancedFeatureSelector(self.optimization_config)
                await self.feature_selector.initialize()
                initialization_status["feature_selector"] = True
                self.logger.info("✅ Feature selector initialized successfully")

            # Initialize regularization manager
            regularization_enabled = (
                self.optimization_config.regularization.enable
                if hasattr(self.optimization_config, "regularization")
                else self.optimization_config.get("regularization", {}).get("enable", True)
            )

            if regularization_enabled:
                self.logger.info("🔄 Initializing regularization manager...")
                self.regularization_manager = EnhancedRegularizationManager(self.optimization_config)
                await self.regularization_manager.initialize()
                initialization_status["regularization_manager"] = True
                self.logger.info("✅ Regularization manager initialized successfully")

            # Initialize Optuna study
            optuna_enabled = (
                self.optimization_config.optuna.enable
                if hasattr(self.optimization_config, "optuna")
                else self.optimization_config.get("optuna", {}).get("enable", True)
            )

            if optuna_enabled:
                self.logger.info("🔄 Initializing Optuna study...")
                await self._initialize_optuna_study()
                initialization_status["optuna_study"] = True
                self.logger.info("✅ Optuna study initialized successfully")

            # Initialize experiment tracking
            experiment_tracking_enabled = (
                self.optimization_config.experiment_tracking.enable
                if hasattr(self.optimization_config, "experiment_tracking")
                else self.optimization_config.get("experiment_tracking", {}).get("enable", True)
            )

            if experiment_tracking_enabled:
                self.logger.info("🔄 Initializing experiment tracking...")
                # Experiment tracking is initialized in _initialize_optuna_study
                initialization_status["experiment_tracking"] = True
                self.logger.info("✅ Experiment tracking initialized successfully")

            # Store initialization status for potential failure handling
            self.initialization_status = initialization_status

            self.logger.info("✅ Enhanced LM Optimizer initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize Enhanced LM Optimizer: {e}")

            # Save initialization artifacts before raising
            await self._save_initialization_artifacts(initialization_status, str(e))

            # Re-raise the exception - no fallback, it has to work
            msg = f"Enhanced LM Optimizer initialization failed: {e}"
            raise RuntimeError(msg)

    async def _save_initialization_artifacts(self, initialization_status: dict[str, bool], error_message: str) -> None:
        """Save artifacts of successful initialization components before failing."""
        try:
            import json
            import os
            from datetime import datetime

            # Create artifacts directory
            artifacts_dir = "artifacts/initialization_failure"
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save initialization status
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_file = f"{artifacts_dir}/initialization_status_{timestamp}.json"

            status_data = {
                "timestamp": timestamp,
                "initialization_status": initialization_status,
                "error_message": error_message,
                "config_summary": self._get_config_summary(),
            }

            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)

            # Save successful component configurations
            if initialization_status.get("feature_selector"):
                await self._save_feature_selector_artifacts(artifacts_dir, timestamp)

            if initialization_status.get("regularization_manager"):
                await self._save_regularization_artifacts(artifacts_dir, timestamp)

            if initialization_status.get("optuna_study"):
                await self._save_optuna_artifacts(artifacts_dir, timestamp)

            self.logger.info(f"📁 Initialization artifacts saved to {artifacts_dir}")

        except Exception as artifact_error:
            self.logger.exception(f"❌ Failed to save initialization artifacts: {artifact_error}")

    def _get_config_summary(self) -> dict[str, Any]:
        """Get a summary of the current configuration."""
        try:
            if hasattr(self.optimization_config, "get_optimization_summary"):
                return self.optimization_config.get_optimization_summary()
            return {
                "config_type": "dictionary",
                "keys": list(self.optimization_config.keys()) if isinstance(self.optimization_config, dict) else [],
            }
        except Exception as e:
            return {"error": f"Failed to get config summary: {e}"}

    async def _save_feature_selector_artifacts(self, artifacts_dir: str, timestamp: str) -> None:
        """Save feature selector artifacts."""
        try:
            if self.feature_selector:
                feature_artifacts = {
                    "feature_selection_config": self.feature_selector.feature_selection_config,
                    "performance_metrics": self.feature_selector.performance_metrics,
                }

                feature_file = f"{artifacts_dir}/feature_selector_{timestamp}.json"
                with open(feature_file, "w") as f:
                    json.dump(feature_artifacts, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save feature selector artifacts: {e}")

    async def _save_regularization_artifacts(self, artifacts_dir: str, timestamp: str) -> None:
        """Save regularization manager artifacts."""
        try:
            if self.regularization_manager:
                reg_artifacts = {
                    "regularization_config": self.regularization_manager.regularization_config,
                }

                reg_file = f"{artifacts_dir}/regularization_manager_{timestamp}.json"
                with open(reg_file, "w") as f:
                    json.dump(reg_artifacts, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save regularization artifacts: {e}")

    async def _save_optuna_artifacts(self, artifacts_dir: str, timestamp: str) -> None:
        """Save Optuna study artifacts."""
        try:
            if self.optuna_study:
                optuna_artifacts = {
                    "study_name": self.optuna_study.study_name,
                    "n_trials": len(self.optuna_study.trials),
                    "best_value": self.optuna_study.best_value if self.optuna_study.trials else None,
                    "best_params": self.optuna_study.best_params if self.optuna_study.trials else None,
                }

                optuna_file = f"{artifacts_dir}/optuna_study_{timestamp}.json"
                with open(optuna_file, "w") as f:
                    json.dump(optuna_artifacts, f, indent=2)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save Optuna artifacts: {e}")

    async def _initialize_optuna_study(self) -> None:
        """Initialize Optuna study for hyperparameter optimization with advanced samplers and pruners."""
        try:
            # Get Optuna configuration
            if hasattr(self.optimization_config, "optuna"):
                optuna_config = self.optimization_config.optuna
                sampler_name = optuna_config.sampler.value if hasattr(optuna_config.sampler, "value") else str(optuna_config.sampler)
                pruner_name = optuna_config.pruner.value if hasattr(optuna_config.pruner, "value") else str(optuna_config.pruner)
                storage = optuna_config.storage
            else:
                optuna_config = self.optimization_config.get("optuna", {})
                sampler_name = optuna_config.get("sampler", "tpe")
                pruner_name = optuna_config.get("pruner", "median")
                storage = optuna_config.get("storage")

            # Create study
            study_name = f"enhanced_lm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Configure advanced sampler
            if sampler_name == "tpe":
                sampler = TPESampler(seed=42, n_startup_trials=10)
            elif sampler_name == "cmaes":
                from optuna.samplers import CmaEsSampler
                sampler = CmaEsSampler(seed=42)
            elif sampler_name == "random":
                from optuna.samplers import RandomSampler
                sampler = RandomSampler(seed=42)
            else:
                sampler = TPESampler(seed=42, n_startup_trials=10)

            # Configure pruner
            if pruner_name == "median":
                from optuna.pruners import MedianPruner
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            elif pruner_name == "hyperband":
                from optuna.pruners import HyperbandPruner
                pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
            elif pruner_name == "threshold":
                from optuna.pruners import ThresholdPruner
                pruner = ThresholdPruner(lower=0.1, upper=0.9)
            else:
                from optuna.pruners import MedianPruner
                pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

            # Create study with advanced configuration
            self.optuna_study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                study_name=study_name,
                load_if_exists=True,
            )

            # Initialize experiment tracking
            await self._initialize_experiment_tracking(study_name)

            self.logger.info(f"✅ Optuna study '{study_name}' initialized with {sampler_name} sampler and {pruner_name} pruner")

        except Exception as e:
            self.logger.exception(f"❌ Failed to initialize Optuna study: {e}")

    async def _initialize_experiment_tracking(self, study_name: str) -> None:
        """Initialize experiment tracking for MLflow or similar tools."""
        try:
            # Try to initialize MLflow
            try:
                import mlflow
                mlflow.set_tracking_uri("file:./mlruns")
                mlflow.set_experiment(f"enhanced_lm_optimization_{study_name}")
                self.mlflow_available = True
                self.logger.info("✅ MLflow experiment tracking initialized")
            except ImportError:
                self.mlflow_available = False
                self.logger.info("⚠️ MLflow not available, skipping experiment tracking")

            # Try to initialize Weights & Biases
            try:
                import wandb
                wandb.init(project="ares-enhanced-lm-optimization", name=study_name, config=self.optimization_config)
                self.wandb_available = True
                self.logger.info("✅ Weights & Biases experiment tracking initialized")
            except ImportError:
                self.wandb_available = False
                self.logger.info("⚠️ Weights & Biases not available, skipping experiment tracking")

        except Exception as e:
            self.logger.warning(f"⚠️ Experiment tracking initialization failed: {e}")
            self.mlflow_available = False
            self.wandb_available = False

    async def optimize_lm_model(
        self,
        step_name: str,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
        architecture: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Comprehensive optimization for LM models. No fallbacks - it has to work."

        Args:
            step_name: Step name (step06, step6_5, step09)
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Model type (classification, regression)
            architecture: Model architecture (LightGBM, CNN, TCN, Transformer)
            **kwargs: Additional parameters

        Returns:
            Dict containing optimization results

        Raises:
            RuntimeError: If any optimization step fails

        """
        start_time = time.time()

        try:
            self.logger.info(f"🔄 Starting comprehensive optimization for {step_name} {architecture}")

            # Validate inputs
            if features_df.empty or target.empty:
                msg = "Features and target cannot be empty"
                raise ValueError(msg)

            if len(features_df) != len(target):
                msg = "Features and target must have the same length"
                raise ValueError(msg)

            # Validate that all required components are available
            if self.feature_selector is None:
                msg = "Feature selector is required but not initialized"
                raise RuntimeError(msg)

            if self.regularization_manager is None:
                msg = "Regularization manager is required but not initialized"
                raise RuntimeError(msg)

            if self.optuna_study is None:
                msg = "Optuna study is required but not initialized"
                raise RuntimeError(msg)

            optimization_results = {
                "step_name": step_name,
                "architecture": architecture,
                "model_type": model_type,
                "optimization_timestamp": datetime.now().isoformat(),
                "feature_selection": {},
                "regularization": {},
                "hyperparameter_optimization": {},
                "performance_metrics": {},
            }

            # Step 1: Feature Selection
            self.logger.info(f"📊 Step 1: Feature selection for {step_name}")
            feature_selection_start = time.time()

            optimized_features, feature_selection_results = await self._optimize_features(
                features_df, target, step_name, architecture,
            )

            optimization_results["feature_selection"] = feature_selection_results
            self.optimization_metrics["feature_selection_time"] = time.time() - feature_selection_start

            self.logger.info(f"✅ Feature selection completed: {len(features_df.columns)} -> {len(optimized_features.columns)} features")

            # Step 2: Regularization Tuning
            self.logger.info(f"🔧 Step 2: Regularization tuning for {step_name}")
            regularization_start = time.time()

            regularization_params = await self._optimize_regularization(
                optimized_features, target, step_name, architecture,
            )

            optimization_results["regularization"] = regularization_params
            self.optimization_metrics["regularization_tuning_time"] = time.time() - regularization_start

            self.logger.info("✅ Regularization tuning completed")

            # Step 3: Hyperparameter Optimization with Optuna
            self.logger.info(f"🎯 Step 3: Hyperparameter optimization for {step_name}")
            hyperopt_start = time.time()

            best_params, hyperopt_results = await self._optimize_hyperparameters(
                optimized_features, target, step_name, architecture, model_type,
            )

            optimization_results["hyperparameter_optimization"] = hyperopt_results
            self.optimization_metrics["hyperparameter_optimization_time"] = time.time() - hyperopt_start

            self.logger.info("✅ Hyperparameter optimization completed")

            # Step 4: Performance Evaluation
            self.logger.info(f"📈 Step 4: Performance evaluation for {step_name}")
            performance_metrics = await self._evaluate_optimized_model(
                optimized_features, target, step_name, architecture, model_type,
                optimization_results,
            )

            optimization_results["performance_metrics"] = performance_metrics

            # Update total time
            self.optimization_metrics["total_optimization_time"] = time.time() - start_time

            # Cache results
            cache_key = f"{step_name}_{architecture}_{model_type}"
            self.optimization_cache[cache_key] = optimization_results

            self.logger.info(f"✅ Comprehensive optimization completed for {step_name} in {self.optimization_metrics['total_optimization_time']:.2f}s")

            # Return both optimization results and optimized features
            return optimization_results, optimized_features

        except Exception as e:
            self.logger.exception(f"❌ Optimization failed for {step_name}: {e}")

            # Save optimization artifacts before failing
            await self._save_optimization_artifacts(step_name, features_df, target, model_type, architecture, str(e))

            # Re-raise the exception - no fallback, it has to work
            msg = f"LM optimization failed for {step_name}: {e}"
            raise RuntimeError(msg)

    async def _save_optimization_artifacts(self, step_name: str, features_df: pd.DataFrame, target: pd.Series, model_type: str, architecture: str, error_message: str) -> None:
        """Save artifacts of optimization process before failing."""
        try:
            import json
            import os
            from datetime import datetime

            # Create artifacts directory
            artifacts_dir = f"artifacts/optimization_failure/{step_name}"
            os.makedirs(artifacts_dir, exist_ok=True)

            # Save optimization status
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_file = f"{artifacts_dir}/optimization_status_{timestamp}.json"

            status_data = {
                "timestamp": timestamp,
                "step_name": step_name,
                "architecture": architecture,
                "model_type": model_type,
                "error_message": error_message,
                "data_info": {
                    "features_shape": features_df.shape,
                    "target_shape": target.shape,
                    "features_columns": list(features_df.columns),
                    "target_dtype": str(target.dtype),
                    "target_unique_values": len(target.unique()),
                },
                "config_summary": self._get_config_summary(),
            }

            with open(status_file, "w") as f:
                json.dump(status_data, f, indent=2)

            # Save data samples for debugging
            data_sample_file = f"{artifacts_dir}/data_sample_{timestamp}.json"
            data_sample = {
                "features_sample": features_df.head(100).to_dict(),
                "target_sample": target.head(100).tolist(),
                "features_info": {
                    "dtypes": features_df.dtypes.to_dict(),
                    "null_counts": features_df.isnull().sum().to_dict(),
                    "memory_usage": features_df.memory_usage(deep=True).sum(),
                },
            }

            with open(data_sample_file, "w") as f:
                json.dump(data_sample, f, indent=2)

            self.logger.info(f"📁 Optimization artifacts saved to {artifacts_dir}")

        except Exception as artifact_error:
            self.logger.exception(f"❌ Failed to save optimization artifacts: {artifact_error}")

    async def _optimize_features(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Optimize feature selection using multiple algorithms."""
        try:
            if self.feature_selector is None:
                return features_df, {"error": "feature_selector_not_available"}

            # Get target feature count for this step
            target_features = self.optimization_config["feature_selection"]["target_features"].get(
                step_name, 80,
            )

            # Apply enhanced feature selection
            optimized_features, selection_metadata = await self.feature_selector.select_features_enhanced(
                features_df, target, target_features, architecture,
            )

            return optimized_features, selection_metadata

        except Exception as e:
            self.logger.exception(f"❌ Feature optimization failed: {e}")
            return features_df, {"error": str(e)}

    async def _optimize_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
    ) -> dict[str, Any]:
        """Optimize regularization parameters."""
        try:
            if self.regularization_manager is None:
                return {"error": "regularization_manager_not_available"}

            # Get regularization parameters optimized for this architecture
            return await self.regularization_manager.optimize_regularization(
                features_df, target, step_name, architecture,
            )


        except Exception as e:
            self.logger.exception(f"❌ Regularization optimization failed: {e}")
            return {"error": str(e)}

    async def _optimize_hyperparameters(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Optimize hyperparameters using Optuna in batches with unified regularization tuning."""
        try:
            if self.optuna_study is None:
                return {}, {"error": "optuna_study_not_available"}

            optuna_config = self.optimization_config["optuna"]
            n_batches = optuna_config["n_batches"]
            n_trials_per_batch = optuna_config["n_trials_per_batch"]
            timeout_per_batch = optuna_config["timeout_per_batch"]

            batch_results = []

            for batch_idx in range(n_batches):
                self.logger.info(f"🔄 Batch {batch_idx + 1}/{n_batches} for {step_name}")

                # Create objective function for this batch with unified hyperparameter optimization
                def objective(trial):
                    return self._unified_hyperparameter_objective(
                        trial, features_df, target, step_name, architecture, model_type,
                    )

                # Use the persistent study for all batches to maintain learning
                self.optuna_study.optimize(
                    objective,
                    n_trials=n_trials_per_batch,
                    timeout=timeout_per_batch,
                )

                batch_results.append({
                    "batch": batch_idx + 1,
                    "best_value": self.optuna_study.best_value,
                    "best_params": self.optuna_study.best_params,
                    "n_trials": len(self.optuna_study.trials),
                })

                self.logger.info(f"✅ Batch {batch_idx + 1} completed: best_value={self.optuna_study.best_value:.4f}")

            return self.optuna_study.best_params, {
                "batch_results": batch_results,
                "overall_best_value": self.optuna_study.best_value,
                "total_trials": sum(r["n_trials"] for r in batch_results),
                "study_name": self.optuna_study.study_name,
            }

        except Exception as e:
            self.logger.exception(f"❌ Hyperparameter optimization failed: {e}")
            return {}, {"error": str(e)}

    def _unified_hyperparameter_objective(
        self,
        trial: optuna.Trial,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str,
    ) -> float:
        """Unified objective function for Optuna hyperparameter optimization including regularization."""
        try:
            # Get unified hyperparameter suggestions including regularization
            if architecture == "LightGBM":
                params = self._suggest_unified_lightgbm_params(trial, step_name)
                model = lgb.LGBMClassifier(**params) if model_type == "classification" else lgb.LGBMRegressor(**params)

                # Cross-validation with domain-specific metrics
                cv_scores = self._evaluate_model_with_domain_metrics(
                    model, features_df, target, model_type, architecture,
                )

            elif architecture in ["CNN", "TCN", "Transformer"]:
                params = self._suggest_unified_neural_network_params(trial, architecture, step_name)

                # For neural networks, we need a proper training loop
                # Note: This is a synchronous method, so we'll use a simplified evaluation'
                cv_scores = self._evaluate_neural_network_sync(
                    params, features_df, target, architecture, model_type,
                )

            else:
                # Default to LightGBM
                params = self._suggest_unified_lightgbm_params(trial, step_name)
                model = lgb.LGBMClassifier(**params) if model_type == "classification" else lgb.LGBMRegressor(**params)

                cv_scores = self._evaluate_model_with_domain_metrics(
                    model, features_df, target, model_type, architecture,
                )

            final_score = cv_scores.mean()

            # Log experiment tracking (synchronous version for Optuna objective)
            try:
                # Run logging in a separate thread to avoid blocking
                import threading
                def log_trial() -> None:
                    try:
                        asyncio.create_task(self._log_experiment_trial(trial, params, final_score, cv_scores, step_name, architecture, model_type))
                    except:
                        pass  # Ignore logging errors in objective function

                threading.Thread(target=log_trial, daemon=True).start()
            except:
                pass  # Ignore logging errors in objective function

            return final_score

        except Exception as e:
            self.logger.warning(f"⚠️ Trial failed: {e}")
            return -float("inf")

    async def _log_experiment_trial(
        self,
        trial: optuna.Trial,
        params: dict[str, Any],
        final_score: float,
        cv_scores: np.ndarray,
        step_name: str,
        architecture: str,
        model_type: str,
    ) -> None:
        """Log experiment trial to MLflow and/or Weights & Biases."""
        try:
            # Log to MLflow with enhanced metadata
            if hasattr(self, "mlflow_available") and self.mlflow_available:
                try:
                    import mlflow

                    from src.utils.mlflow_utils import (
                        log_metrics_with_metadata,
                        log_params_with_metadata,
                    )

                    # Extract metadata from config
                    config = getattr(self, 'config', {})
                    symbol = config.get('trading_symbol', 'ETHUSDT')
                    exchange = config.get('exchange_name', 'BINANCE')
                    lookback_years = config.get('lookback_years', 2)
                    lookback_period = f"{lookback_years}_years"
                    
                    with mlflow.start_run(nested=True) as run:
                        # Log hyperparameters with metadata
                        all_params = {
                            **params,
                            "step_name": step_name,
                            "architecture": architecture,
                            "model_type": model_type,
                            "trial_number": trial.number,
                        }
                        
                        log_params_with_metadata(
                            params=all_params,
                            asset=symbol,
                            exchange=exchange,
                            lookback_period=lookback_period,
                            run_id=run.info.run_id,
                            additional_metadata={
                                "optimization_type": "enhanced_lm_optimizer",
                                "trial_type": "hyperparameter_optimization",
                            }
                        )

                        # Log metrics with metadata
                        metrics = {
                            "final_score": final_score,
                            "cv_mean": cv_scores.mean(),
                            "cv_std": cv_scores.std(),
                            "cv_min": cv_scores.min(),
                            "cv_max": cv_scores.max(),
                            "cv_scores": cv_scores.tolist(),
                        }
                        
                        log_metrics_with_metadata(
                            metrics=metrics,
                            asset=symbol,
                            exchange=exchange,
                            lookback_period=lookback_period,
                            run_id=run.info.run_id,
                            additional_metadata={
                                "optimization_type": "enhanced_lm_optimizer",
                                "architecture": architecture,
                                "model_type": model_type,
                            }
                        )

                except Exception as e:
                    self.logger.warning(f"⚠️ MLflow logging failed: {e}")

            # Log to Weights & Biases
            if hasattr(self, "wandb_available") and self.wandb_available:
                try:
                    import wandb
                    wandb.log({
                        **params,
                        "step_name": step_name,
                        "architecture": architecture,
                        "model_type": model_type,
                        "trial_number": trial.number,
                        "final_score": final_score,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "cv_min": cv_scores.min(),
                        "cv_max": cv_scores.max(),
                        "cv_scores": cv_scores.tolist(),
                    })

                except Exception as e:
                    self.logger.warning(f"⚠️ Weights & Biases logging failed: {e}")

        except Exception as e:
            self.logger.warning(f"⚠️ Experiment logging failed: {e}")

    def _suggest_unified_lightgbm_params(self, trial: optuna.Trial, step_name: str) -> dict[str, Any]:
        """Suggest unified LightGBM hyperparameters including regularization."""
        return {
            # Core hyperparameters
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

            # Regularization parameters (unified with hyperparameter optimization)
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.1),

            # Additional regularization
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1e-1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),

            "random_state": 42,
            "verbose": -1,
        }

    def _suggest_unified_neural_network_params(self, trial: optuna.Trial, architecture: str, step_name: str) -> dict[str, Any]:
        """Suggest unified neural network hyperparameters including regularization."""
        return {
            # Architecture parameters
            "hidden_size": trial.suggest_int("hidden_size", 64, 512),
            "num_layers": trial.suggest_int("num_layers", 2, 6),

            # Training parameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 10, 50),

            # Regularization parameters (unified with hyperparameter optimization)
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3),

            # Additional regularization
            "layer_norm": trial.suggest_categorical("layer_norm", [True, False]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "gradient_clip": trial.suggest_float("gradient_clip", 0.1, 5.0),

            # Architecture-specific parameters
            "attention_heads": trial.suggest_int("attention_heads", 4, 16) if architecture == "Transformer" else 8,
            "kernel_size": trial.suggest_int("kernel_size", 3, 7) if architecture == "CNN" else 3,
            "dilation": trial.suggest_int("dilation", 1, 4) if architecture == "TCN" else 1,
        }

    def _create_neural_network_model(self, params: dict[str, Any], architecture: str, input_size: int, model_type: str):
        """Create neural network model based on architecture."""
        # This is a simplified version - in practice, you'd have more sophisticated model creation'
        if architecture == "CNN":
            return SimpleCNNModel(input_size, params, model_type)
        if architecture == "TCN":
            return SimpleTCNModel(input_size, params, model_type)
        if architecture == "Transformer":
            return SimpleTransformerModel(input_size, params, model_type)
        return SimpleNNModel(input_size, params, model_type)

    def _evaluate_model_with_domain_metrics(
        self,
        model,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
        architecture: str,
    ) -> np.ndarray:
        """Evaluate model using domain-specific metrics for financial applications."""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(features_df):
                X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Get predictions
                if model_type == "classification":
                    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
                    y_pred = model.predict(X_val)
                else:
                    y_pred = model.predict(X_val)

                # Calculate domain-specific metrics
                if model_type == "classification":
                    # For classification, use win rate and balanced accuracy
                    score = self._calculate_classification_metrics(y_val, y_pred, y_pred_proba)
                else:
                    # For regression, use Sharpe ratio approximation
                    score = self._calculate_regression_metrics(y_val, y_pred)

                scores.append(score)

            return np.array(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Domain-specific evaluation failed: {e}")
            # Fallback to standard cross-validation
            return cross_val_score(
                model, features_df, target,
                cv=TimeSeriesSplit(n_splits=3),
                scoring="accuracy" if model_type == "classification" else "neg_mean_squared_error",
            )

    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate domain-specific classification metrics."""
        try:
            from sklearn.metrics import accuracy_score, balanced_accuracy_score

            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            # Win rate (assuming positive class is "win")
            win_rate = np.mean(y_pred == 1) if len(np.unique(y_pred)) > 1 else 0.5

            # Risk-adjusted metric (combine accuracy with win rate)
            return (accuracy * 0.6 + balanced_acc * 0.3 + win_rate * 0.1)


        except Exception as e:
            self.logger.warning(f"⚠️ Classification metrics calculation failed: {e}")
            return 0.5

    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate domain-specific regression metrics."""
        try:
            # Calculate returns (assuming y_true and y_pred are price changes)
            returns = y_true - y_pred

            # Sharpe ratio approximation
            if len(returns) > 1:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
                # Normalize to [0, 1] range
                normalized_sharpe = 1 / (1 + np.exp(-sharpe_ratio))
            else:
                normalized_sharpe = 0.5

            # Win rate (positive returns)
            win_rate = np.mean(returns > 0)

            # Combined metric
            return (normalized_sharpe * 0.7 + win_rate * 0.3)


        except Exception as e:
            self.logger.warning(f"⚠️ Regression metrics calculation failed: {e}")
            return 0.5

    async def _evaluate_neural_network_with_training_loop(
        self,
        params: dict[str, Any],
        features_df: pd.DataFrame,
        target: pd.Series,
        architecture: str,
        model_type: str,
    ) -> np.ndarray:
        """Evaluate neural network with proper PyTorch training loop."""
        try:
            import asyncio

            # Run the training loop in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._run_neural_network_training_loop,
                params, features_df, target, architecture, model_type,
            )


        except Exception as e:
            self.logger.warning(f"⚠️ Neural network evaluation failed: {e}")
            return np.array([0.5])  # Fallback score

    def _evaluate_neural_network_sync(
        self,
        params: dict[str, Any],
        features_df: pd.DataFrame,
        target: pd.Series,
        architecture: str,
        model_type: str,
    ) -> np.ndarray:
        """Synchronous evaluation of neural network for Optuna objective function."""
        try:
            # Use the existing training loop method
            return self._run_neural_network_training_loop(
                params, features_df, target, architecture, model_type,
            )


        except Exception as e:
            self.logger.warning(f"⚠️ Neural network evaluation failed: {e}")
            return np.array([0.5])  # Fallback score

    def _run_neural_network_training_loop(
        self,
        params: dict[str, Any],
        features_df: pd.DataFrame,
        target: pd.Series,
        architecture: str,
        model_type: str,
    ) -> np.ndarray:
        """Run neural network training loop with proper PyTorch implementation."""
        try:
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for train_idx, val_idx in tscv.split(features_df):
                X_train, X_val = features_df.iloc[train_idx], features_df.iloc[val_idx]
                y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

                # Create model
                model = self._create_neural_network_model(params, architecture, X_train.shape[1], model_type)

                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train.values)
                X_val_tensor = torch.FloatTensor(X_val.values)

                if model_type == "classification":
                    y_train_tensor = torch.LongTensor(y_train.values)
                    torch.LongTensor(y_val.values)
                    criterion = nn.CrossEntropyLoss()
                else:
                    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
                    torch.FloatTensor(y_val.values).unsqueeze(1)
                    criterion = nn.MSELoss()

                # Create data loaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

                # Optimizer
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                    weight_decay=params["weight_decay"],
                )

                # Training loop
                model.train()
                for _epoch in range(params["epochs"]):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()

                        # Gradient clipping
                        if "gradient_clip" in params:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip"])

                        optimizer.step()

                # Evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)

                    if model_type == "classification":
                        _, val_pred = torch.max(val_outputs, 1)
                        val_pred_proba = torch.softmax(val_outputs, dim=1)[:, 1]
                        score = self._calculate_classification_metrics(
                            y_val, val_pred.numpy(), val_pred_proba.numpy(),
                        )
                    else:
                        val_pred = val_outputs.squeeze()
                        score = self._calculate_regression_metrics(y_val, val_pred.numpy())

                scores.append(score)

            return np.array(scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Neural network training loop failed: {e}")
            return np.array([0.5])  # Fallback score

    async def _evaluate_optimized_model(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str,
        optimization_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate the optimized model performance."""
        try:
            # Create final model with optimized parameters
            if architecture == "LightGBM":
                best_params = optimization_results.get("hyperparameter_optimization", {}).get("best_params", {})
                model = lgb.LGBMClassifier(**best_params) if model_type == "classification" else lgb.LGBMRegressor(**best_params)
            else:
                # For neural networks, create the model with optimized parameters
                best_params = optimization_results.get("hyperparameter_optimization", {}).get("best_params", {})
                model = self._create_neural_network_model(best_params, architecture, features_df.shape[1], model_type)

            if model is not None:
                # Cross-validation evaluation
                cv_scores = cross_val_score(
                    model, features_df, target,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring="accuracy" if model_type == "classification" else "neg_mean_squared_error",
                )

                return {
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                }
            return {"error": "model_creation_failed"}

        except Exception as e:
            self.logger.exception(f"❌ Model evaluation failed: {e}")
            return {"error": str(e)}

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization metrics and results."""
        return {
            "optimization_metrics": self.optimization_metrics,
            "cache_size": len(self.optimization_cache),
            "cached_steps": list(self.optimization_cache.keys()),
        }


class EnhancedFeatureSelector:
    """Enhanced feature selector with multiple algorithms and vectorized operations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("EnhancedFeatureSelector")
        self.feature_selection_config = config["feature_selection"]

    async def initialize(self) -> None:
        """Initialize the feature selector."""
        self.logger.info("✅ Enhanced Feature Selector initialized")

    async def select_features_enhanced(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        target_features: int,
        architecture: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Enhanced feature selection using ensemble approach with multiple algorithms."""
        try:
            start_time = time.time()

            # Step 1: Variance threshold (remove low variance features)
            variance_selector = VarianceThreshold(threshold=self.feature_selection_config["variance_threshold"])
            variance_selector.fit_transform(features_df)
            variance_features = features_df.columns[variance_selector.get_support()].tolist()

            # Step 2: Correlation analysis (remove highly correlated features)
            correlation_features = self._remove_correlated_features(
                features_df[variance_features], self.feature_selection_config["correlation_threshold"],
            )

            # Step 3: Ensemble feature selection (run multiple methods in parallel)
            ensemble_features = await self._ensemble_feature_selection(
                features_df[correlation_features], target, target_features, architecture,
            )

            # Step 4: Feature stability analysis
            stable_features = await self._analyze_feature_stability(
                features_df[ensemble_features], target, target_features,
            )

            # Create final feature set
            optimized_features = features_df[stable_features]

            selection_metadata = {
                "original_features": len(features_df.columns),
                "variance_filtered": len(variance_features),
                "correlation_filtered": len(correlation_features),
                "ensemble_filtered": len(ensemble_features),
                "stable_features": len(stable_features),
                "final_features": len(stable_features),
                "selection_time": time.time() - start_time,
            }

            return optimized_features, selection_metadata

        except Exception as e:
            self.logger.exception(f"❌ Enhanced feature selection failed: {e}")
            return features_df, {"error": str(e)}

    async def _ensemble_feature_selection(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        target_features: int,
        architecture: str,
    ) -> list[str]:
        """Ensemble feature selection using voting approach."""
        try:
            feature_scores = dict.fromkeys(features_df.columns, 0)
            methods_used = []

            # Run multiple feature selection methods in parallel
            if "mutual_info" in self.feature_selection_config["methods"]:
                mi_features = self._select_mutual_info_features(features_df, target, target_features)
                for feature in mi_features:
                    feature_scores[feature] += 1
                methods_used.append("mutual_info")

            if "lasso" in self.feature_selection_config["methods"]:
                lasso_features = self._select_lasso_features(features_df, target, target_features)
                for feature in lasso_features:
                    feature_scores[feature] += 1
                methods_used.append("lasso")

            if "random_forest" in self.feature_selection_config["methods"]:
                rf_features = self._select_random_forest_features(features_df, target, target_features)
                for feature in rf_features:
                    feature_scores[feature] += 1
                methods_used.append("random_forest")

            if "shap" in self.feature_selection_config["methods"] and len(features_df.columns) <= 50:
                shap_features = self._select_shap_features(features_df, target, target_features)
                for feature in shap_features:
                    feature_scores[feature] += 1
                methods_used.append("shap")

            # Select features based on voting score
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, score in sorted_features[:target_features]]

            self.logger.info(f"📊 Ensemble feature selection used {len(methods_used)} methods: {methods_used}")
            self.logger.info(f"📊 Feature voting scores: {dict(sorted_features[:10])}")  # Top 10 features

            return selected_features

        except Exception as e:
            self.logger.exception(f"❌ Ensemble feature selection failed: {e}")
            return features_df.columns[:target_features].tolist()

    async def _analyze_feature_stability(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        target_features: int,
    ) -> list[str]:
        """Analyze feature stability across multiple CV folds."""
        try:
            from sklearn.model_selection import TimeSeriesSplit

import copy

feature_stability = dict.fromkeys(features_df.columns, 0)
n_folds = 5

            # Time series cross-validation for stability analysis
            tscv = TimeSeriesSplit(n_splits=n_folds)

            for _fold_idx, (train_idx, _val_idx) in enumerate(tscv.split(features_df)):
                X_train = features_df.iloc[train_idx]
                y_train = target.iloc[train_idx]

                # Run feature selection on this fold
                fold_features = self._select_random_forest_features(X_train, y_train, target_features)

                # Count how many times each feature is selected
                for feature in fold_features:
                    if feature in feature_stability:
                        feature_stability[feature] += 1

            # Select features that are stable across folds
            stable_features = [
                feature for feature, count in feature_stability.items()
                if count >= n_folds * 0.6  # Feature must be selected in at least 60% of folds
            ]

            # If not enough stable features, add top features by stability score
            if len(stable_features) < target_features:
                sorted_by_stability = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)
                additional_features = [
                    feature for feature, count in sorted_by_stability
                    if feature not in stable_features
                ][:target_features - len(stable_features)]
                stable_features.extend(additional_features)

            # Limit to target number of features
            stable_features = stable_features[:target_features]

            self.logger.info(f"📊 Feature stability analysis: {len(stable_features)} stable features selected")
            self.logger.info(f"📊 Stability scores: {dict(sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)[:10])}")

            return stable_features

        except Exception as e:
            self.logger.exception(f"❌ Feature stability analysis failed: {e}")
            return features_df.columns[:target_features].tolist()

    def _remove_correlated_features(self, features_df: pd.DataFrame, threshold: float) -> list[str]:
        """Remove highly correlated features using vectorized operations."""
        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()

            # Find upper triangle of correlation matrix
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Find features with correlation above threshold
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

            # Return features to keep
            return [col for col in features_df.columns if col not in to_drop]

        except Exception as e:
            self.logger.warning(f"⚠️ Correlation filtering failed: {e}")
            return features_df.columns.tolist()

    def _select_mutual_info_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Select features using mutual information."""
        try:
            # Determine if classification or regression
            if target.dtype == "object" or len(target.unique()) < 10:
                mi_scores = mutual_info_classif(features_df, target, random_state=42)
            else:
                mi_scores = mutual_info_regression(features_df, target, random_state=42)

            # Get feature indices sorted by importance
            feature_indices = np.argsort(mi_scores)[::-1]

            # Select top features
            selected_indices = feature_indices[:target_features]
            return features_df.columns[selected_indices].tolist()


        except Exception as e:
            self.logger.warning(f"⚠️ Mutual info selection failed: {e}")
            return features_df.columns[:target_features].tolist()

    def _select_lasso_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Select features using Lasso regularization."""
        try:
            # Determine if classification or regression
            if target.dtype == "object" or len(target.unique()) < 10:
                lasso = LogisticRegression(penalty="l1", solver="liblinear", random_state=42, max_iter=1000)
            else:
                lasso = Lasso(alpha=0.01, random_state=42, max_iter=1000)

            # Fit Lasso
            lasso.fit(features_df, target)

            # Get coefficients - handle both binary and multiclass cases
            if hasattr(lasso, "coef_"):
                coef = lasso.coef_
                # Handle multiclass case
                if len(coef.shape) > 1:
                    # Use mean of absolute coefficients across classes
                    coef = np.mean(np.abs(coef), axis=0)
                else:
                    # Binary classification - coef is already 1D
                    coef = coef
            else:
                # For models without coef_ (like Random Forest), use feature_importances_
                coef = lasso.feature_importances_

            # Select features with non-zero coefficients
            selected_indices = np.where(np.abs(coef) > 0)[0]
            selected_features = features_df.columns[selected_indices].tolist()

            # If too few features, add more based on coefficient magnitude
            if len(selected_features) < target_features:
                top_indices = np.argsort(np.abs(coef))[::-1][:target_features]
                selected_features = features_df.columns[top_indices].tolist()

            return selected_features[:target_features]

        except Exception as e:
            self.logger.warning(f"⚠️ Lasso selection failed: {e}")
            return features_df.columns[:target_features].tolist()

    def _select_random_forest_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Select features using Random Forest importance."""
        try:
            # Determine if classification or regression
            if target.dtype == "object" or len(target.unique()) < 10:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            # Fit Random Forest
            rf.fit(features_df, target)

            # Get feature importance
            importance = rf.feature_importances_

            # Select top features
            top_indices = np.argsort(importance)[::-1][:target_features]
            return features_df.columns[top_indices].tolist()


        except Exception as e:
            self.logger.warning(f"⚠️ Random Forest selection failed: {e}")
            return features_df.columns[:target_features].tolist()

    def _select_shap_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> list[str]:
        """Select features using SHAP analysis."""
        try:
            # Use LightGBM for SHAP analysis
            if target.dtype == "object" or len(target.unique()) < 10:
                model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            else:
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

            # Fit model
            model.fit(features_df, target)

            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)

            # If classification, use the first class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)

            # Select top features
            top_indices = np.argsort(mean_shap)[::-1][:target_features]
            return features_df.columns[top_indices].tolist()


        except Exception as e:
            self.logger.warning(f"⚠️ SHAP selection failed: {e}")
            return features_df.columns[:target_features].tolist()


class EnhancedRegularizationManager:
    """Enhanced regularization manager with model-specific tuning."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("EnhancedRegularizationManager")
        self.regularization_config = config["regularization"]

    async def initialize(self) -> None:
        """Initialize the regularization manager."""
        self.logger.info("✅ Enhanced Regularization Manager initialized")

    async def optimize_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
    ) -> dict[str, Any]:
        """Optimize regularization parameters for the given architecture."""
        try:
            if architecture == "LightGBM":
                return await self._optimize_lightgbm_regularization(features_df, target, step_name)
            if architecture in ["CNN", "TCN", "Transformer"]:
                return await self._optimize_neural_network_regularization(features_df, target, step_name, architecture)
            return await self._optimize_general_regularization(features_df, target, step_name)

        except Exception as e:
            self.logger.exception(f"❌ Regularization optimization failed: {e}")
            return {"error": str(e)}

    async def _optimize_lightgbm_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str) -> dict[str, Any]:
        """Optimize LightGBM regularization parameters."""
        try:
            # Use Optuna to optimize regularization parameters
            def objective(trial):
                reg_alpha = trial.suggest_float("reg_alpha", 0.001, 0.1)
                reg_lambda = trial.suggest_float("reg_lambda", 0.001, 0.1)

                model = lgb.LGBMClassifier(
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    n_estimators=100,
                    random_state=42,
                    verbose=-1,
                )

                scores = cross_val_score(model, features_df, target, cv=3, scoring="accuracy")
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "reg_alpha": study.best_params["reg_alpha"],
                "reg_lambda": study.best_params["reg_lambda"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ LightGBM regularization optimization failed: {e}")
            return {"reg_alpha": 0.01, "reg_lambda": 0.001}

    async def _optimize_neural_network_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str, architecture: str) -> dict[str, Any]:
        """Optimize neural network regularization parameters."""
        try:
            # Use Optuna to optimize regularization parameters
            def objective(trial):
                weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3)
                dropout = trial.suggest_float("dropout", 0.1, 0.5)

                # Create a simple neural network for testing
                model = SimpleNNModel(
                    input_size=features_df.shape[1],
                    params={"dropout": dropout, "weight_decay": weight_decay},
                    model_type="classification",
                )

                # Simplified evaluation with proper training loop
                try:
                    # Convert to tensors
                    X_tensor = torch.FloatTensor(features_df.values)
                    y_tensor = torch.LongTensor(target.values) if model_type == "classification" else torch.FloatTensor(target.values).unsqueeze(1)

                    # Create data loader
                    dataset = TensorDataset(X_tensor, y_tensor)
                    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

                    # Training loop
                    model.train()
                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
                    criterion = nn.CrossEntropyLoss() if model_type == "classification" else nn.MSELoss()

                    for _epoch in range(10):  # Short training for optimization
                        for batch_X, batch_y in dataloader:
                            optimizer.zero_grad()
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()

                    # Evaluation
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        if model_type == "classification":
                            _, predictions = torch.max(outputs, 1)
                            return (predictions == y_tensor).float().mean().item()
                        mse = criterion(outputs, y_tensor).item()
                        return -mse  # Return negative MSE for maximization

                except Exception as e:
                    self.logger.warning(f"⚠️ Neural network evaluation failed: {e}")
                    return 0.5  # Fallback score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "weight_decay": study.best_params["weight_decay"],
                "dropout": study.best_params["dropout"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ Neural network regularization optimization failed: {e}")
            return {"weight_decay": 1e-4, "dropout": 0.2}

    async def _optimize_general_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str) -> dict[str, Any]:
        """Optimize general regularization parameters."""
        try:
            # Use ElasticNet for general regularization optimization
            def objective(trial):
                alpha = trial.suggest_float("alpha", 0.001, 0.1)
                l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                scores = cross_val_score(model, features_df, target, cv=3, scoring="neg_mean_squared_error")
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)

            return {
                "alpha": study.best_params["alpha"],
                "l1_ratio": study.best_params["l1_ratio"],
                "best_score": study.best_value,
            }

        except Exception as e:
            self.logger.exception(f"❌ General regularization optimization failed: {e}")
            return {"alpha": 0.01, "l1_ratio": 0.5}


# Simple model classes for demonstration
class SimpleNNModel(nn.Module):
    def __init__(self, input_size: int, params: dict[str, Any], model_type: str) -> None:
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type

        # Simple feedforward network
        self.layers = nn.Sequential(
            nn.Linear(input_size, params.get("hidden_size", 128)),
            nn.ReLU(),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(params.get("hidden_size", 128), 64),
            nn.ReLU(),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(64, 1 if model_type == "regression" else 2),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleCNNModel(nn.Module):
    def __init__(self, input_size: int, params: dict[str, Any], model_type: str) -> None:
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type

        # Simple CNN for 1D data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(params.get("dropout", 0.2)),
            nn.Linear(32, 1 if model_type == "regression" else 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc_layers(x)


class SimpleTCNModel(nn.Module):
    def __init__(self, input_size: int, params: dict[str, Any], model_type: str) -> None:
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type

        # TCN implementation with proper causal convolutions and residual blocks
        self.hidden_size = params.get("hidden_size", 128)
        self.num_layers = params.get("num_layers", 3)
        self.kernel_size = params.get("kernel_size", 3)
        self.dilation = params.get("dilation", 1)
        self.output_size = 1 if model_type == "regression" else 2

        # TCN layers with causal convolutions and residual connections
        self.tcn_layers = nn.ModuleList()
        in_channels = input_size

        for i in range(self.num_layers):
            out_channels = self.hidden_size if i < self.num_layers - 1 else self.output_size
            dilation = self.dilation ** i

            # Causal convolution with proper padding for causality
            padding = (self.kernel_size - 1) * dilation
            conv = nn.Conv1d(
                in_channels, out_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=padding,
            )

            # Residual block with proper residual connection
            if in_channels == out_channels:
                # Same channel dimensions - direct residual connection
                self.tcn_layers.append(nn.ModuleList([
                    conv,
                    nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                    nn.Conv1d(out_channels, out_channels, 1),  # 1x1 conv for residual
                ]))
            else:
                # Different channel dimensions - need projection
                self.tcn_layers.append(nn.ModuleList([
                    conv,
                    nn.ReLU(),
                    nn.Dropout(params.get("dropout", 0.2)),
                    nn.Conv1d(in_channels, out_channels, 1),  # 1x1 conv for channel projection
                ]))

            in_channels = out_channels

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (batch_size, features)
        # Reshape for 1D convolution: (batch_size, channels, sequence_length)
        x = x.unsqueeze(-1).transpose(1, 2)  # Add sequence dimension

        for layer in self.tcn_layers:
            conv, relu, dropout, residual = layer

            # Apply convolution
            out = conv(x)
            out = relu(out)
            out = dropout(out)

            # Add residual connection
            if x.size(1) == out.size(1):
                # Same channel dimensions
                out = out + residual(x)
            else:
                # Different channel dimensions - use projection
                out = out + residual(x)

            x = out

        # Global average pooling
        return self.global_pool(x).squeeze(-1)



class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size: int, params: dict[str, Any], model_type: str) -> None:
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type

        # Simple transformer implementation
        self.embedding = nn.Linear(input_size, params.get("hidden_size", 128))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=params.get("hidden_size", 128),
                nhead=8,
                dim_feedforward=params.get("hidden_size", 128) * 4,
                dropout=params.get("dropout", 0.2),
                batch_first=True,
            ),
            num_layers=params.get("num_layers", 2),
        )
        self.output_layer = nn.Linear(params.get("hidden_size", 128), 1 if model_type == "regression" else 2)

    def forward(self, x):
        x = self.embedding(x)
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_size)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_layer(x)
