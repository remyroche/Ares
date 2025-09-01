"""Wavelet Feature Selection Workflow.

This module implements a comprehensive workflow using the two-model strategy:
1. Discovery Model: Trained on full feature set to identify winning features
2. Production Model: Trained on lean feature set for live deployment

The workflow:
1. Run full extensive wavelet analysis (as in backtesting/training)
2. Build Discovery Model using the rich feature set
3. Perform feature selection using permutation importance and SHAP
4. Identify the most important features
5. Create lean dataset with only winning features
6. Train Production Model on lean dataset
7. Create optimized live trading configurations
"""

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from src.training.steps.precompute_wavelet_features import WaveletFeaturePrecomputer
from src.training.steps.vectorized_advanced_feature_engineering import (
    VectorizedAdvancedFeatureEngineering,
)
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error, failed, initialization_error,
)


@dataclass
class FeatureImportanceResult:
    """Container for feature importance analysis results."""

    feature_name: str
    permutation_importance: float
    shap_importance: float
    combined_score: float
    feature_type: str  # 'wavelet', 'technical', 'other'
    computation_cost: float  # Estimated computation time in ms


class WaveletFeatureSelectionWorkflow:
    """Comprehensive workflow for wavelet feature selection using two-model strategy.

    This workflow:
    1. Runs full wavelet analysis with all features
    2. Builds Discovery Model on the rich feature set
    3. Performs feature selection using multiple methods
    4. Identifies the most important features
    5. Creates lean dataset with only winning features
    6. Trains Production Model on lean dataset
    7. Creates optimized live trading configurations
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("WaveletFeatureSelectionWorkflow")

        # Workflow configuration
        self.workflow_config = config.get("wavelet_feature_selection", {})
        self.output_dir = Path(
            self.workflow_config.get("output_dir", "data/wavelet_feature_selection"),
        )
        self.model_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.configs_dir = self.output_dir / "configs"

        # Feature selection parameters
        self.top_n_features = self.workflow_config.get("top_n_features", 20)
        self.min_importance_threshold = self.workflow_config.get(
            "min_importance_threshold",
            0.01)
        self.max_computation_time = self.workflow_config.get(
            "max_computation_time", 0.1,
        )  # 100ms

        # ML model parameters
        self.test_size = self.workflow_config.get("test_size", 0.2)
        self.random_state = self.workflow_config.get("random_state", 42)
        self.cv_folds = self.workflow_config.get("cv_folds", 5)

        # Model configurations
        self.discovery_model_config = self.workflow_config.get(
            "discovery_model", {})
        self.production_model_config = self.workflow_config.get(
            "production_model", {})

        # Initialize components
        self.feature_precomputer = None
        self.feature_engineering = None
        self.discovery_model = None
        self.production_model = None

        # Results storage
        self.feature_importance_results: List[FeatureImportanceResult] = []
        self.selected_features: List[str] = []
        self.discovery_model_performance: Dict[str, float] = {}
        self.production_model_performance: Dict[str, float] = {}

    @handle_errors(
        exceptions=(Exception,), default_return=False,
        context="wavelet feature selection workflow initialization")
    async def initialize(self) -> bool:
        """Initialize the wavelet feature selection workflow."""
        try:
            self.logger.info("Initializing Wavelet Feature Selection Workflow...")

            # Create output directories
            self._create_output_directories()

            # Initialize feature precomputer
            await self._initialize_feature_precomputer()

            # Initialize feature engineering
            await self._initialize_feature_engineering()

            # Initialize models
            await self._initialize_models()

            self.logger.info("Wavelet Feature Selection Workflow initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing workflow: {e}")
            return False

    def _create_output_directories(self) -> None:
        """Create necessary output directories."""
        try:
            self.logger.debug("Creating output directories...")

            # Create main output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.configs_dir.mkdir(parents=True, exist_ok=True)

            self.logger.debug("Output directories created successfully")

        except Exception as e:
            self.logger.error(f"Error creating output directories: {e}")
            raise

    async def _initialize_feature_precomputer(self) -> None:
        """Initialize the wavelet feature precomputer."""
        try:
            self.logger.debug("Initializing feature precomputer...")

            # Initialize wavelet feature precomputer
            self.feature_precomputer = WaveletFeaturePrecomputer(self.config)
            await self.feature_precomputer.initialize()

            self.logger.debug("Feature precomputer initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing feature precomputer: {e}")
            raise

    async def _initialize_feature_engineering(self) -> None:
        """Initialize the feature engineering component."""
        try:
            self.logger.debug("Initializing feature engineering...")

            # Initialize vectorized advanced feature engineering
            self.feature_engineering = VectorizedAdvancedFeatureEngineering(self.config)
            await self.feature_engineering.initialize()

            self.logger.debug("Feature engineering initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing feature engineering: {e}")
            raise

    async def _initialize_models(self) -> None:
        """Initialize the discovery and production models."""
        try:
            self.logger.debug("Initializing models...")

            # Initialize discovery model
            self.discovery_model = self._create_discovery_model()

            # Initialize production model
            self.production_model = self._create_production_model()

            self.logger.debug("Models initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def _create_discovery_model(self) -> Any:
        """Create the discovery model."""
        try:
            model_type = self.discovery_model_config.get("model_type", "gradient_boosting")
            
            if model_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=self.discovery_model_config.get("n_estimators", 100),
                    learning_rate=self.discovery_model_config.get("learning_rate", 0.1),
                    max_depth=self.discovery_model_config.get("max_depth", 3),
                    random_state=self.random_state
                )
            elif model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=self.discovery_model_config.get("n_estimators", 100),
                    max_depth=self.discovery_model_config.get("max_depth", 10),
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error creating discovery model: {e}")
            raise

    def _create_production_model(self) -> Any:
        """Create the production model."""
        try:
            model_type = self.production_model_config.get("model_type", "gradient_boosting")
            
            if model_type == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=self.production_model_config.get("n_estimators", 50),
                    learning_rate=self.production_model_config.get("learning_rate", 0.1),
                    max_depth=self.production_model_config.get("max_depth", 3),
                    random_state=self.random_state
                )
            elif model_type == "random_forest":
                return RandomForestClassifier(
                    n_estimators=self.production_model_config.get("n_estimators", 50),
                    max_depth=self.production_model_config.get("max_depth", 8),
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error creating production model: {e}")
            raise

    @handle_errors(
        exceptions=(Exception,), default_return=False,
        context="wavelet feature selection execution")
    async def execute_workflow(self, data: pd.DataFrame, labels: pd.Series) -> bool:
        """Execute the complete wavelet feature selection workflow."""
        try:
            self.logger.info("Starting wavelet feature selection workflow...")

            # Step 1: Precompute wavelet features
            wavelet_features = await self._precompute_wavelet_features(data)

            # Step 2: Perform advanced feature engineering
            engineered_features = await self._perform_feature_engineering(data, wavelet_features)

            # Step 3: Train discovery model
            discovery_performance = await self._train_discovery_model(engineered_features, labels)

            # Step 4: Perform feature selection
            selected_features = await self._perform_feature_selection(engineered_features, labels)

            # Step 5: Create lean dataset
            lean_dataset = await self._create_lean_dataset(engineered_features, selected_features)

            # Step 6: Train production model
            production_performance = await self._train_production_model(lean_dataset, labels)

            # Step 7: Save results and configurations
            await self._save_results_and_configs(selected_features, discovery_performance, production_performance)

            self.logger.info("Wavelet feature selection workflow completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error executing workflow: {e}")
            return False

    async def _precompute_wavelet_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Precompute wavelet features for the dataset."""
        try:
            self.logger.info("Precomputing wavelet features...")

            if not self.feature_precomputer:
                raise ValueError("Feature precomputer not initialized")

            # Precompute wavelet features
            wavelet_features = await self.feature_precomputer.precompute_features(data)

            self.logger.info(f"Precomputed {len(wavelet_features.columns)} wavelet features")
            return wavelet_features

        except Exception as e:
            self.logger.error(f"Error precomputing wavelet features: {e}")
            raise

    async def _perform_feature_engineering(self, data: pd.DataFrame, wavelet_features: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced feature engineering."""
        try:
            self.logger.info("Performing advanced feature engineering...")

            if not self.feature_engineering:
                raise ValueError("Feature engineering not initialized")

            # Combine original data with wavelet features
            combined_data = pd.concat([data, wavelet_features], axis=1)

            # Perform advanced feature engineering
            engineered_features = await self.feature_engineering.engineer_features(combined_data)

            self.logger.info(f"Engineered {len(engineered_features.columns)} features")
            return engineered_features

        except Exception as e:
            self.logger.error(f"Error performing feature engineering: {e}")
            raise

    async def _train_discovery_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """Train the discovery model on the full feature set."""
        try:
            self.logger.info("Training discovery model...")

            if not self.discovery_model:
                raise ValueError("Discovery model not initialized")

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=self.test_size, random_state=self.random_state
            )

            # Train model
            self.discovery_model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.discovery_model.predict(X_test)
            y_pred_proba = self.discovery_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            performance = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted'),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }

            # Cross-validation
            cv_scores = cross_val_score(self.discovery_model, features, labels, cv=self.cv_folds, scoring='accuracy')
            performance["cv_accuracy_mean"] = cv_scores.mean()
            performance["cv_accuracy_std"] = cv_scores.std()

            self.discovery_model_performance = performance
            self.logger.info(f"Discovery model trained successfully. CV Accuracy: {performance['cv_accuracy_mean']:.3f} ± {performance['cv_accuracy_std']:.3f}")
            return performance

        except Exception as e:
            self.logger.error(f"Error training discovery model: {e}")
            raise

    async def _perform_feature_selection(self, features: pd.DataFrame, labels: pd.Series) -> List[str]:
        """Perform feature selection using multiple methods."""
        try:
            self.logger.info("Performing feature selection...")

            if not self.discovery_model:
                raise ValueError("Discovery model not trained")

            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.discovery_model, features, labels, 
                n_repeats=10, random_state=self.random_state
            )

            # Create feature importance results
            feature_importance_results = []
            for i, feature_name in enumerate(features.columns):
                importance_result = FeatureImportanceResult(
                    feature_name=feature_name,
                    permutation_importance=perm_importance.importances_mean[i],
                    shap_importance=0.0,  # Would need SHAP library for this
                    combined_score=perm_importance.importances_mean[i],
                    feature_type=self._classify_feature_type(feature_name),
                    computation_cost=self._estimate_computation_cost(feature_name)
                )
                feature_importance_results.append(importance_result)

            # Sort by combined score
            feature_importance_results.sort(key=lambda x: x.combined_score, reverse=True)

            # Filter by importance threshold and computation cost
            selected_features = []
            for result in feature_importance_results:
                if (result.combined_score >= self.min_importance_threshold and 
                    result.computation_cost <= self.max_computation_time):
                    selected_features.append(result.feature_name)
                    if len(selected_features) >= self.top_n_features:
                        break

            self.feature_importance_results = feature_importance_results
            self.selected_features = selected_features

            self.logger.info(f"Selected {len(selected_features)} features out of {len(features.columns)}")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error performing feature selection: {e}")
            raise

    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify the type of a feature."""
        try:
            if "wavelet" in feature_name.lower():
                return "wavelet"
            elif any(tech in feature_name.lower() for tech in ["rsi", "macd", "bollinger", "sma", "ema"]):
                return "technical"
            else:
                return "other"

        except Exception as e:
            self.logger.error(f"Error classifying feature type: {e}")
            return "other"

    def _estimate_computation_cost(self, feature_name: str) -> float:
        """Estimate the computation cost of a feature."""
        try:
            # Simple heuristic based on feature name
            if "wavelet" in feature_name.lower():
                return 0.05  # Wavelet features are computationally expensive
            elif "technical" in feature_name.lower():
                return 0.01  # Technical indicators are moderate
            else:
                return 0.001  # Basic features are cheap

        except Exception as e:
            self.logger.error(f"Error estimating computation cost: {e}")
            return 0.01

    async def _create_lean_dataset(self, features: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """Create lean dataset with only selected features."""
        try:
            self.logger.info("Creating lean dataset...")

            # Select only the chosen features
            lean_dataset = features[selected_features].copy()

            self.logger.info(f"Created lean dataset with {len(lean_dataset.columns)} features")
            return lean_dataset

        except Exception as e:
            self.logger.error(f"Error creating lean dataset: {e}")
            raise

    async def _train_production_model(self, features: pd.DataFrame, labels: pd.Series) -> Dict[str, float]:
        """Train the production model on the lean dataset."""
        try:
            self.logger.info("Training production model...")

            if not self.production_model:
                raise ValueError("Production model not initialized")

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=self.test_size, random_state=self.random_state
            )

            # Train model
            self.production_model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.production_model.predict(X_test)
            y_pred_proba = self.production_model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            performance = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1_score": f1_score(y_test, y_pred, average='weighted'),
                "roc_auc": roc_auc_score(y_test, y_pred_proba)
            }

            # Cross-validation
            cv_scores = cross_val_score(self.production_model, features, labels, cv=self.cv_folds, scoring='accuracy')
            performance["cv_accuracy_mean"] = cv_scores.mean()
            performance["cv_accuracy_std"] = cv_scores.std()

            self.production_model_performance = performance
            self.logger.info(f"Production model trained successfully. CV Accuracy: {performance['cv_accuracy_mean']:.3f} ± {performance['cv_accuracy_std']:.3f}")
            return performance

        except Exception as e:
            self.logger.error(f"Error training production model: {e}")
            raise

    async def _save_results_and_configs(self, selected_features: List[str], 
                                      discovery_performance: Dict[str, float],
                                      production_performance: Dict[str, float]) -> None:
        """Save results and create optimized configurations."""
        try:
            self.logger.info("Saving results and configurations...")

            # Save selected features
            features_file = self.results_dir / "selected_features.pkl"
            with open(features_file, 'wb') as f:
                pickle.dump(selected_features, f)

            # Save model performances
            performance_file = self.results_dir / "model_performances.yaml"
            performance_data = {
                "discovery_model": discovery_performance,
                "production_model": production_performance,
                "feature_selection_summary": {
                    "total_features_analyzed": len(self.feature_importance_results),
                    "selected_features_count": len(selected_features),
                    "importance_threshold": self.min_importance_threshold,
                    "computation_time_limit": self.max_computation_time
                }
            }
            
            with open(performance_file, 'w') as f:
                yaml.dump(performance_data, f, default_flow_style=False)

            # Save models
            discovery_model_file = self.model_dir / "discovery_model.pkl"
            production_model_file = self.model_dir / "production_model.pkl"
            
            with open(discovery_model_file, 'wb') as f:
                pickle.dump(self.discovery_model, f)
            
            with open(production_model_file, 'wb') as f:
                pickle.dump(self.production_model, f)

            # Create optimized live trading configuration
            await self._create_live_trading_config(selected_features, production_performance)

            self.logger.info("Results and configurations saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving results and configurations: {e}")
            raise

    async def _create_live_trading_config(self, selected_features: List[str], 
                                        production_performance: Dict[str, float]) -> None:
        """Create optimized live trading configuration."""
        try:
            self.logger.info("Creating live trading configuration...")

            config = {
                "model": {
                    "type": "production",
                    "file_path": str(self.model_dir / "production_model.pkl"),
                    "performance": production_performance
                },
                "features": {
                    "selected_features": selected_features,
                    "feature_count": len(selected_features),
                    "importance_threshold": self.min_importance_threshold,
                    "computation_time_limit": self.max_computation_time
                },
                "workflow": {
                    "feature_precomputation": True,
                    "feature_engineering": True,
                    "model_prediction": True
                },
                "optimization": {
                    "enabled": True,
                    "batch_size": 1000,
                    "prediction_threshold": 0.5
                }
            }

            config_file = self.configs_dir / "live_trading_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            self.logger.info("Live trading configuration created successfully")

        except Exception as e:
            self.logger.error(f"Error creating live trading configuration: {e}")
            raise

    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance analysis."""
        try:
            if not self.feature_importance_results:
                return {"message": "No feature importance results available"}

            # Calculate summary statistics
            wavelet_features = [f for f in self.feature_importance_results if f.feature_type == "wavelet"]
            technical_features = [f for f in self.feature_importance_results if f.feature_type == "technical"]
            other_features = [f for f in self.feature_importance_results if f.feature_type == "other"]

            summary = {
                "total_features_analyzed": len(self.feature_importance_results),
                "selected_features_count": len(self.selected_features),
                "feature_type_distribution": {
                    "wavelet": len(wavelet_features),
                    "technical": len(technical_features),
                    "other": len(other_features)
                },
                "top_features": [
                    {
                        "name": f.feature_name,
                        "importance": f.combined_score,
                        "type": f.feature_type,
                        "computation_cost": f.computation_cost
                    }
                    for f in self.feature_importance_results[:10]
                ],
                "selected_features": self.selected_features
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting feature importance summary: {e}")
            return {"error": str(e)}

    def get_model_performance_comparison(self) -> Dict[str, Any]:
        """Get comparison of discovery vs production model performance."""
        try:
            comparison = {
                "discovery_model": self.discovery_model_performance,
                "production_model": self.production_model_performance,
                "performance_difference": {}
            }

            # Calculate performance differences
            for metric in self.discovery_model_performance:
                if metric in self.production_model_performance:
                    diff = self.discovery_model_performance[metric] - self.production_model_performance[metric]
                    comparison["performance_difference"][metric] = diff

            return comparison

        except Exception as e:
            self.logger.error(f"Error getting model performance comparison: {e}")
            return {"error": str(e)}

    async def cleanup(self) -> None:
        """Cleanup workflow resources."""
        try:
            self.logger.info("Cleaning up Wavelet Feature Selection Workflow...")

            # Cleanup components
            if self.feature_precomputer:
                await self.feature_precomputer.cleanup()

            if self.feature_engineering:
                await self.feature_engineering.cleanup()

            # Clear results
            self.feature_importance_results.clear()
            self.selected_features.clear()
            self.discovery_model_performance.clear()
            self.production_model_performance.clear()

            self.logger.info("Wavelet Feature Selection Workflow cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
