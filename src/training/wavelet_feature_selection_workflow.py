"""Wavelet Feature Selection Workflow.

This module implements a comprehensive workflow using the two-model strategy:
1. Discovery Model: Trained on full feature set to identify winning features
2. Production Model: Trained on lean feature set for live deployment

The workflow:
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

            # Initialize feature engineering
            await self._initialize_feature_engineering()

            # Initialize models
            await self._initialize_models()

            self.logger.info("Wavelet Feature Selection Workflow initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing workflow: {e}")
            return False


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


            # Step 1: Precompute wavelet features
            wavelet_features = await self._precompute_wavelet_features(data)

            # Step 2: Perform advanced feature engineering
            engineered_features = await self._perform_feature_engineering(data, wavelet_features)

            # Step 3: Train discovery model
            discovery_performance = await self._train_discovery_model(engineered_features, labels)


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

