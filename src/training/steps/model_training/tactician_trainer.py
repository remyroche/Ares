"""
Tactician Trainer

This module provides training utilities for tactician models with comprehensive
type safety and error handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

# Import ML libraries with fallback support
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning

# Import our custom types
from ..types import (
    StepConfig, ModelTrainingResult, ValidationResult, MetricsDict,
    DataFrameType, SeriesType, SignalType, ModelType, PathType,
    ModelTrainingError, ValidationError, ConfigurationError, ArtifactError,
    validate_config, create_error_result, create_success_result,
    is_dataframe, is_series
)

logger = logging.getLogger(__name__)

class TacticianTrainer:
    """
    Trainer for tactician models with support for multiple algorithms.
    
    Provides comprehensive model training with type safety and error handling.
    """

    def __init__(self, config: Optional[StepConfig] = None):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid
            ImportError: If required ML libraries are not available
        """
        try:
            # Validate configuration
            self.config = validate_config(config) if config else {}
            
            # Initialize training state
            self.models: Dict[str, Any] = {}
            self.training_results: Dict[str, ModelTrainingResult] = {}
            
            # Check ML library availability
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for model training but not available")
            
            tprint_success("✅ TacticianTrainer initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianTrainer: {e}")
            raise ConfigurationError(f"Trainer initialization failed: {e}") from e

    async def train_base_model(
        self,
        X: DataFrameType,
        y: SeriesType,
        model_type: str,
        signal_type: SignalType = 'long'
    ) -> ModelTrainingResult:
        """
        Train a base model for the given signal type with comprehensive error handling.

        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            signal_type: Type of signal ('long' or 'short')

        Returns:
            ModelTrainingResult containing training results

        Raises:
            ModelTrainingError: If model training fails
            ValidationError: If input data is invalid
            ConfigurationError: If model type is invalid
        """
        try:
            # Validate input data
            if not is_dataframe(X):
                raise ValidationError("X must be a pandas DataFrame")
            if not is_series(y):
                raise ValidationError("y must be a pandas Series")
            
            if X.empty or y.empty:
                raise ValidationError("Input data cannot be empty")
            
            if len(X) != len(y):
                raise ValidationError(f"X and y must have the same length: X={len(X)}, y={len(y)}")
            
            if signal_type not in ['long', 'short']:
                raise ValidationError(f"Invalid signal_type: {signal_type}. Must be 'long' or 'short'")

            tprint_info(f"🔍 Training {model_type} model for {signal_type} signals...")

            # Validate model type
            valid_model_types = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
            if model_type not in valid_model_types:
                raise ConfigurationError(f"Invalid model_type: {model_type}. Must be one of {valid_model_types}")

            # Handle missing values
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)

            # Validate cleaned data
            if X_clean.empty or y_clean.empty:
                raise ValidationError("Data became empty after cleaning missing values")

            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
                )
                tprint_info(f"📊 Data split: train={len(X_train)}, test={len(X_test)}")
            except Exception as e:
                raise ModelTrainingError(f"Data splitting failed: {e}") from e

            # Train model based on type
            try:
                if model_type == 'random_forest':
                    model = self._train_random_forest(X_train, y_train)
                elif model_type == 'xgboost':
                    model = self._train_xgboost(X_train, y_train)
                elif model_type == 'lightgbm':
                    model = self._train_lightgbm(X_train, y_train)
                elif model_type == 'catboost':
                    model = self._train_catboost(X_train, y_train)
                else:
                    raise ConfigurationError(f"Unknown model type: {model_type}")
                
                tprint_success(f"✅ {model_type} model trained successfully")
            except Exception as e:
                raise ModelTrainingError(f"Model training failed: {e}") from e

            # Evaluate model
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                tprint_success(f"📊 Model evaluation: accuracy={accuracy:.3f}, f1={f1:.3f}")
            except Exception as e:
                raise ModelTrainingError(f"Model evaluation failed: {e}") from e

            # Store model
            model_key = f"{model_type}_{signal_type}"
            self.models[model_key] = model

            # Create training metrics
            training_metrics: MetricsDict = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_clean.shape[1]
            }

            tprint_success(f"✅ {model_type} model trained with accuracy: {accuracy:.3f}")

            return create_success_result(
                model=model,
                accuracy=accuracy,
                model_type=model_type,
                signal_type=signal_type,
                model_key=model_key,
                training_metrics=training_metrics
            )

        except ValidationError:
            raise
        except ModelTrainingError:
            raise
        except ConfigurationError:
            raise
        except Exception as e:
            tprint_error(f"❌ Unexpected error in model training: {e}")
            raise ModelTrainingError(f"Model training failed: {e}") from e

    def _train_random_forest(self, X: DataFrameType, y: SeriesType) -> Any:
        """Train Random Forest model with error handling."""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn not available")

            tprint_info("🌲 Training Random Forest model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            tprint_success("✅ Random Forest model trained")
            return model
        except Exception as e:
            raise ModelTrainingError(f"Random Forest training failed: {e}") from e

    def _train_xgboost(self, X: DataFrameType, y: SeriesType) -> Any:
        """Train XGBoost model with error handling."""
        try:
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")

            tprint_info("🚀 Training XGBoost model...")
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            tprint_success("✅ XGBoost model trained")
            return model
        except Exception as e:
            raise ModelTrainingError(f"XGBoost training failed: {e}") from e

    def _train_lightgbm(self, X: DataFrameType, y: SeriesType) -> Any:
        """Train LightGBM model with error handling."""
        try:
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")

            tprint_info("💡 Training LightGBM model...")
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X, y)
            tprint_success("✅ LightGBM model trained")
            return model
        except Exception as e:
            raise ModelTrainingError(f"LightGBM training failed: {e}") from e

    def _train_catboost(self, X: DataFrameType, y: SeriesType) -> Any:
        """Train CatBoost model with error handling."""
        try:
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")

            tprint_info("🐱 Training CatBoost model...")
            model = CatBoostClassifier(
                iterations=100,
                depth=6,
                random_state=42,
                verbose=False
            )
            model.fit(X, y)
            tprint_success("✅ CatBoost model trained")
            return model
        except Exception as e:
            raise ModelTrainingError(f"CatBoost training failed: {e}") from e

    def save_models(self, output_dir: PathType) -> Dict[str, Any]:
        """
        Save trained models to disk with comprehensive error handling.

        Args:
            output_dir: Directory to save models

        Returns:
            Dictionary with save results

        Raises:
            ArtifactError: If model saving fails
            ValidationError: If output directory is invalid
        """
        try:
            # Validate output directory
            if not output_dir:
                raise ValidationError("Output directory cannot be empty")
            
            output_path = Path(output_dir)
            
            # Create directory if it doesn't exist
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                tprint_info(f"📁 Created output directory: {output_path}")
            except Exception as e:
                raise ArtifactError(f"Failed to create output directory {output_path}: {e}") from e

            if not self.models:
                tprint_warning("⚠️ No models to save")
                return create_success_result(
                    saved_models={},
                    output_dir=str(output_path)
                )

            saved_models = {}
            for model_key, model in self.models.items():
                try:
                    model_path = output_path / f"{model_key}.joblib"
                    joblib.dump(model, model_path)
                    saved_models[model_key] = str(model_path)
                    tprint_info(f"💾 Saved model: {model_key} -> {model_path}")
                except Exception as e:
                    raise ArtifactError(f"Failed to save model {model_key}: {e}") from e

            tprint_success(f"✅ Saved {len(saved_models)} models to {output_dir}")

            return create_success_result(
                saved_models=saved_models,
                output_dir=str(output_path)
            )

        except ValidationError:
            raise
        except ArtifactError:
            raise
        except Exception as e:
            tprint_error(f"❌ Unexpected error saving models: {e}")
            raise ArtifactError(f"Model saving failed: {e}") from e
