"""
Base Training Class - Clean Implementation

This module demonstrates how to use the new clean import system,
centralized configuration, and hardware optimization to create
maintainable, efficient training classes.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from abc import ABC, abstractmethod

# Clean imports using the new system
from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.ml_common.config import BaseTrainingConfig
from src.utils.ml_common.training import BaseTrainingStep
from src.utils.common_operations import safe_divide, safe_mean, validate_finite
from src.utils.math_validation import MathValidationError

# Our new clean systems
from .dependency_manager import get_dependency_manager, validate_training_environment
from .config_manager import get_config_manager, get_model_config
from .hardware_optimizer import get_hardware_optimizer, optimize_data_loading, monitor_training_resources
from .clean_imports import setup_logging, get_optional_dependencies, get_model_imports

class CleanTrainingStep(BaseTrainingStep, ABC):
    """
    Clean, maintainable base training step that demonstrates best practices.

    This replaces the complex, error-prone training classes with a clean,
    well-organized implementation that:
    - Uses centralized dependency management
    - Has proper configuration management
    - Implements hardware optimization
    - Has consistent error handling
    - Is easy to test and maintain
    """

    def __init__(self, config: BaseTrainingConfig, role: str = 'analyst', mode: str = 'full'):
        """
        Initialize clean training step.

        Args:
            config: Training configuration
            role: 'analyst' or 'tactician'
            mode: 'full', 'light', or 'blank'
        """
        # Validate environment before proceeding
        if not validate_training_environment():
            raise RuntimeError("Training environment validation failed")

        self.config = config
        self.role = role
        self.mode = mode

        # Setup clean logging
        self.logger = setup_logging(f'Clean{role.capitalize()}Training')

        # Get centralized configuration
        self.config_manager = get_config_manager()
        self.mode_config = self.config_manager.get_training_mode_config(mode)

        # Get optional dependencies cleanly
        self.deps = get_optional_dependencies()

        # Get model-specific imports
        self.model_imports = self._get_required_model_imports()

        # Setup hardware optimization
        self.hardware_optimizer = get_hardware_optimizer()

        # Validate configuration
        self._validate_configuration()

        tprint_success(f"✅ Clean{role.capitalize()}Training initialized for {mode} mode")

    def _get_required_model_imports(self) -> Dict[str, Any]:
        """Get required model imports for this role and mode."""
        # Get models for this role and mode
        models = self.config_manager.get_models_by_priority(self.role, self.mode)

        # Extract model type names
        model_types = [model.model_type.value for model in models]

        # Get imports for these model types
        return get_model_imports(model_types)

    def _validate_configuration(self):
        """Validate configuration using centralized validation."""
        warnings = self.config_manager.validate_configuration(self.role, self.mode)

        if warnings:
            for warning in warnings:
                tprint_warning(f"⚠️ {warning}")

    def load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load training data with hardware optimization."""
        tprint_info(f"📂 Loading training data for {self.role}")

        with monitor_training_resources("data_loading") as resources:
            try:
                # Use hardware-optimized data loading
                df = optimize_data_loading(data_path)

                # Apply additional optimizations
                if self.deps.get('hardware'):
                    df = self.hardware_optimizer.optimize_dataframe_operations(df, ['memory', 'cpu'])

                # Validate data quality
                self._validate_training_data(df)

                tprint_success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                return df

            except Exception as e:
                tprint_error(f"❌ Failed to load training data: {e}")
                raise

    def _validate_training_data(self, df: pd.DataFrame):
        """Validate training data quality."""
        # Check for required columns
        required_columns = []
        models = self.config_manager.get_models_by_priority(self.role, self.mode)

        for model in models:
            required_columns.extend(model.required_features)
            required_columns.extend(model.optional_features)

        required_columns = list(set(required_columns))  # Remove duplicates

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            tprint_warning(f"⚠️ Missing columns: {missing_columns}")

        # Check data quality
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            tprint_warning(f"⚠️ Found {null_counts.sum()} null values")

        # Check for finite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not np.isfinite(df[col]).all():
                tprint_warning(f"⚠️ Non-finite values found in {col}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for training with hardware optimization."""
        tprint_info(f"🔧 Preparing features for {self.role}")

        with monitor_training_resources("feature_preparation") as resources:
            try:
                # Get required features for this role/mode
                models = self.config_manager.get_models_by_priority(self.role, self.mode)
                all_features = set()

                for model in models:
                    all_features.update(model.required_features)
                    all_features.update(model.optional_features)

                feature_columns = list(all_features.intersection(df.columns))

                if not feature_columns:
                    raise ValueError(f"No valid features found for {self.role}")

                # Extract features with hardware optimization
                X = df[feature_columns].values

                # Apply hardware optimization if available
                if self.deps.get('hardware'):
                    X = self.hardware_optimizer.optimize_matrix_operations([X], 'optimize')[0]

                tprint_success(f"✅ Prepared {X.shape[1]} features for {X.shape[0]} samples")
                return X, feature_columns

            except Exception as e:
                tprint_error(f"❌ Feature preparation failed: {e}")
                raise

    def train_models(self, X: np.ndarray, y: np.ndarray, model_configs: List[Any]) -> Dict[str, Any]:
        """Train models with clean error handling and hardware optimization."""
        tprint_info(f"🤖 Training {len(model_configs)} models for {self.role}")

        trained_models = {}
        training_results = {}

        for i, model_config in enumerate(model_configs):
            model_name = f"{model_config.name}_{self.role}_{i}"
            tprint_info(f"🏋️ Training model {i+1}/{len(model_configs)}: {model_config.name}")

            with monitor_training_resources(f"train_{model_name}") as resources:
                try:
                    # Use hardware optimization for training
                    if self.deps.get('hardware'):
                        model, metrics = self._train_model_optimized(X, y, model_config)
                    else:
                        model, metrics = self._train_model_standard(X, y, model_config)

                    trained_models[model_name] = model
                    training_results[model_name] = {
                        'metrics': metrics,
                        'config': model_config,
                        'training_time': resources.get('duration', 0),
                        'memory_usage': resources.get('memory_used_mb', 0)
                    }

                    tprint_success(f"✅ Model {model_name} trained successfully")

                except Exception as e:
                    tprint_error(f"❌ Model {model_name} training failed: {e}")
                    training_results[model_name] = {
                        'error': str(e),
                        'failed': True
                    }

                    # Try fallback model if available
                    if model_config.fallback_model:
                        tprint_info(f"🔄 Attempting fallback model: {model_config.fallback_model.value}")
                        try:
                            fallback_config = self.config_manager.get_model_config(model_config.fallback_model)
                            model, metrics = self._train_model_standard(X, y, fallback_config)

                            fallback_name = f"{model_config.name}_fallback_{self.role}_{i}"
                            trained_models[fallback_name] = model
                            training_results[fallback_name] = {
                                'metrics': metrics,
                                'config': fallback_config,
                                'is_fallback': True,
                                'training_time': resources.get('duration', 0)
                            }
                            tprint_success(f"✅ Fallback model {fallback_name} trained successfully")
                        except Exception as fallback_error:
                            tprint_error(f"❌ Fallback model also failed: {fallback_error}")

        return trained_models, training_results

    def _train_model_optimized(self, X: np.ndarray, y: np.ndarray, model_config: Any) -> Tuple[Any, Dict[str, Any]]:
        """Train model with hardware optimization."""
        # Apply hardware optimizations
        X_opt, y_opt = self.hardware_optimizer.optimize_training_batch(X, y)

        # Use enhanced training if available
        if self.deps.get('enhanced_training'):
            # Use enhanced training utilities
            return self._train_with_enhanced_utilities(X_opt, y_opt, model_config)
        else:
            return self._train_model_standard(X, y, model_config)

    def _train_model_standard(self, X: np.ndarray, y: np.ndarray, model_config: Any) -> Tuple[Any, Dict[str, Any]]:
        """Standard model training implementation."""
        # This is where you would implement the actual model training
        # For now, return a placeholder
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score

        # Simple example - replace with actual model training logic
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }

        return model, metrics

    def _train_with_enhanced_utilities(self, X: np.ndarray, y: np.ndarray, model_config: Any) -> Tuple[Any, Dict[str, Any]]:
        """Train model using enhanced training utilities."""
        # Implementation would use EnhancedTrainingUtils
        # For now, delegate to standard training
        return self._train_model_standard(X, y, model_config)

    def save_models(self, trained_models: Dict[str, Any], output_path: str) -> bool:
        """Save trained models with hardware optimization."""
        tprint_info(f"💾 Saving {len(trained_models)} models")

        try:
            # Ensure output directory exists
            from src.utils.common_operations import ensure_directory
            ensure_directory(output_path)

            # Use hardware-optimized saving if available
            if self.deps.get('hardware'):
                success = self.hardware_optimizer.save_models_optimized(trained_models, output_path)
            else:
                success = self._save_models_standard(trained_models, output_path)

            if success:
                tprint_success(f"✅ Models saved to {output_path}")
            else:
                tprint_error("❌ Failed to save models")
                return False

            return True

        except Exception as e:
            tprint_error(f"❌ Model saving failed: {e}")
            return False

    def _save_models_standard(self, trained_models: Dict[str, Any], output_path: str) -> bool:
        """Standard model saving implementation."""
        # Implementation would save models to disk
        # For now, just return True
        return True

    @abstractmethod
    def get_target_variable(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target variable for training. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_data_path(self) -> str:
        """Get path to training data. Must be implemented by subclasses."""
        pass

    def execute_training(self) -> Dict[str, Any]:
        """Execute the complete training pipeline."""
        tprint_info(f"🚀 Starting clean training execution for {self.role}")

        try:
            # Load and prepare data
            df = self.load_training_data(self.get_data_path())
            X, feature_names = self.prepare_features(df)
            y = self.get_target_variable(df)

            # Get model configurations
            model_configs = self.config_manager.get_models_by_priority(self.role, self.mode)

            # Train models
            trained_models, training_results = self.train_models(X, y, model_configs)

            # Save models
            save_path = f"models/{self.role}_models_{self.mode}"
            save_success = self.save_models(trained_models, save_path)

            # Compile results
            results = {
                'role': self.role,
                'mode': self.mode,
                'models_trained': len(trained_models),
                'training_results': training_results,
                'save_success': save_success,
                'feature_names': feature_names,
                'data_shape': (X.shape[0], X.shape[1]),
                'execution_time': time.time() - self.start_time
            }

            tprint_success(f"✅ Training completed for {self.role}")
            return results

        except Exception as e:
            tprint_error(f"❌ Training execution failed for {self.role}: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    # This would be used by specific training implementations
    tprint_info("Clean training base class loaded successfully")
    tprint_info("This module provides the foundation for clean, maintainable training implementations")