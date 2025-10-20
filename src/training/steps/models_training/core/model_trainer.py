"""
Model Trainer - Individual Model Training Implementation

This module provides concrete implementations for training individual models
across different roles (Analyst, Tactician) with role-specific optimizations.

Key Features:
- Role-specific training logic (Analyst vs Tactician)
- Model-specific implementations (LightGBM, CatBoost, Neural Networks)
- Optimized training pipelines for different timeframes
- Enhanced feature engineering and selection
- Performance monitoring and optimization
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_debug, tprint_performance
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std, safe_float, safe_int,
    get_memory_usage, optimize_dataframe_memory, memory_checkpoint
)
from src.utils.common_utilities import calculate_data_quality_metrics, get_dataframe_info
from src.utils.math_validation import validate_finite, validate_positive, validate_range
from src.utils.hardware.m1_memory_optimizer import optimize_memory
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
from src.utils.kline_parquet import KlinesParquetManager
from src.core.decorators import handles_errors, traced, log_execution_time

from .base_trainer import (
    BaseTrainer, TrainingConfig, TrainingResult, ValidationResult, 
    PredictionResult, TrainingRole, ModelType
)

# Import SHAP explainability utilities
try:
    from src.utils.ml_common.explainability import (
        SHAPLIMEExplainer, ExplanationConfig, create_explainer, explain_model
    )
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    tprint(f"⚠️ [MODEL_TRAINER] SHAP explainability utilities not available: {e}", color="yellow")


class ModelTrainer(BaseTrainer):
    """
    Individual model trainer implementation.
    
    This class provides concrete implementations for training individual models
    with role-specific optimizations and model-specific training logic.
    """
    
    def __init__(self, config: TrainingConfig, logger: Optional[logging.Logger] = None):
        """Initialize the model trainer."""
        super().__init__(config, logger)
        
        # Role-specific configuration
        self._setup_role_specific_config()
        
        # Model-specific state
        self._model_instances = {}
        self._training_histories = {}
        self._validation_histories = {}
    
    def _setup_role_specific_config(self):
        """Setup role-specific configuration."""
        if self.config.role == TrainingRole.ANALYST:
            # Analyst-specific optimizations
            self.config.custom_params.update({
                'enable_feature_interaction': True,
                'confidence_threshold': 0.4,
                'timeframe_optimization': True
            })
        elif self.config.role == TrainingRole.TACTICIAN:
            # Tactician-specific optimizations
            self.config.custom_params.update({
                'enable_timing_features': True,
                'enable_analyst_signals': True,
                'enable_risk_features': True,
                'precision_optimization': True
            })
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError, MemoryError),
        default_return=TrainingResult(success=False, error_message="Training failed"),
        context="model training"
    )
    async def train(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train individual models with role-specific optimizations.
        
        Args:
            data: Training data
            targets: Target variables
            
        Returns:
            Training result with models and metrics
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.role.value} model training...")
            start_time = time.time()
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Train each model type
            training_results = {}
            best_model = None
            best_metrics = {}
            
            for model_type in self.config.model_types:
                self.logger.info(f"📊 Training {model_type.value} model...")
                
                # Create and train model
                model = self._create_model(model_type)
                if model is None:
                    self.logger.error(f"Failed to create {model_type.value} model")
                    continue
                
                # Train model with role-specific logic
                model_result = await self._train_single_model(
                    model, model_type, processed_data, processed_targets
                )
                
                if model_result.success:
                    training_results[model_type.value] = model_result
                    self._model_instances[model_type.value] = model_result.model
                    
                    # Track best model
                    if not best_model or self._is_better_model(model_result.metrics, best_metrics):
                        best_model = model_result.model
                        best_metrics = model_result.metrics
                        
                    self.logger.info(f"✅ {model_type.value} training completed")
                else:
                    self.logger.error(f"❌ {model_type.value} training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            training_time = time.time() - start_time
            
            # Update state
            self._training_state['training_completed'] = True
            self._training_state['training_started'] = True
            self._update_performance_metrics('training', training_time)
            
            result = TrainingResult(
                success=len(training_results) > 0,
                model=best_model,
                metrics=overall_metrics,
                training_time=training_time,
                metadata={
                    'models_trained': len(training_results),
                    'role': self.config.role.value,
                    'timeframe': self.config.timeframe,
                    'individual_results': training_results
                }
            )
            
            if result.success:
                self.logger.info(f"✅ Training completed successfully in {training_time:.2f}s")
                tprint_success(f"Trained {len(training_results)} models for {self.config.role.value}")
            else:
                self.logger.error("❌ All model training failed")
                tprint_error("Training failed for all models")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time
            )
    
    async def _train_single_model(
        self, 
        model: Any, 
        model_type: ModelType, 
        data: pd.DataFrame, 
        targets: pd.Series
    ) -> TrainingResult:
        """Train a single model with role-specific optimizations."""
        try:
            start_time = time.time()
            
            # Role-specific feature engineering
            if self.config.role == TrainingRole.ANALYST:
                data = self._engineer_analyst_features(data, targets)
            elif self.config.role == TrainingRole.TACTICIAN:
                data = self._engineer_tactician_features(data, targets)
            
            # Model-specific training
            if model_type == ModelType.LIGHTGBM:
                return await self._train_lightgbm_model(model, data, targets)
            elif model_type == ModelType.CATBOOST:
                return await self._train_catboost_model(model, data, targets)
            elif model_type == ModelType.NEURAL_NETWORK:
                return await self._train_neural_network_model(model, data, targets)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Single model training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    def _engineer_analyst_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Analyst role."""
        try:
            
            # Add market condition features
            if 'volume' in data.columns and 'close' in data.columns:
                data['volume_price_trend'] = data['volume'] * data['close'].pct_change()
                data['volume_momentum'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
            
            # Add volatility features
            if 'close' in data.columns:
                data['volatility_5d'] = data['close'].rolling(5).std()
                data['volatility_20d'] = data['close'].rolling(20).std()
                data['volatility_ratio'] = data['volatility_5d'] / data['volatility_20d']
            
            self.logger.info(f"Engineered {len(data.columns)} features for Analyst")
            return data
            
        except Exception as e:
            self.logger.warning(f"Analyst feature engineering failed: {e}")
            return data
    
    def _engineer_tactician_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Tactician role."""
        try:
            # Add timing features
            if 'timestamp' in data.columns:
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
            
            # Add analyst signal features
            analyst_columns = [col for col in data.columns if 'analyst' in col.lower()]
            if analyst_columns:
                data['analyst_signal_strength'] = data[analyst_columns].mean(axis=1)
                data['analyst_signal_consistency'] = data[analyst_columns].std(axis=1)
            
            # Add risk features
            if 'close' in data.columns:
                data['price_momentum'] = data['close'].pct_change(5)
                data['risk_adjusted_return'] = data['price_momentum'] / data['close'].rolling(20).std()
            
            self.logger.info(f"Engineered {len(data.columns)} features for Tactician")
            return data
            
        except Exception as e:
            self.logger.warning(f"Tactician feature engineering failed: {e}")
            return data
    
    async def _train_lightgbm_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train LightGBM model with role-specific parameters."""
        try:
            # Import LightGBM
            import lightgbm as lgb
            
            # Role-specific parameters
            if self.config.role == TrainingRole.ANALYST:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1
                }
            else:  # Tactician
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 63,
                    'learning_rate': 0.03,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.7,
                    'bagging_freq': 5,
                    'verbose': -1
                }
            
            # Create dataset
            train_data = lgb.Dataset(data, label=targets)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
            
            # Get predictions and metrics
            predictions = model.predict(data)
            
            if self.config.role == TrainingRole.ANALYST:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                binary_predictions = (predictions > 0.5).astype(int)
                metrics = {
                    'accuracy': accuracy_score(targets, binary_predictions),
                    'precision': precision_score(targets, binary_predictions),
                    'recall': recall_score(targets, binary_predictions),
                    'f1_score': f1_score(targets, binary_predictions)
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(targets, predictions),
                    'mae': mean_absolute_error(targets, predictions),
                    'r2': r2_score(targets, predictions),
                    'rmse': np.sqrt(mean_squared_error(targets, predictions))
                }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.feature_importance()))
            
            # Generate SHAP explanations
            shap_explanations = None
            if SHAP_AVAILABLE:
                try:
                    tprint("🔍 [MODEL_TRAINER] Generating SHAP explanations for LightGBM", color="cyan")
                    shap_config = ExplanationConfig(
                        enable_shap=True,
                        enable_lime=False,
                        shap_sample_size=min(100, len(data)),
                        shap_max_features=min(50, data.shape[1])
                    )
                    explainer = create_explainer(shap_config)
                    
                    # Prepare data for SHAP
                    X_array = data.values
                    feature_names = list(data.columns)
                    output_names = ["prediction"] if self.config.role == TrainingRole.TACTICIAN else ["class_0", "class_1"]
                    
                    shap_result = explainer.explain_model(
                        model=model,
                        X=X_array,
                        model_name="LightGBM",
                        output_names=output_names,
                        feature_names=feature_names
                    )
                    
                    shap_explanations = {
                        'shap_values': shap_result.shap_values,
                        'base_values': shap_result.shap_base_values,
                        'feature_names': shap_result.shap_feature_names,
                        'explanation_time': shap_result.explanation_time
                    }
                    
                    tprint(f"✅ [MODEL_TRAINER] LightGBM SHAP explanations generated in {shap_result.explanation_time:.3f}s", color="green")
                except Exception as e:
                    tprint(f"⚠️ [MODEL_TRAINER] LightGBM SHAP explanation failed: {e}", color="yellow")
                    shap_explanations = None
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance,
                shap_explanations=shap_explanations
            )
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_catboost_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train CatBoost model with role-specific parameters."""
        try:
            # Import CatBoost
            from catboost import CatBoostClassifier, CatBoostRegressor
            
            # Role-specific model creation
            if self.config.role == TrainingRole.ANALYST:
                model = CatBoostClassifier(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    loss_function='Logloss',
                    eval_metric='AUC',
                    verbose=False
                )
            else:  # Tactician
                model = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.03,
                    depth=8,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    verbose=False
                )
            
            # Train model
            model.fit(data, targets, eval_set=(data, targets), early_stopping_rounds=50)
            
            # Get predictions and metrics
            predictions = model.predict(data)
            
            if self.config.role == TrainingRole.ANALYST:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                binary_predictions = (predictions > 0.5).astype(int)
                metrics = {
                    'accuracy': accuracy_score(targets, binary_predictions),
                    'precision': precision_score(targets, binary_predictions),
                    'recall': recall_score(targets, binary_predictions),
                    'f1_score': f1_score(targets, binary_predictions)
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(targets, predictions),
                    'mae': mean_absolute_error(targets, predictions),
                    'r2': r2_score(targets, predictions),
                    'rmse': np.sqrt(mean_squared_error(targets, predictions))
                }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.get_feature_importance()))
            
            # Generate SHAP explanations
            shap_explanations = None
            if SHAP_AVAILABLE:
                try:
                    tprint("🔍 [MODEL_TRAINER] Generating SHAP explanations for CatBoost", color="cyan")
                    shap_config = ExplanationConfig(
                        enable_shap=True,
                        enable_lime=False,
                        shap_sample_size=min(100, len(data)),
                        shap_max_features=min(50, data.shape[1])
                    )
                    explainer = create_explainer(shap_config)
                    
                    # Prepare data for SHAP
                    X_array = data.values
                    feature_names = list(data.columns)
                    output_names = ["prediction"] if self.config.role == TrainingRole.TACTICIAN else ["class_0", "class_1"]
                    
                    shap_result = explainer.explain_model(
                        model=model,
                        X=X_array,
                        model_name="CatBoost",
                        output_names=output_names,
                        feature_names=feature_names
                    )
                    
                    shap_explanations = {
                        'shap_values': shap_result.shap_values,
                        'base_values': shap_result.shap_base_values,
                        'feature_names': shap_result.shap_feature_names,
                        'explanation_time': shap_result.explanation_time
                    }
                    
                    tprint(f"✅ [MODEL_TRAINER] CatBoost SHAP explanations generated in {shap_result.explanation_time:.3f}s", color="green")
                except Exception as e:
                    tprint(f"⚠️ [MODEL_TRAINER] CatBoost SHAP explanation failed: {e}", color="yellow")
                    shap_explanations = None
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance,
                shap_explanations=shap_explanations
            )
            
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_neural_network_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train neural network model with role-specific architecture."""
        try:
            # Import PyTorch
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(data.values)
            y_tensor = torch.FloatTensor(targets.values)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Role-specific architecture
            if self.config.role == TrainingRole.ANALYST:
                model = nn.Sequential(
                    nn.Linear(data.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                criterion = nn.BCELoss()
            else:  # Tactician
                model = nn.Sequential(
                    nn.Linear(data.shape[1], 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                criterion = nn.MSELoss()
            
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            for epoch in range(100):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
            # Get predictions and metrics
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).squeeze().numpy()
            
            if self.config.role == TrainingRole.ANALYST:
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                binary_predictions = (predictions > 0.5).astype(int)
                metrics = {
                    'accuracy': accuracy_score(targets, binary_predictions),
                    'precision': precision_score(targets, binary_predictions),
                    'recall': recall_score(targets, binary_predictions),
                    'f1_score': f1_score(targets, binary_predictions)
                }
            else:
                # Regression metrics
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics = {
                    'mse': mean_squared_error(targets, predictions),
                    'mae': mean_absolute_error(targets, predictions),
                    'r2': r2_score(targets, predictions),
                    'rmse': np.sqrt(mean_squared_error(targets, predictions))
                }
            
            # Generate SHAP explanations
            shap_explanations = None
            if SHAP_AVAILABLE:
                try:
                    tprint("🔍 [MODEL_TRAINER] Generating SHAP explanations for Neural Network", color="cyan")
                    shap_config = ExplanationConfig(
                        enable_shap=True,
                        enable_lime=False,
                        shap_sample_size=min(100, len(data)),
                        shap_max_features=min(50, data.shape[1])
                    )
                    explainer = create_explainer(shap_config)
                    
                    # Prepare data for SHAP
                    X_array = data.values
                    feature_names = list(data.columns)
                    output_names = ["prediction"] if self.config.role == TrainingRole.TACTICIAN else ["class_0", "class_1"]
                    
                    # Create a wrapper for the PyTorch model to work with SHAP
                    class PyTorchWrapper:
                        def __init__(self, model, role):
                            self.model = model
                            self.role = role
                            
                        def predict(self, X):
                            self.model.eval()
                            with torch.no_grad():
                                X_tensor = torch.FloatTensor(X)
                                if self.role == TrainingRole.ANALYST:
                                    return self.model(X_tensor).squeeze().numpy()
                                else:
                                    return self.model(X_tensor).squeeze().numpy()
                    
                    wrapped_model = PyTorchWrapper(model, self.config.role)
                    
                    shap_result = explainer.explain_model(
                        model=wrapped_model,
                        X=X_array,
                        model_name="Neural Network",
                        output_names=output_names,
                        feature_names=feature_names
                    )
                    
                    shap_explanations = {
                        'shap_values': shap_result.shap_values,
                        'base_values': shap_result.shap_base_values,
                        'feature_names': shap_result.shap_feature_names,
                        'explanation_time': shap_result.explanation_time
                    }
                    
                    tprint(f"✅ [MODEL_TRAINER] Neural Network SHAP explanations generated in {shap_result.explanation_time:.3f}s", color="green")
                except Exception as e:
                    tprint(f"⚠️ [MODEL_TRAINER] Neural Network SHAP explanation failed: {e}", color="yellow")
                    shap_explanations = None
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                shap_explanations=shap_explanations
            )
            
        except Exception as e:
            self.logger.error(f"Neural network training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    def _create_model(self, model_type: ModelType) -> Any:
        """Create model instance based on type."""
        try:
            if model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                return lgb.LGBMClassifier() if self.config.role == TrainingRole.ANALYST else lgb.LGBMRegressor()
            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostClassifier, CatBoostRegressor
                return CatBoostClassifier() if self.config.role == TrainingRole.ANALYST else CatBoostRegressor()
            elif model_type == ModelType.NEURAL_NETWORK:
                # Return None, will be created in training method
                return None
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except ImportError as e:
            self.logger.error(f"Failed to import required library: {e}")
            return None
    
    def _get_feature_importance(self, model: Any) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importance'):
                return dict(zip(self._get_feature_names(), model.feature_importance()))
            elif hasattr(model, 'get_feature_importance'):
                return dict(zip(self._get_feature_names(), model.get_feature_importance()))
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Failed to extract feature importance: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance extraction."""
        # This would be set during preprocessing
        return getattr(self, '_feature_names', [])
    
    def _is_better_model(self, current_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> bool:
        """Check if current model is better than best model."""
        if not best_metrics:
            return True
        
        # Use primary metric for comparison
        primary_metric = 'f1_score' if self.config.role == TrainingRole.ANALYST else 'r2'
        
        if primary_metric in current_metrics and primary_metric in best_metrics:
            return current_metrics[primary_metric] > best_metrics[primary_metric]
        
        return False
    
    def _calculate_overall_metrics(self, training_results: Dict[str, TrainingResult]) -> Dict[str, float]:
        """Calculate overall metrics from individual model results."""
        if not training_results:
            return {}
        
        # Average metrics across all successful models
        all_metrics = {}
        for result in training_results.values():
            if result.success and result.metrics:
                for metric, value in result.metrics.items():
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)
        
        # Calculate averages
        overall_metrics = {}
        for metric, values in all_metrics.items():
            overall_metrics[f'avg_{metric}'] = np.mean(values)
            overall_metrics[f'std_{metric}'] = np.std(values)
        
        return overall_metrics
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=ValidationResult(success=False, error_message="Validation failed"),
        context="model validation"
    )
    async def validate(self, data: pd.DataFrame, targets: Optional[pd.Series] = None) -> ValidationResult:
        """Validate trained models."""
        try:
            if not self._model_instances:
                return ValidationResult(success=False, error_message="No trained models available")
            
            # Use best model for validation
            best_model = max(self._model_instances.values(), key=lambda m: getattr(m, 'score', 0))
            
            # Get predictions
            predictions = best_model.predict(data)
            
            # Calculate validation metrics
            if targets is not None:
                if self.config.role == TrainingRole.ANALYST:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    binary_predictions = (predictions > 0.5).astype(int)
                    metrics = {
                        'accuracy': accuracy_score(targets, binary_predictions),
                        'precision': precision_score(targets, binary_predictions),
                        'recall': recall_score(targets, binary_predictions),
                        'f1_score': f1_score(targets, binary_predictions)
                    }
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    metrics = {
                        'mse': mean_squared_error(targets, predictions),
                        'mae': mean_absolute_error(targets, predictions),
                        'r2': r2_score(targets, predictions),
                        'rmse': np.sqrt(mean_squared_error(targets, predictions))
                    }
            else:
                metrics = {}
            
            return ValidationResult(
                success=True,
                metrics=metrics,
                predictions=predictions
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(success=False, error_message=str(e))
    
    @handles_errors(
        exceptions=(ValueError, RuntimeError),
        default_return=PredictionResult(success=False, error_message="Prediction failed"),
        context="model prediction"
    )
    async def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make predictions with trained models."""
        try:
            if not self._model_instances:
                return PredictionResult(success=False, error_message="No trained models available")
            
            # Use best model for prediction
            best_model = max(self._model_instances.values(), key=lambda m: getattr(m, 'score', 0))
            
            # Get predictions
            predictions = best_model.predict(data)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(best_model, 'predict_proba'):
                probabilities = best_model.predict_proba(data)
            
            return PredictionResult(
                success=True,
                predictions=predictions,
                probabilities=probabilities
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return PredictionResult(success=False, error_message=str(e))
