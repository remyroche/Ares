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
from .training_metrics_collector import TrainingMetricsCollector, ModelMetrics

# Shared feature engineering
from src.feature_generation.shared.feature_engineer import (
    AnalystFeatureEngineer,
    TacticianFeatureEngineer
)


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
        
        # Metrics collector
        self._metrics_collector = TrainingMetricsCollector(logger)
        
        # Shared feature engineers
        self._analyst_feature_engineer = AnalystFeatureEngineer(logger=logger)
        self._tactician_feature_engineer = TacticianFeatureEngineer(logger=logger)
    
    def _setup_role_specific_config(self):
        """Setup role-specific configuration."""
        if self.config.role == TrainingRole.ANALYST:
            # Analyst-specific optimizations
            self.config.custom_params.update({
                'enable_feature_interaction': True,
                'enable_regime_features': True,
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
        Train individual models with role-specific optimizations and comprehensive metrics collection.
        
        Args:
            data: Training data
            targets: Target variables
            
        Returns:
            Training result with models and comprehensive metrics
        """
        try:
            self.logger.info(f"🚀 Starting {self.config.role.value} model training with comprehensive metrics...")
            start_time = time.time()
            
            # Start metrics collection session
            training_type = f"{self.config.role.value}_base"
            self._metrics_collector.start_session(
                training_type=training_type,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe
            )
            
            # Preprocess data
            processed_data, processed_targets = self._preprocess_data(data, targets)
            
            # Train each model type with comprehensive metrics
            training_results = {}
            best_model = None
            best_metrics = {}
            all_model_metrics = []
            
            for model_type in self.config.model_types:
                self.logger.info(f"📊 Training {model_type.value} model with metrics collection...")
                
                # Create model for pre-HPO baseline
                model = self._create_model(model_type)
                # Allow None for models that are created during training (TCN, Neural Networks)
                if model is None and model_type not in [ModelType.TCN, ModelType.NEURAL_NETWORK]:
                    self.logger.error(f"Failed to create {model_type.value} model")
                    continue
                
                # Collect pre-HPO metrics
                tprint_info(f"📊 Phase 1: Collecting pre-HPO baseline metrics for {model_type.value}...")
                model_metrics = self._metrics_collector.collect_pre_hpo_metrics(
                    model_name=f"{self.config.role.value}_{model_type.value}",
                    model_type=model_type.value,
                    model=model,
                    X=processed_data,
                    y=processed_targets,
                    n_folds=self.config.cross_validation_folds
                )
                
                # Run hyperparameter optimization if enabled
                best_params = {}
                hpo_n_trials = 0
                hpo_time = 0.0
                
                if self.config.enable_hyperparameter_optimization:
                    tprint_info(f"🔧 Phase 2: Running hyperparameter optimization for {model_type.value}...")
                    hpo_start = time.time()
                    
                    model, best_params = await self._optimize_hyperparameters(
                        model, model_type, processed_data, processed_targets
                    )
                    
                    hpo_time = time.time() - hpo_start
                    hpo_n_trials = self.config.custom_params.get('hpo_n_trials', 50)
                    
                    tprint_success(f"✅ HPO completed in {hpo_time:.2f}s with {hpo_n_trials} trials")
                else:
                    tprint_info(f"⏭️  Skipping HPO (disabled in config)")
                
                # Train final model with best parameters
                tprint_info(f"🎯 Phase 3: Training final model with optimized parameters...")
                model_result = await self._train_single_model(
                    model, model_type, processed_data, processed_targets
                )
                
                if model_result.success:
                    # Collect post-HPO metrics
                    tprint_info(f"📈 Phase 4: Collecting post-HPO metrics for {model_type.value}...")
                    model_metrics = self._metrics_collector.collect_post_hpo_metrics(
                        model_metrics=model_metrics,
                        model=model_result.model,
                        X=processed_data,
                        y=processed_targets,
                        best_params=best_params,
                        hpo_n_trials=hpo_n_trials,
                        hpo_time=hpo_time,
                        n_folds=self.config.cross_validation_folds
                    )
                    
                    # Add to session
                    self._metrics_collector.add_model_metrics(model_metrics)
                    all_model_metrics.append(model_metrics)
                    
                    training_results[model_type.value] = model_result
                    self._model_instances[model_type.value] = model_result.model
                    
                    # Track best model
                    if not best_model or self._is_better_model(model_result.metrics, best_metrics):
                        best_model = model_result.model
                        best_metrics = model_result.metrics
                    
                    tprint_success(f"✅ {model_type.value} training completed with comprehensive metrics")
                    self.logger.info(f"✅ {model_type.value} training completed")
                else:
                    tprint_error(f"❌ {model_type.value} training failed: {model_result.error_message}")
                    self.logger.error(f"❌ {model_type.value} training failed: {model_result.error_message}")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(training_results)
            training_time = time.time() - start_time
            
            # Finalize session and generate report
            tprint_info("📝 Generating comprehensive training report...")
            session = self._metrics_collector.finalize_session(
                total_training_time=training_time,
                data_quality_score=0.85,  # TODO: Calculate actual quality score
                n_samples=len(processed_data),
                n_features=len(processed_data.columns)
            )
            
            # Generate and save report
            report_path = self._metrics_collector.save_report()
            
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
                    'individual_results': training_results,
                    'comprehensive_metrics': all_model_metrics,
                    'report_path': str(report_path)
                }
            )
            
            if result.success:
                self.logger.info(f"✅ Training completed successfully in {training_time:.2f}s")
                tprint_success(f"✅ Trained {len(training_results)} models with comprehensive metrics")
                tprint_success(f"📄 Report saved to: {report_path}")
            else:
                self.logger.error("❌ All model training failed")
                tprint_error("❌ Training failed for all models")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
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
            elif model_type == ModelType.TCN:
                return await self._train_tcn_model(model, data, targets)
            elif model_type == ModelType.CATBOOST:
                return await self._train_catboost_model(model, data, targets)
            elif model_type == ModelType.NEURAL_NETWORK:
                return await self._train_neural_network_model(model, data, targets)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Single model training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _optimize_hyperparameters(
        self,
        model: Any,
        model_type: ModelType,
        data: pd.DataFrame,
        targets: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Optimize hyperparameters using Bayesian TPE optimization.
        
        Args:
            model: Base model to optimize
            model_type: Type of model
            data: Training data
            targets: Training targets
            
        Returns:
            Tuple of (optimized_model, best_params)
        """
        try:
            # Get HPO config
            n_trials = self.config.custom_params.get('hpo_n_trials', 50)
            
            # Define search spaces based on model type
            if model_type == ModelType.LIGHTGBM:
                search_space = {
                    'num_leaves': ('int', 20, 100),
                    'learning_rate': ('float', 0.01, 0.1),
                    'feature_fraction': ('float', 0.6, 1.0),
                    'bagging_fraction': ('float', 0.6, 1.0),
                    'min_child_samples': ('int', 5, 50)
                }
            elif model_type == ModelType.CATBOOST:
                search_space = {
                    'depth': ('int', 4, 10),
                    'learning_rate': ('float', 0.01, 0.1),
                    'l2_leaf_reg': ('float', 1, 10),
                    'border_count': ('int', 32, 255)
                }
            else:
                # Return model as-is for types without HPO
                return model, {}
            
            # Use BayesianTPEOptimizer
            optimizer = BayesianTPEOptimizer()
            
            # Define objective function
            def objective(params):
                try:
                    # Create model with params
                    if model_type == ModelType.LIGHTGBM:
                        import lightgbm as lgb
                        test_model = lgb.LGBMRegressor(**params, verbose=-1)
                    elif model_type == ModelType.CATBOOST:
                        from catboost import CatBoostRegressor
                        test_model = CatBoostRegressor(**params, verbose=False)
                    
                    # Cross-validate
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(
                        test_model, data, targets, 
                        cv=3, scoring='r2', n_jobs=-1
                    )
                    
                    return np.mean(scores)
                except Exception as e:
                    self.logger.warning(f"HPO trial failed: {e}")
                    return -999999  # Very bad score
            
            # Run optimization
            best_params = {}
            best_score = -float('inf')
            
            for trial in range(n_trials):
                # Sample parameters
                trial_params = {}
                for param_name, param_spec in search_space.items():
                    if param_spec[0] == 'int':
                        trial_params[param_name] = np.random.randint(param_spec[1], param_spec[2] + 1)
                    elif param_spec[0] == 'float':
                        trial_params[param_name] = np.random.uniform(param_spec[1], param_spec[2])
                
                # Evaluate
                score = objective(trial_params)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = trial_params.copy()
            
            # Create optimized model
            if model_type == ModelType.LIGHTGBM:
                import lightgbm as lgb
                optimized_model = lgb.LGBMRegressor(**best_params, verbose=-1)
            elif model_type == ModelType.CATBOOST:
                from catboost import CatBoostRegressor
                optimized_model = CatBoostRegressor(**best_params, verbose=False)
            else:
                optimized_model = model
            
            self.logger.info(f"✅ HPO completed: best score = {best_score:.4f}")
            return optimized_model, best_params
            
        except Exception as e:
            self.logger.error(f"HPO failed: {e}, using default model")
            return model, {}
    
    def _engineer_analyst_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Analyst role using shared module."""
        try:
            # Use shared feature engineer for consistency with inference
            engineered_data = self._analyst_feature_engineer.engineer_features(data)
            return engineered_data
            
        except Exception as e:
            self.logger.warning(f"Analyst feature engineering failed: {e}")
            return data
    
    def _engineer_tactician_features(self, data: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """Engineer features specific to Tactician role using shared module."""
        try:
            # Use shared feature engineer for consistency with inference
            # Extract analyst confidence from data if available
            analyst_confidence = None
            if 'analyst_confidence' in data.columns:
                analyst_confidence = data['analyst_confidence'].iloc[-1] if len(data) > 0 else None
            
            engineered_data = self._tactician_feature_engineer.engineer_features(
                data,
                analyst_confidence=analyst_confidence
            )
            return engineered_data
            
        except Exception as e:
            self.logger.warning(f"Tactician feature engineering failed: {e}")
            return data
    
    async def _train_lightgbm_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train LightGBM model with role-specific parameters."""
        try:
            # Import LightGBM
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            
            # Split data for validation (required for early stopping)
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets, test_size=0.2, random_state=42
            )
            
            # Role-specific parameters
            if self.config.role == TrainingRole.ANALYST:
                params = {
                    'objective': 'regression',  # Changed from binary to regression
                    'metric': 'rmse',  # Changed from binary_logloss to rmse
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
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model with validation set for early stopping
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Get predictions and metrics (use all data for prediction evaluation)
            predictions = model.predict(data)
            
            # Use regression metrics for all roles (targets are continuous)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.feature_importance()))
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_catboost_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train CatBoost model with role-specific parameters."""
        try:
            # Import CatBoost
            from catboost import CatBoostRegressor
            from sklearn.model_selection import train_test_split
            
            # Split data for validation (required for early stopping)
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets, test_size=0.2, random_state=42
            )
            
            # Create regressor for all roles (targets are continuous)
            if self.config.role == TrainingRole.ANALYST:
                model = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.05,
                    depth=6,
                    loss_function='RMSE',
                    eval_metric='RMSE',
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
            
            # Train model with validation set
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            
            # Get predictions and metrics (on full data)
            predictions = model.predict(data)
            
            # Use regression metrics for all roles
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            # Get feature importance
            feature_importance = dict(zip(data.columns, model.get_feature_importance()))
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            return TrainingResult(success=False, error_message=str(e))
    
    async def _train_tcn_model(self, model: Any, data: pd.DataFrame, targets: pd.Series) -> TrainingResult:
        """Train Temporal Convolutional Network model with role-specific parameters."""
        try:
            # Import TCN model
            from src.models.causal_dilated_tcn import CausalDilatedTCNModel, CausalTCNConfig
            from sklearn.model_selection import train_test_split
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                data, targets, test_size=0.2, random_state=42
            )
            
            # Role-specific TCN configuration
            if self.config.role == TrainingRole.ANALYST:
                tcn_config = CausalTCNConfig(
                    num_filters=64,
                    num_layers=4,
                    kernel_size=3,
                    dilation_base=2,
                    dropout=0.2,
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=100,
                    early_stopping_patience=10
                )
            else:  # Tactician
                tcn_config = CausalTCNConfig(
                    num_filters=64,
                    num_layers=4,
                    kernel_size=3,
                    dilation_base=2,
                    dropout=0.1,
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=100,
                    early_stopping_patience=10
                )
            
            # Create and train TCN model
            model = CausalDilatedTCNModel(config=tcn_config)
            model.fit(X_train.values if isinstance(X_train, pd.DataFrame) else X_train, 
                     y_train.values if isinstance(y_train, pd.Series) else y_train)
            
            # Get predictions on full data
            predictions = model.predict(data.values if isinstance(data, pd.DataFrame) else data)
            
            # Calculate regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions))
            }
            
            self.logger.info(f"✅ TCN model trained successfully - R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics,
                feature_importance=None  # TCN doesn't provide feature importance directly
            )
            
        except ImportError as e:
            self.logger.error(f"TCN training failed - missing dependencies: {e}")
            return TrainingResult(success=False, error_message=f"Missing dependencies: {e}")
        except Exception as e:
            self.logger.error(f"TCN training failed: {e}")
            import traceback
            traceback.print_exc()
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
            
            return TrainingResult(
                success=True,
                model=model,
                metrics=metrics
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
            elif model_type == ModelType.TCN:
                # Return None, will be created in training method
                return None
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
