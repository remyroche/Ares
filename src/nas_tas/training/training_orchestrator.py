"""
Unified Training Orchestrator for NAS/TAS Systems

This module provides comprehensive training orchestration that consolidates
training logic previously scattered across NAS and TAS implementations.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

from ..config.base_config import UnifiedArchitectureConfig, ArchitectureType
from ..data.data_processor import UnifiedDataProcessor
from ..evaluation.unified_evaluator import UnifiedEvaluator
from ..results.result_manager import UnifiedArchitectureResult, ArchitectureResult, ResultManager, ResultStatus
from ..error_handling import UnifiedErrorHandler
from ..logging import UnifiedLogger, LoggingConfig


@dataclass
class TrainingConfig:
    """Configuration for training orchestration."""
    
    # Training parameters
    max_training_time_minutes: int = 60
    max_models: int = 10
    early_stopping_patience: int = 5
    validation_split: float = 0.2
    
    # Parallel training
    enable_parallel_training: bool = True
    max_parallel_models: int = 3
    
    # Model selection
    model_selection_strategy: str = "best_performance"  # best_performance, diversity, balanced
    diversity_threshold: float = 0.1
    
    # Resource management
    memory_limit_gb: float = 8.0
    cpu_limit_percent: float = 80.0
    
    # Monitoring
    enable_training_monitoring: bool = True
    monitoring_interval_seconds: int = 30
    
    # Custom training functions
    custom_trainers: List[Callable] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if field_name == 'custom_trainers':
                config_dict[field_name] = [func.__name__ for func in field_value]
            else:
                config_dict[field_name] = field_value
        return config_dict


@dataclass
class TrainingResult:
    """Result of training orchestration."""
    
    # Training status
    training_successful: bool = False
    training_duration_seconds: float = 0.0
    
    # Results
    trained_models: List[Any] = field(default_factory=list)
    model_performance: Dict[str, float] = field(default_factory=dict)
    best_model: Optional[Any] = None
    
    # Training metadata
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    training_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'training_successful': self.training_successful,
            'training_duration_seconds': self.training_duration_seconds,
            'model_performance': self.model_performance,
            'best_model_type': type(self.best_model).__name__ if self.best_model else None,
            'training_config': self.training_config.to_dict(),
            'training_metadata': self.training_metadata,
            'errors': self.errors,
            'warnings': self.warnings,
            'trained_models_count': len(self.trained_models)
        }


class TrainingStatus:
    """Training status tracker."""
    
    def __init__(self):
        self.is_training = False
        self.current_stage = ""
        self.progress_percent = 0.0
        self.models_trained = 0
        self.models_total = 0
        self.start_time = None
        self.current_model = None
        self.errors = []
        self.warnings = []
        self._lock = threading.Lock()
    
    def start_training(self, total_models: int):
        """Start training tracking."""
        with self._lock:
            self.is_training = True
            self.models_total = total_models
            self.start_time = datetime.now()
            self.current_stage = "initialization"
            self.progress_percent = 0.0
            self.models_trained = 0
            self.errors.clear()
            self.warnings.clear()
    
    def update_progress(self, stage: str, models_trained: int):
        """Update training progress."""
        with self._lock:
            self.current_stage = stage
            self.models_trained = models_trained
            if self.models_total > 0:
                self.progress_percent = (models_trained / self.models_total) * 100
    
    def add_error(self, error: str):
        """Add training error."""
        with self._lock:
            self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add training warning."""
        with self._lock:
            self.warnings.append(warning)
    
    def finish_training(self, successful: bool):
        """Finish training tracking."""
        with self._lock:
            self.is_training = False
            self.current_stage = "completed" if successful else "failed"
            self.progress_percent = 100.0 if successful else self.progress_percent
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        with self._lock:
            return {
                'is_training': self.is_training,
                'current_stage': self.current_stage,
                'progress_percent': self.progress_percent,
                'models_trained': self.models_trained,
                'models_total': self.models_total,
                'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0,
                'errors': self.errors.copy(),
                'warnings': self.warnings.copy()
            }


class UnifiedTrainingOrchestrator:
    """
    Unified training orchestrator for NAS/TAS systems.
    
    This class consolidates training orchestration logic that was previously
    scattered across NAS and TAS implementations, providing a unified interface
    for training management and coordination.
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        architecture_config: Optional[UnifiedArchitectureConfig] = None
    ):
        """
        Initialize unified training orchestrator.
        
        Args:
            config: Training configuration
            architecture_config: Architecture configuration
        """
        self.config = config or TrainingConfig()
        self.architecture_config = architecture_config or UnifiedArchitectureConfig()
        
        # Initialize components
        self.data_processor = UnifiedDataProcessor()
        self.evaluator = UnifiedEvaluator()
        self.result_manager = ResultManager()
        self.error_handler = UnifiedErrorHandler()
        self.logger = UnifiedLogger()
        
        # Training state
        self.training_status = TrainingStatus()
        self.training_thread = None
        self.stop_training = False
        
        # Model storage
        self.trained_models = []
        self.model_performance = {}
        
        tprint_info("Unified training orchestrator initialized")
    
    async def execute_training(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None,
        search_interface: Optional[Any] = None
    ) -> UnifiedArchitectureResult:
        """
        Execute comprehensive training pipeline.
        
        Args:
            data: Training data
            target: Target variable (optional)
            search_interface: Architecture search interface
            
        Returns:
            UnifiedArchitectureResult with training results
        """
        tprint_info("Starting unified training execution")
        start_time = datetime.now()
        
        try:
            # Initialize training status
            self.training_status.start_training(self.config.max_models)
            
            # Process data
            processed_data, processed_target, validation_result = await self._process_data(data, target)
            
            if not validation_result.validation_passed:
                raise ValueError(f"Data validation failed: {validation_result.validation_errors}")
            
            # Search architectures if interface provided
            architectures = []
            if search_interface:
                architectures = await self._search_architectures(search_interface, processed_data, processed_target)
            else:
                architectures = await self._create_default_architectures(processed_data, processed_target)
            
            # Train models
            training_result = await self._train_models(architectures, processed_data, processed_target)
            
            # Create unified result
            unified_result = await self._create_unified_result(
                training_result, architectures, start_time
            )
            
            # Store result
            self.result_manager.store_result(unified_result)
            
            # Finish training
            self.training_status.finish_training(True)
            
            tprint_success(f"Training completed successfully in {unified_result.execution_info.duration_seconds:.2f}s")
            
            return unified_result
            
        except Exception as e:
            self.training_status.finish_training(False)
            self.training_status.add_error(str(e))
            
            tprint_error(f"Training failed: {e}")
            
            # Create failed result
            failed_result = UnifiedArchitectureResult()
            failed_result.execution_info.finish_execution(
                status=ResultStatus.FAILED,
                error_message=str(e)
            )
            failed_result.execution_info.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            return failed_result
    
    async def _process_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        target: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
        """Process training data."""
        self.training_status.update_progress("data_processing", 0)
        
        with self.logger.log_execution_time("data_processing", "training"):
            processed_data, processed_target, validation_result = self.data_processor.process_data(
                data, target, fit=True
            )
        
        return processed_data, processed_target, validation_result
    
    async def _search_architectures(
        self,
        search_interface: Any,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> List[ArchitectureResult]:
        """Search for architectures using provided interface."""
        self.training_status.update_progress("architecture_search", 0)
        
        with self.logger.log_execution_time("architecture_search", "training"):
            try:
                search_result = await search_interface.search(data, self.architecture_config)
                return search_result.architectures
            except Exception as e:
                self.error_handler.handle_training_error(e, {
                    "component": "architecture_search",
                    "data_shape": data.shape
                })
                return []
    
    async def _create_default_architectures(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> List[ArchitectureResult]:
        """Create default architectures if no search interface provided."""
        self.training_status.update_progress("default_architecture_creation", 0)
        
        # Create simple default architectures
        architectures = []
        
        # Neural architecture
        neural_arch = ArchitectureResult(
            architecture_type=ArchitectureType.NEURAL_ONLY,
            architecture_config={
                "layers": [data.shape[1], 64, 32, 1] if target is not None else [data.shape[1], 64, 32],
                "activation": "relu",
                "optimizer": "adam"
            }
        )
        architectures.append(neural_arch)
        
        # Tree architecture
        tree_arch = ArchitectureResult(
            architecture_type=ArchitectureType.TREE_ONLY,
            architecture_config={
                "max_depth": 10,
                "n_estimators": 100,
                "random_state": 42
            }
        )
        architectures.append(tree_arch)
        
        return architectures
    
    async def _train_models(
        self,
        architectures: List[ArchitectureResult],
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> TrainingResult:
        """Train models for given architectures."""
        self.training_status.update_progress("model_training", 0)
        
        training_result = TrainingResult()
        training_result.training_config = self.config
        training_result.training_metadata = {
            "data_shape": data.shape,
            "target_shape": target.shape if target is not None else None,
            "architecture_count": len(architectures)
        }
        
        start_time = datetime.now()
        
        try:
            if self.config.enable_parallel_training and len(architectures) > 1:
                # Parallel training
                trained_models = await self._train_models_parallel(architectures, data, target)
            else:
                # Sequential training
                trained_models = await self._train_models_sequential(architectures, data, target)
            
            training_result.trained_models = trained_models
            training_result.training_successful = len(trained_models) > 0
            
            # Evaluate models
            if trained_models:
                training_result.model_performance = await self._evaluate_models(trained_models, data, target)
                training_result.best_model = self._select_best_model(trained_models, training_result.model_performance)
            
            training_result.training_duration_seconds = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            training_result.training_successful = False
            training_result.errors.append(str(e))
            self.error_handler.handle_training_error(e, {
                "component": "model_training",
                "architecture_count": len(architectures)
            })
        
        return training_result
    
    async def _train_models_parallel(
        self,
        architectures: List[ArchitectureResult],
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> List[Any]:
        """Train models in parallel."""
        trained_models = []
        
        # Create tasks for parallel execution
        tasks = []
        for arch in architectures[:self.config.max_models]:
            task = asyncio.create_task(self._train_single_model(arch, data, target))
            tasks.append((task, arch))
        
        # Wait for all tasks to complete
        for task, arch in tasks:
            try:
                model = await task
                if model is not None:
                    trained_models.append(model)
                    self.training_status.update_progress(
                        "model_training",
                        len(trained_models)
                    )
            except Exception as e:
                error_msg = f"Failed to train architecture {arch.architecture_id}: {e}"
                self.training_status.add_error(error_msg)
                self.error_handler.handle_training_error(e, {
                    "component": "single_model_training",
                    "architecture_id": arch.architecture_id
                })
        
        return trained_models
    
    async def _train_models_sequential(
        self,
        architectures: List[ArchitectureResult],
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> List[Any]:
        """Train models sequentially."""
        trained_models = []
        
        for i, arch in enumerate(architectures[:self.config.max_models]):
            try:
                model = await self._train_single_model(arch, data, target)
                if model is not None:
                    trained_models.append(model)
                    self.training_status.update_progress("model_training", i + 1)
                
                # Check if training should stop
                if self.stop_training:
                    break
                    
            except Exception as e:
                error_msg = f"Failed to train architecture {arch.architecture_id}: {e}"
                self.training_status.add_error(error_msg)
                self.error_handler.handle_training_error(e, {
                    "component": "single_model_training",
                    "architecture_id": arch.architecture_id
                })
        
        return trained_models
    
    async def _train_single_model(
        self,
        architecture: ArchitectureResult,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> Optional[Any]:
        """Train a single model for given architecture."""
        try:
            # This is a placeholder for actual model training
            # In practice, you would implement the actual training logic here
            # based on the architecture type and configuration
            
            if architecture.architecture_type == ArchitectureType.NEURAL_ONLY:
                # Train neural network
                model = self._train_neural_model(architecture, data, target)
            elif architecture.architecture_type == ArchitectureType.TREE_ONLY:
                # Train tree model
                model = self._train_tree_model(architecture, data, target)
            else:
                # Default training
                model = self._train_default_model(architecture, data, target)
            
            return model
            
        except Exception as e:
            tprint_error(f"Error training single model: {e}")
            return None
    
    def _train_neural_model(
        self,
        architecture: ArchitectureResult,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> Any:
        """Train neural network model with architecture-specific configuration."""
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        from sklearn.preprocessing import StandardScaler
        
        config = architecture.architecture_config
        
        # Extract architecture-specific parameters
        layers = config.get("layers", [data.shape[1], 64, 32, 1] if target is not None else [data.shape[1], 64, 32])
        activation = config.get("activation", "relu")
        optimizer = config.get("optimizer", "adam")
        learning_rate = config.get("learning_rate", 0.001)
        batch_size = config.get("batch_size", 200)
        max_iter = config.get("max_iter", 1000)
        early_stopping = config.get("early_stopping", True)
        validation_fraction = config.get("validation_fraction", 0.1)
        
        # Scale data for neural networks
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Determine problem type and create appropriate model
        if target is not None:
            if len(np.unique(target)) == 2:
                # Binary classification
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(layers[1:-1]),  # Exclude input and output layers
                    activation=activation,
                    solver=optimizer,
                    learning_rate_init=learning_rate,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    validation_fraction=validation_fraction,
                    random_state=42,
                    verbose=False
                )
            else:
                # Multi-class classification or regression
                if len(np.unique(target)) <= 10:  # Multi-class classification
                    model = MLPClassifier(
                        hidden_layer_sizes=tuple(layers[1:-1]),
                        activation=activation,
                        solver=optimizer,
                        learning_rate_init=learning_rate,
                        batch_size=batch_size,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction,
                        random_state=42,
                        verbose=False
                    )
                else:
                    # Regression
                    model = MLPRegressor(
                        hidden_layer_sizes=tuple(layers[1:-1]),
                        activation=activation,
                        solver=optimizer,
                        learning_rate_init=learning_rate,
                        batch_size=batch_size,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction,
                        random_state=42,
                        verbose=False
                    )
        else:
            # Unsupervised learning - use autoencoder-like approach
            model = MLPRegressor(
                hidden_layer_sizes=tuple(layers[1:-1]),
                activation=activation,
                solver=optimizer,
                learning_rate_init=learning_rate,
                batch_size=batch_size,
                max_iter=max_iter,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                random_state=42,
                verbose=False
            )
        
        # Train the model
        try:
            if target is not None:
                model.fit(data_scaled, target)
            else:
                # For unsupervised learning, reconstruct the input
                model.fit(data_scaled, data_scaled)
            
            # Store the scaler with the model for later use
            model.scaler = scaler
            
            tprint_success(f"Neural model trained: {len(layers)} layers, {activation} activation")
            return model
            
        except Exception as e:
            tprint_error(f"Error training neural model: {e}")
            # Fallback to simpler model
            return self._train_fallback_neural_model(data, target)
    
    def _train_fallback_neural_model(self, data: np.ndarray, target: Optional[np.ndarray]) -> Any:
        """Fallback neural model with simpler configuration."""
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        
        if target is not None:
            if len(np.unique(target)) == 2:
                model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
            else:
                model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
            model.fit(data, target)
        else:
            model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
            model.fit(data, data)
        
        return model
    
    def _train_tree_model(
        self,
        architecture: ArchitectureResult,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> Any:
        """Train tree model with architecture-specific configuration."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        
        config = architecture.architecture_config
        
        # Extract architecture-specific parameters
        n_estimators = config.get("n_estimators", 100)
        max_depth = config.get("max_depth", 10)
        min_samples_split = config.get("min_samples_split", 2)
        min_samples_leaf = config.get("min_samples_leaf", 1)
        max_features = config.get("max_features", "sqrt")
        bootstrap = config.get("bootstrap", True)
        random_state = config.get("random_state", 42)
        tree_type = config.get("tree_type", "random_forest")  # random_forest, gradient_boosting, decision_tree
        
        # Determine problem type and create appropriate model
        if target is not None:
            if len(np.unique(target)) == 2:
                # Binary classification
                if tree_type == "gradient_boosting":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state,
                        verbose=0
                    )
                elif tree_type == "decision_tree":
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                else:  # random_forest
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features,
                        bootstrap=bootstrap,
                        random_state=random_state,
                        n_jobs=-1
                    )
            else:
                # Multi-class classification or regression
                if len(np.unique(target)) <= 10:  # Multi-class classification
                    if tree_type == "gradient_boosting":
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            verbose=0
                        )
                    elif tree_type == "decision_tree":
                        model = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state
                        )
                    else:  # random_forest
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            random_state=random_state,
                            n_jobs=-1
                        )
                else:
                    # Regression
                    if tree_type == "gradient_boosting":
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            verbose=0
                        )
                    elif tree_type == "decision_tree":
                        model = DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state
                        )
                    else:  # random_forest
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            random_state=random_state,
                            n_jobs=-1
                        )
        else:
            # Unsupervised learning - use isolation forest
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                n_estimators=n_estimators,
                max_samples=min(256, len(data)),
                contamination=0.1,
                random_state=random_state
            )
        
        # Train the model
        try:
            if target is not None:
                model.fit(data, target)
            else:
                # For unsupervised learning
                model.fit(data)
            
            tprint_success(f"Tree model trained: {tree_type}, {n_estimators} estimators, max_depth={max_depth}")
            return model
            
        except Exception as e:
            tprint_error(f"Error training tree model: {e}")
            # Fallback to simpler model
            return self._train_fallback_tree_model(data, target)
    
    def _train_fallback_tree_model(self, data: np.ndarray, target: Optional[np.ndarray]) -> Any:
        """Fallback tree model with simpler configuration."""
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        if target is not None:
            if len(np.unique(target)) == 2:
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model.fit(data, target)
        else:
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            model.fit(data, data)
        
        return model
    
    def _train_default_model(
        self,
        architecture: ArchitectureResult,
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> Any:
        """Train default model."""
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        if target is not None:
            if len(np.unique(target)) == 2:
                model = LogisticRegression(random_state=42)
            else:
                model = LinearRegression()
            model.fit(data, target)
        else:
            model = LinearRegression()
            model.fit(data, data)
        
        return model
    
    async def _evaluate_models(
        self,
        models: List[Any],
        data: np.ndarray,
        target: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate trained models."""
        performance = {}
        
        for i, model in enumerate(models):
            try:
                if target is not None:
                    predictions = model.predict(data)
                    if hasattr(model, 'predict_proba'):
                        predictions_proba = model.predict_proba(data)
                        eval_result = await self.evaluator.evaluate_model(model, data, target)
                        performance[f"model_{i}"] = eval_result.evaluation_score
                    else:
                        from sklearn.metrics import mean_squared_error, accuracy_score
                        if len(np.unique(target)) == 2:
                            score = accuracy_score(target, predictions)
                        else:
                            score = -mean_squared_error(target, predictions)  # Negative MSE for maximization
                        performance[f"model_{i}"] = score
                else:
                    # Unsupervised evaluation
                    performance[f"model_{i}"] = 0.5  # Placeholder
                
            except Exception as e:
                performance[f"model_{i}"] = 0.0
                self.error_handler.handle_evaluation_error(e, {
                    "component": "model_evaluation",
                    "model_index": i
                })
        
        return performance
    
    def _select_best_model(
        self,
        models: List[Any],
        performance: Dict[str, float]
    ) -> Any:
        """Select best model based on performance."""
        if not performance:
            return models[0] if models else None
        
        best_model_key = max(performance.keys(), key=lambda k: performance[k])
        best_model_index = int(best_model_key.split("_")[1])
        
        return models[best_model_index] if best_model_index < len(models) else models[0]
    
    async def _create_unified_result(
        self,
        training_result: TrainingResult,
        architectures: List[ArchitectureResult],
        start_time: datetime
    ) -> UnifiedArchitectureResult:
        """Create unified architecture result from training."""
        unified_result = UnifiedArchitectureResult()
        
        # Set basic information
        unified_result.search_type = "unified_training"
        unified_result.search_strategy = "orchestrated"
        unified_result.optimization_mode = self.architecture_config.optimization_mode.value
        
        # Set execution info
        unified_result.execution_info.finish_execution(
            status=ResultStatus.SUCCESS if training_result.training_successful else ResultStatus.FAILED
        )
        unified_result.execution_info.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        # Add architectures
        for arch in architectures:
            unified_result.add_architecture(arch)
        
        # Set training metadata
        unified_result.metadata = {
            "training_result": training_result.to_dict(),
            "architecture_config": self.architecture_config.to_dict(),
            "training_config": self.config.to_dict()
        }
        
        return unified_result
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return self.training_status.get_status()
    
    def stop_training_execution(self):
        """Stop ongoing training execution."""
        self.stop_training = True
        tprint_info("Training stop requested")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "training_status": self.get_training_status(),
            "trained_models_count": len(self.trained_models),
            "model_performance": self.model_performance,
            "config": self.config.to_dict(),
            "architecture_config": self.architecture_config.to_dict()
        }