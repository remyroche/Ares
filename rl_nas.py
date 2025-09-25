"""
Reinforcement Learning Neural Architecture Search (RL-NAS) Optimizer

This module provides a comprehensive RL-NAS optimization system that combines
reinforcement learning with neural architecture search for trading strategy optimization.
It integrates with the existing utility modules for enhanced performance and functionality.

Key Features:
- Multi-objective optimization with Pareto frontier analysis
- M1 Apple Silicon optimization for GPU/CPU acceleration
- Cross-validation and hyperparameter optimization
- Advanced ensemble methods and model stacking
- Real-time performance monitoring and memory optimization
- Comprehensive logging and error handling
"""

import logging
import time
import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from contextlib import contextmanager

# Import utility modules
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory, 
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    integrate_with_m1_optimizers, optimize_memory, get_memory_usage,
    safe_dataframe_operation, validate_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive
)
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns, safe_convert_dtypes, calculate_data_quality_metrics
)
from src.utils.math_validation import (
    MathValidation, safe_correlation, safe_covariance, safe_mean, safe_std,
    validate_correlation_matrix, safe_matrix_inverse
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_performance, tprint_progress, tprint_structured, tprint_timer
)

# Import ML utilities
try:
    from src.utils.ml_common import (
        EnhancedModelFactory, ModelType, ModelConfig, create_model_factory,
        MultiOutputConfig, MultiOutputModel, MultiOutputStackingModel,
        EnhancedModelTrainer, ModelEvaluator, ModelRegistry,
        EnsembleManager, EnsembleType, EnsembleConfig,
        StackingEnsembleManager, StackingEnsembleConfig,
        ModelExplainer, ModelInterpretabilityEngine,
        ParetoOptimizer, ParetoFront, ParetoFrontAnalyzer,
        UnifiedCrossValidator, UnifiedCVResult,
        perform_cross_validation, temporal_cross_validation,
        MemoryOptimizer, ParallelProcessor, UnifiedCache,
        LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler
    )
    ML_UTILS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"ML utilities not available: {e}")
    ML_UTILS_AVAILABLE = False

# Import hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array
    )
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory, optimize_memory
    )
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError as e:
    tprint_warning(f"Hardware utilities not available: {e}")
    HARDWARE_UTILS_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Optimization objectives for RL-NAS."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    STABILITY = "stability"
    COMPLEXITY = "complexity"

class ArchitectureType(Enum):
    """Neural architecture types."""
    FEEDFORWARD = "feedforward"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CONVOLUTIONAL = "convolutional"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"
    STACKING = "stacking"

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture."""
    architecture_type: ArchitectureType
    hidden_layers: List[int]
    activation_functions: List[str]
    dropout_rates: List[float]
    regularization: Dict[str, float]
    learning_rate: float
    batch_size: int
    epochs: int
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'architecture_type': self.architecture_type.value,
            'hidden_layers': self.hidden_layers,
            'activation_functions': self.activation_functions,
            'dropout_rates': self.dropout_rates,
            'regularization': self.regularization,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split
        }

@dataclass
class OptimizationConfig:
    """Configuration for RL-NAS optimization."""
    objectives: List[OptimizationObjective]
    max_generations: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnation: int = 20
    parallel_evaluation: bool = True
    use_m1_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    cross_validation_folds: int = 5
    temporal_validation: bool = True
    lookahead_protection: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'objectives': [obj.value for obj in self.objectives],
            'max_generations': self.max_generations,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elite_size': self.elite_size,
            'tournament_size': self.tournament_size,
            'convergence_threshold': self.convergence_threshold,
            'max_stagnation': self.max_stagnation,
            'parallel_evaluation': self.parallel_evaluation,
            'use_m1_optimization': self.use_m1_optimization,
            'memory_limit_gb': self.memory_limit_gb,
            'cross_validation_folds': self.cross_validation_folds,
            'temporal_validation': self.temporal_validation,
            'lookahead_protection': self.lookahead_protection
        }

@dataclass
class OptimizationResult:
    """Result of RL-NAS optimization."""
    best_architecture: ArchitectureConfig
    best_fitness: Dict[str, float]
    pareto_front: List[Tuple[ArchitectureConfig, Dict[str, float]]]
    optimization_history: List[Dict[str, Any]]
    convergence_info: Dict[str, Any]
    execution_time: float
    memory_usage: Dict[str, Any]
    hardware_utilization: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'best_architecture': self.best_architecture.to_dict(),
            'best_fitness': self.best_fitness,
            'pareto_front': [
                (arch.to_dict(), fitness) for arch, fitness in self.pareto_front
            ],
            'optimization_history': self.optimization_history,
            'convergence_info': self.convergence_info,
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'hardware_utilization': self.hardware_utilization
        }

class RL_NAS_Optimizer:
    """
    Reinforcement Learning Neural Architecture Search Optimizer.
    
    This class provides comprehensive RL-NAS optimization for trading strategies,
    integrating with existing utility modules for enhanced performance.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize RL-NAS Optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.logger = logger.getChild('RL_NAS_Optimizer')
        
        # Initialize utility modules
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize hardware optimizers
        self._init_hardware_optimizers()
        
        # Initialize ML utilities if available
        self._init_ml_utilities()
        
        # Optimization state
        self.population: List[Tuple[ArchitectureConfig, Dict[str, float]]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.pareto_front: List[Tuple[ArchitectureConfig, Dict[str, float]]] = []
        self.convergence_info: Dict[str, Any] = {}
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.memory_checkpoints: List[Dict[str, Any]] = []
        
        tprint_success("🚀 RL-NAS Optimizer initialized successfully")
    
    def _init_hardware_optimizers(self):
        """Initialize hardware optimization utilities."""
        if HARDWARE_UTILS_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring if configured
                if self.config.use_m1_optimization:
                    self.memory_optimizer.start_monitoring()
                    tprint_info("🧠 M1 hardware optimization enabled")
            except Exception as e:
                tprint_warning(f"Hardware optimization initialization failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _init_ml_utilities(self):
        """Initialize ML utilities if available."""
        if ML_UTILS_AVAILABLE:
            try:
                self.model_factory = create_model_factory()
                self.ensemble_manager = EnsembleManager()
                self.cross_validator = UnifiedCrossValidator()
                self.memory_optimizer_ml = MemoryOptimizer()
                self.parallel_processor = ParallelProcessor()
                self.lookahead_protection = LookaheadProtection()
                self.training_safeguards = MLTrainingSafeguards()
                self.error_handler = RobustErrorHandler()
                tprint_info("🤖 ML utilities initialized")
            except Exception as e:
                tprint_warning(f"ML utilities initialization failed: {e}")
                self.model_factory = None
                self.ensemble_manager = None
                self.cross_validator = None
                self.memory_optimizer_ml = None
                self.parallel_processor = None
                self.lookahead_protection = None
                self.training_safeguards = None
                self.error_handler = None
        else:
            self.model_factory = None
            self.ensemble_manager = None
            self.cross_validator = None
            self.memory_optimizer_ml = None
            self.parallel_processor = None
            self.lookahead_protection = None
            self.training_safeguards = None
            self.error_handler = None
    
    def optimize(self, 
                 data: pd.DataFrame, 
                 target_columns: List[str],
                 feature_columns: List[str],
                 strategy_func: Optional[Callable] = None) -> OptimizationResult:
        """
        Perform RL-NAS optimization.
        
        Args:
            data: Input data for optimization
            target_columns: Target columns for prediction
            feature_columns: Feature columns for input
            strategy_func: Optional strategy function for evaluation
            
        Returns:
            Optimization result
        """
        tprint_info("🎯 Starting RL-NAS optimization")
        self.start_time = time.time()
        
        # Validate input data
        if not self._validate_input_data(data, target_columns, feature_columns):
            raise ValueError("Invalid input data")
        
        # Initialize population
        self._initialize_population()
        
        # Main optimization loop
        for generation in range(self.config.max_generations):
            tprint_progress(generation + 1, self.config.max_generations, 
                          f"Generation {generation + 1}")
            
            # Evaluate population
            self._evaluate_population(data, target_columns, feature_columns, strategy_func)
            
            # Update Pareto front
            self._update_pareto_front()
            
            # Check convergence
            if self._check_convergence():
                tprint_info(f"✅ Convergence achieved at generation {generation + 1}")
                break
            
            # Selection and reproduction
            self._selection_and_reproduction()
            
            # Record generation info
            self._record_generation_info(generation)
            
            # Memory optimization
            if self.memory_optimizer:
                self._optimize_memory_usage()
        
        # Finalize optimization
        result = self._finalize_optimization()
        tprint_success("🎉 RL-NAS optimization completed")
        
        return result
    
    def _validate_input_data(self, 
                            data: pd.DataFrame, 
                            target_columns: List[str], 
                            feature_columns: List[str]) -> bool:
        """Validate input data for optimization."""
        try:
            # Check if data is valid DataFrame
            if not validate_dataframe(data):
                tprint_error("❌ Invalid DataFrame provided")
                return False
            
            # Check required columns
            all_columns = target_columns + feature_columns
            if not validate_dataframe_columns(data, all_columns):
                tprint_error("❌ Missing required columns")
                return False
            
            # Check data quality
            quality_report = create_data_quality_report(data)
            if quality_report.get('issues'):
                tprint_warning(f"⚠️ Data quality issues: {quality_report['issues']}")
            
            # Check for sufficient data
            if len(data) < 100:
                tprint_warning("⚠️ Limited data available for optimization")
            
            tprint_info(f"✅ Data validation passed: {data.shape[0]} rows, {data.shape[1]} columns")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _initialize_population(self):
        """Initialize population of neural architectures."""
        tprint_info("🧬 Initializing population")
        
        self.population = []
        for i in range(self.config.population_size):
            architecture = self._generate_random_architecture()
            self.population.append((architecture, {}))
        
        tprint_info(f"✅ Population initialized with {len(self.population)} architectures")
    
    def _generate_random_architecture(self) -> ArchitectureConfig:
        """Generate a random neural architecture."""
        # Random architecture type
        arch_type = np.random.choice(list(ArchitectureType))
        
        # Random hidden layers (1-5 layers)
        num_layers = np.random.randint(1, 6)
        hidden_layers = [np.random.randint(32, 512) for _ in range(num_layers)]
        
        # Random activation functions
        activations = ['relu', 'tanh', 'sigmoid', 'elu', 'swish']
        activation_functions = [np.random.choice(activations) for _ in range(num_layers)]
        
        # Random dropout rates
        dropout_rates = [np.random.uniform(0.1, 0.5) for _ in range(num_layers)]
        
        # Random regularization
        regularization = {
            'l1': np.random.uniform(0.0, 0.01),
            'l2': np.random.uniform(0.0, 0.01)
        }
        
        # Random hyperparameters
        learning_rate = 10 ** np.random.uniform(-4, -1)  # 0.0001 to 0.1
        batch_size = 2 ** np.random.randint(5, 11)  # 32 to 1024
        epochs = np.random.randint(50, 200)
        
        return ArchitectureConfig(
            architecture_type=arch_type,
            hidden_layers=hidden_layers,
            activation_functions=activation_functions,
            dropout_rates=dropout_rates,
            regularization=regularization,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs
        )
    
    def _evaluate_population(self, 
                           data: pd.DataFrame, 
                           target_columns: List[str], 
                           feature_columns: List[str],
                           strategy_func: Optional[Callable] = None):
        """Evaluate population fitness."""
        tprint_info("📊 Evaluating population")
        
        if self.config.parallel_evaluation and self.cpu_optimizer:
            # Parallel evaluation using M1 optimization
            self._evaluate_population_parallel(data, target_columns, feature_columns, strategy_func)
        else:
            # Sequential evaluation
            self._evaluate_population_sequential(data, target_columns, feature_columns, strategy_func)
    
    def _evaluate_population_parallel(self, 
                                    data: pd.DataFrame, 
                                    target_columns: List[str], 
                                    feature_columns: List[str],
                                    strategy_func: Optional[Callable] = None):
        """Evaluate population in parallel."""
        try:
            # Create evaluation tasks
            evaluation_tasks = []
            for i, (architecture, _) in enumerate(self.population):
                task = {
                    'index': i,
                    'architecture': architecture,
                    'data': data,
                    'target_columns': target_columns,
                    'feature_columns': feature_columns,
                    'strategy_func': strategy_func
                }
                evaluation_tasks.append(task)
            
            # Execute parallel evaluation
            if self.cpu_optimizer:
                results = parallel_map_m1(
                    self._evaluate_single_architecture,
                    evaluation_tasks
                )
            else:
                # Fallback to standard parallel processing
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(self._evaluate_single_architecture, evaluation_tasks))
            
            # Update population with results
            for i, result in enumerate(results):
                if i < len(self.population):
                    architecture, _ = self.population[i]
                    self.population[i] = (architecture, result)
            
            tprint_info("✅ Parallel evaluation completed")
            
        except Exception as e:
            tprint_warning(f"Parallel evaluation failed, falling back to sequential: {e}")
            self._evaluate_population_sequential(data, target_columns, feature_columns, strategy_func)
    
    def _evaluate_population_sequential(self, 
                                      data: pd.DataFrame, 
                                      target_columns: List[str], 
                                      feature_columns: List[str],
                                      strategy_func: Optional[Callable] = None):
        """Evaluate population sequentially."""
        for i, (architecture, _) in enumerate(self.population):
            try:
                task = {
                    'index': i,
                    'architecture': architecture,
                    'data': data,
                    'target_columns': target_columns,
                    'feature_columns': feature_columns,
                    'strategy_func': strategy_func
                }
                
                result = self._evaluate_single_architecture(task)
                self.population[i] = (architecture, result)
                
            except Exception as e:
                tprint_warning(f"Evaluation failed for architecture {i}: {e}")
                # Set default fitness values
                self.population[i] = (architecture, self._get_default_fitness())
    
    def _evaluate_single_architecture(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single architecture."""
        try:
            architecture = task['architecture']
            data = task['data']
            target_columns = task['target_columns']
            feature_columns = task['feature_columns']
            strategy_func = task.get('strategy_func')
            
            # Prepare data for evaluation
            X = data[feature_columns].values
            y = data[target_columns].values
            
            # Apply M1 optimization if available
            if self.gpu_manager and self.config.use_m1_optimization:
                X = self.gpu_manager.optimize_tensor_operations(X)
                y = self.gpu_manager.optimize_tensor_operations(y)
            
            # Perform cross-validation if available
            if self.cross_validator and self.config.temporal_validation:
                fitness = self._evaluate_with_cross_validation(
                    X, y, architecture, strategy_func
                )
            else:
                # Simple evaluation
                fitness = self._evaluate_simple(X, y, architecture, strategy_func)
            
            return fitness
            
        except Exception as e:
            tprint_warning(f"Single architecture evaluation failed: {e}")
            return self._get_default_fitness()
    
    def _evaluate_with_cross_validation(self, 
                                      X: np.ndarray, 
                                      y: np.ndarray, 
                                      architecture: ArchitectureConfig,
                                      strategy_func: Optional[Callable] = None) -> Dict[str, float]:
        """Evaluate architecture with cross-validation."""
        try:
            # Perform temporal cross-validation
            cv_results = temporal_cross_validation(
                X, y, 
                n_splits=self.config.cross_validation_folds,
                lookahead_protection=self.config.lookahead_protection
            )
            
            # Calculate fitness metrics
            fitness = {}
            for objective in self.config.objectives:
                if objective == OptimizationObjective.SHARPE_RATIO:
                    fitness[objective.value] = np.mean([r.get('sharpe_ratio', 0) for r in cv_results])
                elif objective == OptimizationObjective.MAX_DRAWDOWN:
                    fitness[objective.value] = -np.mean([r.get('max_drawdown', 0) for r in cv_results])  # Negative for minimization
                elif objective == OptimizationObjective.PROFIT_FACTOR:
                    fitness[objective.value] = np.mean([r.get('profit_factor', 1) for r in cv_results])
                elif objective == OptimizationObjective.WIN_RATE:
                    fitness[objective.value] = np.mean([r.get('win_rate', 0.5) for r in cv_results])
                elif objective == OptimizationObjective.TOTAL_RETURN:
                    fitness[objective.value] = np.mean([r.get('total_return', 0) for r in cv_results])
                else:
                    fitness[objective.value] = 0.0
            
            return fitness
            
        except Exception as e:
            tprint_warning(f"Cross-validation evaluation failed: {e}")
            return self._evaluate_simple(X, y, architecture, strategy_func)
    
    def _evaluate_simple(self, 
                        X: np.ndarray, 
                        y: np.ndarray, 
                        architecture: ArchitectureConfig,
                        strategy_func: Optional[Callable] = None) -> Dict[str, float]:
        """Simple evaluation without cross-validation."""
        try:
            # Split data for training and validation
            split_idx = int(len(X) * (1 - architecture.validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model (simplified)
            model = self._create_model(architecture)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            predictions = model.predict(X_val)
            
            # Calculate fitness metrics
            fitness = {}
            for objective in self.config.objectives:
                if objective == OptimizationObjective.SHARPE_RATIO:
                    fitness[objective.value] = self._calculate_sharpe_ratio(predictions, y_val)
                elif objective == OptimizationObjective.MAX_DRAWDOWN:
                    fitness[objective.value] = -self._calculate_max_drawdown(predictions, y_val)
                elif objective == OptimizationObjective.PROFIT_FACTOR:
                    fitness[objective.value] = self._calculate_profit_factor(predictions, y_val)
                elif objective == OptimizationObjective.WIN_RATE:
                    fitness[objective.value] = self._calculate_win_rate(predictions, y_val)
                elif objective == OptimizationObjective.TOTAL_RETURN:
                    fitness[objective.value] = self._calculate_total_return(predictions, y_val)
                else:
                    fitness[objective.value] = 0.0
            
            return fitness
            
        except Exception as e:
            tprint_warning(f"Simple evaluation failed: {e}")
            return self._get_default_fitness()
    
    def _create_model(self, architecture: ArchitectureConfig):
        """Create model based on architecture configuration."""
        # This is a simplified model creation
        # In practice, you would use the actual ML framework
        class SimpleModel:
            def __init__(self, config):
                self.config = config
                self.weights = None
                
            def fit(self, X, y):
                # Simplified training
                self.weights = np.random.randn(X.shape[1], y.shape[1])
                
            def predict(self, X):
                # Simplified prediction
                return X @ self.weights
        
        return SimpleModel(architecture)
    
    def _calculate_sharpe_ratio(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns = predictions - targets
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            return np.mean(returns) / np.std(returns)
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            returns = predictions - targets
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            return np.min(drawdown)
        except Exception:
            return 0.0
    
    def _calculate_profit_factor(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate profit factor."""
        try:
            returns = predictions - targets
            profits = returns[returns > 0]
            losses = returns[returns < 0]
            
            if len(losses) == 0 or np.sum(losses) == 0:
                return 1.0
            
            return np.sum(profits) / abs(np.sum(losses))
        except Exception:
            return 1.0
    
    def _calculate_win_rate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate win rate."""
        try:
            returns = predictions - targets
            wins = np.sum(returns > 0)
            total = len(returns)
            return wins / total if total > 0 else 0.5
        except Exception:
            return 0.5
    
    def _calculate_total_return(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate total return."""
        try:
            returns = predictions - targets
            return np.sum(returns)
        except Exception:
            return 0.0
    
    def _get_default_fitness(self) -> Dict[str, float]:
        """Get default fitness values."""
        fitness = {}
        for objective in self.config.objectives:
            if objective == OptimizationObjective.MAX_DRAWDOWN:
                fitness[objective.value] = -1.0  # Worst possible drawdown
            else:
                fitness[objective.value] = 0.0
        return fitness
    
    def _update_pareto_front(self):
        """Update Pareto front with non-dominated solutions."""
        try:
            # Get all evaluated architectures
            evaluated = [(arch, fitness) for arch, fitness in self.population if fitness]
            
            if not evaluated:
                return
            
            # Find non-dominated solutions
            non_dominated = []
            for arch, fitness in evaluated:
                is_dominated = False
                for other_arch, other_fitness in evaluated:
                    if arch != other_arch and self._dominates(other_fitness, fitness):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    non_dominated.append((arch, fitness))
            
            # Update Pareto front
            self.pareto_front = non_dominated
            tprint_info(f"📈 Pareto front updated with {len(non_dominated)} solutions")
            
        except Exception as e:
            tprint_warning(f"Pareto front update failed: {e}")
    
    def _dominates(self, fitness1: Dict[str, float], fitness2: Dict[str, float]) -> bool:
        """Check if fitness1 dominates fitness2."""
        try:
            better_count = 0
            worse_count = 0
            
            for objective in self.config.objectives:
                val1 = fitness1.get(objective.value, 0)
                val2 = fitness2.get(objective.value, 0)
                
                # For maximization objectives (higher is better)
                if objective in [OptimizationObjective.SHARPE_RATIO, 
                               OptimizationObjective.PROFIT_FACTOR,
                               OptimizationObjective.WIN_RATE,
                               OptimizationObjective.TOTAL_RETURN]:
                    if val1 > val2:
                        better_count += 1
                    elif val1 < val2:
                        worse_count += 1
                # For minimization objectives (lower is better)
                elif objective in [OptimizationObjective.MAX_DRAWDOWN]:
                    if val1 < val2:
                        better_count += 1
                    elif val1 > val2:
                        worse_count += 1
            
            # fitness1 dominates fitness2 if it's better in at least one objective
            # and not worse in any objective
            return better_count > 0 and worse_count == 0
            
        except Exception:
            return False
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        try:
            if len(self.optimization_history) < 2:
                return False
            
            # Check stagnation
            recent_generations = self.optimization_history[-self.config.max_stagnation:]
            if len(recent_generations) < self.config.max_stagnation:
                return False
            
            # Check if best fitness has improved significantly
            best_fitness_values = [gen.get('best_fitness', {}) for gen in recent_generations]
            if not best_fitness_values:
                return False
            
            # Calculate improvement for each objective
            improvements = []
            for objective in self.config.objectives:
                values = [fitness.get(objective.value, 0) for fitness in best_fitness_values]
                if len(values) >= 2:
                    improvement = abs(values[-1] - values[0])
                    improvements.append(improvement)
            
            # Check if all improvements are below threshold
            if improvements:
                max_improvement = max(improvements)
                return max_improvement < self.config.convergence_threshold
            
            return False
            
        except Exception as e:
            tprint_warning(f"Convergence check failed: {e}")
            return False
    
    def _selection_and_reproduction(self):
        """Perform selection and reproduction for next generation."""
        try:
            tprint_info("🔄 Performing selection and reproduction")
            
            # Sort population by fitness (multi-objective)
            self._sort_population()
            
            # Keep elite solutions
            elite_size = min(self.config.elite_size, len(self.population))
            elite = self.population[:elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.config.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if np.random.random() < self.config.mutation_rate:
                    offspring2 = self._mutate(offspring2)
                
                # Add offspring to new population
                new_population.extend([offspring1, offspring2])
            
            # Trim to population size
            self.population = new_population[:self.config.population_size]
            
            tprint_info(f"✅ New population created with {len(self.population)} individuals")
            
        except Exception as e:
            tprint_warning(f"Selection and reproduction failed: {e}")
    
    def _sort_population(self):
        """Sort population by fitness."""
        try:
            # Multi-objective sorting using Pareto ranking
            def pareto_rank(individual):
                arch, fitness = individual
                rank = 0
                for other_arch, other_fitness in self.population:
                    if individual != (other_arch, other_fitness):
                        if self._dominates(other_fitness, fitness):
                            rank += 1
                return rank
            
            self.population.sort(key=pareto_rank)
            
        except Exception as e:
            tprint_warning(f"Population sorting failed: {e}")
    
    def _tournament_selection(self) -> Tuple[ArchitectureConfig, Dict[str, float]]:
        """Perform tournament selection."""
        try:
            tournament_size = min(self.config.tournament_size, len(self.population))
            tournament = np.random.choice(len(self.population), tournament_size, replace=False)
            
            # Select best from tournament
            best_idx = tournament[0]
            best_fitness = self.population[best_idx][1]
            
            for idx in tournament[1:]:
                current_fitness = self.population[idx][1]
                if self._dominates(current_fitness, best_fitness):
                    best_idx = idx
                    best_fitness = current_fitness
            
            return self.population[best_idx]
            
        except Exception as e:
            tprint_warning(f"Tournament selection failed: {e}")
            # Return random individual as fallback
            return np.random.choice(self.population)
    
    def _crossover(self, parent1: Tuple[ArchitectureConfig, Dict[str, float]], 
                   parent2: Tuple[ArchitectureConfig, Dict[str, float]]) -> Tuple[Tuple, Tuple]:
        """Perform crossover between two parents."""
        try:
            arch1, fitness1 = parent1
            arch2, fitness2 = parent2
            
            # Create offspring architectures
            offspring1 = self._crossover_architectures(arch1, arch2)
            offspring2 = self._crossover_architectures(arch2, arch1)
            
            return (offspring1, {}), (offspring2, {})
            
        except Exception as e:
            tprint_warning(f"Crossover failed: {e}")
            return parent1, parent2
    
    def _crossover_architectures(self, arch1: ArchitectureConfig, arch2: ArchitectureConfig) -> ArchitectureConfig:
        """Crossover two architectures."""
        try:
            # Randomly select attributes from each parent
            new_arch = ArchitectureConfig(
                architecture_type=np.random.choice([arch1.architecture_type, arch2.architecture_type]),
                hidden_layers=self._crossover_lists(arch1.hidden_layers, arch2.hidden_layers),
                activation_functions=self._crossover_lists(arch1.activation_functions, arch2.activation_functions),
                dropout_rates=self._crossover_lists(arch1.dropout_rates, arch2.dropout_rates),
                regularization=self._crossover_dicts(arch1.regularization, arch2.regularization),
                learning_rate=np.random.choice([arch1.learning_rate, arch2.learning_rate]),
                batch_size=np.random.choice([arch1.batch_size, arch2.batch_size]),
                epochs=np.random.choice([arch1.epochs, arch2.epochs]),
                early_stopping_patience=np.random.choice([arch1.early_stopping_patience, arch2.early_stopping_patience]),
                validation_split=np.random.choice([arch1.validation_split, arch2.validation_split])
            )
            
            return new_arch
            
        except Exception as e:
            tprint_warning(f"Architecture crossover failed: {e}")
            return arch1
    
    def _crossover_lists(self, list1: List, list2: List) -> List:
        """Crossover two lists."""
        try:
            if not list1 or not list2:
                return list1 or list2
            
            # Randomly select elements from each list
            result = []
            max_len = max(len(list1), len(list2))
            
            for i in range(max_len):
                if i < len(list1) and i < len(list2):
                    result.append(np.random.choice([list1[i], list2[i]]))
                elif i < len(list1):
                    result.append(list1[i])
                else:
                    result.append(list2[i])
            
            return result
            
        except Exception:
            return list1
    
    def _crossover_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """Crossover two dictionaries."""
        try:
            result = {}
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                if key in dict1 and key in dict2:
                    result[key] = np.random.choice([dict1[key], dict2[key]])
                elif key in dict1:
                    result[key] = dict1[key]
                else:
                    result[key] = dict2[key]
            
            return result
            
        except Exception:
            return dict1
    
    def _mutate(self, individual: Tuple[ArchitectureConfig, Dict[str, float]]) -> Tuple:
        """Mutate an individual."""
        try:
            arch, fitness = individual
            
            # Create mutated architecture
            mutated_arch = self._mutate_architecture(arch)
            
            return (mutated_arch, {})
            
        except Exception as e:
            tprint_warning(f"Mutation failed: {e}")
            return individual
    
    def _mutate_architecture(self, arch: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture."""
        try:
            # Randomly mutate different aspects
            mutations = []
            
            # Mutate hidden layers
            if np.random.random() < 0.3:
                new_layers = arch.hidden_layers.copy()
                if new_layers:
                    idx = np.random.randint(len(new_layers))
                    new_layers[idx] = max(32, min(512, new_layers[idx] + np.random.randint(-64, 65)))
                mutations.append(('hidden_layers', new_layers))
            
            # Mutate learning rate
            if np.random.random() < 0.3:
                new_lr = arch.learning_rate * (2 ** np.random.uniform(-1, 1))
                new_lr = max(1e-5, min(1.0, new_lr))
                mutations.append(('learning_rate', new_lr))
            
            # Mutate batch size
            if np.random.random() < 0.3:
                new_batch = arch.batch_size * (2 ** np.random.randint(-1, 2))
                new_batch = max(16, min(2048, new_batch))
                mutations.append(('batch_size', int(new_batch)))
            
            # Mutate dropout rates
            if np.random.random() < 0.3:
                new_dropout = [max(0.0, min(0.8, rate + np.random.uniform(-0.1, 0.1))) 
                              for rate in arch.dropout_rates]
                mutations.append(('dropout_rates', new_dropout))
            
            # Apply mutations
            new_arch = ArchitectureConfig(
                architecture_type=arch.architecture_type,
                hidden_layers=next((val for key, val in mutations if key == 'hidden_layers'), arch.hidden_layers),
                activation_functions=arch.activation_functions,
                dropout_rates=next((val for key, val in mutations if key == 'dropout_rates'), arch.dropout_rates),
                regularization=arch.regularization,
                learning_rate=next((val for key, val in mutations if key == 'learning_rate'), arch.learning_rate),
                batch_size=next((val for key, val in mutations if key == 'batch_size'), arch.batch_size),
                epochs=arch.epochs,
                early_stopping_patience=arch.early_stopping_patience,
                validation_split=arch.validation_split
            )
            
            return new_arch
            
        except Exception as e:
            tprint_warning(f"Architecture mutation failed: {e}")
            return arch
    
    def _record_generation_info(self, generation: int):
        """Record information for current generation."""
        try:
            # Get best fitness values
            best_fitness = {}
            if self.population:
                # Find best individual (highest Pareto rank)
                best_individual = self.population[0]
                best_fitness = best_individual[1]
            
            # Record generation info
            gen_info = {
                'generation': generation,
                'best_fitness': best_fitness,
                'population_size': len(self.population),
                'pareto_front_size': len(self.pareto_front),
                'timestamp': time.time()
            }
            
            self.optimization_history.append(gen_info)
            
            # Log progress
            if best_fitness:
                fitness_str = ", ".join([f"{k}: {v:.4f}" for k, v in best_fitness.items()])
                tprint_info(f"📊 Generation {generation}: Best fitness - {fitness_str}")
            
        except Exception as e:
            tprint_warning(f"Generation info recording failed: {e}")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage during optimization."""
        try:
            if self.memory_optimizer:
                # Get memory stats
                memory_stats = self.memory_optimizer.get_memory_stats()
                
                # Record memory checkpoint
                checkpoint = {
                    'timestamp': time.time(),
                    'memory_usage': memory_stats,
                    'population_size': len(self.population),
                    'pareto_front_size': len(self.pareto_front)
                }
                self.memory_checkpoints.append(checkpoint)
                
                # Optimize memory if needed
                if memory_stats.get('memory_percent', 0) > 80:
                    tprint_info("🧠 High memory usage detected, optimizing...")
                    self.memory_optimizer.optimize_memory_usage(aggressive=True)
                
        except Exception as e:
            tprint_warning(f"Memory optimization failed: {e}")
    
    def _finalize_optimization(self) -> OptimizationResult:
        """Finalize optimization and create result."""
        try:
            execution_time = time.time() - self.start_time if self.start_time else 0.0
            
            # Get best architecture
            best_architecture = None
            best_fitness = {}
            if self.population:
                best_individual = self.population[0]
                best_architecture, best_fitness = best_individual
            
            # Get convergence info
            convergence_info = {
                'total_generations': len(self.optimization_history),
                'converged': self._check_convergence(),
                'final_population_size': len(self.population),
                'final_pareto_front_size': len(self.pareto_front)
            }
            
            # Get memory usage
            memory_usage = {}
            if self.memory_optimizer:
                memory_usage = self.memory_optimizer.get_memory_stats()
            
            # Get hardware utilization
            hardware_utilization = {}
            if self.gpu_manager:
                hardware_utilization['gpu_info'] = self.gpu_manager.get_gpu_info()
            if self.cpu_optimizer:
                hardware_utilization['cpu_info'] = self.cpu_optimizer.get_cpu_info()
            
            # Create result
            result = OptimizationResult(
                best_architecture=best_architecture or ArchitectureConfig(
                    architecture_type=ArchitectureType.FEEDFORWARD,
                    hidden_layers=[64],
                    activation_functions=['relu'],
                    dropout_rates=[0.2],
                    regularization={'l1': 0.0, 'l2': 0.0},
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=100
                ),
                best_fitness=best_fitness,
                pareto_front=self.pareto_front,
                optimization_history=self.optimization_history,
                convergence_info=convergence_info,
                execution_time=execution_time,
                memory_usage=memory_usage,
                hardware_utilization=hardware_utilization
            )
            
            tprint_success(f"🎉 Optimization completed in {execution_time:.2f} seconds")
            tprint_info(f"📊 Best fitness: {best_fitness}")
            tprint_info(f"🏆 Pareto front size: {len(self.pareto_front)}")
            
            return result
            
        except Exception as e:
            tprint_error(f"Result finalization failed: {e}")
            # Return minimal result
            return OptimizationResult(
                best_architecture=ArchitectureConfig(
                    architecture_type=ArchitectureType.FEEDFORWARD,
                    hidden_layers=[64],
                    activation_functions=['relu'],
                    dropout_rates=[0.2],
                    regularization={'l1': 0.0, 'l2': 0.0},
                    learning_rate=0.001,
                    batch_size=32,
                    epochs=100
                ),
                best_fitness={},
                pareto_front=[],
                optimization_history=[],
                convergence_info={'error': str(e)},
                execution_time=0.0,
                memory_usage={},
                hardware_utilization={}
            )
    
    def save_result(self, result: OptimizationResult, filepath: str) -> bool:
        """Save optimization result to file."""
        try:
            ensure_directory(Path(filepath).parent)
            
            # Convert result to dictionary
            result_dict = result.to_dict()
            
            # Save using appropriate serializer
            if filepath.endswith('.json'):
                return JSONSerializer.save(result_dict, filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                return PickleSerializer.save(result_dict, filepath)
            else:
                return self.serializer.save(result_dict, filepath)
                
        except Exception as e:
            tprint_error(f"Failed to save result: {e}")
            return False
    
    def load_result(self, filepath: str) -> Optional[OptimizationResult]:
        """Load optimization result from file."""
        try:
            # Load result dictionary
            if filepath.endswith('.json'):
                result_dict = JSONSerializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                result_dict = PickleSerializer.load(filepath)
            else:
                result_dict = self.serializer.load(filepath)
            
            if not result_dict:
                return None
            
            # Reconstruct result object
            best_arch_dict = result_dict['best_architecture']
            best_architecture = ArchitectureConfig(
                architecture_type=ArchitectureType(best_arch_dict['architecture_type']),
                hidden_layers=best_arch_dict['hidden_layers'],
                activation_functions=best_arch_dict['activation_functions'],
                dropout_rates=best_arch_dict['dropout_rates'],
                regularization=best_arch_dict['regularization'],
                learning_rate=best_arch_dict['learning_rate'],
                batch_size=best_arch_dict['batch_size'],
                epochs=best_arch_dict['epochs'],
                early_stopping_patience=best_arch_dict.get('early_stopping_patience', 10),
                validation_split=best_arch_dict.get('validation_split', 0.2)
            )
            
            # Reconstruct Pareto front
            pareto_front = []
            for arch_dict, fitness in result_dict['pareto_front']:
                arch = ArchitectureConfig(
                    architecture_type=ArchitectureType(arch_dict['architecture_type']),
                    hidden_layers=arch_dict['hidden_layers'],
                    activation_functions=arch_dict['activation_functions'],
                    dropout_rates=arch_dict['dropout_rates'],
                    regularization=arch_dict['regularization'],
                    learning_rate=arch_dict['learning_rate'],
                    batch_size=arch_dict['batch_size'],
                    epochs=arch_dict['epochs'],
                    early_stopping_patience=arch_dict.get('early_stopping_patience', 10),
                    validation_split=arch_dict.get('validation_split', 0.2)
                )
                pareto_front.append((arch, fitness))
            
            return OptimizationResult(
                best_architecture=best_architecture,
                best_fitness=result_dict['best_fitness'],
                pareto_front=pareto_front,
                optimization_history=result_dict['optimization_history'],
                convergence_info=result_dict['convergence_info'],
                execution_time=result_dict['execution_time'],
                memory_usage=result_dict['memory_usage'],
                hardware_utilization=result_dict['hardware_utilization']
            )
            
        except Exception as e:
            tprint_error(f"Failed to load result: {e}")
            return None
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of current optimization state."""
        try:
            summary = {
                'config': self.config.to_dict(),
                'population_size': len(self.population),
                'pareto_front_size': len(self.pareto_front),
                'generations_completed': len(self.optimization_history),
                'memory_checkpoints': len(self.memory_checkpoints),
                'hardware_available': {
                    'gpu_manager': self.gpu_manager is not None,
                    'memory_optimizer': self.memory_optimizer is not None,
                    'cpu_optimizer': self.cpu_optimizer is not None
                },
                'ml_utilities_available': {
                    'model_factory': self.model_factory is not None,
                    'ensemble_manager': self.ensemble_manager is not None,
                    'cross_validator': self.cross_validator is not None
                }
            }
            
            if self.optimization_history:
                latest_gen = self.optimization_history[-1]
                summary['latest_generation'] = latest_gen
            
            return summary
            
        except Exception as e:
            tprint_warning(f"Failed to get optimization summary: {e}")
            return {'error': str(e)}


# Convenience functions
def create_rl_nas_optimizer(objectives: List[OptimizationObjective], 
                           **kwargs) -> RL_NAS_Optimizer:
    """Create RL-NAS optimizer with default configuration."""
    config = OptimizationConfig(
        objectives=objectives,
        **kwargs
    )
    return RL_NAS_Optimizer(config)


def optimize_architecture(data: pd.DataFrame,
                         target_columns: List[str],
                         feature_columns: List[str],
                         objectives: List[OptimizationObjective],
                         strategy_func: Optional[Callable] = None,
                         **kwargs) -> OptimizationResult:
    """Convenience function for architecture optimization."""
    optimizer = create_rl_nas_optimizer(objectives, **kwargs)
    return optimizer.optimize(data, target_columns, feature_columns, strategy_func)


# Example usage
if __name__ == "__main__":
    # Example configuration
    objectives = [
        OptimizationObjective.SHARPE_RATIO,
        OptimizationObjective.MAX_DRAWDOWN,
        OptimizationObjective.PROFIT_FACTOR
    ]
    
    config = OptimizationConfig(
        objectives=objectives,
        max_generations=50,
        population_size=30,
        parallel_evaluation=True,
        use_m1_optimization=True
    )
    
    # Create optimizer
    optimizer = RL_NAS_Optimizer(config)
    
    # Example data (replace with actual data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_targets = 1
    
    data = pd.DataFrame({
        **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
        **{f'target_{i}': np.random.randn(n_samples) for i in range(n_targets)}
    })
    
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    target_columns = [f'target_{i}' for i in range(n_targets)]
    
    # Run optimization
    result = optimizer.optimize(data, target_columns, feature_columns)
    
    # Print results
    tprint_success("🎉 Optimization completed!")
    tprint_info(f"Best architecture: {result.best_architecture}")
    tprint_info(f"Best fitness: {result.best_fitness}")
    tprint_info(f"Execution time: {result.execution_time:.2f} seconds")
    tprint_info(f"Pareto front size: {len(result.pareto_front)}")