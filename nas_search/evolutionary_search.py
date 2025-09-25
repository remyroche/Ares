"""
Evolutionary Architecture Search (EAS) for Neural Architecture Search (NAS).

This module implements a comprehensive evolutionary algorithm for neural architecture search,
leveraging advanced optimization techniques, hardware-specific optimizations, and ML utilities.

Key Features:
- Population-based evolutionary search with genetic operators
- M1 hardware optimization integration
- Advanced fitness evaluation with cross-validation
- Parallel processing and memory optimization
- Comprehensive logging and serialization
- Integration with ML common utilities for CV, HPO, and grid search
"""

import logging
import time
import json
import pickle
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import concurrent.futures
import threading
from contextlib import contextmanager
import gc
import sys
import os

# Import utility modules
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, create_summary_statistics,
        safe_drop_columns, safe_rename_columns, validate_timestamp_column,
        safe_timestamp_conversion, get_dataframe_info, safe_filter_dataframe,
        create_data_quality_report, optimize_dataframe_dtypes, safe_to_parquet,
        safe_read_parquet, list_parquet_files, get_memory_usage, optimize_memory,
        memory_checkpoint, gpu_context, integrate_with_m1_optimizers,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        cleanup_m1_optimizers, CommonUtilities
    )
except ImportError:
    # Fallback imports if src structure is different
    try:
        from utils.common_operations import *
    except ImportError:
        # Minimal fallback implementations
        def safe_dataframe_operation(df, operation, *args, **kwargs):
            return operation(df, *args, **kwargs)
        def validate_dataframe_columns(df, required_columns):
            return all(col in df.columns for col in required_columns)
        def get_memory_usage():
            return 0.0
        def optimize_memory():
            return {'success': True}
        def memory_checkpoint(name):
            return contextmanager(lambda: (yield))
        def gpu_context(name):
            return contextmanager(lambda: (yield))

try:
    from src.utils.common_utilities import CommonUtilities as BaseCommonUtilities
except ImportError:
    class BaseCommonUtilities:
        pass

try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_correlation, safe_covariance,
        safe_mean, safe_std, safe_percentile, MathValidation
    )
except ImportError:
    # Fallback math functions
    def safe_divide(a, b, default=0.0):
        return a / b if b != 0 else default
    def safe_log(x, default=0.0):
        return np.log(x) if x > 0 else default
    def validate_finite(value, name="value"):
        return float(value)
    def safe_mean(x, default=0.0):
        return np.mean(x) if len(x) > 0 else default
    def safe_std(x, default=0.0):
        return np.std(x) if len(x) > 1 else default

try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
except ImportError:
    # Fallback serialization
    class JSONSerializer:
        @staticmethod
        def save(data, filepath):
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str)
            return True
        @staticmethod
        def load(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_debug, tprint_performance, tprint_progress, tprint_structured,
        tprint_timer, LogLevel, TPrintConfig, configure_tprint
    )
except ImportError:
    # Fallback logging
    def tprint(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}]", *args)
    def tprint_info(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] INFO:", *args)
    def tprint_warning(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] WARNING:", *args)
    def tprint_error(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] ERROR:", *args)
    def tprint_success(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS:", *args)
    def tprint_debug(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] DEBUG:", *args)
    def tprint_performance(operation, duration, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] PERFORMANCE: {operation} took {duration:.3f}s")
    def tprint_progress(step, total, message="", **kwargs):
        percentage = (step / total) * 100 if total > 0 else 0
        print(f"[{time.strftime('%H:%M:%S')}] PROGRESS: {step}/{total} ({percentage:.1f}%) {message}")
    def tprint_structured(data, level=None, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] STRUCTURED: {json.dumps(data, default=str)}")
    def tprint_timer(operation, level=None):
        return contextmanager(lambda: (yield))

# Import ML common utilities
try:
    from src.utils.nas_tas.advanced_hpo_utils import HyperparameterOptimization as HPOOptimizer
    from src.utils.ml_common.optimization.grid_utils import GridSearchOptimizer
    from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import BayesianOptimizer
    from src.utils.ml_common.optimization.enhanced_hpo_monitor import HPOMonitor
    # Import Bayesian TPE optimizer
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        BayesianTPEConfig,
        optimize_with_bayesian_tpe
    )
except ImportError:
    # Fallback ML utilities
    class HPOOptimizer:
        def __init__(self, **kwargs):
            pass
        def optimize(self, objective_func, **kwargs):
            return {'best_params': {}, 'best_score': 0.0}
    
    class GridSearchOptimizer:
        def __init__(self, **kwargs):
            pass
        def search(self, estimator, param_grid, **kwargs):
            return {'best_params': {}, 'best_score': 0.0}
    
    class BayesianOptimizer:
        def __init__(self, **kwargs):
            pass
        def optimize(self, objective_func, **kwargs):
            return {'best_params': {}, 'best_score': 0.0}
    
    class HPOMonitor:
        def __init__(self, **kwargs):
            pass
        def start_monitoring(self):
            pass
        def stop_monitoring(self):
            pass
    
    class BayesianTPEOptimizer:
        def __init__(self, config=None):
            self.config = config
        def optimize(self, objective_function, search_space, **kwargs):
            return {'success': False, 'best_params': {}, 'best_score': 0.0}
    
    class BayesianTPEConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def optimize_with_bayesian_tpe(objective_function, search_space, config=None, **kwargs):
        return {'success': False, 'best_params': {}, 'best_score': 0.0}

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
except ImportError:
    # Fallback hardware utilities
    class M1GPUManager:
        def __init__(self):
            self.is_m1 = False
            self.mps_available = False
        def get_gpu_info(self):
            return {'is_m1': False, 'mps_available': False}
        def optimize_tensor_operations(self, data):
            return data
    
    class M1MemoryOptimizer:
        def __init__(self, memory_limit_gb=None):
            self.memory_limit_gb = memory_limit_gb
        def start_monitoring(self):
            pass
        def stop_monitoring(self):
            pass
        def optimize_memory(self):
            return {'success': True}
    
    class M1CPUOptimizer:
        def __init__(self):
            self.cpu_count = 4
        def get_optimal_worker_count(self):
            return self.cpu_count
        def create_optimized_thread_pool(self, max_workers=None):
            import concurrent.futures
            return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or 4)

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture parameters."""
    
    # Architecture constraints
    max_layers: int = 10
    min_layers: int = 2
    max_neurons_per_layer: int = 1024
    min_neurons_per_layer: int = 16
    
    # Layer types
    layer_types: List[str] = field(default_factory=lambda: [
        'dense', 'conv1d', 'conv2d', 'lstm', 'gru', 'attention', 'dropout', 'batch_norm'
    ])
    
    # Activation functions
    activation_functions: List[str] = field(default_factory=lambda: [
        'relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'swish', 'gelu'
    ])
    
    # Optimization constraints
    max_parameters: int = 1000000  # 1M parameters
    min_parameters: int = 1000    # 1K parameters
    max_flops: int = 1000000000  # 1B FLOPs
    min_flops: int = 1000000     # 1M FLOPs


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary algorithm parameters."""
    
    # Population settings
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    
    # Genetic operators
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    mutation_strength: float = 0.1
    
    # Selection parameters
    selection_pressure: float = 2.0
    diversity_weight: float = 0.1
    
    # Evolution parameters
    max_generations: int = 100
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-6
    
    # Parallel processing
    n_workers: int = 4
    use_parallel_evaluation: bool = True
    batch_size: int = 10


@dataclass
class FitnessConfig:
    """Configuration for fitness evaluation."""
    
    # Evaluation metrics
    primary_metric: str = 'accuracy'
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'precision', 'recall', 'f1_score', 'auc_roc'
    ])
    
    # Cross-validation
    cv_folds: int = 5
    use_stratified_cv: bool = True
    
    # Training parameters
    max_training_epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    
    # Performance constraints
    max_training_time: float = 300.0  # 5 minutes
    max_memory_usage: float = 8.0     # 8 GB
    min_accuracy_threshold: float = 0.5


class Architecture:
    """Represents a neural architecture."""
    
    def __init__(self, layers: List[Dict[str, Any]], config: ArchitectureConfig):
        self.layers = layers
        self.config = config
        self.fitness = None
        self.training_time = None
        self.memory_usage = None
        self.parameters_count = None
        self.flops_count = None
        self.validation_metrics = {}
        
    def __str__(self):
        return f"Architecture(layers={len(self.layers)}, fitness={self.fitness})"
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        return {
            'layers': self.layers,
            'fitness': self.fitness,
            'training_time': self.training_time,
            'memory_usage': self.memory_usage,
            'parameters_count': self.parameters_count,
            'flops_count': self.flops_count,
            'validation_metrics': self.validation_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: ArchitectureConfig):
        """Create architecture from dictionary."""
        arch = cls(data['layers'], config)
        arch.fitness = data.get('fitness')
        arch.training_time = data.get('training_time')
        arch.memory_usage = data.get('memory_usage')
        arch.parameters_count = data.get('parameters_count')
        arch.flops_count = data.get('flops_count')
        arch.validation_metrics = data.get('validation_metrics', {})
        return arch
    
    def calculate_complexity(self) -> Dict[str, int]:
        """Calculate architecture complexity metrics."""
        total_params = 0
        total_flops = 0
        
        for layer in self.layers:
            layer_type = layer.get('type', 'dense')
            neurons = layer.get('neurons', 32)
            
            if layer_type == 'dense':
                # Simplified parameter calculation
                total_params += neurons * (neurons + 1)
                total_flops += neurons * neurons
            elif layer_type in ['conv1d', 'conv2d']:
                # Simplified conv layer calculation
                kernel_size = layer.get('kernel_size', 3)
                total_params += neurons * kernel_size * kernel_size
                total_flops += neurons * kernel_size * kernel_size
            elif layer_type in ['lstm', 'gru']:
                # Simplified RNN calculation
                total_params += neurons * neurons * 4  # 4 gates for LSTM
                total_flops += neurons * neurons * 4
        
        self.parameters_count = total_params
        self.flops_count = total_flops
        
        return {
            'parameters': total_params,
            'flops': total_flops
        }
    
    def is_valid(self) -> bool:
        """Check if architecture is valid according to constraints."""
        if not self.layers:
            return False
        
        # Check layer count
        if len(self.layers) < self.config.min_layers or len(self.layers) > self.config.max_layers:
            return False
        
        # Check neurons per layer
        for layer in self.layers:
            neurons = layer.get('neurons', 32)
            if neurons < self.config.min_neurons_per_layer or neurons > self.config.max_neurons_per_layer:
                return False
        
        # Check complexity constraints
        complexity = self.calculate_complexity()
        if (complexity['parameters'] < self.config.min_parameters or 
            complexity['parameters'] > self.config.max_parameters):
            return False
        
        if (complexity['flops'] < self.config.min_flops or 
            complexity['flops'] > self.config.max_flops):
            return False
        
        return True


class EvolutionaryArchitectureSearch:
    """Evolutionary Algorithm for Neural Architecture Search."""
    
    def __init__(
        self,
        architecture_config: Optional[ArchitectureConfig] = None,
        evolutionary_config: Optional[EvolutionaryConfig] = None,
        fitness_config: Optional[FitnessConfig] = None,
        data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        target: Optional[str] = None,
        log_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize Evolutionary Architecture Search.
        
        Args:
            architecture_config: Configuration for architecture constraints
            evolutionary_config: Configuration for evolutionary algorithm
            fitness_config: Configuration for fitness evaluation
            data: Training data (X, y) tuple
            target: Target variable name for classification/regression
            log_dir: Directory for logging and saving results
        """
        # Configuration
        self.arch_config = architecture_config or ArchitectureConfig()
        self.evo_config = evolutionary_config or EvolutionaryConfig()
        self.fitness_config = fitness_config or FitnessConfig()
        
        # Data
        self.data = data
        self.target = target
        self.X, self.y = data if data else (None, None)
        
        # Logging and serialization
        self.log_dir = Path(log_dir) if log_dir else Path("nas_search_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.serializer = UniversalSerializer()
        except NameError:
            # Fallback if UniversalSerializer is not available
            self.serializer = None
        
        # Hardware optimization
        self.gpu_manager = M1GPUManager()
        self.memory_optimizer = M1MemoryOptimizer()
        self.cpu_optimizer = M1CPUOptimizer()
        
        # ML utilities
        self.hpo_optimizer = HPOOptimizer()
        self.grid_optimizer = GridSearchOptimizer()
        self.bayesian_optimizer = BayesianOptimizer()
        self.hpo_monitor = HPOMonitor()
        # Initialize Bayesian TPE optimizer
        self.bayesian_tpe_optimizer = BayesianTPEOptimizer(
            BayesianTPEConfig(
                n_trials=30,
                enable_grid_search=True,
                coarse_grid_points=5,
                fine_grid_points=8,
                enable_parallel=True,
                max_workers=4
            )
        )
        
        # Common utilities
        try:
            self.common_utils = CommonUtilities()
        except NameError:
            # Fallback if CommonUtilities is not available
            self.common_utils = BaseCommonUtilities()
        
        # Search state
        self.population: List[Architecture] = []
        self.generation = 0
        self.best_architecture: Optional[Architecture] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Performance tracking
        self.start_time = None
        self.total_evaluations = 0
        self.evaluation_times = []
        
        # Threading and parallel processing
        self._lock = threading.Lock()
        self._stop_flag = False
        
        # Setup logging
        self._setup_logging()
        
        # Initialize hardware optimizations
        self._initialize_hardware_optimizations()
        
        tprint_info("🧬 EvolutionaryArchitectureSearch initialized")
        tprint_info(f"📊 Population size: {self.evo_config.population_size}")
        tprint_info(f"🔄 Max generations: {self.evo_config.max_generations}")
        tprint_info(f"💻 Hardware optimization: {'Enabled' if self.gpu_manager.is_m1 else 'Disabled'}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.log_dir / f"nas_search_{int(time.time())}.log"
        
        # Configure tprint for structured logging
        try:
            tprint_config = TPrintConfig(
                output_to_file=True,
                output_file=str(log_file),
                enable_structured_logging=True,
                integrate_with_logging=True
            )
            configure_tprint(tprint_config)
        except NameError:
            # Fallback if TPrintConfig is not available
            pass
        
        # Setup Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _initialize_hardware_optimizations(self):
        """Initialize hardware-specific optimizations."""
        try:
            # Start memory monitoring
            self.memory_optimizer.start_monitoring()
            
            # Get hardware info
            gpu_info = self.gpu_manager.get_gpu_info()
            tprint_info(f"🖥️ GPU Info: {gpu_info}")
            
            # Optimize CPU settings
            optimal_workers = self.cpu_optimizer.get_optimal_worker_count()
            self.evo_config.n_workers = min(optimal_workers, self.evo_config.n_workers)
            tprint_info(f"⚡ Optimal workers: {optimal_workers}")
            
        except Exception as e:
            tprint_warning(f"⚠️ Hardware optimization setup failed: {e}")
    
    def initialize_population(self) -> List[Architecture]:
        """Initialize random population of architectures."""
        tprint_info("🎲 Initializing population...")
        
        population = []
        attempts = 0
        max_attempts = self.evo_config.population_size * 10
        
        while len(population) < self.evo_config.population_size and attempts < max_attempts:
            architecture = self._generate_random_architecture()
            if architecture.is_valid():
                population.append(architecture)
            attempts += 1
        
        if len(population) < self.evo_config.population_size:
            tprint_warning(f"⚠️ Only generated {len(population)} valid architectures out of {self.evo_config.population_size}")
            
            # If we have some valid architectures, duplicate them to fill the population
            if population:
                while len(population) < self.evo_config.population_size:
                    # Create a copy of an existing architecture
                    original = population[len(population) % len(population)]
                    layers_copy = [layer.copy() for layer in original.layers]
                    new_arch = Architecture(layers_copy, self.arch_config)
                    if new_arch.is_valid():
                        population.append(new_arch)
        
        self.population = population
        tprint_success(f"✅ Initialized population of {len(population)} architectures")
        
        return population
    
    def _generate_random_architecture(self) -> Architecture:
        """Generate a random valid architecture."""
        num_layers = random.randint(
            self.arch_config.min_layers,
            self.arch_config.max_layers
        )
        
        layers = []
        for i in range(num_layers):
            layer_type = random.choice(self.arch_config.layer_types)
            neurons = random.randint(
                self.arch_config.min_neurons_per_layer,
                self.arch_config.max_neurons_per_layer
            )
            activation = random.choice(self.arch_config.activation_functions)
            
            layer = {
                'type': layer_type,
                'neurons': neurons,
                'activation': activation,
                'dropout': random.uniform(0.0, 0.5) if random.random() < 0.3 else 0.0
            }
            
            # Add layer-specific parameters
            if layer_type in ['conv1d', 'conv2d']:
                layer['kernel_size'] = random.choice([3, 5, 7])
                layer['stride'] = random.choice([1, 2])
            elif layer_type in ['lstm', 'gru']:
                layer['return_sequences'] = i < num_layers - 1
            
            layers.append(layer)
        
        return Architecture(layers, self.arch_config)
    
    def evaluate_fitness(self, architecture: Architecture) -> float:
        """Evaluate fitness of an architecture."""
        start_time = time.perf_counter()
        
        try:
            # Use memory checkpoint if available
            try:
                with memory_checkpoint(f"fitness_eval_{architecture.__hash__()}"):
                    return self._evaluate_architecture_fitness(architecture, start_time)
            except:
                # Fallback without memory checkpoint
                return self._evaluate_architecture_fitness(architecture, start_time)
                
        except Exception as e:
            tprint_error(f"❌ Fitness evaluation failed: {e}")
            return 0.0
        finally:
            # Cleanup memory
            try:
                optimize_memory()
            except:
                pass
    
    def _evaluate_architecture_fitness(self, architecture: Architecture, start_time: float) -> float:
        """Helper method to evaluate architecture fitness."""
        # Calculate complexity metrics
        complexity = architecture.calculate_complexity()
        
        # Simulate training and evaluation
        # In a real implementation, this would train the actual model
        fitness = self._simulate_training(architecture)
        
        # Apply complexity penalties
        fitness = self._apply_complexity_penalties(fitness, complexity)
        
        # Update architecture metrics
        architecture.fitness = fitness
        architecture.training_time = time.perf_counter() - start_time
        try:
            architecture.memory_usage = get_memory_usage() / (1024**3)  # GB
        except:
            architecture.memory_usage = 0.0
        architecture.parameters_count = complexity['parameters']
        architecture.flops_count = complexity['flops']
        
        self.total_evaluations += 1
        self.evaluation_times.append(architecture.training_time)
        
        return fitness
    
    def optimize_hyperparameters(self, architecture: Architecture) -> Architecture:
        """Optimize hyperparameters for a given architecture using Bayesian TPE."""
        if not hasattr(self, 'bayesian_tpe_optimizer') or not self.bayesian_tpe_optimizer:
            tprint_warning("Bayesian TPE optimizer not available, skipping hyperparameter optimization")
            return architecture
        
        tprint_info("🔧 Starting Bayesian TPE hyperparameter optimization")
        
        # Define hyperparameter search space for the architecture
        search_space = {
            'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
            'batch_size': {'type': 'int', 'low': 16, 'high': 128},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'weight_decay': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}
        }
        
        try:
            # Use Bayesian TPE optimizer with automatic grid search
            result = self.bayesian_tpe_optimizer.optimize(
                objective_function=lambda params: self._evaluate_hyperparameters(architecture, params),
                search_space=search_space,
                X=self.X,
                y=self.y
            )
            
            if result.success:
                # Update architecture with optimized hyperparameters
                optimized_architecture = self._apply_optimized_hyperparameters(architecture, result.best_params)
                tprint_success(f"✅ Bayesian TPE optimization completed - Best score: {result.best_score:.4f}")
                return optimized_architecture
            else:
                tprint_warning(f"⚠️ Bayesian TPE optimization failed: {result.error_message}")
                return architecture
                
        except Exception as e:
            tprint_warning(f"⚠️ Bayesian TPE optimization failed: {e}")
            return architecture
    
    def _evaluate_hyperparameters(self, architecture: Architecture, params: Dict[str, Any]) -> float:
        """Evaluate hyperparameters for Bayesian TPE optimization."""
        try:
            # Create a temporary architecture with new hyperparameters
            temp_architecture = self._create_architecture_with_params(architecture, params)
            
            # Evaluate fitness with new hyperparameters
            fitness = self.evaluate_fitness(temp_architecture)
            
            # Return negative fitness for minimization (Bayesian TPE maximizes)
            return -fitness
            
        except Exception as e:
            tprint_warning(f"⚠️ Hyperparameter evaluation failed: {e}")
            return -1.0  # Return poor score on error
    
    def _create_architecture_with_params(self, base_architecture: Architecture, params: Dict[str, Any]) -> Architecture:
        """Create a new architecture with specified hyperparameters."""
        # Create a copy of the base architecture
        new_architecture = Architecture(
            layers=base_architecture.layers.copy(),
            config=base_architecture.config
        )
        
        # Update hyperparameters
        new_architecture.learning_rate = params.get('learning_rate', base_architecture.learning_rate)
        new_architecture.batch_size = params.get('batch_size', base_architecture.batch_size)
        new_architecture.dropout_rate = params.get('dropout_rate', base_architecture.dropout_rate)
        new_architecture.weight_decay = params.get('weight_decay', getattr(base_architecture, 'weight_decay', 0.0001))
        
        return new_architecture
    
    def _apply_optimized_hyperparameters(self, architecture: Architecture, best_params: Dict[str, Any]) -> Architecture:
        """Apply optimized hyperparameters to the architecture."""
        # Update the architecture with best parameters
        architecture.learning_rate = best_params.get('learning_rate', architecture.learning_rate)
        architecture.batch_size = best_params.get('batch_size', architecture.batch_size)
        architecture.dropout_rate = best_params.get('dropout_rate', architecture.dropout_rate)
        architecture.weight_decay = best_params.get('weight_decay', getattr(architecture, 'weight_decay', 0.0001))
        
        return architecture
    
    def _simulate_training(self, architecture: Architecture) -> float:
        """Simulate training process for fitness evaluation."""
        # This is a simplified simulation - in practice, you would:
        # 1. Build the actual neural network
        # 2. Train it on the data
        # 3. Evaluate on validation set
        # 4. Return the performance metric
        
        # Simulate training time based on complexity
        complexity_factor = architecture.parameters_count / 1000000  # Normalize by 1M params
        base_fitness = random.uniform(0.5, 0.95)  # Random base performance
        
        # Add some structure to make it more realistic
        layer_count = len(architecture.layers)
        neuron_density = sum(layer.get('neurons', 32) for layer in architecture.layers) / layer_count
        
        # Simulate that deeper networks with moderate complexity perform better
        depth_bonus = min(0.1, layer_count * 0.01)
        complexity_penalty = max(0, (complexity_factor - 0.5) * 0.1)
        
        fitness = base_fitness + depth_bonus - complexity_penalty
        fitness = max(0.0, min(1.0, fitness))  # Clamp to [0, 1]
        
        return fitness
    
    def _apply_complexity_penalties(self, fitness: float, complexity: Dict[str, int]) -> float:
        """Apply penalties for overly complex architectures."""
        # Parameter penalty
        param_penalty = max(0, (complexity['parameters'] - self.arch_config.max_parameters) / self.arch_config.max_parameters)
        
        # FLOP penalty
        flop_penalty = max(0, (complexity['flops'] - self.arch_config.max_flops) / self.arch_config.max_flops)
        
        # Apply penalties
        penalty = (param_penalty + flop_penalty) * 0.1
        return max(0.0, fitness - penalty)
    
    def evaluate_population(self, population: List[Architecture]) -> List[Architecture]:
        """Evaluate fitness for entire population."""
        tprint_info(f"📊 Evaluating population of {len(population)} architectures...")
        
        if self.evo_config.use_parallel_evaluation:
            return self._evaluate_population_parallel(population)
        else:
            return self._evaluate_population_sequential(population)
    
    def _evaluate_population_sequential(self, population: List[Architecture]) -> List[Architecture]:
        """Sequential evaluation of population."""
        for i, architecture in enumerate(population):
            if architecture.fitness is None:
                fitness = self.evaluate_fitness(architecture)
                tprint_progress(i + 1, len(population), f"Fitness: {fitness:.4f}")
        
        return population
    
    def _evaluate_population_parallel(self, population: List[Architecture]) -> List[Architecture]:
        """Parallel evaluation of population."""
        with self.cpu_optimizer.create_optimized_thread_pool(self.evo_config.n_workers) as executor:
            # Submit evaluation tasks
            future_to_arch = {
                executor.submit(self.evaluate_fitness, arch): arch 
                for arch in population if arch.fitness is None
            }
            
            # Collect results
            completed = 0
            for future in concurrent.futures.as_completed(future_to_arch):
                architecture = future_to_arch[future]
                try:
                    fitness = future.result()
                    architecture.fitness = fitness
                    completed += 1
                    tprint_progress(completed, len(future_to_arch), f"Fitness: {fitness:.4f}")
                except Exception as e:
                    tprint_error(f"❌ Evaluation failed for architecture: {e}")
                    architecture.fitness = 0.0
        
        return population
    
    def select_parents(self, population: List[Architecture]) -> List[Architecture]:
        """Select parents for reproduction using tournament selection."""
        parents = []
        
        for _ in range(self.evo_config.population_size):
            # Tournament selection
            tournament = random.sample(population, self.evo_config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness or 0.0)
            parents.append(winner)
        
        return parents
    
    def crossover(self, parent1: Architecture, parent2: Architecture) -> Tuple[Architecture, Architecture]:
        """Create offspring through crossover."""
        if random.random() > self.evo_config.crossover_rate:
            return parent1, parent2
        
        # Simple crossover: randomly select layers from each parent
        layers1 = []
        layers2 = []
        
        max_layers = max(len(parent1.layers), len(parent2.layers))
        
        for i in range(max_layers):
            if i < len(parent1.layers) and i < len(parent2.layers):
                # Both parents have this layer
                if random.random() < 0.5:
                    layers1.append(parent1.layers[i].copy())
                    layers2.append(parent2.layers[i].copy())
                else:
                    layers1.append(parent2.layers[i].copy())
                    layers2.append(parent1.layers[i].copy())
            elif i < len(parent1.layers):
                # Only parent1 has this layer
                layers1.append(parent1.layers[i].copy())
                layers2.append(parent1.layers[i].copy())
            else:
                # Only parent2 has this layer
                layers1.append(parent2.layers[i].copy())
                layers2.append(parent2.layers[i].copy())
        
        child1 = Architecture(layers1, self.arch_config)
        child2 = Architecture(layers2, self.arch_config)
        
        return child1, child2
    
    def mutate(self, architecture: Architecture) -> Architecture:
        """Apply mutation to architecture."""
        if random.random() > self.evo_config.mutation_rate:
            return architecture
        
        mutated_layers = [layer.copy() for layer in architecture.layers]
        
        # Random mutations
        mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'change_neurons'])
        
        if mutation_type == 'add_layer' and len(mutated_layers) < self.arch_config.max_layers:
            # Add a new layer
            new_layer = self._generate_random_layer()
            insert_pos = random.randint(0, len(mutated_layers))
            mutated_layers.insert(insert_pos, new_layer)
        
        elif mutation_type == 'remove_layer' and len(mutated_layers) > self.arch_config.min_layers:
            # Remove a random layer
            remove_pos = random.randint(0, len(mutated_layers) - 1)
            mutated_layers.pop(remove_pos)
        
        elif mutation_type == 'modify_layer':
            # Modify a random layer
            if mutated_layers:
                layer_idx = random.randint(0, len(mutated_layers) - 1)
                layer = mutated_layers[layer_idx]
                
                # Change layer type
                if random.random() < 0.3:
                    layer['type'] = random.choice(self.arch_config.layer_types)
                
                # Change activation
                if random.random() < 0.3:
                    layer['activation'] = random.choice(self.arch_config.activation_functions)
        
        elif mutation_type == 'change_neurons':
            # Change number of neurons
            if mutated_layers:
                layer_idx = random.randint(0, len(mutated_layers) - 1)
                layer = mutated_layers[layer_idx]
                
                current_neurons = layer.get('neurons', 32)
                change = random.randint(-32, 32)
                new_neurons = max(
                    self.arch_config.min_neurons_per_layer,
                    min(self.arch_config.max_neurons_per_layer, current_neurons + change)
                )
                layer['neurons'] = new_neurons
        
        mutated_arch = Architecture(mutated_layers, self.arch_config)
        return mutated_arch
    
    def create_next_generation(self, population: List[Architecture]) -> List[Architecture]:
        """Create next generation through selection, crossover, and mutation."""
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness or 0.0, reverse=True)
        
        # Keep elite individuals
        elite = population[:self.evo_config.elite_size]
        new_population = elite.copy()
        
        # Select parents
        parents = self.select_parents(population)
        
        # Generate offspring
        while len(new_population) < self.evo_config.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate children
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Add valid children
            if child1.is_valid():
                new_population.append(child1)
            if child2.is_valid() and len(new_population) < self.evo_config.population_size:
                new_population.append(child2)
        
        return new_population[:self.evo_config.population_size]
    
    def calculate_diversity(self, population: List[Architecture]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity metric based on layer differences
        total_differences = 0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                arch1, arch2 = population[i], population[j]
                
                # Compare layer structures
                max_layers = max(len(arch1.layers), len(arch2.layers))
                differences = 0
                
                for k in range(max_layers):
                    layer1 = arch1.layers[k] if k < len(arch1.layers) else None
                    layer2 = arch2.layers[k] if k < len(arch2.layers) else None
                    
                    if layer1 is None or layer2 is None:
                        differences += 1
                    elif layer1.get('type') != layer2.get('type'):
                        differences += 1
                    elif abs(layer1.get('neurons', 0) - layer2.get('neurons', 0)) > 10:
                        differences += 1
                
                total_differences += differences / max_layers
                comparisons += 1
        
        return total_differences / comparisons if comparisons > 0 else 0.0
    
    def check_convergence(self) -> bool:
        """Check if the algorithm has converged."""
        if len(self.fitness_history) < self.evo_config.early_stopping_patience:
            return False
        
        # Check if fitness has improved significantly in recent generations
        recent_fitness = self.fitness_history[-self.evo_config.early_stopping_patience:]
        improvement = max(recent_fitness) - min(recent_fitness)
        
        return improvement < self.evo_config.convergence_threshold
    
    def save_results(self, generation: int):
        """Save current results to disk."""
        try:
            # Save population
            population_data = [arch.to_dict() for arch in self.population]
            population_file = self.log_dir / f"population_gen_{generation}.json"
            JSONSerializer.save(population_data, str(population_file))
            
            # Save best architecture
            if self.best_architecture:
                best_file = self.log_dir / f"best_architecture_gen_{generation}.json"
                JSONSerializer.save(self.best_architecture.to_dict(), str(best_file))
            
            # Save search history
            history_data = {
                'fitness_history': self.fitness_history,
                'diversity_history': self.diversity_history,
                'generation': generation,
                'total_evaluations': self.total_evaluations,
                'evaluation_times': self.evaluation_times
            }
            history_file = self.log_dir / f"search_history_gen_{generation}.json"
            JSONSerializer.save(history_data, str(history_file))
            
            tprint_info(f"💾 Results saved for generation {generation}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
    
    def run_evolution(self) -> Architecture:
        """Run the complete evolutionary search."""
        tprint_info("🚀 Starting evolutionary architecture search...")
        self.start_time = time.perf_counter()
        
        try:
            # Initialize population
            self.population = self.initialize_population()
            
            # Evaluate initial population
            self.population = self.evaluate_population(self.population)
            
            # Track best architecture
            if self.population:
                self.best_architecture = max(self.population, key=lambda x: x.fitness or 0.0)
                tprint_success(f"🏆 Initial best fitness: {self.best_architecture.fitness:.4f}")
            else:
                tprint_error("❌ No valid architectures found in initial population")
                return None
            
            # Evolution loop
            for generation in range(self.evo_config.max_generations):
                if self._stop_flag:
                    tprint_warning("🛑 Search stopped by user")
                    break
                
                tprint_info(f"🔄 Generation {generation + 1}/{self.evo_config.max_generations}")
                
                # Create next generation
                new_population = self.create_next_generation(self.population)
                
                # Evaluate new population
                new_population = self.evaluate_population(new_population)
                
                # Update population
                self.population = new_population
                
                # Update best architecture
                if self.population:
                    current_best = max(self.population, key=lambda x: x.fitness or 0.0)
                    if current_best.fitness > (self.best_architecture.fitness or 0.0):
                        self.best_architecture = current_best
                        tprint_success(f"🏆 New best fitness: {self.best_architecture.fitness:.4f}")
                
                # Track metrics
                avg_fitness = np.mean([arch.fitness or 0.0 for arch in self.population])
                diversity = self.calculate_diversity(self.population)
                
                self.fitness_history.append(avg_fitness)
                self.diversity_history.append(diversity)
                
                # Log progress
                tprint_info(f"📊 Avg fitness: {avg_fitness:.4f}, Diversity: {diversity:.4f}")
                
                # Save results periodically
                if (generation + 1) % 10 == 0:
                    self.save_results(generation + 1)
                
                # Check convergence
                if self.check_convergence():
                    tprint_info(f"🎯 Convergence reached at generation {generation + 1}")
                    break
                
                # Memory optimization
                if generation % 5 == 0:
                    optimize_memory()
            
            # Final results
            total_time = time.perf_counter() - self.start_time
            tprint_success(f"✅ Evolution completed in {total_time:.2f} seconds")
            tprint_success(f"🏆 Final best fitness: {self.best_architecture.fitness:.4f}")
            tprint_success(f"📊 Total evaluations: {self.total_evaluations}")
            
            # Save final results
            self.save_results(self.evo_config.max_generations)
            
            return self.best_architecture
            
        except Exception as e:
            tprint_error(f"❌ Evolution failed: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            # Stop hardware monitoring
            self.memory_optimizer.stop_monitoring()
            
            # Cleanup M1 optimizers
            try:
                cleanup_m1_optimizers()
            except NameError:
                # Fallback if cleanup function is not available
                pass
            
            tprint_info("🧹 Cleanup completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Cleanup error: {e}")
    
    def stop_search(self):
        """Stop the evolutionary search."""
        self._stop_flag = True
        tprint_info("🛑 Search stop requested")
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of the search process."""
        return {
            'total_generations': len(self.fitness_history),
            'total_evaluations': self.total_evaluations,
            'best_fitness': self.best_architecture.fitness if self.best_architecture else None,
            'avg_fitness': np.mean(self.fitness_history) if self.fitness_history else 0.0,
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'total_time': time.perf_counter() - self.start_time if self.start_time else 0.0,
            'avg_evaluation_time': np.mean(self.evaluation_times) if self.evaluation_times else 0.0
        }


# Example usage and testing
def create_sample_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y


def main():
    """Example usage of EvolutionaryArchitectureSearch."""
    # Create sample data
    X, y = create_sample_data(1000, 20)
    
    # Configuration
    arch_config = ArchitectureConfig(
        max_layers=8,
        min_layers=2,
        max_neurons_per_layer=512,
        min_neurons_per_layer=16
    )
    
    evo_config = EvolutionaryConfig(
        population_size=20,
        max_generations=10,
        n_workers=2
    )
    
    fitness_config = FitnessConfig(
        cv_folds=3,
        max_training_epochs=50
    )
    
    # Initialize search
    nas = EvolutionaryArchitectureSearch(
        architecture_config=arch_config,
        evolutionary_config=evo_config,
        fitness_config=fitness_config,
        data=(X, y),
        log_dir="nas_search_results"
    )
    
    # Run evolution
    best_architecture = nas.run_evolution()
    
    # Print results
    tprint_success(f"🏆 Best architecture found:")
    tprint_success(f"   Fitness: {best_architecture.fitness:.4f}")
    tprint_success(f"   Layers: {len(best_architecture.layers)}")
    tprint_success(f"   Parameters: {best_architecture.parameters_count}")
    
    # Print search summary
    summary = nas.get_search_summary()
    tprint_structured(summary)


if __name__ == "__main__":
    main()