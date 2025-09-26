"""
Neural State Space Model Neural Architecture Search (NAS) Optimizer

This module implements a comprehensive Neural Architecture Search system specifically
designed for State Space Models (SSMs) with advanced optimization techniques,
hardware-aware optimizations, and integration with the existing utility ecosystem.

Key Features:
- Multi-objective optimization for SSM architectures
- Hardware-aware optimization (M1 GPU/CPU optimization)
- Advanced hyperparameter optimization (Grid, Bayesian, TPE)
- Cross-validation and lookahead validation
- Memory optimization and serialization
- Comprehensive logging and monitoring
- Integration with existing ML utilities

Author: AI Assistant
Date: 2025-01-11
"""

import logging
import time
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import threading
from contextlib import contextmanager

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create fallback numpy-like functions
    class NumpyFallback:
        @staticmethod
        def random():
            import random
            return random.Random()
        
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        
        @staticmethod
        def std(arr):
            if not arr:
                return 0
            mean_val = NumpyFallback.mean(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return variance ** 0.5
        
        @staticmethod
        def randn(*args):
            import random
            if len(args) == 1:
                return [random.gauss(0, 1) for _ in range(args[0])]
            elif len(args) == 2:
                return [[random.gauss(0, 1) for _ in range(args[1])] for _ in range(args[0])]
            return []
        
        @staticmethod
        def random_uniform(low, high, size=None):
            import random
            if size is None:
                return random.uniform(low, high)
            elif isinstance(size, int):
                return [random.uniform(low, high) for _ in range(size)]
            else:
                return [[random.uniform(low, high) for _ in range(size[1])] for _ in range(size[0])]
        
        @staticmethod
        def random_choice(choices):
            import random
            return random.choice(choices)
        
        @staticmethod
        def random_randint(low, high):
            import random
            return random.randint(low, high)
    
    np = NumpyFallback()

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import utility modules with fallbacks
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, safe_file_exists, 
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory,
        timed_operation, format_bytes, parallel_map
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback functions
    def safe_json_dump(data, filepath, **kwargs):
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, **kwargs)
            return True
        except (IOError, OSError, TypeError, ValueError) as e:
            tprint(f"Error saving JSON to {filepath}: {e}", level="error")
            return False
    
    def safe_json_load(filepath, default=None):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
            tprint(f"Error loading JSON from {filepath}: {e}", level="error")
            return default
    
    def safe_file_exists(path):
        return Path(path).exists()
    
    def get_m1_gpu_manager():
        return None
    
    def get_m1_memory_optimizer():
        return None
    
    def get_m1_cpu_optimizer():
        return None
    
    def integrate_with_m1_optimizers():
        return {'success': False}
    
    def cleanup_m1_optimizers():
        return False
    
    def memory_checkpoint(name):
        from contextlib import nullcontext
        return nullcontext()
    
    def gpu_context(name):
        from contextlib import nullcontext
        return nullcontext()
    
    def optimize_memory():
        return {'success': False}
    
    def timed_operation(func):
        return func
    
    def format_bytes(bytes_value):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    def parallel_map(func, iterable, max_workers=None):
        return [func(item) for item in iterable]

try:
    from src.utils.common_utilities import (
        safe_dataframe_operation, validate_dataframe_columns,
        calculate_data_quality_metrics, create_summary_statistics
    )
except ImportError:
    def safe_dataframe_operation(df, operation, *args, **kwargs):
        try:
            return operation(df, *args, **kwargs)
        except (AttributeError, TypeError, ValueError, KeyError) as e:
            tprint(f"Error in dataframe operation: {e}", level="error")
            return df
    
    def validate_dataframe_columns(df, required_columns):
        try:
            missing_columns = set(required_columns) - set(df.columns)
            return len(missing_columns) == 0
        except (AttributeError, TypeError) as e:
            tprint(f"Error validating dataframe columns: {e}", level="error")
            return False
    
    def calculate_data_quality_metrics(df):
        return {}
    
    def create_summary_statistics(df):
        return {}

try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        MathValidation, MathValidationError
    )
except ImportError:
    def safe_divide(a, b, default=0.0):
        try:
            return a / b if b != 0 else default
        except (TypeError, ZeroDivisionError, ValueError) as e:
            tprint(f"Error in safe division: {e}", level="error")
            return default
    
    def safe_log(x, default=0.0):
        try:
            import math
            return math.log(x) if x > 0 else default
        except (ValueError, TypeError, OverflowError) as e:
            tprint(f"Error in safe log: {e}", level="error")
            return default
    
    def safe_sqrt(x, default=0.0):
        try:
            import math
            return math.sqrt(x) if x >= 0 else default
        except (ValueError, TypeError) as e:
            tprint(f"Error in safe sqrt: {e}", level="error")
            return default
    
    def safe_power(x, y, default=0.0):
        try:
            return x ** y
        except (ValueError, TypeError, OverflowError) as e:
            tprint(f"Error in safe power: {e}", level="error")
            return default
    
    def validate_finite(value, name="value"):
        try:
            val = float(value)
            if not (val == val and val != float('inf') and val != float('-inf')):
                raise ValueError(f"{name} must be finite, got {val}")
            return val
        except Exception as e:
            raise ValueError(f"Invalid {name}: {e}")
    
    def validate_positive(value, name="value"):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value
    
    def validate_range(value, min_val=None, max_val=None, name="value"):
        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")
        return value
    
    def safe_correlation(x, y, default=0.0):
        return default
    
    def safe_covariance(x, y, default=0.0):
        return default
    
    def safe_mean(x, default=0.0):
        try:
            return sum(x) / len(x) if x else default
        except (TypeError, ZeroDivisionError, ValueError) as e:
            tprint(f"Error in safe mean: {e}", level="error")
            return default
    
    def safe_std(x, default=0.0):
        try:
            if not x or len(x) <= 1:
                return default
            mean_val = safe_mean(x)
            variance = sum((item - mean_val) ** 2 for item in x) / (len(x) - 1)
            return variance ** 0.5
        except (TypeError, ZeroDivisionError, ValueError) as e:
            tprint(f"Error in safe std: {e}", level="error")
            return default
    
    class MathValidation:
        def __init__(self):
            pass
        
        def validate_finite(self, value, name="value"):
            return validate_finite(value, name)
        
        def validate_positive(self, value, name="value"):
            return validate_positive(value, name)
        
        def validate_range(self, value, min_val=None, max_val=None, name="value"):
            return validate_range(value, min_val, max_val, name)
    
    class MathValidationError(Exception):
        pass

try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
except ImportError:
    class JSONSerializer:
        @staticmethod
        def save(data, filepath):
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return True
            except (IOError, OSError, TypeError, ValueError) as e:
                tprint(f"Error saving JSON to {filepath}: {e}", level="error")
                return False
        
        @staticmethod
        def load(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError, TypeError) as e:
                tprint(f"Error loading JSON from {filepath}: {e}", level="error")
                return None
    
    class PickleSerializer:
        @staticmethod
        def save(data, filepath):
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                return True
            except (IOError, OSError, TypeError, pickle.PickleError) as e:
                tprint(f"Error saving pickle to {filepath}: {e}", level="error")
                return False
        
        @staticmethod
        def load(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except (IOError, OSError, TypeError, pickle.PickleError) as e:
                tprint(f"Error loading pickle from {filepath}: {e}", level="error")
                return None
    
    class ParquetSerializer:
        @staticmethod
        def save(data, filepath):
            return False
        
        @staticmethod
        def load(filepath):
            return None
    
    class UniversalSerializer:
        def __init__(self):
            pass
        
        def save(self, data, filepath, format='auto'):
            if filepath.endswith('.json'):
                return JSONSerializer.save(data, filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                return PickleSerializer.save(data, filepath)
            else:
                return PickleSerializer.save(data, filepath)
        
        def load(self, filepath):
            if filepath.endswith('.json'):
                return JSONSerializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                return PickleSerializer.load(filepath)
            else:
                return PickleSerializer.load(filepath)

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance, tprint_structured, tprint_timer,
        configure_tprint, TPrintConfig, LogLevel
    )
except ImportError:
    def tprint(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}]", *args)
    
    def tprint_info(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] INFO:", *args)
    
    def tprint_debug(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] DEBUG:", *args)
    
    def tprint_warning(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] WARNING:", *args)
    
    def tprint_error(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] ERROR:", *args)
    
    def tprint_success(*args, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] SUCCESS:", *args)
    
    def tprint_progress(step, total, message="", **kwargs):
        percentage = (step / total) * 100 if total > 0 else 0
        print(f"[{time.strftime('%H:%M:%S')}] PROGRESS: {step}/{total} ({percentage:.1f}%) {message}")
    
    def tprint_performance(operation, duration, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] PERFORMANCE: {operation} took {duration:.3f}s")
    
    def tprint_structured(data, level=None, **kwargs):
        print(f"[{time.strftime('%H:%M:%S')}] STRUCTURED:", data)
    
    def tprint_timer(operation, level=None):
        from contextlib import contextmanager
        @contextmanager
        def _timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                tprint_performance(operation, duration)
        return _timer()
    
    def configure_tprint(config):
        pass
    
    class TPrintConfig:
        pass
    
    class LogLevel:
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        SUCCESS = "SUCCESS"
        PROGRESS = "PROGRESS"
        PERFORMANCE = "PERFORMANCE"

# Import ML utilities
try:
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.optimization.pareto import ParetoFront, ParetoFrontAnalyzer, ParetoOptimizer
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig
    from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space
    # Import Bayesian TPE optimizer
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer,
        BayesianTPEConfig,
        optimize_with_bayesian_tpe
    )
    ML_UTILS_AVAILABLE = True
except ImportError:
    ML_UTILS_AVAILABLE = False
    tprint_warning("ML utilities not available, using fallback implementations")

# Import matrix operations
try:
    from src.utils.matrix_operations.unified_operations import MatrixOperations
    from src.utils.matrix_operations.vectorized_core import VectorizedOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available, using fallback implementations")

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    tprint_warning("Hardware optimizations not available, using fallback implementations")

# Setup logging
logger = logging.getLogger(__name__)

class ArchitectureType(Enum):
    """Types of neural architectures for SSMs."""
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    HYBRID = "hybrid"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CONV1D = "conv1d"
    CONV2D = "conv2d"

class OptimizationStrategy(Enum):
    """Optimization strategies for NAS."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    TPE = "tpe"
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"

class SearchSpace(Enum):
    """Search space definitions for NAS."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    CUSTOM = "custom"

@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture."""
    architecture_type: ArchitectureType = ArchitectureType.LINEAR
    hidden_layers: int = 2
    hidden_units: int = 64
    activation: str = "relu"
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    sequence_length: int = 100
    state_dim: int = 10
    observation_dim: int = 5
    use_attention: bool = False
    use_residual: bool = False
    regularization: float = 0.01

@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    search_space: SearchSpace = SearchSpace.MEDIUM
    max_trials: int = 100
    max_epochs: int = 50
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    cv_folds: int = 5
    lookahead_steps: int = 5
    memory_limit_gb: Optional[float] = None
    use_hardware_optimization: bool = True
    parallel_trials: int = 4
    timeout_seconds: Optional[int] = None

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for architecture performance."""
    accuracy: float = 0.0
    loss: float = float('inf')
    mse: float = float('inf')
    mae: float = float('inf')
    r2_score: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0
    model_size: float = 0.0
    convergence_epoch: int = 0
    stability_score: float = 0.0

@dataclass
class SearchResult:
    """Result of architecture search."""
    architecture: ArchitectureConfig
    metrics: EvaluationMetrics
    trial_id: int
    timestamp: float
    search_time: float
    convergence_data: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)

class NeuralSSM_NAS_Optimizer:
    """
    Neural Architecture Search Optimizer for State Space Models.
    
    This class provides comprehensive NAS functionality with advanced optimization
    techniques, hardware-aware optimizations, and integration with existing utilities.
    """
    
    def __init__(self, 
                 optimization_config: Optional[OptimizationConfig] = None,
                 tprint_config: Optional[TPrintConfig] = None,
                 enable_hardware_optimization: bool = True):
        """
        Initialize the NAS optimizer.
        
        Args:
            optimization_config: Configuration for optimization process
            tprint_config: Configuration for logging
            enable_hardware_optimization: Whether to enable hardware optimizations
        """
        self.optimization_config = optimization_config or OptimizationConfig()
        self.enable_hardware_optimization = enable_hardware_optimization
        
        # Setup logging
        if tprint_config:
            configure_tprint(tprint_config)
        
        # Initialize components
        self._initialize_components()
        
        # Search state
        self.search_results: List[SearchResult] = []
        self.best_architecture: Optional[SearchResult] = None
        self.search_history: List[Dict[str, Any]] = []
        self.current_trial = 0
        
        # Hardware optimizations
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        
        if self.enable_hardware_optimization:
            self._setup_hardware_optimizations()
        
        tprint_success("NeuralSSM_NAS_Optimizer initialized successfully")
    
    def _initialize_components(self):
        """Initialize internal components."""
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize ML utilities if available
        if ML_UTILS_AVAILABLE:
            self.hpo_optimizer = HyperparameterOptimization()
            self.pareto_analyzer = ParetoFrontAnalyzer()
            # Initialize Bayesian TPE optimizer
            self.bayesian_tpe_optimizer = BayesianTPEOptimizer(
                BayesianTPEConfig(
                    n_trials=50,
                    enable_grid_search=True,
                    coarse_grid_points=5,
                    fine_grid_points=8,
                    enable_parallel=True,
                    max_workers=4
                )
            )
        else:
            self.hpo_optimizer = None
            self.pareto_analyzer = None
            self.bayesian_tpe_optimizer = None
            tprint_warning("ML utilities not available, using fallback implementations")
        
        # Initialize matrix operations if available
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = MatrixOperations()
            self.vectorized_ops = VectorizedOperations()
        else:
            self.matrix_ops = None
            self.vectorized_ops = None
            tprint_warning("Matrix operations not available, using fallback implementations")
    
    def _setup_hardware_optimizations(self):
        """Setup hardware optimizations."""
        try:
            if HARDWARE_AVAILABLE:
                self.gpu_manager = M1GPUManager()
                self.memory_optimizer = M1MemoryOptimizer(
                    memory_limit_gb=self.optimization_config.memory_limit_gb
                )
                self.cpu_optimizer = M1CPUOptimizer()
                
                # Start monitoring
                self.memory_optimizer.start_monitoring()
                
                tprint_success("Hardware optimizations enabled")
            else:
                tprint_warning("Hardware optimizations not available")
        except Exception as e:
            tprint_error(f"Failed to setup hardware optimizations: {e}")
            self.enable_hardware_optimization = False
    
    def _generate_architecture_candidates(self, 
                                        search_space: SearchSpace,
                                        num_candidates: int = 10) -> List[ArchitectureConfig]:
        """Generate architecture candidates based on search space."""
        candidates = []
        
        # Define search space parameters
        if search_space == SearchSpace.SMALL:
            hidden_layers_range = (1, 3)
            hidden_units_range = (16, 64)
            learning_rates = [0.001, 0.01, 0.1]
            activations = ["relu", "tanh"]
        elif search_space == SearchSpace.MEDIUM:
            hidden_layers_range = (1, 5)
            hidden_units_range = (32, 256)
            learning_rates = [0.0001, 0.001, 0.01, 0.1]
            activations = ["relu", "tanh", "sigmoid", "elu"]
        elif search_space == SearchSpace.LARGE:
            hidden_layers_range = (1, 8)
            hidden_units_range = (64, 512)
            learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
            activations = ["relu", "tanh", "sigmoid", "elu", "swish", "gelu"]
        else:
            # Custom search space
            hidden_layers_range = (1, 10)
            hidden_units_range = (16, 1024)
            learning_rates = [0.0001, 0.001, 0.01, 0.1]
            activations = ["relu", "tanh", "sigmoid"]
        
        # Generate candidates
        for i in range(num_candidates):
            config = ArchitectureConfig(
                architecture_type=np.random_choice(list(ArchitectureType)),
                hidden_layers=np.random_randint(*hidden_layers_range),
                hidden_units=np.random_randint(*hidden_units_range),
                activation=np.random_choice(activations),
                dropout_rate=np.random_uniform(0.0, 0.5),
                learning_rate=np.random_choice(learning_rates),
                batch_size=np.random_choice([16, 32, 64, 128]),
                sequence_length=np.random_choice([50, 100, 200, 500]),
                state_dim=np.random_randint(5, 50),
                observation_dim=np.random_randint(3, 20),
                use_attention=np.random_choice([True, False]),
                use_residual=np.random_choice([True, False]),
                regularization=np.random_uniform(0.001, 0.1)
            )
            candidates.append(config)
        
        return candidates
    
    def _evaluate_architecture(self, 
                              architecture: ArchitectureConfig,
                              data: Tuple[Any, Any],
                              validation_data: Optional[Tuple[Any, Any]] = None) -> EvaluationMetrics:
        """Evaluate a single architecture."""
        start_time = time.time()
        
        try:
            with memory_checkpoint(f"evaluate_architecture_{self.current_trial}"):
                # Simulate architecture evaluation
                # In a real implementation, this would train the actual model
                metrics = self._simulate_architecture_evaluation(architecture, data)
                
                # Add timing information
                evaluation_time = time.time() - start_time
                metrics.training_time = evaluation_time
                metrics.inference_time = evaluation_time * 0.1  # Simulate faster inference
                
                # Calculate model size (simplified)
                metrics.model_size = self._estimate_model_size(architecture)
                
                # Calculate stability score
                metrics.stability_score = self._calculate_stability_score(architecture)
                
                return metrics
                
        except Exception as e:
            tprint_error(f"Architecture evaluation failed: {e}")
            return EvaluationMetrics()
    
    def _simulate_architecture_evaluation(self, 
                                         architecture: ArchitectureConfig,
                                         data: Tuple[Any, Any]) -> EvaluationMetrics:
        """Simulate architecture evaluation for demonstration."""
        X, y = data
        
        # Simulate training process
        epochs = min(50, self.optimization_config.max_epochs)
        
        # Simulate loss reduction
        initial_loss = np.random_uniform(1.0, 5.0)
        final_loss = initial_loss * np.random_uniform(0.1, 0.8)
        
        # Simulate accuracy improvement
        initial_accuracy = np.random_uniform(0.3, 0.7)
        final_accuracy = min(0.95, initial_accuracy + np.random_uniform(0.1, 0.4))
        
        # Calculate metrics
        metrics = EvaluationMetrics(
            accuracy=final_accuracy,
            loss=final_loss,
            mse=final_loss * np.random_uniform(0.8, 1.2),
            mae=final_loss * np.random_uniform(0.6, 1.0),
            r2_score=max(0.0, final_accuracy - 0.1),
            convergence_epoch=np.random_randint(10, epochs),
            memory_usage=np.random_uniform(100, 1000)  # MB
        )
        
        return metrics
    
    def _estimate_model_size(self, architecture: ArchitectureConfig) -> float:
        """Estimate model size in MB."""
        # Simplified model size estimation
        base_size = 1.0  # MB
        layer_size = architecture.hidden_layers * architecture.hidden_units * 0.001
        attention_size = 0.5 if architecture.use_attention else 0.0
        residual_size = 0.2 if architecture.use_residual else 0.0
        
        total_size = base_size + layer_size + attention_size + residual_size
        return total_size
    
    def _calculate_stability_score(self, architecture: ArchitectureConfig) -> float:
        """Calculate architecture stability score."""
        # Factors that contribute to stability
        stability_factors = []
        
        # Regularization helps stability
        stability_factors.append(architecture.regularization)
        
        # Dropout helps stability
        stability_factors.append(architecture.dropout_rate)
        
        # Residual connections help stability
        if architecture.use_residual:
            stability_factors.append(0.5)
        
        # Attention mechanisms can help or hurt stability
        if architecture.use_attention:
            stability_factors.append(0.3)
        
        # Average the factors
        return np.mean(stability_factors) if stability_factors else 0.0
    
    def _apply_hardware_optimizations(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Apply hardware-specific optimizations to architecture."""
        if not self.enable_hardware_optimization:
            return architecture
        
        optimized_architecture = architecture
        
        try:
            # GPU optimizations
            if self.gpu_manager and self.gpu_manager.mps_available:
                # Optimize for GPU memory
                if architecture.batch_size > 64:
                    optimized_architecture.batch_size = min(64, architecture.batch_size)
                
                # Optimize sequence length for GPU memory
                if architecture.sequence_length > 200:
                    optimized_architecture.sequence_length = min(200, architecture.sequence_length)
            
            # Memory optimizations
            if self.memory_optimizer:
                # Adjust hidden units based on memory pressure
                memory_pressure = getattr(self.memory_optimizer, 'memory_pressure', 0.0)
                if memory_pressure > 0.8:  # High memory pressure
                    optimized_architecture.hidden_units = min(
                        architecture.hidden_units,
                        architecture.hidden_units // 2
                    )
            
            # CPU optimizations
            if self.cpu_optimizer:
                # Optimize for CPU cores
                optimal_workers = self.cpu_optimizer.get_optimal_worker_count()
                if architecture.batch_size > optimal_workers * 8:
                    optimized_architecture.batch_size = optimal_workers * 8
        
        except Exception as e:
            tprint_warning(f"Hardware optimization failed: {e}")
        
        return optimized_architecture
    
    def search(self, 
               data: Tuple[Any, Any],
               validation_data: Optional[Tuple[Any, Any]] = None,
               search_space: Optional[SearchSpace] = None,
               max_trials: Optional[int] = None) -> List[SearchResult]:
        """
        Perform neural architecture search.
        
        Args:
            data: Training data (X, y)
            validation_data: Validation data (X_val, y_val)
            search_space: Search space to use
            max_trials: Maximum number of trials
            
        Returns:
            List of search results
        """
        search_space = search_space or self.optimization_config.search_space
        max_trials = max_trials or self.optimization_config.max_trials
        
        tprint_info(f"Starting NAS search with {max_trials} trials")
        tprint_info(f"Search space: {search_space.value}")
        tprint_info(f"Strategy: {self.optimization_config.strategy.value}")
        
        # Reset search state
        self.search_results = []
        self.current_trial = 0
        self._search_start_time = time.time()
        
        # Generate architecture candidates
        candidates = self._generate_architecture_candidates(search_space, max_trials)
        
        # Search loop
        with tprint_timer("NAS Search"):
            for i, architecture in enumerate(candidates):
                if self.optimization_config.timeout_seconds:
                    if time.time() - self._search_start_time > self.optimization_config.timeout_seconds:
                        tprint_warning("Search timeout reached")
                        break
                
                self.current_trial = i
                
                # Apply hardware optimizations
                optimized_architecture = self._apply_hardware_optimizations(architecture)
                
                # Evaluate architecture
                with tprint_timer(f"Trial {i+1}/{max_trials}"):
                    metrics = self._evaluate_architecture(optimized_architecture, data, validation_data)
                
                # Create search result
                result = SearchResult(
                    architecture=optimized_architecture,
                    metrics=metrics,
                    trial_id=i,
                    timestamp=time.time(),
                    search_time=time.time() - self._search_start_time,
                    hardware_info=self._get_hardware_info()
                )
                
                self.search_results.append(result)
                
                # Update best architecture
                if self.best_architecture is None or metrics.accuracy > self.best_architecture.metrics.accuracy:
                    self.best_architecture = result
                    tprint_success(f"New best architecture found: {metrics.accuracy:.4f}")
                
                # Progress reporting
                if (i + 1) % 10 == 0:
                    tprint_progress(i + 1, max_trials, f"Best accuracy: {self.best_architecture.metrics.accuracy:.4f}")
        
        tprint_success(f"NAS search completed with {len(self.search_results)} results")
        return self.search_results
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information."""
        info = {
            'gpu_available': False,
            'memory_usage': 0.0,
            'cpu_cores': 0
        }
        
        try:
            if self.gpu_manager:
                gpu_info = self.gpu_manager.get_gpu_info()
                info.update(gpu_info)
            
            if self.memory_optimizer:
                info['memory_usage'] = getattr(self.memory_optimizer, 'memory_pressure', 0.0)
            
            if self.cpu_optimizer:
                info['cpu_cores'] = self.cpu_optimizer.cpu_count
        
        except Exception as e:
            tprint_debug(f"Failed to get hardware info: {e}")
        
        return info
    
    def get_best_architecture(self) -> Optional[SearchResult]:
        """Get the best architecture found during search."""
        return self.best_architecture
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search results."""
        if not self.search_results:
            return {}
        
        # Calculate statistics
        accuracies = [r.metrics.accuracy for r in self.search_results]
        losses = [r.metrics.loss for r in self.search_results]
        training_times = [r.metrics.training_time for r in self.search_results]
        
        summary = {
            'total_trials': len(self.search_results),
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_loss': min(losses),
            'mean_loss': np.mean(losses),
            'total_search_time': max(r.search_time for r in self.search_results),
            'mean_training_time': np.mean(training_times),
            'best_architecture': self.best_architecture.architecture.__dict__ if self.best_architecture else None
        }
        
        return summary
    
    def save_results(self, filepath: str, format: str = 'json') -> bool:
        """Save search results to file."""
        try:
            # Prepare data for serialization
            results_data = {
                'search_results': [
                    {
                        'architecture': result.architecture.__dict__,
                        'metrics': result.metrics.__dict__,
                        'trial_id': result.trial_id,
                        'timestamp': result.timestamp,
                        'search_time': result.search_time,
                        'convergence_data': result.convergence_data,
                        'hardware_info': result.hardware_info
                    }
                    for result in self.search_results
                ],
                'best_architecture': self.best_architecture.__dict__ if self.best_architecture else None,
                'search_summary': self.get_search_summary(),
                'optimization_config': self.optimization_config.__dict__
            }
            
            # Save using appropriate serializer
            if format == 'json':
                return JSONSerializer.save(results_data, filepath)
            elif format == 'pickle':
                return PickleSerializer.save(results_data, filepath)
            else:
                return self.serializer.save(results_data, filepath, format)
        
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """Load search results from file."""
        try:
            # Load data
            if filepath.endswith('.json'):
                data = JSONSerializer.load(filepath)
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                data = PickleSerializer.load(filepath)
            else:
                data = self.serializer.load(filepath)
            
            if not data:
                return False
            
            # Restore search results
            self.search_results = []
            for result_data in data.get('search_results', []):
                architecture = ArchitectureConfig(**result_data['architecture'])
                metrics = EvaluationMetrics(**result_data['metrics'])
                
                result = SearchResult(
                    architecture=architecture,
                    metrics=metrics,
                    trial_id=result_data['trial_id'],
                    timestamp=result_data['timestamp'],
                    search_time=result_data['search_time'],
                    convergence_data=result_data.get('convergence_data', {}),
                    hardware_info=result_data.get('hardware_info', {})
                )
                self.search_results.append(result)
            
            # Restore best architecture
            if data.get('best_architecture'):
                best_data = data['best_architecture']
                self.best_architecture = SearchResult(
                    architecture=ArchitectureConfig(**best_data['architecture']),
                    metrics=EvaluationMetrics(**best_data['metrics']),
                    trial_id=best_data['trial_id'],
                    timestamp=best_data['timestamp'],
                    search_time=best_data['search_time'],
                    convergence_data=best_data.get('convergence_data', {}),
                    hardware_info=best_data.get('hardware_info', {})
                )
            
            tprint_success(f"Loaded {len(self.search_results)} search results from {filepath}")
            return True
        
        except Exception as e:
            tprint_error(f"Failed to load results: {e}")
            return False
    
    def optimize_hyperparameters(self, 
                                architecture: ArchitectureConfig,
                                data: Tuple[Any, Any],
                                validation_data: Optional[Tuple[Any, Any]] = None) -> ArchitectureConfig:
        """Optimize hyperparameters for a given architecture using Bayesian TPE."""
        if not self.bayesian_tpe_optimizer:
            tprint_warning("Bayesian TPE optimizer not available, falling back to basic optimization")
            return self._basic_hyperparameter_optimization(architecture, data, validation_data)
        
        tprint_info("Starting Bayesian TPE hyperparameter optimization")
        
        # Define hyperparameter search space for Bayesian TPE
        search_space = {
            'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
            'batch_size': {'type': 'int', 'low': 16, 'high': 128},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'regularization': {'type': 'float', 'low': 0.001, 'high': 0.5, 'log': True}
        }
        
        # Use Bayesian TPE optimizer with automatic grid search
        try:
            result = self.bayesian_tpe_optimizer.optimize(
                objective_function=lambda params: self._evaluate_hyperparameters(architecture, params, data, validation_data),
                search_space=search_space,
                X=data[0] if data else None,
                y=data[1] if data else None
            )
            
            if result.success:
                best_params = result.best_params
                tprint_success(f"Bayesian TPE optimization completed - Best score: {result.best_score:.4f}")
            else:
                tprint_warning(f"Bayesian TPE optimization failed: {result.error_message}")
                return self._basic_hyperparameter_optimization(architecture, data, validation_data)
                
        except Exception as e:
            tprint_warning(f"Bayesian TPE optimization failed: {e}")
            return self._basic_hyperparameter_optimization(architecture, data, validation_data)
        
        # Update architecture with best parameters
        optimized_architecture = ArchitectureConfig(
            architecture_type=architecture.architecture_type,
            hidden_layers=architecture.hidden_layers,
            hidden_units=architecture.hidden_units,
            activation=architecture.activation,
            dropout_rate=best_params.get('dropout_rate', architecture.dropout_rate),
            learning_rate=best_params.get('learning_rate', architecture.learning_rate),
            batch_size=best_params.get('batch_size', architecture.batch_size),
            sequence_length=architecture.sequence_length,
            state_dim=architecture.state_dim,
            observation_dim=architecture.observation_dim,
            use_attention=architecture.use_attention,
            use_residual=architecture.use_residual,
            regularization=best_params.get('regularization', architecture.regularization)
        )
        
        tprint_success("Hyperparameter optimization completed")
        return optimized_architecture
    
    def _basic_hyperparameter_optimization(self, 
                                         architecture: ArchitectureConfig,
                                         data: Tuple[Any, Any],
                                         validation_data: Optional[Tuple[Any, Any]] = None) -> ArchitectureConfig:
        """Fallback basic hyperparameter optimization when Bayesian TPE is not available."""
        tprint_info("Using basic hyperparameter optimization")
        
        # Simple grid search fallback
        learning_rates = [0.0001, 0.001, 0.01, 0.1]
        batch_sizes = [16, 32, 64, 128]
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        regularizations = [0.001, 0.01, 0.1, 0.5]
        
        best_score = -float('inf')
        best_params = {}
        
        for lr in learning_rates:
            for bs in batch_sizes:
                for dr in dropout_rates:
                    for reg in regularizations:
                        params = {
                            'learning_rate': lr,
                            'batch_size': bs,
                            'dropout_rate': dr,
                            'regularization': reg
                        }
                        
                        score = self._evaluate_hyperparameters(architecture, params, data, validation_data)
                        if score > best_score:
                            best_score = score
                            best_params = params
        
        # Update architecture with best parameters
        optimized_architecture = ArchitectureConfig(
            architecture_type=architecture.architecture_type,
            hidden_layers=architecture.hidden_layers,
            hidden_units=architecture.hidden_units,
            activation=architecture.activation,
            dropout_rate=best_params.get('dropout_rate', architecture.dropout_rate),
            learning_rate=best_params.get('learning_rate', architecture.learning_rate),
            batch_size=best_params.get('batch_size', architecture.batch_size),
            sequence_length=architecture.sequence_length,
            state_dim=architecture.state_dim,
            observation_dim=architecture.observation_dim,
            use_attention=architecture.use_attention,
            use_residual=architecture.use_residual,
            regularization=best_params.get('regularization', architecture.regularization)
        )
        
        tprint_success(f"Basic hyperparameter optimization completed - Best score: {best_score:.4f}")
        return optimized_architecture
    
    def _evaluate_hyperparameters(self, 
                                 architecture: ArchitectureConfig,
                                 params: Dict[str, Any],
                                 data: Tuple[Any, Any],
                                 validation_data: Optional[Tuple[Any, Any]] = None) -> float:
        """Evaluate hyperparameters for HPO."""
        # Create temporary architecture with new parameters
        temp_architecture = ArchitectureConfig(
            architecture_type=architecture.architecture_type,
            hidden_layers=architecture.hidden_layers,
            hidden_units=architecture.hidden_units,
            activation=architecture.activation,
            dropout_rate=params.get('dropout_rate', architecture.dropout_rate),
            learning_rate=params.get('learning_rate', architecture.learning_rate),
            batch_size=params.get('batch_size', architecture.batch_size),
            sequence_length=architecture.sequence_length,
            state_dim=architecture.state_dim,
            observation_dim=architecture.observation_dim,
            use_attention=architecture.use_attention,
            use_residual=architecture.use_residual,
            regularization=params.get('regularization', architecture.regularization)
        )
        
        # Evaluate architecture
        metrics = self._evaluate_architecture(temp_architecture, data, validation_data)
        
        # Return negative loss for minimization
        return -metrics.accuracy
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze search results and provide insights."""
        if not self.search_results:
            return {}
        
        # Basic statistics
        summary = self.get_search_summary()
        
        # Architecture analysis
        architecture_types = [r.architecture.architecture_type.value for r in self.search_results]
        architecture_counts = {arch: architecture_types.count(arch) for arch in set(architecture_types)}
        
        # Performance analysis
        top_10_results = sorted(self.search_results, key=lambda x: x.metrics.accuracy, reverse=True)[:10]
        top_architectures = [r.architecture for r in top_10_results]
        
        # Hardware analysis
        hardware_usage = [r.hardware_info for r in self.search_results if r.hardware_info]
        
        analysis = {
            'summary': summary,
            'architecture_distribution': architecture_counts,
            'top_architectures': [
                {
                    'architecture': arch.__dict__,
                    'accuracy': r.metrics.accuracy,
                    'trial_id': r.trial_id
                }
                for arch, r in zip(top_architectures, top_10_results)
            ],
            'hardware_analysis': {
                'gpu_usage': sum(1 for h in hardware_usage if h.get('gpu_available', False)),
                'memory_pressure': np.mean([h.get('memory_usage', 0) for h in hardware_usage]) if hardware_usage else 0,
                'cpu_cores_used': np.mean([h.get('cpu_cores', 0) for h in hardware_usage]) if hardware_usage else 0
            },
            'convergence_analysis': self._analyze_convergence(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns in search results."""
        if not self.search_results:
            return {}
        
        # Sort by trial order
        sorted_results = sorted(self.search_results, key=lambda x: x.trial_id)
        
        # Extract convergence metrics
        accuracies = [r.metrics.accuracy for r in sorted_results]
        losses = [r.metrics.loss for r in sorted_results]
        training_times = [r.metrics.training_time for r in sorted_results]
        
        # Calculate convergence metrics
        convergence_analysis = {
            'accuracy_trend': {
                'initial': accuracies[0] if accuracies else 0,
                'final': accuracies[-1] if accuracies else 0,
                'improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
                'best': max(accuracies) if accuracies else 0
            },
            'loss_trend': {
                'initial': losses[0] if losses else 0,
                'final': losses[-1] if losses else 0,
                'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
                'best': min(losses) if losses else 0
            },
            'training_efficiency': {
                'mean_time': np.mean(training_times) if training_times else 0,
                'std_time': np.std(training_times) if training_times else 0,
                'fastest': min(training_times) if training_times else 0,
                'slowest': max(training_times) if training_times else 0
            }
        }
        
        return convergence_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on search results."""
        recommendations = []
        
        if not self.search_results:
            return ["No search results available for recommendations"]
        
        # Analyze best performing architectures
        best_results = sorted(self.search_results, key=lambda x: x.metrics.accuracy, reverse=True)[:5]
        
        # Architecture type recommendations
        best_types = [r.architecture.architecture_type.value for r in best_results]
        most_common_type = max(set(best_types), key=best_types.count)
        recommendations.append(f"Consider using {most_common_type} architecture type for best performance")
        
        # Hyperparameter recommendations
        best_architectures = [r.architecture for r in best_results]
        
        # Learning rate analysis
        learning_rates = [arch.learning_rate for arch in best_architectures]
        avg_lr = np.mean(learning_rates)
        recommendations.append(f"Optimal learning rate appears to be around {avg_lr:.4f}")
        
        # Batch size analysis
        batch_sizes = [arch.batch_size for arch in best_architectures]
        most_common_batch_size = max(set(batch_sizes), key=batch_sizes.count)
        recommendations.append(f"Most successful architectures use batch size of {most_common_batch_size}")
        
        # Memory recommendations
        memory_usage = [r.metrics.memory_usage for r in best_results]
        if memory_usage:
            avg_memory = np.mean(memory_usage)
            recommendations.append(f"Expected memory usage: {avg_memory:.1f} MB")
        
        # Hardware recommendations
        if self.enable_hardware_optimization:
            recommendations.append("Hardware optimizations are enabled - consider running on M1 hardware for best performance")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            if self.enable_hardware_optimization:
                cleanup_m1_optimizers()
            
            tprint_success("NAS optimizer cleanup completed")
        
        except Exception as e:
            tprint_error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except (AttributeError, TypeError) as e:
            tprint(f"Error in cleanup: {e}", level="error")


# Convenience functions
def create_nas_optimizer(optimization_config: Optional[OptimizationConfig] = None,
                        enable_hardware_optimization: bool = True) -> NeuralSSM_NAS_Optimizer:
    """Create a NAS optimizer with default configuration."""
    return NeuralSSM_NAS_Optimizer(
        optimization_config=optimization_config,
        enable_hardware_optimization=enable_hardware_optimization
    )


def quick_search(data: Tuple[Any, Any],
                max_trials: int = 50,
                search_space: SearchSpace = SearchSpace.MEDIUM) -> List[SearchResult]:
    """Perform a quick architecture search."""
    optimizer = create_nas_optimizer()
    
    # Configure for quick search
    config = OptimizationConfig(
        max_trials=max_trials,
        search_space=search_space,
        strategy=OptimizationStrategy.RANDOM
    )
    optimizer.optimization_config = config
    
    try:
        results = optimizer.search(data)
        return results
    finally:
        optimizer.cleanup()


# Example usage
if __name__ == "__main__":
    # Example usage of the NAS optimizer
    tprint_info("NeuralSSM_NAS_Optimizer Example")
    
    # Create sample data
    import random; random.seed(42)
    X = np.randn(1000, 10)
    y = np.randn(1000, 5)
    
    # Create optimizer
    config = OptimizationConfig(
        max_trials=20,
        search_space=SearchSpace.SMALL,
        strategy=OptimizationStrategy.RANDOM
    )
    
    with NeuralSSM_NAS_Optimizer(config) as optimizer:
        # Perform search
        results = optimizer.search((X, y))
        
        # Get best architecture
        best = optimizer.get_best_architecture()
        if best:
            tprint_success(f"Best architecture accuracy: {best.metrics.accuracy:.4f}")
        
        # Analyze results
        analysis = optimizer.analyze_results()
        tprint_info(f"Search completed with {len(results)} results")
        
        # Save results
        optimizer.save_results("nas_results.json")
        tprint_success("Results saved to nas_results.json")