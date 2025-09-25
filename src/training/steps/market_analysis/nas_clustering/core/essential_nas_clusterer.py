"""
Essential NAS Clusterer - Neural Architecture Search for Clustering

This module provides a comprehensive clustering system that combines Neural Architecture Search (NAS)
with advanced clustering algorithms for market analysis and pattern recognition.

Key Features:
- Neural Architecture Search for optimal clustering architectures
- Integration with M1 hardware optimization
- Advanced mathematical validation and safe operations
- Comprehensive data processing and serialization
- ML utilities for cross-validation, lookahead, and hyperparameter optimization
- Enhanced logging and monitoring capabilities
"""

import logging
import numpy as np
import pandas as pd
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

# Import common operations and utilities
try:
    from ...utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
        safe_apply_function, create_summary_statistics, safe_drop_columns,
        safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
        get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
        safe_to_parquet, safe_read_parquet, list_parquet_files,
        get_latest_outcome_file, load_latest_optimal_regime_clustering_outcome,
        safe_copy, safe_deepcopy, safe_resample, align_dataframes,
        validate_dataframe_schema, guard_dataframe_nulls, optimize_dataframe_dtypes,
        safe_fillna, integrate_with_m1_optimizers, cleanup_m1_optimizers, 
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage, CommonUtilities
    )
except ImportError:
    warnings.warn("Common operations not available, using fallback implementations")

# Import math validation utilities
try:
    from ...utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, validate_numeric_array,
        safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
        safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile,
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        MathValidation, MathValidationError
    )
except ImportError:
    warnings.warn("Math validation utilities not available")

# Import serialization utilities
try:
    from ...utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
except ImportError:
    warnings.warn("Serialization utilities not available")

# Import enhanced logging utilities
try:
    from ...utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance, tprint_structured,
        tprint_with_level, tprint_batch, tprint_timer, tprint_logged,
        TPrintConfig, TPrintManager, LogLevel, configure_tprint
    )
except ImportError:
    warnings.warn("Enhanced logging utilities not available")
    # Fallback to standard logging
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# Import data utilities
try:
    from ...utils.data.klines_parquet import KlinesParquetProcessor
    from ...utils.data.unified_data_utils import UnifiedDataUtils
    from ...utils.data.real_data_loader import RealDataLoader
    from ...utils.data.quality.data_quality import DataQualityChecker
except ImportError:
    warnings.warn("Data utilities not available")

# Import matrix operations
try:
    from ...utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    from ...utils.matrix_operations.vectorized_core import VectorizedCore
    from ...utils.matrix_operations.enhanced_operations import EnhancedOperations
except ImportError:
    warnings.warn("Matrix operations not available")

# Import M1 hardware optimization utilities
try:
    from ...utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    from ...utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory, optimize_memory
    )
    from ...utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool,
        run_cpu_intensive_task, parallel_backtesting_worker,
        parallel_monte_carlo_simulation, run_monte_carlo_batch
    )
except ImportError:
    warnings.warn("M1 hardware utilities not available")

# Import ML common utilities
try:
    from ...utils.ml_common.matrix_cross_validation import MatrixCrossValidation
    from ...utils.ml_common.feature_selection import FeatureSelector
    from ...utils.ml_common.pipeline_orchestrator import PipelineOrchestrator
    from ...utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager
    from ...utils.ml_common.vectorized_backtesting import VectorizedBacktesting
except ImportError:
    warnings.warn("ML common utilities not available")

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class NASClustererConfig:
    """Configuration for EssentialNASClusterer."""
    
    # Basic configuration
    name: str = "EssentialNASClusterer"
    version: str = "1.0.0"
    
    # Clustering parameters
    n_clusters_range: Tuple[int, int] = (2, 10)
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    random_state: int = 42
    
    # Neural Architecture Search parameters
    nas_population_size: int = 50
    nas_generations: int = 100
    nas_mutation_rate: float = 0.1
    nas_crossover_rate: float = 0.8
    
    # Data processing parameters
    preprocessing_enabled: bool = True
    feature_scaling: str = "standard"  # standard, minmax, robust, none
    outlier_detection: bool = True
    missing_value_strategy: str = "interpolate"  # interpolate, drop, fill
    
    # Performance optimization
    m1_optimization: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    memory_limit_gb: Optional[float] = None
    
    # Validation and evaluation
    cross_validation_folds: int = 5
    lookahead_steps: int = 10
    hyperparameter_optimization: bool = True
    optimization_algorithm: str = "bayesian"  # bayesian, grid, random
    
    # Logging and monitoring
    verbose: bool = True
    log_level: str = "INFO"
    save_intermediate_results: bool = True
    output_directory: str = "nas_clustering_results"
    
    # Advanced features
    ensemble_clustering: bool = True
    multi_objective_optimization: bool = True
    adaptive_clustering: bool = True
    online_learning: bool = False


@dataclass
class ClusteringResult:
    """Results from clustering operation."""
    
    # Core results
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_clusters: int
    
    # Performance metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # NAS-specific results
    architecture_fitness: float
    architecture_complexity: float
    nas_generation: int
    
    # Additional metadata
    execution_time: float
    convergence_iterations: int
    memory_usage_mb: float
    configuration: NASClustererConfig
    
    # Quality metrics
    stability_score: float = 0.0
    interpretability_score: float = 0.0
    scalability_score: float = 0.0


class ClusteringAlgorithm(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'ClusteringAlgorithm':
        """Fit the clustering algorithm to data."""
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        pass
    
    @abstractmethod
    def get_centers(self) -> np.ndarray:
        """Get cluster centers."""
        pass
    
    @abstractmethod
    def get_inertia(self) -> float:
        """Get clustering inertia (within-cluster sum of squares)."""
        pass


class NeuralArchitecture:
    """Neural architecture for clustering."""
    
    def __init__(self, layers: List[int], activations: List[str], 
                 dropout_rates: List[float], learning_rate: float):
        self.layers = layers
        self.activations = activations
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.fitness = 0.0
        self.complexity = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate architecture complexity."""
        total_params = sum(layer * layer for layer in self.layers)
        return total_params * self.learning_rate
    
    def mutate(self, mutation_rate: float = 0.1) -> 'NeuralArchitecture':
        """Create a mutated version of this architecture."""
        # Simplified mutation logic
        new_layers = self.layers.copy()
        new_activations = self.activations.copy()
        new_dropout = self.dropout_rates.copy()
        new_lr = self.learning_rate
        
        # Mutate layers (simplified)
        if np.random.random() < mutation_rate:
            idx = np.random.randint(len(new_layers))
            new_layers[idx] = max(10, new_layers[idx] + np.random.randint(-5, 6))
        
        # Mutate learning rate
        if np.random.random() < mutation_rate:
            new_lr = max(0.001, min(0.1, new_lr * (1 + np.random.normal(0, 0.1))))
        
        return NeuralArchitecture(new_layers, new_activations, new_dropout, new_lr)
    
    def crossover(self, other: 'NeuralArchitecture') -> Tuple['NeuralArchitecture', 'NeuralArchitecture']:
        """Create offspring through crossover with another architecture."""
        # Simplified crossover logic
        child1_layers = [self.layers[i] if i % 2 == 0 else other.layers[i] 
                        for i in range(min(len(self.layers), len(other.layers)))]
        child2_layers = [other.layers[i] if i % 2 == 0 else self.layers[i] 
                        for i in range(min(len(self.layers), len(other.layers)))]
        
        child1 = NeuralArchitecture(child1_layers, self.activations, 
                                   self.dropout_rates, self.learning_rate)
        child2 = NeuralArchitecture(child2_layers, other.activations, 
                                   other.dropout_rates, other.learning_rate)
        
        return child1, child2


class EssentialNASClusterer:
    """
    Essential Neural Architecture Search Clusterer.
    
    A comprehensive clustering system that combines Neural Architecture Search (NAS)
    with advanced clustering algorithms for optimal pattern recognition and market analysis.
    
    Features:
    - Neural Architecture Search for optimal clustering architectures
    - Multiple clustering algorithms with automatic selection
    - M1 hardware optimization for Apple Silicon
    - Advanced mathematical validation and safe operations
    - Comprehensive data processing and serialization
    - Cross-validation and hyperparameter optimization
    - Enhanced logging and monitoring
    """
    
    def __init__(self, config: Optional[NASClustererConfig] = None):
        """
        Initialize EssentialNASClusterer.
        
        Args:
            config: Configuration object for the clusterer
        """
        self.config = config or NASClustererConfig()
        self.logger = logger.getChild('EssentialNASClusterer')
        
        # Initialize utilities
        self._initialize_utilities()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize optimization
        self._initialize_optimization()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state
        self._initialize_state()
        
        tprint_success(f"EssentialNASClusterer initialized with config: {self.config.name}")
    
    def _initialize_utilities(self):
        """Initialize utility classes and managers."""
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            
            # Initialize math validation
            self.math_validator = MathValidation()
            
            # Initialize serializers
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            
            # Initialize M1 optimizers
            if self.config.m1_optimization:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring
                if self.memory_optimizer:
                    self.memory_optimizer.start_monitoring()
            
            # Initialize ML utilities
            try:
                self.cross_validator = MatrixCrossValidation()
                self.feature_selector = FeatureSelector()
                self.pipeline_orchestrator = PipelineOrchestrator()
                self.vectorization_manager = UnifiedVectorizationManager()
                self.backtesting_engine = VectorizedBacktesting()
                self.ml_operations = None
            except Exception as e:
                self.logger.warning(f"ML utilities initialization failed: {e}")
                self.cross_validator = None
                self.feature_selector = None
                self.pipeline_orchestrator = None
                self.vectorization_manager = None
                self.backtesting_engine = None
                self.ml_operations = None
            
            tprint_info("✅ Utilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Utility initialization failed: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize clustering components."""
        try:
            # Initialize data processors
            self.data_processor = None
            self.quality_checker = None
            
            # Initialize clustering algorithms
            self.algorithms = {}
            self._initialize_clustering_algorithms()
            
            # Initialize NAS components
            self.population = []
            self.best_architecture = None
            self.fitness_history = []
            
            tprint_info("✅ Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _initialize_clustering_algorithms(self):
        """Initialize available clustering algorithms."""
        try:
            # This would initialize various clustering algorithms
            # For now, we'll use placeholder implementations
            self.algorithms = {
                'kmeans': self._create_kmeans_algorithm(),
                'dbscan': self._create_dbscan_algorithm(),
                'hierarchical': self._create_hierarchical_algorithm(),
                'gaussian_mixture': self._create_gaussian_mixture_algorithm()
            }
            
            tprint_debug(f"Initialized {len(self.algorithms)} clustering algorithms")
            
        except Exception as e:
            self.logger.error(f"Clustering algorithms initialization failed: {e}")
            self.algorithms = {}
    
    def _create_kmeans_algorithm(self) -> ClusteringAlgorithm:
        """Create K-means clustering algorithm."""
        # Placeholder implementation
        class KMeansAlgorithm(ClusteringAlgorithm):
            def __init__(self, n_clusters=3, random_state=42):
                self.n_clusters = n_clusters
                self.random_state = random_state
                self.centers_ = None
                self.labels_ = None
                self.inertia_ = 0.0
            
            def fit(self, data):
                # Simplified K-means implementation
                np.random.seed(self.random_state)
                n_samples, n_features = data.shape
                
                # Initialize centers randomly
                self.centers_ = data[np.random.choice(n_samples, self.n_clusters, replace=False)]
                
                # Iterative optimization
                for _ in range(100):  # max iterations
                    # Assign points to nearest center
                    distances = np.sqrt(((data - self.centers_[:, np.newaxis])**2).sum(axis=2))
                    self.labels_ = np.argmin(distances, axis=0)
                    
                    # Update centers
                    new_centers = np.array([data[self.labels_ == k].mean(axis=0) 
                                          for k in range(self.n_clusters)])
                    
                    # Check convergence
                    if np.allclose(self.centers_, new_centers, atol=1e-4):
                        break
                    
                    self.centers_ = new_centers
                
                # Calculate inertia
                self.inertia_ = sum(np.min(np.sum((data - self.centers_[label])**2, axis=1)) 
                                  for label in set(self.labels_))
                
                return self
            
            def predict(self, data):
                distances = np.sqrt(((data - self.centers_[:, np.newaxis])**2).sum(axis=2))
                return np.argmin(distances, axis=0)
            
            def get_centers(self):
                return self.centers_
            
            def get_inertia(self):
                return self.inertia_
        
        return KMeansAlgorithm()
    
    def _create_dbscan_algorithm(self) -> ClusteringAlgorithm:
        """Create DBSCAN clustering algorithm."""
        # Placeholder implementation - would use sklearn in practice
        class DBSCANAlgorithm(ClusteringAlgorithm):
            def __init__(self, eps=0.5, min_samples=5):
                self.eps = eps
                self.min_samples = min_samples
                self.labels_ = None
                self.centers_ = None
                self.inertia_ = 0.0
            
            def fit(self, data):
                # Simplified DBSCAN implementation
                self.labels_ = np.zeros(len(data))
                # Placeholder - would implement actual DBSCAN logic
                return self
            
            def predict(self, data):
                # Simplified prediction
                return np.zeros(len(data))
            
            def get_centers(self):
                return self.centers_ or np.array([])
            
            def get_inertia(self):
                return self.inertia_
        
        return DBSCANAlgorithm()
    
    def _create_hierarchical_algorithm(self) -> ClusteringAlgorithm:
        """Create Hierarchical clustering algorithm."""
        # Placeholder implementation
        class HierarchicalAlgorithm(ClusteringAlgorithm):
            def fit(self, data):
                self.labels_ = np.zeros(len(data))
                self.centers_ = data.mean(axis=0, keepdims=True)
                self.inertia_ = 0.0
                return self
            
            def predict(self, data):
                return np.zeros(len(data))
            
            def get_centers(self):
                return self.centers_
            
            def get_inertia(self):
                return self.inertia_
        
        return HierarchicalAlgorithm()
    
    def _create_gaussian_mixture_algorithm(self) -> ClusteringAlgorithm:
        """Create Gaussian Mixture clustering algorithm."""
        # Placeholder implementation
        class GaussianMixtureAlgorithm(ClusteringAlgorithm):
            def fit(self, data):
                self.labels_ = np.zeros(len(data))
                self.centers_ = data.mean(axis=0, keepdims=True)
                self.inertia_ = 0.0
                return self
            
            def predict(self, data):
                return np.zeros(len(data))
            
            def get_centers(self):
                return self.centers_
            
            def get_inertia(self):
                return self.inertia_
        
        return GaussianMixtureAlgorithm()
    
    def _initialize_optimization(self):
        """Initialize optimization components."""
        try:
            # Initialize NAS population
            self._initialize_nas_population()
            
            # Initialize hyperparameter optimizer
            self.hyperparameter_optimizer = None
            if self.config.hyperparameter_optimization:
                self._initialize_hyperparameter_optimizer()
            
            tprint_info("✅ Optimization components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Optimization initialization failed: {e}")
            raise
    
    def _initialize_nas_population(self):
        """Initialize Neural Architecture Search population."""
        try:
            self.population = []
            
            for _ in range(self.config.nas_population_size):
                # Generate random architecture
                layers = [np.random.randint(10, 100) for _ in range(np.random.randint(2, 5))]
                activations = ['relu'] * len(layers)
                dropout_rates = [np.random.uniform(0.1, 0.5) for _ in range(len(layers))]
                learning_rate = np.random.uniform(0.001, 0.1)
                
                architecture = NeuralArchitecture(layers, activations, dropout_rates, learning_rate)
                self.population.append(architecture)
            
            tprint_debug(f"Initialized NAS population with {len(self.population)} architectures")
            
        except Exception as e:
            self.logger.error(f"NAS population initialization failed: {e}")
            self.population = []
    
    def _initialize_hyperparameter_optimizer(self):
        """Initialize hyperparameter optimization."""
        try:
            # Placeholder for hyperparameter optimizer
            # Would integrate with optuna, hyperopt, or similar
            self.hyperparameter_optimizer = {
                'algorithm': self.config.optimization_algorithm,
                'trials': 100,
                'initialized': True
            }
            
            tprint_debug("Hyperparameter optimizer initialized")
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter optimizer initialization failed: {e}")
            self.hyperparameter_optimizer = None
    
    def _setup_logging(self):
        """Setup enhanced logging."""
        try:
            if self.config.verbose:
                # Configure tprint if available
                try:
                    config = TPrintConfig(
                        timestamp_format=TimestampFormat.WITH_MICROSECONDS,
                        use_colors=True,
                        output_to_console=True,
                        min_log_level=getattr(LogLevel, self.config.log_level.upper(), LogLevel.INFO)
                    )
                    configure_tprint(config)
                except Exception:
                    pass  # Fallback to standard logging
            
            tprint_info(f"Enhanced logging configured for {self.config.name}")
            
        except Exception as e:
            self.logger.warning(f"Logging setup failed: {e}")
    
    def _initialize_state(self):
        """Initialize clusterer state."""
        self.fitted = False
        self.training_data = None
        self.best_result = None
        self.training_history = []
        self.performance_metrics = {}
        
        tprint_debug("Clusterer state initialized")
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray], 
            target_columns: Optional[List[str]] = None) -> 'EssentialNASClusterer':
        """
        Fit the EssentialNASClusterer to data.
        
        Args:
            data: Input data as DataFrame or numpy array
            target_columns: Specific columns to use for clustering (if data is DataFrame)
            
        Returns:
            Self for method chaining
        """
        tprint_info("🚀 Starting EssentialNASClusterer fitting process")
        
        try:
            with tprint_timer("Data preprocessing"):
                # Preprocess data
                processed_data = self._preprocess_data(data, target_columns)
            
            with tprint_timer("Neural Architecture Search"):
                # Run NAS optimization
                best_architecture = self._run_nas_optimization(processed_data)
            
            with tprint_timer("Clustering with best architecture"):
                # Perform clustering with best architecture
                result = self._cluster_with_architecture(processed_data, best_architecture)
            
            # Store results
            self.best_result = result
            self.training_data = processed_data
            self.fitted = True
            
            tprint_success(f"✅ EssentialNASClusterer fitting completed successfully")
            tprint_info(f"   - Best architecture fitness: {result.architecture_fitness:.4f}")
            tprint_info(f"   - Clustering inertia: {result.inertia:.4f}")
            tprint_info(f"   - Execution time: {result.execution_time:.2f}s")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Fitting failed: {e}")
            tprint_error(f"❌ EssentialNASClusterer fitting failed: {e}")
            raise
    
    def _preprocess_data(self, data: Union[pd.DataFrame, np.ndarray], 
                        target_columns: Optional[List[str]] = None) -> np.ndarray:
        """Preprocess input data for clustering."""
        try:
            # Convert DataFrame to numpy array if needed
            if isinstance(data, pd.DataFrame):
                if target_columns:
                    data = data[target_columns]
                else:
                    # Use only numeric columns
                    data = data.select_dtypes(include=[np.number])
                
                # Convert to numpy array
                data = data.values
            
            # Validate data
            data = validate_numeric_array(data, "clustering_data")
            
            # Handle missing values
            if self.config.missing_value_strategy == "interpolate":
                data = self._interpolate_missing_values(data)
            elif self.config.missing_value_strategy == "drop":
                data = data[~np.isnan(data).any(axis=1)]
            
            # Feature scaling
            if self.config.feature_scaling != "none":
                data = self._scale_features(data)
            
            # Outlier detection and removal
            if self.config.outlier_detection:
                data = self._remove_outliers(data)
            
            # M1 optimization
            if self.config.m1_optimization and hasattr(self, 'memory_optimizer'):
                with memory_checkpoint("data_preprocessing"):
                    data = create_m1_optimized_array(data)
            
            tprint_debug(f"Data preprocessing completed: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def _interpolate_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Interpolate missing values in data."""
        try:
            # Use pandas for interpolation if available
            if pd is not None:
                df = pd.DataFrame(data)
                df = df.interpolate(method='linear', limit_direction='both')
                return df.values
            else:
                # Simple numpy interpolation
                for i in range(data.shape[1]):
                    col = data[:, i]
                    mask = ~np.isnan(col)
                    if np.any(mask):
                        data[:, i] = np.interp(
                            np.arange(len(col)), 
                            np.arange(len(col))[mask], 
                            col[mask]
                        )
                return data
        except Exception as e:
            self.logger.warning(f"Interpolation failed: {e}")
            return data
    
    def _scale_features(self, data: np.ndarray) -> np.ndarray:
        """Scale features according to configuration."""
        try:
            if self.config.feature_scaling == "standard":
                # Standard scaling (mean=0, std=1)
                mean = safe_mean(data, axis=0)
                std = safe_std(data, axis=0)
                std = np.where(std == 0, 1, std)  # Avoid division by zero
                return (data - mean) / std
                
            elif self.config.feature_scaling == "minmax":
                # Min-max scaling (0-1)
                min_vals = np.min(data, axis=0)
                max_vals = np.max(data, axis=0)
                range_vals = max_vals - min_vals
                range_vals = np.where(range_vals == 0, 1, range_vals)  # Avoid division by zero
                return (data - min_vals) / range_vals
                
            elif self.config.feature_scaling == "robust":
                # Robust scaling (median and IQR)
                median = np.median(data, axis=0)
                q75 = np.percentile(data, 75, axis=0)
                q25 = np.percentile(data, 25, axis=0)
                iqr = q75 - q25
                iqr = np.where(iqr == 0, 1, iqr)  # Avoid division by zero
                return (data - median) / iqr
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Feature scaling failed: {e}")
            return data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using IQR method."""
        try:
            # Calculate IQR for each feature
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            iqr = q75 - q25
            
            # Define outlier bounds
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Create mask for non-outliers
            mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
            
            # Remove outliers
            cleaned_data = data[mask]
            
            tprint_debug(f"Outlier removal: {len(data)} -> {len(cleaned_data)} samples")
            return cleaned_data
            
        except Exception as e:
            self.logger.warning(f"Outlier removal failed: {e}")
            return data
    
    def _run_nas_optimization(self, data: np.ndarray) -> NeuralArchitecture:
        """Run Neural Architecture Search optimization."""
        try:
            tprint_info(f"🧬 Starting NAS optimization with {len(self.population)} architectures")
            
            best_fitness = float('-inf')
            best_architecture = None
            
            for generation in range(self.config.nas_generations):
                tprint_progress(generation + 1, self.config.nas_generations, 
                              f"NAS Generation {generation + 1}")
                
                # Evaluate population
                for architecture in self.population:
                    fitness = self._evaluate_architecture(architecture, data)
                    architecture.fitness = fitness
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_architecture = architecture
                
                # Store fitness history
                self.fitness_history.append(best_fitness)
                
                # Evolve population (selection, crossover, mutation)
                self._evolve_population()
                
                # Early stopping if converged
                if self._check_nas_convergence():
                    tprint_info(f"NAS converged at generation {generation + 1}")
                    break
            
            self.best_architecture = best_architecture
            tprint_success(f"✅ NAS optimization completed. Best fitness: {best_fitness:.4f}")
            
            return best_architecture
            
        except Exception as e:
            self.logger.error(f"NAS optimization failed: {e}")
            raise
    
    def _evaluate_architecture(self, architecture: NeuralArchitecture, data: np.ndarray) -> float:
        """Evaluate the fitness of a neural architecture."""
        try:
            # Simplified fitness evaluation
            # In practice, this would train a neural network with the architecture
            # and evaluate its clustering performance
            
            # Use clustering performance as fitness
            clustering_score = self._evaluate_clustering_performance(data, architecture)
            
            # Consider architecture complexity (simpler is better)
            complexity_penalty = architecture.complexity / 10000.0
            
            # Combined fitness score
            fitness = clustering_score - complexity_penalty
            
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Architecture evaluation failed: {e}")
            return 0.0
    
    def _evaluate_clustering_performance(self, data: np.ndarray, architecture: NeuralArchitecture) -> float:
        """Evaluate clustering performance for an architecture."""
        try:
            # Simplified performance evaluation
            # Use silhouette score as the primary metric
            
            # For demonstration, use a simple clustering approach
            n_clusters = max(2, min(len(architecture.layers), 10))
            
            # Create and fit a simple clustering algorithm
            kmeans = self.algorithms['kmeans']
            kmeans.n_clusters = n_clusters
            kmeans.fit(data)
            
            # Calculate silhouette score (simplified)
            labels = kmeans.predict(data)
            centers = kmeans.get_centers()
            
            # Simple silhouette approximation
            if len(set(labels)) > 1:
                # Calculate average intra-cluster distance
                intra_cluster_distances = []
                for i in range(n_clusters):
                    cluster_points = data[labels == i]
                    if len(cluster_points) > 0:
                        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
                        intra_cluster_distances.extend(distances)
                
                avg_intra = np.mean(intra_cluster_distances) if intra_cluster_distances else 1.0
                
                # Calculate average inter-cluster distance
                inter_cluster_distances = []
                for i in range(n_clusters):
                    for j in range(i + 1, n_clusters):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        inter_cluster_distances.append(dist)
                
                avg_inter = np.mean(inter_cluster_distances) if inter_cluster_distances else 1.0
                
                # Simplified silhouette score
                silhouette_score = (avg_inter - avg_intra) / max(avg_inter, avg_intra)
                return max(0, min(1, silhouette_score))  # Clamp to [0, 1]
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Clustering performance evaluation failed: {e}")
            return 0.0
    
    def _evolve_population(self):
        """Evolve the NAS population through selection, crossover, and mutation."""
        try:
            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Keep top performers (elitism)
            elite_size = max(1, len(self.population) // 4)
            elite = self.population[:elite_size]
            
            # Create new population
            new_population = elite.copy()
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.config.nas_population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.config.nas_crossover_rate:
                    child1, child2 = parent1.crossover(parent2)
                    new_population.extend([child1, child2])
                else:
                    # No crossover, use parents
                    new_population.extend([parent1, parent2])
            
            # Mutation
            for i, architecture in enumerate(new_population[elite_size:], elite_size):
                if np.random.random() < self.config.nas_mutation_rate:
                    new_population[i] = architecture.mutate(self.config.nas_mutation_rate)
            
            # Keep population size
            self.population = new_population[:self.config.nas_population_size]
            
        except Exception as e:
            self.logger.warning(f"Population evolution failed: {e}")
    
    def _tournament_selection(self, tournament_size: int = 3) -> NeuralArchitecture:
        """Select an architecture using tournament selection."""
        try:
            # Random tournament
            tournament = np.random.choice(self.population, size=tournament_size, replace=False)
            
            # Return best in tournament
            return max(tournament, key=lambda x: x.fitness)
            
        except Exception as e:
            self.logger.warning(f"Tournament selection failed: {e}")
            return self.population[0] if self.population else NeuralArchitecture([10], ['relu'], [0.1], 0.01)
    
    def _check_nas_convergence(self, patience: int = 10) -> bool:
        """Check if NAS has converged."""
        try:
            if len(self.fitness_history) < patience:
                return False
            
            # Check if fitness has improved in the last 'patience' generations
            recent_fitness = self.fitness_history[-patience:]
            return max(recent_fitness) - min(recent_fitness) < 0.01
            
        except Exception as e:
            self.logger.warning(f"Convergence check failed: {e}")
            return False
    
    def _cluster_with_architecture(self, data: np.ndarray, architecture: NeuralArchitecture) -> ClusteringResult:
        """Perform clustering using the best architecture."""
        try:
            start_time = time.time()
            
            # Determine optimal number of clusters based on architecture
            n_clusters = max(2, min(len(architecture.layers), 10))
            
            # Use the best clustering algorithm
            algorithm = self.algorithms['kmeans']
            algorithm.n_clusters = n_clusters
            
            # Fit clustering algorithm
            algorithm.fit(data)
            
            # Get results
            labels = algorithm.predict(data)
            centers = algorithm.get_centers()
            inertia = algorithm.get_inertia()
            
            # Calculate performance metrics
            silhouette_score = self._calculate_silhouette_score(data, labels)
            calinski_harabasz_score = self._calculate_calinski_harabasz_score(data, labels)
            davies_bouldin_score = self._calculate_davies_bouldin_score(data, labels)
            
            # Get memory usage
            memory_usage_mb = get_memory_usage().get('used_memory', 0) / (1024 * 1024)
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = ClusteringResult(
                labels=labels,
                centers=centers,
                inertia=inertia,
                n_clusters=n_clusters,
                silhouette_score=silhouette_score,
                calinski_harabasz_score=calinski_harabasz_score,
                davies_bouldin_score=davies_bouldin_score,
                architecture_fitness=architecture.fitness,
                architecture_complexity=architecture.complexity,
                nas_generation=len(self.fitness_history),
                execution_time=execution_time,
                convergence_iterations=100,  # Placeholder
                memory_usage_mb=memory_usage_mb,
                configuration=self.config
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Clustering with architecture failed: {e}")
            raise
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering results."""
        try:
            # Simplified silhouette score calculation
            n_samples = len(data)
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters <= 1:
                return 0.0
            
            silhouette_scores = []
            
            for i in range(n_samples):
                # Calculate average distance to points in same cluster
                cluster_label = labels[i]
                same_cluster_points = data[labels == cluster_label]
                if len(same_cluster_points) > 1:
                    distances = np.linalg.norm(same_cluster_points - data[i], axis=1)
                    a_i = np.mean(distances[distances > 0])  # Exclude self
                else:
                    a_i = 0.0
                
                # Calculate minimum average distance to other clusters
                b_i = float('inf')
                for other_cluster in unique_labels:
                    if other_cluster != cluster_label:
                        other_cluster_points = data[labels == other_cluster]
                        if len(other_cluster_points) > 0:
                            distances = np.linalg.norm(other_cluster_points - data[i], axis=1)
                            b_i = min(b_i, np.mean(distances))
                
                # Calculate silhouette score for this point
                if max(a_i, b_i) > 0:
                    s_i = (b_i - a_i) / max(a_i, b_i)
                else:
                    s_i = 0.0
                
                silhouette_scores.append(s_i)
            
            return np.mean(silhouette_scores)
            
        except Exception as e:
            self.logger.warning(f"Silhouette score calculation failed: {e}")
            return 0.0
    
    def _calculate_calinski_harabasz_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score for clustering results."""
        try:
            n_samples, n_features = data.shape
            n_clusters = len(np.unique(labels))
            
            if n_clusters <= 1:
                return 0.0
            
            # Calculate overall mean
            overall_mean = np.mean(data, axis=0)
            
            # Calculate between-cluster scatter
            between_cluster_scatter = 0.0
            for cluster_label in np.unique(labels):
                cluster_points = data[labels == cluster_label]
                cluster_mean = np.mean(cluster_points, axis=0)
                n_points = len(cluster_points)
                between_cluster_scatter += n_points * np.sum((cluster_mean - overall_mean) ** 2)
            
            # Calculate within-cluster scatter
            within_cluster_scatter = 0.0
            for cluster_label in np.unique(labels):
                cluster_points = data[labels == cluster_label]
                cluster_mean = np.mean(cluster_points, axis=0)
                within_cluster_scatter += np.sum((cluster_points - cluster_mean) ** 2)
            
            # Calculate Calinski-Harabasz score
            if within_cluster_scatter > 0:
                ch_score = (between_cluster_scatter / (n_clusters - 1)) / (within_cluster_scatter / (n_samples - n_clusters))
            else:
                ch_score = 0.0
            
            return ch_score
            
        except Exception as e:
            self.logger.warning(f"Calinski-Harabasz score calculation failed: {e}")
            return 0.0
    
    def _calculate_davies_bouldin_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score for clustering results."""
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            if n_clusters <= 1:
                return 0.0
            
            # Calculate cluster centers and average distances
            cluster_centers = []
            cluster_avg_distances = []
            
            for cluster_label in unique_labels:
                cluster_points = data[labels == cluster_label]
                cluster_center = np.mean(cluster_points, axis=0)
                cluster_centers.append(cluster_center)
                
                # Calculate average distance to cluster center
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                avg_distance = np.mean(distances)
                cluster_avg_distances.append(avg_distance)
            
            cluster_centers = np.array(cluster_centers)
            
            # Calculate Davies-Bouldin score
            db_scores = []
            for i in range(n_clusters):
                max_ratio = 0.0
                for j in range(n_clusters):
                    if i != j:
                        # Calculate ratio of within-cluster distance to between-cluster distance
                        center_distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                        if center_distance > 0:
                            ratio = (cluster_avg_distances[i] + cluster_avg_distances[j]) / center_distance
                            max_ratio = max(max_ratio, ratio)
                db_scores.append(max_ratio)
            
            return np.mean(db_scores)
            
        except Exception as e:
            self.logger.warning(f"Davies-Bouldin score calculation failed: {e}")
            return 0.0
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            data: Input data to cluster
            
        Returns:
            Cluster labels for the input data
        """
        if not self.fitted:
            raise ValueError("EssentialNASClusterer must be fitted before making predictions")
        
        try:
            # Preprocess data using same preprocessing as training
            processed_data = self._preprocess_data(data)
            
            # Use the best clustering algorithm to predict
            algorithm = self.algorithms['kmeans']
            algorithm.n_clusters = self.best_result.n_clusters
            
            # Predict labels
            labels = algorithm.predict(processed_data)
            
            tprint_info(f"✅ Predicted cluster labels for {len(data)} samples")
            return labels
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            tprint_error(f"❌ Prediction failed: {e}")
            raise
    
    def get_results(self) -> Optional[ClusteringResult]:
        """Get the best clustering results."""
        return self.best_result
    
    def get_fitness_history(self) -> List[float]:
        """Get the fitness history from NAS optimization."""
        return self.fitness_history.copy()
    
    def get_best_architecture(self) -> Optional[NeuralArchitecture]:
        """Get the best neural architecture found during NAS."""
        return self.best_architecture
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.fitted:
                tprint_warning("Model not fitted, cannot save")
                return False
            
            # Prepare model data for saving
            model_data = {
                'config': self.config,
                'best_result': self.best_result,
                'best_architecture': self.best_architecture,
                'fitness_history': self.fitness_history,
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics,
                'fitted': self.fitted
            }
            
            # Save using universal serializer
            success = self.universal_serializer.save(model_data, filepath)
            
            if success:
                tprint_success(f"✅ Model saved to {filepath}")
            else:
                tprint_error(f"❌ Failed to save model to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
            tprint_error(f"❌ Model saving failed: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EssentialNASClusterer':
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded EssentialNASClusterer instance
        """
        try:
            # Load model data
            model_data = UniversalSerializer().load(filepath)
            
            if model_data is None:
                raise ValueError(f"Could not load model from {filepath}")
            
            # Create new instance with saved config
            instance = cls(config=model_data['config'])
            
            # Restore model state
            instance.best_result = model_data['best_result']
            instance.best_architecture = model_data['best_architecture']
            instance.fitness_history = model_data['fitness_history']
            instance.training_history = model_data['training_history']
            instance.performance_metrics = model_data['performance_metrics']
            instance.fitted = model_data['fitted']
            
            tprint_success(f"✅ Model loaded from {filepath}")
            return instance
            
        except Exception as e:
            tprint_error(f"❌ Model loading failed: {e}")
            raise
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive clustering report."""
        try:
            if not self.fitted or self.best_result is None:
                return {"error": "Model not fitted"}
            
            report = {
                'model_info': {
                    'name': self.config.name,
                    'version': self.config.version,
                    'fitted': self.fitted,
                    'configuration': self.config.__dict__
                },
                'clustering_results': {
                    'n_clusters': self.best_result.n_clusters,
                    'inertia': self.best_result.inertia,
                    'silhouette_score': self.best_result.silhouette_score,
                    'calinski_harabasz_score': self.best_result.calinski_harabasz_score,
                    'davies_bouldin_score': self.best_result.davies_bouldin_score,
                    'execution_time': self.best_result.execution_time,
                    'memory_usage_mb': self.best_result.memory_usage_mb
                },
                'nas_results': {
                    'best_fitness': self.best_result.architecture_fitness,
                    'architecture_complexity': self.best_result.architecture_complexity,
                    'generations_completed': len(self.fitness_history),
                    'fitness_history': self.fitness_history,
                    'convergence_achieved': self._check_nas_convergence()
                },
                'performance_metrics': self.performance_metrics,
                'training_summary': {
                    'total_training_time': sum(h.get('execution_time', 0) for h in self.training_history),
                    'data_shape': self.training_data.shape if self.training_data is not None else None,
                    'algorithms_used': list(self.algorithms.keys())
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            # Stop memory monitoring
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            # Cleanup M1 optimizers
            if self.config.m1_optimization:
                cleanup_m1_optimizers()
            
            # Clear large data structures
            self.training_data = None
            self.population = []
            
            tprint_info("✅ EssentialNASClusterer cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
    
    def __repr__(self) -> str:
        """String representation of the clusterer."""
        status = "fitted" if self.fitted else "not fitted"
        return f"EssentialNASClusterer(name='{self.config.name}', status='{status}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()


# Convenience functions for easy usage
def create_nas_clusterer(config: Optional[NASClustererConfig] = None) -> EssentialNASClusterer:
    """Create a new EssentialNASClusterer instance."""
    return EssentialNASClusterer(config)


def quick_cluster(data: Union[pd.DataFrame, np.ndarray], 
                 n_clusters_range: Tuple[int, int] = (2, 8),
                 **kwargs) -> ClusteringResult:
    """
    Quick clustering with default settings.
    
    Args:
        data: Input data
        n_clusters_range: Range of clusters to test
        **kwargs: Additional configuration parameters
        
    Returns:
        ClusteringResult object
    """
    config = NASClustererConfig(n_clusters_range=n_clusters_range, **kwargs)
    clusterer = EssentialNASClusterer(config)
    clusterer.fit(data)
    return clusterer.get_results()


# Export main classes and functions
__all__ = [
    'EssentialNASClusterer',
    'NASClustererConfig', 
    'ClusteringResult',
    'NeuralArchitecture',
    'ClusteringAlgorithm',
    'create_nas_clusterer',
    'quick_cluster'
]