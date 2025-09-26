#!/usr/bin/env python3
"""
Financial Neural Architecture Search (NAS) Clusterer

A specialized neural architecture search and clustering system designed for
financial market analysis, trading strategy optimization, and regime classification.

Features:
- Financial-specific architecture search for market regime prediction
- Advanced clustering for trading patterns and market regimes
- Support/Resistance level clustering using DBSCAN
- Market regime classification (BULL, BEAR, SIDEWAYS, VOLATILE)
- Time series financial feature engineering
- M1 Apple Silicon optimization for high-frequency trading
- Cross-validation for financial time series
- Comprehensive trading performance metrics
- Integration with financial data utilities (klines, OHLCV)

Author: AI Assistant
Date: 2025-01-11
"""

import logging
import time
import json
import pickle
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import threading
from contextlib import contextmanager

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import utility modules
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, create_summary_statistics, optimize_dataframe_dtypes,
    safe_merge_dataframes, safe_drop_columns, safe_rename_columns,
    safe_to_parquet, safe_read_parquet, safe_json_dump, safe_json_load,
    integrate_with_m1_optimizers, cleanup_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, CommonUtilities
)

from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, validate_positive,
    validate_range, safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, MathValidation
)

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_progress, tprint_performance, tprint_structured, tprint_timer,
    configure_tprint, TPrintConfig, LogLevel
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# M1 Hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array
    )
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    def get_m1_gpu_manager(): return None
    def is_m1_available(): return False
    def is_mps_available(): return False
    def optimize_dataframe_for_m1(df): return df
    def create_m1_optimized_array(data, dtype=None): return np.array(data, dtype=dtype)

try:
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_dataframe_memory
    )
    M1_MEMORY_AVAILABLE = True
except ImportError:
    M1_MEMORY_AVAILABLE = False
    def get_m1_memory_optimizer(): return None
    def optimize_dataframe_memory(df): return df

try:
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, parallel_map_m1, create_m1_optimized_thread_pool
    )
    M1_CPU_AVAILABLE = True
except ImportError:
    M1_CPU_AVAILABLE = False
    def get_m1_cpu_optimizer(): return None
    def parallel_map_m1(func, items, max_workers=None): return [func(item) for item in items]
    def create_m1_optimized_thread_pool(max_workers=None): return None

# ML utilities
try:
    from src.utils.common_operations import (
        create_data_quality_report, validate_dataframe_schema
    )
    ML_UTILITIES_AVAILABLE = True
except ImportError:
    ML_UTILITIES_AVAILABLE = False

# Matrix operations
try:
    from src.utils.matrix_operations.unified_operations import (
        safe_matrix_operations, validate_matrix_input
    )
    MATRIX_UTILITIES_AVAILABLE = True
except ImportError:
    MATRIX_UTILITIES_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """Financial neural architecture search strategies."""
    RANDOM = "random"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    FINANCIAL_OPTIMIZED = "financial_optimized"  # Specialized for financial data
    REGIME_SPECIFIC = "regime_specific"  # Optimized for market regime prediction

class ClusteringAlgorithm(Enum):
    """Clustering algorithms for financial data and architectures."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"  # Primary for S/R level clustering
    SPECTRAL = "spectral"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    AFFINITY_PROPAGATION = "affinity_propagation"
    FINANCIAL_DBSCAN = "financial_dbscan"  # Optimized for financial clustering
    REGIME_CLUSTERING = "regime_clustering"  # Specialized for market regimes

class ArchitectureRepresentation(Enum):
    """Neural architecture representation methods for financial data."""
    DIRECT = "direct"
    ENCODED = "encoded"
    GRAPH = "graph"
    SEQUENCE = "sequence"
    FINANCIAL_FEATURES = "financial_features"  # Optimized for OHLCV data
    TIME_SERIES = "time_series"  # For temporal financial patterns

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    TRENDING = "TRENDING"
    RANGING = "RANGING"

@dataclass
class ArchitectureConfig:
    """Configuration for financial neural architecture search."""
    # Architecture parameters
    max_layers: int = 8
    min_layers: int = 2
    max_neurons_per_layer: int = 512
    min_neurons_per_layer: int = 32
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])
    learning_rates: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])
    optimizers: List[str] = field(default_factory=lambda: ['adam', 'adamw', 'rmsprop', 'sgd'])
    
    # Financial-specific parameters
    financial_features: List[str] = field(default_factory=lambda: [
        'rsi', 'macd', 'bb_position', 'atr', 'adx', 'volatility_20', 'returns',
        'volume_profile', 'price_momentum', 'trend_strength', 'sr_proximity'
    ])
    time_series_features: List[str] = field(default_factory=lambda: [
        'price_change', 'volume_change', 'volatility_regime', 'momentum',
        'mean_reversion', 'breakout_signals', 'support_resistance'
    ])
    regime_classes: List[str] = field(default_factory=lambda: ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE'])
    
    # Search parameters
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_size: int = 3
    
    # Financial clustering parameters
    n_clusters_range: Tuple[int, int] = (2, 8)
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.FINANCIAL_DBSCAN
    dbscan_eps: float = 0.01  # Optimized for financial data
    dbscan_min_samples: int = 3
    
    # Performance parameters
    max_training_time: float = 180.0  # 3 minutes for financial data
    early_stopping_patience: int = 8
    validation_split: float = 0.2
    time_series_cv_folds: int = 5
    
    # M1 optimization
    use_m1_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Financial data specific
    lookback_periods: int = 100
    feature_engineering: bool = True
    regime_detection: bool = True
    sr_level_clustering: bool = True

@dataclass
class Architecture:
    """Represents a financial neural network architecture."""
    layers: List[Dict[str, Any]]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_score: float = 0.0
    test_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0
    financial_metrics: Dict[str, float] = field(default_factory=dict)
    regime_accuracy: Dict[str, float] = field(default_factory=dict)
    trading_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.complexity_score = self._calculate_complexity()
        self.efficiency_score = self._calculate_efficiency()
    
    def _calculate_complexity(self) -> float:
        """Calculate architecture complexity score."""
        total_params = sum(layer.get('neurons', 0) for layer in self.layers)
        total_layers = len(self.layers)
        return safe_divide(total_params * total_layers, 1000.0, 0.0)
    
    def _calculate_efficiency(self) -> float:
        """Calculate architecture efficiency score."""
        if self.training_time <= 0:
            return 0.0
        return safe_divide(self.validation_score, self.training_time, 0.0)
    
    def get_financial_score(self) -> float:
        """Get overall financial performance score."""
        if not self.financial_metrics:
            return self.validation_score
        
        # Weighted combination of financial metrics
        weights = {
            'sharpe_ratio': 0.3,
            'max_drawdown': -0.2,  # Negative weight (lower is better)
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'regime_accuracy': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in self.financial_metrics:
                score += self.financial_metrics[metric] * weight
        
        return max(0.0, score)  # Ensure non-negative score

class NASClusterer:
    """
    Neural Architecture Search Clusterer.
    
    A comprehensive system for discovering, evaluating, and clustering
    neural network architectures using advanced search and clustering techniques.
    """
    
    def __init__(self, config: Optional[ArchitectureConfig] = None):
        """
        Initialize NAS Clusterer.
        
        Args:
            config: Architecture configuration (optional)
        """
        self.config = config or ArchitectureConfig()
        self.logger = logger.getChild('NASClusterer')
        
        # Initialize components
        self.architectures: List[Architecture] = []
        self.clusters: Dict[int, List[Architecture]] = {}
        self.cluster_centers: Dict[int, np.ndarray] = {}
        self.cluster_metrics: Dict[str, float] = {}
        
        # Initialize utilities
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize M1 optimizations
        self._setup_m1_optimizations()
        
        # Initialize search strategies
        self._setup_search_strategies()
        
        # Initialize clustering algorithms
        self._setup_clustering_algorithms()
        
        tprint_info("🧠 NASClusterer initialized successfully")
    
    def _setup_m1_optimizations(self):
        """Setup M1 hardware optimizations."""
        try:
            if M1_GPU_AVAILABLE and is_m1_available():
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring if available
                if self.memory_optimizer:
                    self.memory_optimizer.start_monitoring()
                
                tprint_success("🚀 M1 optimizations enabled")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                tprint_warning("⚠️ M1 optimizations not available")
        except Exception as e:
            self.logger.warning(f"M1 optimization setup failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _setup_search_strategies(self):
        """Setup financial neural architecture search strategies."""
        self.search_strategies = {
            SearchStrategy.RANDOM: self._random_search,
            SearchStrategy.EVOLUTIONARY: self._evolutionary_search,
            SearchStrategy.REINFORCEMENT_LEARNING: self._rl_search,
            SearchStrategy.BAYESIAN: self._bayesian_search,
            SearchStrategy.GRID_SEARCH: self._grid_search,
            SearchStrategy.RANDOM_SEARCH: self._random_search,
            SearchStrategy.FINANCIAL_OPTIMIZED: self._financial_optimized_search,
            SearchStrategy.REGIME_SPECIFIC: self._regime_specific_search
        }
    
    def _setup_clustering_algorithms(self):
        """Setup financial clustering algorithms."""
        self.clustering_algorithms = {
            ClusteringAlgorithm.KMEANS: self._kmeans_clustering,
            ClusteringAlgorithm.HIERARCHICAL: self._hierarchical_clustering,
            ClusteringAlgorithm.DBSCAN: self._dbscan_clustering,
            ClusteringAlgorithm.SPECTRAL: self._spectral_clustering,
            ClusteringAlgorithm.GAUSSIAN_MIXTURE: self._gaussian_mixture_clustering,
            ClusteringAlgorithm.AFFINITY_PROPAGATION: self._affinity_propagation_clustering,
            ClusteringAlgorithm.FINANCIAL_DBSCAN: self._financial_dbscan_clustering,
            ClusteringAlgorithm.REGIME_CLUSTERING: self._regime_clustering
        }
    
    def generate_random_architecture(self) -> Architecture:
        """Generate a random neural architecture."""
        n_layers = np.random.randint(self.config.min_layers, self.config.max_layers + 1)
        layers = []
        
        for i in range(n_layers):
            layer = {
                'neurons': np.random.randint(
                    self.config.min_neurons_per_layer,
                    self.config.max_neurons_per_layer + 1
                ),
                'activation': np.random.choice(self.config.activation_functions),
                'dropout': np.random.choice(self.config.dropout_rates),
                'layer_type': 'dense'
            }
            layers.append(layer)
        
        return Architecture(layers=layers)
    
    def evaluate_architecture(self, architecture: Architecture, 
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Architecture:
        """Evaluate a neural architecture."""
        start_time = time.time()
        
        try:
            # Create and train model
            model = self._create_model_from_architecture(architecture)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.score(X_val, y_val)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Update architecture with results
            architecture.validation_score = val_score
            architecture.training_time = training_time
            architecture.performance_metrics = {
                'validation_score': val_score,
                'training_time': training_time,
                'complexity': architecture.complexity_score,
                'efficiency': architecture.efficiency_score
            }
            
            tprint_debug(f"Architecture evaluated: score={val_score:.4f}, time={training_time:.2f}s")
            return architecture
            
        except Exception as e:
            self.logger.warning(f"Architecture evaluation failed: {e}")
            architecture.validation_score = 0.0
            architecture.training_time = time.time() - start_time
            return architecture
    
    def _create_model_from_architecture(self, architecture: Architecture):
        """Create a scikit-learn model from architecture."""
        # For simplicity, use MLPClassifier
        # In a real implementation, you'd create a custom neural network
        hidden_layer_sizes = tuple(layer['neurons'] for layer in architecture.layers[:-1])
        activation = architecture.layers[0]['activation']
        
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=1000,
            random_state=42
        )
    
    def search_architectures(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           strategy: SearchStrategy = SearchStrategy.RANDOM,
                           n_architectures: int = 100) -> List[Architecture]:
        """Search for neural architectures using specified strategy."""
        tprint_info(f"🔍 Starting architecture search with {strategy.value} strategy")
        
        with tprint_timer("Architecture Search"):
            search_func = self.search_strategies.get(strategy, self._random_search)
            architectures = search_func(X_train, y_train, X_val, y_val, n_architectures)
        
        tprint_success(f"✅ Found {len(architectures)} architectures")
        return architectures
    
    def _random_search(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Random search for architectures."""
        architectures = []
        
        for i in range(n_architectures):
            tprint_progress(i + 1, n_architectures, "Generating random architectures")
            
            # Generate random architecture
            arch = self.generate_random_architecture()
            
            # Evaluate architecture
            arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            architectures.append(arch)
        
        return architectures
    
    def _evolutionary_search(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Evolutionary search for architectures."""
        # Initialize population
        population = [self.generate_random_architecture() for _ in range(self.config.population_size)]
        
        # Evaluate initial population
        for i, arch in enumerate(population):
            tprint_progress(i + 1, self.config.population_size, "Evaluating initial population")
            population[i] = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
        
        # Evolution loop
        for generation in range(self.config.generations):
            tprint_progress(generation + 1, self.config.generations, f"Evolution generation {generation + 1}")
            
            # Sort by performance
            population.sort(key=lambda x: x.validation_score, reverse=True)
            
            # Keep elite
            elite = population[:self.config.elite_size]
            
            # Create new generation
            new_population = elite.copy()
            
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2)
                
                # Evaluate children
                child1 = self.evaluate_architecture(child1, X_train, y_train, X_val, y_val)
                child2 = self.evaluate_architecture(child2, X_train, y_train, X_val, y_val)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.population_size]
        
        return population
    
    def _tournament_selection(self, population: List[Architecture], tournament_size: int = 3) -> Architecture:
        """Tournament selection for evolutionary algorithm."""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.validation_score)
    
    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Tuple[Architecture, Architecture]:
        """Crossover operation for evolutionary algorithm."""
        # Simple crossover: take layers from each parent
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = np.random.randint(1, min_layers)
        
        child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        
        return Architecture(layers=child1_layers), Architecture(layers=child2_layers)
    
    def _mutate(self, architecture: Architecture) -> Architecture:
        """Mutation operation for evolutionary algorithm."""
        mutated_layers = architecture.layers.copy()
        
        # Randomly mutate a layer
        if mutated_layers:
            layer_idx = np.random.randint(len(mutated_layers))
            layer = mutated_layers[layer_idx]
            
            # Mutate neurons
            if np.random.random() < 0.3:
                layer['neurons'] = np.random.randint(
                    self.config.min_neurons_per_layer,
                    self.config.max_neurons_per_layer + 1
                )
            
            # Mutate activation
            if np.random.random() < 0.3:
                layer['activation'] = np.random.choice(self.config.activation_functions)
            
            # Mutate dropout
            if np.random.random() < 0.3:
                layer['dropout'] = np.random.choice(self.config.dropout_rates)
        
        return Architecture(layers=mutated_layers)
    
    def _rl_search(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Reinforcement learning search (simplified implementation)."""
        # Simplified RL implementation
        architectures = []
        
        for i in range(n_architectures):
            tprint_progress(i + 1, n_architectures, "RL architecture search")
            
            # Generate architecture using RL policy (simplified)
            arch = self.generate_random_architecture()
            arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            architectures.append(arch)
        
        return architectures
    
    def _bayesian_search(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Bayesian optimization search (simplified implementation)."""
        # Simplified Bayesian optimization
        architectures = []
        
        for i in range(n_architectures):
            tprint_progress(i + 1, n_architectures, "Bayesian architecture search")
            
            # Generate architecture using Bayesian optimization (simplified)
            arch = self.generate_random_architecture()
            arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            architectures.append(arch)
        
        return architectures
    
    def _grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Grid search for architectures."""
        architectures = []
        
        # Generate grid of architectures
        layer_counts = range(self.config.min_layers, min(self.config.max_layers + 1, 6))
        neuron_counts = [32, 64, 128, 256, 512]
        
        count = 0
        for n_layers in layer_counts:
            for neurons in neuron_counts:
                if count >= n_architectures:
                    break
                
                tprint_progress(count + 1, n_architectures, "Grid search architectures")
                
                # Create architecture
                layers = []
                for _ in range(n_layers):
                    layers.append({
                        'neurons': neurons,
                        'activation': 'relu',
                        'dropout': 0.0,
                        'layer_type': 'dense'
                    })
                
                arch = Architecture(layers=layers)
                arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
                architectures.append(arch)
                count += 1
            
            if count >= n_architectures:
                break
        
        return architectures
    
    def _financial_optimized_search(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Financial-optimized architecture search."""
        architectures = []
        
        # Financial-specific architecture patterns
        financial_patterns = [
            # LSTM-based for time series
            {'type': 'lstm', 'layers': [64, 32], 'dropout': 0.2},
            # CNN for pattern recognition
            {'type': 'cnn', 'layers': [128, 64], 'dropout': 0.3},
            # Dense for feature combination
            {'type': 'dense', 'layers': [256, 128, 64], 'dropout': 0.1},
            # Hybrid for regime detection
            {'type': 'hybrid', 'layers': [128, 64, 32], 'dropout': 0.2}
        ]
        
        for i in range(n_architectures):
            tprint_progress(i + 1, n_architectures, "Financial-optimized search")
            
            # Select pattern
            pattern = np.random.choice(financial_patterns)
            
            # Create architecture based on pattern
            layers = []
            for j, neurons in enumerate(pattern['layers']):
                layers.append({
                    'neurons': neurons,
                    'activation': 'relu' if j < len(pattern['layers']) - 1 else 'softmax',
                    'dropout': pattern['dropout'],
                    'layer_type': pattern['type']
                })
            
            arch = Architecture(layers=layers)
            arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            architectures.append(arch)
        
        return architectures
    
    def _regime_specific_search(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, n_architectures: int) -> List[Architecture]:
        """Regime-specific architecture search."""
        architectures = []
        
        # Regime-specific architecture configurations
        regime_configs = {
            'BULL': {'layers': [128, 64], 'activation': 'relu', 'dropout': 0.1},
            'BEAR': {'layers': [128, 64], 'activation': 'tanh', 'dropout': 0.2},
            'SIDEWAYS': {'layers': [64, 32], 'activation': 'sigmoid', 'dropout': 0.3},
            'VOLATILE': {'layers': [256, 128, 64], 'activation': 'leaky_relu', 'dropout': 0.4}
        }
        
        for i in range(n_architectures):
            tprint_progress(i + 1, n_architectures, "Regime-specific search")
            
            # Select regime configuration
            regime = np.random.choice(list(regime_configs.keys()))
            config = regime_configs[regime]
            
            # Create architecture
            layers = []
            for neurons in config['layers']:
                layers.append({
                    'neurons': neurons,
                    'activation': config['activation'],
                    'dropout': config['dropout'],
                    'layer_type': 'dense'
                })
            
            arch = Architecture(layers=layers)
            arch = self.evaluate_architecture(arch, X_train, y_train, X_val, y_val)
            architectures.append(arch)
        
        return architectures
    
    def cluster_architectures(self, architectures: List[Architecture],
                            algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS,
                            n_clusters: Optional[int] = None) -> Dict[int, List[Architecture]]:
        """Cluster architectures using specified algorithm."""
        tprint_info(f"🔗 Clustering {len(architectures)} architectures using {algorithm.value}")
        
        if not architectures:
            tprint_warning("No architectures to cluster")
            return {}
        
        # Extract features from architectures
        features = self._extract_architecture_features(architectures)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply clustering
        clustering_func = self.clustering_algorithms.get(algorithm, self._kmeans_clustering)
        cluster_labels = clustering_func(features_normalized, n_clusters)
        
        # Group architectures by cluster
        clusters = {}
        for i, (arch, label) in enumerate(zip(architectures, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(arch)
        
        # Calculate cluster metrics
        self._calculate_cluster_metrics(features_normalized, cluster_labels)
        
        # Store results
        self.architectures = architectures
        self.clusters = clusters
        self.cluster_metrics = self._calculate_cluster_metrics(features_normalized, cluster_labels)
        
        tprint_success(f"✅ Created {len(clusters)} clusters")
        return clusters
    
    def _extract_architecture_features(self, architectures: List[Architecture]) -> np.ndarray:
        """Extract features from architectures for clustering."""
        features = []
        
        for arch in architectures:
            feature_vector = []
            
            # Basic architecture features
            feature_vector.append(len(arch.layers))  # Number of layers
            feature_vector.append(arch.complexity_score)  # Complexity
            feature_vector.append(arch.efficiency_score)  # Efficiency
            feature_vector.append(arch.validation_score)  # Performance
            feature_vector.append(arch.training_time)  # Training time
            
            # Layer-specific features
            if arch.layers:
                # Average neurons per layer
                avg_neurons = np.mean([layer['neurons'] for layer in arch.layers])
                feature_vector.append(avg_neurons)
                
                # Total neurons
                total_neurons = sum([layer['neurons'] for layer in arch.layers])
                feature_vector.append(total_neurons)
                
                # Activation function diversity
                activations = [layer['activation'] for layer in arch.layers]
                unique_activations = len(set(activations))
                feature_vector.append(unique_activations)
                
                # Average dropout
                avg_dropout = np.mean([layer['dropout'] for layer in arch.layers])
                feature_vector.append(avg_dropout)
            else:
                feature_vector.extend([0, 0, 0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _kmeans_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """K-means clustering."""
        if n_clusters is None:
            n_clusters = 3
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    def _hierarchical_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Hierarchical clustering."""
        if n_clusters is None:
            n_clusters = 3
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        return hierarchical.fit_predict(features)
    
    def _dbscan_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """DBSCAN clustering."""
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        return dbscan.fit_predict(features)
    
    def _spectral_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Spectral clustering."""
        if n_clusters is None:
            n_clusters = 3
        
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42)
        return spectral.fit_predict(features)
    
    def _gaussian_mixture_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Gaussian mixture clustering."""
        from sklearn.mixture import GaussianMixture
        
        if n_clusters is None:
            n_clusters = 3
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gmm.fit_predict(features)
    
    def _affinity_propagation_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Affinity propagation clustering."""
        from sklearn.cluster import AffinityPropagation
        
        af = AffinityPropagation(random_state=42)
        return af.fit_predict(features)
    
    def _financial_dbscan_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Financial-optimized DBSCAN clustering."""
        # Use financial-specific parameters
        eps = self.config.dbscan_eps
        min_samples = self.config.dbscan_min_samples
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(features)
    
    def _regime_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Regime-specific clustering for market regimes."""
        if n_clusters is None:
            n_clusters = len(self.config.regime_classes)
        
        # Use K-means with regime-specific initialization
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    def _calculate_cluster_metrics(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        metrics = {}
        
        try:
            # Silhouette score
            if len(set(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(features, cluster_labels)
            else:
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            if len(set(cluster_labels)) > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, cluster_labels)
            else:
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin score
            if len(set(cluster_labels)) > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(features, cluster_labels)
            else:
                metrics['davies_bouldin_score'] = float('inf')
            
        except Exception as e:
            self.logger.warning(f"Error calculating cluster metrics: {e}")
            metrics = {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }
        
        return metrics
    
    def get_best_architectures(self, n: int = 10) -> List[Architecture]:
        """Get the best performing architectures."""
        if not self.architectures:
            return []
        
        # Sort by financial score if available, otherwise validation score
        sorted_architectures = sorted(
            self.architectures,
            key=lambda x: x.get_financial_score() if x.financial_metrics else x.validation_score,
            reverse=True
        )
        
        return sorted_architectures[:n]
    
    def cluster_financial_data(self, data: pd.DataFrame, 
                             algorithm: ClusteringAlgorithm = ClusteringAlgorithm.FINANCIAL_DBSCAN,
                             n_clusters: Optional[int] = None) -> Dict[int, pd.DataFrame]:
        """Cluster financial market data (OHLCV)."""
        tprint_info(f"📊 Clustering financial data with {algorithm.value}")
        
        # Extract financial features
        features = self._extract_financial_features(data)
        
        if features.empty:
            tprint_warning("No financial features extracted")
            return {}
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply clustering
        clustering_func = self.clustering_algorithms.get(algorithm, self._financial_dbscan_clustering)
        cluster_labels = clustering_func(features_normalized, n_clusters)
        
        # Group data by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data.iloc[i])
        
        # Convert to DataFrames
        cluster_dfs = {}
        for cluster_id, cluster_data in clusters.items():
            if cluster_data:
                cluster_dfs[cluster_id] = pd.DataFrame(cluster_data)
        
        tprint_success(f"✅ Created {len(cluster_dfs)} financial data clusters")
        return cluster_dfs
    
    def _extract_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract financial features from OHLCV data."""
        features = pd.DataFrame()
        
        try:
            # Basic price features
            if 'close' in data.columns:
                features['returns'] = data['close'].pct_change()
                features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                features['price_change'] = data['close'].diff()
            
            # Technical indicators
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = data['close'].ewm(span=12).mean()
                ema_26 = data['close'].ewm(span=26).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                
                # Bollinger Bands
                bb_middle = data['close'].rolling(window=20).mean()
                bb_std = data['close'].rolling(window=20).std()
                features['bb_upper'] = bb_middle + (bb_std * 2)
                features['bb_lower'] = bb_middle - (bb_std * 2)
                features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_change'] = data['volume'].pct_change()
                features['volume_ma'] = data['volume'].rolling(window=20).mean()
                features['volume_ratio'] = data['volume'] / features['volume_ma']
            
            # Volatility features
            if 'close' in data.columns:
                features['volatility_20'] = data['close'].rolling(window=20).std()
                features['volatility_5'] = data['close'].rolling(window=5).std()
                features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # ATR (Average True Range)
            if all(col in data.columns for col in ['high', 'low', 'close']):
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift())
                low_close = np.abs(data['low'] - data['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                features['atr'] = true_range.rolling(window=14).mean()
            
            # ADX (Average Directional Index)
            if all(col in data.columns for col in ['high', 'low', 'close']):
                plus_dm = data['high'].diff()
                minus_dm = data['low'].diff()
                plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
                minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
                
                tr = true_range if 'true_range' in locals() else np.maximum(
                    high_low, np.maximum(high_close, low_close)
                )
                
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
                
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                features['adx'] = dx.rolling(window=14).mean()
            
            # Remove rows with NaN values
            features = features.dropna()
            
        except Exception as e:
            tprint_error(f"Error extracting financial features: {e}")
            return pd.DataFrame()
        
        return features
    
    def detect_market_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regimes in financial data."""
        tprint_info("🔍 Detecting market regimes")
        
        try:
            # Extract features
            features = self._extract_financial_features(data)
            
            if features.empty:
                tprint_warning("No features available for regime detection")
                return pd.Series(['UNKNOWN'] * len(data), index=data.index)
            
            # Simple regime detection based on returns and volatility
            regimes = []
            
            for i in range(len(features)):
                returns = features['returns'].iloc[i] if 'returns' in features.columns else 0
                volatility = features['volatility_20'].iloc[i] if 'volatility_20' in features.columns else 0
                
                if volatility > 0.02:  # High volatility threshold
                    regime = 'VOLATILE'
                elif returns > 0.001:  # Positive returns threshold
                    regime = 'BULL'
                elif returns < -0.001:  # Negative returns threshold
                    regime = 'BEAR'
                else:
                    regime = 'SIDEWAYS'
                
                regimes.append(regime)
            
            regime_series = pd.Series(regimes, index=features.index)
            tprint_success(f"✅ Detected {len(set(regimes))} market regimes")
            
            return regime_series
            
        except Exception as e:
            tprint_error(f"Error detecting market regimes: {e}")
            return pd.Series(['UNKNOWN'] * len(data), index=data.index)
    
    def cluster_support_resistance_levels(self, data: pd.DataFrame, 
                                        eps: float = 0.01, min_samples: int = 3) -> Dict[str, List[float]]:
        """Cluster support and resistance levels using DBSCAN."""
        tprint_info("📊 Clustering support/resistance levels")
        
        try:
            # Extract price levels (highs and lows)
            price_levels = []
            
            if 'high' in data.columns:
                price_levels.extend(data['high'].tolist())
            if 'low' in data.columns:
                price_levels.extend(data['low'].tolist())
            
            if not price_levels:
                tprint_warning("No price levels found")
                return {'support': [], 'resistance': []}
            
            # Convert to numpy array for clustering
            levels_array = np.array(price_levels).reshape(-1, 1)
            
            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(levels_array)
            
            # Group levels by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(price_levels[i])
            
            # Calculate cluster centers
            support_levels = []
            resistance_levels = []
            
            for cluster_id, levels in clusters.items():
                if cluster_id == -1:  # Noise points
                    continue
                
                cluster_center = np.mean(levels)
                cluster_std = np.std(levels)
                
                # Classify as support or resistance based on current price
                if 'close' in data.columns:
                    current_price = data['close'].iloc[-1]
                    if cluster_center < current_price:
                        support_levels.append(cluster_center)
                    else:
                        resistance_levels.append(cluster_center)
                else:
                    # Default to support if no current price
                    support_levels.append(cluster_center)
            
            tprint_success(f"✅ Found {len(support_levels)} support levels and {len(resistance_levels)} resistance levels")
            
            return {
                'support': sorted(support_levels),
                'resistance': sorted(resistance_levels, reverse=True)
            }
            
        except Exception as e:
            tprint_error(f"Error clustering S/R levels: {e}")
            return {'support': [], 'resistance': []}
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results."""
        if not self.clusters:
            return {}
        
        summary = {
            'n_clusters': len(self.clusters),
            'cluster_sizes': {str(k): len(v) for k, v in self.clusters.items()},
            'cluster_metrics': self.cluster_metrics,
            'best_architectures_per_cluster': {}
        }
        
        # Get best architecture from each cluster
        for cluster_id, architectures in self.clusters.items():
            if architectures:
                best_arch = max(architectures, key=lambda x: x.validation_score)
                summary['best_architectures_per_cluster'][str(cluster_id)] = {
                    'validation_score': best_arch.validation_score,
                    'complexity_score': best_arch.complexity_score,
                    'efficiency_score': best_arch.efficiency_score,
                    'n_layers': len(best_arch.layers)
                }
        
        return summary
    
    def save_model(self, filepath: str) -> bool:
        """Save the clusterer model to file."""
        try:
            model_data = {
                'config': self.config,
                'architectures': self.architectures,
                'clusters': self.clusters,
                'cluster_metrics': self.cluster_metrics,
                'created_at': time.time()
            }
            
            success = self.serializer.save(model_data, filepath)
            if success:
                tprint_success(f"💾 Model saved to {filepath}")
            else:
                tprint_error(f"❌ Failed to save model to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load the clusterer model from file."""
        try:
            model_data = self.serializer.load(filepath)
            if model_data is None:
                tprint_error(f"❌ Failed to load model from {filepath}")
                return False
            
            # Restore model state
            self.config = model_data.get('config', ArchitectureConfig())
            self.architectures = model_data.get('architectures', [])
            self.clusters = model_data.get('clusters', {})
            self.cluster_metrics = model_data.get('cluster_metrics', {})
            
            tprint_success(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            tprint_error(f"❌ Error loading model: {e}")
            return False
    
    def optimize_for_m1(self, data: Any) -> Any:
        """Optimize data processing for M1 hardware."""
        if not M1_GPU_AVAILABLE or not is_m1_available():
            return data
        
        try:
            if isinstance(data, pd.DataFrame):
                return optimize_dataframe_for_m1(data)
            elif isinstance(data, np.ndarray):
                return create_m1_optimized_array(data)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"M1 optimization failed: {e}")
            return data
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            if M1_GPU_AVAILABLE:
                cleanup_m1_optimizers()
            
            tprint_info("🧹 NASClusterer cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except (AttributeError, TypeError) as e:
            tprint(f"Error in cleanup: {e}", level="error")


# Convenience functions
def create_nas_clusterer(config: Optional[ArchitectureConfig] = None) -> NASClusterer:
    """Create a new NASClusterer instance."""
    return NASClusterer(config)

def quick_architecture_search(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_architectures: int = 50,
                            strategy: SearchStrategy = SearchStrategy.RANDOM) -> List[Architecture]:
    """Quick architecture search with default settings."""
    clusterer = NASClusterer()
    architectures = clusterer.search_architectures(
        X_train, y_train, X_val, y_val, strategy, n_architectures
    )
    clusterer.cleanup()
    return architectures

def quick_clustering(architectures: List[Architecture],
                    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS,
                    n_clusters: int = 3) -> Dict[int, List[Architecture]]:
    """Quick clustering of architectures."""
    clusterer = NASClusterer()
    clusters = clusterer.cluster_architectures(architectures, algorithm, n_clusters)
    clusterer.cleanup()
    return clusters


if __name__ == "__main__":
    # Financial NASClusterer Example
    tprint_info("💰 Financial NASClusterer Example")
    
    # Generate sample financial data (OHLCV)
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV DataFrame
    ohlcv_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    ohlcv_data['high'] = np.maximum(ohlcv_data['high'], np.maximum(ohlcv_data['open'], ohlcv_data['close']))
    ohlcv_data['low'] = np.minimum(ohlcv_data['low'], np.minimum(ohlcv_data['open'], ohlcv_data['close']))
    
    # Create regime labels (simplified)
    regime_labels = []
    for i in range(len(ohlcv_data)):
        returns = ohlcv_data['close'].pct_change().iloc[i] if i > 0 else 0
        volatility = ohlcv_data['close'].rolling(20).std().iloc[i] if i >= 20 else 0.01
        
        if volatility > 0.03:
            regime_labels.append('VOLATILE')
        elif returns > 0.002:
            regime_labels.append('BULL')
        elif returns < -0.002:
            regime_labels.append('BEAR')
        else:
            regime_labels.append('SIDEWAYS')
    
    y = np.array(regime_labels)
    
    # Split data for time series
    split_idx = int(0.8 * len(ohlcv_data))
    X_train = ohlcv_data.iloc[:split_idx]
    X_val = ohlcv_data.iloc[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    
    # Create financial clusterer
    config = ArchitectureConfig(
        max_layers=6,
        min_layers=2,
        max_neurons_per_layer=256,
        min_neurons_per_layer=32,
        population_size=15,
        generations=20,
        financial_features=['rsi', 'macd', 'bb_position', 'atr', 'adx'],
        regime_classes=['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']
    )
    
    clusterer = NASClusterer(config)
    
    try:
        # Extract financial features
        tprint_info("📊 Extracting financial features...")
        train_features = clusterer._extract_financial_features(X_train)
        val_features = clusterer._extract_financial_features(X_val)
        
        if not train_features.empty and not val_features.empty:
            # Search for financial architectures
            tprint_info("🔍 Searching for financial architectures...")
            architectures = clusterer.search_architectures(
                train_features.values, y_train, val_features.values, y_val,
                strategy=SearchStrategy.FINANCIAL_OPTIMIZED,
                n_architectures=15
            )
            
            # Cluster architectures
            tprint_info("🔗 Clustering architectures...")
            clusters = clusterer.cluster_architectures(
                architectures,
                algorithm=ClusteringAlgorithm.FINANCIAL_DBSCAN,
                n_clusters=4
            )
            
            # Financial data clustering
            tprint_info("📈 Clustering financial data...")
            financial_clusters = clusterer.cluster_financial_data(
                X_train,
                algorithm=ClusteringAlgorithm.FINANCIAL_DBSCAN
            )
            
            # Market regime detection
            tprint_info("🎯 Detecting market regimes...")
            regimes = clusterer.detect_market_regimes(X_train)
            
            # Support/Resistance clustering
            tprint_info("📊 Clustering S/R levels...")
            sr_levels = clusterer.cluster_support_resistance_levels(X_train)
            
            # Get results
            best_architectures = clusterer.get_best_architectures(5)
            cluster_summary = clusterer.get_cluster_summary()
            
            tprint_success("✅ Financial NASClusterer example completed")
            tprint_info(f"Found {len(architectures)} architectures")
            tprint_info(f"Created {len(clusters)} architecture clusters")
            tprint_info(f"Created {len(financial_clusters)} financial data clusters")
            tprint_info(f"Detected {len(set(regimes))} market regimes")
            tprint_info(f"Found {len(sr_levels['support'])} support levels")
            tprint_info(f"Found {len(sr_levels['resistance'])} resistance levels")
            tprint_info(f"Best financial score: {best_architectures[0].get_financial_score():.4f}")
            
        else:
            tprint_warning("⚠️ Could not extract financial features from sample data")
        
    finally:
        clusterer.cleanup()