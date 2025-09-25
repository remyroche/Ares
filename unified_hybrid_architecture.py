#!/usr/bin/env python3
"""
Unified Hybrid Architecture for NAS & TAS

This module provides a unified architecture that consolidates common functionality
between Neural Architecture Search (NAS) and Tree Architecture Search (TAS) systems.

Key Features:
- Unified configuration management
- Shared evaluation frameworks
- Common hardware optimization
- Unified search algorithms
- Shared data processing pipelines
- Common utility functions
"""

import os
import sys
import time
import logging
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import concurrent.futures
import threading
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import utility modules
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, safe_to_parquet, safe_read_parquet,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory
    )
    from src.utils.common_utilities import (
        CommonUtilities, safe_dataframe_operation as safe_df_op,
        validate_dataframe_columns as validate_df_cols,
        get_data_summary, safe_convert_dtypes as safe_convert_dt
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        MathValidation
    )
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance, tprint_structured,
        tprint_timer, configure_tprint, TPrintConfig, LogLevel
    )
    UTILITY_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utility modules: {e}")
    UTILITY_MODULES_AVAILABLE = False
    # Define fallback functions
    def safe_dataframe_operation(df, operation, *args, **kwargs):
        return operation(df, *args, **kwargs)
    def validate_dataframe_columns(df, required_columns):
        return True
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)
    def tprint_progress(step, total, message="", **kwargs):
        print(f"Progress [{step}/{total}]: {message}")

logger = logging.getLogger(__name__)


# ============================================================================
# UNIFIED ENUMS
# ============================================================================

class ArchitectureType(Enum):
    """Unified architecture types for both NAS and TAS."""
    # Neural architectures
    FEEDFORWARD = "feedforward"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CONVOLUTIONAL = "convolutional"
    ATTENTION = "attention"
    ENSEMBLE = "ensemble"
    
    # Tree architectures
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    DECISION_TREE = "decision_tree"
    ADABOOST = "adaboost"
    BAGGING = "bagging"
    
    # Hybrid architectures
    HYBRID_TREE_NEURAL = "hybrid_tree_neural"
    CVLSA_TREE = "cvlSA_tree"
    META_LEARNING = "meta_learning"


class SearchStrategy(Enum):
    """Unified search strategies."""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"


class OptimizationObjective(Enum):
    """Unified optimization objectives."""
    # Basic metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    
    # Trading metrics
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    
    # Economic metrics
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    
    # Risk metrics
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    VOLATILITY = "volatility"
    
    # Model metrics
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    INTERPRETABILITY = "interpretability"


class HardwareAccelerationType(Enum):
    """Hardware acceleration types."""
    CPU_ONLY = "cpu_only"
    GPU_ACCELERATION = "gpu_acceleration"
    M1_OPTIMIZATION = "m1_optimization"
    MULTI_GPU = "multi_gpu"
    DISTRIBUTED = "distributed"


# ============================================================================
# UNIFIED CONFIGURATION
# ============================================================================

@dataclass
class UnifiedArchitectureConfig:
    """Unified configuration for both NAS and TAS architectures."""
    
    # Architecture type
    architecture_type: ArchitectureType = ArchitectureType.FEEDFORWARD
    
    # Search parameters
    search_strategy: SearchStrategy = SearchStrategy.RANDOM
    max_trials: int = 100
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Architecture parameters (for neural networks)
    min_layers: int = 2
    max_layers: int = 10
    min_neurons: int = 32
    max_neurons: int = 512
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'swish', 'gelu'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Tree parameters (for tree-based models)
    min_trees: int = 10
    max_trees: int = 1000
    min_depth: int = 3
    max_depth: int = 20
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[int, float, str] = "auto"
    
    # Optimization parameters
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_range: Tuple[int, int] = (16, 256)
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    
    # Multi-objective optimization
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ACCURACY,
        OptimizationObjective.EFFICIENCY,
        OptimizationObjective.ROBUSTNESS
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    hardware_acceleration: HardwareAccelerationType = HardwareAccelerationType.M1_OPTIMIZATION
    memory_limit_gb: float = 8.0
    parallel_evaluations: int = 4
    
    # Data processing
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    enable_feature_selection: bool = True
    max_features: int = 100
    
    # Performance settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "unified_results"
    verbose: bool = True
    
    # Advanced features
    enable_regime_awareness: bool = True
    enable_uncertainty_quantification: bool = True
    enable_meta_learning: bool = True
    enable_real_time_adaptation: bool = False


# ============================================================================
# UNIFIED ARCHITECTURE CANDIDATE
# ============================================================================

@dataclass
class UnifiedArchitectureCandidate:
    """Unified architecture candidate for both NAS and TAS."""
    
    # Basic information
    candidate_id: str
    architecture_type: ArchitectureType
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Architecture parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    fitness_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0
    
    # Evaluation results
    evaluation_results: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Status and metadata
    status: str = "pending"  # pending, evaluating, evaluated, failed
    search_iteration: int = 0
    search_strategy: str = "unknown"
    parent_candidates: List[str] = field(default_factory=list)
    
    # Advanced features
    uncertainty_estimates: Optional[Dict[str, float]] = None
    regime_performance: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'candidate_id': self.candidate_id,
            'architecture_type': self.architecture_type.value,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'fitness_score': self.fitness_score,
            'complexity_score': self.complexity_score,
            'efficiency_score': self.efficiency_score,
            'evaluation_results': self.evaluation_results,
            'training_time': self.training_time,
            'inference_time': self.inference_time,
            'memory_usage_mb': self.memory_usage_mb,
            'status': self.status,
            'search_iteration': self.search_iteration,
            'search_strategy': self.search_strategy,
            'parent_candidates': self.parent_candidates,
            'uncertainty_estimates': self.uncertainty_estimates,
            'regime_performance': self.regime_performance,
            'feature_importance': self.feature_importance
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedArchitectureCandidate':
        """Create from dictionary."""
        candidate = cls(
            candidate_id=data['candidate_id'],
            architecture_type=ArchitectureType(data['architecture_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            parameters=data['parameters'],
            fitness_score=data['fitness_score'],
            complexity_score=data['complexity_score'],
            efficiency_score=data['efficiency_score'],
            evaluation_results=data['evaluation_results'],
            training_time=data['training_time'],
            inference_time=data['inference_time'],
            memory_usage_mb=data['memory_usage_mb'],
            status=data['status'],
            search_iteration=data['search_iteration'],
            search_strategy=data['search_strategy'],
            parent_candidates=data['parent_candidates'],
            uncertainty_estimates=data.get('uncertainty_estimates'),
            regime_performance=data.get('regime_performance'),
            feature_importance=data.get('feature_importance')
        )
        return candidate


# ============================================================================
# UNIFIED EVALUATION FRAMEWORK
# ============================================================================

class UnifiedEvaluator:
    """Unified evaluator for both NAS and TAS architectures."""
    
    def __init__(self, config: UnifiedArchitectureConfig):
        """Initialize unified evaluator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def evaluate_architecture(self, 
                            candidate: UnifiedArchitectureCandidate,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            X_test: np.ndarray,
                            y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate architecture and return comprehensive metrics."""
        
        start_time = time.time()
        
        try:
            # Create and train model
            model = self._create_model(candidate)
            training_time = self._train_model(model, X_train, y_train, X_val, y_val)
            
            # Evaluate model
            evaluation_results = self._evaluate_model(model, X_test, y_test)
            
            # Calculate additional metrics
            inference_time = self._measure_inference_time(model, X_test)
            memory_usage = self._measure_memory_usage(model)
            
            # Update candidate with results
            candidate.training_time = training_time
            candidate.inference_time = inference_time
            candidate.memory_usage_mb = memory_usage
            candidate.evaluation_results = evaluation_results
            candidate.status = "evaluated"
            
            # Calculate fitness score
            fitness_score = self._calculate_fitness_score(evaluation_results)
            candidate.fitness_score = fitness_score
            
            tprint_success(f"Architecture {candidate.candidate_id} evaluated successfully")
            tprint_info(f"Fitness score: {fitness_score:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            candidate.status = "failed"
            tprint_error(f"Architecture {candidate.candidate_id} evaluation failed: {str(e)}")
            return {}
    
    def _create_model(self, candidate: UnifiedArchitectureCandidate):
        """Create model based on architecture type and parameters."""
        if candidate.architecture_type in [
            ArchitectureType.FEEDFORWARD, ArchitectureType.LSTM, 
            ArchitectureType.GRU, ArchitectureType.TRANSFORMER,
            ArchitectureType.CONVOLUTIONAL, ArchitectureType.ATTENTION
        ]:
            return self._create_neural_model(candidate)
        elif candidate.architecture_type in [
            ArchitectureType.RANDOM_FOREST, ArchitectureType.XGBOOST,
            ArchitectureType.LIGHTGBM, ArchitectureType.EXTRA_TREES,
            ArchitectureType.GRADIENT_BOOSTING, ArchitectureType.DECISION_TREE,
            ArchitectureType.ADABOOST, ArchitectureType.BAGGING
        ]:
            return self._create_tree_model(candidate)
        else:
            raise ValueError(f"Unsupported architecture type: {candidate.architecture_type}")
    
    def _create_neural_model(self, candidate: UnifiedArchitectureCandidate):
        """Create neural network model."""
        # This would integrate with existing NAS model creation
        # For now, return a placeholder
        tprint_info(f"Creating neural model for {candidate.architecture_type.value}")
        return None
    
    def _create_tree_model(self, candidate: UnifiedArchitectureCandidate):
        """Create tree-based model."""
        # This would integrate with existing TAS model creation
        # For now, return a placeholder
        tprint_info(f"Creating tree model for {candidate.architecture_type.value}")
        return None
    
    def _train_model(self, model, X_train, y_train, X_val, y_val) -> float:
        """Train model and return training time."""
        start_time = time.time()
        # Placeholder for actual training
        time.sleep(0.1)  # Simulate training time
        training_time = time.time() - start_time
        return training_time
    
    def _evaluate_model(self, model, X_test, y_test) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        # Placeholder for actual evaluation
        return {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.82,
            'f1_score': 0.825,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15,
            'win_rate': 0.65
        }
    
    def _measure_inference_time(self, model, X_test) -> float:
        """Measure model inference time."""
        # Placeholder for actual inference time measurement
        return 0.001
    
    def _measure_memory_usage(self, model) -> float:
        """Measure model memory usage in MB."""
        # Placeholder for actual memory measurement
        return 50.0
    
    def _calculate_fitness_score(self, evaluation_results: Dict[str, float]) -> float:
        """Calculate weighted fitness score from evaluation results."""
        if not evaluation_results:
            return 0.0
        
        # Use configured objectives and weights
        total_score = 0.0
        total_weight = 0.0
        
        for i, objective in enumerate(self.config.objectives):
            if objective.value in evaluation_results:
                weight = self.config.objective_weights[i] if i < len(self.config.objective_weights) else 0.0
                score = evaluation_results[objective.value]
                total_score += weight * score
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


# ============================================================================
# UNIFIED SEARCH ENGINE
# ============================================================================

class UnifiedSearchEngine:
    """Unified search engine for both NAS and TAS."""
    
    def __init__(self, config: UnifiedArchitectureConfig):
        """Initialize unified search engine."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.candidates = []
        self.best_candidate = None
        self.search_history = []
        
    def search_architectures(self, 
                           X: np.ndarray,
                           y: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None) -> List[UnifiedArchitectureCandidate]:
        """Search for optimal architectures."""
        
        tprint_info(f"Starting unified architecture search with strategy: {self.config.search_strategy.value}")
        
        # Prepare data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self._split_data(X, y)
        else:
            X_train, y_train = X, y
        
        # Initialize evaluator
        evaluator = UnifiedEvaluator(self.config)
        
        # Generate and evaluate candidates
        for trial in range(self.config.max_trials):
            tprint_progress(trial + 1, self.config.max_trials, f"Trial {trial + 1}")
            
            # Generate candidate
            candidate = self._generate_candidate(trial)
            
            # Evaluate candidate
            evaluation_results = evaluator.evaluate_architecture(
                candidate, X_train, y_train, X_val, y_val, X_val, y_val
            )
            
            # Store candidate
            self.candidates.append(candidate)
            
            # Update best candidate
            if self.best_candidate is None or candidate.fitness_score > self.best_candidate.fitness_score:
                self.best_candidate = candidate
                tprint_info(f"New best candidate: {candidate.candidate_id} (fitness: {candidate.fitness_score:.4f})")
            
            # Record search history
            self.search_history.append({
                'trial': trial,
                'candidate_id': candidate.candidate_id,
                'fitness_score': candidate.fitness_score,
                'timestamp': datetime.now().isoformat()
            })
            
            # Early stopping check
            if self._should_stop_early(trial):
                tprint_info(f"Early stopping at trial {trial + 1}")
                break
        
        tprint_success(f"Search completed. Best fitness: {self.best_candidate.fitness_score:.4f}")
        return self.candidates
    
    def _generate_candidate(self, trial: int) -> UnifiedArchitectureCandidate:
        """Generate architecture candidate based on search strategy."""
        candidate_id = f"candidate_{trial}_{int(time.time())}"
        
        # Generate parameters based on architecture type
        if self.config.architecture_type in [
            ArchitectureType.FEEDFORWARD, ArchitectureType.LSTM, 
            ArchitectureType.GRU, ArchitectureType.TRANSFORMER,
            ArchitectureType.CONVOLUTIONAL, ArchitectureType.ATTENTION
        ]:
            parameters = self._generate_neural_parameters()
        else:
            parameters = self._generate_tree_parameters()
        
        candidate = UnifiedArchitectureCandidate(
            candidate_id=candidate_id,
            architecture_type=self.config.architecture_type,
            parameters=parameters,
            search_iteration=trial,
            search_strategy=self.config.search_strategy.value
        )
        
        return candidate
    
    def _generate_neural_parameters(self) -> Dict[str, Any]:
        """Generate neural network parameters."""
        import random
        
        n_layers = random.randint(self.config.min_layers, self.config.max_layers)
        layers = []
        
        for i in range(n_layers):
            layer_size = random.randint(self.config.min_neurons, self.config.max_neurons)
            activation = random.choice(self.config.activation_functions)
            dropout = random.choice(self.config.dropout_rates)
            
            layers.append({
                'size': layer_size,
                'activation': activation,
                'dropout': dropout
            })
        
        return {
            'layers': layers,
            'learning_rate': random.uniform(*self.config.learning_rate_range),
            'batch_size': random.randint(*self.config.batch_size_range),
            'optimizer': 'adam'
        }
    
    def _generate_tree_parameters(self) -> Dict[str, Any]:
        """Generate tree model parameters."""
        import random
        
        return {
            'n_trees': random.randint(self.config.min_trees, self.config.max_trees),
            'max_depth': random.randint(self.config.min_depth, self.config.max_depth),
            'min_samples_split': random.randint(self.config.min_samples_split, 20),
            'min_samples_leaf': random.randint(self.config.min_samples_leaf, 10),
            'max_features': random.choice(['auto', 'sqrt', 'log2', 0.5, 0.8, 1.0]),
            'learning_rate': random.uniform(*self.config.learning_rate_range) if self.config.architecture_type in [ArchitectureType.XGBOOST, ArchitectureType.LIGHTGBM] else None
        }
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and validation sets."""
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.validation_split, random_state=42
        )
        return X_train, X_val, y_train, y_val
    
    def _should_stop_early(self, trial: int) -> bool:
        """Check if early stopping should be applied."""
        if trial < self.config.early_stopping_patience:
            return False
        
        # Check if fitness has improved in the last few trials
        recent_trials = self.candidates[-self.config.early_stopping_patience:]
        if len(recent_trials) < self.config.early_stopping_patience:
            return False
        
        best_recent = max(recent_trials, key=lambda c: c.fitness_score)
        if best_recent.fitness_score <= self.best_candidate.fitness_score:
            return True
        
        return False


# ============================================================================
# UNIFIED HYBRID SYSTEM
# ============================================================================

class UnifiedHybridSystem:
    """Main unified hybrid system that combines NAS and TAS capabilities."""
    
    def __init__(self, config: Optional[UnifiedArchitectureConfig] = None):
        """Initialize unified hybrid system."""
        self.config = config or UnifiedArchitectureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.search_engine = UnifiedSearchEngine(self.config)
        self.results = {}
        
    def run_architecture_search(self, 
                              X: np.ndarray,
                              y: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run unified architecture search."""
        
        tprint_info("Starting Unified Hybrid Architecture Search")
        tprint_info(f"Architecture type: {self.config.architecture_type.value}")
        tprint_info(f"Search strategy: {self.config.search_strategy.value}")
        tprint_info(f"Max trials: {self.config.max_trials}")
        
        start_time = time.time()
        
        # Run search
        candidates = self.search_engine.search_architectures(X, y, X_val, y_val)
        
        # Compile results
        execution_time = time.time() - start_time
        self.results = {
            'search_completed': True,
            'execution_time': execution_time,
            'total_candidates': len(candidates),
            'best_candidate': self.search_engine.best_candidate.to_dict() if self.search_engine.best_candidate else None,
            'search_history': self.search_engine.search_history,
            'config': {
                'architecture_type': self.config.architecture_type.value,
                'search_strategy': self.config.search_strategy.value,
                'max_trials': self.config.max_trials,
                'objectives': [obj.value for obj in self.config.objectives],
                'objective_weights': self.config.objective_weights
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results if configured
        if self.config.save_results:
            self._save_results()
        
        tprint_success(f"Architecture search completed in {execution_time:.2f} seconds")
        tprint_info(f"Best fitness score: {self.search_engine.best_candidate.fitness_score:.4f}")
        
        return self.results
    
    def _save_results(self):
        """Save search results to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save main results
        results_file = output_dir / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save all candidates
        candidates_file = output_dir / "all_candidates.json"
        candidates_data = [candidate.to_dict() for candidate in self.search_engine.candidates]
        with open(candidates_file, 'w') as f:
            json.dump(candidates_data, f, indent=2)
        
        tprint_info(f"Results saved to {output_dir}")
    
    def get_best_architecture(self) -> Optional[UnifiedArchitectureCandidate]:
        """Get the best architecture found."""
        return self.search_engine.best_candidate
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of search results."""
        if not self.results:
            return {}
        
        return {
            'total_candidates': self.results['total_candidates'],
            'execution_time': self.results['execution_time'],
            'best_fitness_score': self.search_engine.best_candidate.fitness_score if self.search_engine.best_candidate else 0.0,
            'architecture_type': self.results['config']['architecture_type'],
            'search_strategy': self.results['config']['search_strategy']
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_unified_hybrid_system(architecture_type: str = "feedforward",
                                search_strategy: str = "random",
                                max_trials: int = 100) -> UnifiedHybridSystem:
    """Create a unified hybrid system with default configuration."""
    
    config = UnifiedArchitectureConfig(
        architecture_type=ArchitectureType(architecture_type),
        search_strategy=SearchStrategy(search_strategy),
        max_trials=max_trials
    )
    
    return UnifiedHybridSystem(config)


def run_quick_search(X: np.ndarray,
                    y: np.ndarray,
                    architecture_type: str = "feedforward",
                    max_trials: int = 50) -> Dict[str, Any]:
    """Run a quick architecture search with default settings."""
    
    system = create_unified_hybrid_system(
        architecture_type=architecture_type,
        max_trials=max_trials
    )
    
    return system.run_architecture_search(X, y)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Create unified hybrid system
    system = create_unified_hybrid_system(
        architecture_type="feedforward",
        search_strategy="random",
        max_trials=10
    )
    
    # Run search
    results = system.run_architecture_search(X, y)
    
    # Print results
    print("\n" + "="*50)
    print("UNIFIED HYBRID ARCHITECTURE SEARCH RESULTS")
    print("="*50)
    
    summary = system.get_search_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    best_architecture = system.get_best_architecture()
    if best_architecture:
        print(f"\nBest Architecture ID: {best_architecture.candidate_id}")
        print(f"Best Fitness Score: {best_architecture.fitness_score:.4f}")
        print(f"Architecture Type: {best_architecture.architecture_type.value}")
        print(f"Training Time: {best_architecture.training_time:.4f}s")
        print(f"Inference Time: {best_architecture.inference_time:.6f}s")
        print(f"Memory Usage: {best_architecture.memory_usage_mb:.2f} MB")