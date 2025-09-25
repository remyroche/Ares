"""
NAS Regime Unified Search Integration

This module provides integration between NAS regime detection and the unified
search algorithms framework, replacing legacy search implementations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import unified search algorithms
try:
    from src.utils.nas_tas import (
        SearchManager,
        SearchConfig,
        SearchResult,
        SearchAlgorithmType,
        create_search_manager,
        optimize_with_bayesian,
        optimize_with_evolutionary
    )
    UNIFIED_SEARCH_AVAILABLE = True
except ImportError:
    UNIFIED_SEARCH_AVAILABLE = False

# Import NAS regime components
from ..core.enhanced_perfect_nas_regime_detector import PerfectNASRegimeDetector, PerfectNASConfig

logger = logging.getLogger(__name__)


@dataclass
class NASSearchConfig:
    """Configuration for NAS regime search integration."""
    
    # Search algorithm parameters
    search_algorithm: SearchAlgorithmType = SearchAlgorithmType.BAYESIAN_OPTIMIZATION
    max_iterations: int = 100
    population_size: int = 50
    
    # NAS-specific parameters
    nas_regime_config: PerfectNASConfig = None
    enable_neural_architecture_search: bool = True
    enable_hyperparameter_optimization: bool = True
    
    # Multi-objective parameters
    enable_multi_objective: bool = True
    objectives: List[str] = None
    
    # Performance parameters
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "efficiency", "economic_significance"]


class NASUnifiedSearchIntegration:
    """
    Integration between NAS regime detection and unified search algorithms.
    
    This class replaces legacy NAS search implementations with the unified
    framework while maintaining NAS-specific functionality.
    """
    
    def __init__(self, config: NASSearchConfig):
        """Initialize NAS unified search integration."""
        if not UNIFIED_SEARCH_AVAILABLE:
            raise ImportError("Unified search algorithms framework not available")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize NAS regime detector
        if self.config.nas_regime_config:
            self.nas_detector = PerfectNASRegimeDetector(self.config.nas_regime_config)
        else:
            self.nas_detector = PerfectNASRegimeDetector()
        
        # Initialize unified search manager
        self.search_config = SearchConfig(
            algorithm_type=self.config.search_algorithm,
            max_iterations=self.config.max_iterations,
            population_size=self.config.population_size,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            enable_multi_objective=self.config.enable_multi_objective,
            objectives=self.config.objectives
        )
        self.search_manager = SearchManager(self.search_config)
        
        self.logger.info("NAS unified search integration initialized")
    
    def optimize_nas_regime_detection(
        self,
        data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> SearchResult:
        """
        Optimize NAS regime detection parameters using unified search.
        
        Args:
            data: Training data for regime detection
            validation_data: Validation data (optional)
            
        Returns:
            SearchResult with optimization results
        """
        self.logger.info("Starting NAS regime detection optimization")
        
        # Define parameter space for NAS regime detection
        parameter_space = self._define_nas_parameter_space()
        
        # Create objective function
        objective_function = self._create_nas_objective_function(data, validation_data)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=self.config.search_algorithm
        )
        
        self.logger.info(f"NAS optimization completed: {result.best_score:.4f}")
        return result
    
    def optimize_neural_architecture(
        self,
        data: pd.DataFrame,
        target_column: str = "regime"
    ) -> SearchResult:
        """
        Optimize neural architecture for regime detection.
        
        Args:
            data: Training data
            target_column: Target column name
            
        Returns:
            SearchResult with architecture optimization results
        """
        if not self.config.enable_neural_architecture_search:
            raise ValueError("Neural architecture search not enabled")
        
        self.logger.info("Starting neural architecture optimization")
        
        # Define neural architecture parameter space
        parameter_space = self._define_neural_architecture_space()
        
        # Create architecture objective function
        objective_function = self._create_architecture_objective_function(data, target_column)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=SearchAlgorithmType.EVOLUTIONARY_ALGORITHM
        )
        
        self.logger.info(f"Neural architecture optimization completed: {result.best_score:.4f}")
        return result
    
    def optimize_hyperparameters(
        self,
        data: pd.DataFrame,
        model_type: str = "neural_network"
    ) -> SearchResult:
        """
        Optimize hyperparameters for NAS models.
        
        Args:
            data: Training data
            model_type: Type of model to optimize
            
        Returns:
            SearchResult with hyperparameter optimization results
        """
        if not self.config.enable_hyperparameter_optimization:
            raise ValueError("Hyperparameter optimization not enabled")
        
        self.logger.info(f"Starting hyperparameter optimization for {model_type}")
        
        # Define hyperparameter space
        parameter_space = self._define_hyperparameter_space(model_type)
        
        # Create hyperparameter objective function
        objective_function = self._create_hyperparameter_objective_function(data, model_type)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=SearchAlgorithmType.BAYESIAN_OPTIMIZATION
        )
        
        self.logger.info(f"Hyperparameter optimization completed: {result.best_score:.4f}")
        return result
    
    def _define_nas_parameter_space(self) -> Dict[str, Any]:
        """Define parameter space for NAS regime detection."""
        return {
            'clustering_threshold': {
                'type': 'continuous',
                'min': 0.1,
                'max': 0.9
            },
            'n_clusters': {
                'type': 'integer',
                'min': 2,
                'max': 10
            },
            'regime_stability_threshold': {
                'type': 'continuous',
                'min': 0.5,
                'max': 0.95
            },
            'feature_selection_method': {
                'type': 'discrete',
                'values': ['correlation', 'mutual_info', 'variance', 'all']
            },
            'window_size': {
                'type': 'integer',
                'min': 20,
                'max': 200
            }
        }
    
    def _define_neural_architecture_space(self) -> Dict[str, Any]:
        """Define parameter space for neural architecture."""
        return {
            'n_layers': {
                'type': 'integer',
                'min': 1,
                'max': 5
            },
            'hidden_size': {
                'type': 'integer',
                'min': 32,
                'max': 512
            },
            'dropout_rate': {
                'type': 'continuous',
                'min': 0.0,
                'max': 0.5
            },
            'activation_function': {
                'type': 'discrete',
                'values': ['relu', 'tanh', 'sigmoid', 'swish']
            },
            'learning_rate': {
                'type': 'continuous',
                'min': 1e-4,
                'max': 1e-1
            },
            'batch_size': {
                'type': 'discrete',
                'values': [16, 32, 64, 128]
            }
        }
    
    def _define_hyperparameter_space(self, model_type: str) -> Dict[str, Any]:
        """Define hyperparameter space based on model type."""
        if model_type == "neural_network":
            return {
                'learning_rate': {
                    'type': 'continuous',
                    'min': 1e-4,
                    'max': 1e-1
                },
                'batch_size': {
                    'type': 'discrete',
                    'values': [16, 32, 64, 128]
                },
                'epochs': {
                    'type': 'integer',
                    'min': 10,
                    'max': 100
                },
                'optimizer': {
                    'type': 'discrete',
                    'values': ['adam', 'sgd', 'rmsprop']
                }
            }
        elif model_type == "svm":
            return {
                'C': {
                    'type': 'continuous',
                    'min': 0.1,
                    'max': 10.0
                },
                'gamma': {
                    'type': 'continuous',
                    'min': 1e-4,
                    'max': 1.0
                },
                'kernel': {
                    'type': 'discrete',
                    'values': ['rbf', 'linear', 'poly']
                }
            }
        else:
            # Default parameter space
            return {
                'alpha': {
                    'type': 'continuous',
                    'min': 0.001,
                    'max': 1.0
                },
                'max_iter': {
                    'type': 'integer',
                    'min': 100,
                    'max': 1000
                }
            }
    
    def _create_nas_objective_function(
        self,
        data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Callable:
        """Create objective function for NAS regime detection optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Update NAS detector configuration
                nas_config = PerfectNASConfig(
                    clustering_threshold=parameters['clustering_threshold'],
                    n_clusters=parameters['n_clusters'],
                    regime_stability_threshold=parameters['regime_stability_threshold'],
                    feature_selection_method=parameters['feature_selection_method'],
                    window_size=parameters['window_size']
                )
                
                # Create new detector with updated config
                detector = PerfectNASRegimeDetector(nas_config)
                
                # Train detector
                detector.fit(data)
                
                # Evaluate performance
                if validation_data is not None:
                    predictions = detector.predict(validation_data)
                    actual = validation_data.get('regime', validation_data.iloc[:, -1])
                else:
                    # Use cross-validation
                    predictions = detector.predict(data)
                    actual = data.get('regime', data.iloc[:, -1])
                
                # Calculate accuracy
                accuracy = np.mean(predictions == actual)
                
                # Calculate economic significance
                economic_score = self._calculate_economic_significance(data, predictions)
                
                # Combine scores
                total_score = 0.7 * accuracy + 0.3 * economic_score
                
                return total_score
                
            except Exception as e:
                self.logger.warning(f"Objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _create_architecture_objective_function(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Callable:
        """Create objective function for neural architecture optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Build neural network architecture
                from sklearn.neural_network import MLPClassifier
                
                # Extract features and target
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Create model with specified architecture
                model = MLPClassifier(
                    hidden_layer_sizes=tuple([parameters['hidden_size']] * parameters['n_layers']),
                    dropout=parameters['dropout_rate'],
                    activation=parameters['activation_function'],
                    learning_rate_init=parameters['learning_rate'],
                    batch_size=parameters['batch_size'],
                    max_iter=100,  # Fixed for speed
                    random_state=42
                )
                
                # Train and evaluate
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                
                # Calculate efficiency score (simpler architectures are better)
                complexity_penalty = parameters['n_layers'] * parameters['hidden_size'] / 1000
                efficiency_score = 1.0 / (1.0 + complexity_penalty)
                
                # Combine accuracy and efficiency
                total_score = 0.8 * np.mean(scores) + 0.2 * efficiency_score
                
                return total_score
                
            except Exception as e:
                self.logger.warning(f"Architecture objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _create_hyperparameter_objective_function(
        self,
        data: pd.DataFrame,
        model_type: str
    ) -> Callable:
        """Create objective function for hyperparameter optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                # Split data
                from sklearn.model_selection import train_test_split
                X = data.drop(columns=[data.columns[-1]])  # Assume last column is target
                y = data.iloc[:, -1]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Create model based on type
                if model_type == "neural_network":
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(
                        learning_rate_init=parameters['learning_rate'],
                        batch_size=parameters['batch_size'],
                        max_iter=parameters['epochs'],
                        solver=parameters['optimizer'],
                        random_state=42
                    )
                elif model_type == "svm":
                    from sklearn.svm import SVC
                    model = SVC(
                        C=parameters['C'],
                        gamma=parameters['gamma'],
                        kernel=parameters['kernel'],
                        random_state=42
                    )
                else:
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression(
                        C=parameters['alpha'],
                        max_iter=parameters['max_iter'],
                        random_state=42
                    )
                
                # Train and evaluate
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Hyperparameter objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _calculate_economic_significance(self, data: pd.DataFrame, predictions: np.ndarray) -> float:
        """Calculate economic significance of regime predictions."""
        try:
            # Simple economic significance calculation
            # In practice, this would be more sophisticated
            
            # Calculate regime stability
            regime_changes = np.sum(np.diff(predictions) != 0)
            stability_score = 1.0 / (1.0 + regime_changes / len(predictions))
            
            # Calculate regime distribution balance
            unique_regimes, counts = np.unique(predictions, return_counts=True)
            balance_score = 1.0 - np.std(counts) / np.mean(counts)
            
            # Combine scores
            economic_score = 0.6 * stability_score + 0.4 * balance_score
            
            return economic_score
            
        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def compare_search_algorithms(
        self,
        data: pd.DataFrame,
        n_trials: int = 3
    ) -> Dict[str, SearchResult]:
        """
        Compare different search algorithms for NAS optimization.
        
        Args:
            data: Training data
            n_trials: Number of trials per algorithm
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        self.logger.info("Comparing search algorithms for NAS optimization")
        
        # Define parameter space
        parameter_space = self._define_nas_parameter_space()
        
        # Create objective function
        objective_function = self._create_nas_objective_function(data)
        
        # Compare algorithms
        algorithms = [
            SearchAlgorithmType.BAYESIAN_OPTIMIZATION,
            SearchAlgorithmType.EVOLUTIONARY_ALGORITHM,
            SearchAlgorithmType.RANDOM_SEARCH
        ]
        
        results = self.search_manager.compare_algorithms(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithms=algorithms,
            n_trials=n_trials
        )
        
        self.logger.info("Algorithm comparison completed")
        return results
    
    def generate_search_report(self, result: SearchResult) -> str:
        """Generate NAS search optimization report."""
        report = []
        report.append("=" * 80)
        report.append("NAS REGIME SEARCH OPTIMIZATION REPORT")
        report.append("=" * 80)
        
        # Optimization results
        report.append(f"\nOPTIMIZATION RESULTS:")
        report.append(f"Algorithm Used: {result.algorithm_used}")
        report.append(f"Best Score: {result.best_score:.4f}")
        report.append(f"Number of Evaluations: {result.n_evaluations}")
        report.append(f"Execution Time: {result.execution_time:.2f} seconds")
        report.append(f"Converged: {result.converged}")
        
        # Best parameters
        report.append(f"\nBEST PARAMETERS:")
        for param, value in result.best_parameters.items():
            report.append(f"{param}: {value}")
        
        # Convergence information
        if result.converged:
            report.append(f"\nCONVERGENCE:")
            report.append(f"Converged at iteration: {result.convergence_iteration}")
            report.append(f"Final improvement: {result.final_improvement:.6f}")
        
        # Search history summary
        if result.search_history:
            report.append(f"\nSEARCH HISTORY:")
            report.append(f"Total iterations: {len(result.search_history)}")
            report.append(f"Score range: {min(h['score'] for h in result.search_history):.4f} - {max(h['score'] for h in result.search_history):.4f}")
        
        # Errors and warnings
        if result.warnings:
            report.append(f"\nWARNINGS:")
            for warning in result.warnings:
                report.append(f"- {warning}")
        
        if result.errors:
            report.append(f"\nERRORS:")
            for error in result.errors:
                report.append(f"- {error}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Convenience functions for backward compatibility
def optimize_nas_regime_with_unified_search(
    data: pd.DataFrame,
    config: Optional[NASSearchConfig] = None
) -> SearchResult:
    """Optimize NAS regime detection using unified search (backward compatibility)."""
    if config is None:
        config = NASSearchConfig()
    
    integration = NASUnifiedSearchIntegration(config)
    return integration.optimize_nas_regime_detection(data)


def create_nas_search_config() -> NASSearchConfig:
    """Create default NAS search configuration."""
    return NASSearchConfig(
        search_algorithm=SearchAlgorithmType.BAYESIAN_OPTIMIZATION,
        max_iterations=100,
        population_size=50,
        enable_neural_architecture_search=True,
        enable_hyperparameter_optimization=True,
        enable_multi_objective=True
    )