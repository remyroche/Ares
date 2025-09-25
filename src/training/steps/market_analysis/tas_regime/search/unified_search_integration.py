"""
TAS Regime Unified Search Integration

This module provides integration between TAS regime detection and the unified
search algorithms framework, replacing legacy TAS search implementations.
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

# Import TAS regime components
from ..core.tas_regime_detector import TASRegimeDetector, TASRegimeConfig
from ..search.tree_search_space import TreeSearchSpace

logger = logging.getLogger(__name__)


@dataclass
class TASSearchConfig:
    """Configuration for TAS regime search integration."""
    
    # Search algorithm parameters
    search_algorithm: SearchAlgorithmType = SearchAlgorithmType.TREE_BASED_SEARCH
    max_iterations: int = 100
    population_size: int = 50
    
    # TAS-specific parameters
    tas_regime_config: TASRegimeConfig = None
    enable_tree_architecture_search: bool = True
    enable_feature_selection: bool = True
    
    # Multi-objective parameters
    enable_multi_objective: bool = True
    objectives: List[str] = None
    
    # Performance parameters
    n_jobs: int = -1
    verbose: bool = True
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["accuracy", "interpretability", "economic_significance"]


class TASUnifiedSearchIntegration:
    """
    Integration between TAS regime detection and unified search algorithms.
    
    This class replaces legacy TAS search implementations with the unified
    framework while maintaining TAS-specific functionality.
    """
    
    def __init__(self, config: TASSearchConfig):
        """Initialize TAS unified search integration."""
        if not UNIFIED_SEARCH_AVAILABLE:
            raise ImportError("Unified search algorithms framework not available")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize TAS regime detector
        if self.config.tas_regime_config:
            self.tas_detector = TASRegimeDetector(self.config.tas_regime_config)
        else:
            self.tas_detector = TASRegimeDetector()
        
        # Initialize tree search space
        self.tree_search_space = TreeSearchSpace()
        
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
        
        self.logger.info("TAS unified search integration initialized")
    
    def optimize_tas_regime_detection(
        self,
        data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> SearchResult:
        """
        Optimize TAS regime detection parameters using unified search.
        
        Args:
            data: Training data for regime detection
            validation_data: Validation data (optional)
            
        Returns:
            SearchResult with optimization results
        """
        self.logger.info("Starting TAS regime detection optimization")
        
        # Define parameter space for TAS regime detection
        parameter_space = self._define_tas_parameter_space()
        
        # Create objective function
        objective_function = self._create_tas_objective_function(data, validation_data)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=self.config.search_algorithm
        )
        
        self.logger.info(f"TAS optimization completed: {result.best_score:.4f}")
        return result
    
    def optimize_tree_architecture(
        self,
        data: pd.DataFrame,
        target_column: str = "regime"
    ) -> SearchResult:
        """
        Optimize tree architecture for regime detection.
        
        Args:
            data: Training data
            target_column: Target column name
            
        Returns:
            SearchResult with tree architecture optimization results
        """
        if not self.config.enable_tree_architecture_search:
            raise ValueError("Tree architecture search not enabled")
        
        self.logger.info("Starting tree architecture optimization")
        
        # Define tree architecture parameter space
        parameter_space = self._define_tree_architecture_space()
        
        # Create tree architecture objective function
        objective_function = self._create_tree_architecture_objective_function(data, target_column)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=SearchAlgorithmType.EVOLUTIONARY_ALGORITHM
        )
        
        self.logger.info(f"Tree architecture optimization completed: {result.best_score:.4f}")
        return result
    
    def optimize_feature_selection(
        self,
        data: pd.DataFrame,
        target_column: str = "regime"
    ) -> SearchResult:
        """
        Optimize feature selection for TAS models.
        
        Args:
            data: Training data
            target_column: Target column name
            
        Returns:
            SearchResult with feature selection optimization results
        """
        if not self.config.enable_feature_selection:
            raise ValueError("Feature selection optimization not enabled")
        
        self.logger.info("Starting feature selection optimization")
        
        # Define feature selection parameter space
        parameter_space = self._define_feature_selection_space(data.columns)
        
        # Create feature selection objective function
        objective_function = self._create_feature_selection_objective_function(data, target_column)
        
        # Run optimization
        result = self.search_manager.optimize(
            objective_function=objective_function,
            parameter_space=parameter_space,
            algorithm_type=SearchAlgorithmType.BAYESIAN_OPTIMIZATION
        )
        
        self.logger.info(f"Feature selection optimization completed: {result.best_score:.4f}")
        return result
    
    def _define_tas_parameter_space(self) -> Dict[str, Any]:
        """Define parameter space for TAS regime detection."""
        return {
            'max_depth': {
                'type': 'integer',
                'min': 3,
                'max': 15
            },
            'min_samples_split': {
                'type': 'integer',
                'min': 2,
                'max': 50
            },
            'min_samples_leaf': {
                'type': 'integer',
                'min': 1,
                'max': 20
            },
            'criterion': {
                'type': 'discrete',
                'values': ['gini', 'entropy', 'log_loss']
            },
            'splitter': {
                'type': 'discrete',
                'values': ['best', 'random']
            },
            'max_features': {
                'type': 'discrete',
                'values': ['sqrt', 'log2', None, 0.5, 0.7, 0.9]
            },
            'regime_stability_threshold': {
                'type': 'continuous',
                'min': 0.5,
                'max': 0.95
            },
            'feature_importance_threshold': {
                'type': 'continuous',
                'min': 0.01,
                'max': 0.1
            }
        }
    
    def _define_tree_architecture_space(self) -> Dict[str, Any]:
        """Define parameter space for tree architecture."""
        return {
            'max_depth': {
                'type': 'integer',
                'min': 2,
                'max': 20
            },
            'min_samples_split': {
                'type': 'integer',
                'min': 2,
                'max': 100
            },
            'min_samples_leaf': {
                'type': 'integer',
                'min': 1,
                'max': 50
            },
            'max_leaf_nodes': {
                'type': 'integer',
                'min': 10,
                'max': 1000
            },
            'min_impurity_decrease': {
                'type': 'continuous',
                'min': 0.0,
                'max': 0.1
            },
            'ccp_alpha': {
                'type': 'continuous',
                'min': 0.0,
                'max': 0.01
            }
        }
    
    def _define_feature_selection_space(self, feature_columns: List[str]) -> Dict[str, Any]:
        """Define parameter space for feature selection."""
        return {
            'max_features': {
                'type': 'integer',
                'min': 1,
                'max': min(len(feature_columns), 50)
            },
            'feature_selection_method': {
                'type': 'discrete',
                'values': ['mutual_info', 'f_score', 'chi2', 'variance', 'correlation']
            },
            'selection_threshold': {
                'type': 'continuous',
                'min': 0.01,
                'max': 0.5
            },
            'use_feature_interactions': {
                'type': 'discrete',
                'values': [True, False]
            },
            'interaction_depth': {
                'type': 'integer',
                'min': 1,
                'max': 3
            }
        }
    
    def _create_tas_objective_function(
        self,
        data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Callable:
        """Create objective function for TAS regime detection optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import cross_val_score
                
                # Prepare data
                X = data.drop(columns=[data.columns[-1]])  # Assume last column is target
                y = data.iloc[:, -1]
                
                # Create TAS model with specified parameters
                model = DecisionTreeClassifier(
                    max_depth=parameters['max_depth'],
                    min_samples_split=parameters['min_samples_split'],
                    min_samples_leaf=parameters['min_samples_leaf'],
                    criterion=parameters['criterion'],
                    splitter=parameters['splitter'],
                    max_features=parameters['max_features'],
                    random_state=42
                )
                
                # Train and evaluate
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                accuracy = np.mean(scores)
                
                # Calculate interpretability score (simpler trees are better)
                model.fit(X, y)
                tree_complexity = model.get_depth() * model.get_n_leaves()
                interpretability_score = 1.0 / (1.0 + tree_complexity / 100)
                
                # Calculate economic significance
                economic_score = self._calculate_tas_economic_significance(data, model)
                
                # Combine scores
                total_score = (0.5 * accuracy + 
                             0.3 * interpretability_score + 
                             0.2 * economic_score)
                
                return total_score
                
            except Exception as e:
                self.logger.warning(f"TAS objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _create_tree_architecture_objective_function(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Callable:
        """Create objective function for tree architecture optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import cross_val_score
                
                # Extract features and target
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Create model with specified architecture
                model = DecisionTreeClassifier(
                    max_depth=parameters['max_depth'],
                    min_samples_split=parameters['min_samples_split'],
                    min_samples_leaf=parameters['min_samples_leaf'],
                    max_leaf_nodes=parameters['max_leaf_nodes'],
                    min_impurity_decrease=parameters['min_impurity_decrease'],
                    ccp_alpha=parameters['ccp_alpha'],
                    random_state=42
                )
                
                # Train and evaluate
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                accuracy = np.mean(scores)
                
                # Calculate complexity penalty
                model.fit(X, y)
                complexity_penalty = (model.get_depth() * model.get_n_leaves()) / 1000
                efficiency_score = 1.0 / (1.0 + complexity_penalty)
                
                # Combine accuracy and efficiency
                total_score = 0.8 * accuracy + 0.2 * efficiency_score
                
                return total_score
                
            except Exception as e:
                self.logger.warning(f"Tree architecture objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _create_feature_selection_objective_function(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Callable:
        """Create objective function for feature selection optimization."""
        
        def objective_function(parameters: Dict[str, Any]) -> float:
            try:
                from sklearn.feature_selection import (
                    SelectKBest, f_classif, mutual_info_classif, 
                    chi2, VarianceThreshold
                )
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import cross_val_score
                from sklearn.preprocessing import StandardScaler
                
                # Extract features and target
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Apply feature selection
                max_features = parameters['max_features']
                method = parameters['feature_selection_method']
                
                if method == 'mutual_info':
                    selector = SelectKBest(mutual_info_classif, k=max_features)
                elif method == 'f_score':
                    selector = SelectKBest(f_classif, k=max_features)
                elif method == 'chi2':
                    selector = SelectKBest(chi2, k=max_features)
                elif method == 'variance':
                    selector = VarianceThreshold(threshold=parameters['selection_threshold'])
                else:  # correlation
                    # Simple correlation-based selection
                    correlations = X.corrwith(y).abs()
                    selected_features = correlations.nlargest(max_features).index
                    X_selected = X[selected_features]
                else:
                    X_selected = X
                
                if method != 'correlation':
                    X_selected = selector.fit_transform(X, y)
                
                # Standardize features
                scaler = StandardScaler()
                X_selected = scaler.fit_transform(X_selected)
                
                # Train model on selected features
                model = DecisionTreeClassifier(random_state=42)
                scores = cross_val_score(model, X_selected, y, cv=3, scoring='accuracy')
                
                # Calculate feature efficiency (fewer features is better)
                n_features = X_selected.shape[1]
                feature_efficiency = 1.0 / (1.0 + n_features / 100)
                
                # Combine accuracy and feature efficiency
                total_score = 0.8 * np.mean(scores) + 0.2 * feature_efficiency
                
                return total_score
                
            except Exception as e:
                self.logger.warning(f"Feature selection objective function evaluation failed: {e}")
                return 0.0
        
        return objective_function
    
    def _calculate_tas_economic_significance(self, data: pd.DataFrame, model) -> float:
        """Calculate economic significance of TAS regime predictions."""
        try:
            # Get feature importances
            feature_importances = model.feature_importances_
            
            # Calculate feature importance diversity
            importance_entropy = -np.sum(feature_importances * np.log(feature_importances + 1e-10))
            diversity_score = importance_entropy / np.log(len(feature_importances))
            
            # Calculate regime prediction stability
            X = data.drop(columns=[data.columns[-1]])
            predictions = model.predict(X)
            regime_stability = 1.0 / (1.0 + np.sum(np.diff(predictions) != 0) / len(predictions))
            
            # Calculate interpretability (simpler trees are more interpretable)
            tree_depth = model.get_depth()
            tree_leaves = model.get_n_leaves()
            interpretability = 1.0 / (1.0 + (tree_depth * tree_leaves) / 1000)
            
            # Combine scores
            economic_score = (0.4 * diversity_score + 
                            0.3 * regime_stability + 
                            0.3 * interpretability)
            
            return economic_score
            
        except Exception as e:
            self.logger.warning(f"TAS economic significance calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def compare_search_algorithms(
        self,
        data: pd.DataFrame,
        n_trials: int = 3
    ) -> Dict[str, SearchResult]:
        """
        Compare different search algorithms for TAS optimization.
        
        Args:
            data: Training data
            n_trials: Number of trials per algorithm
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        self.logger.info("Comparing search algorithms for TAS optimization")
        
        # Define parameter space
        parameter_space = self._define_tas_parameter_space()
        
        # Create objective function
        objective_function = self._create_tas_objective_function(data)
        
        # Compare algorithms
        algorithms = [
            SearchAlgorithmType.EVOLUTIONARY_ALGORITHM,
            SearchAlgorithmType.BAYESIAN_OPTIMIZATION,
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
        """Generate TAS search optimization report."""
        report = []
        report.append("=" * 80)
        report.append("TAS REGIME SEARCH OPTIMIZATION REPORT")
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
        
        # Tree-specific analysis
        if result.best_parameters:
            report.append(f"\nTREE ANALYSIS:")
            max_depth = result.best_parameters.get('max_depth', 'N/A')
            min_samples_split = result.best_parameters.get('min_samples_split', 'N/A')
            criterion = result.best_parameters.get('criterion', 'N/A')
            report.append(f"Max Depth: {max_depth}")
            report.append(f"Min Samples Split: {min_samples_split}")
            report.append(f"Criterion: {criterion}")
        
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
def optimize_tas_regime_with_unified_search(
    data: pd.DataFrame,
    config: Optional[TASSearchConfig] = None
) -> SearchResult:
    """Optimize TAS regime detection using unified search (backward compatibility)."""
    if config is None:
        config = TASSearchConfig()
    
    integration = TASUnifiedSearchIntegration(config)
    return integration.optimize_tas_regime_detection(data)


def create_tas_search_config() -> TASSearchConfig:
    """Create default TAS search configuration."""
    return TASSearchConfig(
        search_algorithm=SearchAlgorithmType.TREE_BASED_SEARCH,
        max_iterations=100,
        population_size=50,
        enable_tree_architecture_search=True,
        enable_feature_selection=True,
        enable_multi_objective=True
    )