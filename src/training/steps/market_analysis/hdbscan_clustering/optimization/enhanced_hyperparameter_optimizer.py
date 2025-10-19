"""
Enhanced Hyperparameter Optimization for HDBSCAN Clustering

This module provides comprehensive hyperparameter optimization using
Bayesian TPE optimization and grid search hybrid approaches.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Import Bayesian TPE optimizer
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
    create_bayesian_tpe_optimizer
)

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    hdbscan = None

logger = logging.getLogger(__name__)

@dataclass
class HDBSCANHyperparameterConfig:
    """Configuration for HDBSCAN hyperparameter optimization."""
    # Optimization strategy
    optimization_strategy: str = "hybrid"  # "grid", "tpe", "hybrid"
    n_trials: int = 50
    n_jobs: int = -1
    
    # Grid search parameters
    grid_search_enabled: bool = True
    coarse_grid_trials: int = 20
    fine_grid_trials: int = 30
    
    # TPE parameters
    tpe_enabled: bool = True
    tpe_trials: int = 50
    tpe_sampler_kwargs: Optional[Dict[str, Any]] = None
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold"
    
    # Evaluation metrics
    primary_metric: str = "silhouette"  # "silhouette", "calinski_harabasz", "davies_bouldin"
    secondary_metrics: List[str] = None
    
    # Performance optimization
    enable_parallel: bool = True
    memory_efficient: bool = True
    early_stopping_patience: int = 10

class EnhancedHyperparameterOptimizer:
    """
    Enhanced hyperparameter optimizer for HDBSCAN clustering.
    
    Provides:
    - Bayesian TPE optimization
    - Grid search + TPE hybrid approach
    - Automated hyperparameter optimization with cross-validation
    - Performance-based parameter selection
    """
    
    def __init__(self, config: Optional[HDBSCANHyperparameterConfig] = None):
        """Initialize the enhanced hyperparameter optimizer."""
        self.config = config or HDBSCANHyperparameterConfig()
        
        # Initialize Bayesian TPE optimizer
        if self.config.tpe_enabled:
            tpe_config = OptimizationConfig(
                n_trials=self.config.tpe_trials,
                n_jobs=self.config.n_jobs,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient
            )
            self.tpe_optimizer = create_bayesian_tpe_optimizer(tpe_config)
        else:
            self.tpe_optimizer = None
        
        # Define HDBSCAN parameter search space
        self.parameter_search_space = self._define_parameter_search_space()
        
        # Optimization results
        self.optimization_results = {
            'best_params': None,
            'best_score': -np.inf,
            'optimization_history': [],
            'grid_search_results': [],
            'tpe_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }
        
        logger.info("✅ EnhancedHyperparameterOptimizer initialized")
    
    def _define_parameter_search_space(self) -> Dict[str, List]:
        """Define the HDBSCAN parameter search space."""
        return {
            'min_cluster_size': [5, 10, 15, 20, 25, 30, 40, 50],
            'min_samples': [3, 5, 7, 10, 15, 20, 25, 30],
            'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'cluster_selection_method': ['eom', 'leaf'],
            'metric': ['euclidean', 'manhattan', 'cosine', 'l1', 'l2'],
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'cluster_selection_epsilon': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        }
    
    def optimize_hyperparameters(self, features_df: pd.DataFrame, 
                                labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize HDBSCAN hyperparameters using the configured strategy.
        
        Args:
            features_df: Input features DataFrame
            labels: Optional labels for supervised evaluation
            
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        logger.info(f"🚀 Starting hyperparameter optimization for {features_df.shape[0]} samples")
        
        # Validate input
        self._validate_input(features_df)
        
        # Initialize results
        results = {
            'optimization_strategy': self.config.optimization_strategy,
            'n_trials': self.config.n_trials,
            'best_params': None,
            'best_score': -np.inf,
            'optimization_history': [],
            'grid_search_results': [],
            'tpe_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }
        
        # Execute optimization strategy
        if self.config.optimization_strategy == "grid":
            results = self._grid_search_optimization(features_df, labels)
        elif self.config.optimization_strategy == "tpe":
            results = self._tpe_optimization(features_df, labels)
        elif self.config.optimization_strategy == "hybrid":
            results = self._hybrid_optimization(features_df, labels)
        else:
            raise ValueError(f"Unsupported optimization strategy: {self.config.optimization_strategy}")
        
        # Update results
        optimization_time = time.time() - start_time
        results['optimization_time'] = optimization_time
        
        # Store results
        self.optimization_results = results
        
        logger.info(f"✅ Hyperparameter optimization completed: "
                   f"Best score: {results['best_score']:.3f}, "
                   f"Time: {optimization_time:.2f}s")
        
        return results
    
    def _grid_search_optimization(self, features_df: pd.DataFrame, 
                                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform grid search optimization."""
        logger.info("🔄 Performing grid search optimization")
        
        results = {
            'optimization_strategy': 'grid',
            'best_params': None,
            'best_score': -np.inf,
            'grid_search_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()
        
        # Limit combinations for performance
        if len(param_combinations) > self.config.coarse_grid_trials:
            param_combinations = param_combinations[:self.config.coarse_grid_trials]
        
        # Evaluate each combination
        for i, params in enumerate(param_combinations):
            try:
                score = self._evaluate_parameters(features_df, params, labels)
                
                results['grid_search_results'].append({
                    'params': params,
                    'score': score,
                    'trial': i + 1
                })
                
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_params'] = params
                
                results['total_trials'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Grid search trial {i+1} failed: {e}")
                continue
        
        return results
    
    def _tpe_optimization(self, features_df: pd.DataFrame, 
                         labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform TPE optimization."""
        logger.info("🔄 Performing TPE optimization")
        
        if not self.tpe_optimizer:
            raise ValueError("TPE optimizer not available")
        
        # Define objective function
        def objective(trial):
            params = self._suggest_parameters(trial)
            return self._evaluate_parameters(features_df, params, labels)
        
        # Run TPE optimization
        tpe_results = self.tpe_optimizer.optimize(
            objective=objective,
            n_trials=self.config.tpe_trials
        )
        
        results = {
            'optimization_strategy': 'tpe',
            'best_params': tpe_results['best_params'],
            'best_score': tpe_results['best_score'],
            'tpe_results': tpe_results,
            'total_trials': self.config.tpe_trials,
            'optimization_time': tpe_results.get('optimization_time', 0.0)
        }
        
        return results
    
    def _hybrid_optimization(self, features_df: pd.DataFrame, 
                           labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform hybrid optimization (grid search + TPE)."""
        logger.info("🔄 Performing hybrid optimization (grid search + TPE)")
        
        # Phase 1: Coarse grid search
        logger.info("🔄 Phase 1: Coarse grid search")
        coarse_results = self._coarse_grid_search(features_df, labels)
        
        # Phase 2: Fine grid search around best parameters
        logger.info("🔄 Phase 2: Fine grid search")
        fine_results = self._fine_grid_search(features_df, labels, coarse_results['best_params'])
        
        # Phase 3: TPE optimization
        logger.info("🔄 Phase 3: TPE optimization")
        tpe_results = self._tpe_optimization(features_df, labels)
        
        # Combine results
        results = {
            'optimization_strategy': 'hybrid',
            'best_params': tpe_results['best_params'],
            'best_score': tpe_results['best_score'],
            'coarse_grid_results': coarse_results,
            'fine_grid_results': fine_results,
            'tpe_results': tpe_results,
            'total_trials': (coarse_results['total_trials'] + 
                           fine_results['total_trials'] + 
                           tpe_results['total_trials']),
            'optimization_time': (coarse_results['optimization_time'] + 
                                fine_results['optimization_time'] + 
                                tpe_results['optimization_time'])
        }
        
        return results
    
    def _coarse_grid_search(self, features_df: pd.DataFrame, 
                           labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform coarse grid search."""
        # Use a subset of parameters for coarse search
        coarse_params = {
            'min_cluster_size': [10, 20, 30, 40, 50],
            'min_samples': [5, 10, 15, 20, 25],
            'cluster_selection_epsilon': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            'cluster_selection_method': ['eom', 'leaf'],
            'metric': ['euclidean', 'manhattan', 'cosine']
        }
        
        # Generate combinations
        param_combinations = self._generate_parameter_combinations_from_dict(coarse_params)
        
        # Limit combinations
        if len(param_combinations) > self.config.coarse_grid_trials:
            param_combinations = param_combinations[:self.config.coarse_grid_trials]
        
        results = {
            'best_params': None,
            'best_score': -np.inf,
            'grid_search_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            try:
                score = self._evaluate_parameters(features_df, params, labels)
                
                results['grid_search_results'].append({
                    'params': params,
                    'score': score,
                    'trial': i + 1
                })
                
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_params'] = params
                
                results['total_trials'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Coarse grid trial {i+1} failed: {e}")
                continue
        
        results['optimization_time'] = time.time() - start_time
        return results
    
    def _fine_grid_search(self, features_df: pd.DataFrame, 
                         labels: Optional[np.ndarray] = None,
                         best_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform fine grid search around best parameters."""
        if best_params is None:
            return {'best_params': None, 'best_score': -np.inf, 'total_trials': 0}
        
        # Define fine search space around best parameters
        fine_params = {}
        for param, value in best_params.items():
            if param == 'min_cluster_size':
                fine_params[param] = [max(5, value-5), value, min(100, value+5)]
            elif param == 'min_samples':
                fine_params[param] = [max(3, value-3), value, min(50, value+3)]
            elif param == 'cluster_selection_epsilon':
                fine_params[param] = [max(0.0, value-0.1), value, min(1.0, value+0.1)]
            else:
                fine_params[param] = [value]
        
        # Generate combinations
        param_combinations = self._generate_parameter_combinations_from_dict(fine_params)
        
        # Limit combinations
        if len(param_combinations) > self.config.fine_grid_trials:
            param_combinations = param_combinations[:self.config.fine_grid_trials]
        
        results = {
            'best_params': best_params,
            'best_score': -np.inf,
            'grid_search_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }
        
        start_time = time.time()
        
        for i, params in enumerate(param_combinations):
            try:
                score = self._evaluate_parameters(features_df, params, labels)
                
                results['grid_search_results'].append({
                    'params': params,
                    'score': score,
                    'trial': i + 1
                })
                
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_params'] = params
                
                results['total_trials'] += 1
                
            except Exception as e:
                logger.warning(f"⚠️ Fine grid trial {i+1} failed: {e}")
                continue
        
        results['optimization_time'] = time.time() - start_time
        return results
    
    def _evaluate_parameters(self, features_df: pd.DataFrame, 
                           params: Dict[str, Any], 
                           labels: Optional[np.ndarray] = None) -> float:
        """Evaluate HDBSCAN parameters."""
        try:
            # Create HDBSCAN clusterer
            clusterer = hdbscan.HDBSCAN(**params)
            
            # Fit and predict
            cluster_labels = clusterer.fit_predict(features_df)
            
            # Calculate evaluation metric
            if self.config.primary_metric == "silhouette":
                score = self._calculate_silhouette_score(features_df, cluster_labels)
            elif self.config.primary_metric == "calinski_harabasz":
                score = self._calculate_calinski_harabasz_score(features_df, cluster_labels)
            elif self.config.primary_metric == "davies_bouldin":
                score = self._calculate_davies_bouldin_score(features_df, cluster_labels)
            else:
                score = self._calculate_silhouette_score(features_df, cluster_labels)
            
            return score
            
        except Exception as e:
            logger.debug(f"Parameter evaluation failed: {e}")
            return -np.inf
    
    def _calculate_silhouette_score(self, features_df: pd.DataFrame, 
                                   cluster_labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -1.0
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -1.0
            
            return silhouette_score(valid_features, valid_labels)
        except:
            return -1.0
    
    def _calculate_calinski_harabasz_score(self, features_df: pd.DataFrame, 
                                         cluster_labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return 0.0
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return 0.0
            
            return calinski_harabasz_score(valid_features, valid_labels)
        except:
            return 0.0
    
    def _calculate_davies_bouldin_score(self, features_df: pd.DataFrame, 
                                      cluster_labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score (lower is better)."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return np.inf
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return np.inf
            
            # Davies-Bouldin score (lower is better, so negate)
            return -davies_bouldin_score(valid_features, valid_labels)
        except:
            return np.inf
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations from search space."""
        import itertools
        
        # Get parameter names and values
        param_names = list(self.parameter_search_space.keys())
        param_values = list(self.parameter_search_space.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to dictionaries
        param_combinations = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _generate_parameter_combinations_from_dict(self, param_dict: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate parameter combinations from a dictionary."""
        import itertools
        
        # Get parameter names and values
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to dictionaries
        param_combinations = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _suggest_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters for TPE optimization."""
        params = {}
        
        # Suggest parameters based on search space
        for param_name, param_values in self.parameter_search_space.items():
            if param_name == 'min_cluster_size':
                params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
            elif param_name == 'min_samples':
                params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
            elif param_name == 'cluster_selection_epsilon':
                params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
            elif param_name == 'cluster_selection_method':
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif param_name == 'metric':
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif param_name == 'alpha':
                params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
        
        return params
    
    def _validate_input(self, features_df: pd.DataFrame):
        """Validate input data."""
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("Features must be a pandas DataFrame")
        
        if features_df.empty:
            raise ValueError("Features DataFrame cannot be empty")
        
        if features_df.shape[0] < 10:
            raise ValueError("Not enough samples for hyperparameter optimization")
    
    def get_optimization_results(self) -> Dict[str, Any]:
        """Get optimization results."""
        return self.optimization_results.copy()
    
    def reset_optimization(self):
        """Reset optimization results."""
        self.optimization_results = {
            'best_params': None,
            'best_score': -np.inf,
            'optimization_history': [],
            'grid_search_results': [],
            'tpe_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }

# Convenience function
def create_enhanced_hyperparameter_optimizer(
    optimization_strategy: str = "hybrid",
    n_trials: int = 50,
    primary_metric: str = "silhouette",
    enable_parallel: bool = True,
    memory_efficient: bool = True
) -> EnhancedHyperparameterOptimizer:
    """
    Create an enhanced hyperparameter optimizer with specified configuration.
    
    Args:
        optimization_strategy: Optimization strategy ("grid", "tpe", "hybrid")
        n_trials: Number of optimization trials
        primary_metric: Primary evaluation metric
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        
    Returns:
        EnhancedHyperparameterOptimizer instance
    """
    config = HDBSCANHyperparameterConfig(
        optimization_strategy=optimization_strategy,
        n_trials=n_trials,
        primary_metric=primary_metric,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient
    )
    
    return EnhancedHyperparameterOptimizer(config)
