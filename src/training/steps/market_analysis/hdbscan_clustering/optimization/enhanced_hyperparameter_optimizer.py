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
    OptimizationConfig
)

# Import tprint for enhanced logging
from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning, tprint

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

    # Execution mode for adaptive configuration
    execution_mode: str = "light"  # "full", "light", "blank"

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
    
    # Advanced optimization features
    enable_convergence_detection: bool = False
    convergence_threshold: float = 0.001
    adaptive_trials: bool = True
    smart_sampling: bool = True
    
    # Regime count optimization
    target_regime_count_min: int = 4  # Minimum desired regimes
    target_regime_count_max: int = 8  # Maximum desired regimes
    regime_count_penalty: float = 0.2  # Penalty weight for deviating from target range
    enable_regime_count_objective: bool = True  # Enable regime count objective
    
    # Trial tracking
    _current_trial: int = 0

    def __post_init__(self):
        """Apply execution mode-based optimizations."""
        # Initialize trial counter
        self._current_trial = 0
        
        # Store original n_trials if explicitly set
        original_n_trials = self.n_trials
        
        if self.execution_mode == "light":
            # Light mode: balanced trials for good quality and speed
            self.n_trials = 20  # Good balance for light mode
            self.coarse_grid_trials = 8
            self.fine_grid_trials = 7
            self.tpe_trials = 20
            self.early_stopping_patience = 3
            self.cv_folds = 3
            # Ensure n_startup_trials is less than n_trials
            self.n_startup_trials = 5
        elif self.execution_mode == "blank":
            # Blank mode: minimal trials
            self.n_trials = 1
            self.coarse_grid_trials = 1
            self.fine_grid_trials = 0
            self.tpe_trials = 1
            self.early_stopping_patience = 1
            self.cv_folds = 2
            # Ensure n_startup_trials is less than n_trials
            self.n_startup_trials = 0
        elif self.execution_mode == "full":
            # Full mode: adaptive trials based on data complexity
            # Use early stopping and convergence detection
            self.n_trials = 100  # Increased from default
            self.coarse_grid_trials = 25
            self.fine_grid_trials = 50
            self.tpe_trials = 75
            self.early_stopping_patience = 15  # More patience for full mode
            self.cv_folds = 5
            # Enable convergence detection
            self.enable_convergence_detection = True
            self.convergence_threshold = 0.001
        
        # If n_trials was explicitly set (not default), use that value
        if original_n_trials != 50:  # 50 is the default value
            self.n_trials = original_n_trials

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
        if self.config.tpe_enabled and self.config.execution_mode != "light":
            tpe_config = OptimizationConfig(
                n_trials=self.config.tpe_trials,
                n_startup_trials=self.config.n_startup_trials,  # Use config value
                execution_mode=self.config.execution_mode,
                vectorbt_parallel_workers=self.config.n_jobs,
                enable_vectorbt_optimization=self.config.enable_parallel,
                vectorbt_memory_efficient=self.config.memory_efficient
            )
            self.tpe_optimizer = BayesianTPEOptimizer(tpe_config)
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
        
        # Trial counter for detailed logging
        self._trial_counter = 0
        
        tprint_info("✅ EnhancedHyperparameterOptimizer initialized")
    
    def _define_parameter_search_space(self) -> Dict[str, List]:
        """Define the HDBSCAN parameter search space."""
        # Adaptive search space based on execution mode
        if self.config.execution_mode == "light":
            return {
                'min_cluster_size': [5, 10, 15],
                'min_samples': [3, 5, 7],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2],
                'cluster_selection_method': ['eom'],
                'metric': ['euclidean'],
                'alpha': [1.0]
            }
        elif self.config.execution_mode == "blank":
            return {
                'min_cluster_size': [10],
                'min_samples': [5],
                'cluster_selection_epsilon': [0.0],
                'cluster_selection_method': ['eom'],
                'metric': ['euclidean'],
                'alpha': [1.0]
            }
        else:  # full mode - smart parameter space
            return {
                'min_cluster_size': [5, 8, 12, 15, 20, 25, 30, 40],
                'min_samples': [3, 5, 8, 10, 12, 15, 20],
                'cluster_selection_epsilon': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                'cluster_selection_method': ['eom', 'leaf'],
                'metric': ['euclidean', 'manhattan'],
                'alpha': [0.5, 1.0, 1.5]
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
        tprint_info(f"🚀 Starting hyperparameter optimization for {features_df.shape[0]} samples")
        
        # Initialize convergence tracking
        if self.config.enable_convergence_detection:
            self._convergence_scores = []
            self._convergence_threshold = self.config.convergence_threshold
            tprint(f"🎯 Convergence detection enabled (threshold: {self._convergence_threshold})", "INFO")
        
        # Validate input
        tprint("🔍 Validating input data...", "INFO")
        self._validate_input(features_df)
        tprint("✅ Input validation completed", "SUCCESS")
        
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
        
        tprint_success(f"✅ Hyperparameter optimization completed: "
                      f"Best score: {results['best_score']:.3f}, "
                      f"Time: {optimization_time:.2f}s")
        
        return results
    
    def _grid_search_optimization(self, features_df: pd.DataFrame,
                                 labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform grid search optimization."""
        start_time = time.time()
        tprint_info("🔄 Performing grid search optimization")

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
        tprint_info(f"📊 Generated {len(param_combinations)} parameter combinations")

        # Limit combinations for performance
        if len(param_combinations) > self.config.coarse_grid_trials:
            param_combinations = param_combinations[:self.config.coarse_grid_trials]
            tprint_info(f"🔢 Limited to {len(param_combinations)} trials for performance")

        # Evaluate each combination
        phase_start_time = time.time()
        best_score_so_far = -np.inf

        for i, params in enumerate(param_combinations):
            trial_start_time = time.time()

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
                    tprint_success(f"🎯 New best score: {score:.4f} at trial {i+1}/{len(param_combinations)} "
                                 f"(Params: min_cluster_size={params.get('min_cluster_size', 'N/A')}, "
                                 f"min_samples={params.get('min_samples', 'N/A')}, "
                                 f"epsilon={params.get('cluster_selection_epsilon', 'N/A')})")

                if score > best_score_so_far:
                    best_score_so_far = score

                results['total_trials'] += 1

                # Progress logging every 10% or at key milestones
                progress = (i + 1) / len(param_combinations)
                if (i + 1) % max(1, len(param_combinations) // 10) == 0 or i == 0 or i == len(param_combinations) - 1:
                    elapsed_time = time.time() - phase_start_time
                    trial_time = time.time() - trial_start_time
                    eta = elapsed_time / (i + 1) * (len(param_combinations) - i - 1) if i > 0 else 0

                    tprint_info(f"📈 Grid search progress: {progress*100:.1f}% "
                               f"({i+1}/{len(param_combinations)}) | "
                               f"Current: {score:.4f} | Best: {results['best_score']:.4f} | "
                               f"Trial time: {trial_time:.2f}s | ETA: {eta:.1f}s")

            except Exception as e:
                tprint_warning(f"⚠️ Grid search trial {i+1} failed: {e}")
                continue

        results['optimization_time'] = time.time() - start_time
        tprint_success(f"✅ Grid search completed in {results['optimization_time']:.2f}s | "
                      f"Best score: {results['best_score']:.4f} | Total trials: {results['total_trials']}")

        return results
    
    def _tpe_optimization(self, features_df: pd.DataFrame,
                         labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform TPE optimization."""
        tprint_info("🔄 Performing TPE optimization")
        
        if not self.tpe_optimizer:
            raise ValueError("TPE optimizer not available")
        
        # Define objective function that works with BayesianTPEOptimizer
        def objective(params):
            return self._evaluate_parameters(features_df, params, labels)
        
        # Convert search space to BayesianTPEOptimizer format
        tpe_search_space = {}
        for param_name, param_values in self.parameter_search_space.items():
            if param_name in ['min_cluster_size', 'min_samples']:
                # Integer parameters: convert list to (min, max) range
                tpe_search_space[param_name] = (int(min(param_values)), int(max(param_values)))
            elif param_name in ['cluster_selection_epsilon', 'alpha']:
                # Float parameters: convert list to (min, max) range
                tpe_search_space[param_name] = (float(min(param_values)), float(max(param_values)))
            else:
                # Categorical parameters: keep as list
                tpe_search_space[param_name] = param_values

        # Run TPE optimization
        tpe_results = self.tpe_optimizer.optimize(
            objective=objective,
            search_space=tpe_search_space,
            n_trials=self.config.tpe_trials
        )
        
        results = {
            'optimization_strategy': 'tpe',
            'best_params': tpe_results['best_params'],
            'best_score': tpe_results.get('best_value', float('-inf')),
            'tpe_results': tpe_results,
            'total_trials': self.config.tpe_trials,
            'optimization_time': tpe_results.get('optimization_time', 0.0)
        }
        
        return results
    
    def _hybrid_optimization(self, features_df: pd.DataFrame,
                           labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform hybrid optimization (grid search + TPE)."""
        tprint_info("🔄 Performing hybrid optimization (grid search + TPE)")

        # Phase 1: Coarse grid search
        tprint_info("🔄 Phase 1: Coarse grid search")
        coarse_results = self._coarse_grid_search(features_df, labels)
        
        # Phase 2: Fine grid search around best parameters
        tprint_info("🔄 Phase 2: Fine grid search")
        fine_results = self._fine_grid_search(features_df, labels, coarse_results['best_params'])

        # Phase 3: TPE optimization (skip for light mode)
        if self.config.execution_mode == "light":
            tprint_info("🔄 Phase 3: TPE optimization (skipped for light mode)")
            tpe_results = {
                'best_params': fine_results['best_params'],
                'best_score': fine_results['best_score'],
                'total_trials': 0,
                'optimization_time': 0.0
            }
        else:
            tprint_info("🔄 Phase 3: TPE optimization")
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
        start_time = time.time()
        tprint_info("🔍 Starting coarse grid search phase")

        # Use adaptive search space based on execution mode
        if self.config.execution_mode == "light":
            # Light mode: balanced search space
            coarse_params = {
                'min_cluster_size': [10, 15, 20],
                'min_samples': [5, 10, 15],
                'cluster_selection_epsilon': [0.0, 0.2, 0.4],
                'cluster_selection_method': ['eom'],
                'metric': ['euclidean'],
                'alpha': [0.5, 1.0, 1.5]
            }
        elif self.config.execution_mode == "blank":
            # Blank mode: single trial
            coarse_params = {
                'min_cluster_size': [10],
                'min_samples': [5],
                'cluster_selection_epsilon': [0.0],
                'cluster_selection_method': ['eom'],
                'metric': ['euclidean'],
                'alpha': [1.0]
            }
        else:
            # Full mode: comprehensive search space
            coarse_params = {
                'min_cluster_size': [10, 20, 30, 40, 50],
                'min_samples': [5, 10, 15, 20, 25],
                'cluster_selection_epsilon': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'cluster_selection_method': ['eom', 'leaf'],
                'metric': ['euclidean', 'manhattan'],
                'alpha': [0.5, 1.0, 1.5]
            }

        tprint_info(f"🎛️ Coarse search space: {coarse_params}")

        # Generate combinations
        param_combinations = self._generate_parameter_combinations_from_dict(coarse_params)
        tprint_info(f"📊 Generated {len(param_combinations)} coarse parameter combinations")

        # Limit combinations
        if len(param_combinations) > self.config.coarse_grid_trials:
            param_combinations = param_combinations[:self.config.coarse_grid_trials]
            tprint_info(f"🔢 Limited coarse search to {len(param_combinations)} trials")

        results = {
            'best_params': None,
            'best_score': -np.inf,
            'grid_search_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }

        phase_start_time = time.time()
        best_score_so_far = -np.inf

        for i, params in enumerate(param_combinations):
            trial_start_time = time.time()

            try:
                score = self._evaluate_parameters(features_df, params, labels)

                results['grid_search_results'].append({
                    'params': params,
                    'score': score,
                    'trial': i + 1
                })

                # Increment trial counter
                self._trial_counter += 1
                
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_params'] = params
                    tprint_success(f"🎯 Coarse grid new best: {score:.4f} at trial {self._trial_counter} "
                                 f"(min_cluster_size={params.get('min_cluster_size')}, "
                                 f"min_samples={params.get('min_samples')}, "
                                 f"epsilon={params.get('cluster_selection_epsilon')})")

                if score > best_score_so_far:
                    best_score_so_far = score

                results['total_trials'] += 1

                # Progress logging every 20% or at key milestones
                progress = (i + 1) / len(param_combinations)
                if (i + 1) % max(1, len(param_combinations) // 5) == 0 or i == 0 or i == len(param_combinations) - 1:
                    elapsed_time = time.time() - phase_start_time
                    trial_time = time.time() - trial_start_time
                    eta = elapsed_time / (i + 1) * (len(param_combinations) - i - 1) if i > 0 else 0

                    tprint_info(f"📈 Coarse grid progress: {progress*100:.1f}% "
                               f"({i+1}/{len(param_combinations)}) | "
                               f"Current: {score:.4f} | Best: {results['best_score']:.4f} | "
                               f"Trial time: {trial_time:.2f}s | ETA: {eta:.1f}s")

            except Exception as e:
                tprint_warning(f"⚠️ Coarse grid trial {i+1} failed: {e}")
                continue

        results['optimization_time'] = time.time() - start_time
        tprint_success(f"✅ Coarse grid search completed in {results['optimization_time']:.2f}s | "
                      f"Best score: {results['best_score']:.4f} | Total trials: {results['total_trials']}")

        return results
    
    def _fine_grid_search(self, features_df: pd.DataFrame,
                         labels: Optional[np.ndarray] = None,
                         best_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform fine grid search around best parameters."""
        start_time = time.time()

        if best_params is None:
            tprint_warning("⚠️ No best parameters provided for fine grid search")
            return {'best_params': None, 'best_score': -np.inf, 'total_trials': 0, 'optimization_time': 0.0}

        tprint_info(f"🎯 Starting fine grid search around best parameters: {best_params}")

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

        tprint_info(f"🔬 Fine search space defined: {fine_params}")

        # Generate combinations
        param_combinations = self._generate_parameter_combinations_from_dict(fine_params)
        tprint_info(f"📊 Generated {len(param_combinations)} fine parameter combinations")

        # Limit combinations
        if len(param_combinations) > self.config.fine_grid_trials:
            param_combinations = param_combinations[:self.config.fine_grid_trials]
            tprint_info(f"🔢 Limited fine search to {len(param_combinations)} trials")

        results = {
            'best_params': best_params,
            'best_score': -np.inf,
            'grid_search_results': [],
            'total_trials': 0,
            'optimization_time': 0.0
        }

        phase_start_time = time.time()
        best_score_so_far = -np.inf

        for i, params in enumerate(param_combinations):
            trial_start_time = time.time()

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
                    improvement = score - best_score_so_far if best_score_so_far != -np.inf else score
                    tprint_success(f"🎯 Fine grid new best: {score:.4f} at trial {i+1}/{len(param_combinations)} "
                                 f"(improvement: {improvement:.4f}) | "
                                 f"Params: min_cluster_size={params.get('min_cluster_size')}, "
                                 f"min_samples={params.get('min_samples')}, "
                                 f"epsilon={params.get('cluster_selection_epsilon')}")

                if score > best_score_so_far:
                    best_score_so_far = score

                results['total_trials'] += 1

                # Progress logging every 25% or at key milestones
                progress = (i + 1) / len(param_combinations)
                if (i + 1) % max(1, len(param_combinations) // 4) == 0 or i == 0 or i == len(param_combinations) - 1:
                    elapsed_time = time.time() - phase_start_time
                    trial_time = time.time() - trial_start_time
                    eta = elapsed_time / (i + 1) * (len(param_combinations) - i - 1) if i > 0 else 0

                    logger.info(f"📈 Fine grid progress: {progress*100:.1f}% "
                              f"({i+1}/{len(param_combinations)}) | "
                              f"Current: {score:.4f} | Best: {results['best_score']:.4f} | "
                              f"Trial time: {trial_time:.2f}s | ETA: {eta:.1f}s")

            except Exception as e:
                logger.warning(f"⚠️ Fine grid trial {i+1} failed: {e}")
                continue

        results['optimization_time'] = time.time() - start_time
        logger.info(f"✅ Fine grid search completed in {results['optimization_time']:.2f}s | "
                   f"Best score: {results['best_score']:.4f} | Total trials: {results['total_trials']}")

        return results
    
    def _evaluate_parameters(self, features_df: pd.DataFrame,
                            params: Dict[str, Any],
                            labels: Optional[np.ndarray] = None) -> float:
        """Evaluate HDBSCAN parameters."""
        try:
            # Trial counter removed to avoid attribute errors
            # Create HDBSCAN clusterer with proper parameter types
            # Ensure integer parameters and proper epsilon values
            
            # Helper function to extract single value from list if needed
            def extract_value(value):
                if isinstance(value, list):
                    return value[0] if value else 'eom'
                return value
            
            hdbscan_params = {
                'min_cluster_size': int(params['min_cluster_size']),
                'min_samples': int(params['min_samples']),
                'cluster_selection_epsilon': round(float(params['cluster_selection_epsilon']), 3),
                'cluster_selection_method': extract_value(params.get('cluster_selection_method', 'eom')),
                'metric': extract_value(params.get('metric', 'euclidean')),
                'alpha': float(params['alpha'])
            }
            
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)

            # Ensure only numeric data is passed to HDBSCAN
            numeric_features_df = features_df.select_dtypes(include=[np.number])
            if len(numeric_features_df.columns) < len(features_df.columns):
                logger.warning(f"⚠️ Filtered out {len(features_df.columns) - len(numeric_features_df.columns)} non-numeric columns for HDBSCAN")
            
            # Convert to float64 and handle any remaining data type issues
            numeric_features_df = numeric_features_df.astype(np.float64)
            
            # Remove any infinite or NaN values
            numeric_features_df = numeric_features_df.replace([np.inf, -np.inf], np.nan)
            numeric_features_df = numeric_features_df.fillna(0)
            
            # Ensure all values are finite
            if not np.all(np.isfinite(numeric_features_df.values)):
                logger.error("❌ Non-finite values found in features after cleaning")
                return float('inf')
            
            # Fit and predict
            cluster_labels = clusterer.fit_predict(numeric_features_df)

            # Debug: Log clustering results
            n_noise = (cluster_labels == -1).sum()
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            logger.debug(f"HDBSCAN results: {len(cluster_labels)} samples, {n_noise} noise points, {n_clusters} clusters")
            tprint(f"🔍 Clustering: {n_clusters} clusters, {n_noise} noise points", "INFO")

            # Check if we have enough valid clusters
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                logger.debug(f"Insufficient valid samples ({valid_mask.sum()}) for evaluation")
                return -np.inf

            valid_labels = cluster_labels[valid_mask]
            if len(set(valid_labels)) < 2:
                logger.debug(f"Insufficient clusters ({len(set(valid_labels))}) for evaluation")
                return -np.inf

            # For light mode, use a simple fast scoring approach that differentiates
            if self.config.execution_mode == "light":
                score = self._calculate_fast_light_score(cluster_labels, params)
                tprint(f"⚡ Fast light score: {score:.4f}", "INFO")
            else:
                # Calculate evaluation metric using numeric features
                if self.config.primary_metric == "silhouette":
                    score = self._calculate_silhouette_score(numeric_features_df, cluster_labels)
                elif self.config.primary_metric == "calinski_harabasz":
                    score = self._calculate_calinski_harabasz_score(numeric_features_df, cluster_labels)
                elif self.config.primary_metric == "davies_bouldin":
                    score = self._calculate_davies_bouldin_score(numeric_features_df, cluster_labels)
                else:
                    score = self._calculate_silhouette_score(numeric_features_df, cluster_labels)

            # Increment trial counter
            self._trial_counter += 1
            
            # Format parameters for display
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            tprint(f"📊 Trial {self._trial_counter} completed: {param_str} -> score: {score:.4f}")
            
            # Check for convergence
            if self._check_convergence(score):
                tprint("🛑 Early stopping due to convergence", "SUCCESS")
                
            return score

        except Exception as e:
            tprint(f"DEBUG: Parameter evaluation failed: {e} | params: {params}")
            return -np.inf
    
    def _apply_regime_count_penalty(self, base_score: float, cluster_labels: np.ndarray) -> float:
        """
        Apply penalty for deviating from target regime count range.
        
        Args:
            base_score: Base quality score (silhouette, CH, DBI, etc.)
            cluster_labels: Cluster assignments
            
        Returns:
            Adjusted score with regime count penalty
        """
        if not self.config.enable_regime_count_objective:
            return base_score
        
        # Count actual regimes (excluding noise)
        n_regimes = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        # Calculate penalty for deviating from target range [4, 8]
        target_min = self.config.target_regime_count_min
        target_max = self.config.target_regime_count_max
        
        if target_min <= n_regimes <= target_max:
            # Within target range - no penalty
            penalty = 0.0
        elif n_regimes < target_min:
            # Too few regimes - penalty proportional to deviation
            penalty = self.config.regime_count_penalty * (target_min - n_regimes) / target_min
        else:
            # Too many regimes - penalty proportional to deviation
            penalty = self.config.regime_count_penalty * (n_regimes - target_max) / target_max
        
        adjusted_score = base_score - penalty
        
        tprint(f"🎯 Regime count: {n_regimes} (target: {target_min}-{target_max}) | "
               f"Base: {base_score:.4f} | Penalty: {penalty:.4f} | Adjusted: {adjusted_score:.4f}", "INFO")
        
        return adjusted_score
    
    def _calculate_silhouette_score(self, features_df: pd.DataFrame, 
                                   cluster_labels: np.ndarray) -> float:
        """Calculate silhouette score with regime count objective."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                tprint(f"⚠️ Insufficient valid samples: {valid_mask.sum()}", "WARNING")
                return -1.0
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                tprint(f"⚠️ Insufficient clusters: {len(set(valid_labels))}", "WARNING")
                return -1.0
            
            # Ensure we have numeric data
            valid_features = valid_features.select_dtypes(include=[np.number])
            if valid_features.empty:
                tprint("⚠️ No numeric features for silhouette calculation", "WARNING")
                return -1.0
            
            base_score = silhouette_score(valid_features, valid_labels)
            
            # Apply regime count penalty
            adjusted_score = self._apply_regime_count_penalty(base_score, cluster_labels)
            
            tprint(f"📊 Silhouette score: {adjusted_score:.4f} (base: {base_score:.4f}, "
                   f"clusters: {len(set(valid_labels))}, samples: {len(valid_features)})", "INFO")
            return adjusted_score
        except Exception as e:
            tprint(f"❌ Silhouette calculation failed: {e}", "ERROR")
            return -1.0
    
    def _calculate_simple_cluster_score(self, cluster_labels: np.ndarray) -> float:
        """Calculate a simple cluster quality score based on cluster distribution."""
        try:
            # Count clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = (cluster_labels == -1).sum()
            n_total = len(cluster_labels)
            
            # Calculate cluster balance (prefer balanced clusters)
            if n_clusters == 0:
                return 0.0
            
            # Get cluster sizes
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            if not cluster_sizes:
                return 0.0
            
            # Calculate balance score (lower variance is better)
            mean_size = np.mean(cluster_sizes)
            if mean_size == 0:
                return 0.0
            
            variance = np.var(cluster_sizes) / (mean_size ** 2) if mean_size > 0 else 1.0
            balance_score = 1.0 / (1.0 + variance)
            
            # Calculate noise ratio (lower is better)
            noise_ratio = n_noise / n_total if n_total > 0 else 1.0
            noise_score = 1.0 - noise_ratio
            
            # Combine scores
            score = (balance_score * 0.7 + noise_score * 0.3)
            
            tprint(f"📊 Simple score: {score:.4f} (clusters: {n_clusters}, noise: {n_noise}/{n_total}, balance: {balance_score:.3f})", "INFO")
            return score
            
        except Exception as e:
            tprint(f"❌ Simple cluster score calculation failed: {e}", "ERROR")
            return 0.0
    
    def _calculate_alternative_score(self, features_df: pd.DataFrame, cluster_labels: np.ndarray, params: Dict[str, Any]) -> float:
        """Calculate an alternative score that differentiates between parameter combinations."""
        try:
            # Count clusters and noise points
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = (cluster_labels == -1).sum()
            n_total = len(cluster_labels)
            
            if n_clusters == 0:
                return 0.0
            
            # Get cluster sizes
            cluster_sizes = [np.sum(cluster_labels == i) for i in range(n_clusters)]
            if not cluster_sizes:
                return 0.0
            
            # Calculate various quality metrics
            noise_ratio = n_noise / n_total if n_total > 0 else 1.0
            
            # Cluster balance (prefer balanced clusters)
            mean_size = np.mean(cluster_sizes)
            variance = np.var(cluster_sizes) / (mean_size ** 2) if mean_size > 0 else 1.0
            balance_score = 1.0 / (1.0 + variance)
            
            # Cluster count penalty (prefer reasonable number of clusters)
            optimal_clusters = min(10, max(3, n_total // 100))  # Reasonable cluster count
            cluster_count_score = 1.0 - abs(n_clusters - optimal_clusters) / optimal_clusters
            cluster_count_score = max(0.0, cluster_count_score)
            
            # Parameter-based scoring (prefer certain parameter ranges)
            min_cluster_size = params.get('min_cluster_size', 5)  # Reduced from 10 to target 5-8 clusters
            min_samples = params.get('min_samples', 5)
            epsilon = params.get('cluster_selection_epsilon', 0.0)
            
            # Prefer smaller min_cluster_size for more clusters
            size_score = 1.0 - (min_cluster_size - 5) / 45  # 5-50 range
            size_score = max(0.0, min(1.0, size_score))
            
            # Prefer smaller min_samples for more flexibility
            samples_score = 1.0 - (min_samples - 3) / 27  # 3-30 range
            samples_score = max(0.0, min(1.0, samples_score))
            
            # Prefer moderate epsilon values
            epsilon_score = 1.0 - abs(epsilon - 0.1) / 0.4  # Prefer around 0.1
            epsilon_score = max(0.0, min(1.0, epsilon_score))
            
            # Combine all scores with weights
            score = (
                balance_score * 0.3 +
                (1.0 - noise_ratio) * 0.2 +
                cluster_count_score * 0.2 +
                size_score * 0.1 +
                samples_score * 0.1 +
                epsilon_score * 0.1
            )
            
            tprint(f"📊 Alt score: {score:.4f} (clusters: {n_clusters}, noise: {noise_ratio:.3f}, balance: {balance_score:.3f}, count: {cluster_count_score:.3f})", "INFO")
            return score
            
        except Exception as e:
            tprint(f"❌ Alternative score calculation failed: {e}", "ERROR")
            return 0.0
    
    def _calculate_fast_light_score(self, cluster_labels: np.ndarray, params: Dict[str, Any]) -> float:
        """Calculate a very fast score for light mode optimization."""
        try:
            # Count clusters and noise
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = (cluster_labels == -1).sum()
            n_total = len(cluster_labels)
            
            if n_clusters == 0:
                return 0.0
            
            # Simple scoring based on cluster count and noise ratio
            noise_ratio = n_noise / n_total if n_total > 0 else 1.0
            
            # Prefer 4-8 clusters (updated target range)
            target_min = self.config.target_regime_count_min
            target_max = self.config.target_regime_count_max
            
            if target_min <= n_clusters <= target_max:
                cluster_score = 1.0
            elif n_clusters < target_min:
                cluster_score = n_clusters / target_min
            else:
                cluster_score = max(0.0, 1.0 - (n_clusters - target_max) / 10.0)
            
            # Prefer lower noise ratio
            noise_score = 1.0 - noise_ratio
            
            # Enhanced parameter-based scoring for better differentiation
            min_cluster_size = params.get('min_cluster_size', 5)  # Reduced from 10 to target 5-8 clusters
            min_samples = params.get('min_samples', 5)
            epsilon = params.get('cluster_selection_epsilon', 0.0)
            
            # Size preference (prefer moderate cluster sizes)
            if 5 <= min_cluster_size <= 15:
                size_score = 1.0
            elif min_cluster_size < 5:
                size_score = min_cluster_size / 5.0
            else:
                size_score = max(0.0, 1.0 - (min_cluster_size - 15) / 20.0)
            
            # Samples preference (prefer balanced min_samples)
            if 3 <= min_samples <= 10:
                samples_score = 1.0
            elif min_samples < 3:
                samples_score = min_samples / 3.0
            else:
                samples_score = max(0.0, 1.0 - (min_samples - 10) / 15.0)
            
            # Epsilon preference (prefer moderate epsilon)
            if 0.0 <= epsilon <= 0.2:
                epsilon_score = 1.0
            else:
                epsilon_score = max(0.0, 1.0 - abs(epsilon - 0.1) / 0.3)
            
            # Add parameter diversity bonus
            param_diversity = abs(min_cluster_size - min_samples) / 20.0
            diversity_score = min(1.0, param_diversity)
            
            # Combine scores with more weight on parameter differences
            score = (
                cluster_score * 0.3 + 
                noise_score * 0.2 + 
                size_score * 0.2 + 
                samples_score * 0.15 + 
                epsilon_score * 0.1 + 
                diversity_score * 0.05
            )
            
            tprint(f"⚡ Fast score: {score:.4f} (clusters: {n_clusters}, noise: {noise_ratio:.3f})", "INFO")
            return score

        except Exception as e:
            tprint(f"❌ Fast light score calculation failed: {e}", "ERROR")
            return 0.0

    def _check_convergence(self, score: float) -> bool:
        """Check if optimization has converged based on recent scores."""
        if not self.config.enable_convergence_detection:
            return False
            
        self._convergence_scores.append(score)
        
        # Need at least 10 scores to check convergence
        if len(self._convergence_scores) < 10:
            return False
            
        # Check if recent scores are stable (low variance)
        recent_scores = self._convergence_scores[-10:]
        score_variance = np.var(recent_scores)
        
        if score_variance < self._convergence_threshold:
            tprint(f"🎯 Convergence detected! Score variance: {score_variance:.6f} < {self._convergence_threshold}", "SUCCESS")
            return True
            
        return False
    
    def _calculate_calinski_harabasz_score(self, features_df: pd.DataFrame, 
                                         cluster_labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score with regime count objective."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return 0.0
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return 0.0
            
            # Ensure numeric features
            valid_features = valid_features.select_dtypes(include=[np.number])
            if valid_features.empty:
                return 0.0
            
            base_score = calinski_harabasz_score(valid_features, valid_labels)
            
            # Normalize CH score to [0, 1] range for penalty application
            # CH scores can be large, so use log scaling
            normalized_score = np.log1p(base_score) / 10.0  # Roughly [0, 1]
            
            # Apply regime count penalty
            adjusted_score = self._apply_regime_count_penalty(normalized_score, cluster_labels)
            
            # Scale back to CH range
            final_score = np.expm1(adjusted_score * 10.0)
            
            tprint(f"📊 Calinski-Harabasz score: {final_score:.4f} (base: {base_score:.4f})", "INFO")
            return final_score
        except Exception as e:
            tprint(f"❌ Calinski-Harabasz calculation failed: {e}", "ERROR")
            return 0.0
    
    def _calculate_davies_bouldin_score(self, features_df: pd.DataFrame, 
                                      cluster_labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score with regime count objective (lower is better)."""
        try:
            # Remove noise points for evaluation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return np.inf
            
            valid_features = features_df[valid_mask]
            valid_labels = cluster_labels[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return np.inf
            
            # Ensure numeric features
            valid_features = valid_features.select_dtypes(include=[np.number])
            if valid_features.empty:
                return np.inf
            
            base_db_score = davies_bouldin_score(valid_features, valid_labels)
            
            # Davies-Bouldin score (lower is better, so negate for maximization)
            negated_score = -base_db_score
            
            # Apply regime count penalty (on negated score)
            adjusted_score = self._apply_regime_count_penalty(negated_score, cluster_labels)
            
            tprint(f"📊 Davies-Bouldin score: {adjusted_score:.4f} (base: {negated_score:.4f})", "INFO")
            return adjusted_score
        except Exception as e:
            tprint(f"❌ Davies-Bouldin calculation failed: {e}", "ERROR")
            return -np.inf
    
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
        memory_efficient: bool = True,
        execution_mode: str = "light",
        target_regime_count_min: int = 4,
        target_regime_count_max: int = 8,
        regime_count_penalty: float = 0.2,
        enable_regime_count_objective: bool = True
) -> EnhancedHyperparameterOptimizer:
    """
    Create an enhanced hyperparameter optimizer with specified configuration.

    Args:
        optimization_strategy: Optimization strategy ("grid", "tpe", "hybrid")
        n_trials: Number of optimization trials
        primary_metric: Primary evaluation metric
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        execution_mode: Execution mode ("full", "light", "blank") for adaptive configuration
        target_regime_count_min: Minimum desired number of regimes
        target_regime_count_max: Maximum desired number of regimes
        regime_count_penalty: Penalty weight for deviating from target regime count range
        enable_regime_count_objective: Whether to enable regime count objective

    Returns:
        EnhancedHyperparameterOptimizer instance
    """
    config = HDBSCANHyperparameterConfig(
        optimization_strategy=optimization_strategy,
        n_trials=n_trials,
        primary_metric=primary_metric,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient,
        execution_mode=execution_mode,
        target_regime_count_min=target_regime_count_min,
        target_regime_count_max=target_regime_count_max,
        regime_count_penalty=regime_count_penalty,
        enable_regime_count_objective=enable_regime_count_objective
    )

    return EnhancedHyperparameterOptimizer(config)
