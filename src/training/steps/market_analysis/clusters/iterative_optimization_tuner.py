"""
Iterative Optimization Hyperparameter Tuner

This script uses multi-objective optimization to tune the hyperparameters of iterative_optimization.py
to maximize CV, Silhouette, and DBI while maintaining Balance and Temporal Smoothness.

Objectives:
- Maximize CV (Between/Within Variance Ratio)
- Maximize Silhouette Score  
- Minimize DBI (Davies-Bouldin Index)
- Maintain Balance Score (soft constraint)
- Maintain Temporal Smoothness (soft constraint)

Uses tools from src/utils/ml_common/optimization/
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json

from src.utils.tprint import tprint
from src.utils.ml_common.optimization import (
    HyperparameterOptimization,
    ParetoOptimizer,
    ParetoFrontAnalyzer,
    HierarchicalHPO,
    HierarchicalHPOConfig,
    HPOPhaseConfig
)


@dataclass
class IterativeOptimizationMetrics:
    """Metrics from iterative optimization run."""
    cv_score: float
    silhouette_score: float
    dbi_score: float
    balance_score: float
    temporal_smoothness: float
    n_clusters: int
    cluster_sizes: List[int]
    optimization_time: float
    
    def get_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted composite score."""
        if weights is None:
            weights = {
                'cv': 0.30,
                'silhouette': 0.25,
                'dbi': 0.20,  # Inverted: lower is better
                'balance': 0.15,
                'temporal': 0.10
            }
        
        # Normalize DBI (invert since lower is better)
        dbi_normalized = 1.0 / (1.0 + self.dbi_score) if self.dbi_score > 0 else 0.0
        
        score = (
            weights['cv'] * self.cv_score +
            weights['silhouette'] * max(0, self.silhouette_score) +  # Clip negative
            weights['dbi'] * dbi_normalized +
            weights['balance'] * self.balance_score +
            weights['temporal'] * self.temporal_smoothness
        )
        
        return score
    
    def meets_constraints(self, 
                         min_balance: float = 0.5,
                         min_temporal: float = 0.85,
                         target_clusters: Tuple[int, int] = (6, 8)) -> bool:
        """Check if metrics meet minimum constraints."""
        return (
            self.balance_score >= min_balance and
            self.temporal_smoothness >= min_temporal and
            target_clusters[0] <= self.n_clusters <= target_clusters[1]
        )


@dataclass
class OptimizationParameterSpace:
    """Define the hyperparameter search space for iterative optimization."""
    
    # Core K and size constraints
    K_MIN: Tuple[int, int] = (5, 8)  # Range for minimum clusters
    K_MAX: Tuple[int, int] = (8, 12)  # Range for maximum clusters
    MIN_FRAC: Tuple[float, float] = (0.02, 0.05)  # Minimum cluster size fraction
    MAX_FRAC: Tuple[float, float] = (0.15, 0.25)  # Maximum cluster size fraction
    
    # Objective weights - focus on CV, Silhouette, Temporal
    w_cv: Tuple[float, float] = (0.50, 0.80)  # Weight for CV ratio
    w_sil: Tuple[float, float] = (0.05, 0.20)  # Weight for Silhouette
    w_temp: Tuple[float, float] = (0.10, 0.30)  # Weight for Temporal smoothness
    w_bal: Tuple[float, float] = (0.02, 0.10)  # Weight for Balance
    
    # Optimization thresholds
    eps_std_step1: Tuple[float, float] = (-0.30, -0.10)  # Step 1 threshold
    sil_guard: Tuple[float, float] = (-0.10, -0.05)  # Silhouette guard
    temporal_bonus: Tuple[float, float] = (0.15, 0.35)  # Temporal bonus
    
    # Lexicographic acceptor thresholds
    eps_cv: Tuple[float, float] = (1e-6, 1e-4)  # CV threshold
    eps_sil: Tuple[float, float] = (1e-5, 1e-3)  # Silhouette threshold
    eps_temp: Tuple[float, float] = (1e-5, 1e-3)  # Temporal threshold
    
    # Size-aware parameters
    size_gate_base: Tuple[float, float] = (5e-5, 5e-4)
    size_gate_alpha: Tuple[float, float] = (0.01, 0.05)
    size_gate_beta: Tuple[float, float] = (0.02, 0.08)
    
    # Performance parameters
    max_rounds: Tuple[int, int] = (20, 50)  # Number of optimization rounds
    local_churn_cap: Tuple[int, int] = (3000, 7000)  # Step 1 guard
    knn_size: Tuple[int, int] = (15, 35)  # kNN neighbor consensus
    
    def to_optuna_space(self, trial) -> Dict[str, Any]:
        """Convert to Optuna trial suggestions."""
        params = {}
        
        # Integer parameters
        params['K_MIN'] = trial.suggest_int('K_MIN', self.K_MIN[0], self.K_MIN[1])
        params['K_MAX'] = trial.suggest_int('K_MAX', self.K_MAX[0], self.K_MAX[1])
        params['max_rounds'] = trial.suggest_int('max_rounds', self.max_rounds[0], self.max_rounds[1])
        params['local_churn_cap'] = trial.suggest_int('local_churn_cap', self.local_churn_cap[0], self.local_churn_cap[1])
        params['knn_size'] = trial.suggest_int('knn_size', self.knn_size[0], self.knn_size[1])
        
        # Float parameters - weights
        params['w_cv'] = trial.suggest_float('w_cv', self.w_cv[0], self.w_cv[1])
        params['w_sil'] = trial.suggest_float('w_sil', self.w_sil[0], self.w_sil[1])
        params['w_temp'] = trial.suggest_float('w_temp', self.w_temp[0], self.w_temp[1])
        params['w_bal'] = trial.suggest_float('w_bal', self.w_bal[0], self.w_bal[1])
        
        # Normalize weights to sum to 1.0
        total_weight = params['w_cv'] + params['w_sil'] + params['w_temp'] + params['w_bal']
        if total_weight > 0:
            params['w_cv'] /= total_weight
            params['w_sil'] /= total_weight
            params['w_temp'] /= total_weight
            params['w_bal'] /= total_weight
        
        # Float parameters - thresholds
        params['MIN_FRAC'] = trial.suggest_float('MIN_FRAC', self.MIN_FRAC[0], self.MIN_FRAC[1])
        params['MAX_FRAC'] = trial.suggest_float('MAX_FRAC', self.MAX_FRAC[0], self.MAX_FRAC[1])
        params['eps_std_step1'] = trial.suggest_float('eps_std_step1', self.eps_std_step1[0], self.eps_std_step1[1])
        params['sil_guard'] = trial.suggest_float('sil_guard', self.sil_guard[0], self.sil_guard[1])
        params['temporal_bonus'] = trial.suggest_float('temporal_bonus', self.temporal_bonus[0], self.temporal_bonus[1])
        
        # Log-scale parameters for lexicographic thresholds
        params['eps_cv'] = trial.suggest_float('eps_cv', self.eps_cv[0], self.eps_cv[1], log=True)
        params['eps_sil'] = trial.suggest_float('eps_sil', self.eps_sil[0], self.eps_sil[1], log=True)
        params['eps_temp'] = trial.suggest_float('eps_temp', self.eps_temp[0], self.eps_temp[1], log=True)
        
        # Size-aware parameters
        params['size_gate_base'] = trial.suggest_float('size_gate_base', self.size_gate_base[0], self.size_gate_base[1], log=True)
        params['size_gate_alpha'] = trial.suggest_float('size_gate_alpha', self.size_gate_alpha[0], self.size_gate_alpha[1])
        params['size_gate_beta'] = trial.suggest_float('size_gate_beta', self.size_gate_beta[0], self.size_gate_beta[1])
        
        return params


class IterativeOptimizationTuner:
    """Tunes hyperparameters for iterative optimization to maximize clustering quality."""
    
    def __init__(self, 
                 features: np.ndarray,
                 initial_labels: np.ndarray,
                 market_data: pd.DataFrame,
                 verbose: bool = True):
        """
        Initialize the tuner.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            initial_labels: Initial cluster labels from HDBSCAN
            market_data: Market data DataFrame
            verbose: Enable verbose output
        """
        self.features = features
        self.initial_labels = initial_labels
        self.market_data = market_data
        self.verbose = verbose
        
        # Filter out noise labels for optimization
        self.noise_mask = initial_labels >= 0
        self.filtered_features = features[self.noise_mask]
        self.filtered_labels = initial_labels[self.noise_mask]
        
        tprint(f"🎯 Initialized tuner with {len(self.filtered_labels)} samples ({len(features) - len(self.filtered_labels)} noise filtered)", "INFO")
        
        # Results storage
        self.best_params = None
        self.best_metrics = None
        self.optimization_history = []
        
    def _run_single_trial(self, params: Dict[str, Any]) -> IterativeOptimizationMetrics:
        """
        Run iterative optimization with given parameters and return metrics.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            IterativeOptimizationMetrics object
        """
        try:
            import time
            from src.training.steps.market_analysis.clusters.iterative_optimization import IterativeOptimization
            from src.training.steps.market_analysis.clusters.step1_feature_preparation import ClusteringContext
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            
            start_time = time.time()
            
            # Create configuration from params
            config = self._params_to_config(params)
            
            # Create context
            context = ClusteringContext(
                original_features=self.filtered_features,
                market_data=self.market_data
            )
            context.initial_assignments = self.filtered_labels.copy()
            context.assignments = self.filtered_labels.copy()
            context.optimized_features = self.filtered_features
            context.optimal_k = len(np.unique(self.filtered_labels))
            
            # Run optimization
            optimizer = IterativeOptimization(verbose=False)
            
            # Apply parameters to optimizer config
            optimizer.config.K_MIN = params['K_MIN']
            optimizer.config.K_MAX = params['K_MAX']
            optimizer.config.MIN_FRAC = params['MIN_FRAC']
            optimizer.config.MAX_FRAC = params['MAX_FRAC']
            optimizer.config.w_cv = params['w_cv']
            optimizer.config.w_sil = params['w_sil']
            optimizer.config.w_temp = params['w_temp']
            optimizer.config.w_bal = params['w_bal']
            optimizer.config.eps_std_step1 = params['eps_std_step1']
            optimizer.config.sil_guard = params['sil_guard']
            optimizer.config.temporal_bonus = params['temporal_bonus']
            optimizer.config.eps_cv = params['eps_cv']
            optimizer.config.eps_sil = params['eps_sil']
            optimizer.config.eps_temp = params['eps_temp']
            optimizer.config.max_rounds = params['max_rounds']
            optimizer.config.local_churn_cap = params['local_churn_cap']
            optimizer.config.knn_size = params['knn_size']
            optimizer.config.size_gate_base = params['size_gate_base']
            optimizer.config.size_gate_alpha = params['size_gate_alpha']
            optimizer.config.size_gate_beta = params['size_gate_beta']
            
            # Run optimization synchronously with proper async handling
            try:
                # Handle nested event loop issue
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
                
                # Try to get existing event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Event loop is already running - use ThreadPoolExecutor
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                optimizer.execute_optimization_loop(
                                    context, config,
                                    max_iterations=params['max_rounds'],
                                    enable_risk_mitigation=True
                                )
                            )
                            optimized_context = future.result()
                    else:
                        # Loop exists but not running
                        optimized_context = loop.run_until_complete(
                            optimizer.execute_optimization_loop(
                                context, config,
                                max_iterations=params['max_rounds'],
                                enable_risk_mitigation=True
                            )
                        )
                except RuntimeError:
                    # No event loop exists - create one
                    optimized_context = asyncio.run(
                        optimizer.execute_optimization_loop(
                            context, config,
                            max_iterations=params['max_rounds'],
                            enable_risk_mitigation=True
                        )
                    )
            except Exception as e:
                tprint(f"❌ Trial failed during optimization: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                # Return poor metrics
                return IterativeOptimizationMetrics(
                    cv_score=0.0,
                    silhouette_score=-1.0,
                    dbi_score=10.0,
                    balance_score=0.0,
                    temporal_smoothness=0.0,
                    n_clusters=0,
                    cluster_sizes=[],
                    optimization_time=time.time() - start_time
                )
            
            # Extract results
            optimized_labels = optimized_context.assignments if hasattr(optimized_context, 'assignments') else optimized_context.optimized_assignments
            
            # Calculate metrics
            n_clusters = len(np.unique(optimized_labels))
            cluster_sizes = [int(np.sum(optimized_labels == i)) for i in range(n_clusters)]
            
            # Calculate CV ratio (between/within variance)
            from sklearn.metrics import calinski_harabasz_score
            within_variance = self._calculate_within_variance(self.filtered_features, optimized_labels)
            between_variance = self._calculate_between_variance(self.filtered_features, optimized_labels)
            cv_score = between_variance / within_variance if within_variance > 0 else 0.0
            
            # Calculate silhouette score
            if n_clusters >= 2:
                try:
                    silhouette = silhouette_score(self.filtered_features, optimized_labels)
                except:
                    silhouette = -1.0
            else:
                silhouette = -1.0
            
            # Calculate DBI score
            if n_clusters >= 2:
                try:
                    dbi = davies_bouldin_score(self.filtered_features, optimized_labels)
                except:
                    dbi = 10.0
            else:
                dbi = 10.0
            
            # Calculate balance score
            balance = self._calculate_balance_score(cluster_sizes)
            
            # Calculate temporal smoothness
            temporal = self._calculate_temporal_smoothness(optimized_labels)
            
            optimization_time = time.time() - start_time
            
            return IterativeOptimizationMetrics(
                cv_score=cv_score,
                silhouette_score=silhouette,
                dbi_score=dbi,
                balance_score=balance,
                temporal_smoothness=temporal,
                n_clusters=n_clusters,
                cluster_sizes=cluster_sizes,
                optimization_time=optimization_time
            )
            
        except Exception as e:
            tprint(f"❌ Trial execution failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            # Return poor metrics
            return IterativeOptimizationMetrics(
                cv_score=0.0,
                silhouette_score=-1.0,
                dbi_score=10.0,
                balance_score=0.0,
                temporal_smoothness=0.0,
                n_clusters=0,
                cluster_sizes=[],
                optimization_time=0.0
            )
    
    def _params_to_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert optimization parameters to config dict."""
        return {
            'min_clusters': params['K_MIN'],
            'max_clusters': params['K_MAX'],
            'iterative_max_iterations': params['max_rounds'],
            'iterative_convergence_threshold': 0.001,
            'iterative_enable_risk_mitigation': True
        }
    
    def _calculate_within_variance(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster variance."""
        total_wcss = 0.0
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 0:
                centroid = np.mean(cluster_features, axis=0)
                wcss = np.sum((cluster_features - centroid) ** 2)
                total_wcss += wcss
        return total_wcss / len(features) if len(features) > 0 else 0.0
    
    def _calculate_between_variance(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate between-cluster variance."""
        global_mean = np.mean(features, axis=0)
        total_bcss = 0.0
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 0:
                centroid = np.mean(cluster_features, axis=0)
                bcss = len(cluster_features) * np.sum((centroid - global_mean) ** 2)
                total_bcss += bcss
        return total_bcss / len(features) if len(features) > 0 else 0.0
    
    def _calculate_balance_score(self, cluster_sizes: List[int]) -> float:
        """Calculate cluster balance score (0-1, higher is better)."""
        if not cluster_sizes or len(cluster_sizes) < 2:
            return 0.0
        sizes_array = np.array(cluster_sizes)
        mean_size = np.mean(sizes_array)
        if mean_size == 0:
            return 0.0
        cv = np.std(sizes_array) / mean_size  # Coefficient of variation
        # Convert to 0-1 score (lower CV is better balance)
        balance = 1.0 / (1.0 + cv)
        return balance
    
    def _calculate_temporal_smoothness(self, labels: np.ndarray) -> float:
        """Calculate temporal smoothness (ratio of consecutive identical labels)."""
        if len(labels) < 2:
            return 0.0
        changes = np.sum(labels[1:] != labels[:-1])
        total_pairs = len(labels) - 1
        smoothness = 1.0 - (changes / total_pairs)
        return smoothness
    
    def _objective_function(self, trial) -> float:
        """
        Objective function for Optuna optimization.
        Returns composite score to maximize.
        """
        # Get parameter suggestions from trial
        param_space = OptimizationParameterSpace()
        params = param_space.to_optuna_space(trial)
        
        # Ensure K_MIN < K_MAX
        if params['K_MIN'] >= params['K_MAX']:
            params['K_MAX'] = params['K_MIN'] + 2
        
        # Run trial
        metrics = self._run_single_trial(params)
        
        # Store history
        self.optimization_history.append({
            'trial': trial.number,
            'params': params,
            'metrics': metrics
        })
        
        # Check if constraints are met
        if not metrics.meets_constraints():
            # Penalize trials that don't meet constraints
            penalty = -10.0
            tprint(f"❌ Trial {trial.number} failed constraints: {metrics.n_clusters} clusters, balance={metrics.balance_score:.3f}, temporal={metrics.temporal_smoothness:.3f}", "WARNING")
            return penalty
        
        # Calculate composite score
        composite = metrics.get_composite_score()
        
        if self.verbose:
            tprint(f"✅ Trial {trial.number}: CV={metrics.cv_score:.3f}, Sil={metrics.silhouette_score:.3f}, DBI={metrics.dbi_score:.3f}, Balance={metrics.balance_score:.3f}, Temporal={metrics.temporal_smoothness:.3f}, K={metrics.n_clusters}, Score={composite:.4f}", "INFO")
        
        # Store as multi-objective values for Pareto analysis
        trial.set_user_attr('cv_score', metrics.cv_score)
        trial.set_user_attr('silhouette_score', metrics.silhouette_score)
        trial.set_user_attr('dbi_score', metrics.dbi_score)
        trial.set_user_attr('balance_score', metrics.balance_score)
        trial.set_user_attr('temporal_smoothness', metrics.temporal_smoothness)
        trial.set_user_attr('n_clusters', metrics.n_clusters)
        
        return composite
    
    def optimize_bayesian(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        Run Bayesian optimization using Optuna TPE sampler.
        
        Args:
            n_trials: Number of trials to run
            
        Returns:
            Dictionary with best parameters and metrics
        """
        tprint(f"🚀 Starting Bayesian hyperparameter optimization ({n_trials} trials)...", "INFO")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            # Create study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
                study_name=f"iterative_opt_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Run optimization
            study.optimize(self._objective_function, n_trials=n_trials, show_progress_bar=True)
            
            # Get best trial
            best_trial = study.best_trial
            best_params = best_trial.params
            best_score = best_trial.value
            
            # Extract metrics from best trial
            best_metrics = IterativeOptimizationMetrics(
                cv_score=best_trial.user_attrs['cv_score'],
                silhouette_score=best_trial.user_attrs['silhouette_score'],
                dbi_score=best_trial.user_attrs['dbi_score'],
                balance_score=best_trial.user_attrs['balance_score'],
                temporal_smoothness=best_trial.user_attrs['temporal_smoothness'],
                n_clusters=best_trial.user_attrs['n_clusters'],
                cluster_sizes=[],
                optimization_time=0.0
            )
            
            self.best_params = best_params
            self.best_metrics = best_metrics
            
            tprint(f"✅ Bayesian optimization completed!", "SUCCESS")
            tprint(f"📊 Best composite score: {best_score:.4f}", "SUCCESS")
            tprint(f"🎯 Best parameters: CV={best_metrics.cv_score:.3f}, Sil={best_metrics.silhouette_score:.3f}, DBI={best_metrics.dbi_score:.3f}, K={best_metrics.n_clusters}", "SUCCESS")
            
            return {
                'best_params': best_params,
                'best_metrics': best_metrics,
                'best_score': best_score,
                'study': study,
                'optimization_history': self.optimization_history
            }
            
        except Exception as e:
            tprint(f"❌ Bayesian optimization failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None
    
    def optimize_multiobjective(self, n_trials: int = 30) -> Dict[str, Any]:
        """
        Run multi-objective optimization to find Pareto-optimal solutions.
        
        Optimizes:
        - Maximize CV score
        - Maximize Silhouette score
        - Minimize DBI score
        - Maintain Balance >= 0.5
        - Maintain Temporal >= 0.85
        
        Args:
            n_trials: Number of trials
            
        Returns:
            Dictionary with Pareto front and best compromise solution
        """
        tprint(f"🎯 Starting multi-objective optimization ({n_trials} trials)...", "INFO")
        
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def multiobjective_func(trial):
                """Return tuple of objectives to optimize."""
                param_space = OptimizationParameterSpace()
                params = param_space.to_optuna_space(trial)
                
                # Ensure K_MIN < K_MAX
                if params['K_MIN'] >= params['K_MAX']:
                    params['K_MAX'] = params['K_MIN'] + 2
                
                metrics = self._run_single_trial(params)
                
                # Store history
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': params,
                    'metrics': metrics
                })
                
                # Return multiple objectives (Optuna will find Pareto front)
                # Objectives: maximize CV, maximize Silhouette, minimize DBI
                return (
                    metrics.cv_score,  # Maximize
                    metrics.silhouette_score,  # Maximize
                    -metrics.dbi_score  # Maximize negative (i.e., minimize DBI)
                )
            
            # Create multi-objective study
            study = optuna.create_study(
                directions=['maximize', 'maximize', 'maximize'],
                sampler=optuna.samplers.NSGAIISampler(seed=42),
                study_name=f"multiobjective_iterative_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Run optimization
            study.optimize(multiobjective_func, n_trials=n_trials, show_progress_bar=True)
            
            # Analyze Pareto front
            pareto_trials = []
            for trial in study.best_trials:
                if all(attr in trial.user_attrs for attr in ['cv_score', 'silhouette_score', 'dbi_score', 'balance_score', 'temporal_smoothness', 'n_clusters']):
                    metrics = IterativeOptimizationMetrics(
                        cv_score=trial.values[0],
                        silhouette_score=trial.values[1],
                        dbi_score=-trial.values[2],  # Convert back to original scale
                        balance_score=trial.user_attrs.get('balance_score', 0.0),
                        temporal_smoothness=trial.user_attrs.get('temporal_smoothness', 0.0),
                        n_clusters=trial.user_attrs.get('n_clusters', 0),
                        cluster_sizes=[],
                        optimization_time=0.0
                    )
                    pareto_trials.append({
                        'trial_number': trial.number,
                        'params': trial.params,
                        'metrics': metrics
                    })
            
            # Find best compromise solution from Pareto front
            best_compromise = self._find_best_compromise(pareto_trials)
            
            tprint(f"✅ Multi-objective optimization completed!", "SUCCESS")
            tprint(f"📊 Found {len(pareto_trials)} Pareto-optimal solutions", "SUCCESS")
            if best_compromise:
                tprint(f"🎯 Best compromise: CV={best_compromise['metrics'].cv_score:.3f}, Sil={best_compromise['metrics'].silhouette_score:.3f}, DBI={best_compromise['metrics'].dbi_score:.3f}", "SUCCESS")
            
            return {
                'pareto_front': pareto_trials,
                'best_compromise': best_compromise,
                'study': study,
                'optimization_history': self.optimization_history
            }
            
        except Exception as e:
            tprint(f"❌ Multi-objective optimization failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_best_compromise(self, pareto_trials: List[Dict]) -> Optional[Dict]:
        """Find the best compromise solution from Pareto front."""
        if not pareto_trials:
            return None
        
        # Score each solution by composite metric with constraints
        best_solution = None
        best_score = -float('inf')
        
        for trial in pareto_trials:
            metrics = trial['metrics']
            
            # Check constraints
            if not metrics.meets_constraints():
                continue
            
            # Calculate composite score
            composite = metrics.get_composite_score()
            
            if composite > best_score:
                best_score = composite
                best_solution = trial
        
        return best_solution
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save optimization results to file."""
        try:
            # Convert results to serializable format
            serializable_results = {
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(self.filtered_labels),
                'n_features': self.filtered_features.shape[1],
                'best_params': results.get('best_params'),
                'best_metrics': {
                    'cv_score': results['best_metrics'].cv_score,
                    'silhouette_score': results['best_metrics'].silhouette_score,
                    'dbi_score': results['best_metrics'].dbi_score,
                    'balance_score': results['best_metrics'].balance_score,
                    'temporal_smoothness': results['best_metrics'].temporal_smoothness,
                    'n_clusters': results['best_metrics'].n_clusters,
                    'cluster_sizes': results['best_metrics'].cluster_sizes
                } if 'best_metrics' in results and results['best_metrics'] else None,
                'n_trials': len(self.optimization_history)
            }
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            tprint(f"✅ Results saved to: {output_path}", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to save results: {e}", "ERROR")
    
    def generate_report(self, results: Dict[str, Any], output_path: str):
        """Generate comprehensive optimization report."""
        try:
            report = []
            report.append("# Iterative Optimization Hyperparameter Tuning Report\n")
            report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report.append(f"**Dataset**: {len(self.filtered_labels)} samples, {self.filtered_features.shape[1]} features\n")
            report.append("\n## Optimization Summary\n")
            
            if 'best_params' in results and 'best_metrics' in results:
                metrics = results['best_metrics']
                report.append(f"**Total Trials**: {len(self.optimization_history)}\n")
                report.append(f"**Best Composite Score**: {results.get('best_score', 'N/A'):.4f}\n")
                report.append("\n### Best Configuration Metrics\n")
                report.append("| Metric | Value | Status |\n")
                report.append("|--------|-------|--------|\n")
                report.append(f"| CV Score | {metrics.cv_score:.4f} | {'✅' if metrics.cv_score > 1.0 else '⚠️'} |\n")
                report.append(f"| Silhouette Score | {metrics.silhouette_score:.4f} | {'✅' if metrics.silhouette_score > 0.2 else '⚠️'} |\n")
                report.append(f"| DBI Score | {metrics.dbi_score:.4f} | {'✅' if metrics.dbi_score < 1.5 else '⚠️'} |\n")
                report.append(f"| Balance Score | {metrics.balance_score:.4f} | {'✅' if metrics.balance_score > 0.5 else '⚠️'} |\n")
                report.append(f"| Temporal Smoothness | {metrics.temporal_smoothness:.4f} | {'✅' if metrics.temporal_smoothness > 0.85 else '⚠️'} |\n")
                report.append(f"| Number of Clusters | {metrics.n_clusters} | {'✅' if 6 <= metrics.n_clusters <= 8 else '⚠️'} |\n")
                
                report.append("\n### Best Parameters\n")
                report.append("```json\n")
                report.append(json.dumps(results['best_params'], indent=2))
                report.append("\n```\n")
            
            # Save report
            with open(output_path, 'w') as f:
                f.writelines(report)
            
            tprint(f"✅ Report saved to: {output_path}", "SUCCESS")
            
        except Exception as e:
            tprint(f"❌ Failed to generate report: {e}", "ERROR")


def run_tuning_pipeline(
    features: np.ndarray,
    initial_labels: np.ndarray,
    market_data: pd.DataFrame,
    n_trials: int = 30,
    method: str = 'bayesian',
    output_dir: str = 'artifacts/hyperparameter_tuning/'
) -> Dict[str, Any]:
    """
    Run the complete hyperparameter tuning pipeline.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        initial_labels: Initial cluster labels from HDBSCAN
        market_data: Market data DataFrame
        n_trials: Number of optimization trials
        method: 'bayesian' or 'multiobjective'
        output_dir: Directory to save results
        
    Returns:
        Dictionary with optimization results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tuner
    tuner = IterativeOptimizationTuner(features, initial_labels, market_data, verbose=True)
    
    # Run optimization
    if method == 'bayesian':
        results = tuner.optimize_bayesian(n_trials=n_trials)
    elif method == 'multiobjective':
        results = tuner.optimize_multiobjective(n_trials=n_trials)
    else:
        tprint(f"❌ Unknown method: {method}", "ERROR")
        return None
    
    if results is None:
        return None
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_dir, f"optimization_results_{timestamp}.json")
    report_path = os.path.join(output_dir, f"optimization_report_{timestamp}.md")
    
    tuner.save_results(results, results_path)
    tuner.generate_report(results, report_path)
    
    return results


# Example usage function
if __name__ == "__main__":
    """
    Example usage:
    
    from src.training.steps.market_analysis.clusters.iterative_optimization_tuner import run_tuning_pipeline
    
    # Load your data
    features = ...  # From regime_feature_selection
    initial_labels = ...  # From HDBSCAN
    market_data = ...  # From feature_generation
    
    # Run tuning
    results = run_tuning_pipeline(
        features=features,
        initial_labels=initial_labels,
        market_data=market_data,
        n_trials=30,  # Adjust based on time budget
        method='bayesian'  # or 'multiobjective'
    )
    
    # Apply best parameters to OptConfig in iterative_optimization.py
    # Edit lines 2489-2562 with the best_params from results
    """
    tprint("💡 This is a utility module. Import and use run_tuning_pipeline() to optimize hyperparameters.", "INFO")

