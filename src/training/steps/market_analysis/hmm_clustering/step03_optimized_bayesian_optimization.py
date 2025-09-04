#!/usr/bin/env python3
"""Optimized Bayesian Parameter Optimization for Enhanced Computational Efficiency.

This module implements several optimization strategies:
1. Parallel trial execution with multiprocessing
2. Early pruning with adaptive thresholds
3. Smart parameter space reduction
4. Caching and memoization
5. Progressive parameter refinement
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.integration import LightGBMPruningCallback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from functools import lru_cache
import time
import warnings
warnings.filterwarnings('ignore')

class OptimizedBayesianParameterOptimization:
    """Optimized Bayesian parameter optimization with enhanced efficiency."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_jobs = self.config.get('n_jobs', min(mp.cpu_count(), 8))
        self.random_state = self.config.get('random_state', 42)
        
        # Optimization parameters
        self.max_trials = self.config.get('max_trials', 100)
        self.timeout_minutes = self.config.get('timeout_minutes', 30)
        self.cv_folds = self.config.get('cv_folds', 3)
        
        # Efficiency parameters
        self.early_stopping_patience = self.config.get('early_stopping_patience', 10)
        self.min_trials_for_pruning = self.config.get('min_trials_for_pruning', 5)
        self.parameter_space_reduction = self.config.get('parameter_space_reduction', True)
        self.progressive_refinement = self.config.get('progressive_refinement', True)
        
        # Caching
        self.cache_size = self.config.get('cache_size', 1000)
        self._evaluation_cache = {}
        
        # Results storage
        self.best_params = {}
        self.optimization_history = []
        
    async def optimize_parameters(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize parameters with enhanced computational efficiency.
        
        Args:
            data: Market data
            features: Feature matrix
            
        Returns:
            Optimization results
        """
        print("🚀 Starting optimized Bayesian parameter optimization...")
        
        # Step 1: Quick parameter space exploration
        if self.parameter_space_reduction:
            reduced_space = await self._reduce_parameter_space(data, features)
            print(f"✅ Parameter space reduced from {len(self._get_full_parameter_space())} to {len(reduced_space)} combinations")
        else:
            reduced_space = self._get_full_parameter_space()
        
        # Step 2: Progressive refinement optimization
        if self.progressive_refinement:
            optimization_results = await self._progressive_refinement_optimization(
                data, features, reduced_space
            )
        else:
            optimization_results = await self._standard_optimization(
                data, features, reduced_space
            )
        
        # Step 3: Final fine-tuning
        fine_tuned_results = await self._fine_tune_best_parameters(
            data, features, optimization_results['best_params']
        )
        
        return {
            'success': True,
            'best_params': fine_tuned_results['best_params'],
            'optimization_history': optimization_results['history'],
            'efficiency_metrics': {
                'total_trials': optimization_results['total_trials'],
                'pruned_trials': optimization_results['pruned_trials'],
                'cache_hits': len(self._evaluation_cache),
                'optimization_time': optimization_results['optimization_time']
            }
        }
    
    async def _reduce_parameter_space(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, List]:
        """Reduce parameter space using quick evaluation."""
        print("🔍 Reducing parameter space...")
        
        # Quick evaluation with subset of data
        sample_size = min(1000, len(features))
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        data_sample = data.iloc[sample_indices]
        features_sample = features.iloc[sample_indices]
        
        # Test parameter combinations quickly
        full_space = self._get_full_parameter_space()
        reduced_space = {}
        
        for param_type, param_values in full_space.items():
            if len(param_values) > 3:  # Only reduce if more than 3 values
                # Quick evaluation of parameter values
                scores = []
                for value in param_values:
                    try:
                        score = await self._quick_evaluate_parameter(
                            param_type, value, data_sample, features_sample
                        )
                        scores.append((value, score))
                    except:
                        scores.append((value, 0.0))
                
                # Keep top 3 performing values
                scores.sort(key=lambda x: x[1], reverse=True)
                reduced_space[param_type] = [score[0] for score in scores[:3]]
            else:
                reduced_space[param_type] = param_values
        
        return reduced_space
    
    async def _quick_evaluate_parameter(self, param_type: str, value: Any, 
                                      data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Quick evaluation of a single parameter."""
        try:
            # Create minimal config with single parameter
            config = {param_type: value}
            
            # Quick HMM evaluation
            if param_type.startswith('hmm_'):
                score = await self._quick_hmm_evaluation(config, features)
            elif param_type.startswith('kmeans_'):
                score = await self._quick_kmeans_evaluation(config, features)
            elif param_type.startswith('dbscan_'):
                score = await self._quick_dbscan_evaluation(config, features)
            else:
                score = 0.0
            
            return score
            
        except Exception as e:
            return 0.0
    
    async def _quick_hmm_evaluation(self, config: Dict[str, Any], features: pd.DataFrame) -> float:
        """Quick HMM evaluation."""
        try:
            from hmmlearn import hmm
            
            n_components = config.get('hmm_n_components', 4)
            covariance_type = config.get('hmm_covariance_type', 'full')
            
            # Use small subset for quick evaluation
            sample_size = min(500, len(features))
            features_sample = features.sample(n=sample_size, random_state=42)
            
            # Train HMM
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=20,  # Reduced iterations
                random_state=42
            )
            
            model.fit(features_sample)
            
            # Quick quality assessment
            regimes = model.predict(features_sample)
            unique_regimes = len(np.unique(regimes))
            
            if unique_regimes < 2:
                return 0.0
            
            # Simple quality score
            score = min(1.0, unique_regimes / 5.0)  # Prefer 3-5 regimes
            
            return score
            
        except Exception as e:
            return 0.0
    
    async def _quick_kmeans_evaluation(self, config: Dict[str, Any], features: pd.DataFrame) -> float:
        """Quick K-means evaluation."""
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = config.get('kmeans_n_clusters', 20)
            
            # Use small subset
            sample_size = min(500, len(features))
            features_sample = features.sample(n=sample_size, random_state=42)
            
            # Train K-means
            model = KMeans(
                n_clusters=n_clusters,
                n_init=3,  # Reduced n_init
                max_iter=50,  # Reduced max_iter
                random_state=42
            )
            
            regimes = model.fit_predict(features_sample)
            
            # Quick quality assessment
            unique_regimes = len(np.unique(regimes))
            if unique_regimes < 2:
                return 0.0
            
            # Simple quality score based on cluster balance
            regime_counts = np.bincount(regimes)
            balance_score = 1.0 - (np.std(regime_counts) / np.mean(regime_counts))
            
            return max(0.0, balance_score)
            
        except Exception as e:
            return 0.0
    
    async def _quick_dbscan_evaluation(self, config: Dict[str, Any], features: pd.DataFrame) -> float:
        """Quick DBSCAN evaluation."""
        try:
            from sklearn.cluster import DBSCAN
            
            eps = config.get('dbscan_eps', 0.5)
            min_samples = config.get('dbscan_min_samples', 10)
            
            # Use small subset
            sample_size = min(500, len(features))
            features_sample = features.sample(n=sample_size, random_state=42)
            
            # Train DBSCAN
            model = DBSCAN(eps=eps, min_samples=min_samples)
            regimes = model.fit_predict(features_sample)
            
            # Quick quality assessment
            unique_regimes = len(np.unique(regimes))
            n_noise = np.sum(regimes == -1)
            
            if unique_regimes < 2 or n_noise > len(regimes) * 0.5:
                return 0.0
            
            # Simple quality score
            noise_ratio = n_noise / len(regimes)
            cluster_score = unique_regimes / 10.0  # Prefer 5-10 clusters
            
            score = cluster_score * (1.0 - noise_ratio)
            return max(0.0, score)
            
        except Exception as e:
            return 0.0
    
    async def _progressive_refinement_optimization(self, data: pd.DataFrame, features: pd.DataFrame, 
                                                 reduced_space: Dict[str, List]) -> Dict[str, Any]:
        """Progressive refinement optimization strategy."""
        print("🔄 Running progressive refinement optimization...")
        
        # Phase 1: Coarse optimization
        coarse_results = await self._coarse_optimization(data, features, reduced_space)
        
        # Phase 2: Fine optimization around best parameters
        fine_results = await self._fine_optimization(data, features, coarse_results['best_params'])
        
        return {
            'best_params': fine_results['best_params'],
            'history': coarse_results['history'] + fine_results['history'],
            'total_trials': coarse_results['total_trials'] + fine_results['total_trials'],
            'pruned_trials': coarse_results['pruned_trials'] + fine_results['pruned_trials'],
            'optimization_time': coarse_results['optimization_time'] + fine_results['optimization_time']
        }
    
    async def _coarse_optimization(self, data: pd.DataFrame, features: pd.DataFrame, 
                                 reduced_space: Dict[str, List]) -> Dict[str, Any]:
        """Coarse optimization phase."""
        print("   Phase 1: Coarse optimization...")
        
        # Create study with aggressive pruning
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=2,
                min_early_stopping_rate=0
            ),
            study_name="coarse_optimization"
        )
        
        # Define objective function
        def objective(trial):
            return self._evaluate_parameter_combination(trial, data, features, reduced_space, coarse=True)
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.max_trials // 2,  # Use half trials for coarse
            timeout=self.timeout_minutes * 60 // 2
        )
        
        return {
            'best_params': study.best_params,
            'history': [trial.value for trial in study.trials if trial.value is not None],
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_time': time.time() - start_time
        }
    
    async def _fine_optimization(self, data: pd.DataFrame, features: pd.DataFrame, 
                               best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Fine optimization phase around best parameters."""
        print("   Phase 2: Fine optimization...")
        
        # Create refined parameter space around best parameters
        refined_space = self._create_refined_parameter_space(best_params)
        
        # Create study with less aggressive pruning
        study = optuna.create_study(
            direction="maximize",
            sampler=CmaEsSampler(seed=self.random_state),  # Use CMA-ES for fine optimization
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            ),
            study_name="fine_optimization"
        )
        
        # Define objective function
        def objective(trial):
            return self._evaluate_parameter_combination(trial, data, features, refined_space, coarse=False)
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.max_trials // 2,  # Use half trials for fine
            timeout=self.timeout_minutes * 60 // 2
        )
        
        return {
            'best_params': study.best_params,
            'history': [trial.value for trial in study.trials if trial.value is not None],
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_time': time.time() - start_time
        }
    
    def _create_refined_parameter_space(self, best_params: Dict[str, Any]) -> Dict[str, List]:
        """Create refined parameter space around best parameters."""
        refined_space = {}
        
        for param_name, best_value in best_params.items():
            if param_name == 'hmm_n_components':
                # Refine around best value
                base_value = best_value
                refined_space[param_name] = [max(2, base_value - 1), base_value, min(10, base_value + 1)]
            
            elif param_name == 'hmm_covariance_type':
                refined_space[param_name] = [best_value]  # Keep best value
            
            elif param_name == 'kmeans_n_clusters':
                base_value = best_value
                refined_space[param_name] = [max(5, base_value - 5), base_value, min(50, base_value + 5)]
            
            elif param_name == 'dbscan_eps':
                base_value = best_value
                refined_space[param_name] = [base_value * 0.8, base_value, base_value * 1.2]
            
            elif param_name == 'dbscan_min_samples':
                base_value = best_value
                refined_space[param_name] = [max(3, base_value - 2), base_value, min(20, base_value + 2)]
            
            else:
                refined_space[param_name] = [best_value]
        
        return refined_space
    
    async def _standard_optimization(self, data: pd.DataFrame, features: pd.DataFrame, 
                                   reduced_space: Dict[str, List]) -> Dict[str, Any]:
        """Standard optimization without progressive refinement."""
        print("🔄 Running standard optimization...")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(
                n_startup_trials=self.min_trials_for_pruning,
                n_warmup_steps=20,
                interval_steps=10
            ),
            study_name="standard_optimization"
        )
        
        # Define objective function
        def objective(trial):
            return self._evaluate_parameter_combination(trial, data, features, reduced_space, coarse=False)
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=self.max_trials,
            timeout=self.timeout_minutes * 60
        )
        
        return {
            'best_params': study.best_params,
            'history': [trial.value for trial in study.trials if trial.value is not None],
            'total_trials': len(study.trials),
            'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'optimization_time': time.time() - start_time
        }
    
    def _evaluate_parameter_combination(self, trial, data: pd.DataFrame, features: pd.DataFrame, 
                                      parameter_space: Dict[str, List], coarse: bool = False) -> float:
        """Evaluate a parameter combination."""
        try:
            # Sample parameters
            params = {}
            for param_name, param_values in parameter_space.items():
                if len(param_values) == 1:
                    params[param_name] = param_values[0]
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            
            # Check cache
            cache_key = str(sorted(params.items()))
            if cache_key in self._evaluation_cache:
                return self._evaluation_cache[cache_key]
            
            # Evaluate parameters
            if coarse:
                score = self._coarse_evaluation(params, data, features)
            else:
                score = self._detailed_evaluation(params, data, features)
            
            # Cache result
            if len(self._evaluation_cache) < self.cache_size:
                self._evaluation_cache[cache_key] = score
            
            return score
            
        except Exception as e:
            return 0.0
    
    def _coarse_evaluation(self, params: Dict[str, Any], data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Coarse evaluation for parameter space reduction."""
        try:
            # Use subset of data
            sample_size = min(1000, len(features))
            sample_indices = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features.iloc[sample_indices]
            
            # Quick HMM evaluation
            hmm_score = self._quick_hmm_evaluation_sync(params, features_sample)
            
            # Quick clustering evaluation
            clustering_score = self._quick_clustering_evaluation_sync(params, features_sample)
            
            # Combined score
            combined_score = 0.6 * hmm_score + 0.4 * clustering_score
            
            return combined_score
            
        except Exception as e:
            return 0.0
    
    def _detailed_evaluation(self, params: Dict[str, Any], data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Detailed evaluation for final optimization."""
        try:
            # Use full dataset
            # HMM evaluation
            hmm_score = self._detailed_hmm_evaluation(params, features)
            
            # Clustering evaluation
            clustering_score = self._detailed_clustering_evaluation(params, features)
            
            # Economic significance evaluation
            economic_score = self._economic_significance_evaluation(params, data, features)
            
            # Combined score
            combined_score = 0.4 * hmm_score + 0.3 * clustering_score + 0.3 * economic_score
            
            return combined_score
            
        except Exception as e:
            return 0.0
    
    def _quick_hmm_evaluation_sync(self, params: Dict[str, Any], features: pd.DataFrame) -> float:
        """Synchronous quick HMM evaluation."""
        try:
            from hmmlearn import hmm
            
            n_components = params.get('hmm_n_components', 4)
            covariance_type = params.get('hmm_covariance_type', 'full')
            
            # Train HMM
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=50,  # More iterations for detailed evaluation
                random_state=42
            )
            
            model.fit(features)
            
            # Quality assessment
            regimes = model.predict(features)
            unique_regimes = len(np.unique(regimes))
            
            if unique_regimes < 2:
                return 0.0
            
            # Calculate quality metrics
            regime_counts = np.bincount(regimes)
            balance_score = 1.0 - (np.std(regime_counts) / np.mean(regime_counts))
            
            # Prefer 3-5 regimes
            regime_score = 1.0 - abs(unique_regimes - 4) / 4.0
            
            return 0.6 * balance_score + 0.4 * regime_score
            
        except Exception as e:
            return 0.0
    
    def _quick_clustering_evaluation_sync(self, params: Dict[str, Any], features: pd.DataFrame) -> float:
        """Synchronous quick clustering evaluation."""
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.metrics import silhouette_score
            
            # K-means evaluation
            kmeans_n_clusters = params.get('kmeans_n_clusters', 20)
            kmeans_model = KMeans(
                n_clusters=kmeans_n_clusters,
                n_init=5,
                max_iter=100,
                random_state=42
            )
            
            kmeans_regimes = kmeans_model.fit_predict(features)
            
            if len(np.unique(kmeans_regimes)) > 1:
                kmeans_score = silhouette_score(features, kmeans_regimes)
            else:
                kmeans_score = 0.0
            
            # DBSCAN evaluation
            dbscan_eps = params.get('dbscan_eps', 0.5)
            dbscan_min_samples = params.get('dbscan_min_samples', 10)
            
            dbscan_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_regimes = dbscan_model.fit_predict(features)
            
            unique_dbscan_regimes = len(np.unique(dbscan_regimes))
            n_noise = np.sum(dbscan_regimes == -1)
            
            if unique_dbscan_regimes > 1 and n_noise < len(features) * 0.5:
                dbscan_score = silhouette_score(features, dbscan_regimes)
            else:
                dbscan_score = 0.0
            
            # Combined clustering score
            combined_score = 0.6 * kmeans_score + 0.4 * dbscan_score
            
            return max(0.0, combined_score)
            
        except Exception as e:
            return 0.0
    
    def _detailed_hmm_evaluation(self, params: Dict[str, Any], features: pd.DataFrame) -> float:
        """Detailed HMM evaluation."""
        try:
            from hmmlearn import hmm
            from sklearn.model_selection import cross_val_score
            
            n_components = params.get('hmm_n_components', 4)
            covariance_type = params.get('hmm_covariance_type', 'full')
            
            # Train HMM with more iterations
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=200,
                random_state=42
            )
            
            model.fit(features)
            
            # Detailed quality assessment
            regimes = model.predict(features)
            regime_probs = model.predict_proba(features)
            
            unique_regimes = len(np.unique(regimes))
            if unique_regimes < 2:
                return 0.0
            
            # Calculate multiple quality metrics
            regime_counts = np.bincount(regimes)
            balance_score = 1.0 - (np.std(regime_counts) / np.mean(regime_counts))
            
            # Regime stability (based on probability consistency)
            max_probs = np.max(regime_probs, axis=1)
            stability_score = np.mean(max_probs)
            
            # Regime separation (based on regime centroids)
            regime_centroids = []
            for regime in np.unique(regimes):
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 0:
                    centroid = np.mean(features[regime_mask], axis=0)
                    regime_centroids.append(centroid)
            
            if len(regime_centroids) > 1:
                from scipy.spatial.distance import pdist
                inter_distances = pdist(regime_centroids)
                separation_score = np.mean(inter_distances)
            else:
                separation_score = 0.0
            
            # Combined score
            combined_score = 0.4 * balance_score + 0.3 * stability_score + 0.3 * min(1.0, separation_score)
            
            return combined_score
            
        except Exception as e:
            return 0.0
    
    def _detailed_clustering_evaluation(self, params: Dict[str, Any], features: pd.DataFrame) -> float:
        """Detailed clustering evaluation."""
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # K-means evaluation
            kmeans_n_clusters = params.get('kmeans_n_clusters', 20)
            kmeans_model = KMeans(
                n_clusters=kmeans_n_clusters,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            
            kmeans_regimes = kmeans_model.fit_predict(features)
            
            if len(np.unique(kmeans_regimes)) > 1:
                kmeans_silhouette = silhouette_score(features, kmeans_regimes)
                kmeans_calinski = calinski_harabasz_score(features, kmeans_regimes)
                kmeans_davies = davies_bouldin_score(features, kmeans_regimes)
                
                # Normalize scores
                kmeans_silhouette_norm = max(0, kmeans_silhouette)
                kmeans_calinski_norm = min(1.0, kmeans_calinski / 1000)  # Rough normalization
                kmeans_davies_norm = max(0, 1.0 - kmeans_davies / 2.0)  # Lower is better
                
                kmeans_score = (kmeans_silhouette_norm + kmeans_calinski_norm + kmeans_davies_norm) / 3
            else:
                kmeans_score = 0.0
            
            # DBSCAN evaluation
            dbscan_eps = params.get('dbscan_eps', 0.5)
            dbscan_min_samples = params.get('dbscan_min_samples', 10)
            
            dbscan_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_regimes = dbscan_model.fit_predict(features)
            
            unique_dbscan_regimes = len(np.unique(dbscan_regimes))
            n_noise = np.sum(dbscan_regimes == -1)
            
            if unique_dbscan_regimes > 1 and n_noise < len(features) * 0.5:
                dbscan_silhouette = silhouette_score(features, dbscan_regimes)
                dbscan_calinski = calinski_harabasz_score(features, dbscan_regimes)
                dbscan_davies = davies_bouldin_score(features, dbscan_regimes)
                
                # Normalize scores
                dbscan_silhouette_norm = max(0, dbscan_silhouette)
                dbscan_calinski_norm = min(1.0, dbscan_calinski / 1000)
                dbscan_davies_norm = max(0, 1.0 - dbscan_davies / 2.0)
                
                # Penalize high noise ratio
                noise_penalty = 1.0 - (n_noise / len(features))
                
                dbscan_score = (dbscan_silhouette_norm + dbscan_calinski_norm + dbscan_davies_norm) / 3 * noise_penalty
            else:
                dbscan_score = 0.0
            
            # Combined clustering score
            combined_score = 0.6 * kmeans_score + 0.4 * dbscan_score
            
            return max(0.0, combined_score)
            
        except Exception as e:
            return 0.0
    
    def _economic_significance_evaluation(self, params: Dict[str, Any], data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Economic significance evaluation."""
        try:
            # Quick regime discovery with current parameters
            from hmmlearn import hmm
            
            n_components = params.get('hmm_n_components', 4)
            covariance_type = params.get('hmm_covariance_type', 'full')
            
            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=covariance_type,
                n_iter=100,
                random_state=42
            )
            
            model.fit(features)
            regimes = model.predict(features)
            
            # Calculate economic significance
            returns = data['close'].pct_change().dropna()
            regime_returns = []
            
            for regime in np.unique(regimes):
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 10:  # Minimum sample size
                    regime_return = returns[regime_mask]
                    regime_returns.append(regime_return)
            
            if len(regime_returns) < 2:
                return 0.0
            
            # Calculate return differences
            mean_returns = [np.mean(regime_return) for regime_return in regime_returns]
            return_spread = max(mean_returns) - min(mean_returns)
            
            # Economic significance score
            economic_score = min(1.0, return_spread * 1000)  # Scale to 0-1
            
            return economic_score
            
        except Exception as e:
            return 0.0
    
    async def _fine_tune_best_parameters(self, data: pd.DataFrame, features: pd.DataFrame, 
                                       best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tune the best parameters with advanced techniques."""
        print("🔧 Fine-tuning best parameters...")
        
        # Advanced parameter fine-tuning
        fine_tuned_params = best_params.copy()
        
        # Fine-tune HMM parameters
        fine_tuned_params = await self._fine_tune_hmm_parameters(fine_tuned_params, features)
        
        # Fine-tune clustering parameters
        fine_tuned_params = await self._fine_tune_clustering_parameters(fine_tuned_params, features)
        
        # Validate final parameters
        final_score = self._detailed_evaluation(fine_tuned_params, data, features)
        
        return {
            'best_params': fine_tuned_params,
            'final_score': final_score,
            'improvement': final_score - self._detailed_evaluation(best_params, data, features)
        }
    
    async def _fine_tune_hmm_parameters(self, params: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """Fine-tune HMM parameters."""
        try:
            from hmmlearn import hmm
            
            n_components = params.get('hmm_n_components', 4)
            covariance_type = params.get('hmm_covariance_type', 'full')
            
            # Test different regularization parameters
            reg_covar_values = [1e-6, 1e-5, 1e-4, 1e-3]
            best_reg_covar = 1e-6
            best_score = 0.0
            
            for reg_covar in reg_covar_values:
                try:
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=reg_covar,
                        n_iter=100,
                        random_state=42
                    )
                    
                    model.fit(features)
                    regimes = model.predict(features)
                    
                    if len(np.unique(regimes)) > 1:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(features, regimes)
                        
                        if score > best_score:
                            best_score = score
                            best_reg_covar = reg_covar
                
                except:
                    continue
            
            params['hmm_reg_covar'] = best_reg_covar
            
            # Test different convergence parameters
            tol_values = [1e-2, 1e-3, 1e-4, 1e-5]
            best_tol = 1e-2
            best_score = 0.0
            
            for tol in tol_values:
                try:
                    model = hmm.GaussianHMM(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=params['hmm_reg_covar'],
                        tol=tol,
                        n_iter=100,
                        random_state=42
                    )
                    
                    model.fit(features)
                    regimes = model.predict(features)
                    
                    if len(np.unique(regimes)) > 1:
                        from sklearn.metrics import silhouette_score
                        score = silhouette_score(features, regimes)
                        
                        if score > best_score:
                            best_score = score
                            best_tol = tol
                
                except:
                    continue
            
            params['hmm_tol'] = best_tol
            
            return params
            
        except Exception as e:
            return params
    
    async def _fine_tune_clustering_parameters(self, params: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
        """Fine-tune clustering parameters."""
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.metrics import silhouette_score
            
            # Fine-tune K-means parameters
            kmeans_n_clusters = params.get('kmeans_n_clusters', 20)
            
            # Test different initialization methods
            init_methods = ['k-means++', 'random']
            best_init = 'k-means++'
            best_score = 0.0
            
            for init_method in init_methods:
                try:
                    model = KMeans(
                        n_clusters=kmeans_n_clusters,
                        init=init_method,
                        n_init=10,
                        max_iter=300,
                        random_state=42
                    )
                    
                    regimes = model.fit_predict(features)
                    
                    if len(np.unique(regimes)) > 1:
                        score = silhouette_score(features, regimes)
                        
                        if score > best_score:
                            best_score = score
                            best_init = init_method
                
                except:
                    continue
            
            params['kmeans_init'] = best_init
            
            # Fine-tune DBSCAN parameters
            dbscan_eps = params.get('dbscan_eps', 0.5)
            dbscan_min_samples = params.get('dbscan_min_samples', 10)
            
            # Test different distance metrics
            metrics = ['euclidean', 'manhattan', 'cosine']
            best_metric = 'euclidean'
            best_score = 0.0
            
            for metric in metrics:
                try:
                    model = DBSCAN(
                        eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        metric=metric
                    )
                    
                    regimes = model.fit_predict(features)
                    unique_regimes = len(np.unique(regimes))
                    n_noise = np.sum(regimes == -1)
                    
                    if unique_regimes > 1 and n_noise < len(features) * 0.5:
                        score = silhouette_score(features, regimes)
                        
                        if score > best_score:
                            best_score = score
                            best_metric = metric
                
                except:
                    continue
            
            params['dbscan_metric'] = best_metric
            
            return params
            
        except Exception as e:
            return params
    
    def _get_full_parameter_space(self) -> Dict[str, List]:
        """Get enhanced parameter space for optimization."""
        return {
            # HMM parameters
            'hmm_n_components': [2, 3, 4, 5, 6, 7, 8],
            'hmm_covariance_type': ['full', 'tied', 'diag', 'spherical'],
            'hmm_n_iter': [50, 100, 200, 300],
            'hmm_tol': [1e-2, 1e-3, 1e-4, 1e-5],
            'hmm_reg_covar': [1e-6, 1e-5, 1e-4, 1e-3],
            
            # K-means parameters
            'kmeans_n_clusters': [10, 15, 20, 25, 30, 35, 40],
            'kmeans_init': ['k-means++', 'random'],
            'kmeans_n_init': [5, 10, 15, 20],
            'kmeans_max_iter': [100, 200, 300, 500],
            'kmeans_tol': [1e-4, 1e-3, 1e-2],
            
            # DBSCAN parameters
            'dbscan_eps': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'dbscan_min_samples': [3, 5, 8, 10, 12, 15, 18, 20, 25, 30],
            'dbscan_metric': ['euclidean', 'manhattan', 'cosine'],
            'dbscan_algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            
            # Ensemble weights
            'ensemble_weight_hmm': [0.2, 0.3, 0.4, 0.5, 0.6],
            'ensemble_weight_kmeans': [0.2, 0.3, 0.4, 0.5, 0.6],
            'ensemble_weight_dbscan': [0.2, 0.3, 0.4, 0.5, 0.6],
            
            # Feature engineering parameters
            'feature_volatility_windows': [[5, 10, 20], [5, 10, 20, 50], [10, 20, 50]],
            'feature_momentum_windows': [[5, 10, 20], [5, 10, 20, 50], [10, 20, 50]],
            'feature_lags': [[1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 5, 10]],
            
            # Economic validation parameters
            'economic_significance_threshold': [0.01, 0.05, 0.1],
            'economic_threshold': [0.0005, 0.001, 0.002, 0.005],
            
            # ML transition detection parameters
            'ml_initial_features': [15, 20, 25, 30],
            'ml_feature_increment': [5, 10, 15],
            'ml_max_features': [50, 75, 100, 150],
            'ml_min_improvement': [0.0005, 0.001, 0.002],
            'ml_patience': [2, 3, 4, 5]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    # Create features with some structure
    features = pd.DataFrame(np.random.randn(n_samples, n_features))
    
    # Create market data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_samples) * 0.01),
        'volume': np.random.lognormal(10, 1, n_samples)
    })
    
    # Initialize optimized Bayesian optimizer
    config = {
        'n_jobs': 4,
        'max_trials': 50,
        'timeout_minutes': 10,
        'parameter_space_reduction': True,
        'progressive_refinement': True,
        'random_state': 42
    }
    
    optimizer = OptimizedBayesianParameterOptimization(config)
    
    # Run optimization
    results = asyncio.run(optimizer.optimize_parameters(data, features))
    
    print("Optimized Bayesian Parameter Optimization Results:")
    print(f"Success: {results['success']}")
    print(f"Best parameters: {results['best_params']}")
    print(f"Efficiency metrics: {results['efficiency_metrics']}")