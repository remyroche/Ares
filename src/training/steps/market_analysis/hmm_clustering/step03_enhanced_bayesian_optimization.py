#!/usr/bin/env python3
"""Enhanced Bayesian Parameter Optimization with Expanded Search Space.

This module significantly expands the parameter search space for HMM regime discovery
with advanced optimization strategies and multi-objective optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import asyncio
import time
import logging
from dataclasses import dataclass
import json

# Import centralized systems
from .step03_imports import get_import_manager, safe_import, check_feature_availability
from .step03_config import Step03Config
from .step03_memory_manager import get_memory_manager, memory_aware_processing

logger = logging.getLogger(__name__)

# Import optimization libraries
optuna = safe_import('optuna')
sklearn = safe_import('sklearn')
hmmlearn = safe_import('hmmlearn')


@dataclass
class AdvancedParameterSpace:
    """Advanced parameter space definition for comprehensive optimization."""
    
    # HMM Parameters - Expanded ranges
    n_components_range: Tuple[int, int] = (2, 12)  # Increased from (2, 8)
    covariance_types: List[str] = None  # Will be set to all available types
    n_iter_range: Tuple[int, int] = (50, 500)  # Increased from (50, 200)
    tol_range: Tuple[float, float] = (1e-8, 1e-1)  # Expanded range
    reg_covar_range: Tuple[float, float] = (1e-9, 1e-1)  # Expanded range
    
    # Advanced HMM parameters
    init_params: List[str] = None  # Will be set to all available
    algorithm: List[str] = None  # Will be set to all available
    random_state_range: Tuple[int, int] = (1, 1000)
    
    # Feature Engineering Parameters
    feature_selection_methods: List[str] = None  # Will be set to comprehensive list
    max_features_range: Tuple[int, int] = (10, 200)  # Increased from (20, 100)
    feature_scaling_methods: List[str] = None  # Will be set to all methods
    dimensionality_reduction_methods: List[str] = None  # Will be set to all methods
    
    # Ensemble Parameters
    ensemble_methods: List[str] = None  # Will be set to comprehensive list
    ensemble_weights_ranges: Dict[str, Tuple[float, float]] = None  # Will be set
    
    # Advanced Optimization Parameters
    multi_objective_weights: Dict[str, float] = None  # Will be set
    constraint_parameters: Dict[str, Any] = None  # Will be set
    
    def __post_init__(self):
        """Initialize comprehensive parameter spaces."""
        if self.covariance_types is None:
            self.covariance_types = ["full", "tied", "diag", "spherical"]
        
        if self.init_params is None:
            self.init_params = ["kmeans", "random"]
        
        if self.algorithm is None:
            self.algorithm = ["viterbi", "map"]
        
        if self.feature_selection_methods is None:
            self.feature_selection_methods = [
                "variance", "correlation", "mutual_info", "f_score", 
                "chi2", "rfe", "lasso", "elastic_net", "boruta"
            ]
        
        if self.feature_scaling_methods is None:
            self.feature_scaling_methods = [
                "standard", "minmax", "robust", "quantile", "power", "yeo_johnson"
            ]
        
        if self.dimensionality_reduction_methods is None:
            self.dimensionality_reduction_methods = [
                "pca", "ica", "truncated_svd", "factor_analysis", 
                "lda", "tsne", "umap", "autoencoder"
            ]
        
        if self.ensemble_methods is None:
            self.ensemble_methods = [
                "hmm", "kmeans", "dbscan", "gaussian_mixture", 
                "spectral", "agglomerative", "birch", "optics"
            ]
        
        if self.ensemble_weights_ranges is None:
            self.ensemble_weights_ranges = {
                "hmm": (0.1, 0.6),
                "kmeans": (0.1, 0.5),
                "dbscan": (0.1, 0.5),
                "gaussian_mixture": (0.1, 0.4),
                "spectral": (0.1, 0.4),
                "agglomerative": (0.1, 0.3),
                "birch": (0.1, 0.3),
                "optics": (0.1, 0.3)
            }
        
        if self.multi_objective_weights is None:
            self.multi_objective_weights = {
                "hmm_score": 0.25,
                "silhouette_score": 0.20,
                "calinski_harabasz_score": 0.15,
                "davies_bouldin_score": 0.15,
                "regime_stability": 0.10,
                "regime_balance": 0.05,
                "economic_significance": 0.10
            }
        
        if self.constraint_parameters is None:
            self.constraint_parameters = {
                "min_regime_size": 0.05,  # Minimum 5% of data per regime
                "max_regime_size": 0.80,  # Maximum 80% of data per regime
                "min_regime_duration": 10,  # Minimum 10 periods per regime
                "max_transition_frequency": 0.1,  # Maximum 10% transition rate
                "min_economic_significance": 0.001  # Minimum economic significance
            }


class EnhancedBayesianOptimizer:
    """Enhanced Bayesian optimizer with expanded parameter space and multi-objective optimization."""
    
    def __init__(self, config: Step03Config):
        self.config = config
        self.logger = logging.getLogger('EnhancedBayesianOptimizer')
        self.parameter_space = AdvancedParameterSpace()
        self.memory_manager = get_memory_manager(config.memory.__dict__)
        
        # Initialize Optuna study with advanced settings
        self._initialize_optuna_study()
        
        # Optimization history
        self.optimization_history = []
        self.best_trials = []
        
    def _initialize_optuna_study(self):
        """Initialize Optuna study with advanced configuration."""
        if not optuna:
            raise ImportError("Optuna is required for enhanced Bayesian optimization")
        
        # Advanced sampler with more exploration
        self.sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.config.bayesian_optimization.n_startup_trials,
            n_ei_candidates=50,  # Increased from default 24
            gamma=lambda x: min(25, x // 4),  # More exploration
            prior_weight=1.0,  # Equal weight to prior
            consider_magic_clip=True,
            consider_endpoints=True,
            multivariate=True,  # Enable multivariate optimization
            group=True,  # Enable group optimization
            warn_independent_sampling=True,
            seed=self.config.bayesian_optimization.random_state
        )
        
        # Advanced pruner with more sophisticated pruning
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=self.config.bayesian_optimization.n_startup_trials,
            n_warmup_steps=self.config.bayesian_optimization.n_warmup_steps,
            interval_steps=1,
            n_min_trials=5
        )
        
        # Create study with multi-objective optimization
        self.study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],  # Multi-objective
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"enhanced_hmm_optimization_{int(time.time())}"
        )
    
    def _suggest_advanced_parameters(self, trial) -> Dict[str, Any]:
        """Suggest parameters from expanded search space."""
        params = {}
        
        # HMM Parameters - Expanded
        params['n_components'] = trial.suggest_int(
            "n_components", 
            self.parameter_space.n_components_range[0], 
            self.parameter_space.n_components_range[1]
        )
        
        params['covariance_type'] = trial.suggest_categorical(
            "covariance_type", 
            self.parameter_space.covariance_types
        )
        
        params['n_iter'] = trial.suggest_int(
            "n_iter", 
            self.parameter_space.n_iter_range[0], 
            self.parameter_space.n_iter_range[1]
        )
        
        params['tol'] = trial.suggest_float(
            "tol", 
            self.parameter_space.tol_range[0], 
            self.parameter_space.tol_range[1], 
            log=True
        )
        
        params['reg_covar'] = trial.suggest_float(
            "reg_covar", 
            self.parameter_space.reg_covar_range[0], 
            self.parameter_space.reg_covar_range[1], 
            log=True
        )
        
        # Advanced HMM Parameters
        params['init_params'] = trial.suggest_categorical(
            "init_params", 
            self.parameter_space.init_params
        )
        
        params['algorithm'] = trial.suggest_categorical(
            "algorithm", 
            self.parameter_space.algorithm
        )
        
        # Feature Engineering Parameters
        params['feature_selection_method'] = trial.suggest_categorical(
            "feature_selection_method", 
            self.parameter_space.feature_selection_methods
        )
        
        params['max_features'] = trial.suggest_int(
            "max_features", 
            self.parameter_space.max_features_range[0], 
            self.parameter_space.max_features_range[1]
        )
        
        params['feature_scaling_method'] = trial.suggest_categorical(
            "feature_scaling_method", 
            self.parameter_space.feature_scaling_methods
        )
        
        params['dimensionality_reduction_method'] = trial.suggest_categorical(
            "dimensionality_reduction_method", 
            self.parameter_space.dimensionality_reduction_methods
        )
        
        # Ensemble Parameters
        params['ensemble_methods'] = trial.suggest_categorical(
            "ensemble_methods", 
            [method for method in self.parameter_space.ensemble_methods]
        )
        
        # Ensemble weights (dynamic based on selected methods)
        selected_methods = params['ensemble_methods']
        if isinstance(selected_methods, str):
            selected_methods = [selected_methods]
        
        ensemble_weights = {}
        for method in selected_methods:
            if method in self.parameter_space.ensemble_weights_ranges:
                weight_range = self.parameter_space.ensemble_weights_ranges[method]
                ensemble_weights[method] = trial.suggest_float(
                    f"ensemble_weight_{method}", 
                    weight_range[0], 
                    weight_range[1]
                )
        
        # Normalize weights
        total_weight = sum(ensemble_weights.values())
        if total_weight > 0:
            ensemble_weights = {k: v/total_weight for k, v in ensemble_weights.items()}
        
        params['ensemble_weights'] = ensemble_weights
        
        # Advanced Optimization Parameters
        params['multi_objective_weights'] = {}
        for objective, weight_range in self.parameter_space.multi_objective_weights.items():
            params['multi_objective_weights'][objective] = trial.suggest_float(
                f"weight_{objective}", 
                0.0, 
                1.0
            )
        
        # Normalize multi-objective weights
        total_weight = sum(params['multi_objective_weights'].values())
        if total_weight > 0:
            params['multi_objective_weights'] = {
                k: v/total_weight for k, v in params['multi_objective_weights'].items()
            }
        
        return params
    
    def _multi_objective_function(self, trial, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[float, float, float]:
        """Multi-objective optimization function."""
        try:
            # Get suggested parameters
            params = self._suggest_advanced_parameters(trial)
            
            # Process features with suggested parameters
            processed_features = self._process_features_with_params(features, params)
            
            # Train HMM model
            hmm_model = self._train_hmm_model(processed_features, params)
            
            # Get regime predictions
            regimes = hmm_model.predict(processed_features)
            regime_probs = hmm_model.predict_proba(processed_features)
            
            # Calculate multiple objectives
            objectives = self._calculate_multi_objectives(
                processed_features, regimes, regime_probs, hmm_model, data, params
            )
            
            # Report intermediate values for pruning
            trial.report(objectives[0], 0)  # Primary objective
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return objectives
            
        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return (-float('inf'), -float('inf'), float('inf'))
    
    def _process_features_with_params(self, features: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Process features using suggested parameters."""
        # Feature selection
        if params['feature_selection_method'] == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            features_selected = selector.fit_transform(features)
        elif params['feature_selection_method'] == 'correlation':
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            features_selected = features.drop(columns=to_drop).values
        else:
            features_selected = features.values
        
        # Limit features
        max_features = min(params['max_features'], features_selected.shape[1])
        if features_selected.shape[1] > max_features:
            # Select top features by variance
            feature_vars = np.var(features_selected, axis=0)
            top_features_idx = np.argsort(feature_vars)[-max_features:]
            features_selected = features_selected[:, top_features_idx]
        
        # Feature scaling
        if params['feature_scaling_method'] == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif params['feature_scaling_method'] == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif params['feature_scaling_method'] == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        features_scaled = scaler.fit_transform(features_selected)
        
        # Dimensionality reduction
        if params['dimensionality_reduction_method'] == 'pca':
            from sklearn.decomposition import PCA
            n_components = min(50, features_scaled.shape[1] // 2)
            reducer = PCA(n_components=n_components)
            features_reduced = reducer.fit_transform(features_scaled)
        elif params['dimensionality_reduction_method'] == 'ica':
            from sklearn.decomposition import FastICA
            n_components = min(30, features_scaled.shape[1] // 2)
            reducer = FastICA(n_components=n_components, random_state=42)
            features_reduced = reducer.fit_transform(features_scaled)
        else:
            features_reduced = features_scaled
        
        return features_reduced
    
    def _train_hmm_model(self, features: np.ndarray, params: Dict[str, Any]):
        """Train HMM model with suggested parameters."""
        if not hmmlearn:
            raise ImportError("hmmlearn is required for HMM training")
        
        # Use subset for faster training
        max_samples = min(self.config.hmm.max_samples, features.shape[0])
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace=False)
            features_subset = features[indices]
        else:
            features_subset = features
        
        # Create HMM model
        hmm_model = hmmlearn.hmm.GaussianHMM(
            n_components=params['n_components'],
            covariance_type=params['covariance_type'],
            n_iter=params['n_iter'],
            tol=params['tol'],
            reg_covar=params['reg_covar'],
            init_params=params['init_params'],
            algorithm=params['algorithm'],
            random_state=42
        )
        
        # Train model
        hmm_model.fit(features_subset)
        
        return hmm_model
    
    def _calculate_multi_objectives(self, features: np.ndarray, regimes: np.ndarray, 
                                  regime_probs: np.ndarray, hmm_model, data: pd.DataFrame, 
                                  params: Dict[str, Any]) -> Tuple[float, float, float]:
        """Calculate multiple optimization objectives."""
        # Objective 1: HMM Quality Score (maximize)
        hmm_score = hmm_model.score(features)
        hmm_score_normalized = max(0, min(1, (hmm_score + 1000) / 1000))
        
        # Objective 2: Clustering Quality (maximize)
        if len(np.unique(regimes)) > 1:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(features, regimes)
            except:
                silhouette = 0.0
        else:
            silhouette = 0.0
        
        # Objective 3: Regime Stability (minimize transition frequency)
        regime_changes = np.sum(np.diff(regimes) != 0)
        transition_frequency = regime_changes / len(regimes)
        
        # Economic significance bonus
        economic_bonus = self._calculate_economic_significance_bonus(data, regimes)
        
        # Combine objectives with weights
        weights = params['multi_objective_weights']
        combined_score = (
            weights.get('hmm_score', 0.25) * hmm_score_normalized +
            weights.get('silhouette_score', 0.20) * max(0, silhouette) +
            weights.get('regime_stability', 0.10) * (1 - transition_frequency) +
            weights.get('economic_significance', 0.10) * economic_bonus
        )
        
        return (combined_score, silhouette, transition_frequency)
    
    def _calculate_economic_significance_bonus(self, data: pd.DataFrame, regimes: np.ndarray) -> float:
        """Calculate economic significance bonus."""
        try:
            returns = data['close'].pct_change().dropna()
            regime_returns = {}
            
            for regime in np.unique(regimes):
                regime_mask = regimes == regime
                if np.sum(regime_mask) > 10:
                    regime_returns[regime] = returns[regime_mask]
            
            if len(regime_returns) < 2:
                return 0.0
            
            # Calculate return differences
            return_diffs = []
            regime_list = list(regime_returns.keys())
            for i in range(len(regime_list)):
                for j in range(i + 1, len(regime_list)):
                    mean_diff = abs(np.mean(regime_returns[regime_list[i]]) - 
                                  np.mean(regime_returns[regime_list[j]]))
                    return_diffs.append(mean_diff)
            
            # Economic significance threshold
            economic_threshold = 0.001
            significant_diffs = sum(1 for diff in return_diffs if diff > economic_threshold)
            
            return significant_diffs / len(return_diffs) if return_diffs else 0.0
            
        except Exception:
            return 0.0
    
    async def optimize_parameters(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Run enhanced Bayesian optimization."""
        self.logger.info("🚀 Starting enhanced Bayesian parameter optimization...")
        
        with memory_aware_processing("bayesian_optimization", self.config.memory.__dict__):
            # Define optimization objective
            def objective(trial):
                return self._multi_objective_function(trial, data, features)
            
            # Run optimization
            self.study.optimize(
                objective,
                n_trials=self.config.bayesian_optimization.n_trials,
                timeout=self.config.bayesian_optimization.timeout_minutes * 60,
                show_progress_bar=True
            )
        
        # Get best trial
        best_trial = self.study.best_trial
        
        # Extract best parameters
        best_params = best_trial.params
        
        # Get best values
        best_values = best_trial.values
        
        self.logger.info(f"✅ Enhanced optimization completed")
        self.logger.info(f"   Best combined score: {best_values[0]:.4f}")
        self.logger.info(f"   Best silhouette score: {best_values[1]:.4f}")
        self.logger.info(f"   Best transition frequency: {best_values[2]:.4f}")
        
        return {
            'best_params': best_params,
            'best_values': best_values,
            'n_trials': len(self.study.trials),
            'study': self.study,
            'optimization_history': self.optimization_history
        }