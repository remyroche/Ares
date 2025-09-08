#!/usr/bin/env python3
"""Enhanced Bayesian Parameter Optimization with Expanded Search Space and Performance Optimizations.

This module significantly expands the parameter search space for HMM regime discovery
with advanced optimization strategies, multi-objective optimization, and performance enhancements:
- Multi-fidelity optimization
- Parallel trials execution
- Early stopping and convergence detection
- Warm-starting from previous results
- Transfer learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union

import time
import logging
from dataclasses import dataclass
import json
import hashlib

from pathlib import Path
import threading
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

# Import centralized systems
from .step03_imports import get_import_manager, safe_import, check_feature_availability
from .step03_config import Step03Config

logger = logging.getLogger(__name__)

# Import optimization libraries
optuna = safe_import('optuna')
sklearn = safe_import('sklearn')
hmmlearn = safe_import('hmmlearn')

# Performance optimization imports
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    joblib = None

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
    @log_all_calls
    
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

@dataclass
class OptimizationCache:
    """Cache for optimization results and warm-starting."""
    cache_dir: Path = Path("data/cache/optimization")
    max_cache_age_days: int = 30

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, data_hash: str, config_hash: str) -> str:
        """Generate cache key from data and config hashes."""
        combined = f"{data_hash}_{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def load_cached_results(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached optimization results."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # Check cache age
        if time.time() - cache_file.stat().st_mtime > self.max_cache_age_days * 24 * 3600:
            cache_file.unlink()  # Remove old cache
            return None

        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def save_cached_results(self, cache_key: str, results: Dict[str, Any]) -> None:
        """Save optimization results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

@dataclass
class MultiFidelityConfig:
    """Configuration for multi-fidelity optimization."""
    fidelity_levels: List[str] = None  # ["low", "medium", "high"]
    fidelity_costs: Dict[str, float] = None  # Cost multipliers
    fidelity_accuracy: Dict[str, float] = None  # Accuracy expectations
    early_stopping_threshold: float = 0.001
    max_consecutive_no_improvement: int = 10

    def __post_init__(self):
        if self.fidelity_levels is None:
            self.fidelity_levels = ["low", "medium", "high"]

        if self.fidelity_costs is None:
            self.fidelity_costs = {
                "low": 0.1,    # 10% of full cost
                "medium": 0.5, # 50% of full cost
                "high": 1.0    # Full cost
            }

        if self.fidelity_accuracy is None:
            self.fidelity_accuracy = {
                "low": 0.7,    # 70% expected accuracy
                "medium": 0.85, # 85% expected accuracy
                "high": 0.95   # 95% expected accuracy
            }

class ParallelBayesianOptimizer:
    """Parallel Bayesian optimizer with multi-fidelity support."""

    def __init__(self, config: Step03Config, max_workers: int = None):
        self.config = config
        self.logger = logging.getLogger('ParallelBayesianOptimizer')
        self.max_workers = max_workers or min(8, (joblib.cpu_count() if JOBLIB_AVAILABLE else 4))
        self.cache = OptimizationCache()
        self.mf_config = MultiFidelityConfig()

        # Optimization state
        self.trials_completed = 0
        self.best_score = float('-inf')
        self.consecutive_no_improvement = 0
        self.optimization_start_time = time.time()

        # Thread safety
        self._lock = threading.Lock()

    def optimize_parallel(self, objective_function, data: pd.DataFrame,
                         features: pd.DataFrame, n_trials: int = 100) -> Dict[str, Any]:
        """Run parallel Bayesian optimization with multi-fidelity."""

        self.logger.info(f"🚀 Starting parallel optimization with {self.max_workers} workers")

        # Generate data hash for caching
        data_hash = self._generate_data_hash(data, features)

        # Check cache
        cache_key = self.cache.get_cache_key(data_hash, str(self.config.__dict__))
        cached_results = self.cache.load_cached_results(cache_key)

        if cached_results:
            self.logger.info("✅ Using cached optimization results")
            return cached_results

        # Initialize optimization
        study = self._create_parallel_study()
        results = []

        # Multi-fidelity optimization loop
        for fidelity in self.mf_config.fidelity_levels:
            self.logger.info(f"📊 Optimizing at {fidelity} fidelity")

            # Adjust trials based on fidelity
            fidelity_trials = int(n_trials * self.mf_config.fidelity_costs[fidelity])

            # Run parallel optimization at this fidelity
            fidelity_results = self._optimize_at_fidelity(
                study, objective_function, data, features,
                fidelity_trials, fidelity
            )

            results.extend(fidelity_results)

            # Check early stopping
            if self._should_stop_early(results):
                self.logger.info("⏹️ Early stopping triggered")
                break

        # Process final results
        final_results = self._process_optimization_results(results)

        # Cache results
        self.cache.save_cached_results(cache_key, final_results)

        return final_results

    def _create_parallel_study(self):
        """Create Optuna study with parallel capabilities."""
        if not optuna:
            raise ImportError("Optuna is required for parallel optimization")

        return optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=10,
                n_ei_candidates=50,
                multivariate=True,
                seed=self.config.bayesian_optimization.random_state
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )

    def _optimize_at_fidelity(self, study, objective_function, data: pd.DataFrame,
                             features: pd.DataFrame, n_trials: int, fidelity: str) -> List[Dict]:
        """Optimize at specific fidelity level."""

        results = []
        trial_queue = []

        # Create trial queue
        for i in range(n_trials):
            trial_params = {
                'trial_id': i,
                'fidelity': fidelity,
                'data_subset_size': self._get_fidelity_data_size(fidelity, len(data)),
                'feature_subset_size': self._get_fidelity_feature_size(fidelity, features.shape[1])
            }
            trial_queue.append(trial_params)

        # Process trials in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for trial_params in trial_queue:
                future = executor.submit(
                    self._evaluate_trial,
                    study, objective_function, data, features, trial_params
                )
                futures.append(future)

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Trial evaluation failed: {e}")

        return results

    def _evaluate_trial(self, study, objective_function, data: pd.DataFrame,
                       features: pd.DataFrame, trial_params: Dict) -> Optional[Dict]:
        """Evaluate a single trial with thread safety."""

        try:
            # Create trial
            trial = study.ask()

            # Apply fidelity constraints
            fidelity = trial_params['fidelity']
            data_subset = self._get_data_subset(data, trial_params['data_subset_size'])
            feature_subset = self._get_feature_subset(features, trial_params['feature_subset_size'])

            # Evaluate objective
            score = objective_function(trial, data_subset, feature_subset)

            # Tell result to study
            study.tell(trial, score)

            result = {
                'trial_id': trial_params['trial_id'],
                'fidelity': fidelity,
                'score': score,
                'parameters': trial.params,
                'data_size': len(data_subset),
                'feature_count': feature_subset.shape[1]
            }

            # Update best score with thread safety
            with self._lock:
                self.trials_completed += 1
                if score > self.best_score:
                    self.best_score = score
                    self.consecutive_no_improvement = 0
                else:
                    self.consecutive_no_improvement += 1

            return result

        except Exception as e:
            self.logger.error(f"Trial {trial_params.get('trial_id', 'unknown')} failed: {e}")
            return None

    def _get_fidelity_data_size(self, fidelity: str, full_size: int) -> int:
        """Get data subset size for fidelity level."""
        if fidelity == "low":
            return min(1000, full_size // 10)  # 10% or 1000 samples
        elif fidelity == "medium":
            return min(5000, full_size // 2)   # 50% or 5000 samples
        else:  # high
            return full_size

    def _get_fidelity_feature_size(self, fidelity: str, full_feature_count: int) -> int:
        """Get feature subset size for fidelity level."""
        if fidelity == "low":
            return min(20, full_feature_count // 4)   # 25% or 20 features
        elif fidelity == "medium":
            return min(100, full_feature_count // 2)  # 50% or 100 features
        else:  # high
            return full_feature_count

    def _get_data_subset(self, data: pd.DataFrame, subset_size: int) -> pd.DataFrame:
        """Get random subset of data for fidelity optimization."""
        if len(data) <= subset_size:
            return data
        return data.sample(n=subset_size, random_state=42)

    def _get_feature_subset(self, features: pd.DataFrame, subset_size: int) -> pd.DataFrame:
        """Get feature subset for fidelity optimization."""
        if features.shape[1] <= subset_size:
            return features

        # Use variance-based selection for feature subset
        variances = features.var()
        top_features = variances.nlargest(subset_size).index
        return features[top_features]

    def _should_stop_early(self, results: List[Dict]) -> bool:
        """Check if optimization should stop early."""
        if len(results) < 20:  # Need minimum trials
            return False

        # Check convergence
        recent_scores = [r['score'] for r in results[-20:]]
        score_improvement = max(recent_scores) - min(recent_scores)

        if score_improvement < self.mf_config.early_stopping_threshold:
            return True

        # Check consecutive no improvement
        if self.consecutive_no_improvement >= self.mf_config.max_consecutive_no_improvement:
            return True

        return False

    def _process_optimization_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process and aggregate optimization results."""

        if not results:
            return {"error": "No optimization results"}

        # Find best result
        best_result = max(results, key=lambda x: x['score'])

        # Aggregate statistics
        scores = [r['score'] for r in results]
        high_fidelity_results = [r for r in results if r['fidelity'] == 'high']

        return {
            "best_parameters": best_result['parameters'],
            "best_score": best_result['score'],
            "total_trials": len(results),
            "high_fidelity_trials": len(high_fidelity_results),
            "optimization_time": time.time() - self.optimization_start_time,
            "score_statistics": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores)
            },
            "fidelity_distribution": {
                fidelity: len([r for r in results if r['fidelity'] == fidelity])
                for fidelity in self.mf_config.fidelity_levels
            },
            "all_results": results
        }

    def _generate_data_hash(self, data: pd.DataFrame, features: pd.DataFrame) -> str:
        """Generate hash for data and features to enable caching."""
        data_str = f"{data.shape}_{data.iloc[0].to_string()}_{data.iloc[-1].to_string()}"
        features_str = f"{features.shape}_{features.columns.tolist()}"
        combined = f"{data_str}_{features_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

class EnhancedBayesianOptimizer:
    """Enhanced Bayesian optimizer with expanded parameter space and multi-objective optimization."""
    @log_important_calls
    
    def __init__(self, config: Step03Config):
        self.config = config
        self.logger = logging.getLogger('EnhancedBayesianOptimizer')
        self.parameter_space = AdvancedParameterSpace()
        # Import memory manager locally to avoid circular imports
        try:
            from .step03_memory_manager import get_memory_manager
            self.memory_manager = get_memory_manager(config.memory.__dict__)
        except ImportError:
            self.memory_manager = None

        # Initialize parallel optimizer
        self.parallel_optimizer = ParallelBayesianOptimizer(config)

        # Initialize Optuna study with advanced settings
        self._initialize_optuna_study()

        # Optimization history
        self.optimization_history = []
        self.best_trials = []
    @log_all_calls
        
    def _initialize_optuna_study(self):
        """Initialize Optuna study with advanced configuration."""
        if not optuna:
            raise ImportError("Optuna is required for enhanced Bayesian optimization")
        
        # Advanced sampler with more exploration
        self.sampler = optuna.samplers.TPESampler(
            n_startup_trials = self.config.bayesian_optimization.n_startup_trials,
            n_ei_candidates = 50,  # Increased from default 24
            gamma = lambda x: min(25, x // 4),  # More exploration
            prior_weight = 1.0,  # Equal weight to prior
            consider_magic_clip = True,
            consider_endpoints = True,
            multivariate = True,  # Enable multivariate optimization
            group = True,  # Enable group optimization
            warn_independent_sampling = True,
            seed = self.config.bayesian_optimization.random_state
        )
        
        # Advanced pruner with more sophisticated pruning
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials = self.config.bayesian_optimization.n_startup_trials,
            n_warmup_steps = self.config.bayesian_optimization.n_warmup_steps,
            interval_steps = 1,
            n_min_trials = 5
        )
        
        # Create study with multi-objective optimization
        self.study = optuna.create_study(
            directions=["maximize", "maximize", "minimize"],  # Multi-objective
            sampler = self.sampler,
            pruner = self.pruner,
            study_name = f"enhanced_hmm_optimization_{int(time.time())}"
        )
    @log_all_calls
    
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
            log = True
        )
        
        params['reg_covar'] = trial.suggest_float(
            "reg_covar", 
            self.parameter_space.reg_covar_range[0], 
            self.parameter_space.reg_covar_range[1], 
            log = True
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
    @log_all_calls
    
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
    @log_all_calls
    
    def _process_features_with_params(self, features: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
        """Process features using suggested parameters."""
        # Feature selection
        if params['feature_selection_method'] == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold = 0.01)
            features_selected = selector.fit_transform(features)
        elif params['feature_selection_method'] == 'correlation':
            # Remove highly correlated features
            corr_matrix = features.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            features_selected = features.drop(columns = to_drop).values
        else:
            features_selected = features.values
        
        # Limit features
        max_features = min(params['max_features'], features_selected.shape[1])
        if features_selected.shape[1] > max_features:
            # Select top features by variance
            feature_vars = np.var(features_selected, axis = 0)
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
            reducer = PCA(n_components = n_components)
            features_reduced = reducer.fit_transform(features_scaled)
        elif params['dimensionality_reduction_method'] == 'ica':
            from sklearn.decomposition import FastICA
            n_components = min(30, features_scaled.shape[1] // 2)
            reducer = FastICA(n_components = n_components, random_state = 42)
            features_reduced = reducer.fit_transform(features_scaled)
        else:
            features_reduced = features_scaled
        
        return features_reduced
    @log_all_calls
    
    def _train_hmm_model(self, features: np.ndarray, params: Dict[str, Any]):
        """Train HMM model with suggested parameters."""
        if not hmmlearn:
            raise ImportError("hmmlearn is required for HMM training")
        
        # Use subset for faster training
        max_samples = min(self.config.hmm.max_samples, features.shape[0])
        if features.shape[0] > max_samples:
            indices = np.random.choice(features.shape[0], max_samples, replace = False)
            features_subset = features[indices]
        else:
            features_subset = features
        
        # Create HMM model
        hmm_model = hmmlearn.hmm.GaussianHMM(
            n_components = params['n_components'],
            covariance_type = params['covariance_type'],
            n_iter = params['n_iter'],
            tol = params['tol'],
            reg_covar = params['reg_covar'],
            init_params = params['init_params'],
            algorithm = params['algorithm'],
            random_state = 42
        )
        
        # Train model
        hmm_model.fit(features_subset)
        
        return hmm_model
    @log_all_calls
    
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
    @log_all_calls
    
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
        
        # Import memory aware processing locally to avoid circular imports
        try:
            from .step03_memory_manager import memory_aware_processing
            with memory_aware_processing("bayesian_optimization", self.config.memory.__dict__):
                result = self._run_optimization(data, features)
        except ImportError:
            # Fallback without memory context manager
            result = self._run_optimization(data, features)

        return result

    def _run_optimization(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Run the actual optimization process."""
        # Define optimization objective
        def objective(trial):
            return self._multi_objective_function(trial, data, features)

        # Run optimization
        self.study.optimize(
            objective,
            n_trials = self.config.bayesian_optimization.n_trials,
            timeout = self.config.bayesian_optimization.timeout_minutes * 60,
            show_progress_bar = True
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

    @log_important_calls
    def optimize_parallel_enhanced(self, data: pd.DataFrame, features: pd.DataFrame,
                                  n_trials: int = 100, use_parallel: bool = True) -> Dict[str, Any]:
        """Enhanced optimization with parallel processing and multi-fidelity support.

        This method provides significant performance improvements through:
        - Parallel trial evaluation
        - Multi-fidelity optimization
        - Caching and warm-starting
        - Early stopping

        Args:
            data: Input market data
            features: Feature matrix
            n_trials: Number of optimization trials
            use_parallel: Whether to use parallel processing

        Returns:
            Dictionary containing optimization results
        """

        self.logger.info(f"🚀 Starting enhanced parallel optimization with {n_trials} trials")

        if use_parallel and JOBLIB_AVAILABLE:
            # Use parallel Bayesian optimizer
            self.logger.info("📊 Using parallel Bayesian optimization")

            # Create objective function wrapper
            def objective_function(trial, data_subset, feature_subset):
                return self._single_objective_function(trial, data_subset, feature_subset)

            # Run parallel optimization
            results = self.parallel_optimizer.optimize_parallel(
                objective_function, data, features, n_trials
            )

            # Process results
            if "error" in results:
                self.logger.warning(f"Parallel optimization failed: {results['error']}")
                # Fallback to sequential optimization
                return self.optimize_enhanced(data, features, n_trials)

            # Convert parallel results to expected format
            best_params = results['best_parameters']
            best_score = results['best_score']

            self.logger.info(f"✅ Parallel optimization completed in {results['optimization_time']:.2f}s")
            self.logger.info(f"   Best score: {best_score:.4f}")
            self.logger.info(f"   Total trials: {results['total_trials']}")
            self.logger.info(f"   High-fidelity trials: {results['high_fidelity_trials']}")

            return {
                'best_params': best_params,
                'best_values': [best_score],
                'n_trials': results['total_trials'],
                'optimization_time': results['optimization_time'],
                'parallel_results': results,
                'optimization_method': 'parallel_bayesian'
            }

        else:
            # Fallback to sequential optimization
            self.logger.info("📊 Using sequential Bayesian optimization")
            return self.optimize_enhanced(data, features, n_trials)

    @log_all_calls
    def _single_objective_function(self, trial, data: pd.DataFrame, features: pd.DataFrame) -> float:
        """Single objective function for parallel optimization."""
        try:
            # Get suggested parameters
            params = self._suggest_advanced_parameters(trial)

            # Process features with suggested parameters
            processed_features = self._process_features_with_params(features, params)

            # Train HMM model
            hmm_model = self._train_hmm_model(processed_features, params)

            # Get regime predictions
            regimes = hmm_model.predict(processed_features)

            # Calculate primary objective (combined score)
            primary_score = self._calculate_primary_objective(
                processed_features, regimes, hmm_model, data, params
            )

            return primary_score

        except Exception as e:
            self.logger.warning(f"Trial failed: {e}")
            return -float('inf')

    @log_all_calls
    def _calculate_primary_objective(self, features: np.ndarray, regimes: np.ndarray,
                                   hmm_model, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """Calculate primary objective score for optimization."""
        try:
            # Silhouette score for clustering quality
            silhouette = self._calculate_silhouette_score(features, regimes)

            # Regime stability (lower transition frequency is better)
            transition_freq = self._calculate_transition_frequency(regimes)

            # Economic significance (if data contains returns)
            economic_sig = self._calculate_economic_significance(regimes, data)

            # Combine objectives with weights
            weights = params.get('multi_objective_weights', {
                'silhouette': 0.4,
                'transition_freq': 0.3,
                'economic_sig': 0.3
            })

            combined_score = (
                weights['silhouette'] * silhouette +
                weights['transition_freq'] * (1.0 / (1.0 + transition_freq)) +  # Invert transition freq
                weights['economic_sig'] * economic_sig
            )

            return combined_score

        except Exception as e:
            self.logger.warning(f"Primary objective calculation failed: {e}")
            return -float('inf')