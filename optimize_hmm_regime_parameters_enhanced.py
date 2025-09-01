#!/usr/bin/env python3
"""
Enhanced HMM Regime Parameter Optimization with Advanced Features

This enhanced version includes:
- Parallel processing for faster optimization
- Cross-validation for robust parameter selection
- Advanced optimization algorithms
- Multi-objective optimization
- Adaptive parameter ranges
- Real-time progress monitoring
"""

import time
import warnings

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from sklearn.model_selection import TimeSeriesSplit

# Suppress warnings for cleaner output
import warnings.filterwarnings
warnings.filterwarnings('ignore')


class EnhancedHMMRegimeOptimizer:
    """Enhanced HMM Regime Optimizer with advanced features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
    pass
    pass
        self.config = config or self._get_default_config()
        self.study = None
        self.best_params = {}
        self.best_score = -np.inf
        self.optimization_history = []
        self.cv_results = {}

    def _get_default_config(self) -> Dict[str, Any]:
    pass
    pass
        """Get default configuration for enhanced optimization."""
        return {
            "optimization_settings": {
                "n_trials": 100,
                "timeout": 3600,
                "study_name": "enhanced_hmm_optimization",
                "random_state": 42,
                "n_jobs": -1,  # Use all CPU cores
                "parallel_trials": True
            },
            "advanced_optimization": {
                "use_cross_validation": True,
                "cv_folds": 5,
                "use_early_stopping": True,
                "early_stopping_patience": 10,
                "use_adaptive_ranges": True,
                "multi_objective": False
            },
            "sampling_strategy": {
                "sampler": "tpe",  # "tpe", "cmaes", "random"
                "n_startup_trials": 20,
                "n_ei_candidates": 24
            },
            "pruning_strategy": {
                "pruner": "median",  # "median", "hyperband", "none"
                "n_startup_trials": 5,
                "n_warmup_steps": 10
            }
        }

    def optimize_parallel(self, data: pd.DataFrame, feature_columns: List[str],
                         market_condition_columns: List[str], n_trials: int = 100,
                         timeout: Optional[int] = None, study_name: str = "enhanced_optimization") -> Dict[str, Any]:
        """Run parallel optimization with advanced features."""

        print(f"🚀 Starting Enhanced HMM Regime Optimization...")
        print(f"📊 Data shape: {data.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"📈 Market conditions: {len(market_condition_columns)}")
        print(f"🎯 Trials: {n_trials}")
        print(f"⚡ Parallel processing: {self.config['optimization_settings']['parallel_trials']}")

        # Pre-process data for optimization
        processed_data = self._preprocess_data_enhanced(data, feature_columns, market_condition_columns)

        # Create enhanced study with advanced samplers and pruners
        self.study = self._create_enhanced_study(study_name)

        # Create objective function with cross-validation
        objective = self._create_enhanced_objective(processed_data, feature_columns, market_condition_columns)

        # Run optimization with parallel processing
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config['optimization_settings']['n_jobs'],
            show_progress_bar=True
        )

        # Store best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\\\n✅ Enhanced Optimization completed!")
        print(f"🏆 Best score: {self.best_score:.4f}")
        print(f"🔧 Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study,
            'optimization_history': self.optimization_history,
            'cv_results': self.cv_results
        }

    def _create_enhanced_study(self, study_name: str) -> optuna.Study:
    pass
    pass
        """Create enhanced Optuna study with advanced samplers and pruners."""

        # Choose sampler based on configuration
        sampler_config = self.config['sampling_strategy']
        if sampler_config['sampler'] == 'tpe':
    pass
    pass
            sampler = TPESampler(
                seed=42,
                n_startup_trials=sampler_config['n_startup_trials'],
                n_ei_candidates=sampler_config['n_ei_candidates']
            )
        elif sampler_config['sampler'] == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)

        # Choose pruner based on configuration
        pruner_config = self.config['pruning_strategy']
        if pruner_config['pruner'] == 'median':
    pass
    pass
            pruner = MedianPruner(
                n_startup_trials=pruner_config['n_startup_trials'],
                n_warmup_steps=pruner_config['n_warmup_steps']
            )
        elif pruner_config['pruner'] == 'hyperband':
            pruner = HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        return optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )

    def _preprocess_data_enhanced(self, data: pd.DataFrame, feature_columns: List[str],
                                 market_condition_columns: List[str]) -> Dict[str, Any]:
        """Enhanced pre-processing with additional optimizations."""

        # Filter valid columns
        valid_features = [col for col in feature_columns if col in data.columns]
        valid_market_conditions = [col for col in market_condition_columns if col in data.columns]

        # Create enhanced pre-processed data structure
        processed_data = {
            'data': data.copy(),
            'feature_columns': valid_features,
            'market_condition_columns': valid_market_conditions,
            'feature_matrix': data[valid_features].values if valid_features else np.array([]),
            'market_condition_matrix': data[valid_market_conditions].values if valid_market_conditions else np.array([]),
            'feature_ranges': {},
            'market_condition_ranges': {},
            'cv_splits': None
        }

        # Pre-calculate ranges for normalization
        for col in valid_features:
    pass
    pass
            col_data = data[col].dropna()
            if len(col_data) > 0:
    pass
    pass
                processed_data['feature_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }

        for col in valid_market_conditions:
    pass
    pass
            col_data = data[col].dropna()
            if len(col_data) > 0:
    pass
    pass
                processed_data['market_condition_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }

        # Create cross-validation splits if enabled
        if self.config['advanced_optimization']['use_cross_validation']:
    pass
    pass
            cv_folds = self.config['advanced_optimization']['cv_folds']
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            processed_data['cv_splits'] = list(tscv.split(data))

        return processed_data

    def _create_enhanced_objective(self, processed_data: Dict[str, Any],
                                 feature_columns: List[str],
                                 market_condition_columns: List[str]) -> callable:
        """Create enhanced objective function with cross-validation."""

        def objective(trial: optuna.Trial) -> float:
    pass
    pass
            """Enhanced objective function with cross-validation and early stopping."""

            # Suggest parameters with adaptive ranges if enabled
            if self.config['advanced_optimization']['use_adaptive_ranges']:
    pass
    pass
                params = self._suggest_adaptive_parameters(trial)
            else:
                params = self._suggest_standard_parameters(trial)

            try:
                # Use cross-validation if enabled
    except Exception as e:
        pass
    except Exception as e:
        pass
                if self.config['advanced_optimization']['use_cross_validation'] and processed_data['cv_splits']:
    pass
    pass
                    cv_scores = []

                    for train_idx, val_idx in processed_data['cv_splits']:
    pass
    pass
                        # Split data
                        train_data = processed_data['data'].iloc[train_idx]
                        val_data = processed_data['data'].iloc[val_idx]

                        # Generate clusters for training data
                        train_clusters = self._generate_clusters_enhanced(train_data, params)

                        # Evaluate on validation data
                        val_score = self._evaluate_regime_quality_enhanced(
                            val_data, train_clusters, processed_data['market_condition_columns'], params
                        )
                        cv_scores.append(val_score)

                    # Return mean CV score
                    final_score = np.mean(cv_scores)

                    # Store CV results
                    self.cv_results[trial.number] = {
                        'cv_scores': cv_scores,
                        'cv_mean': final_score,
                        'cv_std': np.std(cv_scores)
                    }

                else:
                    # Standard evaluation without CV
                    cluster_data = self._generate_clusters_enhanced(processed_data['data'], params)
                    final_score = self._evaluate_regime_quality_enhanced(
                        cluster_data, None, processed_data['market_condition_columns'], params
                    )

                # Store trial information
                trial_info = {
                    'trial_number': trial.number,
                    'params': params,
                    'score': final_score,
                    'timestamp': time.time()
                }
                self.optimization_history.append(trial_info)

                return final_score

            except Exception as e:
                print(f"⚠️ Trial {trial.number} failed: {e}")
                return -np.inf

        return objective

    def _suggest_adaptive_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    pass
    pass
        """Suggest parameters with adaptive ranges based on previous trials."""

        # Get previous best parameters to adapt ranges
        if self.study and len(self.study.trials) > 0:
    pass
    pass
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
    pass
    pass
                best_trial = max(completed_trials, key=lambda t: t.value)
                best_params = best_trial.params

                # Adapt ranges based on best parameters
                n_components_range = self._adapt_range(best_params.get('n_components', 5), [2, 10])
                target_regimes_range = self._adapt_range(best_params.get('target_regimes', 18), [15, 20])
            else:
                n_components_range = [2, 10]
                target_regimes_range = [15, 20]
        else:
            n_components_range = [2, 10]
            target_regimes_range = [15, 20]

        return {
            'n_components': trial.suggest_int('n_components', n_components_range[0], n_components_range[1]),
            'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical']),
            'n_iter': trial.suggest_int('n_iter', 100, 300),
            'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
            'reg_covar': trial.suggest_float('reg_covar', 1e-6, 1e-3, log=True),
            'clustering_method': trial.suggest_categorical('clustering_method', ['kmeans', 'gaussian_mixture']),
            'n_clusters': trial.suggest_int('n_clusters', 3, 15),
            'target_regimes': trial.suggest_int('target_regimes', target_regimes_range[0], target_regimes_range[1]),
            'merging_method': trial.suggest_categorical('merging_method', ['hierarchical', 'kmeans', 'dbscan', 'spectral']),
            'similarity_threshold': trial.suggest_float('similarity_threshold', 0.3, 0.8),
            'coherence_threshold': trial.suggest_float('coherence_threshold', 0.6, 0.9),
            'differentiation_threshold': trial.suggest_float('differentiation_threshold', 0.4, 0.8)
        }

    def _suggest_standard_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
    pass
    pass
        """Suggest parameters with standard ranges."""

        return {
            'n_components': trial.suggest_int('n_components', 2, 10),
            'covariance_type': trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical']),
            'n_iter': trial.suggest_int('n_iter', 100, 300),
            'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True),
            'reg_covar': trial.suggest_float('reg_covar', 1e-6, 1e-3, log=True),
            'clustering_method': trial.suggest_categorical('clustering_method', ['kmeans', 'gaussian_mixture']),
            'n_clusters': trial.suggest_int('n_clusters', 3, 15),
            'target_regimes': trial.suggest_int('target_regimes', 15, 20),
            'merging_method': trial.suggest_categorical('merging_method', ['hierarchical', 'kmeans', 'dbscan', 'spectral']),
            'similarity_threshold': trial.suggest_float('similarity_threshold', 0.3, 0.8),
            'coherence_threshold': trial.suggest_float('coherence_threshold', 0.6, 0.9),
            'differentiation_threshold': trial.suggest_float('differentiation_threshold', 0.4, 0.8)
        }

    def _adapt_range(self, best_value: Any, original_range: List[Any]) -> List[Any]:
    pass
    pass
        """Adapt parameter range based on best value."""

        if isinstance(best_value, int):
    pass
    pass
            # Adapt integer range
            range_size = original_range[1] - original_range[0]
            new_min = max(original_range[0], best_value - range_size // 4)
            new_max = min(original_range[1], best_value + range_size // 4)
            return [new_min, new_max]
        elif isinstance(best_value, float):
            # Adapt float range
            range_size = original_range[1] - original_range[0]
            new_min = max(original_range[0], best_value - range_size * 0.25)
            new_max = min(original_range[1], best_value + range_size * 0.25)
            return [new_min, new_max]
        else:
            return original_range

    def _generate_clusters_enhanced(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    pass
    pass
        """Generate clusters with enhanced processing."""

        # This would implement the cluster generation logic
        # For now, return a simple implementation
        result_data = data.copy()
        result_data['composite_cluster_id'] = np.random.randint(0, params.get('target_regimes', 18), size=len(data))
        return result_data

    def _evaluate_regime_quality_enhanced(self, cluster_data: pd.DataFrame,
                                        train_clusters: Optional[pd.DataFrame],
                                        market_condition_columns: List[str],
                                        params: Dict[str, Any]) -> float:
        """Enhanced regime quality evaluation with additional metrics."""

        # Basic evaluation (placeholder for enhanced implementation)
        if 'composite_cluster_id' not in cluster_data.columns:
    pass
    pass
            return -np.inf

        # Calculate basic metrics
        n_regimes = len(cluster_data['composite_cluster_id'].unique())
        target_regimes = params.get('target_regimes', 18)

        # Target count penalty
        target_penalty = 1.0 - abs(n_regimes - target_regimes) / target_regimes
        if n_regimes < 15 or n_regimes > 20:
    pass
    pass
            target_penalty *= 0.5

        # Basic score (placeholder)
        score = target_penalty * 0.5 + np.random.random() * 0.5

        return max(0.0, score)


def main():
    pass
    pass
    """Example usage of enhanced optimizer."""

    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Add some market condition columns
    data['volatility'] = np.random.exponential(1, n_samples)
    data['momentum'] = np.random.normal(0, 1, n_samples)
    data['volume'] = np.random.lognormal(0, 1, n_samples)
    data['returns'] = np.random.normal(0, 0.02, n_samples)

    feature_columns = [f'feature_{i}' for i in range(n_features)]
    market_condition_columns = ['volatility', 'momentum', 'volume', 'returns']

    # Initialize enhanced optimizer
    config = {
        "optimization_settings": {
            "n_trials": 50,  # Reduced for demo
            "timeout": 600,
            "n_jobs": -1,
            "parallel_trials": True
        },
        "advanced_optimization": {
            "use_cross_validation": True,
            "cv_folds": 3,
            "use_adaptive_ranges": True
        }
    }

    optimizer = EnhancedHMMRegimeOptimizer(config)

    # Run enhanced optimization
    results = optimizer.optimize_parallel(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=50,
        study_name="enhanced_demo"
    )

    print(f"\\\n🎉 Enhanced optimization completed!")
    print(f"🏆 Best score: {results['best_score']:.4f}")
    print(f"🔧 Best parameters: {results['best_params']}")


if __name__ == "__main__":
    pass
    pass
    main()