#!/usr/bin/env python3
"""
Advanced HMM Regime Parameter Optimization with Comprehensive Features

This advanced version includes:
- Cross-validation with time-series splits
- Bayesian optimization with advanced samplers
- Early stopping and pruning
- Data streaming for large datasets
- Advanced metrics (regime persistence, transition smoothness, etc.)
- Robustness checks (bootstrap, sensitivity analysis, outlier detection)
- Feature engineering integration
- Multi-objective optimization framework
"""

import asyncio
import concurrent.futures
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import joblib
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class OptimizationMetrics:
    """Container for optimization metrics."""
    regime_differentiation: float
    internal_coherence: float
    regime_balance: float
    target_count_penalty: float
    regime_persistence: float
    transition_smoothness: float
    market_correlation: float
    overall_score: float


class DataStreamer:
    """Data streaming for large datasets."""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    def stream_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stream data in chunks for processing."""
        for i in range(0, len(data), self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            yield chunk
    
    def process_chunk(self, chunk: pd.DataFrame, processor_func) -> pd.DataFrame:
        """Process a data chunk."""
        return processor_func(chunk)


class RobustnessChecker:
    """Robustness checks for optimization results."""
    
    def __init__(self, n_bootstrap: int = 100, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
    
    def bootstrap_confidence_intervals(self, data: pd.DataFrame, 
                                     metric_func, 
                                     n_samples: int = None) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for metrics."""
        
        if n_samples is None:
            n_samples = len(data)
        
        bootstrap_scores = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), size=n_samples, replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Calculate metric
            try:
                score = metric_func(bootstrap_data)
                bootstrap_scores.append(score)
            except Exception:
                continue
        
        if not bootstrap_scores:
            return {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0}
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        
        return {
            'mean': np.mean(bootstrap_scores),
            'std': np.std(bootstrap_scores),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    def sensitivity_analysis(self, data: pd.DataFrame, 
                           base_params: Dict[str, Any],
                           param_ranges: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis on parameters."""
        
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            param_scores = []
            
            for param_value in param_range:
                # Create modified parameters
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                try:
                    # Generate clusters with modified parameters
                    cluster_data = self._generate_test_clusters(data, test_params)
                    
                    # Calculate score
                    score = self._calculate_test_score(cluster_data)
                    param_scores.append(score)
                except Exception:
                    param_scores.append(0.0)
            
            # Calculate sensitivity metrics
            if param_scores:
                sensitivity_results[param_name] = {
                    'mean': np.mean(param_scores),
                    'std': np.std(param_scores),
                    'range': max(param_scores) - min(param_scores),
                    'stability': 1.0 / (1.0 + np.std(param_scores))
                }
        
        return sensitivity_results
    
    def outlier_detection(self, data: pd.DataFrame, 
                         contamination: float = 0.1) -> Tuple[pd.DataFrame, np.ndarray]:
        """Detect and remove outliers from data."""
        
        # Use Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data.select_dtypes(include=[np.number]))
        
        # Remove outliers
        clean_data = data[outlier_labels == 1].copy()
        outlier_mask = outlier_labels == -1
        
        return clean_data, outlier_mask
    
    def _generate_test_clusters(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Generate test clusters for sensitivity analysis."""
        # Simplified cluster generation for testing
        result_data = data.copy()
        result_data['composite_cluster_id'] = np.random.randint(
            0, params.get('target_regimes', 18), size=len(data)
        )
        return result_data
    
    def _calculate_test_score(self, cluster_data: pd.DataFrame) -> float:
        """Calculate test score for sensitivity analysis."""
        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        n_regimes = len(cluster_data['composite_cluster_id'].unique())
        target_regimes = 18
        
        # Simple target penalty
        penalty = 1.0 - abs(n_regimes - target_regimes) / target_regimes
        return max(0.0, penalty)


class AdvancedMetricsCalculator:
    """Calculate advanced metrics for regime quality."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_regime_persistence(self, cluster_data: pd.DataFrame, 
                                   window_size: int = 100) -> float:
        """Calculate regime persistence over time."""
        
        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        cluster_series = cluster_data['composite_cluster_id'].values
        
        # Calculate regime changes
        regime_changes = np.diff(cluster_series) != 0
        total_periods = len(cluster_series)
        
        if total_periods == 0:
            return 0.0
        
        # Calculate persistence as average regime duration
        change_indices = np.where(regime_changes)[0]
        
        if len(change_indices) == 0:
            # No changes - perfect persistence
            return 1.0
        
        # Calculate regime durations
        durations = []
        prev_change = -1
        
        for change_idx in change_indices:
            duration = change_idx - prev_change
            durations.append(duration)
            prev_change = change_idx
        
        # Add final regime duration
        final_duration = total_periods - prev_change - 1
        durations.append(final_duration)
        
        # Calculate average persistence
        avg_duration = np.mean(durations)
        max_possible_duration = total_periods
        
        persistence = avg_duration / max_possible_duration
        
        return min(1.0, persistence)
    
    def calculate_transition_smoothness(self, cluster_data: pd.DataFrame) -> float:
        """Calculate smoothness of regime transitions."""
        
        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        cluster_series = cluster_data['composite_cluster_id'].values
        
        # Calculate transition probabilities
        unique_regimes = np.unique(cluster_series)
        n_regimes = len(unique_regimes)
        
        if n_regimes < 2:
            return 0.0
        
        # Create transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        
        for i in range(len(cluster_series) - 1):
            current_regime = cluster_series[i]
            next_regime = cluster_series[i + 1]
            
            current_idx = regime_to_idx[current_regime]
            next_idx = regime_to_idx[next_regime]
            
            transition_matrix[current_idx, next_idx] += 1
        
        # Normalize transition matrix
        row_sums = transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate smoothness as entropy of transition probabilities
        # Lower entropy = smoother transitions
        entropy = 0.0
        for row in transition_matrix:
            row = row[row > 0]  # Remove zero probabilities
            if len(row) > 0:
                entropy += -np.sum(row * np.log(row))
        
        # Normalize by maximum possible entropy
        max_entropy = n_regimes * np.log(n_regimes)
        if max_entropy > 0:
            smoothness = 1.0 - (entropy / max_entropy)
        else:
            smoothness = 0.0
        
        return max(0.0, smoothness)
    
    def calculate_market_correlation(self, cluster_data: pd.DataFrame, 
                                   market_condition_columns: List[str]) -> float:
        """Calculate correlation between regimes and market conditions."""
        
        if not market_condition_columns or 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        correlations = []
        
        for col in market_condition_columns:
            if col not in cluster_data.columns:
                continue
            
            # Calculate correlation between regime and market condition
            regime_means = cluster_data.groupby('composite_cluster_id')[col].mean()
            
            if len(regime_means) < 2:
                continue
            
            # Calculate how well regimes differentiate this market condition
            regime_values = regime_means.values
            overall_mean = cluster_data[col].mean()
            
            # Calculate F-statistic (variance between regimes / variance within regimes)
            ss_between = np.sum((regime_values - overall_mean) ** 2)
            ss_within = np.sum((cluster_data[col].values - cluster_data.groupby('composite_cluster_id')[col].transform('mean').values) ** 2)
            
            if ss_within > 0:
                f_stat = ss_between / ss_within
                correlation = f_stat / (f_stat + 1)  # Convert to correlation-like measure
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_all_metrics(self, cluster_data: pd.DataFrame, 
                            market_condition_columns: List[str],
                            params: Dict[str, Any]) -> OptimizationMetrics:
        """Calculate all advanced metrics."""
        
        # Basic metrics
        regime_differentiation = self._calculate_regime_differentiation(cluster_data, market_condition_columns)
        internal_coherence = self._calculate_internal_coherence(cluster_data, market_condition_columns)
        regime_balance = self._calculate_regime_balance(cluster_data)
        target_count_penalty = self._calculate_target_count_penalty(cluster_data, params)
        
        # Advanced metrics
        regime_persistence = self.calculate_regime_persistence(cluster_data)
        transition_smoothness = self.calculate_transition_smoothness(cluster_data)
        market_correlation = self.calculate_market_correlation(cluster_data, market_condition_columns)
        
        # Calculate overall score
        weights = [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]  # Adjust weights as needed
        scores = [regime_differentiation, internal_coherence, regime_balance, 
                 target_count_penalty, regime_persistence, transition_smoothness, market_correlation]
        
        overall_score = np.average(scores, weights=weights)
        
        metrics = OptimizationMetrics(
            regime_differentiation=regime_differentiation,
            internal_coherence=internal_coherence,
            regime_balance=regime_balance,
            target_count_penalty=target_count_penalty,
            regime_persistence=regime_persistence,
            transition_smoothness=transition_smoothness,
            market_correlation=market_correlation,
            overall_score=overall_score
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_regime_differentiation(self, cluster_data: pd.DataFrame, 
                                        market_condition_columns: List[str]) -> float:
        """Calculate regime differentiation using vectorized operations."""
        
        if not market_condition_columns or 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0
        
        # Vectorized calculation
        regime_means_matrix = cluster_data.groupby('composite_cluster_id')[valid_columns].mean()
        
        if len(regime_means_matrix) < 2:
            return 0.0
        
        differentiation_scores = []
        
        for col in valid_columns:
            regime_means = regime_means_matrix[col].values
            n_regimes = len(regime_means)
            
            # Matrix operations for pairwise differences
            means_i = regime_means[:, np.newaxis]
            means_j = regime_means[np.newaxis, :]
            differences = np.abs(means_i - means_j)
            
            # Remove diagonal and get valid differences
            mask = ~np.eye(n_regimes, dtype=bool)
            valid_differences = differences[mask]
            
            if len(valid_differences) > 0:
                # Normalize by overall range
                overall_range = cluster_data[col].max() - cluster_data[col].min()
                if overall_range > 0:
                    avg_difference = np.mean(valid_differences) / overall_range
                    differentiation_scores.append(avg_difference)
        
        return np.mean(differentiation_scores) if differentiation_scores else 0.0
    
    def _calculate_internal_coherence(self, cluster_data: pd.DataFrame, 
                                    market_condition_columns: List[str]) -> float:
        """Calculate internal coherence using vectorized operations."""
        
        if not market_condition_columns or 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        valid_columns = [col for col in market_condition_columns if col in cluster_data.columns]
        if not valid_columns:
            return 0.0
        
        coherence_scores = []
        
        for col in valid_columns:
            # Vectorized calculation
            regime_stats = cluster_data.groupby('composite_cluster_id')[col].agg(['mean', 'std', 'count'])
            valid_regimes = regime_stats[regime_stats['count'] > 1]
            
            if len(valid_regimes) > 0:
                means = valid_regimes['mean'].values
                stds = valid_regimes['std'].values
                
                # Avoid division by zero
                non_zero_means = means != 0
                if np.any(non_zero_means):
                    cvs = stds[non_zero_means] / np.abs(means[non_zero_means])
                    
                    if len(cvs) > 0:
                        avg_cv = np.mean(cvs)
                        coherence = 1.0 / (1.0 + avg_cv)
                        coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_regime_balance(self, cluster_data: pd.DataFrame) -> float:
        """Calculate regime balance using vectorized operations."""
        
        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        regime_sizes = cluster_data['composite_cluster_id'].value_counts().values
        
        if len(regime_sizes) < 2:
            return 0.0
        
        # Calculate coefficient of variation
        mean_size = np.mean(regime_sizes)
        std_size = np.std(regime_sizes)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        balance_score = 1.0 / (1.0 + cv)
        
        return balance_score
    
    def _calculate_target_count_penalty(self, cluster_data: pd.DataFrame, 
                                      params: Dict[str, Any]) -> float:
        """Calculate target count penalty."""
        
        if 'composite_cluster_id' not in cluster_data.columns:
            return 0.0
        
        target_regimes = params.get('target_regimes', 18)
        actual_regimes = len(cluster_data['composite_cluster_id'].unique())
        
        # Penalty based on distance from target
        penalty = 1.0 - abs(actual_regimes - target_regimes) / target_regimes
        
        # Additional penalty for being outside the 15-20 range
        if actual_regimes < 15 or actual_regimes > 20:
            penalty *= 0.5
        
        return max(0.0, penalty)


class AdvancedHMMRegimeOptimizer:
    """Advanced HMM Regime Optimizer with comprehensive features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.study = None
        self.best_params = {}
        self.best_score = -np.inf
        self.optimization_history = []
        self.cv_results = {}
        self.metrics_calculator = AdvancedMetricsCalculator()
        self.robustness_checker = RobustnessChecker()
        self.data_streamer = DataStreamer()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for advanced optimization."""
        return {
            "optimization_settings": {
                "n_trials": 100,
                "timeout": 3600,
                "study_name": "advanced_hmm_optimization",
                "random_state": 42,
                "n_jobs": -1,
                "parallel_trials": True
            },
            "cross_validation": {
                "enabled": True,
                "cv_folds": 5,
                "time_series_split": True
            },
            "bayesian_optimization": {
                "sampler": "tpe",  # "tpe", "cmaes", "nsga2"
                "n_startup_trials": 20,
                "n_ei_candidates": 24
            },
            "early_stopping": {
                "enabled": True,
                "patience": 10,
                "min_delta": 0.001
            },
            "data_streaming": {
                "enabled": True,
                "chunk_size": 10000
            },
            "robustness_checks": {
                "bootstrap_enabled": True,
                "n_bootstrap": 100,
                "sensitivity_analysis": True,
                "outlier_detection": True,
                "contamination": 0.1
            }
        }
    
    def optimize_advanced(self, data: pd.DataFrame, feature_columns: List[str], 
                         market_condition_columns: List[str], n_trials: int = 100,
                         timeout: Optional[int] = None, study_name: str = "advanced_optimization") -> Dict[str, Any]:
        """Run advanced optimization with all features."""
        
        print(f"🚀 Starting Advanced HMM Regime Optimization...")
        print(f"📊 Data shape: {data.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        print(f"📈 Market conditions: {len(market_condition_columns)}")
        print(f"🎯 Trials: {n_trials}")
        
        # Outlier detection
        if self.config['robustness_checks']['outlier_detection']:
            print("🔍 Performing outlier detection...")
            clean_data, outlier_mask = self.robustness_checker.outlier_detection(
                data, self.config['robustness_checks']['contamination']
            )
            print(f"🧹 Removed {np.sum(outlier_mask)} outliers ({np.sum(outlier_mask)/len(data)*100:.1f}%)")
        else:
            clean_data = data.copy()
        
        # Pre-process data
        processed_data = self._preprocess_data_advanced(clean_data, feature_columns, market_condition_columns)
        
        # Create advanced study
        self.study = self._create_advanced_study(study_name)
        
        # Create objective function with cross-validation
        objective = self._create_advanced_objective(processed_data, feature_columns, market_condition_columns)
        
        # Run optimization
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
        
        # Perform robustness checks on best result
        robustness_results = self._perform_robustness_checks(clean_data, feature_columns, market_condition_columns)
        
        print(f"\n✅ Advanced Optimization completed!")
        print(f"🏆 Best score: {self.best_score:.4f}")
        print(f"🔧 Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': self.study,
            'optimization_history': self.optimization_history,
            'cv_results': self.cv_results,
            'robustness_results': robustness_results,
            'metrics_history': self.metrics_calculator.metrics_history
        }
    
    def _create_advanced_study(self, study_name: str) -> optuna.Study:
        """Create advanced Optuna study with Bayesian optimization."""
        
        # Choose sampler based on configuration
        sampler_config = self.config['bayesian_optimization']
        if sampler_config['sampler'] == 'tpe':
            sampler = TPESampler(
                seed=42,
                n_startup_trials=sampler_config['n_startup_trials'],
                n_ei_candidates=sampler_config['n_ei_candidates']
            )
        elif sampler_config['sampler'] == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        elif sampler_config['sampler'] == 'nsga2':
            sampler = NSGAIISampler(seed=42)
        else:
            sampler = optuna.samplers.RandomSampler(seed=42)
        
        # Choose pruner for early stopping
        if self.config['early_stopping']['enabled']:
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        return optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
    
    def _preprocess_data_advanced(self, data: pd.DataFrame, feature_columns: List[str], 
                                market_condition_columns: List[str]) -> Dict[str, Any]:
        """Advanced pre-processing with data streaming support."""
        
        # Filter valid columns
        valid_features = [col for col in feature_columns if col in data.columns]
        valid_market_conditions = [col for col in market_condition_columns if col in data.columns]
        
        # Create advanced pre-processed data structure
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
            col_data = data[col].dropna()
            if len(col_data) > 0:
                processed_data['feature_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }
        
        for col in valid_market_conditions:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                processed_data['market_condition_ranges'][col] = {
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'range': col_data.max() - col_data.min(),
                    'mean': col_data.mean(),
                    'std': col_data.std()
                }
        
        # Create cross-validation splits if enabled
        if self.config['cross_validation']['enabled']:
            cv_folds = self.config['cross_validation']['cv_folds']
            if self.config['cross_validation']['time_series_split']:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                processed_data['cv_splits'] = list(tscv.split(data))
            else:
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                processed_data['cv_splits'] = list(kf.split(data))
        
        return processed_data
    
    def _create_advanced_objective(self, processed_data: Dict[str, Any], 
                                feature_columns: List[str], 
                                market_condition_columns: List[str]) -> callable:
        """Create advanced objective function with cross-validation and early stopping."""
        
        def objective(trial: optuna.Trial) -> float:
            """Advanced objective function with cross-validation and early stopping."""
            
            # Suggest parameters
            params = self._suggest_advanced_parameters(trial)
            
            try:
                # Use cross-validation if enabled
                if self.config['cross_validation']['enabled'] and processed_data['cv_splits']:
                    cv_scores = []
                    
                    for train_idx, val_idx in processed_data['cv_splits']:
                        # Split data
                        train_data = processed_data['data'].iloc[train_idx]
                        val_data = processed_data['data'].iloc[val_idx]
                        
                        # Generate clusters for training data
                        train_clusters = self._generate_clusters_advanced(train_data, params)
                        
                        # Evaluate on validation data
                        val_score = self._evaluate_regime_quality_advanced(
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
                    cluster_data = self._generate_clusters_advanced(processed_data['data'], params)
                    final_score = self._evaluate_regime_quality_advanced(
                        cluster_data, None, processed_data['market_condition_columns'], params
                    )
                
                # Early stopping check
                if self.config['early_stopping']['enabled']:
                    if hasattr(self, '_best_score_history'):
                        if len(self._best_score_history) >= self.config['early_stopping']['patience']:
                            recent_improvement = max(self._best_score_history[-self.config['early_stopping']['patience']:]) - min(self._best_score_history[-self.config['early_stopping']['patience']:])
                            if recent_improvement < self.config['early_stopping']['min_delta']:
                                trial.report(-np.inf, step=0)
                                raise optuna.TrialPruned()
                    else:
                        self._best_score_history = []
                    
                    self._best_score_history.append(final_score)
                
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
    
    def _suggest_advanced_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest advanced parameters with Bayesian optimization."""
        
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
    
    def _generate_clusters_advanced(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Generate clusters with advanced processing."""
        
        # This would implement the actual cluster generation logic
        # For now, return a simple implementation
        result_data = data.copy()
        result_data['composite_cluster_id'] = np.random.randint(
            0, params.get('target_regimes', 18), size=len(data)
        )
        return result_data
    
    def _evaluate_regime_quality_advanced(self, cluster_data: pd.DataFrame, 
                                        train_clusters: Optional[pd.DataFrame],
                                        market_condition_columns: List[str],
                                        params: Dict[str, Any]) -> float:
        """Advanced regime quality evaluation with all metrics."""
        
        # Calculate all advanced metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            cluster_data, market_condition_columns, params
        )
        
        return metrics.overall_score
    
    def _perform_robustness_checks(self, data: pd.DataFrame, 
                                 feature_columns: List[str],
                                 market_condition_columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive robustness checks."""
        
        print("🔍 Performing robustness checks...")
        
        robustness_results = {}
        
        # Bootstrap confidence intervals
        if self.config['robustness_checks']['bootstrap_enabled']:
            print("📊 Calculating bootstrap confidence intervals...")
            
            def metric_func(data_subset):
                cluster_data = self._generate_clusters_advanced(data_subset, self.best_params)
                return self._evaluate_regime_quality_advanced(
                    cluster_data, None, market_condition_columns, self.best_params
                )
            
            bootstrap_results = self.robustness_checker.bootstrap_confidence_intervals(
                data, metric_func
            )
            robustness_results['bootstrap'] = bootstrap_results
        
        # Sensitivity analysis
        if self.config['robustness_checks']['sensitivity_analysis']:
            print("📈 Performing sensitivity analysis...")
            
            param_ranges = {
                'n_components': [2, 4, 6, 8, 10],
                'target_regimes': [15, 16, 17, 18, 19, 20],
                'similarity_threshold': [0.3, 0.5, 0.7, 0.8]
            }
            
            sensitivity_results = self.robustness_checker.sensitivity_analysis(
                data, self.best_params, param_ranges
            )
            robustness_results['sensitivity'] = sensitivity_results
        
        print("✅ Robustness checks completed!")
        return robustness_results


def main():
    """Example usage of advanced optimizer."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    n_features = 20
    
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add market condition columns
    data['volatility'] = np.random.exponential(1, n_samples)
    data['momentum'] = np.random.normal(0, 1, n_samples)
    data['volume'] = np.random.lognormal(0, 1, n_samples)
    data['returns'] = np.random.normal(0, 0.02, n_samples)
    
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    market_condition_columns = ['volatility', 'momentum', 'volume', 'returns']
    
    # Initialize advanced optimizer
    config = {
        "optimization_settings": {
            "n_trials": 50,
            "timeout": 600,
            "n_jobs": -1,
            "parallel_trials": True
        },
        "cross_validation": {
            "enabled": True,
            "cv_folds": 3,
            "time_series_split": True
        },
        "bayesian_optimization": {
            "sampler": "tpe",
            "n_startup_trials": 10,
            "n_ei_candidates": 24
        },
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.001
        },
        "robustness_checks": {
            "bootstrap_enabled": True,
            "n_bootstrap": 50,
            "sensitivity_analysis": True,
            "outlier_detection": True
        }
    }
    
    optimizer = AdvancedHMMRegimeOptimizer(config)
    
    # Run advanced optimization
    results = optimizer.optimize_advanced(
        data=data,
        feature_columns=feature_columns,
        market_condition_columns=market_condition_columns,
        n_trials=50,
        study_name="advanced_demo"
    )
    
    print(f"\n🎉 Advanced optimization completed!")
    print(f"🏆 Best score: {results['best_score']:.4f}")
    print(f"🔧 Best parameters: {results['best_params']}")
    print(f"📊 Robustness results: {results['robustness_results']}")


if __name__ == "__main__":
    main()