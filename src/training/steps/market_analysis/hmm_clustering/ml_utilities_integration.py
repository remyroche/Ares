#!/usr/bin/env python3
"""
ML Common Utilities Integration for HMM Clustering

This module provides comprehensive integration with ML common utilities
including cross-validation, hyperparameter optimization, and validation frameworks.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings

# Core dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# HMM dependencies
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    hmm = None

# Import ML common utilities
from src.utils.ml_common.validation.cross_validation import TimeSeriesCrossValidator
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.ml_common.hmm_regime_detection import HMMRegimeDetector
from src.utils.ml_common.data_processing.regime_processing import RegimeDataProcessor
from src.utils.ml_common.evaluation.evaluation_utils import ClusteringEvaluator
from src.utils.ml_common.models.ensemble_models import EnsembleModel
from src.utils.logger import system_logger

# Setup logging
logger = system_logger.getChild('MLUtilitiesIntegration')

@dataclass
class MLUtilitiesConfig:
    """Configuration for ML utilities integration."""
    # Cross-validation settings
    cv_method: str = 'timeseries'  # 'timeseries', 'kfold', 'stratified'
    n_splits: int = 5
    test_size: float = 0.2
    gap: int = 0  # Gap between train and test for time series
    
    # Hyperparameter optimization
    enable_hpo: bool = True
    hpo_method: str = 'bayesian'  # 'bayesian', 'grid', 'random'
    n_trials: int = 50
    timeout: int = 300
    scoring_metric: str = 'silhouette'  # 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    
    # Validation settings
    enable_validation: bool = True
    validation_threshold: float = 0.5
    enable_early_stopping: bool = True
    patience: int = 10
    
    # Ensemble settings
    enable_ensemble: bool = False
    ensemble_methods: List[str] = None  # Will be set to ['hmm', 'kmeans', 'gmm'] if None
    ensemble_weights: Optional[List[float]] = None
    
    # Performance settings
    enable_profiling: bool = False
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    
    # Data processing
    enable_regime_processing: bool = True
    regime_detection_method: str = 'hmm'  # 'hmm', 'kmeans', 'gmm'
    
    def __post_init__(self):
        if self.ensemble_methods is None:
            self.ensemble_methods = ['hmm', 'kmeans', 'gmm']

class MLUtilitiesIntegration:
    """
    ML utilities integration for HMM clustering.
    
    This class provides comprehensive integration with ML common utilities
    for advanced HMM clustering workflows.
    """
    
    def __init__(self, config: MLUtilitiesConfig):
        """Initialize ML utilities integration."""
        self.config = config
        self.logger = logger.getChild('MLUtilitiesIntegration')
        
        # Initialize ML utilities
        self.cv_validator = TimeSeriesCrossValidator()
        self.hpo_optimizer = HyperparameterOptimizer()
        self.hmm_regime_detector = HMMRegimeDetector()
        self.regime_processor = RegimeDataProcessor()
        self.clustering_evaluator = ClusteringEvaluator()
        self.ensemble_model = EnsembleModel()
        
        # Performance tracking
        self.performance_metrics = {}
        self.cv_scores = {}
        self.hpo_results = {}
        self.validation_results = {}
        
        # State
        self.is_fitted = False
        self.best_params = None
        self.best_model = None
        self.cv_results = None
        
        self.logger.info("🔧 ML Utilities Integration initialized")
        self._log_capabilities()
    
    def _log_capabilities(self):
        """Log available ML utilities capabilities."""
        self.logger.info("🔧 ML Utilities Capabilities:")
        self.logger.info(f"   Cross-Validation: {'✅ Available' if self.cv_validator else '❌ Not Available'}")
        self.logger.info(f"   Hyperparameter Optimization: {'✅ Available' if self.hpo_optimizer else '❌ Not Available'}")
        self.logger.info(f"   HMM Regime Detection: {'✅ Available' if self.hmm_regime_detector else '❌ Not Available'}")
        self.logger.info(f"   Regime Processing: {'✅ Available' if self.regime_processor else '❌ Not Available'}")
        self.logger.info(f"   Clustering Evaluation: {'✅ Available' if self.clustering_evaluator else '❌ Not Available'}")
        self.logger.info(f"   Ensemble Models: {'✅ Available' if self.ensemble_model else '❌ Not Available'}")
    
    def _create_cv_splitter(self) -> Any:
        """Create appropriate cross-validation splitter."""
        if self.config.cv_method == 'timeseries':
            return TimeSeriesSplit(
                n_splits=self.config.n_splits,
                test_size=int(self.config.test_size * 1000),  # Assuming 1000 samples
                gap=self.config.gap
            )
        elif self.config.cv_method == 'kfold':
            return KFold(n_splits=self.config.n_splits, shuffle=True, random_state=42)
        elif self.config.cv_method == 'stratified':
            return StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=42)
        else:
            return TimeSeriesSplit(n_splits=self.config.n_splits)
    
    def _create_scoring_function(self) -> Callable:
        """Create scoring function based on configuration."""
        def silhouette_scorer(estimator, X, y=None):
            try:
                labels = estimator.predict(X)
                if len(np.unique(labels)) > 1:
                    return silhouette_score(X, labels)
                else:
                    return 0.0
            except:
                return 0.0
        
        def calinski_harabasz_scorer(estimator, X, y=None):
            try:
                labels = estimator.predict(X)
                if len(np.unique(labels)) > 1:
                    return calinski_harabasz_score(X, labels)
                else:
                    return 0.0
            except:
                return 0.0
        
        def davies_bouldin_scorer(estimator, X, y=None):
            try:
                labels = estimator.predict(X)
                if len(np.unique(labels)) > 1:
                    return -davies_bouldin_score(X, labels)  # Negative because lower is better
                else:
                    return 0.0
            except:
                return 0.0
        
        if self.config.scoring_metric == 'silhouette':
            return silhouette_scorer
        elif self.config.scoring_metric == 'calinski_harabasz':
            return calinski_harabasz_scorer
        elif self.config.scoring_metric == 'davies_bouldin':
            return davies_bouldin_scorer
        else:
            return silhouette_scorer
    
    def _create_hmm_model(self, params: Dict[str, Any]) -> Any:
        """Create HMM model with given parameters."""
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn not available")
        
        return hmm.GaussianHMM(
            n_components=params.get('n_components', 3),
            covariance_type=params.get('covariance_type', 'full'),
            n_iter=params.get('n_iter', 100),
            random_state=params.get('random_state', 42)
        )
    
    def _create_param_grid(self) -> Dict[str, List]:
        """Create parameter grid for hyperparameter optimization."""
        return {
            'n_components': [2, 3, 4, 5, 6, 7, 8],
            'covariance_type': ['full', 'tied', 'diag'],
            'n_iter': [50, 100, 200, 500],
            'random_state': [42]
        }
    
    def run_cross_validation(self, data: np.ndarray, model_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run cross-validation for HMM clustering."""
        self.logger.info("🔄 Running cross-validation...")
        
        try:
            # Create model
            model = self._create_hmm_model(model_params)
            
            # Create CV splitter
            cv_splitter = self._create_cv_splitter()
            
            # Create scoring function
            scoring_func = self._create_scoring_function()
            
            # Run cross-validation
            start_time = time.time()
            
            if self.cv_validator and hasattr(self.cv_validator, 'cross_validate'):
                cv_results = self.cv_validator.cross_validate(
                    model, data, cv=cv_splitter, scoring=scoring_func
                )
            else:
                # Fallback cross-validation
                cv_results = self._fallback_cross_validation(model, data, cv_splitter, scoring_func)
            
            cv_time = time.time() - start_time
            
            # Store results
            self.cv_results = cv_results
            self.cv_scores = {
                'mean_score': np.mean(cv_results['test_score']),
                'std_score': np.std(cv_results['test_score']),
                'scores': cv_results['test_score'].tolist(),
                'cv_time': cv_time
            }
            
            self.logger.info(f"✅ Cross-validation completed in {cv_time:.3f}s")
            self.logger.info(f"   Mean score: {self.cv_scores['mean_score']:.3f} ± {self.cv_scores['std_score']:.3f}")
            
            return self.cv_scores
            
        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")
            raise
    
    def _fallback_cross_validation(self, model: Any, data: np.ndarray, 
                                 cv_splitter: Any, scoring_func: Callable) -> Dict[str, Any]:
        """Fallback cross-validation implementation."""
        scores = []
        fit_times = []
        score_times = []
        
        for train_idx, test_idx in cv_splitter.split(data):
            X_train, X_test = data[train_idx], data[test_idx]
            
            # Fit model
            fit_start = time.time()
            model.fit(X_train)
            fit_times.append(time.time() - fit_start)
            
            # Score model
            score_start = time.time()
            score = scoring_func(model, X_test)
            score_times.append(time.time() - score_start)
            
            scores.append(score)
        
        return {
            'test_score': np.array(scores),
            'fit_time': np.array(fit_times),
            'score_time': np.array(score_times)
        }
    
    def run_hyperparameter_optimization(self, data: np.ndarray) -> Dict[str, Any]:
        """Run hyperparameter optimization for HMM clustering."""
        if not self.config.enable_hpo:
            self.logger.info("⚠️ Hyperparameter optimization disabled")
            return {}
        
        self.logger.info("🔧 Running hyperparameter optimization...")
        
        try:
            # Create parameter grid
            param_grid = self._create_param_grid()
            
            # Create scoring function
            scoring_func = self._create_scoring_function()
            
            # Create CV splitter
            cv_splitter = self._create_cv_splitter()
            
            # Run optimization
            start_time = time.time()
            
            if self.hpo_optimizer and hasattr(self.hpo_optimizer, 'optimize'):
                best_params = self.hpo_optimizer.optimize(
                    model_class=hmm.GaussianHMM,
                    param_grid=param_grid,
                    X=data,
                    cv=cv_splitter,
                    scoring=scoring_func,
                    n_trials=self.config.n_trials,
                    timeout=self.config.timeout
                )
            else:
                # Fallback optimization
                best_params = self._fallback_hyperparameter_optimization(
                    data, param_grid, cv_splitter, scoring_func
                )
            
            hpo_time = time.time() - start_time
            
            # Store results
            self.hpo_results = {
                'best_params': best_params,
                'hpo_time': hpo_time,
                'method': self.config.hpo_method
            }
            self.best_params = best_params
            
            self.logger.info(f"✅ Hyperparameter optimization completed in {hpo_time:.3f}s")
            self.logger.info(f"   Best parameters: {best_params}")
            
            return self.hpo_results
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            raise
    
    def _fallback_hyperparameter_optimization(self, data: np.ndarray, param_grid: Dict[str, List],
                                            cv_splitter: Any, scoring_func: Callable) -> Dict[str, Any]:
        """Fallback hyperparameter optimization implementation."""
        best_score = -np.inf
        best_params = {}
        
        # Simple grid search
        from itertools import product
        
        param_combinations = list(product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        for i, param_combo in enumerate(param_combinations[:self.config.n_trials]):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create model
                model = self._create_hmm_model(params)
                
                # Run cross-validation
                cv_results = self._fallback_cross_validation(model, data, cv_splitter, scoring_func)
                mean_score = np.mean(cv_results['test_score'])
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                
            except Exception as e:
                self.logger.warning(f"⚠️ Parameter combination failed: {params}, {e}")
                continue
        
        return best_params
    
    def run_regime_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Run regime detection using HMM."""
        if not self.config.enable_regime_processing:
            self.logger.info("⚠️ Regime processing disabled")
            return {}
        
        self.logger.info("🎯 Running regime detection...")
        
        try:
            start_time = time.time()
            
            if self.hmm_regime_detector and hasattr(self.hmm_regime_detector, 'detect_regimes'):
                regime_results = self.hmm_regime_detector.detect_regimes(
                    data, method=self.config.regime_detection_method
                )
            else:
                # Fallback regime detection
                regime_results = self._fallback_regime_detection(data)
            
            regime_time = time.time() - start_time
            
            self.logger.info(f"✅ Regime detection completed in {regime_time:.3f}s")
            
            return {
                'regime_results': regime_results,
                'regime_time': regime_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Regime detection failed: {e}")
            raise
    
    def _fallback_regime_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback regime detection implementation."""
        # Use K-means as fallback
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(data)
        
        return {
            'labels': labels,
            'n_regimes': len(np.unique(labels)),
            'regime_centers': kmeans.cluster_centers_
        }
    
    def run_ensemble_clustering(self, data: np.ndarray) -> Dict[str, Any]:
        """Run ensemble clustering combining multiple methods."""
        if not self.config.enable_ensemble:
            self.logger.info("⚠️ Ensemble clustering disabled")
            return {}
        
        self.logger.info("🎭 Running ensemble clustering...")
        
        try:
            start_time = time.time()
            
            # Create ensemble models
            ensemble_models = {}
            
            # HMM model
            if 'hmm' in self.config.ensemble_methods and HMM_AVAILABLE:
                hmm_model = hmm.GaussianHMM(n_components=3, random_state=42)
                hmm_model.fit(data)
                ensemble_models['hmm'] = hmm_model
            
            # K-means model
            if 'kmeans' in self.config.ensemble_methods:
                kmeans_model = KMeans(n_clusters=3, random_state=42)
                kmeans_model.fit(data)
                ensemble_models['kmeans'] = kmeans_model
            
            # GMM model
            if 'gmm' in self.config.ensemble_methods:
                from sklearn.mixture import GaussianMixture
                gmm_model = GaussianMixture(n_components=3, random_state=42)
                gmm_model.fit(data)
                ensemble_models['gmm'] = gmm_model
            
            # Get predictions from all models
            predictions = {}
            for name, model in ensemble_models.items():
                if hasattr(model, 'predict'):
                    predictions[name] = model.predict(data)
                elif hasattr(model, 'labels_'):
                    predictions[name] = model.labels_
            
            # Combine predictions (simple voting)
            ensemble_labels = self._combine_ensemble_predictions(predictions)
            
            ensemble_time = time.time() - start_time
            
            self.logger.info(f"✅ Ensemble clustering completed in {ensemble_time:.3f}s")
            
            return {
                'ensemble_labels': ensemble_labels,
                'individual_predictions': predictions,
                'ensemble_time': ensemble_time,
                'n_models': len(ensemble_models)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble clustering failed: {e}")
            raise
    
    def _combine_ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine ensemble predictions using voting."""
        if not predictions:
            return np.array([])
        
        # Get the first prediction as base
        base_pred = list(predictions.values())[0]
        n_samples = len(base_pred)
        
        # Create voting matrix
        voting_matrix = np.zeros((n_samples, max(max(pred) for pred in predictions.values()) + 1))
        
        for pred in predictions.values():
            for i, label in enumerate(pred):
                voting_matrix[i, label] += 1
        
        # Get majority vote
        ensemble_labels = np.argmax(voting_matrix, axis=1)
        
        return ensemble_labels
    
    def evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate clustering quality using multiple metrics."""
        self.logger.info("📊 Evaluating clustering quality...")
        
        try:
            if self.clustering_evaluator and hasattr(self.clustering_evaluator, 'evaluate'):
                evaluation_results = self.clustering_evaluator.evaluate(data, labels)
            else:
                # Fallback evaluation
                evaluation_results = self._fallback_clustering_evaluation(data, labels)
            
            self.logger.info("✅ Clustering evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Clustering evaluation failed: {e}")
            raise
    
    def _fallback_clustering_evaluation(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Fallback clustering evaluation implementation."""
        metrics = {}
        
        try:
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(data, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data, labels)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['calinski_harabasz_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
            
            metrics['n_clusters'] = len(np.unique(labels))
            metrics['n_samples'] = len(labels)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Clustering evaluation failed: {e}")
            metrics = {'error': str(e)}
        
        return metrics
    
    def run_comprehensive_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Run comprehensive ML analysis including all utilities."""
        self.logger.info("🚀 Running comprehensive ML analysis...")
        
        try:
            results = {}
            
            # Run hyperparameter optimization
            if self.config.enable_hpo:
                hpo_results = self.run_hyperparameter_optimization(data)
                results['hyperparameter_optimization'] = hpo_results
                
                # Use best parameters for subsequent analysis
                if self.best_params:
                    model_params = self.best_params
                else:
                    model_params = {'n_components': 3, 'covariance_type': 'full', 'n_iter': 100}
            else:
                model_params = {'n_components': 3, 'covariance_type': 'full', 'n_iter': 100}
            
            # Run cross-validation
            cv_results = self.run_cross_validation(data, model_params)
            results['cross_validation'] = cv_results
            
            # Run regime detection
            regime_results = self.run_regime_detection(data)
            results['regime_detection'] = regime_results
            
            # Run ensemble clustering
            ensemble_results = self.run_ensemble_clustering(data)
            results['ensemble_clustering'] = ensemble_results
            
            # Train final model
            final_model = self._create_hmm_model(model_params)
            final_model.fit(data)
            final_labels = final_model.predict(data)
            
            # Evaluate final model
            evaluation_results = self.evaluate_clustering(data, final_labels)
            results['final_evaluation'] = evaluation_results
            
            # Store final model
            self.best_model = final_model
            self.is_fitted = True
            
            self.logger.info("✅ Comprehensive ML analysis completed!")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive ML analysis failed: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'is_fitted': self.is_fitted,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'hpo_results': self.hpo_results,
            'validation_results': self.validation_results,
            'performance_metrics': self.performance_metrics,
            'config': self.config.__dict__
        }


def create_ml_utilities_integration(config: Optional[MLUtilitiesConfig] = None) -> MLUtilitiesIntegration:
    """Factory function to create ML utilities integration instance."""
    if config is None:
        config = MLUtilitiesConfig()
    
    return MLUtilitiesIntegration(config)


# Example usage
if __name__ == "__main__":
    # Example usage
    logger.info("🔧 ML Utilities Integration Example")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Generate sample data with 3 clusters
    cluster1 = np.random.multivariate_normal([0, 0, 0, 0, 0], np.eye(5), n_samples // 3)
    cluster2 = np.random.multivariate_normal([3, 3, 3, 3, 3], np.eye(5), n_samples // 3)
    cluster3 = np.random.multivariate_normal([-3, -3, -3, -3, -3], np.eye(5), n_samples - 2 * (n_samples // 3))
    
    sample_data = np.vstack([cluster1, cluster2, cluster3])
    
    # Create configuration
    config = MLUtilitiesConfig(
        enable_hpo=True,
        hpo_method='bayesian',
        n_trials=20,
        enable_ensemble=True,
        ensemble_methods=['hmm', 'kmeans'],
        enable_regime_processing=True
    )
    
    # Create and run ML utilities integration
    ml_utils = create_ml_utilities_integration(config)
    results = ml_utils.run_comprehensive_analysis(sample_data)
    
    # Print results
    print("📊 ML Utilities Integration Results:")
    print(f"Cross-validation scores: {results['cross_validation']}")
    print(f"Hyperparameter optimization: {results['hyperparameter_optimization']}")
    print(f"Final evaluation: {results['final_evaluation']}")
    
    # Get performance summary
    summary = ml_utils.get_performance_summary()
    print(f"Performance summary: {summary}")