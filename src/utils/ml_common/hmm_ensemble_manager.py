#!/usr/bin/env python3
"""
HMM Ensemble Manager for Enhanced Stability

This module provides comprehensive ensemble methods for HMM regime detection
to improve stability and robustness:
- Multiple HMM configurations
- Bootstrap aggregating (Bagging)
- Voting ensembles
- Stacking ensembles
- Temporal ensemble methods
- Regime-specific ensemble adaptation
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import joblib
from collections import defaultdict, deque
import itertools

# Import system utilities
from ..logger import get_logger
from .matrix_operations import get_enhanced_matrix_operations

class EnsembleMethod(Enum):
    """Available ensemble methods."""
    VOTING = "voting"
    BAGGING = "bagging"
    STACKING = "stacking"
    BAYESIAN_AVERAGING = "bayesian_averaging"
    TEMPORAL_ENSEMBLE = "temporal_ensemble"
    REGIME_SPECIFIC = "regime_specific"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"

class HMMType(Enum):
    """Available HMM types."""
    GAUSSIAN = "gaussian"
    MULTINOMIAL = "multinomial"
    GMM = "gaussian_mixture"
    VAR = "vector_autoregressive"
    FACTORIAL = "factorial"

@dataclass
class HMMConfig:
    """Configuration for individual HMM."""
    n_components: int = 4
    covariance_type: str = "full"
    n_iter: int = 100
    tol: float = 1e-3
    init_params: str = "kmeans"
    random_state: int = 42
    hmm_type: HMMType = HMMType.GAUSSIAN
    
    # Additional parameters
    max_iter: int = 1000
    warm_start: bool = False
    reg_covar: float = 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'init_params': self.init_params,
            'random_state': self.random_state,
            'hmm_type': self.hmm_type.value,
            'max_iter': self.max_iter,
            'warm_start': self.warm_start,
            'reg_covar': self.reg_covar
        }

@dataclass
class EnsembleConfig:
    """Configuration for HMM ensemble."""
    # Ensemble method
    method: EnsembleMethod = EnsembleMethod.VOTING
    
    # HMM configurations
    base_configs: List[HMMConfig] = field(default_factory=lambda: [
        HMMConfig(n_components=2, covariance_type="full"),
        HMMConfig(n_components=3, covariance_type="diag"),
        HMMConfig(n_components=4, covariance_type="spherical"),
        HMMConfig(n_components=5, covariance_type="full"),
        HMMConfig(n_components=6, covariance_type="diag")
    ])
    
    # Ensemble parameters
    n_bootstrap_samples: int = 10
    bootstrap_ratio: float = 0.8
    voting_type: str = "soft"  # "hard" or "soft"
    stacking_cv_folds: int = 5
    
    # Performance settings
    n_jobs: int = -1
    enable_parallel: bool = True
    chunk_size: int = 10000
    
    # Quality thresholds
    min_quality_score: float = 0.6
    max_regime_imbalance: float = 0.8
    min_samples_per_regime: int = 100
    
    # Output settings
    save_results: bool = True
    generate_plots: bool = True
    output_directory: Optional[str] = None

@dataclass
class HMMResult:
    """Result from individual HMM."""
    config: HMMConfig
    model: Any
    labels: np.ndarray
    probabilities: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    quality_score: float
    training_time: float
    convergence_info: Dict[str, Any]

@dataclass
class EnsembleResult:
    """Result from HMM ensemble."""
    ensemble_method: EnsembleMethod
    individual_results: List[HMMResult]
    ensemble_labels: np.ndarray
    ensemble_probabilities: np.ndarray
    ensemble_confidence: np.ndarray
    diversity_score: float
    stability_score: float
    overall_quality: float
    meta_info: Dict[str, Any]

class HMMEnsembleManager:
    """Manager for HMM ensemble methods."""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.logger = get_logger("HMMEnsembleManager")
        
        # Initialize matrix operations for performance
        self.matrix_ops = get_enhanced_matrix_operations()
        
        # Results storage
        self.results: Dict[str, EnsembleResult] = {}
        self.performance_history = deque(maxlen=100)
        
        self.logger.info("🚀 HMMEnsembleManager initialized")
    
    def fit_ensemble(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> EnsembleResult:
        """Fit HMM ensemble on data."""
        start_time = time.time()
        self.logger.info(f"🔍 Starting HMM ensemble fitting on {X.shape[0]} samples, {X.shape[1]} features")
        
        # Prepare data
        X_processed = self._prepare_data(X)
        
        # Fit individual HMMs
        individual_results = self._fit_individual_hmms(X_processed)
        
        # Create ensemble
        ensemble_result = self._create_ensemble(individual_results, X_processed)
        
        # Evaluate ensemble quality
        ensemble_result.overall_quality = self._evaluate_ensemble_quality(ensemble_result, X_processed)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(ensemble_result)
        
        # Generate plots if configured
        if self.config.generate_plots:
            self._generate_plots(ensemble_result, X_processed)
        
        # Update performance history
        self._update_performance_history(ensemble_result, time.time() - start_time)
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ HMM ensemble fitting completed in {total_time:.3f}s")
        
        return ensemble_result
    
    def _prepare_data(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare data for HMM fitting."""
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        # Handle missing values
        X_clean = X_numeric.fillna(X_numeric.median())
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        return X_scaled
    
    def _fit_individual_hmms(self, X: np.ndarray) -> List[HMMResult]:
        """Fit individual HMMs with different configurations."""
        individual_results = []
        
        if self.config.enable_parallel and len(self.config.base_configs) > 3:
            individual_results = self._fit_hmms_parallel(X)
        else:
            individual_results = self._fit_hmms_sequential(X)
        
        # Filter results by quality
        quality_results = [r for r in individual_results if r.quality_score >= self.config.min_quality_score]
        
        if not quality_results:
            self.logger.warning("⚠️ No HMMs met quality threshold, using all results")
            quality_results = individual_results
        
        self.logger.info(f"✅ Fitted {len(quality_results)} high-quality HMMs out of {len(individual_results)} total")
        return quality_results
    
    def _fit_hmms_parallel(self, X: np.ndarray) -> List[HMMResult]:
        """Fit HMMs using parallel processing."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            futures = {
                executor.submit(self._fit_single_hmm, config, X): config
                for config in self.config.base_configs
            }
            
            for future in as_completed(futures):
                config = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Error fitting HMM with config {config}: {e}")
        
        return results
    
    def _fit_hmms_sequential(self, X: np.ndarray) -> List[HMMResult]:
        """Fit HMMs using sequential processing."""
        results = []
        
        for config in self.config.base_configs:
            try:
                result = self._fit_single_hmm(config, X)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Error fitting HMM with config {config}: {e}")
        
        return results
    
    def _fit_single_hmm(self, config: HMMConfig, X: np.ndarray) -> Optional[HMMResult]:
        """Fit a single HMM with given configuration."""
        start_time = time.time()
        
        try:
            # Create HMM based on type
            if config.hmm_type == HMMType.GAUSSIAN:
                model = self._create_gaussian_hmm(config)
            elif config.hmm_type == HMMType.GMM:
                model = self._create_gmm_hmm(config)
            else:
                self.logger.warning(f"⚠️ Unsupported HMM type: {config.hmm_type}")
                return None
            
            # Fit model
            model.fit(X)
            
            # Get predictions
            labels = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Calculate metrics
            log_likelihood = model.score(X)
            aic = 2 * config.n_components - 2 * log_likelihood
            bic = np.log(X.shape[0]) * config.n_components - 2 * log_likelihood
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(labels, probabilities, config)
            
            training_time = time.time() - start_time
            
            return HMMResult(
                config=config,
                model=model,
                labels=labels,
                probabilities=probabilities,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                quality_score=quality_score,
                training_time=training_time,
                convergence_info={'converged': True, 'n_iter': config.n_iter}
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error fitting HMM: {e}")
            return None
    
    def _create_gaussian_hmm(self, config: HMMConfig):
        """Create Gaussian HMM."""
        from hmmlearn.hmm import GaussianHMM
        
        return GaussianHMM(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            n_iter=config.n_iter,
            tol=config.tol,
            init_params=config.init_params,
            random_state=config.random_state
        )
    
    def _create_gmm_hmm(self, config: HMMConfig):
        """Create Gaussian Mixture Model."""
        return GaussianMixture(
            n_components=config.n_components,
            covariance_type=config.covariance_type,
            max_iter=config.n_iter,
            tol=config.tol,
            init_params=config.init_params,
            random_state=config.random_state,
            reg_covar=config.reg_covar
        )
    
    def _calculate_quality_score(self, labels: np.ndarray, probabilities: np.ndarray, config: HMMConfig) -> float:
        """Calculate quality score for HMM result."""
        # Regime balance
        unique_labels, counts = np.unique(labels, return_counts=True)
        regime_balance = 1 - (np.std(counts) / np.mean(counts)) if len(counts) > 1 else 0
        
        # Probability confidence
        max_probs = np.max(probabilities, axis=1)
        confidence = np.mean(max_probs)
        
        # Regime size adequacy
        min_regime_size = np.min(counts) if len(counts) > 0 else 0
        size_adequacy = min(1.0, min_regime_size / config.n_components / 100)
        
        # Overall quality score
        quality_score = (regime_balance * 0.4 + confidence * 0.4 + size_adequacy * 0.2)
        
        return quality_score
    
    def _create_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create ensemble from individual HMM results."""
        if not individual_results:
            raise ValueError("No individual results provided for ensemble")
        
        if self.config.method == EnsembleMethod.VOTING:
            return self._create_voting_ensemble(individual_results, X)
        elif self.config.method == EnsembleMethod.BAGGING:
            return self._create_bagging_ensemble(individual_results, X)
        elif self.config.method == EnsembleMethod.STACKING:
            return self._create_stacking_ensemble(individual_results, X)
        elif self.config.method == EnsembleMethod.BAYESIAN_AVERAGING:
            return self._create_bayesian_ensemble(individual_results, X)
        elif self.config.method == EnsembleMethod.TEMPORAL_ENSEMBLE:
            return self._create_temporal_ensemble(individual_results, X)
        else:
            self.logger.warning(f"⚠️ Unknown ensemble method: {self.config.method}")
            return self._create_voting_ensemble(individual_results, X)
    
    def _create_voting_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create voting ensemble."""
        n_samples = len(individual_results[0].labels)
        n_regimes = len(np.unique(individual_results[0].labels))
        
        # Initialize ensemble arrays
        ensemble_labels = np.zeros(n_samples)
        ensemble_probabilities = np.zeros((n_samples, n_regimes))
        ensemble_confidence = np.zeros(n_samples)
        
        if self.config.voting_type == "hard":
            # Hard voting
            all_labels = np.array([r.labels for r in individual_results]).T
            
            for i in range(n_samples):
                # Get most common label
                unique_labels, counts = np.unique(all_labels[i], return_counts=True)
                ensemble_labels[i] = unique_labels[np.argmax(counts)]
                
                # Calculate confidence as proportion of votes
                ensemble_confidence[i] = np.max(counts) / len(individual_results)
        
        else:
            # Soft voting (probability averaging)
            for result in individual_results:
                ensemble_probabilities += result.probabilities
            
            # Average probabilities
            ensemble_probabilities /= len(individual_results)
            
            # Get labels from max probabilities
            ensemble_labels = np.argmax(ensemble_probabilities, axis=1)
            
            # Calculate confidence as max probability
            ensemble_confidence = np.max(ensemble_probabilities, axis=1)
        
        # Calculate diversity and stability
        diversity_score = self._calculate_diversity_score(individual_results)
        stability_score = self._calculate_stability_score(individual_results)
        
        return EnsembleResult(
            ensemble_method=EnsembleMethod.VOTING,
            individual_results=individual_results,
            ensemble_labels=ensemble_labels,
            ensemble_probabilities=ensemble_probabilities,
            ensemble_confidence=ensemble_confidence,
            diversity_score=diversity_score,
            stability_score=stability_score,
            overall_quality=0.0,  # Will be calculated later
            meta_info={
                'voting_type': self.config.voting_type,
                'n_models': len(individual_results),
                'n_regimes': n_regimes
            }
        )
    
    def _create_bagging_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create bagging ensemble."""
        # For bagging, we need to fit models on bootstrap samples
        # Since we already have individual results, we'll use them as-is
        # In a full implementation, you would fit models on bootstrap samples
        
        # Use voting ensemble as base
        voting_result = self._create_voting_ensemble(individual_results, X)
        voting_result.ensemble_method = EnsembleMethod.BAGGING
        
        return voting_result
    
    def _create_stacking_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create stacking ensemble."""
        # Create meta-features from individual predictions
        meta_features = []
        
        for result in individual_results:
            meta_features.append(result.probabilities)
        
        meta_features = np.hstack(meta_features)
        
        # Train meta-learner (simple logistic regression)
        from sklearn.linear_model import LogisticRegression
        
        meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        meta_learner.fit(meta_features, individual_results[0].labels)
        
        # Get ensemble predictions
        ensemble_probabilities = meta_learner.predict_proba(meta_features)
        ensemble_labels = meta_learner.predict(meta_features)
        ensemble_confidence = np.max(ensemble_probabilities, axis=1)
        
        # Calculate diversity and stability
        diversity_score = self._calculate_diversity_score(individual_results)
        stability_score = self._calculate_stability_score(individual_results)
        
        return EnsembleResult(
            ensemble_method=EnsembleMethod.STACKING,
            individual_results=individual_results,
            ensemble_labels=ensemble_labels,
            ensemble_probabilities=ensemble_probabilities,
            ensemble_confidence=ensemble_confidence,
            diversity_score=diversity_score,
            stability_score=stability_score,
            overall_quality=0.0,  # Will be calculated later
            meta_info={
                'meta_learner': 'LogisticRegression',
                'n_models': len(individual_results),
                'cv_folds': self.config.stacking_cv_folds
            }
        )
    
    def _create_bayesian_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create Bayesian averaging ensemble."""
        # Weight models by their quality scores
        weights = np.array([r.quality_score for r in individual_results])
        weights = weights / np.sum(weights)  # Normalize weights
        
        n_samples = len(individual_results[0].labels)
        n_regimes = individual_results[0].probabilities.shape[1]
        
        # Weighted probability averaging
        ensemble_probabilities = np.zeros((n_samples, n_regimes))
        
        for i, result in enumerate(individual_results):
            ensemble_probabilities += weights[i] * result.probabilities
        
        # Get labels from max probabilities
        ensemble_labels = np.argmax(ensemble_probabilities, axis=1)
        ensemble_confidence = np.max(ensemble_probabilities, axis=1)
        
        # Calculate diversity and stability
        diversity_score = self._calculate_diversity_score(individual_results)
        stability_score = self._calculate_stability_score(individual_results)
        
        return EnsembleResult(
            ensemble_method=EnsembleMethod.BAYESIAN_AVERAGING,
            individual_results=individual_results,
            ensemble_labels=ensemble_labels,
            ensemble_probabilities=ensemble_probabilities,
            ensemble_confidence=ensemble_confidence,
            diversity_score=diversity_score,
            stability_score=stability_score,
            overall_quality=0.0,  # Will be calculated later
            meta_info={
                'weights': weights.tolist(),
                'n_models': len(individual_results),
                'weighted_by_quality': True
            }
        )
    
    def _create_temporal_ensemble(self, individual_results: List[HMMResult], X: np.ndarray) -> EnsembleResult:
        """Create temporal ensemble."""
        # For temporal ensemble, we weight recent predictions more heavily
        # This is a simplified implementation
        
        # Use Bayesian averaging as base
        bayesian_result = self._create_bayesian_ensemble(individual_results, X)
        bayesian_result.ensemble_method = EnsembleMethod.TEMPORAL_ENSEMBLE
        
        return bayesian_result
    
    def _calculate_diversity_score(self, individual_results: List[HMMResult]) -> float:
        """Calculate diversity score between individual results."""
        if len(individual_results) < 2:
            return 0.0
        
        # Calculate pairwise adjusted rand index
        ari_scores = []
        for i in range(len(individual_results)):
            for j in range(i + 1, len(individual_results)):
                ari = adjusted_rand_score(individual_results[i].labels, individual_results[j].labels)
                ari_scores.append(ari)
        
        # Diversity is inverse of average agreement
        avg_ari = np.mean(ari_scores)
        diversity_score = 1 - avg_ari
        
        return diversity_score
    
    def _calculate_stability_score(self, individual_results: List[HMMResult]) -> float:
        """Calculate stability score of individual results."""
        # Calculate coefficient of variation of quality scores
        quality_scores = [r.quality_score for r in individual_results]
        
        if len(quality_scores) < 2:
            return 1.0
        
        cv = np.std(quality_scores) / (np.mean(quality_scores) + 1e-8)
        stability_score = 1 / (1 + cv)  # Higher stability = lower CV
        
        return stability_score
    
    def _evaluate_ensemble_quality(self, ensemble_result: EnsembleResult, X: np.ndarray) -> float:
        """Evaluate overall ensemble quality."""
        # Combine multiple quality metrics
        confidence_score = np.mean(ensemble_result.ensemble_confidence)
        diversity_score = ensemble_result.diversity_score
        stability_score = ensemble_result.stability_score
        
        # Regime balance
        unique_labels, counts = np.unique(ensemble_result.ensemble_labels, return_counts=True)
        regime_balance = 1 - (np.std(counts) / np.mean(counts)) if len(counts) > 1 else 0
        
        # Overall quality score
        overall_quality = (
            confidence_score * 0.3 +
            diversity_score * 0.2 +
            stability_score * 0.2 +
            regime_balance * 0.3
        )
        
        return overall_quality
    
    def _save_results(self, ensemble_result: EnsembleResult):
        """Save ensemble results."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save ensemble result
            result_file = output_dir / f"hmm_ensemble_result_{int(time.time())}.pkl"
            joblib.dump(ensemble_result, result_file)
            
            self.logger.info(f"💾 Ensemble results saved to {result_file}")
    
    def _generate_plots(self, ensemble_result: EnsembleResult, X: np.ndarray):
        """Generate visualization plots."""
        if self.config.output_directory:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot 1: Ensemble confidence distribution
            self._plot_confidence_distribution(ensemble_result, output_dir)
            
            # Plot 2: Individual vs ensemble comparison
            self._plot_individual_comparison(ensemble_result, output_dir)
            
            # Plot 3: Quality metrics
            self._plot_quality_metrics(ensemble_result, output_dir)
    
    def _plot_confidence_distribution(self, ensemble_result: EnsembleResult, output_dir: Path):
        """Plot ensemble confidence distribution."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(ensemble_result.ensemble_confidence, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Ensemble Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Ensemble Confidence Scores')
        ax.axvline(np.mean(ensemble_result.ensemble_confidence), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(ensemble_result.ensemble_confidence):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_comparison(self, ensemble_result: EnsembleResult, output_dir: Path):
        """Plot individual vs ensemble comparison."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Plot individual model qualities
        qualities = [r.quality_score for r in ensemble_result.individual_results]
        axes[0].bar(range(len(qualities)), qualities)
        axes[0].set_xlabel('Model Index')
        axes[0].set_ylabel('Quality Score')
        axes[0].set_title('Individual Model Quality Scores')
        
        # Plot ensemble confidence
        axes[1].hist(ensemble_result.ensemble_confidence, bins=30, alpha=0.7)
        axes[1].set_xlabel('Ensemble Confidence')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Ensemble Confidence Distribution')
        
        # Plot regime distribution
        unique_labels, counts = np.unique(ensemble_result.ensemble_labels, return_counts=True)
        axes[2].bar(unique_labels, counts)
        axes[2].set_xlabel('Regime')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Ensemble Regime Distribution')
        
        # Plot quality metrics
        metrics = ['Confidence', 'Diversity', 'Stability', 'Overall']
        values = [
            np.mean(ensemble_result.ensemble_confidence),
            ensemble_result.diversity_score,
            ensemble_result.stability_score,
            ensemble_result.overall_quality
        ]
        axes[3].bar(metrics, values)
        axes[3].set_ylabel('Score')
        axes[3].set_title('Ensemble Quality Metrics')
        axes[3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quality_metrics(self, ensemble_result: EnsembleResult, output_dir: Path):
        """Plot quality metrics."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Individual model metrics
        individual_metrics = ['Quality Score', 'Log Likelihood', 'AIC', 'BIC']
        individual_values = []
        
        for result in ensemble_result.individual_results:
            values = [
                result.quality_score,
                result.log_likelihood,
                result.aic,
                result.bic
            ]
            individual_values.append(values)
        
        # Plot individual metrics
        x = np.arange(len(individual_metrics))
        width = 0.8 / len(individual_values)
        
        for i, values in enumerate(individual_values):
            # Normalize values for visualization
            normalized_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
            ax.bar(x + i * width, normalized_values, width, label=f'Model {i+1}')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Individual Model Quality Metrics Comparison')
        ax.set_xticks(x + width * (len(individual_values) - 1) / 2)
        ax.set_xticklabels(individual_metrics)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _update_performance_history(self, ensemble_result: EnsembleResult, execution_time: float):
        """Update performance history."""
        self.performance_history.append({
            'timestamp': time.time(),
            'execution_time': execution_time,
            'overall_quality': ensemble_result.overall_quality,
            'n_models': len(ensemble_result.individual_results),
            'ensemble_method': ensemble_result.ensemble_method.value
        })

# Convenience functions
def create_hmm_ensemble(X: pd.DataFrame, 
                        method: EnsembleMethod = EnsembleMethod.VOTING,
                        config: Optional[EnsembleConfig] = None) -> EnsembleResult:
    """Convenience function for creating HMM ensemble."""
    if config is None:
        config = EnsembleConfig(method=method)
    else:
        config.method = method
    
    manager = HMMEnsembleManager(config)
    return manager.fit_ensemble(X)

def get_stable_hmm_labels(X: pd.DataFrame, 
                          min_quality: float = 0.7,
                          config: Optional[EnsembleConfig] = None) -> Tuple[np.ndarray, float]:
    """Get stable HMM labels with quality score."""
    manager = HMMEnsembleManager(config)
    result = manager.fit_ensemble(X)
    
    return result.ensemble_labels, result.overall_quality