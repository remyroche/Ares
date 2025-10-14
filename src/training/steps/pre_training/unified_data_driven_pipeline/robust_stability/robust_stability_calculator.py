"""
Robust Stability Calculator

Implements comprehensive stability metrics beyond simple Jaccard similarity
to ensure robust feature selection and model validation.

Key Features:
- Multiple stability metrics
- Coefficient-path stability
- Bootstrapped importance rank correlation
- Stability consensus scoring
- Robust stability assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from enum import Enum
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

logger = logging.getLogger(__name__)


class StabilityMetric(Enum):
    """Types of stability metrics."""
    JACCARD_SIMILARITY = "jaccard_similarity"
    COEFFICIENT_PATH = "coefficient_path"
    RANK_CORRELATION = "rank_correlation"
    BOOTSTRAP_STABILITY = "bootstrap_stability"
    VARIANCE_STABILITY = "variance_stability"
    CORRELATION_STABILITY = "correlation_stability"
    CONSENSUS_STABILITY = "consensus_stability"


@dataclass
class StabilityResult:
    """Result from a single stability metric calculation."""
    metric: StabilityMetric
    feature_scores: Dict[str, float]
    average_score: float
    score_std: float
    calculation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobustStabilityConfig:
    """Configuration for robust stability calculation."""
    
    # Stability metrics to calculate
    stability_metrics: List[StabilityMetric] = field(default_factory=lambda: [
        StabilityMetric.JACCARD_SIMILARITY,
        StabilityMetric.COEFFICIENT_PATH,
        StabilityMetric.RANK_CORRELATION,
        StabilityMetric.BOOTSTRAP_STABILITY
    ])
    
    # Jaccard similarity parameters
    jaccard_min_intersection: int = 1
    jaccard_weighted: bool = False
    
    # Coefficient path parameters
    coefficient_path_window: int = 20
    coefficient_path_min_samples: int = 10
    coefficient_path_correlation_method: str = 'pearson'  # 'pearson', 'spearman'
    
    # Rank correlation parameters
    rank_correlation_method: str = 'spearman'  # 'spearman', 'kendall'
    rank_correlation_min_ranks: int = 5
    
    # Bootstrap parameters
    bootstrap_samples: int = 100
    bootstrap_sample_size: float = 0.8  # Fraction of data to sample
    bootstrap_confidence_level: float = 0.95
    
    # Variance stability parameters
    variance_stability_window: int = 10
    variance_stability_min_samples: int = 5
    
    # Correlation stability parameters
    correlation_stability_window: int = 15
    correlation_stability_min_samples: int = 8
    
    # Consensus parameters
    consensus_weights: Dict[StabilityMetric, float] = field(default_factory=lambda: {
        StabilityMetric.JACCARD_SIMILARITY: 0.2,
        StabilityMetric.COEFFICIENT_PATH: 0.3,
        StabilityMetric.RANK_CORRELATION: 0.2,
        StabilityMetric.BOOTSTRAP_STABILITY: 0.3
    })
    
    # Performance parameters
    enable_parallel: bool = True
    max_workers: int = 4
    chunk_size: int = 100
    
    # Validation parameters
    validate_inputs: bool = True
    strict_validation: bool = False


@dataclass
class RobustStabilityResult:
    """Result from robust stability calculation."""
    
    # Individual stability results
    stability_results: Dict[StabilityMetric, StabilityResult]
    
    # Combined stability scores
    combined_stability_scores: Dict[str, float]
    consensus_stability_scores: Dict[str, float]
    
    # Summary statistics
    n_features: int
    n_metrics: int
    average_combined_stability: float
    stability_std: float
    
    # Performance metrics
    total_calculation_time: float
    memory_usage_mb: float
    parallel_operations: int
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class RobustStabilityCalculator:
    """
    Calculator for robust stability metrics beyond simple Jaccard similarity.
    
    This class provides comprehensive stability assessment using multiple
    metrics to ensure robust feature selection and model validation.
    """
    
    def __init__(self, config: Optional[RobustStabilityConfig] = None):
        """
        Initialize the robust stability calculator.
        
        Args:
            config: Configuration for stability calculation
        """
        self.config = config or RobustStabilityConfig()
        self.logger = logger
        
        # Set up parallel processing
        if self.config.enable_parallel:
            self.max_workers = min(self.config.max_workers, mp.cpu_count())
        else:
            self.max_workers = 1
        
        tprint_info("📊 Robust Stability Calculator initialized")
        tprint_debug(f"📊 Stability metrics: {len(self.config.stability_metrics)}")
        tprint_debug(f"📊 Max workers: {self.max_workers}")
        tprint_debug(f"📊 Bootstrap samples: {self.config.bootstrap_samples}")
    
    def calculate_robust_stability(self, 
                                 feature_importances: Dict[str, List[float]],
                                 feature_names: Optional[List[str]] = None) -> RobustStabilityResult:
        """
        Calculate robust stability metrics for features.
        
        Args:
            feature_importances: Dictionary of feature names to importance lists
            feature_names: Optional list of feature names to analyze
            
        Returns:
            RobustStabilityResult with comprehensive stability metrics
        """
        start_time = time.time()
        
        tprint_info("📊 Calculating robust stability metrics...")
        tprint_debug(f"📊 Features: {len(feature_importances)}")
        tprint_debug(f"📊 Metrics: {len(self.config.stability_metrics)}")
        
        try:
            # Validate inputs
            if self.config.validate_inputs:
                self._validate_inputs(feature_importances, feature_names)
            
            # Filter features if specified
            if feature_names is not None:
                feature_importances = {
                    name: importances for name, importances in feature_importances.items()
                    if name in feature_names
                }
            
            # Calculate individual stability metrics
            stability_results = {}
            parallel_operations = 0
            
            if self.config.enable_parallel and len(self.config.stability_metrics) > 1:
                # Parallel processing
                tprint_debug("Calculating stability metrics in parallel...")
                stability_results, parallel_ops = self._calculate_parallel_stability(feature_importances)
                parallel_operations = parallel_ops
            else:
                # Sequential processing
                tprint_debug("Calculating stability metrics sequentially...")
                stability_results = self._calculate_sequential_stability(feature_importances)
            
            # Calculate combined stability scores
            tprint_debug("Calculating combined stability scores...")
            combined_scores = self._calculate_combined_stability_scores(stability_results)
            
            # Calculate consensus stability scores
            tprint_debug("Calculating consensus stability scores...")
            consensus_scores = self._calculate_consensus_stability_scores(stability_results)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(
                combined_scores, consensus_scores, stability_results
            )
            
            # Calculate performance metrics
            total_calculation_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(feature_importances, stability_results)
            
            result = RobustStabilityResult(
                stability_results=stability_results,
                combined_stability_scores=combined_scores,
                consensus_stability_scores=consensus_scores,
                n_features=len(feature_importances),
                n_metrics=len(stability_results),
                average_combined_stability=summary_stats['average_combined_stability'],
                stability_std=summary_stats['stability_std'],
                total_calculation_time=total_calculation_time,
                memory_usage_mb=memory_usage,
                parallel_operations=parallel_operations,
                metadata={
                    'config': self.config.__dict__,
                    'metrics_used': [metric.value for metric in self.config.stability_metrics]
                }
            )
            
            tprint_success(f"✅ Robust stability calculation completed in {total_calculation_time:.3f}s")
            tprint_info(f"📊 Features analyzed: {len(feature_importances)}")
            tprint_info(f"📊 Metrics calculated: {len(stability_results)}")
            tprint_info(f"📊 Average stability: {summary_stats['average_combined_stability']:.3f}")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Robust stability calculation failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _validate_inputs(self, 
                        feature_importances: Dict[str, List[float]],
                        feature_names: Optional[List[str]]) -> None:
        """Validate input parameters."""
        if not feature_importances:
            raise ValueError("Feature importances dictionary cannot be empty")
        
        if self.config.strict_validation:
            # Check for valid importance lists
            for feature, importances in feature_importances.items():
                if not isinstance(importances, list):
                    raise ValueError(f"Importances for {feature} must be a list")
                if len(importances) < 2:
                    raise ValueError(f"Insufficient importance values for {feature}: {len(importances)}")
                if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in importances):
                    raise ValueError(f"Non-finite importance values for {feature}")
    
    def _calculate_parallel_stability(self, 
                                    feature_importances: Dict[str, List[float]]) -> Tuple[Dict[StabilityMetric, StabilityResult], int]:
        """Calculate stability metrics in parallel."""
        stability_results = {}
        parallel_operations = 0
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit stability calculation tasks
                future_to_metric = {}
                for metric in self.config.stability_metrics:
                    future = executor.submit(
                        self._calculate_single_stability_metric, feature_importances, metric
                    )
                    future_to_metric[future] = metric
                
                # Collect results
                for future in future_to_metric:
                    metric = future_to_metric[future]
                    try:
                        result = future.result()
                        stability_results[metric] = result
                        parallel_operations += 1
                    except Exception as e:
                        tprint_warning(f"⚠️ Stability metric {metric.value} failed: {e}")
                        continue
            
        except Exception as e:
            tprint_error(f"❌ Parallel stability calculation failed: {e}")
            return {}, 0
        
        return stability_results, parallel_operations
    
    def _calculate_sequential_stability(self, 
                                      feature_importances: Dict[str, List[float]]) -> Dict[StabilityMetric, StabilityResult]:
        """Calculate stability metrics sequentially."""
        stability_results = {}
        
        for metric in self.config.stability_metrics:
            try:
                tprint_debug(f"Calculating {metric.value} stability...")
                result = self._calculate_single_stability_metric(feature_importances, metric)
                stability_results[metric] = result
                tprint_success(f"✅ {metric.value} completed: {result.average_score:.3f}")
            except Exception as e:
                tprint_warning(f"⚠️ {metric.value} stability calculation failed: {e}")
                continue
        
        return stability_results
    
    def _calculate_single_stability_metric(self, 
                                         feature_importances: Dict[str, List[float]],
                                         metric: StabilityMetric) -> StabilityResult:
        """Calculate a single stability metric."""
        start_time = time.time()
        
        try:
            if metric == StabilityMetric.JACCARD_SIMILARITY:
                return self._calculate_jaccard_stability(feature_importances, start_time)
            elif metric == StabilityMetric.COEFFICIENT_PATH:
                return self._calculate_coefficient_path_stability(feature_importances, start_time)
            elif metric == StabilityMetric.RANK_CORRELATION:
                return self._calculate_rank_correlation_stability(feature_importances, start_time)
            elif metric == StabilityMetric.BOOTSTRAP_STABILITY:
                return self._calculate_bootstrap_stability(feature_importances, start_time)
            elif metric == StabilityMetric.VARIANCE_STABILITY:
                return self._calculate_variance_stability(feature_importances, start_time)
            elif metric == StabilityMetric.CORRELATION_STABILITY:
                return self._calculate_correlation_stability(feature_importances, start_time)
            else:
                raise ValueError(f"Unknown stability metric: {metric}")
                
        except Exception as e:
            tprint_error(f"❌ {metric.value} stability calculation failed: {e}")
            return StabilityResult(
                metric=metric,
                feature_scores={},
                average_score=0.0,
                score_std=0.0,
                calculation_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _calculate_jaccard_stability(self, 
                                   feature_importances: Dict[str, List[float]],
                                   start_time: float) -> StabilityResult:
        """Calculate Jaccard similarity stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < 2:
                    feature_scores[feature] = 0.0
                    continue
                
                # Calculate Jaccard similarity between consecutive importance values
                jaccard_similarities = []
                
                for i in range(len(importances) - 1):
                    # Convert importance values to binary (above/below median)
                    median_importance = np.median(importances)
                    set1 = set([1 if x >= median_importance else 0 for x in [importances[i]]])
                    set2 = set([1 if x >= median_importance else 0 for x in [importances[i + 1]]])
                    
                    # Calculate Jaccard similarity
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard = intersection / union if union > 0 else 0.0
                    jaccard_similarities.append(jaccard)
                
                # Average Jaccard similarity
                feature_scores[feature] = np.mean(jaccard_similarities) if jaccard_similarities else 0.0
            
            return StabilityResult(
                metric=StabilityMetric.JACCARD_SIMILARITY,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={'method': 'consecutive_pairs'}
            )
            
        except Exception as e:
            tprint_error(f"❌ Jaccard stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.JACCARD_SIMILARITY, start_time, str(e))
    
    def _calculate_coefficient_path_stability(self, 
                                            feature_importances: Dict[str, List[float]],
                                            start_time: float) -> StabilityResult:
        """Calculate coefficient path stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < self.config.coefficient_path_min_samples:
                    feature_scores[feature] = 0.0
                    continue
                
                # Calculate rolling correlation of importance values
                window = min(self.config.coefficient_path_window, len(importances) // 2)
                if window < 2:
                    feature_scores[feature] = 0.0
                    continue
                
                correlations = []
                for i in range(len(importances) - window):
                    window1 = importances[i:i + window // 2]
                    window2 = importances[i + window // 2:i + window]
                    
                    if len(window1) >= 2 and len(window2) >= 2:
                        if self.config.coefficient_path_correlation_method == 'pearson':
                            corr, _ = pearsonr(window1, window2)
                        else:  # spearman
                            corr, _ = spearmanr(window1, window2)
                        
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                
                # Average correlation as stability measure
                feature_scores[feature] = np.mean(correlations) if correlations else 0.0
            
            return StabilityResult(
                metric=StabilityMetric.COEFFICIENT_PATH,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={
                    'window_size': self.config.coefficient_path_window,
                    'correlation_method': self.config.coefficient_path_correlation_method
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Coefficient path stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.COEFFICIENT_PATH, start_time, str(e))
    
    def _calculate_rank_correlation_stability(self, 
                                            feature_importances: Dict[str, List[float]],
                                            start_time: float) -> StabilityResult:
        """Calculate rank correlation stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < self.config.rank_correlation_min_ranks:
                    feature_scores[feature] = 0.0
                    continue
                
                # Calculate rank correlations between consecutive importance values
                rank_correlations = []
                
                for i in range(len(importances) - 1):
                    if self.config.rank_correlation_method == 'spearman':
                        corr, _ = spearmanr([importances[i]], [importances[i + 1]])
                    else:  # kendall
                        from scipy.stats import kendalltau
                        corr, _ = kendalltau([importances[i]], [importances[i + 1]])
                    
                    if not np.isnan(corr):
                        rank_correlations.append(abs(corr))
                
                # Average rank correlation as stability measure
                feature_scores[feature] = np.mean(rank_correlations) if rank_correlations else 0.0
            
            return StabilityResult(
                metric=StabilityMetric.RANK_CORRELATION,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={'method': self.config.rank_correlation_method}
            )
            
        except Exception as e:
            tprint_error(f"❌ Rank correlation stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.RANK_CORRELATION, start_time, str(e))
    
    def _calculate_bootstrap_stability(self, 
                                     feature_importances: Dict[str, List[float]],
                                     start_time: float) -> StabilityResult:
        """Calculate bootstrap stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < 5:  # Need minimum samples for bootstrap
                    feature_scores[feature] = 0.0
                    continue
                
                # Bootstrap sampling
                bootstrap_correlations = []
                n_samples = min(self.config.bootstrap_samples, len(importances) * 2)
                
                for _ in range(n_samples):
                    # Sample with replacement
                    sample_size = int(len(importances) * self.config.bootstrap_sample_size)
                    if sample_size < 2:
                        continue
                    
                    sample_indices = np.random.choice(len(importances), size=sample_size, replace=True)
                    sample_importances = [importances[i] for i in sample_indices]
                    
                    # Calculate correlation between first and second half
                    mid_point = len(sample_importances) // 2
                    if mid_point >= 1:
                        first_half = sample_importances[:mid_point]
                        second_half = sample_importances[mid_point:]
                        
                        if len(first_half) >= 2 and len(second_half) >= 2:
                            corr, _ = pearsonr(first_half, second_half)
                            if not np.isnan(corr):
                                bootstrap_correlations.append(abs(corr))
                
                # Average bootstrap correlation as stability measure
                feature_scores[feature] = np.mean(bootstrap_correlations) if bootstrap_correlations else 0.0
            
            return StabilityResult(
                metric=StabilityMetric.BOOTSTRAP_STABILITY,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={
                    'bootstrap_samples': self.config.bootstrap_samples,
                    'sample_size': self.config.bootstrap_sample_size
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Bootstrap stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.BOOTSTRAP_STABILITY, start_time, str(e))
    
    def _calculate_variance_stability(self, 
                                    feature_importances: Dict[str, List[float]],
                                    start_time: float) -> StabilityResult:
        """Calculate variance stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < self.config.variance_stability_min_samples:
                    feature_scores[feature] = 0.0
                    continue
                
                # Calculate rolling variance
                window = min(self.config.variance_stability_window, len(importances) // 2)
                if window < 2:
                    feature_scores[feature] = 0.0
                    continue
                
                rolling_variances = []
                for i in range(len(importances) - window + 1):
                    window_data = importances[i:i + window]
                    rolling_variances.append(np.var(window_data))
                
                # Stability is inverse of variance of rolling variances
                if rolling_variances:
                    variance_of_variances = np.var(rolling_variances)
                    mean_variance = np.mean(rolling_variances)
                    stability = 1.0 / (1.0 + variance_of_variances / (mean_variance + 1e-10))
                else:
                    stability = 0.0
                
                feature_scores[feature] = stability
            
            return StabilityResult(
                metric=StabilityMetric.VARIANCE_STABILITY,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={'window_size': self.config.variance_stability_window}
            )
            
        except Exception as e:
            tprint_error(f"❌ Variance stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.VARIANCE_STABILITY, start_time, str(e))
    
    def _calculate_correlation_stability(self, 
                                       feature_importances: Dict[str, List[float]],
                                       start_time: float) -> StabilityResult:
        """Calculate correlation stability."""
        try:
            feature_scores = {}
            
            for feature, importances in feature_importances.items():
                if len(importances) < self.config.correlation_stability_min_samples:
                    feature_scores[feature] = 0.0
                    continue
                
                # Calculate rolling correlation with overall trend
                window = min(self.config.correlation_stability_window, len(importances) // 2)
                if window < 2:
                    feature_scores[feature] = 0.0
                    continue
                
                # Create trend line
                x = np.arange(len(importances))
                trend = np.polyfit(x, importances, 1)[0] * x + np.polyfit(x, importances, 1)[1]
                
                rolling_correlations = []
                for i in range(len(importances) - window + 1):
                    window_data = importances[i:i + window]
                    window_trend = trend[i:i + window]
                    
                    if len(window_data) >= 2 and len(window_trend) >= 2:
                        corr, _ = pearsonr(window_data, window_trend)
                        if not np.isnan(corr):
                            rolling_correlations.append(abs(corr))
                
                # Average correlation as stability measure
                feature_scores[feature] = np.mean(rolling_correlations) if rolling_correlations else 0.0
            
            return StabilityResult(
                metric=StabilityMetric.CORRELATION_STABILITY,
                feature_scores=feature_scores,
                average_score=np.mean(list(feature_scores.values())),
                score_std=np.std(list(feature_scores.values())),
                calculation_time=time.time() - start_time,
                metadata={'window_size': self.config.correlation_stability_window}
            )
            
        except Exception as e:
            tprint_error(f"❌ Correlation stability calculation failed: {e}")
            return self._create_empty_stability_result(StabilityMetric.CORRELATION_STABILITY, start_time, str(e))
    
    def _calculate_combined_stability_scores(self, 
                                           stability_results: Dict[StabilityMetric, StabilityResult]) -> Dict[str, float]:
        """Calculate combined stability scores from individual metrics."""
        combined_scores = {}
        
        try:
            # Get all features
            all_features = set()
            for result in stability_results.values():
                all_features.update(result.feature_scores.keys())
            
            # Calculate combined scores
            for feature in all_features:
                scores = []
                for result in stability_results.values():
                    if feature in result.feature_scores:
                        scores.append(result.feature_scores[feature])
                
                if scores:
                    combined_scores[feature] = np.mean(scores)
                else:
                    combined_scores[feature] = 0.0
            
        except Exception as e:
            tprint_error(f"❌ Combined stability score calculation failed: {e}")
            return {}
        
        return combined_scores
    
    def _calculate_consensus_stability_scores(self, 
                                            stability_results: Dict[StabilityMetric, StabilityResult]) -> Dict[str, float]:
        """Calculate consensus stability scores using weighted combination."""
        consensus_scores = {}
        
        try:
            # Get all features
            all_features = set()
            for result in stability_results.values():
                all_features.update(result.feature_scores.keys())
            
            # Calculate consensus scores
            for feature in all_features:
                weighted_score = 0.0
                total_weight = 0.0
                
                for metric, result in stability_results.items():
                    if feature in result.feature_scores:
                        weight = self.config.consensus_weights.get(metric, 1.0)
                        weighted_score += result.feature_scores[feature] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    consensus_scores[feature] = weighted_score / total_weight
                else:
                    consensus_scores[feature] = 0.0
            
        except Exception as e:
            tprint_error(f"❌ Consensus stability score calculation failed: {e}")
            return {}
        
        return consensus_scores
    
    def _calculate_summary_statistics(self, 
                                    combined_scores: Dict[str, float],
                                    consensus_scores: Dict[str, float],
                                    stability_results: Dict[StabilityMetric, StabilityResult]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        try:
            return {
                'average_combined_stability': np.mean(list(combined_scores.values())) if combined_scores else 0.0,
                'stability_std': np.std(list(combined_scores.values())) if combined_scores else 0.0,
                'consensus_stability': np.mean(list(consensus_scores.values())) if consensus_scores else 0.0,
                'n_metrics_calculated': len(stability_results),
                'n_features_analyzed': len(combined_scores)
            }
        except Exception:
            return {
                'average_combined_stability': 0.0,
                'stability_std': 0.0,
                'consensus_stability': 0.0,
                'n_metrics_calculated': 0,
                'n_features_analyzed': 0
            }
    
    def _estimate_memory_usage(self, 
                             feature_importances: Dict[str, List[float]],
                             stability_results: Dict[StabilityMetric, StabilityResult]) -> float:
        """Estimate memory usage in MB."""
        try:
            memory_usage = 0.0
            
            # Add memory for feature importances
            for importances in feature_importances.values():
                memory_usage += len(importances) * 8 / 1024 / 1024  # 8 bytes per float
            
            # Add memory for stability results
            for result in stability_results.values():
                memory_usage += len(result.feature_scores) * 8 / 1024 / 1024
            
            return memory_usage
            
        except Exception:
            return 0.0
    
    def _create_empty_stability_result(self, 
                                     metric: StabilityMetric,
                                     start_time: float,
                                     error_message: str) -> StabilityResult:
        """Create empty stability result for failed metric."""
        return StabilityResult(
            metric=metric,
            feature_scores={},
            average_score=0.0,
            score_std=0.0,
            calculation_time=time.time() - start_time,
            metadata={'error': error_message}
        )
    
    def _create_error_result(self, start_time: float, error_message: str) -> RobustStabilityResult:
        """Create error result for failed calculation."""
        return RobustStabilityResult(
            stability_results={},
            combined_stability_scores={},
            consensus_stability_scores={},
            n_features=0,
            n_metrics=0,
            average_combined_stability=0.0,
            stability_std=0.0,
            total_calculation_time=time.time() - start_time,
            memory_usage_mb=0.0,
            parallel_operations=0,
            metadata={'error': True, 'error_message': error_message}
        )


# Convenience functions
def calculate_robust_stability(feature_importances: Dict[str, List[float]],
                             feature_names: Optional[List[str]] = None,
                             config: Optional[RobustStabilityConfig] = None) -> RobustStabilityResult:
    """
    Convenience function to calculate robust stability metrics.
    
    Args:
        feature_importances: Dictionary of feature names to importance lists
        feature_names: Optional list of feature names to analyze
        config: Configuration for stability calculation
        
    Returns:
        RobustStabilityResult with comprehensive stability metrics
    """
    calculator = RobustStabilityCalculator(config)
    return calculator.calculate_robust_stability(feature_importances, feature_names)


# Export main classes and functions
__all__ = [
    'RobustStabilityCalculator',
    'RobustStabilityConfig',
    'RobustStabilityResult',
    'StabilityResult',
    'StabilityMetric',
    'calculate_robust_stability'
]