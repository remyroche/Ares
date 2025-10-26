"""
Automated HDBSCAN Parameter Tuning System

This module provides intelligent parameter tuning for HDBSCAN clustering using
the existing ML Common optimization infrastructure. It leverages AutoTuner,
BayesianTPEOptimizer, and advanced metrics for comprehensive parameter optimization.

Key Features:
- Dataset-aware parameter selection using AutoTuner
- Bayesian optimization for parameter search
- Quality assessment with advanced metrics
- Automatic fallback strategies
- Integration with existing HDBSCAN regime discovery
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# Import ML Common optimization tools
try:
    from src.utils.ml_common.optimization.auto_tuner import AutoTuner, DatasetCharacteristics
    from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import HPODiagnostics
    from src.utils.ml_common.optimization.shared_utils.advanced_metrics import RiskMetrics, RegimeMetrics
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO, HPOPhaseConfig
    from src.utils.ml_common.optimization.regime_hpo_wrapper import RegimeHPOConfig
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logging.warning(f"ML Common optimization tools not available: {e}")

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available")

# Import VectorBT for optimized computations
try:
    import vectorbt as vbt
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    VECTORBT_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    VECTORBT_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"VectorBT optimization tools not available: {e}")

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    logging.warning(f"Hardware optimization tools not available: {e}")

# Import math validation utilities
try:
    from src.utils.math_validation import (
        validate_positive, safe_divide, safe_log, safe_sqrt,
        validate_finite
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    logging.warning(f"Math validation tools not available: {e}")

from src.utils.tprint import tprint
from src.utils.logger import system_logger

logger = system_logger.getChild('AutomatedHDBSCANTuner')

@dataclass
class HDBSCANParameterSpace:
    """Parameter space for HDBSCAN optimization."""
    min_cluster_size: Tuple[int, int] = (10, 100)  # (min, max)
    min_samples: Tuple[int, int] = (5, 50)
    cluster_selection_epsilon: Tuple[float, float] = (0.0, 0.5)
    cluster_selection_method: List[str] = field(default_factory=lambda: ['leaf', 'eom'])  # Prioritize 'leaf' method
    metric: List[str] = field(default_factory=lambda: ['euclidean', 'manhattan', 'cosine'])
    alpha: Tuple[float, float] = (0.5, 2.0)
    cluster_selection_epsilon: Tuple[float, float] = (0.0, 0.5)

@dataclass
class ClusteringQualityMetrics:
    """Comprehensive clustering quality metrics optimized for regime discovery."""
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    n_clusters: int = 0
    n_noise_points: int = 0
    noise_ratio: float = 0.0
    dbcv_score: Optional[float] = None
    economic_separation: float = 0.0
    
    # Target-specific metrics
    within_cluster_cv: Optional[float] = None  # Lower is better
    between_cluster_cv: Optional[float] = None  # Higher is better
    temporal_stability: Optional[float] = None  # Higher is better
    regime_persistence: Optional[float] = None  # Higher is better
    
    # Cluster distribution metrics
    cluster_distributions: Optional[List[float]] = None  # Distribution percentages for each cluster
    min_cluster_size_pct: Optional[float] = None  # Minimum cluster size as percentage
    max_cluster_size_pct: Optional[float] = None  # Maximum cluster size as percentage
    distribution_balanced: Optional[bool] = None  # Whether distribution meets 2%-20% constraint
    
    def is_poor_quality(self) -> bool:
        """Determine if clustering quality is poor based on regime discovery targets."""
        # Check basic quality criteria
        basic_poor = (
            (self.silhouette_score is not None and self.silhouette_score < 0.0) or
            self.n_clusters < 2 or
            self.noise_ratio > 0.5 or
            (self.calinski_harabasz_score is not None and self.calinski_harabasz_score < 10.0) or
            (self.davies_bouldin_score is not None and self.davies_bouldin_score > 5.0)
        )
        
        # Check regime-specific criteria
        regime_poor = (
            self.n_clusters < 4 or self.n_clusters > 8 or  # Target: 4-8 clusters
            (self.within_cluster_cv is not None and self.within_cluster_cv > 0.3) or  # Lower within-cluster CV
            (self.between_cluster_cv is not None and self.between_cluster_cv < 0.1) or  # Higher between-cluster CV
            (self.economic_separation < 0.25) or  # Minimum economic separation (increased threshold)
            (self.distribution_balanced is not None and not self.distribution_balanced) or  # Cluster distribution constraint
            (self.silhouette_score is not None and self.silhouette_score < 0.1)  # Poor silhouette score
        )
        
        return basic_poor or regime_poor
    
    def calculate_composite_score(self) -> float:
        """Calculate composite quality score for optimization."""
        scores = []
        
        # Silhouette score (higher is better, range -1 to 1)
        if self.silhouette_score is not None:
            scores.append(max(0, self.silhouette_score))  # Normalize to 0-1
        
        # Davies-Bouldin score (lower is better, invert)
        if self.davies_bouldin_score is not None:
            scores.append(max(0, 1 - min(1, self.davies_bouldin_score / 5.0)))  # Normalize to 0-1
        
        # Cluster count preference (4-8 clusters optimal)
        if 4 <= self.n_clusters <= 8:
            cluster_score = 1.0
        elif self.n_clusters in [3, 9]:
            cluster_score = 0.8
        elif self.n_clusters in [2, 10]:
            cluster_score = 0.6
        else:
            cluster_score = 0.2
        scores.append(cluster_score)
        
        # Within-cluster CV (lower is better)
        if self.within_cluster_cv is not None:
            scores.append(max(0, 1 - min(1, self.within_cluster_cv / 0.5)))  # Normalize to 0-1
        
        # Between-cluster CV (higher is better)
        if self.between_cluster_cv is not None:
            scores.append(min(1, self.between_cluster_cv / 0.3))  # Normalize to 0-1
        
        # Economic separation (higher is better)
        if self.economic_separation > 0:
            scores.append(min(1, self.economic_separation / 0.2))  # Normalize to 0-1
        
        # Noise ratio (lower is better)
        noise_score = max(0, 1 - self.noise_ratio)
        scores.append(noise_score)
        
        # Cluster distribution balance (higher is better)
        if self.distribution_balanced is not None:
            distribution_score = 1.0 if self.distribution_balanced else 0.0
            scores.append(distribution_score)
        
        return np.mean(scores) if scores else 0.0

@dataclass
class FallbackStrategy:
    """Fallback strategy configuration."""
    name: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1  # Lower number = higher priority

class AutomatedHDBSCANTuner:
    """
    Automated HDBSCAN parameter tuning system using ML Common optimization tools.
    
    This class provides intelligent parameter optimization for HDBSCAN clustering
    by leveraging the existing ML Common infrastructure for dataset analysis,
    parameter search, and quality assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the automated HDBSCAN tuner with optimized components."""
        self.config = config or {}
        self.parameter_space = HDBSCANParameterSpace()
        self.auto_tuner = None
        self.bayesian_optimizer = None
        
        # Initialize optimization components
        self.vectorbt_optimizer = None
        self.vectorization_manager = None
        self.hardware_manager = None
        self.memory_optimizer = None
        
        if ML_COMMON_AVAILABLE:
            self._initialize_ml_common_tools()
        else:
            tprint("⚠️ ML Common tools not available - using basic fallback strategies", "WARNING")
        
        if VECTORBT_OPTIMIZATION_AVAILABLE:
            self._initialize_vectorbt_optimization()
        else:
            tprint("⚠️ VectorBT optimization not available - using standard computations", "WARNING")
        
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self._initialize_hardware_optimization()
        else:
            tprint("⚠️ Hardware optimization not available - using standard processing", "WARNING")
    
    def _initialize_ml_common_tools(self):
        """Initialize ML Common optimization tools."""
        try:
            self.auto_tuner = AutoTuner()
            self.bayesian_optimizer = BayesianTPEOptimizer()
            tprint("✅ ML Common optimization tools initialized", "SUCCESS")
        except Exception as e:
            logger.warning(f"Failed to initialize ML Common tools: {e}")
            self.auto_tuner = None
            self.bayesian_optimizer = None
    
    def _initialize_vectorbt_optimization(self):
        """Initialize VectorBT optimization components."""
        try:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
            self.vectorization_manager = get_unified_vectorization_manager()
            tprint("✅ VectorBT optimization components initialized", "SUCCESS")
        except Exception as e:
            logger.warning(f"Failed to initialize VectorBT optimization: {e}")
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            self.hardware_manager = UnifiedHardwareManager()
            self.memory_optimizer = M1MemoryOptimizer()
            
            # Optimize for clustering workload
            self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            tprint("✅ Hardware optimization components initialized", "SUCCESS")
        except Exception as e:
            logger.warning(f"Failed to initialize hardware optimization: {e}")
            self.hardware_manager = None
            self.memory_optimizer = None
    
    def analyze_dataset_characteristics(self, data: pd.DataFrame) -> DatasetCharacteristics:
        """Analyze dataset characteristics using ML Common AutoTuner."""
        if not ML_COMMON_AVAILABLE or self.auto_tuner is None:
            # Fallback to basic characteristics
            return DatasetCharacteristics(
                n_samples=data.shape[0],
                n_features=data.shape[1],
                feature_complexity=0.5,
                class_imbalance=0.0,
                data_quality_score=0.8,
                temporal_dependency=0.7
            )
        
        try:
            # Use AutoTuner to analyze dataset
            characteristics = self.auto_tuner.analyze_dataset_characteristics(data)
            tprint(f"📊 Dataset analysis: {data.shape[0]} samples, {data.shape[1]} features", "INFO")
            return characteristics
        except Exception as e:
            logger.warning(f"Error analyzing dataset characteristics: {e}")
            # Return fallback characteristics
            return DatasetCharacteristics(
                n_samples=data.shape[0],
                n_features=data.shape[1],
                feature_complexity=0.5,
                class_imbalance=0.0,
                data_quality_score=0.8,
                temporal_dependency=0.7
            )
    
    def create_parameter_search_space(self, characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """Create parameter search space based on dataset characteristics."""
        if not ML_COMMON_AVAILABLE or self.auto_tuner is None:
            # Fallback to basic search space
            return {
                'min_cluster_size': (max(10, characteristics.n_samples // 50), 
                                   min(100, characteristics.n_samples // 10)),
                'min_samples': (max(5, characteristics.n_samples // 100), 
                              min(50, characteristics.n_samples // 20)),
                'cluster_selection_epsilon': (0.0, 0.3),
                'cluster_selection_method': ['eom', 'leaf'],
                'metric': ['euclidean', 'manhattan']
            }
        
        try:
            # Use AutoTuner to create intelligent search space
            search_space = self.auto_tuner.create_hdbscan_search_space(characteristics)
            tprint(f"🎯 Created parameter search space with {len(search_space)} parameters", "INFO")
            return search_space
        except Exception as e:
            logger.warning(f"Error creating parameter search space: {e}")
            # Return fallback search space
            return {
                'min_cluster_size': (max(10, characteristics.n_samples // 50), 
                                   min(100, characteristics.n_samples // 10)),
                'min_samples': (max(5, characteristics.n_samples // 100), 
                              min(50, characteristics.n_samples // 20)),
                'cluster_selection_epsilon': (0.0, 0.3),
                'cluster_selection_method': ['eom', 'leaf'],
                'metric': ['euclidean', 'manhattan']
            }
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        search_space: Dict[str, Any],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize HDBSCAN parameters using Bayesian optimization."""
        if not ML_COMMON_AVAILABLE or self.bayesian_optimizer is None:
            # Fallback to basic parameter selection
            return self._basic_parameter_selection(data, search_space)
        
        try:
            tprint(f"🔍 Starting Bayesian parameter optimization with {n_trials} trials", "INFO")
            
            # Create optimization configuration
            optimization_config = OptimizationConfig(
                n_trials=n_trials,
                timeout=timeout,
                direction='maximize',  # Maximize clustering quality
                study_name='hdbscan_parameter_optimization'
            )
            
            # Define objective function
            def objective(trial):
                params = {
                    'min_cluster_size': trial.suggest_int('min_cluster_size', 
                                                        search_space['min_cluster_size'][0],
                                                        search_space['min_cluster_size'][1]),
                    'min_samples': trial.suggest_int('min_samples',
                                                   search_space['min_samples'][0],
                                                   search_space['min_samples'][1]),
                    'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon',
                                                                    search_space['cluster_selection_epsilon'][0],
                                                                    search_space['cluster_selection_epsilon'][1]),
                    'cluster_selection_method': trial.suggest_categorical('cluster_selection_method',
                                                                          search_space['cluster_selection_method']),
                    'metric': trial.suggest_categorical('metric', search_space['metric'])
                }
                
                # Evaluate clustering quality
                quality_metrics = self._evaluate_clustering_quality(data, params)
                return quality_metrics.calculate_composite_score()
            
            # Run optimization
            best_params = self.bayesian_optimizer.optimize(
                objective=objective,
                config=optimization_config
            )
            
            tprint(f"✅ Parameter optimization completed", "SUCCESS")
            tprint(f"🏆 Best parameters: {best_params}", "SUCCESS")
            
            return best_params
            
        except Exception as e:
            logger.warning(f"Bayesian optimization failed: {e}")
            tprint(f"⚠️ Falling back to basic parameter selection: {e}", "WARNING")
            return self._basic_parameter_selection(data, search_space)
    
    def _basic_parameter_selection(self, data: pd.DataFrame, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Basic parameter selection fallback."""
        n_samples = data.shape[0]
        
        return {
            'min_cluster_size': max(20, n_samples // 30),
            'min_samples': max(10, n_samples // 60),
            'cluster_selection_epsilon': 0.05,
            'cluster_selection_method': 'eom',
            'metric': 'euclidean'
        }
    
    def _evaluate_clustering_quality(self, data: pd.DataFrame, params: Dict[str, Any]) -> ClusteringQualityMetrics:
        """Evaluate clustering quality for given parameters with regime-specific metrics."""
        if not HDBSCAN_AVAILABLE:
            return ClusteringQualityMetrics()
        
        try:
            # Create HDBSCAN clusterer with given parameters
            clusterer = hdbscan.HDBSCAN(**params)
            cluster_labels = clusterer.fit_predict(data)
            
            # Calculate basic metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            noise_ratio = n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
            
            # Calculate advanced metrics
            silhouette_score = None
            calinski_harabasz_score = None
            davies_bouldin_score = None
            within_cluster_cv = None
            between_cluster_cv = None
            temporal_stability = None
            economic_separation = 0.0
            
            if n_clusters > 1:
                try:
                    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                    
                    # Remove noise points for evaluation
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data[valid_mask]
                        valid_labels = cluster_labels[valid_mask]
                        
                        if len(set(valid_labels)) > 1:
                            silhouette_score = silhouette_score(valid_data, valid_labels)
                            calinski_harabasz_score = calinski_harabasz_score(valid_data, valid_labels)
                            davies_bouldin_score = davies_bouldin_score(valid_data, valid_labels)
                            
                            # Calculate within-cluster and between-cluster CV
                            within_cluster_cv, between_cluster_cv = self._calculate_cv_metrics(valid_data, valid_labels)
                            
                            # Calculate temporal stability (if data has temporal structure)
                            temporal_stability = self._calculate_temporal_stability(cluster_labels)
                            
                            # Calculate economic separation (if returns data available)
                            economic_separation = self._calculate_economic_separation(data, cluster_labels)
                            
                except Exception as e:
                    logger.warning(f"Error calculating advanced metrics: {e}")
            
            return ClusteringQualityMetrics(
                silhouette_score=silhouette_score,
                calinski_harabasz_score=calinski_harabasz_score,
                davies_bouldin_score=davies_bouldin_score,
                n_clusters=n_clusters,
                n_noise_points=n_noise_points,
                noise_ratio=noise_ratio,
                within_cluster_cv=within_cluster_cv,
                between_cluster_cv=between_cluster_cv,
                temporal_stability=temporal_stability,
                economic_separation=economic_separation
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating clustering quality: {e}")
            return ClusteringQualityMetrics()
    
    def _calculate_cv_metrics(self, data: np.ndarray, labels: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate within-cluster and between-cluster coefficient of variation using optimized methods."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return None, None
            
            # Use VectorBT for optimized calculations if available
            if VECTORBT_OPTIMIZATION_AVAILABLE and self.vectorbt_optimizer is not None:
                return self._calculate_cv_metrics_vectorbt(data, labels, unique_labels)
            else:
                return self._calculate_cv_metrics_standard(data, labels, unique_labels)
            
        except Exception as e:
            logger.warning(f"Error calculating CV metrics: {e}")
            return None, None
    
    def _calculate_cv_metrics_vectorbt(self, data: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate CV metrics using VectorBT optimization."""
        try:
            # Convert to pandas for VectorBT operations
            data_df = pd.DataFrame(data)
            
            # Calculate within-cluster CV using VectorBT
            within_cvs = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_data = data_df[cluster_mask]
                
                if len(cluster_data) > 1:
                    # Use VectorBT for efficient std and mean calculations
                    cluster_std = cluster_data.std()
                    cluster_mean = cluster_data.mean()
                    
                    # Safe division with math validation
                    if MATH_VALIDATION_AVAILABLE:
                        cluster_cv = safe_divide(cluster_std, np.abs(cluster_mean) + 1e-8).mean()
                    else:
                        cluster_cv = np.mean(cluster_std / (np.abs(cluster_mean) + 1e-8))
                    
                    within_cvs.append(cluster_cv)
            
            within_cluster_cv = np.mean(within_cvs) if within_cvs else None
            
            # Calculate between-cluster CV
            cluster_means = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_data = data_df[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_means.append(cluster_data.mean().values)
            
            if len(cluster_means) > 1:
                cluster_means_df = pd.DataFrame(cluster_means)
                between_cluster_std = cluster_means_df.std()
                between_cluster_mean = cluster_means_df.mean()
                
                if MATH_VALIDATION_AVAILABLE:
                    between_cluster_cv = safe_divide(between_cluster_std, np.abs(between_cluster_mean) + 1e-8).mean()
                else:
                    between_cluster_cv = np.mean(between_cluster_std / (np.abs(between_cluster_mean) + 1e-8))
            else:
                between_cluster_cv = None
            
            return within_cluster_cv, between_cluster_cv
            
        except Exception as e:
            logger.warning(f"Error in VectorBT CV calculation: {e}")
            return self._calculate_cv_metrics_standard(data, labels, unique_labels)
    
    def _calculate_cv_metrics_standard(self, data: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate CV metrics using standard numpy operations."""
        try:
            # Calculate within-cluster CV (lower is better)
            within_cvs = []
            for label in unique_labels:
                cluster_data = data[labels == label]
                if len(cluster_data) > 1:
                    cluster_std = np.std(cluster_data, axis=0)
                    cluster_mean = np.mean(cluster_data, axis=0)
                    
                    # Use safe division if available
                    if MATH_VALIDATION_AVAILABLE:
                        cluster_cv = safe_divide(cluster_std, np.abs(cluster_mean) + 1e-8).mean()
                    else:
                        cluster_cv = np.mean(cluster_std / (np.abs(cluster_mean) + 1e-8))
                    
                    within_cvs.append(cluster_cv)
            
            within_cluster_cv = np.mean(within_cvs) if within_cvs else None
            
            # Calculate between-cluster CV (higher is better)
            cluster_means = []
            for label in unique_labels:
                cluster_data = data[labels == label]
                if len(cluster_data) > 0:
                    cluster_means.append(np.mean(cluster_data, axis=0))
            
            if len(cluster_means) > 1:
                cluster_means = np.array(cluster_means)
                between_cluster_std = np.std(cluster_means, axis=0)
                between_cluster_mean = np.mean(cluster_means, axis=0)
                
                if MATH_VALIDATION_AVAILABLE:
                    between_cluster_cv = safe_divide(between_cluster_std, np.abs(between_cluster_mean) + 1e-8).mean()
                else:
                    between_cluster_cv = np.mean(between_cluster_std / (np.abs(between_cluster_mean) + 1e-8))
            else:
                between_cluster_cv = None
            
            return within_cluster_cv, between_cluster_cv
            
        except Exception as e:
            logger.warning(f"Error in standard CV calculation: {e}")
            return None, None
    
    def _calculate_temporal_stability(self, cluster_labels: np.ndarray) -> Optional[float]:
        """Calculate temporal stability of clustering using optimized methods."""
        try:
            # Use VectorBT for optimized temporal calculations if available
            if VECTORBT_OPTIMIZATION_AVAILABLE and self.vectorbt_optimizer is not None:
                return self._calculate_temporal_stability_vectorbt(cluster_labels)
            else:
                return self._calculate_temporal_stability_standard(cluster_labels)
                
        except Exception as e:
            logger.warning(f"Error calculating temporal stability: {e}")
            return None
    
    def _calculate_temporal_stability_vectorbt(self, cluster_labels: np.ndarray) -> Optional[float]:
        """Calculate temporal stability using VectorBT optimization."""
        try:
            # Convert to pandas Series for VectorBT operations
            labels_series = pd.Series(cluster_labels)
            
            # Calculate regime changes using VectorBT
            regime_changes = labels_series.diff() != 0
            regime_changes.iloc[0] = False  # First element is always False
            
            # Use VectorBT rolling operations for efficient calculation
            regime_durations = []
            current_duration = 1
            
            for change in regime_changes:
                if change:
                    regime_durations.append(current_duration)
                    current_duration = 1
                else:
                    current_duration += 1
            regime_durations.append(current_duration)
            
            if regime_durations:
                # Use VectorBT for statistical calculations
                durations_series = pd.Series(regime_durations)
                avg_duration = durations_series.mean()
                
                # Calculate expected duration with math validation
                unique_labels = len(labels_series.unique())
                if MATH_VALIDATION_AVAILABLE:
                    expected_duration = safe_divide(len(cluster_labels), unique_labels)
                else:
                    expected_duration = len(cluster_labels) / unique_labels
                
                if expected_duration > 0:
                    stability = min(1.0, avg_duration / expected_duration)
                    return stability
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in VectorBT temporal stability calculation: {e}")
            return self._calculate_temporal_stability_standard(cluster_labels)
    
    def _calculate_temporal_stability_standard(self, cluster_labels: np.ndarray) -> Optional[float]:
        """Calculate temporal stability using standard numpy operations."""
        try:
            # Calculate regime persistence (how long regimes last)
            regime_changes = np.diff(cluster_labels) != 0
            regime_durations = []
            
            current_duration = 1
            for change in regime_changes:
                if change:
                    regime_durations.append(current_duration)
                    current_duration = 1
                else:
                    current_duration += 1
            regime_durations.append(current_duration)
            
            if regime_durations:
                # Higher average duration = more stable
                avg_duration = np.mean(regime_durations)
                
                # Calculate expected duration with math validation
                unique_labels = len(np.unique(cluster_labels))
                if MATH_VALIDATION_AVAILABLE:
                    expected_duration = safe_divide(len(cluster_labels), unique_labels)
                else:
                    expected_duration = len(cluster_labels) / unique_labels
                
                if expected_duration > 0:
                    stability = min(1.0, avg_duration / expected_duration)
                    return stability
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in standard temporal stability calculation: {e}")
            return None
    
    def _calculate_economic_separation(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> float:
        """Calculate economic separation between regimes using optimized methods."""
        try:
            # Use VectorBT for optimized calculations if available
            if VECTORBT_OPTIMIZATION_AVAILABLE and self.vectorbt_optimizer is not None:
                return self._calculate_economic_separation_vectorbt(data, cluster_labels)
            else:
                return self._calculate_economic_separation_standard(data, cluster_labels)
                
        except Exception as e:
            logger.warning(f"Error calculating economic separation: {e}")
            return 0.0
    
    def _calculate_economic_separation_vectorbt(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> float:
        """Calculate economic separation using VectorBT optimization."""
        try:
            # Look for returns column (common names)
            returns_col = None
            for col in ['returns', 'return', 'pct_change', 'close']:
                if col in data.columns:
                    returns_col = col
                    break
            
            if returns_col is None:
                return 0.0
            
            # Calculate returns using VectorBT if available
            if returns_col == 'close':
                # Use VectorBT for efficient pct_change calculation
                if VECTORBT_AVAILABLE:
                    returns = vbt.pct_change(data[returns_col]).fillna(0)
                else:
                    returns = data[returns_col].pct_change().fillna(0)
            else:
                returns = data[returns_col]
            
            # Calculate economic separation between clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                return 0.0
            
            # Use VectorBT for efficient cluster return calculations
            cluster_returns = []
            for label in unique_labels:
                if label != -1:  # Exclude noise
                    cluster_mask = cluster_labels == label
                    if cluster_mask.sum() > 0:
                        cluster_return = returns[cluster_mask].mean()
                        cluster_returns.append(cluster_return)
            
            if len(cluster_returns) >= 2:
                # Calculate pairwise differences using VectorBT
                cluster_returns_series = pd.Series(cluster_returns)
                
                # Use VectorBT for efficient pairwise calculations
                return_diffs = []
                for i in range(len(cluster_returns)):
                    for j in range(i + 1, len(cluster_returns)):
                        diff = abs(cluster_returns[i] - cluster_returns[j])
                        return_diffs.append(diff)
                
                # Economic separation is the average difference
                if return_diffs:
                    return np.mean(return_diffs)
                else:
                    return 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in VectorBT economic separation calculation: {e}")
            return self._calculate_economic_separation_standard(data, cluster_labels)
    
    def _calculate_economic_separation_standard(self, data: pd.DataFrame, cluster_labels: np.ndarray) -> float:
        """Calculate economic separation using standard pandas operations."""
        try:
            # Look for returns column (common names)
            returns_col = None
            for col in ['returns', 'return', 'pct_change', 'close']:
                if col in data.columns:
                    returns_col = col
                    break
            
            if returns_col is None:
                return 0.0
            
            # Calculate returns if using close prices
            if returns_col == 'close':
                returns = data[returns_col].pct_change().fillna(0).values
            else:
                returns = data[returns_col].values
            
            # Calculate economic separation between clusters
            unique_labels = np.unique(cluster_labels)
            if len(unique_labels) < 2:
                return 0.0
            
            cluster_returns = []
            for label in unique_labels:
                if label != -1:  # Exclude noise
                    cluster_mask = cluster_labels == label
                    if cluster_mask.sum() > 0:
                        cluster_return = np.mean(returns[cluster_mask])
                        cluster_returns.append(cluster_return)
            
            if len(cluster_returns) >= 2:
                # Calculate pairwise differences in returns
                return_diffs = []
                for i in range(len(cluster_returns)):
                    for j in range(i + 1, len(cluster_returns)):
                        return_diffs.append(abs(cluster_returns[i] - cluster_returns[j]))
                
                # Economic separation is the average difference
                return np.mean(return_diffs) if return_diffs else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error in standard economic separation calculation: {e}")
            return 0.0
    
    def _calculate_cluster_distribution_metrics(self, cluster_labels: np.ndarray) -> Tuple[Optional[List[float]], Optional[float], Optional[float], Optional[bool]]:
        """Calculate cluster distribution metrics with 2%-20% constraint validation."""
        try:
            # Calculate cluster sizes
            unique_labels = np.unique(cluster_labels)
            total_samples = len(cluster_labels)
            
            cluster_sizes = []
            for label in unique_labels:
                if label != -1:  # Exclude noise
                    cluster_size = np.sum(cluster_labels == label)
                    cluster_sizes.append(cluster_size)
            
            if not cluster_sizes:
                return None, None, None, None
            
            # Calculate distribution percentages
            cluster_distributions = [size / total_samples * 100 for size in cluster_sizes]
            
            # Calculate min and max cluster sizes as percentages
            min_cluster_size_pct = min(cluster_distributions)
            max_cluster_size_pct = max(cluster_distributions)
            
            # Check if distribution meets 2%-20% constraint
            distribution_balanced = all(2.0 <= pct <= 20.0 for pct in cluster_distributions)
            
            return cluster_distributions, min_cluster_size_pct, max_cluster_size_pct, distribution_balanced
            
        except Exception as e:
            logger.warning(f"Error calculating cluster distribution metrics: {e}")
            return None, None, None, None
    
    def create_fallback_strategies(self, data: pd.DataFrame, characteristics: DatasetCharacteristics) -> List[FallbackStrategy]:
        """Create intelligent fallback strategies optimized for 4-8 clusters."""
        strategies = []
        
        # Strategy 1: Target 4-6 clusters with leaf method (enhanced)
        strategies.append(FallbackStrategy(
            name='target_4_6_clusters_leaf_enhanced',
            description='Target 4-6 clusters using leaf method with enhanced epsilon range',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.05,  # Increased from 0.01 for better separation
                'min_cluster_size': max(25, characteristics.n_samples // 25),  # Target ~4-6 clusters
                'min_samples': max(12, characteristics.n_samples // 50),
                'metric': 'euclidean'
            },
            priority=1
        ))
        
        # Strategy 1.5: Enhanced leaf method for better separation
        strategies.append(FallbackStrategy(
            name='enhanced_leaf_separation',
            description='Enhanced leaf method for better cluster separation',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.15,  # Optimized epsilon for better separation
                'min_cluster_size': max(20, characteristics.n_samples // 40),  # Balanced cluster size
                'min_samples': max(8, characteristics.n_samples // 80),  # Balanced samples
                'metric': 'euclidean'  # Use euclidean for better regime detection
            },
            priority=1
        ))
        
        # Strategy 1.6: Ultra-aggressive for balanced distribution
        strategies.append(FallbackStrategy(
            name='ultra_balanced_distribution',
            description='Ultra-aggressive parameters for balanced cluster distribution',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.3,  # High epsilon for aggressive clustering
                'min_cluster_size': max(15, characteristics.n_samples // 50),  # Much smaller clusters
                'min_samples': max(5, characteristics.n_samples // 100),  # Much smaller samples
                'metric': 'manhattan'  # Use manhattan for better separation
            },
            priority=2
        ))
        
        # Strategy 2: Target 6-8 clusters with EOM method
        strategies.append(FallbackStrategy(
            name='target_6_8_clusters_eom',
            description='Target 6-8 clusters using EOM method',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.05,
                'min_cluster_size': max(20, characteristics.n_samples // 35),  # Target ~6-8 clusters
                'min_samples': max(10, characteristics.n_samples // 70),
                'metric': 'euclidean'
            },
            priority=2
        ))
        
        # Strategy 3: Aggressive clustering for more clusters
        strategies.append(FallbackStrategy(
            name='aggressive_more_clusters',
            description='Aggressive clustering to get more clusters',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.02,
                'min_cluster_size': max(15, characteristics.n_samples // 50),  # Target 8+ clusters
                'min_samples': max(8, characteristics.n_samples // 100),
                'metric': 'euclidean'
            },
            priority=3
        ))
        
        # Strategy 4: Conservative for fewer clusters
        strategies.append(FallbackStrategy(
            name='conservative_fewer_clusters',
            description='Conservative approach for fewer clusters',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.1,
                'min_cluster_size': max(40, characteristics.n_samples // 15),  # Target 3-4 clusters
                'min_samples': max(20, characteristics.n_samples // 30),
                'metric': 'manhattan'
            },
            priority=4
        ))
        
        # Strategy 5: Alternative metrics for better separation
        strategies.append(FallbackStrategy(
            name='alternative_metrics_separation',
            description='Alternative metrics for better cluster separation',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.05,
                'min_cluster_size': max(25, characteristics.n_samples // 30),
                'min_samples': max(12, characteristics.n_samples // 60),
                'metric': 'cosine'
            },
            priority=5
        ))
        
        # Strategy 6: Balanced distribution targeting
        strategies.append(FallbackStrategy(
            name='balanced_distribution',
            description='Target balanced cluster distribution (2%-20%)',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.08,
                'min_cluster_size': max(20, characteristics.n_samples // 25),  # Target ~4% minimum
                'min_samples': max(10, characteristics.n_samples // 50),
                'metric': 'euclidean'
            },
            priority=6
        ))
        
        # Strategy 7: Conservative balanced approach
        strategies.append(FallbackStrategy(
            name='conservative_balanced',
            description='Conservative approach for balanced distribution',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.03,
                'min_cluster_size': max(30, characteristics.n_samples // 20),  # Target ~5% minimum
                'min_samples': max(15, characteristics.n_samples // 40),
                'metric': 'manhattan'
            },
            priority=7
        ))
        
        # Strategy 8: Aggressive cluster count targeting
        strategies.append(FallbackStrategy(
            name='aggressive_cluster_count',
            description='Aggressive approach to achieve 4-8 clusters',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.01,  # Very tight epsilon
                'min_cluster_size': max(10, characteristics.n_samples // 100),  # Target ~1% minimum
                'min_samples': max(5, characteristics.n_samples // 200),
                'metric': 'euclidean'
            },
            priority=8
        ))
        
        # Strategy 9: Alternative metrics for better separation
        strategies.append(FallbackStrategy(
            name='cosine_metric_separation',
            description='Cosine metric for better feature separation',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.05,
                'min_cluster_size': max(15, characteristics.n_samples // 50),
                'min_samples': max(8, characteristics.n_samples // 100),
                'metric': 'cosine'
            },
            priority=9
        ))
        
        # Strategy 10: Manhattan metric for robust clustering
        strategies.append(FallbackStrategy(
            name='manhattan_robust_clustering',
            description='Manhattan metric for robust regime detection',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.08,
                'min_cluster_size': max(20, characteristics.n_samples // 40),
                'min_samples': max(10, characteristics.n_samples // 80),
                'metric': 'manhattan'
            },
            priority=10
        ))
        
        # Strategy 11: Feature engineering approach
        strategies.append(FallbackStrategy(
            name='feature_engineering_approach',
            description='Enhanced feature engineering for better discrimination',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.03,
                'min_cluster_size': max(12, characteristics.n_samples // 60),
                'min_samples': max(6, characteristics.n_samples // 120),
                'metric': 'euclidean'
            },
            priority=11
        ))
        
        # Strategy 12: Multi-metric ensemble approach
        strategies.append(FallbackStrategy(
            name='multi_metric_ensemble',
            description='Multi-metric approach for comprehensive clustering',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.06,
                'min_cluster_size': max(18, characteristics.n_samples // 35),
                'min_samples': max(9, characteristics.n_samples // 70),
                'metric': 'cosine'
            },
            priority=12
        ))
        
        return strategies
    
    def execute_parameter_fallback(
        self, 
        data: pd.DataFrame, 
        initial_result: ClusteringQualityMetrics,
        max_retries: int = 3
    ) -> Tuple[Dict[str, Any], ClusteringQualityMetrics]:
        """Execute parameter fallback with intelligent strategy selection."""
        tprint("🔄 Starting parameter fallback system...", "INFO")
        
        # Analyze dataset characteristics
        characteristics = self.analyze_dataset_characteristics(data)
        
        # Create fallback strategies
        fallback_strategies = self.create_fallback_strategies(data, characteristics)
        
        best_result = initial_result
        best_params = {}
        fallback_attempts = []
        
        for i, strategy in enumerate(fallback_strategies[:max_retries]):
            tprint(f"🔄 Attempting fallback strategy {i+1}: {strategy.name}", "INFO")
            
            try:
                # Apply feature engineering for certain strategies
                test_data = data
                if strategy.name in ['feature_engineering_approach', 'multi_metric_ensemble']:
                    tprint(f"🔧 Applying feature engineering for {strategy.name}...", "INFO")
                    test_data = self._apply_feature_engineering(data)
                
                # Evaluate clustering with fallback parameters
                quality_metrics = self._evaluate_clustering_quality(test_data, strategy.parameters)
                
                # Calculate quality improvement
                quality_improvement = self._calculate_quality_improvement(initial_result, quality_metrics)
                
                fallback_attempts.append({
                    'strategy': strategy.name,
                    'parameters': strategy.parameters,
                    'quality_metrics': quality_metrics,
                    'quality_improvement': quality_improvement
                })
                
                # If quality improved significantly, use this result
                if quality_improvement > 0.1:  # 10% improvement threshold
                    tprint(f"✅ Fallback successful: {strategy.name} improved quality by {quality_improvement:.2%}", "SUCCESS")
                    return strategy.parameters, quality_metrics
                
                # Keep track of best result so far
                if quality_improvement > 0:
                    best_result = quality_metrics
                    best_params = strategy.parameters
                
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.name} failed: {e}")
                continue
        
        # Return best result from all attempts
        if fallback_attempts:
            best_attempt = max(fallback_attempts, key=lambda x: x['quality_improvement'])
            tprint(f"📊 Best fallback result: {best_attempt['strategy']} (improvement: {best_attempt['quality_improvement']:.2%})", "INFO")
            return best_attempt['parameters'], best_attempt['quality_metrics']
        else:
            tprint("⚠️ All fallback strategies failed, returning original parameters", "WARNING")
            return {}, initial_result
    
    def _calculate_quality_improvement(
        self, 
        original: ClusteringQualityMetrics, 
        improved: ClusteringQualityMetrics
    ) -> float:
        """Calculate quality improvement percentage."""
        improvements = []
        
        # Silhouette score improvement
        if (original.silhouette_score is not None and 
            improved.silhouette_score is not None):
            if original.silhouette_score != 0:
                sil_improvement = (improved.silhouette_score - original.silhouette_score) / abs(original.silhouette_score)
                improvements.append(sil_improvement)
        
        # Cluster count improvement (prefer more clusters if reasonable)
        if improved.n_clusters > original.n_clusters and improved.n_clusters <= 8:
            cluster_improvement = (improved.n_clusters - original.n_clusters) / max(original.n_clusters, 1)
            improvements.append(cluster_improvement)
        
        # Noise ratio improvement (prefer less noise)
        if improved.noise_ratio < original.noise_ratio:
            noise_improvement = (original.noise_ratio - improved.noise_ratio) / max(original.noise_ratio, 0.01)
            improvements.append(noise_improvement)
        
        # Calinski-Harabasz score improvement
        if (original.calinski_harabasz_score is not None and 
            improved.calinski_harabasz_score is not None):
            if original.calinski_harabasz_score != 0:
                ch_improvement = (improved.calinski_harabasz_score - original.calinski_harabasz_score) / abs(original.calinski_harabasz_score)
                improvements.append(ch_improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def tune_parameters(
        self, 
        data: pd.DataFrame, 
        n_trials: int = 50,
        timeout: Optional[int] = None,
        enable_fallback: bool = True
    ) -> Tuple[Dict[str, Any], ClusteringQualityMetrics]:
        """
        Main method to tune HDBSCAN parameters with automatic fallback and hardware optimization.
        
        Args:
            data: Input data for clustering
            n_trials: Number of optimization trials
            timeout: Maximum time for optimization
            enable_fallback: Whether to enable parameter fallback
            
        Returns:
            Tuple of (best_parameters, quality_metrics)
        """
        tprint("🎯 Starting automated HDBSCAN parameter tuning...", "INFO")
        
        # Step 0: Hardware optimization setup
        if HARDWARE_OPTIMIZATION_AVAILABLE and self.hardware_manager is not None:
            tprint("🔧 Optimizing hardware for clustering workload...", "INFO")
            self.hardware_manager.optimize_for_workload(WorkloadType.FEATURE_ENGINEERING)
            
            # Memory optimization
            if self.memory_optimizer is not None:
                memory_info = self.memory_optimizer.get_memory_info()
                tprint(f"💾 Available memory: {memory_info.get('available_gb', 0):.1f} GB", "INFO")
        
        # Step 1: Analyze dataset characteristics
        characteristics = self.analyze_dataset_characteristics(data)
        tprint(f"📊 Dataset: {data.shape[0]} samples, {data.shape[1]} features", "INFO")
        
        # Step 2: Create parameter search space
        search_space = self.create_parameter_search_space(characteristics)
        tprint(f"🔍 Search space created with {len(search_space)} parameters", "INFO")
        
        # Step 3: Optimize parameters with hardware optimization
        best_params = self.optimize_parameters(data, search_space, n_trials, timeout)
        tprint(f"🏆 Initial optimization completed", "SUCCESS")
        
        # Step 4: Evaluate initial result with optimized calculations
        initial_quality = self._evaluate_clustering_quality_optimized(data, best_params)
        tprint(f"📊 Initial quality: Silhouette={initial_quality.silhouette_score:.3f}, "
               f"Clusters={initial_quality.n_clusters}, Noise={initial_quality.noise_ratio:.3f}", "INFO")
        
        # Step 5: Check if fallback is needed
        if enable_fallback and initial_quality.is_poor_quality():
            tprint("⚠️ Poor clustering quality detected - attempting parameter fallback", "WARNING")
            
            try:
                best_params, final_quality = self.execute_parameter_fallback(data, initial_quality)
                tprint(f"✅ Parameter fallback completed", "SUCCESS")
                tprint(f"📊 Final quality: Silhouette={final_quality.silhouette_score:.3f}, "
                       f"Clusters={final_quality.n_clusters}, Noise={final_quality.noise_ratio:.3f}", "INFO")
            except Exception as e:
                logger.warning(f"Parameter fallback failed: {e}")
                tprint(f"⚠️ Parameter fallback failed: {e}", "WARNING")
                final_quality = initial_quality
        else:
            final_quality = initial_quality
        
        tprint("✅ Automated HDBSCAN parameter tuning completed", "SUCCESS")
        return best_params, final_quality
    
    def _evaluate_clustering_quality_optimized(self, data: pd.DataFrame, params: Dict[str, Any]) -> ClusteringQualityMetrics:
        """Evaluate clustering quality with hardware and VectorBT optimizations."""
        if not HDBSCAN_AVAILABLE:
            return ClusteringQualityMetrics()
        
        try:
            # Use hardware-optimized matrix operations if available
            if HARDWARE_OPTIMIZATION_AVAILABLE and self.hardware_manager is not None:
                # Optimize data for hardware
                data_optimized = self._optimize_data_for_hardware(data)
            else:
                data_optimized = data
            
            # Create HDBSCAN clusterer with given parameters
            clusterer = hdbscan.HDBSCAN(**params)
            cluster_labels = clusterer.fit_predict(data_optimized)
            
            # Calculate basic metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise_points = list(cluster_labels).count(-1)
            noise_ratio = n_noise_points / len(cluster_labels) if len(cluster_labels) > 0 else 0.0
            
            # Calculate advanced metrics with optimizations
            silhouette_score = None
            calinski_harabasz_score = None
            davies_bouldin_score = None
            within_cluster_cv = None
            between_cluster_cv = None
            temporal_stability = None
            economic_separation = 0.0
            
            # Calculate cluster distribution metrics
            cluster_distributions, min_cluster_size_pct, max_cluster_size_pct, distribution_balanced = self._calculate_cluster_distribution_metrics(cluster_labels)
            
            if n_clusters > 1:
                try:
                    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                    
                    # Remove noise points for evaluation
                    valid_mask = cluster_labels != -1
                    if valid_mask.sum() > 1:
                        valid_data = data_optimized[valid_mask]
                        valid_labels = cluster_labels[valid_mask]
                        
                        if len(set(valid_labels)) > 1:
                            silhouette_score = silhouette_score(valid_data, valid_labels)
                            calinski_harabasz_score = calinski_harabasz_score(valid_data, valid_labels)
                            davies_bouldin_score = davies_bouldin_score(valid_data, valid_labels)
                            
                            # Calculate optimized metrics
                            within_cluster_cv, between_cluster_cv = self._calculate_cv_metrics(valid_data, valid_labels)
                            temporal_stability = self._calculate_temporal_stability(cluster_labels)
                            economic_separation = self._calculate_economic_separation(data, cluster_labels)
                            
                except Exception as e:
                    logger.warning(f"Error calculating advanced metrics: {e}")
            
            return ClusteringQualityMetrics(
                silhouette_score=silhouette_score,
                calinski_harabasz_score=calinski_harabasz_score,
                davies_bouldin_score=davies_bouldin_score,
                n_clusters=n_clusters,
                n_noise_points=n_noise_points,
                noise_ratio=noise_ratio,
                within_cluster_cv=within_cluster_cv,
                between_cluster_cv=between_cluster_cv,
                temporal_stability=temporal_stability,
                economic_separation=economic_separation,
                cluster_distributions=cluster_distributions,
                min_cluster_size_pct=min_cluster_size_pct,
                max_cluster_size_pct=max_cluster_size_pct,
                distribution_balanced=distribution_balanced
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating clustering quality: {e}")
            return ClusteringQualityMetrics()
    
    def _optimize_data_for_hardware(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for hardware processing."""
        try:
            if HARDWARE_OPTIMIZATION_AVAILABLE and self.hardware_manager is not None:
                # Use M1 GPU utils for optimized matrix operations
                gpu_utils = M1GPUManager()
                
                # Convert to numpy for GPU optimization
                data_array = data.values
                
                # Optimize data layout for M1 GPU
                optimized_array = gpu_utils.optimize_array_layout(data_array)
                
                # Convert back to DataFrame
                optimized_data = pd.DataFrame(optimized_array, columns=data.columns, index=data.index)
                return optimized_data
            else:
                return data
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
            return data
    
    def generate_optimization_report(
        self, 
        best_params: Dict[str, Any], 
        quality_metrics: ClusteringQualityMetrics,
        optimization_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'optimization_summary': {
                'best_parameters': best_params,
                'optimization_time_seconds': optimization_time,
                'composite_score': quality_metrics.calculate_composite_score()
            },
            'quality_metrics': {
                'silhouette_score': quality_metrics.silhouette_score,
                'calinski_harabasz_score': quality_metrics.calinski_harabasz_score,
                'davies_bouldin_score': quality_metrics.davies_bouldin_score,
                'n_clusters': quality_metrics.n_clusters,
                'noise_ratio': quality_metrics.noise_ratio,
                'within_cluster_cv': quality_metrics.within_cluster_cv,
                'between_cluster_cv': quality_metrics.between_cluster_cv,
                'temporal_stability': quality_metrics.temporal_stability,
                'economic_separation': quality_metrics.economic_separation,
                'cluster_distributions': quality_metrics.cluster_distributions,
                'min_cluster_size_pct': quality_metrics.min_cluster_size_pct,
                'max_cluster_size_pct': quality_metrics.max_cluster_size_pct,
                'distribution_balanced': quality_metrics.distribution_balanced
            },
            'target_assessment': {
                'cluster_count_optimal': 4 <= quality_metrics.n_clusters <= 8,
                'within_cluster_cv_optimal': quality_metrics.within_cluster_cv is not None and quality_metrics.within_cluster_cv < 0.3,
                'between_cluster_cv_optimal': quality_metrics.between_cluster_cv is not None and quality_metrics.between_cluster_cv > 0.1,
                'silhouette_optimal': quality_metrics.silhouette_score is not None and quality_metrics.silhouette_score > 0.0,
                'dbi_optimal': quality_metrics.davies_bouldin_score is not None and quality_metrics.davies_bouldin_score < 2.0,
                'economic_separation_optimal': quality_metrics.economic_separation > 0.05,
                'distribution_balanced': quality_metrics.distribution_balanced is not None and quality_metrics.distribution_balanced
            },
            'recommendations': self._generate_recommendations(quality_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, quality_metrics: ClusteringQualityMetrics) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []
        
        # Cluster count recommendations
        if quality_metrics.n_clusters < 4:
            recommendations.append("Consider reducing min_cluster_size to get more clusters (target: 4-8)")
        elif quality_metrics.n_clusters > 8:
            recommendations.append("Consider increasing min_cluster_size to get fewer clusters (target: 4-8)")
        
        # Within-cluster CV recommendations
        if quality_metrics.within_cluster_cv is not None and quality_metrics.within_cluster_cv > 0.3:
            recommendations.append("High within-cluster CV detected - consider feature selection or different metrics")
        
        # Between-cluster CV recommendations
        if quality_metrics.between_cluster_cv is not None and quality_metrics.between_cluster_cv < 0.1:
            recommendations.append("Low between-cluster CV detected - clusters may not be well separated")
        
        # Silhouette score recommendations
        if quality_metrics.silhouette_score is not None and quality_metrics.silhouette_score < 0.0:
            recommendations.append("Negative silhouette score - clustering quality is poor, consider parameter adjustment")
        
        # Economic separation recommendations
        if quality_metrics.economic_separation < 0.05:
            recommendations.append("Low economic separation - regimes may not be economically distinct")
        
        # Noise ratio recommendations
        if quality_metrics.noise_ratio > 0.3:
            recommendations.append("High noise ratio - consider adjusting min_samples or cluster_selection_epsilon")
        
        # Cluster distribution recommendations
        if quality_metrics.distribution_balanced is not None and not quality_metrics.distribution_balanced:
            if quality_metrics.min_cluster_size_pct is not None and quality_metrics.min_cluster_size_pct < 2.0:
                recommendations.append(f"Cluster too small ({quality_metrics.min_cluster_size_pct:.1f}%) - increase min_cluster_size")
            if quality_metrics.max_cluster_size_pct is not None and quality_metrics.max_cluster_size_pct > 20.0:
                recommendations.append(f"Cluster too large ({quality_metrics.max_cluster_size_pct:.1f}%) - decrease min_cluster_size or increase min_samples")
        
        if not recommendations:
            recommendations.append("Clustering quality meets all target criteria - no adjustments needed")
        
        return recommendations
    
    def validate_optimization_targets(self, quality_metrics: ClusteringQualityMetrics) -> Dict[str, Any]:
        """Validate that all optimization goals and hard caps are met."""
        validation_results = {
            'goals_achieved': {},
            'hard_caps_met': {},
            'overall_success': False,
            'recommendations': []
        }
        
        # OPTIMIZATION GOALS (✅ = Achieved, ❌ = Not Achieved)
        goals = {
            'lower_within_cluster_cv': {
                'achieved': quality_metrics.within_cluster_cv is not None and quality_metrics.within_cluster_cv < 0.3,
                'value': quality_metrics.within_cluster_cv,
                'target': '< 0.3',
                'description': 'Lower within-cluster CV'
            },
            'higher_between_cluster_cv': {
                'achieved': quality_metrics.between_cluster_cv is not None and quality_metrics.between_cluster_cv > 0.1,
                'value': quality_metrics.between_cluster_cv,
                'target': '> 0.1',
                'description': 'Higher between-cluster CV'
            },
            'optimized_silhouette_dbi_economic': {
                'achieved': (
                    (quality_metrics.silhouette_score is not None and quality_metrics.silhouette_score > 0.0) and
                    (quality_metrics.davies_bouldin_score is not None and quality_metrics.davies_bouldin_score < 2.0) and
                    (quality_metrics.economic_separation > 0.05)
                ),
                'value': {
                    'silhouette': quality_metrics.silhouette_score,
                    'dbi': quality_metrics.davies_bouldin_score,
                    'economic_separation': quality_metrics.economic_separation
                },
                'target': 'Silhouette > 0, DBI < 2, Economic > 0.05',
                'description': 'Optimized Silhouette, DBI, economic metrics'
            },
            'temporal_stability': {
                'achieved': quality_metrics.temporal_stability is not None and quality_metrics.temporal_stability > 0.5,
                'value': quality_metrics.temporal_stability,
                'target': '> 0.5',
                'description': 'Temporal stability analysis'
            },
            'balanced_cluster_distribution': {
                'achieved': quality_metrics.distribution_balanced is not None and quality_metrics.distribution_balanced,
                'value': quality_metrics.distribution_balanced,
                'target': 'True (2%-20% per cluster)',
                'description': 'Balanced cluster distribution'
            }
        }
        
        # HARD CAPS (✅ = Met, ❌ = Violated)
        hard_caps = {
            'distribution_constraint_2_20': {
                'met': quality_metrics.distribution_balanced is not None and quality_metrics.distribution_balanced,
                'value': {
                    'min_cluster_pct': quality_metrics.min_cluster_size_pct,
                    'max_cluster_pct': quality_metrics.max_cluster_size_pct,
                    'distributions': quality_metrics.cluster_distributions
                },
                'target': '2% ≤ cluster_size ≤ 20%',
                'description': '2%-20% distribution constraint'
            },
            'cluster_count_4_8': {
                'met': 4 <= quality_metrics.n_clusters <= 8,
                'value': quality_metrics.n_clusters,
                'target': '4 ≤ clusters ≤ 8',
                'description': '4-8 clusters target'
            }
        }
        
        # Evaluate goals
        goals_achieved_count = 0
        for goal_name, goal_data in goals.items():
            validation_results['goals_achieved'][goal_name] = {
                'achieved': goal_data['achieved'],
                'value': goal_data['value'],
                'target': goal_data['target'],
                'description': goal_data['description']
            }
            if goal_data['achieved']:
                goals_achieved_count += 1
        
        # Evaluate hard caps
        hard_caps_met_count = 0
        for cap_name, cap_data in hard_caps.items():
            validation_results['hard_caps_met'][cap_name] = {
                'met': cap_data['met'],
                'value': cap_data['value'],
                'target': cap_data['target'],
                'description': cap_data['description']
            }
            if cap_data['met']:
                hard_caps_met_count += 1
        
        # Overall success: All hard caps must be met, and at least 80% of goals achieved
        validation_results['overall_success'] = (
            hard_caps_met_count == len(hard_caps) and  # All hard caps met
            goals_achieved_count >= len(goals) * 0.8   # At least 80% of goals achieved
        )
        
        # Generate specific recommendations
        recommendations = []
        
        # Hard cap violations (critical)
        for cap_name, cap_data in validation_results['hard_caps_met'].items():
            if not cap_data['met']:
                if cap_name == 'distribution_constraint_2_20':
                    if quality_metrics.min_cluster_size_pct is not None and quality_metrics.min_cluster_size_pct < 2.0:
                        recommendations.append(f"🚨 CRITICAL: Cluster too small ({quality_metrics.min_cluster_size_pct:.1f}%) - increase min_cluster_size")
                    if quality_metrics.max_cluster_size_pct is not None and quality_metrics.max_cluster_size_pct > 20.0:
                        recommendations.append(f"🚨 CRITICAL: Cluster too large ({quality_metrics.max_cluster_size_pct:.1f}%) - decrease min_cluster_size")
                elif cap_name == 'cluster_count_4_8':
                    if quality_metrics.n_clusters < 4:
                        recommendations.append(f"🚨 CRITICAL: Too few clusters ({quality_metrics.n_clusters}) - decrease min_cluster_size")
                    elif quality_metrics.n_clusters > 8:
                        recommendations.append(f"🚨 CRITICAL: Too many clusters ({quality_metrics.n_clusters}) - increase min_cluster_size")
        
        # Goal improvements (important)
        for goal_name, goal_data in validation_results['goals_achieved'].items():
            if not goal_data['achieved']:
                if goal_name == 'lower_within_cluster_cv':
                    recommendations.append(f"⚠️ IMPORTANT: High within-cluster CV ({goal_data['value']:.3f}) - improve feature selection")
                elif goal_name == 'higher_between_cluster_cv':
                    recommendations.append(f"⚠️ IMPORTANT: Low between-cluster CV ({goal_data['value']:.3f}) - improve cluster separation")
                elif goal_name == 'optimized_silhouette_dbi_economic':
                    recommendations.append(f"⚠️ IMPORTANT: Poor clustering metrics - adjust parameters")
                elif goal_name == 'temporal_stability':
                    recommendations.append(f"⚠️ IMPORTANT: Low temporal stability ({goal_data['value']:.3f}) - improve regime persistence")
                elif goal_name == 'balanced_cluster_distribution':
                    recommendations.append(f"⚠️ IMPORTANT: Unbalanced distribution - adjust min_cluster_size")
        
        validation_results['recommendations'] = recommendations
        
        return validation_results
    
    def _apply_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced feature engineering to improve cluster discrimination."""
        try:
            enhanced_data = data.copy()
            
            # 1. Technical indicators for regime detection
            if 'close' in data.columns:
                close_prices = data['close']
                
                # Moving averages with different periods
                for period in [5, 10, 20, 50]:
                    enhanced_data[f'sma_{period}'] = close_prices.rolling(period).mean()
                    enhanced_data[f'ema_{period}'] = close_prices.ewm(span=period).mean()
                
                # Volatility indicators
                enhanced_data['volatility_5'] = close_prices.rolling(5).std()
                enhanced_data['volatility_20'] = close_prices.rolling(20).std()
                enhanced_data['volatility_ratio'] = enhanced_data['volatility_5'] / enhanced_data['volatility_20']
                
                # Momentum indicators
                enhanced_data['rsi_14'] = self._calculate_rsi(close_prices, 14)
                enhanced_data['rsi_21'] = self._calculate_rsi(close_prices, 21)
                enhanced_data['momentum_5'] = close_prices.pct_change(5)
                enhanced_data['momentum_10'] = close_prices.pct_change(10)
                
                # Trend indicators
                enhanced_data['trend_strength'] = self._calculate_trend_strength(close_prices)
                enhanced_data['regime_persistence'] = self._calculate_regime_persistence(close_prices)
                
                # Price action features
                enhanced_data['price_position'] = (close_prices - close_prices.rolling(20).min()) / (close_prices.rolling(20).max() - close_prices.rolling(20).min())
                enhanced_data['volatility_regime'] = self._classify_volatility_regime(enhanced_data['volatility_20'])
            
            # 2. Cross-feature interactions
            if 'returns' in data.columns:
                returns = data['returns']
                enhanced_data['returns_abs'] = abs(returns)
                enhanced_data['returns_squared'] = returns ** 2
                enhanced_data['returns_cumsum'] = returns.cumsum()
                
                # Regime change indicators
                enhanced_data['regime_change'] = self._detect_regime_changes(returns)
                enhanced_data['volatility_clustering'] = self._detect_volatility_clustering(returns)
            
            # 3. Statistical features
            numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['close', 'returns']:  # Avoid recursive features
                    # Rolling statistics
                    enhanced_data[f'{col}_zscore'] = (enhanced_data[col] - enhanced_data[col].rolling(20).mean()) / enhanced_data[col].rolling(20).std()
                    enhanced_data[f'{col}_percentile'] = enhanced_data[col].rolling(20).rank(pct=True)
            
            # 4. Regime-specific features
            enhanced_data['market_regime'] = self._classify_market_regime(enhanced_data)
            enhanced_data['regime_transition'] = self._detect_regime_transitions(enhanced_data)
            
            # Fill NaN values
            enhanced_data = enhanced_data.fillna(method='ffill').fillna(0)
            
            logger.info(f"✅ Feature engineering applied: {enhanced_data.shape[1]} features (was {data.shape[1]})")
            return enhanced_data
            
        except Exception as e:
            logger.warning(f"Feature engineering failed: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate trend strength indicator."""
        try:
            sma_short = prices.rolling(10).mean()
            sma_long = prices.rolling(30).mean()
            trend_strength = (sma_short - sma_long) / sma_long
            return trend_strength.fillna(0)
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _calculate_regime_persistence(self, prices: pd.Series) -> pd.Series:
        """Calculate regime persistence indicator."""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(20).std()
            persistence = volatility.rolling(10).std()  # Volatility of volatility
            return persistence.fillna(0)
        except:
            return pd.Series([0] * len(prices), index=prices.index)
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Classify volatility regime (low, medium, high)."""
        try:
            vol_quantiles = volatility.rolling(100).quantile([0.33, 0.67])
            low_threshold = vol_quantiles[0.33]
            high_threshold = vol_quantiles[0.67]
            
            regime = pd.Series([1] * len(volatility), index=volatility.index)  # Default to medium
            regime[volatility < low_threshold] = 0  # Low volatility
            regime[volatility > high_threshold] = 2  # High volatility
            
            return regime.fillna(1)
        except:
            return pd.Series([1] * len(volatility), index=volatility.index)
    
    def _detect_regime_changes(self, returns: pd.Series) -> pd.Series:
        """Detect regime changes in returns."""
        try:
            volatility = returns.rolling(20).std()
            vol_changes = volatility.diff().abs()
            regime_changes = (vol_changes > vol_changes.rolling(50).quantile(0.8)).astype(int)
            return regime_changes.fillna(0)
        except:
            return pd.Series([0] * len(returns), index=returns.index)
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> pd.Series:
        """Detect volatility clustering."""
        try:
            abs_returns = abs(returns)
            clustering = abs_returns.rolling(10).corr(abs_returns.shift(1))
            return clustering.fillna(0)
        except:
            return pd.Series([0] * len(returns), index=returns.index)
    
    def _classify_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify overall market regime."""
        try:
            if 'close' in data.columns and 'volatility_20' in data.columns:
                close = data['close']
                volatility = data['volatility_20']
                
                # Simple regime classification
                price_trend = close.rolling(20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
                vol_regime = volatility.rolling(50).rank(pct=True)
                
                regime = pd.Series([1] * len(close), index=close.index)  # Default to normal
                regime[(price_trend > 0) & (vol_regime < 0.3)] = 0  # Bull low vol
                regime[(price_trend < 0) & (vol_regime < 0.3)] = 2  # Bear low vol
                regime[vol_regime > 0.7] = 3  # High volatility
                
                return regime.fillna(1)
            else:
                return pd.Series([1] * len(data), index=data.index)
        except:
            return pd.Series([1] * len(data), index=data.index)
    
    def _detect_regime_transitions(self, data: pd.DataFrame) -> pd.Series:
        """Detect regime transitions."""
        try:
            if 'market_regime' in data.columns:
                regime = data['market_regime']
                transitions = (regime != regime.shift(1)).astype(int)
                return transitions.fillna(0)
            else:
                return pd.Series([0] * len(data), index=data.index)
        except:
            return pd.Series([0] * len(data), index=data.index)


def create_automated_hdbscan_tuner(config: Optional[Dict[str, Any]] = None) -> AutomatedHDBSCANTuner:
    """Factory function to create an AutomatedHDBSCANTuner instance."""
    return AutomatedHDBSCANTuner(config)
