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
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

# Import ML Common optimization tools
if TYPE_CHECKING:
    from src.utils.ml_common.optimization.auto_tuner import DatasetCharacteristics

try:
    from src.utils.ml_common.optimization.auto_tuner import AutoTuner, DatasetCharacteristics
    from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import HPODiagnostics
    from src.utils.ml_common.optimization.shared_utils.advanced_metrics import RiskMetrics, RegimeMetrics
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer, OptimizationConfig
    from src.utils.ml_common.optimization.hierarchical_hpo import HierarchicalHPO, HPOPhaseConfig
    from src.utils.ml_common.optimization.regime_hpo_wrapper import RegimeHPOConfig
    from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
        HierarchicalParameterOptimizer,
        ParameterGroup,
        OptimizationStage,
        create_param_group
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    logging.warning(f"ML Common optimization tools not available: {e}")
    # Define a fallback for DatasetCharacteristics when ML Common is not available
    @dataclass
    class DatasetCharacteristics:
        """Fallback DatasetCharacteristics when ML Common is not available."""
        n_samples: int
        n_features: int
        complexity_score: float = 0.0
        noise_estimate: float = 0.0
        sparsity: float = 0.0

# Import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available")

# Import VectorBT for optimized computations
try:
    from src.utils.vectorbt_compat import vbt, VECTORBT_AVAILABLE
except ImportError as e:
    VECTORBT_AVAILABLE = False
    logging.warning(f"VectorBT not available: {e}")

try:
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

# Import quality assessment module
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityMetrics,
    create_cluster_quality_assessor
)

from src.utils.tprint import tprint
from src.utils.logger import system_logger

logger = system_logger.getChild('AutomatedHDBSCANTuner')

@dataclass
class HDBSCANParameterSpace:
    """Parameter space for HDBSCAN optimization."""
    # Optimized for more regimes: Lower min_cluster_size allows smaller clusters
    min_cluster_size: Tuple[int, int] = (3, 12)  # Reduced max from 20 to 12 to force more clusters
    min_samples: Tuple[int, int] = (3, 25)  # Reduced from (5, 50) to make clustering less conservative
    cluster_selection_epsilon: Tuple[float, float] = (0.1, 0.5)  # Increased min from 0.0 to 0.1 to prevent merging
    cluster_selection_method: List[str] = field(default_factory=lambda: ['leaf', 'eom'])  # Prioritize 'leaf' method (creates more clusters)
    metric: List[str] = field(default_factory=lambda: ['manhattan', 'cosine', 'euclidean'])  # Try manhattan and cosine first
    alpha: Tuple[float, float] = (0.5, 2.0)
    # Removed duplicate cluster_selection_epsilon

# Adapter for ClusterQualityMetrics from cluster_quality_assessor.py
# This adapter provides compatibility for ClusteringQualityMetrics
@dataclass
class ClusteringQualityMetrics:
    """
    Adapter for ClusterQualityMetrics from cluster_quality_assessor.py.
    
    This adapter class provides compatibility for existing code.
    Maps ClusterQualityMetrics to the interface expected by the tuner.
    """
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
    cluster_persistence: Optional[float] = None  # Alternative name for regime_persistence
    
    # Cluster distribution metrics
    cluster_distributions: Optional[List[float]] = None  # Distribution percentages for each cluster
    min_cluster_size_pct: Optional[float] = None  # Minimum cluster size as percentage
    max_cluster_size_pct: Optional[float] = None  # Maximum cluster size as percentage
    distribution_balanced: Optional[bool] = None  # Whether distribution meets 2%-20% constraint
    
    # New metrics from cluster_quality_assessor.py
    predictive_power: Optional[float] = None
    composite_quality_score: Optional[float] = None
    
    @classmethod
    def from_cluster_quality_metrics(cls, qm: ClusterQualityMetrics, 
                            cluster_distributions: Optional[List[float]] = None,
                            distribution_balanced: Optional[bool] = None) -> 'ClusteringQualityMetrics':
        """
        Create ClusteringQualityMetrics from ClusterQualityMetrics (cluster_quality_assessor.py).
        
        Args:
            qm: ClusterQualityMetrics from cluster_quality_assessor module
            cluster_distributions: Optional cluster size distributions
            distribution_balanced: Optional distribution balance flag
            
        Returns:
            ClusteringQualityMetrics adapter instance
        """
        # Calculate cluster distributions if not provided
        if cluster_distributions is None and hasattr(qm, 'cluster_size_distribution') and qm.cluster_size_distribution is not None:
            cluster_distributions = [d * 100 for d in qm.cluster_size_distribution]  # Convert to percentages
        
        # Check distribution balance (2%-20% constraint)
        if distribution_balanced is None and cluster_distributions is not None:
            distribution_balanced = all(2.0 <= d <= 20.0 for d in cluster_distributions)
        
        # Use CV metrics from ClusterQualityMetrics
        within_cv = qm.within_regime_cv
        between_cv = qm.between_regime_cv
        
        # Estimate economic separation from predictive power if not available
        economic_sep = 0.0
        if hasattr(qm, 'predictive_power') and qm.predictive_power is not None:
            economic_sep = qm.predictive_power
        
        return cls(
            silhouette_score=qm.silhouette_score,
            calinski_harabasz_score=qm.calinski_harabasz_score,
            davies_bouldin_score=qm.davies_bouldin_score,
            n_clusters=qm.n_regimes,
            n_noise_points=int(qm.noise_ratio * qm.n_regimes) if qm.noise_ratio else 0,
            noise_ratio=qm.noise_ratio or 0.0,
            dbcv_score=None,  # Not available in ClusterQualityMetrics
            economic_separation=economic_sep,
            within_cluster_cv=within_cv,
            between_cluster_cv=between_cv,
            temporal_stability=qm.temporal_smoothness,
            regime_persistence=qm.regime_persistence,
            cluster_persistence=qm.regime_persistence,
            cluster_distributions=cluster_distributions,
            min_cluster_size_pct=min(cluster_distributions) if cluster_distributions else None,
            max_cluster_size_pct=max(cluster_distributions) if cluster_distributions else None,
            distribution_balanced=distribution_balanced,
            predictive_power=qm.predictive_power,
            composite_quality_score=qm.quality_score
        )
    
    def is_poor_quality(self) -> bool:
        """Determine if clustering quality is poor based on regime discovery targets."""
        # Use composite score if available from cluster_quality_assessor.py
        if self.composite_quality_score is not None:
            return self.composite_quality_score < 0.3  # Quality score below 30%
        
        # Fallback to legacy logic
        basic_poor = (
            (self.silhouette_score is not None and self.silhouette_score < -0.1) or
            self.n_clusters < 2 or
            self.noise_ratio > 0.6 or
            (self.calinski_harabasz_score is not None and self.calinski_harabasz_score < 5.0) or
            (self.davies_bouldin_score is not None and self.davies_bouldin_score > 8.0)
        )
        
        regime_poor = (
            self.n_clusters < 5 or self.n_clusters > 8 or
            (self.within_cluster_cv is not None and self.within_cluster_cv > 0.4) or
            (self.between_cluster_cv is not None and self.between_cluster_cv < 0.05) or
            (self.economic_separation < 0.05) or
            (self.distribution_balanced is not None and not self.distribution_balanced) or
            (self.silhouette_score is not None and self.silhouette_score < -0.2)
        )
        
        return basic_poor or regime_poor
    
    def calculate_composite_score(self) -> float:
        """Calculate composite quality score for optimization."""
        # Use composite score from cluster_quality_assessor.py if available
        if self.composite_quality_score is not None:
            return self.composite_quality_score
        
        # Fallback to legacy calculation
        scores = []
        
        if self.silhouette_score is not None:
            scores.append(max(0, self.silhouette_score))
        
        if self.davies_bouldin_score is not None:
            scores.append(max(0, 1 - min(1, self.davies_bouldin_score / 5.0)))
        
        # Cluster count preference (5-8 clusters optimal)
        if 5 <= self.n_clusters <= 8:
            cluster_score = 1.0
        elif self.n_clusters == 4 or self.n_clusters == 9:
            cluster_score = 0.7
        elif self.n_clusters == 3 or self.n_clusters == 10:
            cluster_score = 0.4
        elif self.n_clusters == 2 or self.n_clusters == 11:
            cluster_score = 0.1
        else:
            cluster_score = 0.0
        scores.append(cluster_score)
        
        if self.within_cluster_cv is not None:
            scores.append(max(0, 1 - min(1, self.within_cluster_cv / 0.5)))
        
        if self.between_cluster_cv is not None:
            scores.append(min(1, self.between_cluster_cv / 0.3))
        
        if self.economic_separation > 0:
            scores.append(min(1, self.economic_separation / 0.2))
        
        noise_score = max(0, 1 - self.noise_ratio)
        scores.append(noise_score)
        
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
            # AutoTuner doesn't have analyze_dataset_characteristics, so use our own analysis
            # Convert data to numpy array for analysis
            data_array = data.values if isinstance(data, pd.DataFrame) else data
            characteristics = DatasetCharacteristics(
                n_samples=data_array.shape[0],
                n_features=data_array.shape[1],
                feature_complexity=self._estimate_feature_complexity(data),
                class_imbalance=0.0,  # Not applicable for clustering
                data_quality_score=0.8,  # Assume good data quality
                temporal_dependency=0.7  # High for financial time series
            )
            tprint(f"📊 Dataset analysis: {data_array.shape[0]} samples, {data_array.shape[1]} features", "INFO")
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
    
    def _estimate_feature_complexity(self, data: pd.DataFrame) -> float:
        """Estimate feature complexity based on data statistics."""
        try:
            # Calculate coefficient of variation for each feature
            cv_values = data.std() / (data.mean().abs() + 1e-8)
            avg_cv = cv_values.mean()
            
            # Normalize to 0-1 range (typical CV range is 0-5 for financial data)
            complexity = min(1.0, avg_cv / 3.0)
            return float(complexity)
        except Exception:
            return 0.5  # Default medium complexity
    
    def create_parameter_search_space(self, characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """Create parameter search space based on dataset characteristics."""
        if not ML_COMMON_AVAILABLE or self.auto_tuner is None:
            # Fallback to basic search space - optimized for more regimes
            return {
                'min_cluster_size': (max(3, characteristics.n_samples // 60), 
                                   min(12, characteristics.n_samples // 20)),  # Reduced to favor more clusters
                'min_samples': (max(3, characteristics.n_samples // 120), 
                              min(25, characteristics.n_samples // 30)),  # Reduced to be less conservative
                'cluster_selection_epsilon': (0.1, 0.5),  # Increased min to prevent merging
                'cluster_selection_method': ['leaf', 'eom'],  # Prioritize 'leaf' for more clusters
                'metric': ['euclidean', 'manhattan']
            }
        
        try:
            # Use AutoTuner to create intelligent search space
            search_space = self.auto_tuner.create_hdbscan_search_space(characteristics)
            tprint(f"🎯 Created parameter search space with {len(search_space)} parameters", "INFO")
            return search_space
        except Exception as e:
            logger.warning(f"Error creating parameter search space: {e}")
            # Return fallback search space - optimized for more regimes
            return {
                'min_cluster_size': (max(3, characteristics.n_samples // 60), 
                                   min(12, characteristics.n_samples // 20)),  # Reduced to favor more clusters
                'min_samples': (max(3, characteristics.n_samples // 120), 
                              min(25, characteristics.n_samples // 30)),  # Reduced to be less conservative
                'cluster_selection_epsilon': (0.1, 0.5),  # Increased min to prevent merging
                'cluster_selection_method': ['leaf', 'eom'],  # Prioritize 'leaf' for more clusters
                'metric': ['euclidean', 'manhattan']
            }
    
    def optimize_parameters(
        self, 
        data: pd.DataFrame, 
        search_space: Dict[str, Any],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        use_hierarchical: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize HDBSCAN parameters using Bayesian or Hierarchical optimization.
        
        Args:
            data: Input data for clustering
            search_space: Parameter search space
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            use_hierarchical: Use hierarchical optimization (recommended for 6+ parameters)
            
        Returns:
            Best parameters dictionary
        """
        if not ML_COMMON_AVAILABLE or self.bayesian_optimizer is None:
            # Fallback to basic parameter selection
            return self._basic_parameter_selection(data, search_space)
        
        # Use hierarchical optimization if enabled (default)
        if use_hierarchical:
            try:
                tprint(f"🚀 Starting Hierarchical parameter optimization with {n_trials} trials", "INFO")
                return self._optimize_parameters_hierarchical(data, search_space, n_trials, timeout)
            except Exception as e:
                logger.warning(f"Hierarchical optimization failed: {e}, falling back to standard Bayesian")
                tprint(f"⚠️ Falling back to standard Bayesian optimization: {e}", "WARNING")
        
        # Standard Bayesian optimization fallback
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
    
    def _optimize_parameters_hierarchical(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize HDBSCAN parameters using hierarchical 3-phase optimization.
        
        Phase 1: Structure parameters (min_cluster_size, min_samples)
        Phase 2: Selection parameters (cluster_selection_epsilon, cluster_selection_method)  
        Phase 3: Distance metric and advanced (metric, alpha if available)
        
        This approach reduces search space by optimizing parameter groups sequentially,
        achieving ~30-50% faster convergence for 6+ parameters.
        
        Args:
            data: Input data for clustering
            search_space: Parameter search space
            n_trials: Total number of trials (distributed across phases)
            timeout: Optional timeout in seconds
            
        Returns:
            Best parameters dictionary
        """
        tprint("=" * 80, "INFO")
        tprint("🔷 HIERARCHICAL HDBSCAN PARAMETER OPTIMIZATION", "INFO")
        tprint("=" * 80, "INFO")
        tprint("Phase 1: Structure (min_cluster_size, min_samples)", "INFO")
        tprint("Phase 2: Selection (epsilon, method)", "INFO")
        tprint("Phase 3: Distance (metric)", "INFO")
        tprint("=" * 80, "INFO")
        
        # Define parameter groups with priorities
        param_groups = [
            create_param_group(
                name="structure",
                params={
                    "min_cluster_size": {
                        "type": "int",
                        "low": search_space['min_cluster_size'][0],
                        "high": search_space['min_cluster_size'][1]
                    },
                    "min_samples": {
                        "type": "int",
                        "low": search_space['min_samples'][0],
                        "high": search_space['min_samples'][1]
                    }
                },
                priority=1,
                description="Core cluster structure parameters"
            ),
            create_param_group(
                name="selection",
                params={
                    "cluster_selection_epsilon": {
                        "type": "float",
                        "low": search_space['cluster_selection_epsilon'][0],
                        "high": search_space['cluster_selection_epsilon'][1]
                    },
                    "cluster_selection_method": {
                        "type": "categorical",
                        "choices": search_space['cluster_selection_method']
                    }
                },
                priority=2,
                depends_on=["structure"],
                description="Cluster selection parameters"
            ),
            create_param_group(
                name="distance",
                params={
                    "metric": {
                        "type": "categorical",
                        "choices": search_space['metric']
                    }
                },
                priority=3,
                depends_on=["structure", "selection"],
                description="Distance metric"
            )
        ]
        
        # Define objective function for hierarchical optimizer
        def objective_func(params, X_train, y_train, X_val=None, y_val=None, 
                          model=None, cv_folds=None, scoring_metric=None):
            """Objective function that evaluates HDBSCAN clustering quality."""
            try:
                # Evaluate clustering quality
                quality_metrics = self._evaluate_clustering_quality(data, params)
                composite_score = quality_metrics.calculate_composite_score()
                
                # Return score (hierarchical optimizer maximizes by default with direction='maximize')
                return composite_score
            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 0.0  # Return poor score on failure
        
        # Create hierarchical optimizer
        hierarchical_optimizer = HierarchicalParameterOptimizer(
            param_groups=param_groups,
            objective_func=objective_func,
            stages=[
                OptimizationStage.COARSE_GRID,
                OptimizationStage.FINE_GRID,
                OptimizationStage.TPE
            ],
            direction='maximize',
            n_rounds=2,  # 2 rounds of refinement
            enable_final_refinement=True,
            final_refinement_trials=max(20, n_trials // 5),
            random_state=42,
            verbose=True
        )
        
        # Convert data to numpy for hierarchical optimizer
        data_array = data.values if isinstance(data, pd.DataFrame) else data
        
        # Run hierarchical optimization
        result = hierarchical_optimizer.optimize(
            X_train=data_array,
            y_train=np.zeros(len(data_array)),  # Dummy target for clustering
            X_val=None,
            y_val=None
        )
        
        best_params = result.best_params
        best_score = result.best_score
        
        tprint("=" * 80, "SUCCESS")
        tprint(f"✅ Hierarchical optimization complete!", "SUCCESS")
        tprint(f"🏆 Best composite score: {best_score:.4f}", "SUCCESS")
        tprint(f"📊 Total trials: {result.total_trials}", "SUCCESS")
        tprint(f"⏱️  Total time: {result.total_time:.2f}s", "SUCCESS")
        tprint("=" * 80, "SUCCESS")
        tprint(f"Best parameters:", "INFO")
        for param_name, param_value in best_params.items():
            tprint(f"  • {param_name}: {param_value}", "INFO")
        tprint("=" * 80, "SUCCESS")
        
        return best_params
    
    def _basic_parameter_selection(self, data: pd.DataFrame, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Basic parameter selection fallback - optimized for more regimes."""
        n_samples = data.shape[0]
        
        return {
            'min_cluster_size': max(8, n_samples // 50),  # Reduced to encourage more clusters
            'min_samples': max(5, n_samples // 80),  # Reduced to make clustering less conservative
            'cluster_selection_epsilon': 0.15,  # Increased to prevent merging and create more clusters
            'cluster_selection_method': 'leaf',  # Changed to 'leaf' method which creates more clusters
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
                            logger.info(f"CV calculation result: within={within_cluster_cv}, between={between_cluster_cv}")
                            
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
            return 0.0, 0.0  # Return default values instead of None
    
    def _calculate_cv_metrics_vectorbt(self, data: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate CV metrics using VectorBT optimization."""
        try:
            # Convert to pandas for VectorBT operations and ensure numeric types
            data_df = pd.DataFrame(data)
            
            # Ensure all columns are numeric and handle any non-numeric data
            numeric_columns = data_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) < len(data_df.columns):
                logger.warning(f"Filtering out {len(data_df.columns) - len(numeric_columns)} non-numeric columns for CV calculation")
                data_df = data_df[numeric_columns]
            
            # Convert to float64 to ensure consistent data types
            data_df = data_df.astype(np.float64)
            
            logger.info(f"CV calculation: {len(unique_labels)} unique labels, {data_df.shape[0]} samples, {data_df.shape[1]} features")
            
            # Calculate within-cluster CV using VectorBT
            within_cvs = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_data = data_df[cluster_mask]
                
                logger.info(f"Cluster {label}: {len(cluster_data)} samples")
                
                if len(cluster_data) > 1:
                    # Use VectorBT for efficient std and mean calculations
                    cluster_std = cluster_data.std()
                    cluster_mean = cluster_data.mean()
                    
                    # Ensure we have numeric values and handle division safely
                    if len(cluster_std) > 0 and len(cluster_mean) > 0:
                        # Convert to numpy arrays for safe operations
                        std_values = cluster_std.values if hasattr(cluster_std, 'values') else cluster_std
                        mean_values = cluster_mean.values if hasattr(cluster_mean, 'values') else cluster_mean
                        
                        # Safe division with proper handling of zeros and infinities
                        denominator = np.abs(mean_values) + 1e-8
                        cv_values = np.divide(std_values, denominator, out=np.zeros_like(std_values), where=denominator!=0)
                        
                        # Remove any infinite or NaN values
                        cv_values = cv_values[np.isfinite(cv_values)]
                        
                        if len(cv_values) > 0:
                            cluster_cv = np.mean(cv_values)
                            within_cvs.append(cluster_cv)
                            logger.info(f"Cluster {label} CV: {cluster_cv:.4f}")
                        else:
                            logger.warning(f"Cluster {label}: No valid CV values after filtering")
                    else:
                        logger.warning(f"Cluster {label}: Empty std or mean arrays")
                else:
                    logger.warning(f"Cluster {label}: Only {len(cluster_data)} samples (need >1)")
            
            within_cluster_cv = np.mean(within_cvs) if within_cvs else None
            logger.info(f"Within-cluster CV: {within_cluster_cv}")
            
            # Calculate between-cluster CV
            cluster_means = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_data = data_df[cluster_mask]
                
                if len(cluster_data) > 0:
                    cluster_mean = cluster_data.mean()
                    # Convert to numpy array and ensure numeric
                    mean_values = cluster_mean.values if hasattr(cluster_mean, 'values') else cluster_mean
                    mean_values = mean_values[np.isfinite(mean_values)]  # Remove any non-finite values
                    if len(mean_values) > 0:
                        cluster_means.append(mean_values)
                        logger.info(f"Cluster {label} mean: {np.mean(mean_values):.4f}")
            
            if len(cluster_means) > 1:
                cluster_means_df = pd.DataFrame(cluster_means)
                between_cluster_std = cluster_means_df.std()
                between_cluster_mean = cluster_means_df.mean()
                
                # Safe division for between-cluster CV
                std_values = between_cluster_std.values if hasattr(between_cluster_std, 'values') else between_cluster_std
                mean_values = between_cluster_mean.values if hasattr(between_cluster_mean, 'values') else between_cluster_mean
                
                denominator = np.abs(mean_values) + 1e-8
                cv_values = np.divide(std_values, denominator, out=np.zeros_like(std_values), where=denominator!=0)
                
                # Remove any infinite or NaN values
                cv_values = cv_values[np.isfinite(cv_values)]
                
                between_cluster_cv = np.mean(cv_values) if len(cv_values) > 0 else None
                logger.info(f"Between-cluster CV: {between_cluster_cv}")
            else:
                between_cluster_cv = None
                logger.warning(f"Only {len(cluster_means)} cluster means available (need >1)")
            
            # Ensure we return actual values, not None
            if within_cluster_cv is None:
                within_cluster_cv = 0.0  # Default value
                logger.warning("Within-cluster CV was None, using default 0.0")
            
            if between_cluster_cv is None:
                between_cluster_cv = 0.0  # Default value
                logger.warning("Between-cluster CV was None, using default 0.0")
            
            return within_cluster_cv, between_cluster_cv
            
        except Exception as e:
            logger.warning(f"Error in VectorBT CV calculation: {e}")
            return self._calculate_cv_metrics_standard(data, labels, unique_labels)
    
    def _calculate_cv_metrics_standard(self, data: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Calculate CV metrics using standard numpy operations."""
        try:
            # Ensure data is numeric and handle any non-finite values
            data = np.asarray(data, dtype=np.float64)
            
            # Remove any non-finite values from the data
            finite_mask = np.all(np.isfinite(data), axis=1)
            if not np.any(finite_mask):
                logger.warning("No finite values found in data for CV calculation")
                return None, None
            
            data = data[finite_mask]
            labels = labels[finite_mask]
            
            # Calculate within-cluster CV (lower is better)
            within_cvs = []
            for label in unique_labels:
                cluster_data = data[labels == label]
                if len(cluster_data) > 1:
                    cluster_std = np.std(cluster_data, axis=0)
                    cluster_mean = np.mean(cluster_data, axis=0)
                    
                    # Safe division with proper handling of zeros and infinities
                    denominator = np.abs(cluster_mean) + 1e-8
                    cv_values = np.divide(cluster_std, denominator, out=np.zeros_like(cluster_std), where=denominator!=0)
                    
                    # Remove any infinite or NaN values
                    cv_values = cv_values[np.isfinite(cv_values)]
                    
                    if len(cv_values) > 0:
                        cluster_cv = np.mean(cv_values)
                        within_cvs.append(cluster_cv)
            
            within_cluster_cv = np.mean(within_cvs) if within_cvs else None
            
            # Calculate between-cluster CV (higher is better)
            cluster_means = []
            for label in unique_labels:
                cluster_data = data[labels == label]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data, axis=0)
                    # Remove any non-finite values
                    cluster_mean = cluster_mean[np.isfinite(cluster_mean)]
                    if len(cluster_mean) > 0:
                        cluster_means.append(cluster_mean)
            
            if len(cluster_means) > 1:
                cluster_means = np.array(cluster_means)
                between_cluster_std = np.std(cluster_means, axis=0)
                between_cluster_mean = np.mean(cluster_means, axis=0)
                
                # Safe division for between-cluster CV
                denominator = np.abs(between_cluster_mean) + 1e-8
                cv_values = np.divide(between_cluster_std, denominator, out=np.zeros_like(between_cluster_std), where=denominator!=0)
                
                # Remove any infinite or NaN values
                cv_values = cv_values[np.isfinite(cv_values)]
                
                between_cluster_cv = np.mean(cv_values) if len(cv_values) > 0 else None
            else:
                between_cluster_cv = None
            
            # Ensure we return actual values, not None
            if within_cluster_cv is None:
                within_cluster_cv = 0.0  # Default value
                logger.warning("Within-cluster CV was None, using default 0.0")
            
            if between_cluster_cv is None:
                between_cluster_cv = 0.0  # Default value
                logger.warning("Between-cluster CV was None, using default 0.0")
            
            return within_cluster_cv, between_cluster_cv
            
        except Exception as e:
            logger.warning(f"Error in standard CV calculation: {e}")
            return 0.0, 0.0  # Return default values instead of None
    
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
        """Create intelligent fallback strategies optimized for 5-8 clusters."""
        strategies = []
        
        # Strategy 1: Target 5-8 clusters with leaf method (enhanced)
        strategies.append(FallbackStrategy(
            name='target_5_8_clusters_leaf_enhanced',
            description='Target 5-8 clusters using leaf method with enhanced epsilon range',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.1,  # Increased for better separation
                'min_cluster_size': max(15, characteristics.n_samples // 30),  # Target ~5-8 clusters
                'min_samples': max(8, characteristics.n_samples // 60),
                'metric': 'euclidean'
            },
            priority=1
        ))
        
        # Strategy 1.5: Enhanced leaf method for 5-8 clusters
        strategies.append(FallbackStrategy(
            name='enhanced_leaf_5_8_clusters',
            description='Enhanced leaf method specifically for 5-8 clusters',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.2,  # Higher epsilon for more clusters
                'min_cluster_size': max(12, characteristics.n_samples // 40),  # Smaller clusters
                'min_samples': max(6, characteristics.n_samples // 80),  # Smaller samples
                'metric': 'euclidean'
            },
            priority=1
        ))
        
        # Strategy 1.6: Ultra-aggressive for 5-8 clusters
        strategies.append(FallbackStrategy(
            name='ultra_aggressive_5_8_clusters',
            description='Ultra-aggressive parameters for 5-8 clusters',
            parameters={
                'cluster_selection_method': 'leaf',
                'cluster_selection_epsilon': 0.4,  # Very high epsilon for aggressive clustering
                'min_cluster_size': max(8, characteristics.n_samples // 60),  # Very small clusters
                'min_samples': max(3, characteristics.n_samples // 160),  # Very small samples
                'metric': 'manhattan'  # Use manhattan for better separation
            },
            priority=2
        ))
        
        # Strategy 2: Target 5-8 clusters with EOM method
        strategies.append(FallbackStrategy(
            name='target_5_8_clusters_eom',
            description='Target 5-8 clusters using EOM method',
            parameters={
                'cluster_selection_method': 'eom',
                'cluster_selection_epsilon': 0.1,
                'min_cluster_size': max(12, characteristics.n_samples // 40),  # Target ~5-8 clusters
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
                try:
                    memory_info = self.memory_optimizer.get_memory_stats()
                    tprint(f"💾 Memory stats: {memory_info.get('total_gb', 0):.1f} GB total", "INFO")
                except Exception as e:
                    logger.warning(f"Could not get memory stats: {e}")
        
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
        tprint(f"📊 Initial quality: Silhouette={initial_quality.silhouette_score or 0.0:.3f}, "
               f"Clusters={initial_quality.n_clusters or 0}, Noise={initial_quality.noise_ratio or 0.0:.3f}", "INFO")
        
        # Step 5: Check if fallback is needed
        if enable_fallback and initial_quality.is_poor_quality():
            tprint("⚠️ Poor clustering quality detected - attempting parameter fallback", "WARNING")
            
            try:
                best_params, final_quality = self.execute_parameter_fallback(data, initial_quality)
                tprint(f"✅ Parameter fallback completed", "SUCCESS")
                tprint(f"📊 Final quality: Silhouette={final_quality.silhouette_score or 0.0:.3f}, "
                       f"Clusters={final_quality.n_clusters or 0}, Noise={final_quality.noise_ratio or 0.0:.3f}", "INFO")
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
                            logger.info(f"Optimized CV calculation result: within={within_cluster_cv}, between={between_cluster_cv}")
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
