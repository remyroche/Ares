"""
Label Fusion Module

Provides regime optimization and label fusion services for production use.
Implements advanced optimization algorithms, intelligent label fusion, and comprehensive validation.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import json
from datetime import datetime

# Import optimization and validation utilities
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from scipy.optimize import minimize
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

# Import project utilities
from src.utils.logger import get_logger
from src.training.steps.market_analysis.shared_utils.core import (
    validate_regime_count, normalize_weights, validate_algorithm_type
)

logger = get_logger(__name__)


class OptimizationMethod(Enum):
    """Optimization methods for regime detection."""
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    RANDOM_SEARCH = "random_search"


class FusionMethod(Enum):
    """Label fusion methods."""
    MAJORITY_VOTING = "majority_voting"
    WEIGHTED_AVERAGE = "weighted_average"
    DAWID_SKENE = "dawid_skene"
    CONSENSUS_CLUSTERING = "consensus_clustering"
    BAYESIAN_FUSION = "bayesian_fusion"


@dataclass
class OptimizationConfig:
    """Configuration for regime optimization."""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    n_regimes_range: Tuple[int, int] = (3, 8)
    algorithms: List[str] = field(default_factory=lambda: ['kmeans', 'gmm', 'agglomerative'])
    max_iterations: int = 100
    cv_folds: int = 5
    random_state: int = 42
    quality_threshold: float = 0.6
    enable_parallel: bool = True


@dataclass
class FusionConfig:
    """Configuration for label fusion."""
    method: FusionMethod = FusionMethod.WEIGHTED_AVERAGE
    max_iterations: int = 50
    tolerance: float = 1e-6
    confidence_threshold: float = 0.7
    enable_quality_weighting: bool = True
    enable_temporal_smoothing: bool = True


@dataclass
class ValidationConfig:
    """Configuration for regime validation."""
    min_regime_persistence: float = 0.7
    max_feature_noise_ratio: float = 0.3
    min_temporal_stability: float = 0.6
    min_samples_per_regime: int = 10
    max_regime_count: int = 15
    enable_economic_validation: bool = True


@dataclass
class OptimizationResult:
    """Result from regime optimization."""
    success: bool
    optimized_parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]
    optimization_time: float
    algorithm_used: str
    n_regimes: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class FusionResult:
    """Result from label fusion."""
    success: bool
    fused_labels: np.ndarray
    confidence_scores: np.ndarray
    fusion_weights: List[float]
    quality_improvement: float
    method_used: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result from regime validation."""
    valid: bool
    quality_score: float
    regime_count: int
    regime_statistics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RegimeOptimizationService:
    """Production-ready service for regime optimization and label fusion."""
    
    def __init__(self, 
                 optimization_config: Optional[OptimizationConfig] = None,
                 fusion_config: Optional[FusionConfig] = None,
                 validation_config: Optional[ValidationConfig] = None):
        """Initialize the regime optimization service."""
        self.optimization_config = optimization_config or OptimizationConfig()
        self.fusion_config = fusion_config or FusionConfig()
        self.validation_config = validation_config or ValidationConfig()
        self.logger = get_logger('RegimeOptimizationService')
        
        # Initialize scaler for feature normalization
        self.scaler = RobustScaler()
        self.is_fitted = False
        
        self.logger.info("Regime optimization service initialized with production-ready algorithms")
    
    def optimize_regimes(self, data: Union[np.ndarray, pd.DataFrame], 
                        config: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize regime detection parameters using advanced algorithms.
        
        Args:
            data: Input data (features or DataFrame)
            config: Optional configuration overrides
            
        Returns:
            OptimizationResult with optimal parameters and quality metrics
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting advanced regime optimization")
            
            # Prepare data
            features = self._prepare_features(data)
            if features is None or features.shape[0] < 10:
                return OptimizationResult(
                    success=False,
                    optimized_parameters={},
                    quality_metrics={},
                    optimization_time=0.0,
                    algorithm_used="none",
                    n_regimes=0,
                    errors=["Insufficient data for optimization"]
                )
            
            # Merge config
            opt_config = self._merge_config(config)
            
            # Create objective function
            objective_function = create_optimization_objective(
                features, 
                opt_config['algorithms']
            )
            
            # Define parameter bounds
            parameter_bounds = {
                'n_clusters': (opt_config['n_regimes_range'][0], opt_config['n_regimes_range'][1])
            }
            categorical_params = {
                'algorithm': opt_config['algorithms']
            }
            
            # Perform optimization based on method
            if opt_config['method'] == OptimizationMethod.GRID_SEARCH:
                optimizer = AdvancedGridSearch()
                best_params = optimizer.optimize(objective_function, parameter_bounds, categorical_params)
            elif opt_config['method'] == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                bayesian_config = BayesianOptimizationConfig(
                    n_iterations=opt_config['max_iterations'],
                    random_state=opt_config['random_state']
                )
                optimizer = BayesianOptimizer(bayesian_config)
                best_params = optimizer.optimize(objective_function, parameter_bounds, categorical_params)
            elif opt_config['method'] == OptimizationMethod.EVOLUTIONARY:
                evolutionary_config = EvolutionaryConfig(
                    n_generations=opt_config['max_iterations'],
                    random_state=opt_config['random_state']
                )
                optimizer = EvolutionaryOptimizer(evolutionary_config)
                best_params = optimizer.optimize(objective_function, parameter_bounds, categorical_params)
            else:  # RANDOM_SEARCH
                result = self._random_search_optimization(features, opt_config)
                best_params = result.optimized_parameters
            
            # Calculate final metrics
            if best_params:
                final_score = objective_function(best_params)
                
                # Create final model and calculate detailed metrics
                final_model = self._create_model(best_params['algorithm'], best_params)
                final_labels = final_model.fit_predict(features)
                
                quality_metrics = {
                    'silhouette_score': silhouette_score(features, final_labels),
                    'calinski_harabasz_score': calinski_harabasz_score(features, final_labels),
                    'davies_bouldin_score': davies_bouldin_score(features, final_labels),
                    'combined_score': final_score
                }
                
                result = OptimizationResult(
                    success=True,
                    optimized_parameters=best_params,
                    quality_metrics=quality_metrics,
                    optimization_time=0.0,  # Will be set below
                    algorithm_used=best_params['algorithm'],
                    n_regimes=best_params['n_clusters']
                )
            else:
                result = OptimizationResult(
                    success=False,
                    optimized_parameters={},
                    quality_metrics={},
                    optimization_time=0.0,
                    algorithm_used="none",
                    n_regimes=0,
                    errors=["No valid parameters found"]
                )
            
            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time
            
            self.logger.info(f"Regime optimization completed in {optimization_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Regime optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimized_parameters={},
                quality_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                algorithm_used="none",
                n_regimes=0,
                errors=[str(e)]
            )
    
    def fuse_labels(self, labels: List[np.ndarray], 
                   weights: Optional[List[float]] = None,
                   quality_scores: Optional[List[float]] = None,
                   temporal_data: Optional[np.ndarray] = None) -> FusionResult:
        """
        Fuse multiple label sets using advanced algorithms.
        
        Args:
            labels: List of label arrays
            weights: Optional weights for each label set
            quality_scores: Optional quality scores for weighting
            temporal_data: Optional temporal data for smoothing
            
        Returns:
            FusionResult with fused labels and confidence scores
        """
        try:
            if not labels:
                return FusionResult(
                    success=False,
                    fused_labels=np.array([]),
                    confidence_scores=np.array([]),
                    fusion_weights=[],
                    quality_improvement=0.0,
                    method_used="none",
                    errors=["No labels provided"]
                )
            
            if len(labels) == 1:
                return FusionResult(
                    success=True,
                    fused_labels=labels[0],
                    confidence_scores=np.ones(len(labels[0])),
                    fusion_weights=[1.0],
                    quality_improvement=0.0,
                    method_used="single_label"
                )
            
            self.logger.info(f"Fusing {len(labels)} label sets using {self.fusion_config.method.value}")
            
            # Use advanced fusion engine
            fusion_engine = LabelFusionEngine(
                algorithm=FusionAlgorithm(self.fusion_config.method.value)
            )
            
            # Perform fusion
            fused_labels, confidence_scores, fusion_metrics = fusion_engine.fuse_labels(
                labels, weights, quality_scores, temporal_data
            )
            
            # Calculate quality improvement
            quality_improvement = fusion_metrics.overall_quality
            
            # Prepare fusion weights
            fusion_weights = self._prepare_fusion_weights(weights, quality_scores, len(labels))
            
            self.logger.info(f"Label fusion completed with {quality_improvement:.3f} quality improvement")
            
            return FusionResult(
                success=True,
                fused_labels=fused_labels,
                confidence_scores=confidence_scores,
                fusion_weights=fusion_weights,
                quality_improvement=quality_improvement,
                method_used=self.fusion_config.method.value
            )
            
        except Exception as e:
            self.logger.error(f"Label fusion failed: {e}")
            return FusionResult(
                success=False,
                fused_labels=np.array([]),
                confidence_scores=np.array([]),
                fusion_weights=[],
                quality_improvement=0.0,
                method_used="none",
                errors=[str(e)]
            )
    
    def validate_regimes(self, regimes: np.ndarray, 
                        data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                        market_data: Optional[pd.DataFrame] = None,
                        temporal_data: Optional[np.ndarray] = None) -> ValidationResult:
        """
        Validate regime quality using comprehensive metrics.
        
        Args:
            regimes: Regime labels array
            data: Optional input data for validation
            market_data: Optional market data for economic validation
            temporal_data: Optional temporal data for stability analysis
            
        Returns:
            ValidationResult with comprehensive validation metrics
        """
        try:
            self.logger.info(f"Validating {len(np.unique(regimes))} regimes")
            
            # Use advanced validation system
            validator = RegimeValidator(self.validation_config.__dict__)
            
            # Perform comprehensive validation
            quality_metrics = validator.validate_regimes(
                regimes, data, market_data, temporal_data
            )
            
            # Convert to ValidationResult format
            regime_statistics = {
                'n_regimes': quality_metrics.n_regimes,
                'regime_sizes': quality_metrics.regime_sizes,
                'regime_balance': quality_metrics.regime_balance,
                'silhouette_score': quality_metrics.silhouette_score,
                'calinski_harabasz_score': quality_metrics.calinski_harabasz_score,
                'davies_bouldin_score': quality_metrics.davies_bouldin_score,
                'persistence_score': quality_metrics.persistence_score,
                'stability_score': quality_metrics.stability_score,
                'transition_rate': quality_metrics.transition_rate,
                'economic_consistency': quality_metrics.economic_consistency,
                'volatility_separation': quality_metrics.volatility_separation,
                'return_separation': quality_metrics.return_separation
            }
            
            self.logger.info(f"Regime validation completed - Valid: {quality_metrics.validation_passed}, Quality: {quality_metrics.overall_quality:.3f}")
            
            return ValidationResult(
                valid=quality_metrics.validation_passed,
                quality_score=quality_metrics.overall_quality,
                regime_count=quality_metrics.n_regimes,
                regime_statistics=regime_statistics,
                warnings=quality_metrics.warnings,
                errors=quality_metrics.critical_issues
            )
            
        except Exception as e:
            self.logger.error(f"Regime validation failed: {e}")
            return ValidationResult(
                valid=False,
                quality_score=0.0,
                regime_count=0,
                regime_statistics={},
                errors=[str(e)]
            )
    
    def _prepare_features(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[np.ndarray]:
        """Prepare and normalize features for optimization."""
        try:
            if isinstance(data, pd.DataFrame):
                # Extract numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    self.logger.warning("No numeric columns found in DataFrame")
                    return None
                features = data[numeric_cols].values
            else:
                features = np.array(data)
            
            # Check for valid data
            if features.size == 0 or np.isnan(features).all():
                self.logger.warning("Invalid or empty feature data")
                return None
            
            # Handle missing values
            if np.isnan(features).any():
                self.logger.warning("Handling missing values in features")
                features = np.nan_to_num(features, nan=0.0)
            
            # Normalize features
            if not self.is_fitted:
                features = self.scaler.fit_transform(features)
                self.is_fitted = True
            else:
                features = self.scaler.transform(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature preparation failed: {e}")
            return None
    
    def _merge_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge provided config with default configuration."""
        merged = {
            'method': self.optimization_config.method,
            'n_regimes_range': self.optimization_config.n_regimes_range,
            'algorithms': self.optimization_config.algorithms,
            'max_iterations': self.optimization_config.max_iterations,
            'cv_folds': self.optimization_config.cv_folds,
            'random_state': self.optimization_config.random_state,
            'quality_threshold': self.optimization_config.quality_threshold,
            'enable_parallel': self.optimization_config.enable_parallel
        }
        
        if config:
            merged.update(config)
        
        return merged
    
    
    
    def _create_model(self, algorithm: str, params: Dict[str, Any]):
        """Create clustering model with given parameters."""
        if algorithm == 'kmeans':
            return KMeans(**params)
        elif algorithm == 'gmm':
            return GaussianMixture(**params)
        elif algorithm == 'agglomerative':
            return AgglomerativeClustering(**params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _prepare_fusion_weights(self, weights: Optional[List[float]], 
                               quality_scores: Optional[List[float]], 
                               n_labels: int) -> List[float]:
        """Prepare fusion weights."""
        if weights is not None and len(weights) == n_labels:
            return normalize_weights(weights)
        
        if quality_scores is not None and len(quality_scores) == n_labels:
            return normalize_weights(quality_scores)
        
        # Default to equal weights
        return [1.0 / n_labels] * n_labels


# Import advanced modules
from .optimization_algorithms import (
    BayesianOptimizer, EvolutionaryOptimizer, AdvancedGridSearch,
    create_optimization_objective, BayesianOptimizationConfig, EvolutionaryConfig
)
from .fusion_algorithms import (
    LabelFusionEngine, FusionAlgorithm, FusionMetrics
)
from .validation_metrics import (
    RegimeValidator, RegimeQualityMetrics, ValidationType, ValidationResult
)

# Export the main class and supporting classes
__all__ = [
    'RegimeOptimizationService',
    'OptimizationMethod',
    'FusionMethod',
    'OptimizationConfig',
    'FusionConfig',
    'ValidationConfig',
    'OptimizationResult',
    'FusionResult',
    'ValidationResult',
    # Advanced modules
    'BayesianOptimizer',
    'EvolutionaryOptimizer', 
    'AdvancedGridSearch',
    'LabelFusionEngine',
    'RegimeValidator',
    'RegimeQualityMetrics',
    'FusionAlgorithm',
    'FusionMetrics',
    'ValidationType',
    'create_optimization_objective'
]