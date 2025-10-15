"""
Feature Generation Optimization

This module provides data-driven optimization for feature generation parameters,
particularly lookback periods for time-series features. It uses statistical
analysis and cross-validation to determine optimal parameters for each feature.

Key Features:
- Data-driven lookback period optimization
- Feature performance analysis across different time windows
- Feature stability assessment
- Optimal feature window selection
- Feature decay analysis
- Cross-validation for feature parameters
- Regime-aware feature optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from src.utils.tprint import tprint
from datetime import datetime, timedelta
import logging
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings
from dataclasses import dataclass
from enum import Enum

from ..utils.math_validation import safe_divide, safe_log
from src.utils.common_operations import create_fallback_logger
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.parallel_processing_optimizer import ParallelProcessor

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited optimization functionality")

class OptimizationMethod(Enum):
    """Optimization methods for feature parameters."""
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    INFORMATION_THEORY = "information_theory"
    REGIME_AWARE = "regime_aware"
    ADAPTIVE = "adaptive"

@dataclass
class FeatureOptimizationConfig:
    """Configuration for feature optimization."""
    min_lookback: int = 5
    max_lookback: int = 252  # 1 year of daily data
    step_size: int = 1
    optimization_method: OptimizationMethod = OptimizationMethod.CROSS_VALIDATION
    cv_folds: int = 5
    stability_threshold: float = 0.8
    performance_threshold: float = 0.6
    regime_aware: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    chunk_size: int = 1000
    # Add methods parameter for backward compatibility
    methods: Optional[List[str]] = None
    optimization_metric: str = "sharpe_ratio"
    
    # Stability Enhancement Parameters
    l1_regularization: float = 0.01  # L1 regularization for feature selection
    l2_regularization: float = 0.001  # L2 regularization for stability
    max_lookback_variance: float = 0.2  # Maximum variance between feature lookbacks
    lookback_range_penalty: float = 0.1  # Penalty for wide lookback ranges
    temporal_consistency_weight: float = 0.3  # Weight for temporal consistency
    stability_weight: float = 0.4  # Balance performance vs stability
    
    # Rolling Window Parameters
    rolling_window_size: str = "30D"  # Rolling window size for optimization
    rolling_step_size: str = "7D"  # Step size for rolling optimization
    min_stability_score: float = 0.7  # Minimum required stability score
    
    # Cross-Validation Stability Parameters
    cv_stability_metric: str = "coefficient_variance"  # Stability metric for CV
    stability_cv_folds: int = 3  # Additional CV folds for stability assessment

@dataclass
class FeatureOptimizationResult:
    """Result of feature optimization."""
    feature_name: str
    optimal_lookback: int
    performance_score: float
    stability_score: float
    confidence_interval: Tuple[float, float]
    optimization_method: str
    regime_specific_results: Optional[Dict[str, Any]] = None
    decay_analysis: Optional[Dict[str, Any]] = None
    validation_scores: Optional[List[float]] = None

class FeatureGenerationOptimizer:
    """
    Optimizes feature generation parameters using data-driven approaches.
    
    This class provides comprehensive optimization for feature parameters,
    particularly lookback periods, using various statistical and machine learning
    methods to determine optimal values for each feature.
    """
    
    def __init__(self, config: Optional[FeatureOptimizationConfig] = None):
        """Initialize the feature generation optimizer."""
        self.logger = logger.getChild('FeatureGenerationOptimizer')
        self.logger.info("🚀 Initializing FeatureGenerationOptimizer...")
        start_time = time.time()
        
        self.config = config or FeatureOptimizationConfig()
        self.logger.info(f"📊 Configuration loaded: {self.config.optimization_method.value}")
        
        # Initialize components
        self.logger.debug("🔧 Initializing GPU manager...")
        try:
            self.gpu_manager = M1GPUManager() if self.config.parallel_processing else None
            if self.gpu_manager:
                self.logger.debug("✅ GPU manager initialized")
            else:
                self.logger.debug("ℹ️ GPU manager not initialized (parallel processing disabled)")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize GPU manager: {e}")
            self.gpu_manager = None
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ FeatureGenerationOptimizer initialized in {init_time:.3f}s")
        self.logger.info(f"📊 Min lookback: {self.config.min_lookback}, Max lookback: {self.config.max_lookback}")
        self.logger.info(f"📊 CV folds: {self.config.cv_folds}, Parallel processing: {self.config.parallel_processing}")
        self.parallel_processor = ParallelProcessor(max_workers=self.config.max_workers)
        
        # Cache for optimization results
        self._optimization_cache: Dict[str, FeatureOptimizationResult] = {}
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the optimization configuration."""
        if self.config.min_lookback >= self.config.max_lookback:
            raise ValueError("min_lookback must be less than max_lookback")
        
        if self.config.step_size <= 0:
            raise ValueError("step_size must be positive")
        
        if not SKLEARN_AVAILABLE and self.config.optimization_method == OptimizationMethod.CROSS_VALIDATION:
            self.logger.warning("Scikit-learn not available, falling back to statistical analysis")
            self.config.optimization_method = OptimizationMethod.STATISTICAL_ANALYSIS
    
    async def optimize_feature_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable[[pd.DataFrame, int], pd.Series],
        regime_column: Optional[str] = None
    ) -> FeatureOptimizationResult:
        """
        Optimize the lookback period for a specific feature.
        
        Args:
            data: Input data DataFrame
            feature_name: Name of the feature to optimize
            target_column: Name of the target column
            feature_generator: Function that generates the feature given data and lookback
            regime_column: Optional regime column for regime-aware optimization
            
        Returns:
            FeatureOptimizationResult with optimal parameters
        """
        self.logger.info(f"Optimizing lookback period for feature: {feature_name}")
        
        # Check cache first
        cache_key = f"{feature_name}_{hash(str(data.shape))}"
        if cache_key in self._optimization_cache:
            self.logger.info(f"Using cached optimization result for {feature_name}")
            return self._optimization_cache[cache_key]
        
        try:
            # Generate lookback range
            lookback_range = range(
                self.config.min_lookback,
                self.config.max_lookback + 1,
                self.config.step_size
            )
            
            # Optimize based on method
            if self.config.optimization_method == OptimizationMethod.CROSS_VALIDATION:
                result = await self._optimize_with_cross_validation(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.STATISTICAL_ANALYSIS:
                result = await self._optimize_with_statistical_analysis(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.INFORMATION_THEORY:
                result = await self._optimize_with_information_theory(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            elif self.config.optimization_method == OptimizationMethod.REGIME_AWARE:
                result = await self._optimize_with_regime_awareness(
                    data, feature_name, target_column, feature_generator, lookback_range, regime_column
                )
            else:
                result = await self._optimize_adaptive(
                    data, feature_name, target_column, feature_generator, lookback_range
                )
            
            # Add regime-specific analysis if regime column provided
            if regime_column and regime_column in data.columns:
                result.regime_specific_results = await self._analyze_regime_specific_performance(
                    data, feature_name, target_column, feature_generator, result.optimal_lookback, regime_column
                )
            
            # Add decay analysis
            result.decay_analysis = await self._analyze_feature_decay(
                data, feature_name, feature_generator, result.optimal_lookback
            )
            
            # Cache result
            self._optimization_cache[cache_key] = result
            
            self.logger.info(f"Optimization completed for {feature_name}: optimal_lookback={result.optimal_lookback}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing feature {feature_name}: {e}")
            raise
    
    async def _optimize_with_cross_validation(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using cross-validation approach."""
        self.logger.info(f"Using cross-validation optimization for {feature_name}")
        
        best_score = -np.inf
        best_lookback = self.config.min_lookback
        validation_scores = []
        
        for lookback in lookback_range:
            try:
                # Generate feature with current lookback
                feature_values = feature_generator(data, lookback)
                
                # Prepare data for cross-validation
                valid_indices = ~(feature_values.isna() | data[target_column].isna())
                X = feature_values[valid_indices].values.reshape(-1, 1)
                y = data[target_column][valid_indices].values
                
                if len(X) < 10:  # Need minimum data for CV
                    continue
                
                # Perform time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_val)
                    score = -mean_squared_error(y_val, y_pred)  # Negative MSE for maximization
                    scores.append(score)
                
                avg_score = np.mean(scores)
                validation_scores.append(avg_score)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_lookback = lookback
                    
            except Exception as e:
                self.logger.warning(f"Error in cross-validation for lookback {lookback}: {e}")
                continue
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(validation_scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(validation_scores)
        
        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.CROSS_VALIDATION.value,
            validation_scores=validation_scores
        )
    
    async def _optimize_with_statistical_analysis(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using statistical analysis approach."""
        self.logger.info(f"Using statistical analysis optimization for {feature_name}")
        
        best_score = -np.inf
        best_lookback = self.config.min_lookback
        scores = []
        
        for lookback in lookback_range:
            try:
                # Generate feature with current lookback
                feature_values = feature_generator(data, lookback)
                
                # Calculate correlation with target
                valid_indices = ~(feature_values.isna() | data[target_column].isna())
                if valid_indices.sum() < 10:
                    continue
                
                correlation = abs(feature_values[valid_indices].corr(data[target_column][valid_indices]))
                
                # Calculate feature stability (low variance is better)
                feature_std = feature_values[valid_indices].std()
                feature_mean = feature_values[valid_indices].mean()
                stability = 1 / (1 + feature_std / abs(feature_mean)) if feature_mean != 0 else 0
                
                # Combined score
                score = correlation * stability
                scores.append(score)
                
                if score > best_score:
                    best_score = score
                    best_lookback = lookback
                    
            except Exception as e:
                self.logger.warning(f"Error in statistical analysis for lookback {lookback}: {e}")
                continue
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)
        
        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS.value,
            validation_scores=scores
        )
    
    async def _optimize_with_information_theory(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Optimize using information theory approach."""
        self.logger.info(f"Using information theory optimization for {feature_name}")
        
        best_score = -np.inf
        best_lookback = self.config.min_lookback
        scores = []
        
        for lookback in lookback_range:
            try:
                # Generate feature with current lookback
                feature_values = feature_generator(data, lookback)
                
                # Calculate mutual information
                valid_indices = ~(feature_values.isna() | data[target_column].isna())
                if valid_indices.sum() < 10:
                    continue
                
                # Discretize for mutual information calculation
                feature_discrete = pd.cut(feature_values[valid_indices], bins=10, labels=False)
                target_discrete = pd.cut(data[target_column][valid_indices], bins=10, labels=False)
                
                # Calculate mutual information
                mi_score = self._calculate_mutual_information(feature_discrete, target_discrete)
                scores.append(mi_score)
                
                if mi_score > best_score:
                    best_score = mi_score
                    best_lookback = lookback
                    
            except Exception as e:
                self.logger.warning(f"Error in information theory analysis for lookback {lookback}: {e}")
                continue
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)
        
        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.INFORMATION_THEORY.value,
            validation_scores=scores
        )
    
    async def _optimize_with_regime_awareness(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range,
        regime_column: str
    ) -> FeatureOptimizationResult:
        """Optimize using regime-aware approach."""
        self.logger.info(f"Using regime-aware optimization for {feature_name}")
        
        regime_results = {}
        overall_scores = []
        
        # Get unique regimes
        regimes = data[regime_column].unique()
        
        for regime in regimes:
            regime_data = data[data[regime_column] == regime]
            if len(regime_data) < 20:  # Need minimum data per regime
                continue
            
            regime_scores = []
            best_regime_score = -np.inf
            best_regime_lookback = self.config.min_lookback
            
            for lookback in lookback_range:
                try:
                    # Generate feature for this regime
                    feature_values = feature_generator(regime_data, lookback)
                    
                    # Calculate performance for this regime
                    valid_indices = ~(feature_values.isna() | regime_data[target_column].isna())
                    if valid_indices.sum() < 5:
                        continue
                    
                    correlation = abs(feature_values[valid_indices].corr(regime_data[target_column][valid_indices]))
                    regime_scores.append(correlation)
                    
                    if correlation > best_regime_score:
                        best_regime_score = correlation
                        best_regime_lookback = lookback
                        
                except Exception as e:
                    self.logger.warning(f"Error in regime-aware analysis for regime {regime}, lookback {lookback}: {e}")
                    continue
            
            regime_results[regime] = {
                'optimal_lookback': best_regime_lookback,
                'performance_score': best_regime_score,
                'scores': regime_scores
            }
            overall_scores.extend(regime_scores)
        
        # Calculate overall optimal lookback (weighted average)
        if regime_results:
            weighted_lookback = sum(
                result['optimal_lookback'] * result['performance_score']
                for result in regime_results.values()
            ) / sum(result['performance_score'] for result in regime_results.values())
            optimal_lookback = int(round(weighted_lookback))
        else:
            optimal_lookback = self.config.min_lookback
        
        # Calculate overall performance score
        overall_performance = np.mean(overall_scores) if overall_scores else 0
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(overall_scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(overall_scores)
        
        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=optimal_lookback,
            performance_score=overall_performance,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.REGIME_AWARE.value,
            regime_specific_results=regime_results,
            validation_scores=overall_scores
        )
    
    async def _optimize_adaptive(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        lookback_range: range
    ) -> FeatureOptimizationResult:
        """Adaptive optimization that combines multiple methods."""
        self.logger.info(f"Using adaptive optimization for {feature_name}")
        
        # Try different methods and combine results
        methods = [
            OptimizationMethod.STATISTICAL_ANALYSIS,
            OptimizationMethod.INFORMATION_THEORY
        ]
        
        if SKLEARN_AVAILABLE:
            methods.append(OptimizationMethod.CROSS_VALIDATION)
        
        results = []
        for method in methods:
            try:
                if method == OptimizationMethod.CROSS_VALIDATION:
                    result = await self._optimize_with_cross_validation(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                elif method == OptimizationMethod.STATISTICAL_ANALYSIS:
                    result = await self._optimize_with_statistical_analysis(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                elif method == OptimizationMethod.INFORMATION_THEORY:
                    result = await self._optimize_with_information_theory(
                        data, feature_name, target_column, feature_generator, lookback_range
                    )
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Error in adaptive optimization with method {method}: {e}")
                continue
        
        if not results:
            # Fallback to statistical analysis
            return await self._optimize_with_statistical_analysis(
                data, feature_name, target_column, feature_generator, lookback_range
            )
        
        # Combine results (weighted average)
        weights = [r.performance_score for r in results]
        total_weight = sum(weights)
        
        if total_weight > 0:
            optimal_lookback = int(round(
                sum(r.optimal_lookback * w for r, w in zip(results, weights)) / total_weight
            ))
            performance_score = sum(r.performance_score * w for r, w in zip(results, weights)) / total_weight
        else:
            optimal_lookback = results[0].optimal_lookback
            performance_score = results[0].performance_score
        
        # Calculate combined stability score
        all_scores = []
        for result in results:
            if result.validation_scores:
                all_scores.extend(result.validation_scores)
        
        stability_score = self._calculate_stability_score(all_scores)
        confidence_interval = self._calculate_confidence_interval(all_scores)
        
        return FeatureOptimizationResult(
            feature_name=feature_name,
            optimal_lookback=optimal_lookback,
            performance_score=performance_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.ADAPTIVE.value,
            validation_scores=all_scores
        )
    
    async def _analyze_regime_specific_performance(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        feature_generator: Callable,
        optimal_lookback: int,
        regime_column: str
    ) -> Dict[str, Any]:
        """Analyze performance across different regimes."""
        regime_analysis = {}
        regimes = data[regime_column].unique()
        
        for regime in regimes:
            regime_data = data[data[regime_column] == regime]
            if len(regime_data) < 10:
                continue
            
            try:
                feature_values = feature_generator(regime_data, optimal_lookback)
                valid_indices = ~(feature_values.isna() | regime_data[target_column].isna())
                
                if valid_indices.sum() > 5:
                    correlation = feature_values[valid_indices].corr(regime_data[target_column][valid_indices])
                    regime_analysis[regime] = {
                        'correlation': correlation,
                        'sample_size': valid_indices.sum(),
                        'feature_mean': feature_values[valid_indices].mean(),
                        'feature_std': feature_values[valid_indices].std()
                    }
            except Exception as e:
                self.logger.warning(f"Error analyzing regime {regime}: {e}")
                continue
        
        return regime_analysis
    
    async def _analyze_feature_decay(
        self,
        data: pd.DataFrame,
        feature_name: str,
        feature_generator: Callable,
        optimal_lookback: int
    ) -> Dict[str, Any]:
        """Analyze how feature performance decays with different lookback periods."""
        decay_analysis = {}
        
        # Test lookback periods around the optimal
        test_lookbacks = range(
            max(1, optimal_lookback - 10),
            min(optimal_lookback + 11, self.config.max_lookback + 1)
        )
        
        correlations = []
        for lookback in test_lookbacks:
            try:
                feature_values = feature_generator(data, lookback)
                # Calculate autocorrelation as a proxy for information content
                autocorr = feature_values.autocorr(lag=1)
                correlations.append(autocorr if not pd.isna(autocorr) else 0)
            except Exception as e:
                self.logger.warning(f"Error in decay analysis for lookback {lookback}: {e}")
                correlations.append(0)
        
        if correlations:
            decay_analysis = {
                'lookbacks': list(test_lookbacks),
                'correlations': correlations,
                'decay_rate': np.polyfit(test_lookbacks, correlations, 1)[0] if len(correlations) > 1 else 0,
                'peak_lookback': test_lookbacks[np.argmax(correlations)] if correlations else optimal_lookback
            }
        
        return decay_analysis
    
    def _calculate_stability_score(self, scores: List[float]) -> float:
        """Calculate stability score from a list of scores."""
        if not scores or len(scores) < 2:
            return 0.0
        
        # Stability is inverse of coefficient of variation
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / abs(mean_score)
        stability = 1 / (1 + cv)
        return min(1.0, max(0.0, stability))
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if not scores or len(scores) < 2:
            return (0.0, 0.0)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy.stats import t
            t_val = t.ppf((1 + confidence) / 2, n - 1)
        else:
            from scipy.stats import norm
            t_val = norm.ppf((1 + confidence) / 2)
        
        margin_error = t_val * (std_score / np.sqrt(n))
        
        return (mean_score - margin_error, mean_score + margin_error)
    
    def _calculate_mutual_information(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate mutual information between two discrete series."""
        try:
            # Create contingency table
            contingency = pd.crosstab(x, y)
            
            # Calculate mutual information
            n = contingency.sum().sum()
            mi = 0
            
            for i in range(contingency.shape[0]):
                for j in range(contingency.shape[1]):
                    if contingency.iloc[i, j] > 0:
                        p_ij = contingency.iloc[i, j] / n
                        p_i = contingency.iloc[i, :].sum() / n
                        p_j = contingency.iloc[:, j].sum() / n
                        mi += p_ij * np.log2(p_ij / (p_i * p_j))
            
            return mi
        except Exception as e:
            self.logger.warning(f"Error calculating mutual information: {e}")
            return 0.0
    
    async def optimize_multiple_features(
        self,
        data: pd.DataFrame,
        feature_configs: Dict[str, Dict[str, Any]],
        target_column: str,
        regime_column: Optional[str] = None
    ) -> Dict[str, FeatureOptimizationResult]:
        """
        Optimize multiple features in parallel.
        
        Args:
            data: Input data DataFrame
            feature_configs: Dictionary mapping feature names to their configurations
            target_column: Name of the target column
            regime_column: Optional regime column for regime-aware optimization
            
        Returns:
            Dictionary mapping feature names to optimization results
        """
        self.logger.info(f"Optimizing {len(feature_configs)} features in parallel")
        
        results = {}
        
        if self.config.parallel_processing and len(feature_configs) > 1:
            # Parallel optimization
            tasks = []
            for feature_name, config in feature_configs.items():
                feature_generator = config['generator']
                task = self.optimize_feature_lookback(
                    data, feature_name, target_column, feature_generator, regime_column
                )
                tasks.append((feature_name, task))
            
            # Execute tasks
            for feature_name, task in tasks:
                try:
                    result = await task
                    results[feature_name] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {feature_name}: {e}")
                    continue
        else:
            # Sequential optimization
            for feature_name, config in feature_configs.items():
                try:
                    feature_generator = config['generator']
                    result = await self.optimize_feature_lookback(
                        data, feature_name, target_column, feature_generator, regime_column
                    )
                    results[feature_name] = result
                except Exception as e:
                    self.logger.error(f"Error optimizing feature {feature_name}: {e}")
                    continue
        
        self.logger.info(f"Completed optimization for {len(results)} features")
        return results
    
    def get_optimization_summary(self, results: Dict[str, FeatureOptimizationResult]) -> Dict[str, Any]:
        """Generate a summary of optimization results."""
        if not results:
            return {}
        
        summary = {
            'total_features': len(results),
            'optimization_methods': {},
            'lookback_distribution': {},
            'performance_stats': {},
            'stability_stats': {},
            'recommendations': []
        }
        
        # Analyze methods used
        for result in results.values():
            method = result.optimization_method
            summary['optimization_methods'][method] = summary['optimization_methods'].get(method, 0) + 1
        
        # Analyze lookback distribution
        lookbacks = [result.optimal_lookback for result in results.values()]
        summary['lookback_distribution'] = {
            'mean': np.mean(lookbacks),
            'median': np.median(lookbacks),
            'std': np.std(lookbacks),
            'min': np.min(lookbacks),
            'max': np.max(lookbacks)
        }
        
        # Analyze performance
        performances = [result.performance_score for result in results.values()]
        summary['performance_stats'] = {
            'mean': np.mean(performances),
            'median': np.median(performances),
            'std': np.std(performances),
            'min': np.min(performances),
            'max': np.max(performances)
        }
        
        # Analyze stability
        stabilities = [result.stability_score for result in results.values()]
        summary['stability_stats'] = {
            'mean': np.mean(stabilities),
            'median': np.median(stabilities),
            'std': np.std(stabilities),
            'min': np.min(stabilities),
            'max': np.max(stabilities)
        }
        
        # Generate recommendations
        low_performance = [name for name, result in results.items() 
                          if result.performance_score < self.config.performance_threshold]
        low_stability = [name for name, result in results.items() 
                        if result.stability_score < self.config.stability_threshold]
        
        if low_performance:
            summary['recommendations'].append(
                f"Consider removing or redesigning features with low performance: {low_performance}"
            )
        
        if low_stability:
            summary['recommendations'].append(
                f"Consider stabilizing features with low stability: {low_stability}"
            )
        
        return summary

    async def optimize_features(
        self,
        data: pd.DataFrame,
        config: FeatureOptimizationConfig
    ) -> Dict[str, Any]:
        """
        Optimize features based on the provided configuration.
        This is a wrapper method for backward compatibility.

        Args:
            data: Input data DataFrame
            config: Feature optimization configuration

        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Starting feature optimization with method: {config.optimization_method}")

        try:
            # Validate input data type
            if not isinstance(data, pd.DataFrame):
                error_msg = f"❌ Expected DataFrame but got {type(data)}. Cannot perform feature optimization."
                self.logger.error(error_msg)
                return {
                    'best_lookback_period': 20,  # Default fallback
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': error_msg,
                    'fallback_reason': 'invalid_data_type'
                }
            
            # Enhanced optimization with stability constraints
            results = {}
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter out raw OHLCV data and basic transformations - focus on REAL technical indicators
            excluded_columns = [
                'timestamp', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'returns', 'close_return',
                'close_time', 'quote_volume', 'trades', 'day', 'close_log_return', 'volume_return', 
                'volume_log_return', 'price_range', 'price_range_pct', 'body_size', 'body_size_pct', 
                'hour', 'day_of_week', 'is_weekend', 'exchange', 'timeframe', 'symbol', 'interval'
            ]
            feature_columns = [col for col in numeric_columns if col not in excluded_columns]
            
            # Check for real technical indicators (RSI, SMA, EMA, etc.)
            ta_indicators = [col for col in feature_columns if any(indicator in col.lower() for indicator in 
                           ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stoch', 'williams', 'cci', 'roc'])]
            
            # If no real technical indicators available, use the FeatureBank to generate comprehensive features
            if not ta_indicators:
                tprint("⚠️ No technical indicators found, generating comprehensive features using FeatureBank...")
                try:
                    # Import and use the proper FeatureBank system via factory
                    from ..core.factory import get_feature_bank, list_available_categories
                    from ..core.feature_generator import FeatureCategory
                    
                    # Get the global feature bank and manually register generators
                    feature_bank = get_feature_bank()
                    
                    # Manually register ALL available generators using correct imports
                    try:
                        # Direct class imports for main generators
                        from ..categories.momentum import MomentumFeatureGenerator
                        from ..categories.volatility import VolatilityFeatureGenerator
                        from ..categories.trend import TrendFeatureGenerator
                        from ..categories.oscillator import OscillatorFeatureGenerator
                        from ..categories.volume import VolumeFeatureGenerator
                        from ..categories.returns import ReturnsFeatureGenerator
                        from ..categories.support_resistance import SupportResistanceFeatureGenerator
                        from ..categories.candlestick_pattern import CandlestickPatternFeatureGenerator
                        from ..categories.interaction import InteractionFeatureGenerator
                        
                        # Factory functions for complex generators
                        from ..categories.microstructure import create_default_microstructure_generators
                        from ..categories.order_flow import create_default_order_flow_generators
                        from ..categories.cross_timeframe import create_default_cross_timeframe_generators
                        from ..categories.entropy import create_default_entropy_generators
                        from ..categories.time import create_default_time_generators

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None
                        
                        # Register main generators
                        generators_to_register = [
                            MomentumFeatureGenerator(),
                            VolatilityFeatureGenerator(),
                            TrendFeatureGenerator(),
                            OscillatorFeatureGenerator(),
                            VolumeFeatureGenerator(),
                            ReturnsFeatureGenerator(),
                            SupportResistanceFeatureGenerator(),
                            CandlestickPatternFeatureGenerator(),
                            InteractionFeatureGenerator()
                        ]
                        
                        # Add generators from factory functions (with error handling for missing data)
                        try:
                            microstructure_gens = create_default_microstructure_generators()
                            # Filter out generators that require bid/ask data
                            filtered_microstructure = []
                            for gen in microstructure_gens:
                                required_cols = getattr(gen.config, 'required_columns', [])
                                if not any(col in ['bid', 'ask'] for col in required_cols):
                                    filtered_microstructure.append(gen)
                            generators_to_register.extend(filtered_microstructure)
                            tprint(f"✅ Added {len(filtered_microstructure)} microstructure generators (filtered from {len(microstructure_gens)})")
                        except Exception as e:
                            tprint(f"⚠️ Skipping microstructure generators: {e}")
                        
                        try:
                            generators_to_register.extend(create_default_order_flow_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping order flow generators: {e}")
                        
                        try:
                            generators_to_register.extend(create_default_cross_timeframe_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping cross-timeframe generators: {e}")
                        
                        try:
                            generators_to_register.extend(create_default_entropy_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping entropy generators: {e}")
                        
                        try:
                            generators_to_register.extend(create_default_time_generators())
                        except Exception as e:
                            tprint(f"⚠️ Skipping time generators: {e}")
                        
                        for generator in generators_to_register:
                            feature_bank.register_generator(generator)
                        
                        tprint(f"✅ Registered {len(generators_to_register)} feature generators")
                        
                    except Exception as reg_error:
                        tprint(f"⚠️ Failed to register generators: {reg_error}")
                        raise Exception("Failed to register generators")
                    
                    # Check what categories are available now
                    available_categories = list_available_categories()
                    tprint(f"📊 Available feature categories: {len(available_categories)}")
                    
                    if not available_categories:
                        tprint("⚠️ No feature generators available after registration, will use fallback indicators...")
                        raise Exception("No feature generators available")
                    
                    # Generate features for ALL available categories
                    categories_to_generate = [
                        FeatureCategory.MOMENTUM,
                        FeatureCategory.VOLATILITY,
                        FeatureCategory.TREND,
                        FeatureCategory.OSCILLATOR,
                        FeatureCategory.VOLUME,
                        FeatureCategory.RETURNS,
                        FeatureCategory.SUPPORT_RESISTANCE,
                        FeatureCategory.CANDLESTICK_PATTERN,
                        FeatureCategory.MICROSTRUCTURE,
                        FeatureCategory.ORDER_FLOW,
                        FeatureCategory.CROSS_TIMEFRAME,
                        FeatureCategory.ENTROPY,
                        FeatureCategory.TIME
                        # Note: CUSTOM and LEGACY categories available too
                    ]
                    
                    tprint(f"🚀 Generating features for {len(categories_to_generate)} categories...")
                    
                    # Generate features using the FeatureBank with reduced hardware optimization
                    feature_df = feature_bank.generate_features(
                        data=data,
                        categories=categories_to_generate,
                        target_column='returns',
                        lookback_optimization=False,  # Disable to avoid optimize_lookback method error
                        # Pass hardware optimization parameters as kwargs
                        cpu_optimization_level='CONSERVATIVE',  # Reduce CPU intensity
                        enable_thermal_monitoring=False,        # Disable thermal monitoring
                        enable_adaptive_optimization=False,     # Disable adaptive optimization
                        monitoring_interval=30.0,              # Reduce monitoring frequency
                        cpu_usage_threshold=70.0,              # Lower CPU threshold
                        memory_usage_threshold=80.0,           # Lower memory threshold
                        gpu_usage_threshold=60.0,              # Lower GPU threshold
                        temperature_threshold=70.0             # Lower temperature threshold
                    )
                    
                    # Merge generated features with original data
                    if not feature_df.empty:
                        # Add generated features to data
                        for col in feature_df.columns:
                            if col not in data.columns:
                                data[col] = feature_df[col]
                        
                        # Update feature columns to include generated features
                        feature_columns = [col for col in feature_df.columns if col not in excluded_columns]
                        tprint(f"✅ Generated {len(feature_columns)} features using FeatureBank system")
                    else:
                        tprint("⚠️ FeatureBank returned empty results, using fallback basic indicators...")
                        # Fallback to basic indicators if FeatureBank fails
                        self._create_basic_technical_indicators(data)
                        feature_columns = ['sma_20', 'ema_12', 'rsi_14', 'volatility_20']
                        
                except Exception as e:
                    tprint(f"⚠️ FeatureBank failed: {e}, creating basic technical indicators...")
                    # Fallback to basic indicators
                    self._create_basic_technical_indicators(data)
                    feature_columns = ['sma_20', 'ema_12', 'rsi_14', 'volatility_20']
            else:
                # Use existing real technical indicators
                feature_columns = ta_indicators
                tprint(f"✅ Found {len(feature_columns)} existing technical indicators: {feature_columns[:5]}...")
            
            all_scores = []
            all_lookbacks = []
            
            # Limit to reasonable number of features for optimization
            optimization_features = feature_columns[:8] if len(feature_columns) > 8 else feature_columns
            tprint(f"🎯 Optimizing {len(optimization_features)} engineered features: {optimization_features}")
            
            for i, col in enumerate(optimization_features):
                # Generate candidate lookback values
                candidate_lookbacks = list(range(config.min_lookback, min(config.max_lookback, 50), 5))
                candidate_scores = []
                
                for lookback in candidate_lookbacks:
                    # Simple correlation-based scoring
                    try:
                        if len(data) > lookback:
                            rolling_feature = data[col].rolling(window=lookback).mean()
                            # FORCE bi-directional targets first - we know directional_confidence exists
                            target_options = [
                                # FORCE: Use directional_confidence first (we know this exists)
                                'directional_confidence',        # Strength of directional bias - CONFIRMED WORKING
                                
                                # Try other bi-directional targets
                                'opportunity_asymmetry',         # Long-short bias indicator
                                'long_overall_opportunity',      # Long opportunity score
                                'short_overall_opportunity',     # Short opportunity score  
                                
                                # Original targets (backward compatibility) - LOWER PRIORITY
                                'leverage_adjusted_score',       # Primary multi-horizon target (long-biased)
                                'immediate_opportunity',         # Secondary multi-horizon target
                                'short_term_opportunity',        # Tertiary multi-horizon target
                                'returns',                       # Fallback to basic returns
                                'close_return',                  # Alternative returns name
                                'close'                         # Last resort
                            ]
                            
                            target_col = None
                            for target_option in target_options:
                                if target_option in data.columns:
                                    target_col = target_option
                                    break
                            
                            if target_col is None:
                                correlation = 0.0
                                raw_correlation = 0.0
                                tprint(f"⚠️ No suitable target column found for {col}")
                            else:
                                raw_correlation = rolling_feature.corr(data[target_col])
                                correlation = abs(raw_correlation)  # Use absolute for optimization
                                
                                # Enhanced logging for bi-directional targets
                                if target_col in ['long_overall_opportunity', 'short_overall_opportunity', 'opportunity_asymmetry', 'directional_confidence']:
                                    direction = "positive" if raw_correlation > 0 else "negative"
                                    if target_col == 'directional_confidence':
                                        interpretation = "Higher feature → Stronger directional signal" if raw_correlation > 0 else "Higher feature → Weaker directional signal"
                                        tprint(f"🎉 BREAKTHROUGH: Using DIRECTIONAL_CONFIDENCE target for {col} optimization!")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'long_overall_opportunity':
                                        interpretation = "Higher feature → Higher LONG opportunity" if raw_correlation > 0 else "Higher feature → Lower LONG opportunity (contrarian)"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'short_overall_opportunity':  
                                        interpretation = "Higher feature → Higher SHORT opportunity" if raw_correlation > 0 else "Higher feature → Lower SHORT opportunity (contrarian)"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    elif target_col == 'opportunity_asymmetry':
                                        interpretation = "Higher feature → LONG bias" if raw_correlation > 0 else "Higher feature → SHORT bias"
                                        tprint(f"🎯 Using BI-DIRECTIONAL target '{target_col}' for {col} optimization")
                                        tprint(f"   📊 Correlation: {raw_correlation:.4f} ({direction}) - {interpretation}")
                                    
                                elif target_col in ['leverage_adjusted_score', 'immediate_opportunity', 'short_term_opportunity']:
                                    direction = "positive" if raw_correlation > 0 else "negative" 
                                    tprint(f"🎯 Using multi-horizon target '{target_col}' for {col} optimization (correlation: {raw_correlation:.4f} - {direction})")
                            if not np.isnan(correlation):
                                candidate_scores.append(correlation)
                            else:
                                candidate_scores.append(0.0)
                        else:
                            candidate_scores.append(0.0)
                    except Exception:
                        candidate_scores.append(0.0)
                
                # Apply regularization
                regularized_scores = self._apply_regularization(candidate_scores, candidate_lookbacks)
                
                # Apply stability constraints
                constrained_scores = self._calculate_stability_constraints(candidate_lookbacks, regularized_scores)
                
                # Find best with stability weighting
                stability_metrics = self._calculate_stability_metrics(constrained_scores, candidate_lookbacks)
                
                # Combine performance and stability
                if constrained_scores:
                    best_idx = np.argmax(constrained_scores)
                    performance_score = constrained_scores[best_idx]
                    stability_score = stability_metrics['overall_stability']
                    
                    # Weighted final score
                    final_score = (1 - config.stability_weight) * performance_score + config.stability_weight * stability_score
                    
                    optimal_lookback = candidate_lookbacks[best_idx]
                    all_scores.append(final_score)
                    all_lookbacks.append(optimal_lookback)
                    
                    results[col] = {
                        'optimal_lookback': optimal_lookback,
                        'performance_score': performance_score,
                        'stability_score': stability_score,
                        'final_score': final_score,
                        'confidence_interval': (final_score - 0.1, final_score + 0.1),
                        'stability_metrics': stability_metrics
                    }
                else:
                    results[col] = {
                        'optimal_lookback': config.min_lookback,
                        'performance_score': 0.5,
                        'stability_score': 0.5,
                        'final_score': 0.5,
                        'confidence_interval': (0.4, 0.6)
                    }
            
            # Calculate overall stability metrics
            overall_stability_metrics = {}
            if all_scores and all_lookbacks:
                overall_stability_metrics = self._calculate_stability_metrics(all_scores, all_lookbacks)
            
            metadata = {
                'optimization_method': config.optimization_method.value,
                'features_processed': len(numeric_columns),
                'config_used': {
                    'min_lookback': config.min_lookback,
                    'max_lookback': config.max_lookback,
                    'cv_folds': config.cv_folds,
                    'parallel_processing': config.parallel_processing,
                    'l1_regularization': config.l1_regularization,
                    'l2_regularization': config.l2_regularization,
                    'stability_weight': config.stability_weight,
                    'max_lookback_variance': config.max_lookback_variance
                },
                'stability_analysis': {
                    'overall_stability': overall_stability_metrics.get('overall_stability', 0.5),
                    'score_coefficient_variation': overall_stability_metrics.get('score_cv', 1.0),
                    'lookback_coefficient_variation': overall_stability_metrics.get('lookback_cv', 1.0),
                    'range_consistency': overall_stability_metrics.get('range_consistency', 0.5),
                    'regularization_applied': config.l1_regularization > 0 or config.l2_regularization > 0,
                    'stability_constraints_applied': True,
                    'features_optimized': len(results),
                    'average_stability_score': np.mean([r.get('stability_score', 0.5) for r in results.values()]) if results else 0.5
                }
            }

            return {
                'results': results,
                'metadata': metadata
            }

        except Exception as e:
            self.logger.error(f"Feature optimization failed: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)  # Neutral RSI fallback
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = self._vectorbt_rolling_operation(true_range, "mean", period)
            return atr
        except Exception:
            return pd.Series([1.0] * len(data), index=data.index)  # Fallback ATR
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        try:
            sma = self._vectorbt_rolling_operation(prices, "mean", period)
            std = self._vectorbt_rolling_operation(prices, "std", period)
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, lower_band
        except Exception:
            # Fallback bands
            return pd.Series(prices * 1.02, index=prices.index), pd.Series(prices * 0.98, index=prices.index)
    
    def _create_basic_technical_indicators(self, data: pd.DataFrame) -> None:
        """Create basic technical indicators as fallback."""
        try:
            if 'close' in data.columns:
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['ema_12'] = data['close'].ewm(span=12).mean()
                data['rsi_14'] = self._calculate_rsi(data['close'], 14)
                data['volatility_20'] = data['close'].rolling(window=20).std()
                tprint("✅ Created 4 basic technical indicators as fallback")
        except Exception as e:
            tprint(f"⚠️ Failed to create basic indicators: {e}")
    
    def _apply_regularization(self, scores: List[float], lookback_values: List[int]) -> List[float]:
        """Apply L1/L2 regularization to optimization scores."""
        try:
            regularized_scores = scores.copy()
            
            # L1 regularization - penalize extreme lookback values
            if self.config.l1_regularization > 0:
                lookback_array = np.array(lookback_values)
                l1_penalty = self.config.l1_regularization * np.abs(lookback_array - np.mean(lookback_array))
                regularized_scores = [score - penalty for score, penalty in zip(regularized_scores, l1_penalty)]
            
            # L2 regularization - penalize variance in lookback values
            if self.config.l2_regularization > 0:
                lookback_variance = np.var(lookback_values)
                l2_penalty = self.config.l2_regularization * lookback_variance
                regularized_scores = [score - l2_penalty for score in regularized_scores]
            
            return regularized_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regularization failed: {e}")
            return scores
    
    def _calculate_stability_constraints(self, lookback_values: List[int], scores: List[float]) -> List[float]:
        """Apply stability constraints to optimization scores."""
        try:
            constrained_scores = scores.copy()
            
            # Calculate lookback variance penalty (reduced to prevent excessive penalties)
            if len(lookback_values) > 1:
                lookback_variance = np.var(lookback_values) / np.mean(lookback_values)  # Coefficient of variation
                
                # Apply much smaller penalty to avoid turning positive correlations negative
                max_variance_threshold = getattr(self.config, 'max_lookback_variance', 1.0)
                penalty_weight = getattr(self.config, 'lookback_range_penalty', 0.1)
                
                # Reduce penalty weight to prevent sign flips
                reduced_penalty_weight = min(penalty_weight, 0.05)  # Cap at 5% penalty
                
                if lookback_variance > max_variance_threshold:
                    variance_penalty = reduced_penalty_weight * (lookback_variance - max_variance_threshold)
                    # Ensure penalty doesn't exceed 50% of the original score
                    max_penalty = max([abs(score) * 0.5 for score in constrained_scores])
                    variance_penalty = min(variance_penalty, max_penalty)
                    constrained_scores = [score - variance_penalty for score in constrained_scores]
            
            return constrained_scores
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stability constraints failed: {e}")
            return scores
    
    def _rolling_window_optimization(self, data: pd.DataFrame, feature_name: str, 
                                   target_column: str, feature_generator: Callable) -> Dict[str, Any]:
        """Perform rolling window optimization for temporal stability."""
        try:
            if 'timestamp' not in data.columns:
                self.logger.warning("⚠️ No timestamp column for rolling window optimization")
                return {}
            
            # Convert rolling window size to timedelta
            window_size = pd.Timedelta(self.config.rolling_window_size)
            step_size = pd.Timedelta(self.config.rolling_step_size)
            
            rolling_results = []
            start_time = data['timestamp'].min()
            end_time = data['timestamp'].max()
            
            current_time = start_time + window_size
            while current_time <= end_time:
                # Get window data
                window_start = current_time - window_size
                window_data = data[(data['timestamp'] >= window_start) & (data['timestamp'] < current_time)]
                
                if len(window_data) < 100:  # Minimum data requirement
                    current_time += step_size
                    continue
                
                # Optimize for this window
                window_results = []
                for lookback in range(self.config.min_lookback, min(self.config.max_lookback, len(window_data)//4)):
                    try:
                        feature_values = feature_generator(window_data, lookback)
                        if len(feature_values) > 10:
                            correlation = abs(feature_values.corr(window_data[target_column]))
                            if not np.isnan(correlation):
                                window_results.append({
                                    'lookback': lookback,
                                    'score': correlation,
                                    'window_start': window_start,
                                    'window_end': current_time
                                })
                    except Exception:
                        continue
                
                if window_results:
                    best_window = max(window_results, key=lambda x: x['score'])
                    rolling_results.append(best_window)
                
                current_time += step_size
            
            # Calculate temporal consistency
            if rolling_results:
                lookback_values = [r['lookback'] for r in rolling_results]
                score_values = [r['score'] for r in rolling_results]
                
                temporal_stability = 1.0 - (np.std(lookback_values) / np.mean(lookback_values)) if np.mean(lookback_values) > 0 else 0.0
                
                return {
                    'rolling_results': rolling_results,
                    'temporal_stability': temporal_stability,
                    'optimal_lookback': int(np.median(lookback_values)),
                    'stability_score': temporal_stability,
                    'window_count': len(rolling_results)
                }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Rolling window optimization failed: {e}")
            return {}
    
    def _calculate_stability_metrics(self, scores: List[float], lookback_values: List[int]) -> Dict[str, float]:
        """Calculate comprehensive stability metrics."""
        try:
            metrics = {}
            
            # Coefficient of variation for scores
            if len(scores) > 1:
                metrics['score_cv'] = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0.0
            else:
                metrics['score_cv'] = 0.0
            
            # Coefficient of variation for lookback values
            if len(lookback_values) > 1:
                metrics['lookback_cv'] = np.std(lookback_values) / np.mean(lookback_values) if np.mean(lookback_values) > 0 else 0.0
            else:
                metrics['lookback_cv'] = 0.0
            
            # Overall stability score (lower CV = higher stability)
            metrics['overall_stability'] = 1.0 - min(1.0, (metrics['score_cv'] + metrics['lookback_cv']) / 2.0)
            
            # Temporal consistency (based on lookback range)
            if len(lookback_values) > 1:
                lookback_range = max(lookback_values) - min(lookback_values)
                max_possible_range = self.config.max_lookback - self.config.min_lookback
                metrics['range_consistency'] = 1.0 - (lookback_range / max_possible_range)
            else:
                metrics['range_consistency'] = 1.0
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stability metrics calculation failed: {e}")
            return {'overall_stability': 0.5, 'score_cv': 1.0, 'lookback_cv': 1.0, 'range_consistency': 0.5}

# Convenience functions
def get_feature_optimizer(config: Optional[FeatureOptimizationConfig] = None) -> FeatureGenerationOptimizer:
    """Get a configured feature generation optimizer."""
    return FeatureGenerationOptimizer(config)

async def optimize_feature_lookback(
    data: pd.DataFrame,
    feature_name: str,
    target_column: str,
    feature_generator: Callable,
    config: Optional[FeatureOptimizationConfig] = None,
    regime_column: Optional[str] = None
) -> FeatureOptimizationResult:
    """Convenience function for optimizing a single feature."""
    optimizer = get_feature_optimizer(config)
    return await optimizer.optimize_feature_lookback(
        data, feature_name, target_column, feature_generator, regime_column
    )
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
