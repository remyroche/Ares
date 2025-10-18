"""
Lookback Optimization System

This module provides the lookback optimization system that leverages the existing
feature generation optimization code to automatically determine optimal lookback
periods for features.
"""

import warnings

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np

# Lazy import to avoid circular dependency
def _get_feature_generator_imports():
    try:
        from ...core.feature_generator import Any, FeatureConfig
        return Any, FeatureConfig
    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Any import failed: {e}")
        return None, None

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

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Optimization methods for lookback periods."""
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    INFORMATION_THEORY = "information_theory"
    REGIME_AWARE = "regime_aware"
    ADAPTIVE = "adaptive"

@dataclass
class FeatureOptimizationConfig:
    """Configuration for feature optimization."""
    min_lookback: int = 5
    max_lookback: int = 252
    step_size: int = 1
    optimization_method: OptimizationMethod = OptimizationMethod.STATISTICAL_ANALYSIS
    cv_folds: int = 5
    stability_threshold: float = 0.8
    performance_threshold: float = 0.6
    regime_aware: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    chunk_size: int = 1000
    optimization_metric: str = "sharpe_ratio"

@dataclass
class FeatureOptimizationResult:
    """Result of feature optimization."""
    feature_name: str
    optimal_lookback: int
    performance_score: float
    stability_score: float
    confidence_interval: tuple
    optimization_method: str
    regime_specific_results: Optional[Dict[str, Any]] = None
    decay_analysis: Optional[Dict[str, Any]] = None
    validation_scores: Optional[List[float]] = None

class LookbackOptimizer:
    """
    Optimizer for feature lookback periods.

    This class leverages the existing feature generation optimization code
    to automatically determine optimal lookback periods for features.
    """

    def __init__(self, config: Optional[FeatureOptimizationConfig] = None):
        """
        Initialize the lookback optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config or FeatureOptimizationConfig()
        self.logger = logger.getChild('LookbackOptimizer')

        # Try to import the existing optimization system
        try:
            from ..feature_generation_optimization import (
                FeatureGenerationOptimizer,
                FeatureOptimizationConfig as LegacyConfig
            )
            self.legacy_optimizer = FeatureGenerationOptimizer()
            self.legacy_available = True
            self.logger.info("✅ Legacy optimization system available")
        except ImportError:
            self.legacy_optimizer = None
            self.legacy_available = False
            self.logger.warning("⚠️ Legacy optimization system not available")

        # Cache for optimization results
        self._optimization_cache: Dict[str, FeatureOptimizationResult] = {}

        self.logger.info("✅ LookbackOptimizer initialized")

    def optimize_lookback(self,
                         generator: Any,
                         data: pd.DataFrame,
                         target_column: str,
                         regime_column: Optional[str] = None) -> int:
        """
        Optimize lookback period for a feature generator.

        Args:
            generator: Feature generator to optimize
            data: Input data
            target_column: Target column for optimization
            regime_column: Optional regime column

        Returns:
            Optimal lookback period
        """
        self.logger.info(f"Optimizing lookback for {generator.config.name}")

        # Check cache first
        cache_key = f"{generator.config.name}_{hash(str(data.shape))}"
        if cache_key in self._optimization_cache:
            result = self._optimization_cache[cache_key]
            self.logger.info(f"Using cached optimization result: {result.optimal_lookback}")
            return result.optimal_lookback

        # Use legacy optimizer if available
        if self.legacy_available and self.legacy_optimizer:
            try:
                result = self._optimize_with_legacy_system(
                    generator, data, target_column, regime_column
                )
                self._optimization_cache[cache_key] = result
                return result.optimal_lookback
            except Exception as e:
                self.logger.warning(f"Legacy optimization failed: {e}")

        # Fallback to simple optimization
        result = self._simple_optimization(generator, data, target_column)
        self._optimization_cache[cache_key] = result
        return result.optimal_lookback

    def _optimize_with_legacy_system(self,
                                   generator: Any,
                                   data: pd.DataFrame,
                                   target_column: str,
                                   regime_column: Optional[str] = None) -> FeatureOptimizationResult:
        """Use the legacy optimization system."""
        # Create a feature generator function for the legacy system
        def feature_generator_func(df: pd.DataFrame, lookback: int) -> pd.Series:
            # Create a temporary generator with the specified lookback
            temp_config = generator.config
            temp_config.default_lookback = lookback
            temp_generator = generator.__class__(temp_config)
            result = temp_generator.generate(df)
            return result.data

        # Use the legacy optimizer
        legacy_result = self.legacy_optimizer.optimize_feature_lookback(
            data=data,
            feature_name=generator.config.name,
            target_column=target_column,
            feature_generator=feature_generator_func,
            regime_column=regime_column
        )

        # Convert to our result format
        return FeatureOptimizationResult(
            feature_name=generator.config.name,
            optimal_lookback=legacy_result.optimal_lookback,
            performance_score=legacy_result.performance_score,
            stability_score=legacy_result.stability_score,
            confidence_interval=legacy_result.confidence_interval,
            optimization_method=legacy_result.optimization_method,
            regime_specific_results=legacy_result.regime_specific_results,
            decay_analysis=legacy_result.decay_analysis,
            validation_scores=legacy_result.validation_scores
        )

    def _simple_optimization(self,
                           generator: Any,
                           data: pd.DataFrame,
                           target_column: str) -> FeatureOptimizationResult:
        """Simple optimization fallback."""
        self.logger.info(f"Using simple optimization for {generator.config.name}")

        best_score = -np.inf
        best_lookback = self.config.min_lookback
        scores = []

        # Test different lookback periods
        for lookback in range(self.config.min_lookback,
                            min(self.config.max_lookback + 1, 50),
                            self.config.step_size):
            try:
                # Generate feature with current lookback
                if generator.supports_lookback_optimization():
                    result = generator.generate_with_lookback(data, lookback)
                else:
                    result = generator.generate(data)

                if not result.success:
                    continue

                # Calculate correlation with target
                valid_indices = ~(result.data.isna() | data[target_column].isna())
                if valid_indices.sum() < 10:
                    continue

                correlation = abs(result.data[valid_indices].corr(data[target_column][valid_indices]))
                scores.append(correlation)

                if correlation > best_score:
                    best_score = correlation
                    best_lookback = lookback

            except Exception as e:
                self.logger.warning(f"Error in optimization for lookback {lookback}: {e}")
                continue

        # Calculate stability score
        stability_score = self._calculate_stability_score(scores)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)

        return FeatureOptimizationResult(
            feature_name=generator.config.name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            stability_score=stability_score,
            confidence_interval=confidence_interval,
            optimization_method=OptimizationMethod.STATISTICAL_ANALYSIS.value,
            validation_scores=scores
        )

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

    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for scores."""
        if not scores or len(scores) < 2:
            return (0.0, 0.0)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)

        # Use t-distribution for small samples
        if n < 30:
            try:
                from scipy.stats import t
                t_val = t.ppf((1 + confidence) / 2, n - 1)
            except ImportError:
                t_val = 2.0  # Fallback
        else:
            try:
                from scipy.stats import norm
                t_val = norm.ppf((1 + confidence) / 2)
            except ImportError:
                t_val = 1.96  # Fallback

        margin_error = t_val * (std_score / np.sqrt(n))

        return (mean_score - margin_error, mean_score + margin_error)

    def optimize_multiple_features(self,
                                 generators: List[Any],
                                 data: pd.DataFrame,
                                 target_column: str,
                                 regime_column: Optional[str] = None) -> Dict[str, int]:
        """
        Optimize lookback periods for multiple features.

        Args:
            generators: List of feature generators
            data: Input data
            target_column: Target column
            regime_column: Optional regime column

        Returns:
            Dictionary mapping feature names to optimal lookback periods
        """
        self.logger.info(f"Optimizing {len(generators)} features")

        results = {}

        if self.config.parallel_processing and len(generators) > 1:
            # Parallel optimization
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_generator = {
                    executor.submit(self.optimize_lookback, gen, data, target_column, regime_column): gen
                    for gen in generators
                }

                for future in as_completed(future_to_generator):
                    generator = future_to_generator[future]
                    try:
                        optimal_lookback = future.result()
                        results[generator.config.name] = optimal_lookback
                    except Exception as e:
                        self.logger.error(f"Error optimizing {generator.config.name}: {e}")
                        results[generator.config.name] = generator.config.default_lookback
        else:
            # Sequential optimization
            for generator in generators:
                try:
                    optimal_lookback = self.optimize_lookback(generator, data, target_column, regime_column)
                    results[generator.config.name] = optimal_lookback
                except Exception as e:
                    self.logger.error(f"Error optimizing {generator.config.name}: {e}")
                    results[generator.config.name] = generator.config.default_lookback

        self.logger.info(f"Completed optimization for {len(results)} features")
        return results

    def get_optimization_summary(self, results: Dict[str, int]) -> Dict[str, Any]:
        """Generate a summary of optimization results."""
        if not results:
            return {}

        lookbacks = list(results.values())

        summary = {
            'total_features': len(results),
            'lookback_distribution': {
                'mean': np.mean(lookbacks),
                'median': np.median(lookbacks),
                'std': np.std(lookbacks),
                'min': np.min(lookbacks),
                'max': np.max(lookbacks)
            },
            'recommendations': []
        }

        # Generate recommendations
        high_lookback_features = [name for name, lookback in results.items() if lookback > 50]
        low_lookback_features = [name for name, lookback in results.items() if lookback < 10]

        if high_lookback_features:
            summary['recommendations'].append(
                f"Features with high lookback periods (>50): {high_lookback_features}"
            )

        if low_lookback_features:
            summary['recommendations'].append(
                f"Features with low lookback periods (<10): {low_lookback_features}"
            )

        return summary

# Convenience functions
def optimize_feature_lookbacks(generators: List[Any],
                             data: pd.DataFrame,
                             target_column: str,
                             config: Optional[FeatureOptimizationConfig] = None,
                             regime_column: Optional[str] = None) -> Dict[str, int]:
    """
    Optimize lookback periods for multiple features.

    Args:
        generators: List of feature generators
        data: Input data
        target_column: Target column
        config: Optimization configuration
        regime_column: Optional regime column

    Returns:
        Dictionary mapping feature names to optimal lookback periods
    """
    optimizer = LookbackOptimizer(config)
    return optimizer.optimize_multiple_features(generators, data, target_column, regime_column)

def get_optimization_config(**kwargs) -> FeatureOptimizationConfig:
    """
    Create an optimization configuration with the given parameters.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Feature optimization configuration
    """
    return FeatureOptimizationConfig(**kwargs)
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
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
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
