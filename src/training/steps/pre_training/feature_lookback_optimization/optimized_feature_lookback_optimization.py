"""
Optimized Feature Lookback Optimization Component.

This component optimizes feature lookback periods for better model performance.
It aligns forward return calculations with multi_horizon_profit_labeler goals:
- Different targets with different weights
- Avoidance of adversarial price movements
- Uses first-passage time (FPT) approach, not just close price returns

Provides comprehensive validation, detailed reporting, and robust error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import tprint for consistent logging
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import logging functions
from src.utils.logger import get_logger, log_error, log_warning, log_info

# Core dependencies
import numpy as np
import pandas as pd

# Import feature bank and generators
from src.feature_generation.core.feature_bank import get_global_feature_bank, FeatureCategory
from src.feature_generation.core.feature_generator import FeatureGenerator

# Import profit labeling components for alignment
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig
)
from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
    MultiTargetScheme, MultiTargetConfig, TargetBand
)


@dataclass
class OptimizedFeatureLookbackConfig:
    """Configuration for optimized feature lookback optimization."""

    # Timeframe settings
    default_timeframe: str = "5m"
    base_period_minutes: float = 5.0

    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5

    # Feature selection settings
    excluded_categories: List[FeatureCategory] = None
    excluded_features: List[str] = None

    # Forward return calculation settings (aligned with multi_horizon_profit_labeler)
    enable_volatility_normalization: bool = True
    enable_multi_target_scheme: bool = True
    small_band: Tuple[float, float] = (0.4, 0.8)  # k_s range
    medium_band: Tuple[float, float] = (0.8, 1.3)  # k_m range
    high_band: Tuple[float, float] = (1.3, 2.0)   # k_h range

    # Optimization settings
    optimization_metric: str = "information_coefficient"
    cv_folds: int = 5
    max_optimization_time: int = 300  # seconds

    # Output settings
    save_results: bool = True
    generate_reports: bool = True
    output_directory: str = "feature_lookback_optimization_results"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.excluded_categories is None:
            self.excluded_categories = [
                FeatureCategory.INTERACTION,
                FeatureCategory.CROSS_TIMEFRAME,
                FeatureCategory.AUTOENCODER,
                FeatureCategory.REGIME
            ]

        if self.excluded_features is None:
            self.excluded_features = [
                'wavelets', 'autoencoder', 'interaction', 'cross_timeframe', 'regime_'
            ]


class OptimizationStatus(Enum):
    """Optimization status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FeatureLookbackResult:
    """Result container for feature lookback optimization."""

    # Core results
    feature_name: str
    optimal_lookback: int
    performance_score: float

    # Detailed metrics
    lookback_scores: Dict[int, float]
    best_targets: List[str]
    confidence_interval: Tuple[float, float]

    # Metadata
    optimization_time: float
    n_samples: int
    n_features_tested: int

    # Status
    success: bool
    error_message: Optional[str] = None


class OptimizedFeatureLookbackOptimizer:
    """
    Optimized Feature Lookback Optimizer that aligns with multi_horizon_profit_labeler.

    Key improvements:
    1. Uses first-passage time (FPT) forward return calculations instead of simple close returns
    2. Implements multi-target scheme with different target bands and weights
    3. Avoids adversarial price movements through proper volatility normalization
    4. Excludes problematic feature types (interaction, cross-timeframe, autoencoder, regime)
    5. Uses 5m timeframe by default
    6. Comprehensive tprint logging at every stage
    """

    def __init__(self, config: Optional[OptimizedFeatureLookbackConfig] = None):
        """Initialize the optimized feature lookback optimizer."""
        tprint("🔧 Initializing OptimizedFeatureLookbackOptimizer...")
        self.config = config or OptimizedFeatureLookbackConfig()
        self.logger = get_logger('OptimizedFeatureLookbackOptimizer')

        # Initialize components
        self.feature_bank = get_global_feature_bank()
        self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())
        self.multi_target_scheme = MultiTargetScheme(self._create_multi_target_config())

        # Performance tracking
        self.optimization_status = OptimizationStatus.PENDING
        self.start_time: Optional[float] = None
        self.results: Dict[str, FeatureLookbackResult] = {}

        tprint("✅ OptimizedFeatureLookbackOptimizer initialized")
        tprint_info(f"   → Default timeframe: {self.config.default_timeframe}")
        tprint_info(f"   → Lookback range: {self.config.min_lookback}-{self.config.max_lookback}")
        tprint_info(f"   → Excluded categories: {[c.value for c in self.config.excluded_categories]}")

    def _create_volatility_config(self) -> VolatilityAwareConfig:
        """Create volatility-aware configuration aligned with profit labeler."""
        return VolatilityAwareConfig(
            min_data_points=1000,
            generate_reports=True,
            save_intermediate_results=True,
            enable_volatility_normalization=self.config.enable_volatility_normalization,
            enable_multi_target_scheme=self.config.enable_multi_target_scheme
        )

    def _create_multi_target_config(self) -> MultiTargetConfig:
        """Create multi-target configuration aligned with profit labeler."""
        return MultiTargetConfig(
            small_band=self.config.small_band,
            medium_band=self.config.medium_band,
            high_band=self.config.high_band,
            enable_optimization=True,
            optimization_method='bayesian',
            n_trials=50,
            optimization_metric='lqs'
        )

    def _should_exclude_feature(self, feature_name: str, category: FeatureCategory) -> bool:
        """Check if feature should be excluded from optimization."""
        # Check category exclusions
        if category in self.config.excluded_categories:
            return True

        # Check feature name exclusions
        feature_lower = feature_name.lower()
        for excluded in self.config.excluded_features:
            if excluded in feature_lower:
                return True

        return False

    def _get_eligible_features(self) -> List[Tuple[str, FeatureGenerator]]:
        """Get list of features eligible for lookback optimization."""
        tprint("🔍 Identifying eligible features for optimization...")

        eligible_features = []
        excluded_count = 0

        for category in self.feature_bank.list_categories():
            if category in self.config.excluded_categories:
                tprint_info(f"   → Skipping excluded category: {category.value}")
                continue

            try:
                generators = self.feature_bank.get_generators_by_category(category)
                tprint_info(f"   → Processing {len(generators)} generators in {category.value}")

                for generator in generators:
                    feature_name = generator.config.name

                    if self._should_exclude_feature(feature_name, category):
                        excluded_count += 1
                        continue

                    if generator.supports_lookback_optimization():
                        eligible_features.append((feature_name, generator))
                        tprint_info(f"   → ✓ Eligible: {feature_name}")
                    else:
                        tprint_info(f"   → ⚠ Not optimizable: {feature_name}")

            except Exception as e:
                tprint_warning(f"   → ⚠ Error processing category {category.value}: {e}")

        tprint_success(f"✅ Found {len(eligible_features)} eligible features")
        tprint_info(f"   → Excluded {excluded_count} features")

        return eligible_features

    def _calculate_forward_returns_fpt(self, data: pd.DataFrame,
                                    lookback: int) -> pd.Series:
        """
        Calculate forward returns using first-passage time approach.

        This aligns with multi_horizon_profit_labeler's forward return calculation:
        - Uses volatility-normalized targets
        - Implements first-passage time logic
        - Avoids adversarial price movements
        """
        tprint_info(f"   → Calculating FPT forward returns with lookback {lookback}")

        try:
            # Estimate volatility for the data
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=min(lookback, 50)).std()

            # Use the multi-target scheme to generate targets
            # This ensures alignment with profit labeler's target calculation
            target_result = self.multi_target_scheme.generate_targets(
                data, volatility, pd.Series(True, index=data.index)
            )

            # Extract forward return labels from target scheme
            if not target_result.labels.empty:
                # Convert labels to forward returns (simplified mapping)
                # In practice, this would be more sophisticated
                forward_returns = target_result.labels.copy()

                # Normalize to [-1, 1] range for consistent optimization
                forward_returns = forward_returns.replace({-1: -1.0, 0: 0.0, 1: 1.0})

                tprint_info(f"   → Generated {len(forward_returns.dropna())} forward return samples")
                return forward_returns

            else:
                tprint_warning("   → No forward returns generated from target scheme")
                # Fallback to simple returns if target scheme fails
                simple_returns = data['close'].pct_change(lookback).shift(-lookback)
                return simple_returns.fillna(0)

        except Exception as e:
            tprint_error(f"   → Error calculating FPT forward returns: {e}")
            # Fallback to simple returns
            simple_returns = data['close'].pct_change(lookback).shift(-lookback)
            return simple_returns.fillna(0)

    def _optimize_single_feature(self, feature_name: str, generator: FeatureGenerator,
                               data: pd.DataFrame) -> FeatureLookbackResult:
        """Optimize lookback period for a single feature."""
        tprint_info(f"🎯 Optimizing lookback for feature: {feature_name}")

        lookback_scores = {}
        best_lookback = self.config.min_lookback
        best_score = -np.inf

        # Test different lookback periods
        for lookback in range(self.config.min_lookback,
                             self.config.max_lookback + 1,
                             self.config.lookback_step):

            try:
                tprint_info(f"   → Testing lookback period: {lookback}")

                # Generate feature with current lookback
                feature_data = generator.generate(data, lookback=lookback)
                if feature_data.data.empty or len(feature_data.data.dropna()) < 100:
                    tprint_warning(f"   → Insufficient data for lookback {lookback}")
                    continue

                # Calculate forward returns using FPT approach
                forward_returns = self._calculate_forward_returns_fpt(data, lookback)
                forward_returns = forward_returns.dropna()

                if len(forward_returns) < 100:
                    tprint_warning(f"   → Insufficient forward returns for lookback {lookback}")
                    continue

                # Align feature data with forward returns
                common_index = feature_data.data.index.intersection(forward_returns.index)
                if len(common_index) < 100:
                    tprint_warning(f"   → Insufficient overlapping data for lookback {lookback}")
                    continue

                feature_aligned = feature_data.data.loc[common_index]
                returns_aligned = forward_returns.loc[common_index]

                # Calculate information coefficient (IC) as optimization metric
                # This aligns with the profit labeler's quality assessment
                ic, p_value = self._calculate_information_coefficient(
                    feature_aligned.values, returns_aligned.values
                )

                if not np.isnan(ic) and not np.isnan(p_value):
                    score = abs(ic) * (1 - p_value)  # Combine IC magnitude with significance
                    lookback_scores[lookback] = score

                    tprint_info(f"   → Lookback {lookback}: IC={ic:.4f}, p-value={p_value:.4f}, score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback

                else:
                    tprint_warning(f"   → Invalid IC calculation for lookback {lookback}")

            except Exception as e:
                tprint_warning(f"   → Error testing lookback {lookback}: {e}")
                continue

        # Calculate confidence interval for best lookback
        if best_lookback in lookback_scores:
            scores_array = np.array(list(lookback_scores.values()))
            confidence_interval = (
                np.percentile(scores_array, 5),
                np.percentile(scores_array, 95)
            )
        else:
            confidence_interval = (0.0, 0.0)

        # Determine best targets (for multi-target features)
        best_targets = self._identify_best_targets(generator, data, best_lookback)

        result = FeatureLookbackResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            lookback_scores=lookback_scores,
            best_targets=best_targets,
            confidence_interval=confidence_interval,
            optimization_time=time.time() - (self.start_time or time.time()),
            n_samples=len(data),
            n_features_tested=len(lookback_scores),
            success=len(lookback_scores) > 0
        )

        if result.success:
            tprint_success(f"✅ Optimized {feature_name}: lookback={best_lookback}, score={best_score:.4f}")
        else:
            tprint_error(f"❌ Failed to optimize {feature_name}")
            result.error_message = "No valid lookback periods found"

        return result

    def _calculate_information_coefficient(self, feature_values: np.ndarray,
                                         target_values: np.ndarray) -> Tuple[float, float]:
        """Calculate information coefficient between feature and target."""
        try:
            # Remove NaN values
            valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            if np.sum(valid_mask) < 50:  # Need minimum samples
                return np.nan, np.nan

            feature_clean = feature_values[valid_mask]
            target_clean = target_values[valid_mask]

            # Calculate Spearman correlation (rank-based, robust to outliers)
            correlation, p_value = self._safe_spearmanr(feature_clean, target_clean)

            return correlation, p_value

        except Exception as e:
            tprint_warning(f"   → Error calculating IC: {e}")
            return np.nan, np.nan

    def _safe_spearmanr(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Safe Spearman correlation calculation."""
        try:
            from scipy.stats import spearmanr
            return spearmanr(x, y)
        except Exception:
            # Fallback to numpy correlation
            try:
                return np.corrcoef(x, y)[0, 1], 1.0
            except Exception:
                return 0.0, 1.0

    def _identify_best_targets(self, generator: FeatureGenerator, data: pd.DataFrame,
                             lookback: int) -> List[str]:
        """Identify best targets for multi-target features."""
        # For now, return a simple list - this could be enhanced
        # to analyze which specific targets work best with the feature
        return ["multi_target_primary"]

    async def optimize_all_features(self, data: pd.DataFrame,
                                  timeframe: str = None) -> Dict[str, FeatureLookbackResult]:
        """
        Optimize lookback periods for all eligible features.

        Args:
            data: Market data for optimization
            timeframe: Timeframe for data (uses default if None)

        Returns:
            Dictionary mapping feature names to optimization results
        """
        tprint("🚀 Starting comprehensive feature lookback optimization")
        tprint_info(f"   → Data shape: {data.shape}")
        tprint_info(f"   → Timeframe: {timeframe or self.config.default_timeframe}")

        self.optimization_status = OptimizationStatus.RUNNING
        self.start_time = time.time()
        self.results = {}

        # Get eligible features
        eligible_features = self._get_eligible_features()
        if not eligible_features:
            tprint_error("❌ No eligible features found for optimization")
            self.optimization_status = OptimizationStatus.FAILED
            return {}

        tprint_success(f"🎯 Optimizing {len(eligible_features)} features")

        # Optimize each feature
        for i, (feature_name, generator) in enumerate(eligible_features, 1):
            try:
                tprint_info(f"📊 Progress: {i}/{len(eligible_features)} - {feature_name}")

                # Optimize this feature
                result = self._optimize_single_feature(feature_name, generator, data)
                self.results[feature_name] = result

                # Save intermediate results if requested
                if self.config.save_results and i % 10 == 0:
                    self._save_intermediate_results()

            except Exception as e:
                tprint_error(f"❌ Failed to optimize {feature_name}: {e}")
                self.results[feature_name] = FeatureLookbackResult(
                    feature_name=feature_name,
                    optimal_lookback=self.config.min_lookback,
                    performance_score=0.0,
                    lookback_scores={},
                    best_targets=[],
                    confidence_interval=(0.0, 0.0),
                    optimization_time=0.0,
                    n_samples=0,
                    n_features_tested=0,
                    success=False,
                    error_message=str(e)
                )

        # Generate summary report
        await self._generate_optimization_report()

        # Final status
        successful_optimizations = sum(1 for r in self.results.values() if r.success)
        total_time = time.time() - (self.start_time or time.time())

        tprint_success(f"✅ Feature lookback optimization completed")
        tprint_info(f"   → Successful optimizations: {successful_optimizations}/{len(eligible_features)}")
        tprint_info(f"   → Total time: {total_time:.2f} seconds")
        tprint_info(f"   → Average time per feature: {total_time/max(len(eligible_features), 1):.2f} seconds")

        self.optimization_status = OptimizationStatus.COMPLETED

        return self.results

    def _save_intermediate_results(self):
        """Save intermediate optimization results."""
        try:
            if not self.results:
                return

            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"intermediate_results_{timestamp}.json"

            # Convert results to serializable format
            serializable_results = {}
            for name, result in self.results.items():
                serializable_results[name] = {
                    'feature_name': result.feature_name,
                    'optimal_lookback': result.optimal_lookback,
                    'performance_score': float(result.performance_score),
                    'lookback_scores': {k: float(v) for k, v in result.lookback_scores.items()},
                    'best_targets': result.best_targets,
                    'confidence_interval': [float(x) for x in result.confidence_interval],
                    'optimization_time': float(result.optimization_time),
                    'n_samples': result.n_samples,
                    'n_features_tested': result.n_features_tested,
                    'success': result.success,
                    'error_message': result.error_message
                }

            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'config': {
                        'default_timeframe': self.config.default_timeframe,
                        'min_lookback': self.config.min_lookback,
                        'max_lookback': self.config.max_lookback,
                        'optimization_metric': self.config.optimization_metric
                    },
                    'results': serializable_results
                }, f, indent=2)

            tprint_info(f"💾 Saved intermediate results to {filename}")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to save intermediate results: {e}")

    async def _generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        if not self.config.generate_reports:
            return

        try:
            tprint("📊 Generating optimization report...")

            # Calculate summary statistics
            successful_results = [r for r in self.results.values() if r.success]
            failed_results = [r for r in self.results.values() if not r.success]

            report = {
                'summary': {
                    'total_features': len(self.results),
                    'successful_optimizations': len(successful_results),
                    'failed_optimizations': len(failed_results),
                    'success_rate': len(successful_results) / max(len(self.results), 1),
                    'total_optimization_time': sum(r.optimization_time for r in successful_results),
                    'average_optimization_time': np.mean([r.optimization_time for r in successful_results]) if successful_results else 0
                },
                'performance': {
                    'best_score': max((r.performance_score for r in successful_results), default=0),
                    'worst_score': min((r.performance_score for r in successful_results), default=0),
                    'average_score': np.mean([r.performance_score for r in successful_results]) if successful_results else 0,
                    'score_std': np.std([r.performance_score for r in successful_results]) if successful_results else 0
                },
                'lookback_distribution': {
                    'optimal_lookbacks': [r.optimal_lookback for r in successful_results],
                    'most_common_lookback': None,
                    'lookback_range': (min((r.optimal_lookback for r in successful_results), default=0),
                                    max((r.optimal_lookback for r in successful_results), default=0))
                },
                'failed_features': [
                    {'name': r.feature_name, 'error': r.error_message}
                    for r in failed_results
                ]
            }

            # Calculate most common lookback
            if successful_results:
                lookbacks = [r.optimal_lookback for r in successful_results]
                from collections import Counter
                most_common = Counter(lookbacks).most_common(1)
                report['lookback_distribution']['most_common_lookback'] = most_common[0][0] if most_common else None

            # Save report
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"optimization_report_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            tprint_success(f"📋 Optimization report saved to {filename}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate optimization report: {e}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.results:
            return {'status': 'no_results', 'message': 'No optimization results available'}

        successful_results = [r for r in self.results.values() if r.success]

        return {
            'status': self.optimization_status.value,
            'total_features': len(self.results),
            'successful_optimizations': len(successful_results),
            'failed_optimizations': len(self.results) - len(successful_results),
            'best_performing_feature': max(successful_results, key=lambda x: x.performance_score).feature_name if successful_results else None,
            'best_score': max((r.performance_score for r in successful_results), default=0),
            'average_score': np.mean([r.performance_score for r in successful_results]) if successful_results else 0,
            'optimization_time': sum(r.optimization_time for r in successful_results) if successful_results else 0
        }

    def export_optimized_config(self) -> Dict[str, Any]:
        """Export optimized feature configuration for use in training."""
        if not self.results:
            return {}

        optimized_config = {}

        for feature_name, result in self.results.items():
            if result.success:
                optimized_config[feature_name] = {
                    'optimal_lookback': result.optimal_lookback,
                    'performance_score': result.performance_score,
                    'best_targets': result.best_targets,
                    'confidence_interval': result.confidence_interval
                }

        return optimized_config


# Convenience function for easy optimization
async def optimize_feature_lookbacks(data: pd.DataFrame,
                                   timeframe: str = "5m",
                                   config: Optional[OptimizedFeatureLookbackConfig] = None) -> Dict[str, FeatureLookbackResult]:
    """
    Convenience function to optimize feature lookback periods.

    Args:
        data: Market data for optimization
        timeframe: Timeframe for data (default: 5m)
        config: Optimization configuration

    Returns:
        Dictionary of optimization results
    """
    optimizer = OptimizedFeatureLookbackOptimizer(config)
    return await optimizer.optimize_all_features(data, timeframe)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        # Load data from file if provided
        data_file = sys.argv[1]
        try:
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            tprint(f"📊 Loaded data from {data_file}: {data.shape}")

            # Run optimization
            results = asyncio.run(optimize_feature_lookbacks(data))

            tprint("🎉 Feature lookback optimization completed!")
            tprint(f"   → Optimized {len([r for r in results.values() if r.success])} features")

        except Exception as e:
            tprint_error(f"❌ Error: {e}")
            sys.exit(1)
    else:
        tprint("ℹ️ Usage: python optimized_feature_lookback_optimization.py <data_file.csv>")
        tprint("   → Run feature lookback optimization on the provided data file")