"""
Regime Feature Integration

This module integrates all regime-focused feature generators for use in
NAS-TAS clustering. Provides a unified interface for regime classification
features while excluding trading-relevant features.

Key Features:
- Unified regime feature generation
- Trading feature exclusion
- Regime-focused feature selection
- 15-minute timeframe optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Import tprint for consistent logging
from src.utils.tprint import tprint

# Import parallel processing utilities
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache

# Import regime-focused feature generators
from .regime_volatility import RegimeVolatilityFeatureGenerator
from .regime_volume import RegimeVolumeFeatureGenerator
from .regime_structural_trend import RegimeStructuralTrendFeatureGenerator
from .regime_statistical import RegimeStatisticalFeatureGenerator
from src.config.regime_feature_thresholds import get_regime_feature_thresholds

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime-focused feature generation."""
    # Regime feature categories to include
    include_volatility_regime: bool = True
    include_volume_regime: bool = True
    include_structural_trend: bool = True
    include_statistical_regime: bool = True
    
    # Feature quality filters (moderately relaxed for regime signal)
    min_regime_persistence: Optional[float] = None
    max_feature_noise_ratio: Optional[float] = None
    min_temporal_stability: Optional[float] = None
    
    # Enhanced regime quality features
    include_regime_quality_metrics: bool = True
    include_economic_significance: bool = True
    include_trading_viability: bool = True
    
    # Performance optimizations
    enable_parallel_processing: bool = True
    enable_matrix_optimization: bool = True
    max_parallel_workers: int = 4
    
    # 15-minute timeframe optimization
    optimize_for_15m: bool = True
    trade_duration_minutes: Tuple[int, int] = (5, 30)
    
    # Feature selection
    max_features_per_category: int = 30
    total_max_features: int = 100
    enable_feature_selection: bool = True

    # Composite scoring weights (exposed for NAS/TAS tuning)
    persistence_weight: float = 0.5
    noise_penalty_weight: float = 0.3
    stability_weight: float = 0.2

    # Intensity weighting controls
    persistence_scale: float = 0.5
    probability_scale: float = 0.75

    def __post_init__(self) -> None:
        thresholds = get_regime_feature_thresholds()
        quality_thresholds = thresholds.get("quality_thresholds", {})

        if self.min_regime_persistence is None:
            self.min_regime_persistence = quality_thresholds.get("min_regime_persistence", 0.2)

        if self.max_feature_noise_ratio is None:
            self.max_feature_noise_ratio = quality_thresholds.get("max_feature_noise_ratio", 1.2)

        if self.min_temporal_stability is None:
            self.min_temporal_stability = quality_thresholds.get("min_temporal_stability", 0.1)

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class RegimeFeatureIntegration(VectorizedFeatureGenerator):
    """Unified regime feature generator that excludes trading features."""
    
    def __init__(self, config: Optional[Union[RegimeFeatureConfig, FeatureConfig]] = None):
        if config is None:
            config = RegimeFeatureConfig()
        elif isinstance(config, FeatureConfig) and not isinstance(config, RegimeFeatureConfig):
            # Convert FeatureConfig to RegimeFeatureConfig
            config = RegimeFeatureConfig(
                include_volatility_regime=True,
                include_volume_regime=True,
                include_structural_trend=True,
                include_statistical_regime=True,
                min_regime_persistence=0.7,
                max_feature_noise_ratio=0.3,
                min_temporal_stability=0.6,
                optimize_for_15m=True,
                trade_duration_minutes=(5, 30),
                max_features_per_category=30,
                total_max_features=100,
                enable_feature_selection=True,
                persistence_weight=0.5,
                noise_penalty_weight=0.3,
                stability_weight=0.2
            )

        self.regime_config = config
        self.config = config

        # Track the most recent selection metadata for downstream reporting
        self._latest_quality_stats: Dict[str, Dict[str, float]] = {}
        self._latest_selection_scores: Dict[str, float] = {}
        self._latest_category_counts: Dict[str, int] = {}
        self._latest_target_count: int = getattr(config, 'total_max_features', 100)
        self._latest_intensity_scalers: Dict[str, float] = {}
        
        # Initialize regime-focused feature generators
        self.volatility_generator = RegimeVolatilityFeatureGenerator() if config.include_volatility_regime else None
        self.volume_generator = RegimeVolumeFeatureGenerator() if config.include_volume_regime else None
        self.structural_trend_generator = RegimeStructuralTrendFeatureGenerator() if config.include_structural_trend else None
        self.statistical_generator = RegimeStatisticalFeatureGenerator() if config.include_statistical_regime else None
        
        # Initialize base config
        base_config = FeatureConfig(
            name="regime_feature_integration",
            category=FeatureCategory.REGIME,
            description="Unified regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,
            min_lookback=8,
            max_lookback=128,
            parameters={},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        
        super().__init__(base_config, enable_matrix_ops=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate unified regime features as a single feature series."""
        try:
            # Generate all regime features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple feature if no features generated
                return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            error_msg = f"Regime feature generation failed: {e}"
            tprint(error_msg)
            raise ValueError(error_msg) from e

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate unified regime features, excluding trading features."""
        start_time = time.time()
        features = {}
        feature_names = []
        
        try:
            # 🚀 OPTIMIZATION: Data preprocessing for matrix operations
            tprint(f"🚀 Starting optimized regime feature generation...")
            optimized_data = data
            
            # Check if matrix optimization is enabled (with fallback for old configs)
            enable_matrix_opt = getattr(self.regime_config, 'enable_matrix_optimization', True)
            if enable_matrix_opt:
                optimization_start = time.time()
                optimized_data = self._optimize_matrix_operations(data)
                optimization_time = time.time() - optimization_start
                tprint(f"⚡ Data optimization completed in {optimization_time:.2f}s")
            
            # Prepare generators for execution
            generators = []
            if self.volatility_generator:
                generators.append(("volatility", self.volatility_generator))
            if self.volume_generator:
                generators.append(("volume", self.volume_generator))
            if self.structural_trend_generator:
                generators.append(("structural_trend", self.structural_trend_generator))
            if self.statistical_generator:
                generators.append(("statistical", self.statistical_generator))
            
            # Execute generators (parallel or sequential based on config)
            if generators:
                # Check if parallel processing is enabled (with fallback for old configs)
                enable_parallel = getattr(self.regime_config, 'enable_parallel_processing', True)
                if enable_parallel and len(generators) > 1:
                    parallel_results = self._parallel_feature_generation(generators, optimized_data, **kwargs)
                else:
                    # Sequential execution for debugging or single-threaded environments
                    parallel_results = self._sequential_feature_generation(generators, optimized_data, **kwargs)
                
                # Merge results
                for generator_name, generator_features in parallel_results.items():
                    if generator_features:
                        features.update(generator_features)
                        feature_names.extend(generator_features.keys())
                        tprint(f"✅ {generator_name}: {len(generator_features)} features")
            
            # Generate enhanced regime quality features (sequential - depends on other features)
            include_quality_metrics = getattr(self.regime_config, 'include_regime_quality_metrics', False)
            if include_quality_metrics:
                tprint(f"🔧 Generating regime quality metrics...")
                quality_start = time.time()
                quality_features = self._generate_regime_quality_features(optimized_data, **kwargs)
                quality_time = time.time() - quality_start
                tprint(f"Generated {len(quality_features)} quality features in {quality_time:.2f}s")
                features.update(quality_features)
                feature_names.extend(quality_features.keys())
            
            # OPTIMIZED: Apply quality filters only (no trading feature filter needed - all features are regime-focused)
            if getattr(self.regime_config, 'enable_feature_selection', True):
                filter_start = time.time()
                filtered_features, quality_stats = self._apply_quality_filters(features, optimized_data)

                # Apply intensity weighting prior to feature selection
                filtered_features, intensity_scalers, quality_stats = self._apply_intensity_weighting(
                    filtered_features,
                    quality_stats
                )
                if intensity_scalers:
                    self._latest_intensity_scalers = intensity_scalers
                else:
                    self._latest_intensity_scalers = {
                        name: 1.0 for name in filtered_features.keys()
                    }

                # Ensure we keep exactly the configured number of features for optimal performance
                target_features = getattr(self.regime_config, 'total_max_features', 100)
                self._latest_target_count = target_features

                if len(filtered_features) > target_features:
                    tprint(f"🔍 Feature selection: {len(filtered_features)} → {target_features} features")
                    filtered_features, quality_stats = self._select_top_features(
                        filtered_features,
                        quality_stats,
                        target_features
                    )
                elif len(filtered_features) < target_features:
                    tprint(f"⚠️ Only {len(filtered_features)} features available (target: {target_features})")
                else:
                    tprint(f"✅ Perfect: {len(filtered_features)} features (target: {target_features})")

                features = filtered_features
                # Persist the latest stats aligned with the selected features
                self._latest_quality_stats = {
                    name: quality_stats.get(name, {})
                    for name in features.keys()
                }
                self._latest_intensity_scalers = {
                    name: self._latest_quality_stats.get(name, {}).get('intensity_scaler', 1.0)
                    for name in features.keys()
                }
                if len(filtered_features) <= target_features:
                    persistence_weight = getattr(self.regime_config, 'persistence_weight', 0.5)
                    noise_penalty_weight = getattr(self.regime_config, 'noise_penalty_weight', 0.3)
                    stability_weight = getattr(self.regime_config, 'stability_weight', 0.2)
                    self._latest_selection_scores = {
                        name: (
                            persistence_weight * stats.get('persistence', 0.0)
                            - noise_penalty_weight * stats.get('noise_ratio', 0.0)
                            + stability_weight * stats.get('temporal_stability', 0.0)
                        ) if stats else 0.0
                        for name, stats in self._latest_quality_stats.items()
                    }
                    self._latest_category_counts = self._compute_category_counts(features.keys())

                filter_time = time.time() - filter_start
                tprint(f"Feature filtering and quality checks completed in {filter_time:.2f}s")
            
            total_time = time.time() - start_time
            tprint(f"🎯 Total regime feature generation completed in {total_time:.2f}s")
            return features
            
        except Exception as e:
            error_msg = f"Regime feature generation failed: {e}"
            tprint(error_msg)
            raise ValueError(error_msg) from e
    
    def _parallel_feature_generation(self, generators: List[Tuple[str, Any]], data: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute feature generators in parallel for maximum performance."""
        results = {}
        
        # OPTIMIZED: Determine optimal number of workers based on system resources
        max_workers_config = getattr(self.regime_config, 'max_parallel_workers', 4)
        # Use CPU count for optimal parallelization
        import os

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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
        cpu_count = os.cpu_count() or 4
        max_workers = min(max_workers_config, len(generators), cpu_count)
        
        def generate_features_worker(generator_info):
            """Worker function for parallel feature generation."""
            name, generator = generator_info
            try:
                start_time = time.time()
                features = generator.generate_features(data, **kwargs)
                generation_time = time.time() - start_time
                tprint(f"⚡ {name}: {len(features) if features else 0} features in {generation_time:.2f}s")
                return name, features
            except Exception as e:
                tprint(f"❌ {name} generation failed: {e}")
                return name, {}
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(generate_features_worker, gen_info): gen_info[0] 
                for gen_info in generators
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_name):
                name, features = future.result()
                results[name] = features
        
        return results
    
    def _sequential_feature_generation(self, generators: List[Tuple[str, Any]], data: pd.DataFrame, **kwargs) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute feature generators sequentially for debugging or single-threaded environments."""
        results = {}
        
        for name, generator in generators:
            try:
                start_time = time.time()
                features = generator.generate_features(data, **kwargs)
                generation_time = time.time() - start_time
                tprint(f"⚡ {name}: {len(features) if features else 0} features in {generation_time:.2f}s")
                results[name] = features
            except Exception as e:
                tprint(f"❌ {name} generation failed: {e}")
                results[name] = {}
        
        return results
    
    def _optimize_matrix_operations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for matrix operations by ensuring proper data types and alignment."""
        # Convert to optimal dtypes for matrix operations
        optimized_data = data.copy()
        
        # Count columns that need conversion
        numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
        conversion_count = 0
        
        # Ensure numeric columns are float32 for better memory usage and speed
        for col in numeric_columns:
            if optimized_data[col].dtype != np.float32:
                optimized_data[col] = optimized_data[col].astype(np.float32)
                conversion_count += 1
        
        # Log optimization details
        if conversion_count > 0:
            tprint(f"⚡ Converted {conversion_count} columns to float32 for matrix optimization")
        else:
            tprint(f"⚡ Data already optimized (all numeric columns are float32)")
        
        # Ensure data is aligned and contiguous for matrix operations
        optimized_data = optimized_data.copy()  # Force contiguous memory layout
        
        return optimized_data
    
    @lru_cache(maxsize=128)
    def _cached_data_hash(self, data_hash: str) -> str:
        """Cache data hash for repeated operations."""
        return data_hash
    
    def _filter_trading_features(self, features: Dict[str, np.ndarray], feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Filter out any remaining trading-relevant features."""
        trading_patterns = [
            'rsi', 'macd', 'stochastic', 'williams', 'momentum',
            'oscillator', 'signal', 'crossover', 'divergence',
            'candlestick', 'pattern', 'breakout', 'support', 'resistance',
            'bollinger', 'atr', 'cci', 'roc', 'mfi', 'obv', 'ema', 'sma'
        ]
        
        # OPTIMIZED: Use dictionary comprehension for faster filtering
        regime_patterns = {
            'volatility', 'volume_regime', 'trend_persistence', 
            'regime_stability', 'correlation', 'distribution',
            'clustering', 'persistence', 'structural', 'statistical',
            'vol_persistence', 'vol_clustering', 'vol_stability',
            'vol_regime', 'trend_strength', 'market_structure'
        }
        
        filtered_features = {
            name: feature_array for name, feature_array in features.items()
            if not any(pattern in name.lower() for pattern in trading_patterns)
            and any(pattern in name.lower() for pattern in regime_patterns)
        }
        
        return filtered_features
    
    def _apply_quality_filters(self, features: Dict[str, np.ndarray], data: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """Apply quality filters and compute per-feature quality statistics."""
        filtered_features: Dict[str, np.ndarray] = {}
        quality_stats: Dict[str, Dict[str, float]] = {}

        # Relaxed thresholds for statistical features
        statistical_patterns = ['statistical', 'distribution', 'returns_', 'skewness', 'kurtosis', 'autocorr', 'entropy']

        # OPTIMIZED: Use vectorized filtering with batch processing
        valid_features = {
            name: feature_array for name, feature_array in features.items()
            if feature_array is not None and len(feature_array) > 0
        }
        
        # Batch process features by type for efficiency
        statistical_features = {
            name: feature_array for name, feature_array in valid_features.items()
            if any(pattern in name.lower() for pattern in statistical_patterns)
        }
        
        other_features = {
            name: feature_array for name, feature_array in valid_features.items()
            if not any(pattern in name.lower() for pattern in statistical_patterns)
        }
        
        # Process statistical features with relaxed criteria
        for name, feature_array in statistical_features.items():
            passed, metrics = self._is_high_quality_regime_feature(feature_array, relaxed=True)
            if passed:
                filtered_features[name] = feature_array
                if metrics:
                    quality_stats[name] = metrics

        # Process other features with standard criteria
        for name, feature_array in other_features.items():
            passed, metrics = self._is_high_quality_regime_feature(feature_array)
            if passed:
                filtered_features[name] = feature_array
                if metrics:
                    quality_stats[name] = metrics

        tprint(f"📊 Quality filter results: {len(filtered_features)}/{len(features)} features passed")
        if quality_stats:
            avg_persistence = np.mean([m['persistence'] for m in quality_stats.values()])
            avg_noise = np.mean([m['noise_ratio'] for m in quality_stats.values()])
            avg_stability = np.mean([m['temporal_stability'] for m in quality_stats.values()])
            tprint(
                "   ➤ Avg quality metrics — "
                f"persistence: {avg_persistence:.3f}, "
                f"noise: {avg_noise:.3f}, "
                f"stability: {avg_stability:.3f}"
            )

        self._latest_quality_stats = quality_stats
        return filtered_features, quality_stats

    def _apply_intensity_weighting(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Scale features using persistence and probability based intensity multipliers."""

        if not features:
            return features, {}, quality_stats

        persistence_scale = getattr(self.regime_config, 'persistence_scale', 0.0)
        probability_scale = getattr(self.regime_config, 'probability_scale', 0.0)

        updated_features: Dict[str, np.ndarray] = {}
        intensity_scalers: Dict[str, float] = {}
        updated_quality_stats: Dict[str, Dict[str, float]] = dict(quality_stats)

        for name, feature_array in features.items():
            metrics = dict(quality_stats.get(name, {}))

            persistence = float(metrics.get('persistence', 0.0) or 0.0)
            persistence = max(persistence, 0.0)
            scale = 1.0 + (persistence_scale * persistence if persistence_scale else 0.0)

            probability_value = metrics.get('probability')
            if probability_value is None and probability_scale and feature_array is not None and 'prob' in name.lower():
                valid_values = feature_array[~np.isnan(feature_array)]
                if len(valid_values) > 0:
                    probability_value = float(np.clip(np.nanmean(valid_values), 0.0, 1.0))

            if probability_value is not None:
                probability_value = float(np.clip(probability_value, 0.0, 1.0))
                probability_boost = max(probability_value - 0.5, 0.0)
                scale *= 1.0 + (probability_scale * probability_boost if probability_scale else 0.0)
                metrics['probability'] = probability_value

            if scale <= 0:
                scale = 1.0

            metrics['intensity_scaler'] = scale
            intensity_scalers[name] = scale
            updated_quality_stats[name] = metrics

            if feature_array is not None:
                updated_features[name] = np.asarray(feature_array) * scale
            else:
                updated_features[name] = feature_array

        return updated_features, intensity_scalers, updated_quality_stats

    def _determine_feature_category(self, feature_name: str) -> str:
        """Classify feature names into high-level regime categories."""
        name = feature_name.lower()

        if 'volatility' in name or 'vol_' in name:
            return 'volatility_regime'
        if 'volume' in name or 'liquidity' in name:
            return 'volume_regime'
        if 'trend' in name or 'structural' in name:
            return 'structural_trend'
        if 'statistical' in name or 'distribution' in name or 'entropy' in name:
            return 'statistical_regime'
        if 'economic' in name or 'macro' in name:
            return 'economic_quality'
        if 'trading' in name or 'position' in name:
            return 'trading_viability'
        if 'stability' in name or 'persistence' in name or 'consistency' in name or 'quality' in name:
            return 'regime_quality'

        return 'other'

    def _compute_category_counts(self, feature_names: List[str]) -> Dict[str, int]:
        """Compute category counts for reporting."""
        counts: Dict[str, int] = defaultdict(int)
        for name in feature_names:
            counts[self._determine_feature_category(name)] += 1
        return dict(counts)
    
    def _select_top_features(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Dict[str, Dict[str, float]],
        target_count: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
        """Select top features using composite scoring with category caps."""
        try:
            if not features:
                return features, quality_stats

            persistence_weight = getattr(self.regime_config, 'persistence_weight', 0.5)
            noise_penalty_weight = getattr(self.regime_config, 'noise_penalty_weight', 0.3)
            stability_weight = getattr(self.regime_config, 'stability_weight', 0.2)
            max_per_category = getattr(self.regime_config, 'max_features_per_category', target_count)

            composite_scores: Dict[str, float] = {}
            variances: Dict[str, float] = {}

            for name, feature_array in features.items():
                valid_values = feature_array[~np.isnan(feature_array)] if feature_array is not None else np.array([])
                if len(valid_values) > 1:
                    variances[name] = float(np.var(valid_values))
                else:
                    variances[name] = 0.0

                metrics = quality_stats.get(name, {})
                composite_score = (
                    persistence_weight * metrics.get('persistence', 0.0)
                    - noise_penalty_weight * metrics.get('noise_ratio', 0.0)
                    + stability_weight * metrics.get('temporal_stability', 0.0)
                )

                # Fallback to variance if metrics are missing (e.g., relaxed filters)
                if not metrics:
                    composite_score += variances[name]

                composite_scores[name] = composite_score

            # Sort features by composite score then variance as tie-breaker
            sorted_feature_names = sorted(
                features.keys(),
                key=lambda n: (composite_scores.get(n, float('-inf')), variances.get(n, 0.0)),
                reverse=True
            )

            selected_features: Dict[str, np.ndarray] = {}
            selected_stats: Dict[str, Dict[str, float]] = {}
            category_counts: Dict[str, int] = defaultdict(int)
            categories_capped: List[str] = []

            for name in sorted_feature_names:
                if len(selected_features) >= target_count:
                    break

                category = self._determine_feature_category(name)
                if category_counts[category] >= max_per_category:
                    if category not in categories_capped:
                        categories_capped.append(category)
                    continue

                selected_features[name] = features[name]
                if name in quality_stats:
                    selected_stats[name] = quality_stats[name]
                category_counts[category] += 1

            if len(selected_features) < target_count:
                tprint(
                    f"⚠️ Category caps limited selection to {len(selected_features)}/{target_count} features."
                )

            # Log selection summary for verification
            tprint(
                "🎯 Composite feature selection completed: "
                f"{len(selected_features)}/{target_count} features retained"
            )
            tprint(
                "   ➤ Weights — "
                f"persistence: {persistence_weight:.2f}, "
                f"noise penalty: {noise_penalty_weight:.2f}, "
                f"stability: {stability_weight:.2f}"
            )
            if categories_capped:
                tprint(f"   ➤ Category caps reached for: {', '.join(categories_capped)}")

            preview_count = min(5, len(selected_features))
            if preview_count:
                top_preview = list(selected_features.keys())[:preview_count]
                tprint("   ➤ Top features by composite score:")
                for feature_name in top_preview:
                    tprint(
                        f"      • {feature_name}: "
                        f"score={composite_scores.get(feature_name, 0.0):.4f}, "
                        f"variance={variances.get(feature_name, 0.0):.4f}"
                    )

            self._latest_selection_scores = {
                name: composite_scores.get(name, 0.0)
                for name in selected_features.keys()
            }
            self._latest_category_counts = dict(category_counts)

            return selected_features, selected_stats

        except Exception as e:
            tprint(f"⚠️ Feature selection failed: {e}, returning original features")
            return features, quality_stats
    
    def _generate_regime_quality_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime quality assessment features."""
        features = {}
        
        try:
            # Economic significance features
            if getattr(self.regime_config, 'include_economic_significance', False):
                features.update(self._generate_economic_significance_features(data))
            
            # Trading viability features  
            if getattr(self.regime_config, 'include_trading_viability', False):
                features.update(self._generate_trading_viability_features(data))
            
            # Regime stability features
            features.update(self._generate_regime_stability_features(data))
            
        except Exception as e:
            tprint(f"⚠️ Regime quality feature generation failed: {e}")
        
        return features
    
    def _generate_economic_significance_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate economic significance features for regime quality."""
        features = {}
        
        try:
            # Price impact significance
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                price_volatility = returns.rolling(20).std()
                price_impact = price_volatility / price_volatility.mean()
                features['economic_price_impact'] = price_impact.fillna(0).values
            
            # Volume significance
            if 'volume' in data.columns:
                volume_ma = data['volume'].rolling(20).mean()
                volume_significance = data['volume'] / volume_ma
                features['economic_volume_significance'] = volume_significance.fillna(1).values
            
            # Market efficiency
            if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
                price_range = (data['high'] - data['low']) / data['close']
                efficiency = 1.0 / (1.0 + price_range.rolling(20).mean())
                features['economic_market_efficiency'] = efficiency.fillna(0.5).values
                
        except Exception as e:
            tprint(f"⚠️ Economic significance features failed: {e}")
        
        return features
    
    def _generate_trading_viability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trading viability features for regime quality."""
        features = {}
        
        try:
            # Trading frequency viability
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                volatility = returns.rolling(20).std()
                trading_frequency = 1.0 / (1.0 + volatility)
                features['trading_frequency_viability'] = trading_frequency.fillna(0.5).values
            
            # Position duration viability - OPTIMIZED
            if 'close' in data.columns:
                # OPTIMIZED: Use vectorized trend strength calculation
                close_prices = data['close']
                price_changes = close_prices.diff()
                
                # Vectorized trend strength using rolling slope approximation
                trend_strength = price_changes.rolling(20).mean().abs()
                position_duration = 1.0 / (1.0 + trend_strength)
                features['trading_position_duration'] = position_duration.fillna(0.5).values
            
            # Liquidity viability
            if 'volume' in data.columns and 'close' in data.columns:
                liquidity = data['volume'] * data['close']
                liquidity_viability = liquidity / liquidity.rolling(20).mean()
                features['trading_liquidity_viability'] = liquidity_viability.fillna(1).values
                
        except Exception as e:
            tprint(f"⚠️ Trading viability features failed: {e}")
        
        return features
    
    def _generate_regime_stability_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate regime stability features for quality assessment."""
        features = {}
        
        try:
            # Regime persistence - OPTIMIZED
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                # OPTIMIZED: Use vectorized autocorrelation calculation
                autocorr = returns.rolling(20).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    raw=False
                ).fillna(0)
                features['regime_persistence'] = autocorr.values
            
            # Regime transition stability
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                regime_changes = (returns.rolling(5).std().diff() != 0).astype(int)
                stability = 1.0 - regime_changes.rolling(20).mean()
                features['regime_transition_stability'] = stability.fillna(0.5).values
            
            # Regime consistency
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
                consistency = 1.0 / (1.0 + returns.rolling(20).std())
                features['regime_consistency'] = consistency.fillna(0.5).values
                
        except Exception as e:
            tprint(f"⚠️ Regime stability features failed: {e}")
        
        return features
    
    def _is_high_quality_regime_feature(self, feature_array: np.ndarray, relaxed: bool = False) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Check if a feature meets quality standards and compute its quality metrics."""
        try:
            # Remove NaN values for analysis
            valid_values = feature_array[~np.isnan(feature_array)]

            if len(valid_values) < 5:
                return False, None

            # Test 1: Regime persistence (autocorrelation)
            if len(valid_values) > 1:
                corr = np.corrcoef(valid_values[:-1], valid_values[1:])[0, 1]
                regime_persistence = corr if not np.isnan(corr) else 0.0
            else:
                regime_persistence = 0.0

            # Test 2: Low noise-to-signal ratio
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            noise_ratio = std_val / (abs(mean_val) + 1e-8)

            # Test 3: Temporal stability
            if len(valid_values) > 5:
                window = min(5, len(valid_values) // 2)
                rolling_means = []
                for i in range(window, len(valid_values)):
                    rolling_means.append(np.mean(valid_values[i-window:i]))

                if len(rolling_means) > 1:
                    temporal_stability = 1.0 - (np.std(rolling_means) / (np.mean(np.abs(rolling_means)) + 1e-8))
                else:
                    temporal_stability = 0.0
            else:
                temporal_stability = 0.0

            metrics = {
                'persistence': regime_persistence,
                'noise_ratio': noise_ratio,
                'temporal_stability': temporal_stability,
                'valid_length': float(len(valid_values))
            }

            # Apply very lenient quality thresholds to preserve ~100 features
            # Regime features are expected to change with market regimes, so be permissive
            if relaxed:
                # Very lenient thresholds for statistical features
                result = (regime_persistence > 0.05 and  # Very low bar for autocorrelation
                         noise_ratio < 5.0 and      # Allow high variability for regime changes
                         temporal_stability > -1.0 and # Allow negative stability (regime transitions)
                         len(valid_values) >= 3)
                tprint(f"   Statistical feature: persistence={regime_persistence:.3f}, noise={noise_ratio:.3f}, stability={temporal_stability:.3f}, valid_vals={len(valid_values)}, result={result}")
                return result, metrics
            else:
                # Very lenient thresholds for ALL regime features
                # Goal: Keep ~100 features instead of filtering to 37
                result = (regime_persistence > 0.05 and  # Very low bar for autocorrelation
                         noise_ratio < 4.0 and      # Allow high noise for regime transitions
                         temporal_stability > -0.5 and # Allow negative stability for regime changes
                         len(valid_values) >= 3)
                tprint(f"   Regime feature: persistence={regime_persistence:.3f}, noise={noise_ratio:.3f}, stability={temporal_stability:.3f}, valid_vals={len(valid_values)}, result={result}")
                return result, metrics

        except:
            return False, None
    
    def get_feature_summary(
        self,
        features: Dict[str, np.ndarray],
        quality_stats: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Get summary of generated regime features including selection metadata."""
        stats = quality_stats or self._latest_quality_stats or {}
        total_features = len(features)
        category_counts = self._latest_category_counts or self._compute_category_counts(features.keys())
        max_per_category = getattr(self.regime_config, 'max_features_per_category', total_features or 1)

        if stats:
            avg_persistence = float(np.mean([m.get('persistence', 0.0) for m in stats.values()]))
            avg_noise = float(np.mean([m.get('noise_ratio', 0.0) for m in stats.values()]))
            avg_stability = float(np.mean([m.get('temporal_stability', 0.0) for m in stats.values()]))
        else:
            avg_persistence = 0.0
            avg_noise = 0.0
            avg_stability = 0.0

        selection_scores = self._latest_selection_scores or {}
        top_ranked = sorted(selection_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        summary = {
            'total_features': total_features,
            'feature_categories': category_counts,
            'quality_metrics': {
                'avg_persistence': avg_persistence,
                'avg_noise_ratio': avg_noise,
                'avg_temporal_stability': avg_stability
            },
            'selection': {
                'target': self._latest_target_count,
                'weights': {
                    'persistence': getattr(self.regime_config, 'persistence_weight', 0.5),
                    'noise_penalty': getattr(self.regime_config, 'noise_penalty_weight', 0.3),
                    'stability': getattr(self.regime_config, 'stability_weight', 0.2)
                },
                'intensity_scalers': self._latest_intensity_scalers or {
                    name: stats.get(name, {}).get('intensity_scaler', 1.0)
                    for name in features.keys()
                },
                'category_quota': {
                    category: {
                        'count': count,
                        'max': max_per_category
                    }
                    for category, count in category_counts.items()
                },
                'composite_scores': selection_scores,
                'top_ranked_features': top_ranked
            }
        }

        return summary

# Analyst Features - Regime generators
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystRegimeProbTrendingGenerator(VectorizedFeatureGenerator):
    """Generator for regime probability trending feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_regime_prob_trending",
            category=FeatureCategory.REGIME,
            description="Analyst probability of trending regime",
            required_columns=[],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime probability trending feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
            if current_regime == 'trending':
                prob_trending = 1.0
            elif current_regime == 'choppy':
                prob_trending = 0.0
            else:
                prob_trending = 0.5
        else:
            prob_trending = 0.5

        prob_trending_series = pd.Series([prob_trending] * len(data), index=data.index, name=self.config.name)
        return prob_trending_series

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystRegimeProbChoppyGenerator(VectorizedFeatureGenerator):
    """Generator for regime probability choppy feature."""

    def __init__(self):
        config = FeatureConfig(
            name="analyst_regime_prob_choppy",
            category=FeatureCategory.REGIME,
            description="Analyst probability of choppy regime",
            required_columns=[],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime probability choppy feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
            if current_regime == 'choppy':
                prob_choppy = 1.0
            elif current_regime == 'trending':
                prob_choppy = 0.0
            else:
                prob_choppy = 0.5
        else:
            prob_choppy = 0.5

        prob_choppy_series = pd.Series([prob_choppy] * len(data), index=data.index, name=self.config.name)
        return prob_choppy_series

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class AnalystRegimeStabilityGenerator(VectorizedFeatureGenerator):
    """Generator for regime stability feature."""

    def __init__(self, lookback: int = 50):
        config = FeatureConfig(
            name="analyst_regime_stability",
            category=FeatureCategory.REGIME,
            description="Analyst regime stability (1 - regime_entropy)",
            required_columns=[],
            default_lookback=lookback,
            min_lookback=20,
            max_lookback=200,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.Series:
        """Generate regime stability feature."""
        if regime_data is not None and 'regime' in regime_data.columns:
            regime = regime_data['regime']

            # Shannon entropy calculation
            regime_counts = regime.value_counts()
            total_regimes = len(regime_counts)
            if total_regimes > 0:
                regime_probs = regime_counts / len(regime)
                entropy = -np.sum(regime_probs * np.log2(regime_probs.replace(0, 1)))
                max_entropy = np.log2(total_regimes) if total_regimes > 1 else 1
                stability = 1 - (entropy / max_entropy)
            else:
                stability = 0.5
        else:
            stability = 0.5

        stability_series = pd.Series([stability] * len(data), index=data.index, name=self.config.name)
        return stability_series

# Convenience function for easy integration
def generate_regime_features(data: pd.DataFrame, 
                           config: Optional[RegimeFeatureConfig] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate regime-focused features for clustering.
    
    Args:
        data: Market data DataFrame with OHLCV columns
        config: Configuration for regime feature generation
        
    Returns:
        Tuple of (features_dict, summary_dict)
    """
    if config is None:
        config = RegimeFeatureConfig()
    
    generator = RegimeFeatureIntegration(config)
    features = generator.generate_features(data)
    summary = generator.get_feature_summary(features, generator._latest_quality_stats)

    return features, summary
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
