"""
Optimized Variant Generator for Feature Engineering

Enhanced version with:
- Shared computation caching
- Vectorized operations using VectorBT
- Hardware optimization
- Cheaper proxy calculations
- Performance monitoring
- Data validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
import hashlib
from collections import OrderedDict

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Math validation imports
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    safe_correlation, safe_covariance, safe_mean, safe_std, MathValidation
)

# Hardware optimization
try:
    from src.utils.hardware.unified_hardware_manager import (
        UnifiedHardwareManager, WorkloadType, OptimizationLevel, get_unified_hardware_manager
    )
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPT_AVAILABLE = True
except ImportError:
    HARDWARE_OPT_AVAILABLE = False

# Feature common utilities
try:
    from src.utils.feature_common.caching import get_shared_cache, get_feature_cache
    from src.utils.feature_common.monitoring import get_performance_monitor, get_resource_tracker
    from src.utils.feature_common.validation import get_data_validator
    FEATURE_COMMON_AVAILABLE = True
except ImportError:
    FEATURE_COMMON_AVAILABLE = False

# ML common utilities
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
    )
    from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEIntegration
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False

from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error, tprint_warning, tprint_performance
from src.utils.logger import system_logger

logger = system_logger.getChild('OptimizedVariantGenerator')

@dataclass
class OptimizedVariantConfig:
    """Configuration for optimized variant generation."""
    feature_name: str
    category: str
    optimal_lookback: int
    enable_vol_norm: bool = True
    enable_vwap: bool = True
    enable_trend_adj: bool = True
    enable_vectorbt: bool = True
    enable_hardware_opt: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    robust_scaler_quantile_range: Tuple[float, float] = (1.0, 99.0)
    max_workers: int = 4
    chunk_size: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour TTL
    max_cache_size: int = 1000  # Maximum number of cached items
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.optimal_lookback < 1:
            raise ValueError("optimal_lookback must be >= 1")
        if not 0 < self.robust_scaler_quantile_range[0] < self.robust_scaler_quantile_range[1] < 100:
            raise ValueError("Invalid quantile range")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

class OptimizedVariantGenerator:
    """
    Optimized variant generator with comprehensive performance improvements.
    
    Features:
    - VectorBT integration for vectorized operations
    - Hardware optimization for Apple Silicon
    - Shared computation caching
    - Performance monitoring
    - Data validation
    - Cheaper proxy calculations
    """
    
    def __init__(self, config: Optional[OptimizedVariantConfig] = None):
        """Initialize optimized variant generator."""
        self.config = config or OptimizedVariantConfig(
            feature_name="", category="", optimal_lookback=20
        )
        self.logger = system_logger.getChild('OptimizedVariantGenerator')
        
        # Initialize hardware optimization
        self.hardware_manager = None
        self.memory_optimizer = None
        if HARDWARE_OPT_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.memory_optimizer = M1MemoryOptimizer(memory_limit_gb=8.0)
                tprint_info("✅ Hardware optimization initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Hardware optimization failed: {e}")
        
        # Initialize VectorBT components
        self.vectorization_manager = None
        if ML_COMMON_AVAILABLE and VECTORBT_AVAILABLE:
            try:
                self.vectorization_manager = get_unified_vectorization_manager()
                tprint_info("✅ VectorBT components initialized")
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
        
        # Initialize shared utilities
        self.shared_cache = get_shared_cache() if FEATURE_COMMON_AVAILABLE else None
        self.feature_cache = get_feature_cache() if FEATURE_COMMON_AVAILABLE else None
        self.performance_monitor = get_performance_monitor() if FEATURE_COMMON_AVAILABLE else None
        self.resource_tracker = get_resource_tracker() if FEATURE_COMMON_AVAILABLE else None
        self.data_validator = get_data_validator() if FEATURE_COMMON_AVAILABLE else None
        
        # Initialize math validation
        self.math_validator = MathValidation()
        
        # Initialize reusable scalers cache
        self._scalers_cache = {}
        
        # Initialize content-based cache with TTL
        self._content_cache = OrderedDict()
        self._cache_timestamps = {}
        
        # Track statistics
        self.stats = {
            'total_variants_generated': 0,
            'failed_variants': [],
            'clipping_stats': {},
            'variants_by_type': {
                'base': 0,
                'volnorm': 0,
                'vwap': 0,
                'trend_adj': 0
            },
            'performance_metrics': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        tprint_info("🔧 Initialized OptimizedVariantGenerator")
    
    def generate_variants(
        self,
        data: pd.DataFrame,
        feature_name: str,
        category: str,
        optimal_lookback: int,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Generate optimized variants for a feature.
        
        Args:
            data: DataFrame containing the feature
            feature_name: Name of the feature to generate variants for
            category: Feature category
            optimal_lookback: Optimal lookback period
            ohlcv_data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary mapping variant names to Series
        """
        if self.performance_monitor:
            return self.performance_monitor.monitor_operation("generate_variants")(
                self._generate_variants_impl
            )(data, feature_name, category, optimal_lookback, ohlcv_data)
        else:
            return self._generate_variants_impl(data, feature_name, category, optimal_lookback, ohlcv_data)
    
    def _generate_variants_impl(
        self,
        data: pd.DataFrame,
        feature_name: str,
        category: str,
        optimal_lookback: int,
        ohlcv_data: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """Implementation of variant generation with optimizations."""
        variants = {}
        
        try:
            # 1. Base variant (original feature)
            base_variant = self._generate_base_variant(data[feature_name])
            variants[f"{feature_name}_base"] = base_variant
            self.stats['variants_by_type']['base'] += 1
            
            # 2. Volatility-normalized (skip if volatility feature)
            if category.lower() != 'volatility':
                vol_norm = self._generate_volatility_normalized_optimized(
                    data[feature_name], ohlcv_data['close'], optimal_lookback
                )
                if vol_norm is not None:
                    variants[f"{feature_name}_volnorm"] = vol_norm
                    self.stats['variants_by_type']['volnorm'] += 1
            
            # 3. VWAP-weighted (only for price-based features)
            if category.lower() not in ['volume'] and self._is_price_based_feature(feature_name, category):
                vwap_weighted = self._generate_vwap_weighted_optimized(
                    data[feature_name], ohlcv_data['volume'], optimal_lookback
                )
                if vwap_weighted is not None:
                    variants[f"{feature_name}_vwap"] = vwap_weighted
                    self.stats['variants_by_type']['vwap'] += 1
            
            # 4. Trend-adjusted (only for oscillators/momentum)
            if category.lower() in ['oscillator', 'momentum']:
                trend_adj = self._generate_trend_adjusted_optimized(
                    data[feature_name], ohlcv_data, optimal_lookback
                )
                if trend_adj is not None:
                    variants[f"{feature_name}_trend_adj"] = trend_adj
                    self.stats['variants_by_type']['trend_adj'] += 1
            
            self.stats['total_variants_generated'] += len(variants)
            
        except Exception as e:
            self.logger.error(f"Failed to generate variants for {feature_name}: {e}")
            # Return at least base variant
            if f"{feature_name}_base" in variants:
                return {f"{feature_name}_base": variants[f"{feature_name}_base"]}
            return {}
        
        return variants
    
    def _generate_base_variant(self, feature: pd.Series) -> pd.Series:
        """Generate base variant with causality enforcement."""
        return self._apply_causality_shift(feature)
    
    def _generate_volatility_normalized_optimized(
        self,
        feature: pd.Series,
        close_prices: pd.Series,
        lookback: int
    ) -> Optional[pd.Series]:
        """Generate volatility-normalized variant with optimizations using VectorBTRollingOptimizer."""
        try:
            # Use VectorBTRollingOptimizer for efficient rolling calculations
            if self.vectorization_manager and VECTORBT_AVAILABLE:
                try:
                    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    if rolling_optimizer:
                        returns = close_prices.pct_change()
                        rolling_vol = rolling_optimizer.rolling_std(returns, window=lookback)
                    else:
                        # Fallback to pandas
                        returns = close_prices.pct_change()
                        rolling_vol = returns.rolling(window=lookback, min_periods=max(1, lookback // 2)).std()
                except Exception:
                    # Fallback to pandas
                    returns = close_prices.pct_change()
                    rolling_vol = returns.rolling(window=lookback, min_periods=max(1, lookback // 2)).std()
            else:
                # Standard pandas implementation
                returns = close_prices.pct_change()
                rolling_vol = returns.rolling(window=lookback, min_periods=max(1, lookback // 2)).std()
            
            # Use safe division from math_validation
            rolling_vol = rolling_vol.replace(0, np.nan)
            rolling_vol = rolling_vol.fillna(rolling_vol.mean())
            
            # Normalize with safe division
            vol_normalized = self.math_validator.safe_divide(feature, rolling_vol, default=0.0)
            
            # Apply robust scaling with caching
            vol_normalized = self._apply_robust_scaling_cached(vol_normalized, "volnorm")
            
            return vol_normalized
            
        except Exception as e:
            self.logger.error(f"Volatility normalization failed: {e}")
            return None
    
    def _generate_vwap_weighted_optimized(
        self,
        feature: pd.Series,
        volume: pd.Series,
        lookback: int
    ) -> Optional[pd.Series]:
        """Generate VWAP-weighted variant with optimizations using VectorBTRollingOptimizer."""
        try:
            # Use VectorBTRollingOptimizer for efficient rolling calculations
            if self.vectorization_manager and VECTORBT_AVAILABLE:
                try:
                    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    if rolling_optimizer:
                        rolling_vol_mean = rolling_optimizer.rolling_mean(volume, window=lookback)
                    else:
                        # Fallback to pandas
                        rolling_vol_mean = volume.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
                except Exception:
                    # Fallback to pandas
                    rolling_vol_mean = volume.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            else:
                # Standard pandas implementation
                rolling_vol_mean = volume.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            
            # Use safe division from math_validation
            rolling_vol_mean = rolling_vol_mean.replace(0, np.nan)
            rolling_vol_mean = rolling_vol_mean.fillna(volume.mean())
            
            # Calculate volume ratio with safe division
            volume_ratio = self.math_validator.safe_divide(volume, rolling_vol_mean, default=1.0)
            
            # Weight feature by volume ratio
            vwap_weighted = feature * volume_ratio
            
            # Apply robust scaling with caching
            vwap_weighted = self._apply_robust_scaling_cached(vwap_weighted, "vwap")
            
            return vwap_weighted
            
        except Exception as e:
            self.logger.error(f"VWAP weighting failed: {e}")
            return None
    
    def _generate_trend_adjusted_optimized(
        self,
        feature: pd.Series,
        ohlcv_data: pd.DataFrame,
        lookback: int
    ) -> Optional[pd.Series]:
        """Generate trend-adjusted variant with optimizations using VectorBTRollingOptimizer."""
        try:
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            
            # Use cheaper trend strength proxy instead of full ADX
            trend_strength = self._calculate_cheap_trend_strength(high, low, close, lookback)
            
            # Calculate trend direction using VectorBTRollingOptimizer
            if self.vectorization_manager and VECTORBT_AVAILABLE:
                try:
                    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    if rolling_optimizer:
                        sma = rolling_optimizer.rolling_mean(close, window=lookback)
                    else:
                        # Fallback to pandas
                        sma = close.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
                except Exception:
                    # Fallback to pandas
                    sma = close.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            else:
                # Standard pandas implementation
                sma = close.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
            
            trend_direction = np.sign(sma - close.shift(1))
            
            # Combine: feature * trend_strength * trend_direction
            trend_adjusted = feature * trend_strength * trend_direction
            
            # Apply robust scaling with caching
            trend_adjusted = self._apply_robust_scaling_cached(trend_adjusted, "trend_adj")
            
            return trend_adjusted
            
        except Exception as e:
            self.logger.error(f"Trend adjustment failed: {e}")
            return None
    
    def _calculate_cheap_trend_strength(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int
    ) -> pd.Series:
        """
        Calculate cheap trend strength proxy using math_validation.py for safe operations.
        
        Uses improved trend strength calculation with proper mathematical validation.
        """
        try:
            # Validate inputs using math_validation
            high = self.math_validator.validate_finite(high, "high prices")
            low = self.math_validator.validate_finite(low, "low prices")
            close = self.math_validator.validate_finite(close, "close prices")
            
            # Calculate price range with safe division
            price_range = high - low
            
            # Calculate price momentum with safe operations
            price_momentum = abs(close.pct_change().fillna(0))
            
            # Improved trend strength calculation
            # Use additive approach instead of multiplicative to avoid numerical issues
            price_range_norm = self.math_validator.safe_divide(
                price_range, close + 1e-10, default=0.0
            )
            
            # Combine components additively for better numerical stability
            trend_strength = price_range_norm + (price_momentum * 0.1)  # Scale momentum
            
            # Smooth with rolling mean using VectorBT if available
            if self.vectorization_manager and VECTORBT_AVAILABLE:
                try:
                    # Use VectorBTRollingOptimizer for efficient rolling operations
                    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    if rolling_optimizer:
                        trend_strength = rolling_optimizer.rolling_mean(trend_strength, window=14)
                    else:
                        trend_strength = trend_strength.rolling(window=14, min_periods=7).mean()
                except Exception:
                    trend_strength = trend_strength.rolling(window=14, min_periods=7).mean()
            else:
                trend_strength = trend_strength.rolling(window=14, min_periods=7).mean()
            
            # Normalize to 0-1 range with safe operations
            max_val = trend_strength.max()
            if max_val > 0:
                trend_strength = self.math_validator.safe_divide(
                    trend_strength, max_val + 1e-10, default=0.5
                )
            else:
                trend_strength = pd.Series(0.5, index=close.index)
            
            # Ensure values are in valid range
            trend_strength = trend_strength.clip(0.0, 1.0)
            
            return trend_strength.fillna(0.5)  # Default neutral trend
            
        except Exception as e:
            self.logger.error(f"Cheap trend strength calculation failed: {e}")
            # Return default trend strength
            return pd.Series(0.5, index=close.index)
    
    def _get_series_hash(self, series: pd.Series) -> str:
        """Generate content-based hash for series."""
        try:
            # Use content hash instead of object id
            content = series.values.tobytes() + str(series.index).encode()
            return hashlib.md5(content).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.md5(str(series).encode()).hexdigest()
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.config.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._content_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _apply_robust_scaling_cached(self, series: pd.Series, variant_type: str) -> pd.Series:
        """Apply robust scaling with content-based caching and reusable scalers."""
        try:
            # Generate content-based cache key
            series_hash = self._get_series_hash(series)
            cache_key = f"robust_scaling_{variant_type}_{series_hash}"
            
            # Check content-based cache first
            if cache_key in self._content_cache:
                # Check if cache entry is still valid
                if time.time() - self._cache_timestamps.get(cache_key, 0) < self.config.cache_ttl_seconds:
                    self.stats['cache_hits'] += 1
                    return self._content_cache[cache_key]
                else:
                    # Remove expired entry
                    self._content_cache.pop(cache_key, None)
                    self._cache_timestamps.pop(cache_key, None)
            
            self.stats['cache_misses'] += 1
            
            # Clean cache if it's getting too large
            if len(self._content_cache) > self.config.max_cache_size:
                self._clean_cache()
            
            # Remove NaN values for scaling
            valid_mask = ~series.isna()
            valid_data = series[valid_mask].values.reshape(-1, 1)
            
            if len(valid_data) == 0:
                return series
            
            # Use reusable scaler
            scaler_key = f"scaler_{variant_type}"
            if scaler_key not in self._scalers_cache:
                from sklearn.preprocessing import RobustScaler
                self._scalers_cache[scaler_key] = RobustScaler(
                    quantile_range=self.config.robust_scaler_quantile_range
                )
                # Fit the scaler
                self._scalers_cache[scaler_key].fit(valid_data)
            
            # Transform data using fitted scaler
            scaled_data = self._scalers_cache[scaler_key].transform(valid_data)
            
            # Create result series
            result = series.copy()
            result[valid_mask] = scaled_data.flatten()
            
            # Cache result with timestamp
            self._content_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            
            # Track clipping statistics
            if variant_type not in self.stats['clipping_stats']:
                self.stats['clipping_stats'][variant_type] = {
                    'count': 0,
                    'original_ranges': [],
                    'scaled_ranges': []
                }
            
            self.stats['clipping_stats'][variant_type]['count'] += 1
            self.stats['clipping_stats'][variant_type]['original_ranges'].append(
                (valid_data.min(), valid_data.max())
            )
            self.stats['clipping_stats'][variant_type]['scaled_ranges'].append(
                (scaled_data.min(), scaled_data.max())
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Robust scaling failed for {variant_type}: {e}")
            return series
    
    def _apply_causality_shift(self, series: pd.Series) -> pd.Series:
        """Apply shift(1) to enforce causality and prevent lookahead bias."""
        return series.shift(1)
    
    def _is_price_based_feature(self, feature_name: str, category: str) -> bool:
        """Determine if a feature is price-based (suitable for VWAP weighting)."""
        price_based_categories = ['return', 'returns', 'momentum', 'oscillator', 'trend']
        
        if category.lower() in price_based_categories:
            return True
        
        # Check feature name for price-related keywords
        price_keywords = ['price', 'return', 'rsi', 'macd', 'sma', 'ema', 'momentum', 'roc']
        return any(keyword in feature_name.lower() for keyword in price_keywords)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        stats = self.stats.copy()
        
        # Add cache statistics
        if self.shared_cache:
            cache_stats = self.shared_cache.get_stats()
            stats['cache_stats'] = cache_stats
        
        # Add performance metrics
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_performance_summary()
            stats['performance_stats'] = perf_stats
        
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_variants_generated': 0,
            'failed_variants': [],
            'clipping_stats': {},
            'variants_by_type': {
                'base': 0,
                'volnorm': 0,
                'vwap': 0,
                'trend_adj': 0
            },
            'performance_metrics': {},
            'cache_hits': 0,
            'cache_misses': 0
        }

def generate_all_variants_optimized(
    features_df: pd.DataFrame,
    selected_features: List[Dict[str, Any]],
    ohlcv_data: pd.DataFrame,
    max_workers: int = 4
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate optimized variants for all selected features.
    
    Args:
        features_df: DataFrame containing all features
        selected_features: List of dicts with keys: feature_name, category, optimal_lookback
        ohlcv_data: DataFrame with OHLCV columns
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (variants_df, statistics)
    """
    generator = OptimizedVariantGenerator()
    all_variants = {}
    
    tprint_info(f"🔄 Generating optimized variants for {len(selected_features)} features...")
    
    # Use parallel processing if available
    if max_workers > 1 and len(selected_features) > 10:
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for feature_info in selected_features:
                    feature_name = feature_info['feature_name']
                    category = feature_info['category']
                    optimal_lookback = feature_info['optimal_lookback']
                    
                    if feature_name not in features_df.columns:
                        tprint_warning(f"⚠️ Feature {feature_name} not found in DataFrame, skipping...")
                        continue
                    
                    future = executor.submit(
                        generator.generate_variants,
                        features_df,
                        feature_name,
                        category,
                        optimal_lookback,
                        ohlcv_data
                    )
                    futures.append((feature_name, future))
                
                # Collect results
                for feature_name, future in futures:
                    try:
                        variants = future.result(timeout=60)  # 60 second timeout
                        all_variants.update(variants)
                    except Exception as e:
                        tprint_error(f"❌ Failed to generate variants for {feature_name}: {e}")
                        generator.stats['failed_variants'].append(feature_name)
        
        except Exception as e:
            tprint_warning(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            for i, feature_info in enumerate(selected_features):
                feature_name = feature_info['feature_name']
                category = feature_info['category']
                optimal_lookback = feature_info['optimal_lookback']
                
                if feature_name not in features_df.columns:
                    tprint_warning(f"⚠️ Feature {feature_name} not found in DataFrame, skipping...")
                    continue
                
                try:
                    variants = generator.generate_variants(
                        features_df,
                        feature_name,
                        category,
                        optimal_lookback,
                        ohlcv_data
                    )
                    all_variants.update(variants)
                    
                    if (i + 1) % 10 == 0:
                        tprint_info(f"  Progress: {i+1}/{len(selected_features)} features processed")
                        
                except Exception as e:
                    tprint_error(f"❌ Failed to generate variants for {feature_name}: {e}")
                    generator.stats['failed_variants'].append(feature_name)
    
    else:
        # Sequential processing
        for i, feature_info in enumerate(selected_features):
            feature_name = feature_info['feature_name']
            category = feature_info['category']
            optimal_lookback = feature_info['optimal_lookback']
            
            if feature_name not in features_df.columns:
                tprint_warning(f"⚠️ Feature {feature_name} not found in DataFrame, skipping...")
                continue
            
            try:
                variants = generator.generate_variants(
                    features_df,
                    feature_name,
                    category,
                    optimal_lookback,
                    ohlcv_data
                )
                all_variants.update(variants)
                
                if (i + 1) % 10 == 0:
                    tprint_info(f"  Progress: {i+1}/{len(selected_features)} features processed")
                    
            except Exception as e:
                tprint_error(f"❌ Failed to generate variants for {feature_name}: {e}")
                generator.stats['failed_variants'].append(feature_name)
    
    # Create DataFrame from variants
    variants_df = pd.DataFrame(all_variants, index=features_df.index)
    
    stats = generator.get_stats()
    tprint_success(f"✅ Generated {len(variants_df.columns)} total variants from {len(selected_features)} features")
    tprint_info(f"  Breakdown: {stats['variants_by_type']}")
    
    if stats['failed_variants']:
        tprint_warning(f"⚠️ Failed variants: {len(stats['failed_variants'])}")
    
    # Log performance metrics
    if 'performance_stats' in stats:
        perf_stats = stats['performance_stats']
        tprint_info(f"📊 Performance: {perf_stats.get('total_execution_time', 0):.2f}s total")
        tprint_info(f"📊 Cache hit rate: {stats['cache_hits']/(stats['cache_hits']+stats['cache_misses']+1e-10)*100:.1f}%")
    
    return variants_df, stats
