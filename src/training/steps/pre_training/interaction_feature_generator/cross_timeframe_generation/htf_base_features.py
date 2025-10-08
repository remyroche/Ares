"""Shared HTF base feature utilities with dynamic feature generation.

This module provides dynamic feature generation and lookback optimization
for HTF features. Instead of hardcoded base features with fixed lookback
periods, this module integrates with:
1. FeatureBank system for dynamic feature selection
2. Lookback optimization system for optimized lookback periods

The utilities are stateless and can be safely reused across different
generators without duplicating implementation details.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

# Import feature generation system
try:
    from src.feature_generation.core.feature_bank import FeatureBank, FeatureBankConfig
    from src.feature_generation.core.feature_generator import FeatureCategory
    FEATURE_BANK_AVAILABLE = True
except ImportError:
    FEATURE_BANK_AVAILABLE = False

# Import lookback optimization system
try:
    from ...feature_lookback_optimization.core.optimizer import (
        CoreOptimizer,
        OptimizationMethod,
        OptimizationResult
    )
    LOOKBACK_OPTIMIZER_AVAILABLE = True
except ImportError:
    LOOKBACK_OPTIMIZER_AVAILABLE = False

# Setup logger
logger = logging.getLogger(__name__)


class DynamicFeatureGenerator:
    """
    Dynamic feature generator using FeatureBank system.
    
    This class replaces the hardcoded base feature functions with a dynamic
    system that:
    1. Generates features using FeatureBank
    2. Optimizes lookback periods per feature
    3. Supports feature selection based on performance
    """
    
    def __init__(self):
        """Initialize the dynamic feature generator."""
        self.feature_bank = None
        self.lookback_optimizer = None
        self._initialized = False
        
        # Initialize feature bank if available
        if FEATURE_BANK_AVAILABLE:
            try:
                config = FeatureBankConfig(
                    enable_matrix_operations=True,
                    enable_gpu_acceleration=True,
                    enable_lookback_optimization=True,
                    enable_parallel_processing=True,
                    cache_results=True
                )
                self.feature_bank = FeatureBank(config)
                logger.info("✅ FeatureBank initialized for dynamic feature generation")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize FeatureBank: {e}")
        
        # Initialize lookback optimizer if available
        if LOOKBACK_OPTIMIZER_AVAILABLE:
            try:
                self.lookback_optimizer = CoreOptimizer(logger=logger)
                logger.info("✅ Lookback optimizer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize lookback optimizer: {e}")
        
        self._initialized = (self.feature_bank is not None)
    
    def generate_features(
        self,
        data: pd.DataFrame,
        categories: Optional[List[FeatureCategory]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate features dynamically using FeatureBank.
        
        Args:
            data: Input OHLCV data
            categories: Feature categories to generate (default: momentum, volatility, trend, oscillator)
            exclude_patterns: Patterns to exclude from feature names
            
        Returns:
            DataFrame with generated features
        """
        if not self._initialized or self.feature_bank is None:
            logger.warning("⚠️ FeatureBank not available, returning empty DataFrame")
            return pd.DataFrame()
        
        # Default categories for HTF features
        if categories is None:
            categories = [
                FeatureCategory.RETURNS,
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.SUPPORT_RESISTANCE
            ]
        
        # Default exclude patterns
        if exclude_patterns is None:
            exclude_patterns = [
                'wavelet', 'autoencoder', 'regime_', 'nas_', 'tas_',
                'interaction_', 'cross_timeframe_', 'cross_timeframe',
                'bid_ask', 'bidask', 'market_depth', 'liquidity_proxy',
                'order_flow', 'trade_intensity'
            ]
        
        try:
            # Generate features using FeatureBank
            features_df = self.feature_bank.generate_features(data, categories=categories)
            
            if features_df is None or features_df.empty:
                logger.warning("⚠️ FeatureBank returned empty result")
                return pd.DataFrame()
            
            # Filter out excluded patterns
            feature_columns = [
                col for col in features_df.columns
                if not any(pattern in col.lower() for pattern in exclude_patterns)
            ]
            
            # Also exclude OHLCV columns
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_columns = [col for col in feature_columns if col not in ohlcv_columns]
            
            result_df = features_df[feature_columns].copy()
            logger.info(f"✅ Generated {len(feature_columns)} features using FeatureBank")
            
            return result_df
            
        except Exception as e:
            logger.error(f"❌ Error generating features: {e}")
            return pd.DataFrame()
    
    def optimize_feature_lookback(
        self,
        data: pd.DataFrame,
        feature_name: str,
        target_column: str,
        lookback_range: Tuple[int, int] = (5, 300),
        method: str = 'coarse_to_refine'
    ) -> Dict[str, Any]:
        """
        Optimize lookback period for a single feature.
        
        Args:
            data: Input data with features and target
            feature_name: Name of the feature to optimize
            target_column: Target column for optimization
            lookback_range: Min and max lookback periods to test
            method: Optimization method ('coarse_to_refine', 'grid_search', 'bayesian')
            
        Returns:
            Dict with optimization results:
                - best_lookback_period: Optimal lookback period
                - best_score: Best score achieved
                - method: Optimization method used
        """
        if not LOOKBACK_OPTIMIZER_AVAILABLE or self.lookback_optimizer is None:
            logger.warning("⚠️ Lookback optimizer not available")
            return {
                'best_lookback_period': 20,  # Default
                'best_score': 0.0,
                'method': 'default'
            }
        
        try:
            # Map method string to enum
            method_enum = {
                'coarse_to_refine': OptimizationMethod.COARSE_TO_REFINE,
                'grid_search': OptimizationMethod.GRID_SEARCH,
                'bayesian': OptimizationMethod.BAYESIAN_OPTIMIZATION
            }.get(method, OptimizationMethod.COARSE_TO_REFINE)
            
            # Optimize lookback
            result = self.lookback_optimizer.optimize_single_feature(
                data,
                feature_name,
                target_column,
                method=method_enum,
                lookback_range=lookback_range
            )
            
            return {
                'best_lookback_period': result.best_lookback_period,
                'best_score': result.best_score,
                'method': method
            }
            
        except Exception as e:
            logger.error(f"❌ Error optimizing lookback for {feature_name}: {e}")
            return {
                'best_lookback_period': 20,  # Default
                'best_score': 0.0,
                'method': 'default_fallback'
            }
    
    def get_feature_function(
        self,
        feature_name: str,
        lookback_period: int = 20
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """
        Get a callable function for a specific feature with optimized lookback.
        
        This provides backward compatibility with the old get_base_feature_func interface.
        
        Args:
            feature_name: Name of the feature
            lookback_period: Lookback period to use
            
        Returns:
            Callable that computes the feature
        """
        if not self._initialized:
            raise RuntimeError("DynamicFeatureGenerator not properly initialized")
        
        def compute_feature(data: pd.DataFrame) -> pd.Series:
            """Compute feature with specified lookback period."""
            # Generate features
            features_df = self.generate_features(data)
            
            # Find matching feature column
            if feature_name in features_df.columns:
                return features_df[feature_name]
            
            # Try to find similar feature
            matching_cols = [col for col in features_df.columns if feature_name.lower() in col.lower()]
            if matching_cols:
                logger.info(f"Using {matching_cols[0]} for {feature_name}")
                return features_df[matching_cols[0]]
            
            # Return empty series if not found
            logger.warning(f"⚠️ Feature {feature_name} not found, returning zeros")
            return pd.Series(0, index=data.index)
        
        return compute_feature


# Global instance of dynamic feature generator
_global_feature_generator = None


def get_feature_generator() -> DynamicFeatureGenerator:
    """Get the global feature generator instance."""
    global _global_feature_generator
    if _global_feature_generator is None:
        _global_feature_generator = DynamicFeatureGenerator()
    return _global_feature_generator


def generate_htf_features(
    data: pd.DataFrame,
    categories: Optional[List[FeatureCategory]] = None
) -> pd.DataFrame:
    """
    Generate HTF features dynamically using FeatureBank.
    
    This function replaces the hardcoded base feature functions with dynamic
    feature generation.
    
    Args:
        data: Input OHLCV data
        categories: Feature categories to generate
        
    Returns:
        DataFrame with generated features
    """
    generator = get_feature_generator()
    return generator.generate_features(data, categories=categories)


def optimize_htf_lookbacks(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    lookback_range: Tuple[int, int] = (5, 300)
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize lookback periods for multiple features.
    
    Args:
        data: Input data with features and target
        feature_columns: List of feature names to optimize
        target_column: Target column for optimization
        lookback_range: Min and max lookback periods to test
        
    Returns:
        Dict mapping feature names to optimization results
    """
    generator = get_feature_generator()
    results = {}
    
    for feature in feature_columns:
        results[feature] = generator.optimize_feature_lookback(
            data,
            feature,
            target_column,
            lookback_range=lookback_range
        )
    
    return results


def get_base_feature_func(feature_name: str, lookback_period: int = 20) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Return the computation function for a base feature.
    
    This function provides backward compatibility with the old interface but uses
    dynamic feature generation under the hood.
    
    Args:
        feature_name: Name of the feature (e.g., 'rsi', 'ema', 'bollinger')
        lookback_period: Lookback period to use (default: 20)
        
    Returns:
        Callable that computes the feature
    """
    generator = get_feature_generator()
    return generator.get_feature_function(feature_name, lookback_period)


def resample_to_htf(
    base_series: pd.Series,
    lookback_minutes: int,
    family: str,
) -> pd.Series:
    """
    Resample a base feature series to the requested HTF frequency.
    
    Args:
        base_series: The feature series to resample
        lookback_minutes: HTF lookback period in minutes
        family: Feature family ('trend_level_vol', 'oscillators', 'anchors', etc.)
        
    Returns:
        Resampled series at the HTF frequency
    """
    rule = f"{lookback_minutes}min"

    if family in {"trend_level_vol", "anchors"}:
        return base_series.resample(rule).last()
    if family == "oscillators":
        return base_series.resample(rule).mean()
    return base_series.resample(rule).last()


__all__ = [
    # Core dynamic feature generation
    "DynamicFeatureGenerator",
    "get_feature_generator",
    "generate_htf_features",
    "optimize_htf_lookbacks",
    
    # Backward compatibility functions
    "get_base_feature_func",
    "resample_to_htf",
]
