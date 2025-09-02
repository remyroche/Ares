"""
Data Type Optimization Utilities

This module provides utilities for optimizing data types to reduce memory usage
and improve computational efficiency in feature engineering.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    target_memory_reduction: float = 0.5,
    preserve_categorical: bool = True,
) -> pd.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage.
    
    Args:
        df: Input DataFrame
        target_memory_reduction: Target memory reduction (0.0 to 1.0)
        preserve_categorical: Whether to preserve existing categorical columns
        
    Returns:
        Optimized DataFrame with reduced memory usage
    """
    initial_memory = df.memory_usage(deep=True).sum()
    logger.info(
        f"🔧 Optimizing data types - Initial memory: {initial_memory / 1024**2:.2f} MB",
    )

    optimized_df = df.copy()

    # Optimize numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        col_type = df[col].dtype

        # Skip if already optimized
        if col_type in ["int8", "int16", "int32", "float16", "float32"]:
            continue

        # Optimize integers
        if col_type in ["int64"]:
            c_min = df[col].min()
            c_max = df[col].max()

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                optimized_df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                optimized_df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                optimized_df[col] = df[col].astype(np.int32)

        # Optimize floats
        elif col_type in ["float64"]:
            # Check if we can use float32 (lose some precision but save memory)
            if df[col].isnull().sum() == 0:  # No NaN values
                try:
                    # Test if conversion preserves values within tolerance
                    float32_vals = df[col].astype(np.float32)
                    if np.allclose(df[col], float32_vals, rtol=1e-5):
                        optimized_df[col] = float32_vals
                except Exception:
                    pass

    # Optimize categorical columns
    if preserve_categorical:
        for col in df.select_dtypes(include=["object"]).columns:
            if len(df) > 0 and df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                optimized_df[col] = df[col].astype("category")

    # Optimize boolean columns
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].isin([True, False, 1, 0, "True", "False", "1", "0"]).all():
                optimized_df[col] = (
                    df[col]
                    .map(
                        {
                            "True": True,
                            "False": False,
                            "1": True,
                            "0": False,
                            1: True,
                            0: False,
                        },
                    )
                    .astype("bool")
                )

    final_memory = optimized_df.memory_usage(deep=True).sum()
    memory_reduction = (initial_memory - final_memory) / initial_memory if initial_memory else 0.0

    logger.info("🔧 Data type optimization complete:")
    logger.info(f"   Initial memory: {initial_memory / 1024**2:.2f} MB")
    logger.info(f"   Final memory: {final_memory / 1024**2:.2f} MB")
    logger.info(f"   Memory reduction: {memory_reduction:.1%}")

    return optimized_df

def get_optimal_dtypes_for_features() -> Dict[str, str]:
    """
    Get optimal data types for common feature patterns.
    
    Returns:
        Dictionary mapping feature patterns to optimal data types
    """
    return {
        # Price-based features (typically float32 is sufficient)
        "price_": "float32",
        "close_": "float32",
        "high_": "float32",
        "low_": "float32",
        "open_": "float32",
        # Volume features (can often use int32)
        "volume_": "int32",
        "vol_": "int32",
        # Technical indicators (float32 is sufficient)
        "rsi_": "float32",
        "sma_": "float32",
        "ema_": "float32",
        "bb_": "float32",
        "macd_": "float32",
        "stoch_": "float32",
        # Cluster features (categorical or int8)
        "cluster_": "int8",
        "intensity_cluster_": "float32",
        # Correlation features (float32)
        "correlation_": "float32",
        "corr_": "float32",
        # Volatility features (float32)
        "volatility_": "float32",
        "vol_": "float32",
        # Momentum features (float32)
        "momentum_": "float32",
        "mom_": "float32",
        # Spread features (float32)
        "spread_": "float32",
        "bid_ask_": "float32",
        # Impact features (float32)
        "impact_": "float32",
        "price_impact": "float32",
        "volume_impact": "float32",
    }

def apply_feature_specific_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature-specific data type optimizations based on feature names.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with feature-specific optimizations applied
    """
    optimal_dtypes = get_optimal_dtypes_for_features()
    optimized_df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        # Find matching pattern
        for pattern, dtype in optimal_dtypes.items():
            if pattern in col_lower:
                try:
                    if dtype == "int8":
                        # For cluster IDs, ensure they're small integers
                        if col_lower.startswith("cluster_") or "cluster" in col_lower:
                            optimized_df[col] = df[col].astype("int8")
                    elif dtype == "float32":
                        # For float features, use float32 if no precision loss
                        if df[col].dtype == "float64":
                            try:
                                float32_vals = df[col].astype(np.float32)
                                if np.allclose(df[col], float32_vals, rtol=1e-5):
                                    optimized_df[col] = float32_vals
                            except Exception:
                                pass
                    elif dtype == "int32":
                        # For volume features, use int32 if possible
                        if df[col].dtype == "int64":
                            c_min = df[col].min()
                            c_max = df[col].max()
                            if (
                                c_min > np.iinfo(np.int32).min
                                and c_max < np.iinfo(np.int32).max
                            ):
                                optimized_df[col] = df[col].astype(np.int32)
                except Exception as e:
                    logger.debug(f"Could not optimize {col} to {dtype}: {e}")
                break

    return optimized_df

def optimize_feature_engineering_pipeline(
    df: pd.DataFrame,
    stage: str = "input",
) -> pd.DataFrame:
    """
    Optimize DataFrame for feature engineering pipeline stages.
    
    Args:
        df: Input DataFrame
        stage: Pipeline stage ("input", "intermediate", "output")
        
    Returns:
        Optimized DataFrame
    """
    if stage == "input":
        # For input data, be conservative with optimizations
        return optimize_dataframe_dtypes(
            df,
            target_memory_reduction=0.3,
            preserve_categorical=True,
        )
    
    elif stage == "intermediate":
        # For intermediate calculations, be more aggressive
        return optimize_dataframe_dtypes(
            df,
            target_memory_reduction=0.6,
            preserve_categorical=False,
        )
    
    elif stage == "output":
        # For final output, apply feature-specific optimizations
        return apply_feature_specific_optimization(df)
    
    return df

def estimate_memory_savings(df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate potential memory savings from data type optimization.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory savings estimates
    """
    current_memory = df.memory_usage(deep=True)
    total_current = current_memory.sum()
    
    # Simulate optimization
    optimized = optimize_dataframe_dtypes(df, preserve_categorical=True)
    optimized_memory = optimized.memory_usage(deep=True)
    total_optimized = optimized_memory.sum()
    
    savings = {
        "current_memory_mb": total_current / 1024**2,
        "optimized_memory_mb": total_optimized / 1024**2,
        "absolute_savings_mb": (total_current - total_optimized) / 1024**2,
        "percentage_savings": ((total_current - total_optimized) / total_current) * 100,
        "column_savings": {}
    }
    
    # Calculate savings per column
    for col in df.columns:
        if col in current_memory and col in optimized_memory:
            col_savings = (current_memory[col] - optimized_memory[col]) / 1024**2
            savings["column_savings"][col] = col_savings
    
    return savings

def get_optimization_recommendations(df: pd.DataFrame) -> list[str]:
    """
    Get specific recommendations for data type optimization.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of optimization recommendations
    """
    recommendations = []
    
    # Check for large integer columns
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
            recommendations.append(f"Column '{col}' can be optimized from int64 to int8")
        elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
            recommendations.append(f"Column '{col}' can be optimized from int64 to int16")
        elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
            recommendations.append(f"Column '{col}' can be optimized from int64 to int32")
    
    # Check for float64 columns
    for col in df.select_dtypes(include=["float64"]).columns:
        if df[col].isnull().sum() == 0:  # No NaN values
            recommendations.append(f"Column '{col}' can be optimized from float64 to float32")
    
    # Check for object columns that could be categorical
    for col in df.select_dtypes(include=["object"]).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5:
            recommendations.append(f"Column '{col}' can be optimized from object to category (unique ratio: {unique_ratio:.2%})")
    
    return recommendations
