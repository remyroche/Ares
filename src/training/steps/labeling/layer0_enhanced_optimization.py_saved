"""
Enhanced Layer0 Configuration for Advanced Price Filtering

This module extends the Layer0 parameter optimization to include:
- Median Filter optimization
- Adaptive Kalman filtering with noise estimation
- Robust VWAP with adaptive window sizing
- Enhanced parameter search spaces
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FilterType(Enum):
    """Available filtering methods for price processing."""
    STANDARD = "standard"
    MEDIAN_FILTER = "median_filter"
    HAMPEL_FILTER = "hampel_filter"
    ADAPTIVE_KALMAN = "adaptive_kalman"
    ROBUST_VWAP = "robust_vwap"

@dataclass
class Layer0EnhancedConfig:
    """Enhanced Layer0 configuration with advanced filtering options."""
    # Standard Kalman parameters
    kalman_Q_range: Tuple[float, float] = (1e-6, 1e-2)
    kalman_R_range: Tuple[float, float] = (1e-4, 1e-1)
    
    # VWAP parameters
    vwap_weight_range: Tuple[float, float] = (0.0, 1.0)
    vwap_lookback_range: Tuple[int, int] = (10, 200)
    
    # Median Filter parameters
    median_filter_enabled: bool = False
    median_window_range: Tuple[int, int] = (3, 15)
    
    # Hampel Filter parameters
    hampel_filter_enabled: bool = False
    hampel_window_range: Tuple[int, int] = (3, 15)
    hampel_threshold_range: Tuple[float, float] = (2.0, 5.0)
    
    # Adaptive Kalman parameters
    adaptive_kalman_enabled: bool = False
    adaptive_noise_window_range: Tuple[int, int] = (20, 100)
    adaptive_adaptation_rate_range: Tuple[float, float] = (0.05, 0.3)
    
    # Robust VWAP parameters
    robust_vwap_enabled: bool = False
    robust_min_lookback_range: Tuple[int, int] = (10, 50)
    robust_max_lookback_range: Tuple[int, int] = (100, 500)
    robust_volatility_window_range: Tuple[int, int] = (10, 50)

def get_enhanced_parameter_grid(config: Layer0EnhancedConfig = None) -> Dict[str, List]:
    """
    Generate enhanced parameter grid for Layer0 optimization including advanced filtering.
    
    Args:
        config: Enhanced configuration object
        
    Returns:
        Dictionary of parameter grids for optimization
    """
    if config is None:
        config = Layer0EnhancedConfig()
    
    # Standard parameter grid (base case)
    base_grid = {
        'kalman_Q': np.logspace(np.log10(config.kalman_Q_range[0]), np.log10(config.kalman_Q_range[1]), 20),
        'kalman_R': np.logspace(np.log10(config.kalman_R_range[0]), np.log10(config.kalman_R_range[1]), 20),
        'vwap_weight': np.linspace(config.vwap_weight_range[0], config.vwap_weight_range[1], 10),
        'vwap_lookback': np.linspace(config.vwap_lookback_range[0], config.vwap_lookback_range[1], 10)
    }
    
    # Enhanced filtering parameters
    enhanced_grid = base_grid.copy()
    
    # Median Filter parameters
    if config.median_filter_enabled:
        enhanced_grid['median_window'] = list(range(config.median_window_range[0], config.median_window_range[1] + 1, 2))
    
    # Hampel Filter parameters
    if config.hampel_filter_enabled:
        enhanced_grid['hampel_window'] = list(range(config.hampel_window_range[0], config.hampel_window_range[1] + 1, 2))
        enhanced_grid['hampel_threshold'] = np.linspace(config.hampel_threshold_range[0], config.hampel_threshold_range[1], 6)
    
    # Adaptive Kalman parameters
    if config.adaptive_kalman_enabled:
        enhanced_grid['adaptive_noise_window'] = list(range(config.adaptive_noise_window_range[0], config.adaptive_noise_window_range[1] + 5, 10))
        enhanced_grid['adaptive_adaptation_rate'] = np.linspace(config.adaptive_adaptation_rate_range[0], config.adaptive_adaptation_rate_range[1], 8)
    
    # Robust VWAP parameters
    if config.robust_vwap_enabled:
        enhanced_grid['robust_min_lookback'] = list(range(config.robust_min_lookback_range[0], config.robust_min_lookback_range[1] + 5, 10))
        enhanced_grid['robust_max_lookback'] = list(range(config.robust_max_lookback_range[0], config.robust_max_lookback_range[1] + 10, 25))
        enhanced_grid['robust_volatility_window'] = list(range(config.robust_volatility_window_range[0], config.robust_volatility_window_range[1] + 5, 5))
    
    return enhanced_grid

def evaluate_filter_performance(df: pd.DataFrame, 
                             params: dict,
                             filter_type: FilterType = FilterType.STANDARD) -> Dict[str, float]:
    """
    Evaluate performance of different filtering methods.
    
    Args:
        df: Market data DataFrame
        params: Filter parameters
        filter_type: Type of filtering to evaluate
        
    Returns:
        Performance metrics for the filtering method
    """
    try:
        from .unified_price_layer2 import (
            generate_unified_layer2_price,
            apply_median_filter,
            generate_adaptive_kalman_price,
            generate_robust_vwap_price
        )
        
        # Generate filtered price based on type
        if filter_type == FilterType.STANDARD:
            filtered_price = generate_unified_layer2_price(df, params)
        elif filter_type == FilterType.MEDIAN_FILTER:
            # Standard Kalman + Median Filter
            params['median_filter_enabled'] = True
            filtered_price = generate_unified_layer2_price(df, params)
        elif filter_type == FilterType.ADAPTIVE_KALMAN:
            # Adaptive Kalman only
            params['adaptive_kalman_enabled'] = True
            filtered_price = generate_adaptive_kalman_price(
                df, 
                params['kalman_Q'], 
                params['kalman_R'],
                params['adaptive_noise_window'],
                params['adaptive_adaptation_rate']
            )
        elif filter_type == FilterType.ROBUST_VWAP:
            # Robust VWAP only
            params['robust_vwap_enabled'] = True
            filtered_price = generate_robust_vwap_price(
                df,
                params['vwap_lookback'],
                params['robust_min_lookback'],
                params['robust_max_lookback'],
                params['robust_volatility_window']
            )
        else:
            filtered_price = df['close']
        
        # Calculate performance metrics
        raw_price = df['close']
        
        # 1. Noise reduction (lower is better)
        raw_volatility = raw_price.pct_change().std()
        filtered_volatility = filtered_price.pct_change().std()
        noise_reduction = 1 - (filtered_volatility / raw_volatility)
        
        # 2. Tracking accuracy (lower error is better)
        tracking_error = np.mean((filtered_price - raw_price) ** 2)
        max_acceptable_error = (raw_price.std() * 0.01) ** 2
        tracking_score = 1 - min(tracking_error / max_acceptable_error, 1.0)
        
        # 3. Edge preservation (how well edges are maintained)
        raw_edges = detect_price_edges(raw_price)
        filtered_edges = detect_price_edges(filtered_price)
        edge_preservation = calculate_edge_preservation(raw_edges, filtered_edges)
        
        # 4. Computational efficiency (lower time is better)
        import time
        start_time = time.time()
        for _ in range(10):  # Simulate multiple calls
            _ = generate_unified_layer2_price(df, params)
        computation_time = (time.time() - start_time) / 10
        
        return {
            'noise_reduction': noise_reduction,
            'tracking_score': tracking_score,
            'edge_preservation': edge_preservation,
            'computation_time': computation_time,
            'overall_score': (noise_reduction * 0.3 + tracking_score * 0.4 + edge_preservation * 0.3)
        }
        
    except Exception as e:
        logger.error(f"Filter evaluation failed: {e}")
        return {'overall_score': 0.0, 'error': str(e)}

def detect_price_edges(price_series: pd.Series, threshold: float = 0.5) -> pd.DatetimeIndex:
    """Detect price edges (significant price changes)."""
    price_changes = price_series.pct_change()
    return price_series.index[abs(price_changes) > threshold]

def calculate_edge_preservation(raw_edges: pd.DatetimeIndex, 
                               filtered_edges: pd.DatetimeIndex) -> float:
    """Calculate how well filtered price preserves edges from raw price."""
    if len(raw_edges) == 0 or len(filtered_edges) == 0:
        return 0.0
    
    # Calculate Jaccard similarity between edge sets
    intersection = len(raw_edges.intersection(filtered_edges))
    union = len(raw_edges.union(filtered_edges))
    
    return intersection / union if union > 0 else 0.0

def optimize_enhanced_parameters(df: pd.DataFrame,
                               config: Layer0EnhancedConfig = None) -> Dict[str, Any]:
    """
    Optimize enhanced Layer0 parameters using grid search.
    
    Args:
        df: Market data for optimization
        config: Enhanced configuration
        
    Returns:
        Optimization results with best parameters
    """
    if config is None:
        config = Layer0EnhancedConfig()
    
    logger.info("Starting enhanced Layer0 parameter optimization...")
    
    # Get parameter grid
    param_grid = get_enhanced_parameter_grid(config)
    
    best_score = 0.0
    best_params = {}
    best_filter_type = FilterType.STANDARD
    
    # Test each parameter combination
    for i, (kalman_Q, kalman_R, vwap_weight, vwap_lookback) in enumerate(
        np.array(np.meshgrid(
            param_grid['kalman_Q'],
            param_grid['kalman_R'], 
            param_grid['vwap_weight'],
            param_grid['vwap_lookback']
        ).T
    ):
        params = {
            'kalman_Q': kalman_Q,
            'kalman_R': kalman_R,
            'vwap_weight': vwap_weight,
            'vwap_lookback': int(vwap_lookback),
            'median_filter_enabled': config.median_filter_enabled,
            'median_window': config.median_window_range[1] if config.median_filter_enabled else 5,
            'adaptive_kalman_enabled': config.adaptive_kalman_enabled,
            'adaptive_noise_window': config.adaptive_noise_window_range[1] if config.adaptive_kalman_enabled else 50,
            'adaptive_adaptation_rate': config.adaptive_adaptation_range[1] if config.adaptive_kalman_enabled else 0.1,
            'robust_vwap_enabled': config.robust_vwap_enabled,
            'robust_min_lookback': config.robust_min_lookback_range[1] if config.robust_vwap_enabled else 20,
            'robust_max_lookback': config.robust_max_lookback_range[1] if config.robust_vwap_enabled else 100,
            'robust_volatility_window': config.robust_volatility_window_range[1] if config.robust_vwap_enabled else 20
        }
        
        # Test standard filtering
        standard_score = evaluate_filter_performance(df, params, FilterType.STANDARD)
        
        # Test enhanced filtering if enabled
        enhanced_score = standard_score
        current_filter_type = FilterType.STANDARD
        
        if config.median_filter_enabled:
            median_score = evaluate_filter_performance(df, params, FilterType.MEDIAN_FILTER)
            if median_score['overall_score'] > enhanced_score['overall_score']:
                enhanced_score = median_score
                current_filter_type = FilterType.MEDIAN_FILTER
        
        if config.adaptive_kalman_enabled:
            adaptive_score = evaluate_filter_performance(df, params, FilterType.ADAPTIVE_KALMAN)
            if adaptive_score['overall_score'] > enhanced_score['overall_score']:
                enhanced_score = adaptive_score
                current_filter_type = FilterType.ADAPTIVE_KALMAN
        
        if config.robust_vwap_enabled:
            robust_score = evaluate_filter_performance(df, params, FilterType.ROBUST_VWAP)
            if robust_score['overall_score'] > enhanced_score['overall_score']:
                enhanced_score = robust_score
                current_filter_type = FilterType.ROBUST_VWAP
        
        # Update best parameters
        if enhanced_score['overall_score'] > best_score:
            best_score = enhanced_score['overall_score']
            best_params = params.copy()
            best_filter_type = current_filter_type
        
        # Progress logging
        if (i + 1) % 50 == 0:
            logger.info(f"Optimization progress: {i+1}/{len(param_grid['kalman_Q'])} combinations")
            logger.info(f"Current best score: {best_score:.4f} using {current_filter_type.value}")
    
    # Final results
    logger.info(f"Enhanced optimization complete. Best score: {best_score:.4f}")
    logger.info(f"Best filter type: {best_filter_type.value}")
    logger.info(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_filter_type': best_filter_type,
        'all_scores': best_score,
        'param_grid': param_grid
    }

def save_enhanced_results(results: Dict[str, Any], filepath: str = None):
    """Save enhanced optimization results."""
    if filepath is None:
        filepath = f"outcomes/layer0_enhanced_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Enhanced results saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save enhanced results: {e}")

def compare_filtering_methods(df: pd.DataFrame, 
                               param_sets: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compare different filtering methods side-by-side.
    
    Args:
        df: Market data for comparison
        param_sets: List of parameter sets to compare
        
    Returns:
        Comparison results with performance metrics
    """
    if param_sets is None:
        # Default parameter sets for comparison
        param_sets = [
            {'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4, 'vwap_lookback': 50},
            {'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.6, 'vwap_lookback': 50},
            {'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4, 'vwap_lookback': 100},
        ]
    
    comparison_results = {}
    
    filter_types = [
        FilterType.STANDARD,
        FilterType.MEDIAN_FILTER,
        FilterType.ADAPTIVE_KALMAN,
        FilterType.ROBUST_VWAP
    ]
    
    for filter_type in filter_types:
        comparison_results[filter_type.value] = {}
        
        for i, params in enumerate(param_sets):
            try:
                # Set appropriate flags for this filter type
                temp_params = params.copy()
                temp_params['median_filter_enabled'] = (filter_type == FilterType.MEDIAN_FILTER)
                temp_params['adaptive_kalman_enabled'] = (filter_type == FilterType.ADAPTIVE_KALMAN)
                temp_params['robust_vwap_enabled'] = (filter_type == FilterType.ROBUST_VWAP)
                
                # Evaluate performance
                metrics = evaluate_filter_performance(df, temp_params, filter_type)
                comparison_results[filter_type.value][f'param_set_{i}'] = metrics
                
            except Exception as e:
                logger.error(f"Failed to evaluate {filter_type.value} with params {i}: {e}")
                comparison_results[filter_type.value][f'param_set_{i}'] = {'overall_score': 0.0, 'error': str(e)}
    
    return comparison_results

# Example usage and testing
if __name__name__ == "__main__":
    # Test enhanced filtering with sample data
    np.random.seed(42)
    
    # Generate sample price data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1H')
    prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)) + np.random.normal(0, 0.5, len(dates))  # Add some noise
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Add volume data
    df['volume'] = np.abs(np.random.normal(1000000, 500000, len(dates)))
    
    # Test enhanced configuration
    config = Layer0EnhancedConfig(
        median_filter_enabled=True,
        adaptive_kalman_enabled=True,
        robust_vwap_enabled=True,
        median_window_range=(3, 11),
        adaptive_noise_window_range=(30, 70),
        robust_min_lookback_range=(15, 35),
        robust_max_lookback_range=(100, 300)
    )
    
    # Run optimization
    results = optimize_enhanced_parameters(df, config)
    
    # Save results
    save_enhanced_results(results)
    
    # Compare methods
    comparison = compare_filtering_methods(df)
    
    print("\n=== Enhanced Layer2 Filtering Results ===")
    print(f"Best overall score: {results['best_score']:.4f}")
    print(f"Best filter type: {results['best_filter_type']}")
    print(f"Best parameters: {results['best_params']}")
    
    print("\n=== Method Comparison ===")
    for filter_type, results in comparison.items():
        scores = [r['overall_score'] for r in results.values() if 'overall_score' in r]
        print(f"{filter_type}:")
        print(f"  Mean score: {np.mean(scores):.4f}")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Worst score: {min(scores):.4f}")
