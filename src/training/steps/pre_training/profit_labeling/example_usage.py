"""
Example Usage of Volatility-Aware Multi-Horizon Profit Labeling System

This module provides comprehensive examples of how to use the volatility-aware
labeling system for creating high-quality, learnable labels.

Key Examples:
- Basic usage with default configuration
- Custom configuration for specific use cases
- Integration with existing ML pipelines
- Performance monitoring and optimization
- Batch processing for multiple datasets
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any

# Import the volatility-aware labeling system
from .volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelQualityScore,
    LabelingResult
)

from .bar_construction import BarConstructionConfig, BarType
from .volatility_modeling import VolatilityConfig, VolatilityMethod
from .noise_gating import NoiseGatingConfig, NoiseGateType
from .quality_scoring import QualityScoringConfig
from .multi_target_scheme import MultiTargetConfig, TargetBand

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning


def generate_sample_data(n_samples: int = 10000, 
                        start_date: str = "2023-01-01",
                        timeframe_minutes: int = 15) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.
    
    Args:
        n_samples: Number of samples to generate
        start_date: Start date for the data
        timeframe_minutes: Timeframe in minutes
        
    Returns:
        DataFrame with OHLCV data
    """
    tprint_info("📊 Generating sample data")
    
    # Create datetime index
    start_dt = pd.to_datetime(start_date)
    timestamps = pd.date_range(
        start=start_dt,
        periods=n_samples,
        freq=f'{timeframe_minutes}T'
    )
    
    # Generate price data with trend and volatility
    np.random.seed(42)
    
    # Base price
    base_price = 100.0
    
    # Generate returns with trend and volatility clustering
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Add volatility clustering
    volatility = np.ones(n_samples) * 0.02
    for i in range(1, n_samples):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * 0.02 + 0.01 * np.random.normal()
    
    returns = returns * volatility
    
    # Add some trend
    trend = np.linspace(0, 0.1, n_samples)  # 10% trend over the period
    returns = returns + trend / n_samples
    
    # Calculate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Generate intraday volatility
        intraday_vol = volatility[i] * 0.1
        
        # Generate OHLC
        open_price = price
        high_price = price * (1 + abs(np.random.normal(0, intraday_vol)))
        low_price = price * (1 - abs(np.random.normal(0, intraday_vol)))
        close_price = price * (1 + np.random.normal(0, intraday_vol))
        
        # Ensure OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume
        volume = np.random.lognormal(10, 0.5)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    tprint_success(f"✅ Generated {len(df)} samples of sample data")
    return df


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    tprint_info("🚀 Example 1: Basic Usage")
    
    # Generate sample data
    market_data = generate_sample_data(n_samples=5000)
    
    # Create labeler with default configuration
    labeler = VolatilityAwareMultiHorizonLabeler()
    
    # Generate labels
    result = labeler.generate_labels(market_data)
    
    # Display results
    tprint_success("✅ Basic usage completed")
    tprint_info(f"   → Generated {len(result.labels)} label samples")
    tprint_info(f"   → Number of targets: {result.n_targets}")
    tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
    
    # Show label distribution
    if not result.labels.empty:
        tprint_info("📊 Label distribution:")
        for col in result.labels.columns:
            value_counts = result.labels[col].value_counts()
            tprint_info(f"   → {col}: {dict(value_counts)}")
    
    return result


def example_custom_configuration():
    """Example 2: Custom configuration for specific use cases."""
    tprint_info("🚀 Example 2: Custom Configuration")
    
    # Generate sample data
    market_data = generate_sample_data(n_samples=3000)
    
    # Create custom configuration
    config = VolatilityAwareConfig(
        min_data_points=1000,
        enable_caching=True,
        parallel_processing=True
    )
    
    # Customize bar construction
    config.bar_construction = BarConstructionConfig(
        bar_type=BarType.DOLLAR,
        bar_size=500000.0,  # $500k bars
        enable_microstructure_filter=True,
        min_spread_ratio=0.0005
    )
    
    # Customize volatility modeling
    config.volatility = VolatilityConfig(
        method=VolatilityMethod.COMBINED,
        rv_window=30,
        atr_window=20,
        ewma_alpha=0.05
    )
    
    # Customize noise gating
    config.noise_gating = NoiseGatingConfig(
        gate_type=NoiseGateType.COMBINED,
        enable_micro_range_gating=True,
        min_move_ratio=2.0,
        enable_liquidity_gating=True
    )
    
    # Customize quality scoring
    config.quality_scoring = QualityScoringConfig(
        baseline_models=['logistic', 'random_forest'],
        n_splits=3,
        min_auc_threshold=0.6
    )
    
    # Customize multi-target scheme
    config.multi_target = MultiTargetConfig(
        small_band=(0.5, 0.9),
        medium_band=(0.9, 1.4),
        high_band=(1.4, 2.2),
        max_targets_per_band=1,
        min_lqs_score=0.4
    )
    
    # Create labeler with custom configuration
    labeler = VolatilityAwareMultiHorizonLabeler(config)
    
    # Generate labels
    result = labeler.generate_labels(market_data)
    
    # Display results
    tprint_success("✅ Custom configuration completed")
    tprint_info(f"   → Generated {len(result.labels)} label samples")
    tprint_info(f"   → Number of targets: {result.n_targets}")
    tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
    
    return result


def example_ml_integration():
    """Example 3: Integration with ML pipelines."""
    tprint_info("🚀 Example 3: ML Integration")
    
    # Generate sample data
    market_data = generate_sample_data(n_samples=4000)
    
    # Create labeler
    labeler = VolatilityAwareMultiHorizonLabeler()
    
    # Generate labels
    result = labeler.generate_labels(market_data)
    
    if result.labels.empty:
        tprint_warning("⚠️ No labels generated")
        return None
    
    # Prepare data for ML
    # Generate features (simplified example)
    features = pd.DataFrame(index=market_data.index)
    features['returns'] = market_data['close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    features['volume_ratio'] = market_data['volume'] / market_data['volume'].rolling(50).mean()
    features['price_momentum'] = market_data['close'] / market_data['close'].shift(20) - 1
    
    # Align features with labels
    common_index = features.index.intersection(result.labels.index)
    features_aligned = features.loc[common_index]
    labels_aligned = result.labels.loc[common_index]
    
    # Remove NaN values
    valid_mask = features_aligned.notna().all(axis=1) & labels_aligned.notna().all(axis=1)
    features_clean = features_aligned[valid_mask]
    labels_clean = labels_aligned[valid_mask]
    
    tprint_success("✅ ML integration data prepared")
    tprint_info(f"   → Features shape: {features_clean.shape}")
    tprint_info(f"   → Labels shape: {labels_clean.shape}")
    
    # Example: Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    # Use first target for demonstration
    target_col = labels_clean.columns[0]
    y = labels_clean[target_col]
    
    # Convert to binary classification
    y_binary = (y > 0).astype(int)
    
    if y_binary.nunique() < 2:
        tprint_warning("⚠️ Only one class in labels, skipping ML training")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, y_binary, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import roc_auc_score

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
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    tprint_success(f"✅ ML model trained with AUC: {auc_score:.3f}")
    
    return {
        'model': model,
        'features': features_clean,
        'labels': labels_clean,
        'auc_score': auc_score
    }


def example_performance_monitoring():
    """Example 4: Performance monitoring and optimization."""
    tprint_info("🚀 Example 4: Performance Monitoring")
    
    # Generate multiple datasets for monitoring
    datasets = []
    for i in range(5):
        data = generate_sample_data(n_samples=2000, start_date=f"2023-{i+1:02d}-01")
        datasets.append(data)
    
    # Create labeler
    labeler = VolatilityAwareMultiHorizonLabeler()
    
    # Process each dataset and monitor performance
    results = []
    for i, data in enumerate(datasets):
        tprint_info(f"📊 Processing dataset {i+1}/5")
        result = labeler.generate_labels(data)
        results.append(result)
    
    # Get performance summary
    perf_summary = labeler.get_performance_summary()
    
    tprint_success("✅ Performance monitoring completed")
    tprint_info("📊 Performance Summary:")
    for metric, stats in perf_summary.items():
        if isinstance(stats, dict) and 'current' in stats:
            tprint_info(f"   → {metric}: {stats['current']:.3f} (μ={stats['mean']:.3f}, σ={stats['std']:.3f})")
        else:
            tprint_info(f"   → {metric}: {stats}")
    
    return results, perf_summary


def example_batch_processing():
    """Example 5: Batch processing for multiple datasets."""
    tprint_info("🚀 Example 5: Batch Processing")
    
    # Generate multiple datasets
    datasets = {
        'ETH_1m': generate_sample_data(n_samples=10000, timeframe_minutes=1),
        'ETH_5m': generate_sample_data(n_samples=5000, timeframe_minutes=5),
        'ETH_15m': generate_sample_data(n_samples=2000, timeframe_minutes=15),
        'ETH_1h': generate_sample_data(n_samples=500, timeframe_minutes=60)
    }
    
    # Create labeler
    labeler = VolatilityAwareMultiHorizonLabeler()
    
    # Process each dataset
    batch_results = {}
    for name, data in datasets.items():
        tprint_info(f"📊 Processing {name}")
        result = labeler.generate_labels(data)
        batch_results[name] = result
        
        tprint_info(f"   → Generated {len(result.labels)} labels")
        tprint_info(f"   → Processing time: {result.processing_time:.2f}s")
    
    # Compare results across timeframes
    tprint_success("✅ Batch processing completed")
    tprint_info("📊 Results comparison:")
    for name, result in batch_results.items():
        tprint_info(f"   → {name}: {result.n_targets} targets, {result.n_samples} samples")
    
    return batch_results


def example_quality_analysis():
    """Example 6: Quality analysis and visualization."""
    tprint_info("🚀 Example 6: Quality Analysis")
    
    # Generate sample data
    market_data = generate_sample_data(n_samples=3000)
    
    # Create labeler
    labeler = VolatilityAwareMultiHorizonLabeler()
    
    # Generate labels
    result = labeler.generate_labels(market_data)
    
    if result.labels.empty:
        tprint_warning("⚠️ No labels generated")
        return None
    
    # Analyze quality scores
    if result.quality_scores:
        tprint_info("📊 Quality Analysis:")
        for target_name, quality_score in result.quality_scores.items():
            tprint_info(f"   → {target_name}:")
            tprint_info(f"     - Overall Quality: {quality_score.overall_quality:.3f}")
            tprint_info(f"     - Predictability: {quality_score.predictability:.3f}")
            tprint_info(f"     - Stability: {quality_score.stability:.3f}")
            tprint_info(f"     - Consistency: {quality_score.consistency:.3f}")
            tprint_info(f"     - Balance: {quality_score.balance:.3f}")
            tprint_info(f"     - SNR Proxy: {quality_score.snr_proxy:.3f}")
    
    # Analyze label distribution
    tprint_info("📊 Label Distribution Analysis:")
    for col in result.labels.columns:
        value_counts = result.labels[col].value_counts()
        total = len(result.labels)
        tprint_info(f"   → {col}:")
        for value, count in value_counts.items():
            percentage = (count / total) * 100
            tprint_info(f"     - {value}: {count} ({percentage:.1f}%)")
    
    return result


def run_all_examples():
    """Run all examples."""
    tprint_info("🚀 Running All Examples")
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("ML Integration", example_ml_integration),
        ("Performance Monitoring", example_performance_monitoring),
        ("Batch Processing", example_batch_processing),
        ("Quality Analysis", example_quality_analysis)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            tprint_info(f"\n{'='*50}")
            tprint_info(f"Running: {name}")
            tprint_info(f"{'='*50}")
            
            result = example_func()
            results[name] = result
            
            tprint_success(f"✅ {name} completed successfully")
            
        except Exception as e:
            tprint_warning(f"⚠️ {name} failed: {e}")
            results[name] = None
    
    tprint_success("\n🎉 All examples completed!")
    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Print summary
    tprint_info("\n📊 Summary:")
    for name, result in results.items():
        status = "✅ Success" if result is not None else "❌ Failed"
        tprint_info(f"   → {name}: {status}")

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
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
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
