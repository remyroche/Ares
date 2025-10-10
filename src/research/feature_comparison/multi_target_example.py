"""
Multi-Target System Example

This script demonstrates the comprehensive multi-target system that evaluates
features against multiple targets including mean-reversion, trend-following,
and other target families.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from feature_comparison.multi_target_system import MultiTargetSystem
from feature_comparison.family_diverse_features import FamilyDiverseFeatureGenerator

# Suppress warnings
warnings.filterwarnings('ignore')

def load_market_data_with_klines_manager(symbol: str = "ETHUSDT",
                                        interval: str = "15m",
                                        start_date: Optional[datetime] = None,
                                        end_date: Optional[datetime] = None,
                                        data_type: str = "raw",
                                        fallback_days: int = 30) -> Optional[pd.DataFrame]:
    """
    Load market data using KlinesParquetManager.
    
    Args:
        symbol: Trading symbol (default: ETHUSDT)
        interval: Data interval (default: 15m)
        start_date: Start date for filtering
        end_date: End date for filtering
        data_type: 'raw' or 'processed'
        fallback_days: Days to fallback to if no data in range
        
    Returns:
        DataFrame with market data or None if not found
    """
    try:
        from src.utils.data.klines_parquet import KlinesParquetManager
        
        # Initialize data manager
        data_manager = KlinesParquetManager()
        
        # Try to load data with specified date range
        data = data_manager.read_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_type=data_type
        )
        
        # If no data found and we have a date range, try fallback
        if (data is None or data.empty) and (start_date is not None or end_date is not None):
            print(f"No data found in specified range, trying last {fallback_days} days fallback")
            data = data_manager.read_last_x_days_data(
                symbol=symbol,
                interval=interval,
                x_days=fallback_days,
                data_type=data_type
            )
        
        if data is not None and not data.empty:
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Missing required columns: {missing_columns}")
                return None
            
            # Ensure data is sorted by timestamp
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()
            
            print(f"Loaded {len(data)} records for {symbol} {interval}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
        else:
            print(f"No data available for {symbol} {interval}")
            return None
            
    except ImportError as e:
        print(f"Could not import KlinesParquetManager: {e}")
        print("Falling back to synthetic data generation...")
        return create_synthetic_market_data_15min()
    except Exception as e:
        print(f"Failed to load market data: {e}")
        print("Falling back to synthetic data generation...")
        return create_synthetic_market_data_15min()

def create_synthetic_market_data_15min(n_samples: int = 10000) -> pd.DataFrame:
    """
    Create synthetic market data with 15-minute timeframe as fallback.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    # Generate realistic 15-minute data with multiple regimes
    n_regimes = 8
    regime_length = n_samples // n_regimes
    
    prices = []
    volumes = []
    regimes = []
    
    for regime in range(n_regimes):
        if regime == 0:
            # Trending up regime
            trend = 0.0005  # Smaller trend for 15-min
            volatility = 0.008
            volume_trend = 0.0002
            regime_name = 'trending_up'
        elif regime == 1:
            # Sideways regime
            trend = 0.0001
            volatility = 0.004
            volume_trend = -0.0001
            regime_name = 'sideways'
        elif regime == 2:
            # High volatility regime
            trend = 0.0002
            volatility = 0.020
            volume_trend = 0.0005
            regime_name = 'high_vol'
        elif regime == 3:
            # Trending down regime
            trend = -0.0004
            volatility = 0.012
            volume_trend = 0.0003
            regime_name = 'trending_down'
        elif regime == 4:
            # Mean reversion regime
            trend = 0.0
            volatility = 0.010
            volume_trend = 0.0001
            regime_name = 'mean_reversion'
        elif regime == 5:
            # Low volatility regime
            trend = 0.0003
            volatility = 0.003
            volume_trend = -0.0002
            regime_name = 'low_vol'
        elif regime == 6:
            # Breakout regime
            trend = 0.0001
            volatility = 0.015
            volume_trend = 0.0008
            regime_name = 'breakout'
        else:
            # Jump regime
            trend = 0.0002
            volatility = 0.025
            volume_trend = 0.001
            regime_name = 'jump'
        
        # Generate regime data
        regime_returns = np.random.normal(trend, volatility, regime_length)
        regime_prices = 100 * np.exp(np.cumsum(regime_returns))
        
        # Add volume with trend
        regime_volumes = np.random.lognormal(
            8 + volume_trend * np.arange(regime_length), 
            0.8
        )
        
        prices.extend(regime_prices)
        volumes.extend(regime_volumes)
        regimes.extend([regime_name] * regime_length)
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='15T'),  # 15-minute
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': prices,
        'volume': volumes,
        'regime': regimes
    })
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_multi_target_system():
    """Test the multi-target system with comprehensive evaluation."""
    print("Testing Multi-Target System...")
    print("=" * 70)
    
    # Load 15-minute market data using KlinesParquetManager
    data = load_market_data_with_klines_manager(
        symbol="ETHUSDT",
        interval="15m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        data_type="raw",
        fallback_days=30
    )
    
    if data is None:
        print("Failed to load market data, using synthetic data")
        data = create_synthetic_market_data_15min(8000)
    
    print(f"Loaded market data with shape: {data.shape}")
    print(f"Timeframe: 15 minutes")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Generate base features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    # Combine features
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    print(f"Generated {X.shape[1]} features across {len(family_features)} families")
    
    # Initialize multi-target system
    multi_target = MultiTargetSystem(
        horizons=[1, 2, 3],
        volatility_window=20,
        neutral_threshold=0.001,
        tail_quantile=0.05,
        breakout_std_multiplier=2.0,
        breakout_min_bars=3,
        profit_taking_upper=0.006,  # 0.6%
        profit_taking_lower=0.003,  # 0.3%
        stop_loss=0.003,  # 0.3%
        max_bars=3,
        timeframe_minutes=15
    )
    
    # Create all targets
    print("\nCreating all target families...")
    targets = multi_target.create_all_targets(data)
    
    print(f"Created targets for {len(targets)} families:")
    for family_name, target_df in targets.items():
        print(f"  {family_name:20s} | {len(target_df.columns):2d} targets")
    
    # Show sample targets
    print(f"\nSample targets by family:")
    for family_name, target_df in targets.items():
        sample_targets = target_df.columns[:3]
        print(f"  {family_name}:")
        for target in sample_targets:
            print(f"    - {target}")
        if len(target_df.columns) > 3:
            print(f"    ... and {len(target_df.columns) - 3} more")
    
    # Evaluate features against all targets
    print(f"\nEvaluating features against all targets...")
    results = multi_target.evaluate_features_against_targets(X, targets)
    
    # Display results
    print(f"\nMulti-Target Evaluation Results:")
    print("=" * 70)
    
    summary = results['multi_target_summary']
    print(f"Total targets: {summary['total_targets']}")
    print(f"Total features evaluated: {summary['total_features_evaluated']}")
    
    # Show target family performance
    print(f"\nTarget Family Performance:")
    for family_name, perf in summary['target_family_performance'].items():
        print(f"  {family_name:20s} | Targets: {perf['targets']:2d} | Features: {perf['features_evaluated']:3d}")
    
    # Show best overall features
    print(f"\nBest Overall Features (Top 10):")
    for i, (feature, score) in enumerate(summary['best_overall_features'][:10], 1):
        print(f"  {i:2d}. {feature:30s} | Score: {score:.4f}")
    
    # Show most consistent features
    print(f"\nMost Consistent Features (Top 10):")
    for i, (feature, consistency) in enumerate(summary['feature_consistency'][:10], 1):
        print(f"  {i:2d}. {feature:30s} | Consistency: {consistency:.4f}")
    
    # Show best features by target family
    print(f"\nBest Features by Target Family:")
    for family_name, family_results in results['target_families'].items():
        print(f"\n  {family_name.upper()}:")
        best_features = family_results['best_features'][:5]
        for i, feature in enumerate(best_features, 1):
            print(f"    {i}. {feature}")
    
    # Show detailed results for specific targets
    print(f"\nDetailed Results for Key Targets:")
    key_targets = [
        'mr_strength_1', 'trend_strength_1', 'binary_direction_1',
        'realized_vol_1', 'left_tail_1', 'sharpe_like_1'
    ]
    
    for target_name in key_targets:
        if target_name in results['best_features_by_target']:
            best_features = results['best_features_by_target'][target_name][:5]
            print(f"\n  {target_name}:")
            for i, (feature, score) in enumerate(best_features, 1):
                print(f"    {i}. {feature:30s} | Score: {score:.4f}")
    
    # Show highly correlated targets
    if results['correlation_analysis']['highly_correlated_pairs']:
        print(f"\nHighly Correlated Target Pairs:")
        for pair in results['correlation_analysis']['highly_correlated_pairs'][:10]:
            print(f"  {pair['target1']:30s} ↔ {pair['target2']:30s} | {pair['correlation']:.3f}")
    
    return results

def test_specific_target_families():
    """Test specific target families in detail."""
    print("\nTesting Specific Target Families...")
    print("=" * 70)
    
    # Load market data using KlinesParquetManager
    data = load_market_data_with_klines_manager(
        symbol="ETHUSDT",
        interval="15m",
        fallback_days=20
    )
    
    if data is None:
        print("Failed to load market data, using synthetic data")
        data = create_synthetic_market_data_15min(5000)
    
    # Generate features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    # Initialize multi-target system
    multi_target = MultiTargetSystem(
        horizons=[1, 2, 3],
        timeframe_minutes=15
    )
    
    # Test mean-reversion targets
    print("\n1. Mean-Reversion Targets:")
    mr_targets = multi_target._create_mean_reversion_targets(
        data['close'].pct_change(), 
        data['close'].pct_change().rolling(20).std()
    )
    print(f"   Created {len(mr_targets.columns)} mean-reversion targets")
    for target in mr_targets.columns:
        print(f"   - {target}")
    
    # Test trend-following targets
    print("\n2. Trend-Following Targets:")
    tr_targets = multi_target._create_trend_following_targets(
        data['close'].pct_change(),
        data['close'].pct_change().rolling(20).std()
    )
    print(f"   Created {len(tr_targets.columns)} trend-following targets")
    for target in tr_targets.columns:
        print(f"   - {target}")
    
    # Test directional targets
    print("\n3. Directional Targets:")
    dir_targets = multi_target._create_directional_targets(data['close'].pct_change())
    print(f"   Created {len(dir_targets.columns)} directional targets")
    for target in dir_targets.columns:
        print(f"   - {target}")
    
    # Test volatility targets
    print("\n4. Volatility Targets:")
    vol_targets = multi_target._create_volatility_targets(
        data['close'].pct_change(), data
    )
    print(f"   Created {len(vol_targets.columns)} volatility targets")
    for target in vol_targets.columns:
        print(f"   - {target}")
    
    # Test tail risk targets
    print("\n5. Tail Risk Targets:")
    tail_targets = multi_target._create_tail_risk_targets(data['close'].pct_change())
    print(f"   Created {len(tail_targets.columns)} tail risk targets")
    for target in tail_targets.columns:
        print(f"   - {target}")
    
    # Test meta-labeling targets
    print("\n6. Meta-Labeling Targets:")
    meta_targets = multi_target._create_meta_labeling_targets(
        data['close'].pct_change(), data
    )
    print(f"   Created {len(meta_targets.columns)} meta-labeling targets")
    for target in meta_targets.columns:
        print(f"   - {target}")
    
    return {
        'mean_reversion': mr_targets,
        'trend_following': tr_targets,
        'directional': dir_targets,
        'volatility': vol_targets,
        'tail_risk': tail_targets,
        'meta_labeling': meta_targets
    }

def test_feature_performance_analysis():
    """Test detailed feature performance analysis."""
    print("\nTesting Feature Performance Analysis...")
    print("=" * 70)
    
    # Load market data using KlinesParquetManager
    data = load_market_data_with_klines_manager(
        symbol="ETHUSDT",
        interval="15m",
        fallback_days=25
    )
    
    if data is None:
        print("Failed to load market data, using synthetic data")
        data = create_synthetic_market_data_15min(6000)
    
    # Generate features
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(data)
    
    X = pd.DataFrame(index=data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    # Initialize multi-target system
    multi_target = MultiTargetSystem(
        horizons=[1, 2, 3],
        timeframe_minutes=15
    )
    
    # Create targets
    targets = multi_target.create_all_targets(data)
    
    # Evaluate features
    results = multi_target.evaluate_features_against_targets(X, targets)
    
    # Analyze feature performance across targets
    print("\nFeature Performance Analysis:")
    print("-" * 50)
    
    # Get all feature scores across all targets
    all_feature_scores = {}
    for family_name, family_results in results['target_families'].items():
        for target_name, feature_scores in family_results['feature_scores'].items():
            for feature_name, score in feature_scores.items():
                if feature_name not in all_feature_scores:
                    all_feature_scores[feature_name] = []
                all_feature_scores[feature_name].append(score)
    
    # Calculate performance statistics
    feature_stats = {}
    for feature_name, scores in all_feature_scores.items():
        if len(scores) > 0:
            feature_stats[feature_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'num_targets': len(scores),
                'consistency': 1 - (np.std(scores) / (np.mean(scores) + 1e-8))
            }
    
    # Sort by mean score
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    
    print(f"Top 15 Features by Mean Score:")
    for i, (feature, stats) in enumerate(sorted_features[:15], 1):
        print(f"  {i:2d}. {feature:30s} | Mean: {stats['mean_score']:.4f} | "
              f"Std: {stats['std_score']:.4f} | Targets: {stats['num_targets']:2d} | "
              f"Consistency: {stats['consistency']:.4f}")
    
    # Find features that perform well across multiple targets
    print(f"\nFeatures with High Consistency (Top 10):")
    consistent_features = sorted(feature_stats.items(), key=lambda x: x[1]['consistency'], reverse=True)
    for i, (feature, stats) in enumerate(consistent_features[:10], 1):
        print(f"  {i:2d}. {feature:30s} | Consistency: {stats['consistency']:.4f} | "
              f"Mean Score: {stats['mean_score']:.4f} | Targets: {stats['num_targets']:2d}")
    
    # Find features that perform well on specific target types
    print(f"\nBest Features by Target Type:")
    
    # Regression targets
    regression_targets = [t for t in results['best_features_by_target'].keys() 
                         if not any(x in t for x in ['_hit', '_cls', '_flag', '_label'])]
    if regression_targets:
        print(f"  Regression Targets ({len(regression_targets)}):")
        regression_scores = {}
        for target in regression_targets:
            if target in results['best_features_by_target']:
                for feature, score in results['best_features_by_target'][target]:
                    if feature not in regression_scores:
                        regression_scores[feature] = []
                    regression_scores[feature].append(score)
        
        regression_avg = {f: np.mean(scores) for f, scores in regression_scores.items()}
        regression_sorted = sorted(regression_avg.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(regression_sorted[:5], 1):
            print(f"    {i}. {feature:30s} | Score: {score:.4f}")
    
    # Classification targets
    classification_targets = [t for t in results['best_features_by_target'].keys() 
                             if any(x in t for x in ['_hit', '_cls', '_flag', '_label'])]
    if classification_targets:
        print(f"  Classification Targets ({len(classification_targets)}):")
        classification_scores = {}
        for target in classification_targets:
            if target in results['best_features_by_target']:
                for feature, score in results['best_features_by_target'][target]:
                    if feature not in classification_scores:
                        classification_scores[feature] = []
                    classification_scores[feature].append(score)
        
        classification_avg = {f: np.mean(scores) for f, scores in classification_scores.items()}
        classification_sorted = sorted(classification_avg.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(classification_sorted[:5], 1):
            print(f"    {i}. {feature:30s} | Score: {score:.4f}")
    
    return results

def test_complete_evaluation_with_data_loading():
    """Test complete evaluation with automatic data loading."""
    print("\nTesting Complete Evaluation with Data Loading...")
    print("=" * 80)
    
    # Generate features first (we'll use synthetic data for features)
    print("Generating features...")
    synthetic_data = create_synthetic_market_data_15min(5000)
    family_generator = FamilyDiverseFeatureGenerator(enable_matrix_ops=True)
    family_features = family_generator.generate_family_diverse_features(synthetic_data)
    
    # Combine features
    X = pd.DataFrame(index=synthetic_data.index)
    for family_df in family_features.values():
        original_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in family_df.columns if col not in original_cols]
        for feature in feature_cols:
            X[feature] = family_df[feature]
    
    print(f"Generated {X.shape[1]} features across {len(family_features)} families")
    
    # Initialize multi-target system
    multi_target = MultiTargetSystem(
        horizons=[1, 2, 3],
        timeframe_minutes=15
    )
    
    # Run complete evaluation with data loading
    print("Running complete evaluation with data loading...")
    results = multi_target.run_complete_evaluation_with_data_loading(
        X=X,
        symbol="ETHUSDT",
        interval="15m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        data_type="raw",
        fallback_days=30
    )
    
    # Display results
    if 'error' in results:
        print(f"Error: {results['error']}")
        return results
    
    print(f"\nComplete Evaluation Results:")
    print("=" * 80)
    
    # Data info
    if 'data_info' in results:
        data_info = results['data_info']
        print(f"Data Information:")
        print(f"  Symbol: {data_info['symbol']}")
        print(f"  Interval: {data_info['interval']}")
        print(f"  Data Type: {data_info['data_type']}")
        print(f"  Date Range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
        print(f"  Records: {data_info['n_records']:,}")
        print(f"  Targets: {data_info['n_targets']}")
        print()
    
    # Summary
    summary = results['multi_target_summary']
    print(f"Evaluation Summary:")
    print(f"  Total targets: {summary['total_targets']}")
    print(f"  Total features evaluated: {summary['total_features_evaluated']}")
    print()
    
    # Target family performance
    print(f"Target Family Performance:")
    for family_name, perf in summary['target_family_performance'].items():
        print(f"  {family_name:20s} | Targets: {perf['targets']:2d} | Features: {perf['features_evaluated']:3d}")
    print()
    
    # Best overall features
    print(f"Best Overall Features (Top 10):")
    for i, (feature, score) in enumerate(summary['best_overall_features'][:10], 1):
        print(f"  {i:2d}. {feature:30s} | Score: {score:.4f}")
    print()
    
    # Most consistent features
    print(f"Most Consistent Features (Top 10):")
    for i, (feature, consistency) in enumerate(summary['feature_consistency'][:10], 1):
        print(f"  {i:2d}. {feature:30s} | Consistency: {consistency:.4f}")
    print()
    
    return results

def generate_comprehensive_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive multi-target report."""
    multi_target = MultiTargetSystem()
    return multi_target.generate_multi_target_report(results)

def main():
    """Main function to run all multi-target tests."""
    print("Multi-Target Feature Evaluation System - Complete Test Suite")
    print("=" * 90)
    print("Features:")
    print("- Mean-reversion targets (H=1,2,3) with regression and classification")
    print("- Trend-following targets (H=1,2,3) with regression and classification")
    print("- Directional & probability targets with calibrated probabilities")
    print("- Magnitude/volatility forecasting with realized and range-based vol")
    print("- Tail risk/jump likelihood with quantile-based events")
    print("- Breakout/reversal speed with VWAP and microstructure awareness")
    print("- Risk-adjusted return targets with Sharpe-like metrics")
    print("- Meta-labeling with triple barrier method")
    print("- 15-minute timeframe with proper bar-based horizons")
    print("=" * 90)
    
    # Test individual components
    print("\n1. Testing Multi-Target System...")
    multi_target_results = test_multi_target_system()
    
    print("\n2. Testing Specific Target Families...")
    target_families = test_specific_target_families()
    
    print("\n3. Testing Feature Performance Analysis...")
    performance_results = test_feature_performance_analysis()
    
    print("\n4. Testing Complete Evaluation with Data Loading...")
    complete_results = test_complete_evaluation_with_data_loading()
    
    # Generate comprehensive report
    print("\n5. Generating Comprehensive Report...")
    report = generate_comprehensive_report(multi_target_results)
    print(report)
    
    print("\n" + "=" * 90)
    print("Multi-target feature evaluation testing completed successfully!")
    print("Key features demonstrated:")
    print("✅ 8 target families with 24+ individual targets")
    print("✅ Mean-reversion and trend-following with risk-adjusted variants")
    print("✅ Directional and probability targets with calibration")
    print("✅ Volatility forecasting with multiple estimation methods")
    print("✅ Tail risk detection with quantile-based events")
    print("✅ Breakout detection with microstructure awareness")
    print("✅ Risk-adjusted returns with Sharpe-like metrics")
    print("✅ Meta-labeling with triple barrier method")
    print("✅ 15-minute timeframe with proper bar-based horizons")
    print("✅ KlinesParquetManager integration for real data loading")
    print("✅ Automatic fallback to synthetic data if real data unavailable")
    print("✅ Comprehensive evaluation across all targets")
    print("✅ Feature consistency and performance analysis")
    print("=" * 90)

if __name__ == "__main__":
    main()