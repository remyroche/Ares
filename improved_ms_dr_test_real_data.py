#!/usr/bin/env python3
"""
Improved MS-DR Clustering Test with REAL Market Data

This version loads real market data through KlinesParquetManager/ArtifactManager
instead of using synthetic data.

Usage:
    python3 improved_ms_dr_test_real_data.py --symbol ETHUSDT --timeframe 1h --limit 2000
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor
from improved_ms_dr_signal import create_improved_regime_signal
from improved_ms_dr_test import enhanced_burn_in_removal

# Try to import data loaders
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    KLINES_MANAGER_AVAILABLE = True
    print("✅ KlinesParquetManager available")
except ImportError as e:
    KLINES_MANAGER_AVAILABLE = False
    print(f"⚠️ KlinesParquetManager not available: {e}")

try:
    from src.training.steps.pre_training.utils.artifact_manager import (
        get_pretraining_artifact_manager,
        artifact_context
    )
    ARTIFACT_MANAGER_AVAILABLE = True
    print("✅ PreTraining ArtifactManager available")
except ImportError as e:
    ARTIFACT_MANAGER_AVAILABLE = False
    print(f"⚠️ ArtifactManager not available: {e}")


def load_market_data(symbol: str = "ETHUSDT",
                     exchange: str = "binance",
                     timeframe: str = "1h",
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None,
                     limit: int = 2000) -> Optional[pd.DataFrame]:
    """
    Load real market data using available data managers.
    
    Tries multiple sources in order:
    1. KlinesParquetManager
    2. PreTraining ArtifactManager
    3. Generate synthetic data as fallback
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '1h', '15m', '5m')
        start_date: Start date for data
        end_date: End date for data
        limit: Maximum number of records to load
        
    Returns:
        DataFrame with OHLCV data or None
    """
    print("\n" + "=" * 80)
    print(f"📊 LOADING REAL MARKET DATA")
    print("=" * 80)
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Limit: {limit} records")
    
    # Try KlinesParquetManager first (most common)
    if KLINES_MANAGER_AVAILABLE:
        print("\n🔍 Trying KlinesParquetManager...")
        try:
            klines_manager = get_klines_manager(data_dir='data_cache', exchange=exchange)
            
            # Load data
            df = klines_manager.read_data(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                data_type="processed"
            )
            
            if df is not None and not df.empty:
                # Limit to requested number of records (most recent)
                if len(df) > limit:
                    df = df.tail(limit)
                
                print(f"✅ Loaded {len(df)} records from KlinesParquetManager")
                print(f"   Date range: {df.index.min()} to {df.index.max()}")
                print(f"   Columns: {list(df.columns)}")
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    return df
                else:
                    print(f"⚠️ Missing required columns. Has: {list(df.columns)}")
        
        except Exception as e:
            print(f"⚠️ KlinesParquetManager failed: {e}")
    
    # Try PreTraining ArtifactManager
    if ARTIFACT_MANAGER_AVAILABLE:
        print("\n🔍 Trying PreTraining ArtifactManager...")
        try:
            with artifact_context(symbol=symbol, exchange=exchange, timeframe=timeframe) as am:
                # Try to load from various steps
                for step_name in ['data_collection', 'data_preparation', 'feature_generation']:
                    df = am.load(step_name, 'cleaned_dataframe')
                    if df is not None and not df.empty:
                        # Limit to requested number of records
                        if len(df) > limit:
                            df = df.tail(limit)
                        
                        print(f"✅ Loaded {len(df)} records from ArtifactManager ({step_name})")
                        print(f"   Date range: {df.index.min()} to {df.index.max()}")
                        return df
        
        except Exception as e:
            print(f"⚠️ ArtifactManager failed: {e}")
    
    # Fallback: Generate synthetic data
    print("\n⚠️ No real data available, generating synthetic data for testing...")
    print("   (This is for demonstration only - use real data in production!)")
    
    return generate_synthetic_data(limit, symbol, timeframe)


def generate_synthetic_data(n_samples: int = 2000,
                           symbol: str = "ETHUSDT",
                           timeframe: str = "1h") -> pd.DataFrame:
    """Generate synthetic market data as fallback."""
    print(f"\n🔨 Generating {n_samples} synthetic {timeframe} candles...")
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq=timeframe)
    base_price = 3000.0
    
    # Create 3 distinct market regimes with realistic parameters
    regime_lengths = [int(n_samples * 0.35), int(n_samples * 0.30), int(n_samples * 0.35)]
    regime_params = [
        {'volatility': 0.02, 'trend': 0.001, 'volume': 1.5, 'name': 'Bull'},
        {'volatility': 0.05, 'trend': -0.0005, 'volume': 0.8, 'name': 'Bear'},
        {'volatility': 0.01, 'trend': 0.0, 'volume': 1.0, 'name': 'Sideways'}
    ]
    
    prices = [base_price]
    volumes = []
    regime_idx = 0
    regime_counter = 0
    
    for i in range(n_samples):
        if regime_counter >= regime_lengths[regime_idx]:
            regime_idx = (regime_idx + 1) % 3
            regime_counter = 0
            print(f"   Switching to {regime_params[regime_idx]['name']} regime at candle {i}")
        
        params = regime_params[regime_idx]
        price_change = np.random.normal(params['trend'], params['volatility'])
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
        
        volume = np.random.uniform(500 * params['volume'], 2000 * params['volume'])
        volumes.append(volume)
        
        regime_counter += 1
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Generated {len(df)} synthetic candles")
    return df


def run_improved_ms_dr_test_real_data(symbol: str = "ETHUSDT",
                                      exchange: str = "binance",
                                      timeframe: str = "1h",
                                      limit: int = 2000,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None):
    """Run MS-DR test with real market data."""
    
    print("=" * 80)
    print("🎯 IMPROVED MS-DR CLUSTERING TEST WITH REAL DATA")
    print("=" * 80)
    
    # Create outcomes directory
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"improved_ms_dr_real_data_{symbol}_{timeframe}_{timestamp}.md"
    
    print(f"📝 Report will be saved to: {report_filename}")
    
    # === STEP 1: Load real market data ===
    df = load_market_data(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    if df is None or df.empty:
        print("\n❌ Failed to load market data!")
        return None, None, None, None
    
    print(f"\n✅ Market data loaded successfully: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Columns: {list(df.columns)}")
    
    # === STEP 2: Create improved regime signal ===
    print("\n" + "=" * 80)
    print("🔧 STEP 2: Creating Improved Regime Signal")
    print("=" * 80)
    
    try:
        regime_signal, signal_diagnostics = create_improved_regime_signal(
            df,
            use_nonlinear=True,
            use_multiscale=True,
            use_adaptive_weights=True
        )
        
        # Validate signal quality
        print("\n📊 Signal Quality Check:")
        quality = signal_diagnostics['signal_quality']
        
        warnings = []
        if quality['std'] < 0.5:
            warnings.append(f"Low variance (std={quality['std']:.3f})")
        if quality['range'] < 3.0:
            warnings.append(f"Narrow range ({quality['range']:.3f})")
        if quality['transition_rate'] < 0.1:
            warnings.append(f"Low transition rate ({quality['transition_rate']:.3f})")
        
        if warnings:
            print("⚠️ Signal quality warnings:")
            for w in warnings:
                print(f"   - {w}")
            print("   (This may indicate uniform market conditions)")
        else:
            print("✅ Signal quality is good")
        
    except Exception as e:
        print(f"\n❌ Failed to create regime signal: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    # Prepare data for MS-DR
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✅ Prepared data for MS-DR: {data.shape}")
    print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"   Data mean: {data.mean():.4f}, std: {data.std():.4f}")
    
    # === STEP 3: Run MS-DR clustering ===
    print("\n" + "=" * 80)
    print("🚀 STEP 3: Running MS-DR Clustering")
    print("=" * 80)
    
    config = MSDRConfig(
        n_regimes=3,
        auto_select_regimes=True,
        model_type='autoregression',
        switching_variance=True,
        enable_pca=False,
        min_regimes=2,
        max_regimes=5,
        ic_criterion='bic',
        order=2,
        max_iter=3000,
        method='powell',
        random_state=42,
        use_memory_optimization=True,
        use_hardware_acceleration=True,
        show_progress=True
    )
    
    try:
        clusterer = MSDRClusterer(config)
        result = clusterer.fit_predict(data)
        
        print(f"\n✅ MS-DR clustering completed")
        print(f"   Success: {result.success}")
        print(f"   N clusters: {result.n_clusters}")
        print(f"   Processing time: {result.processing_time:.2f}s")
        
    except Exception as e:
        print(f"\n❌ MS-DR clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    
    # === STEP 4: Apply enhanced burn-in detection ===
    print("\n" + "=" * 80)
    print("🔧 STEP 4: Enhanced Burn-in Detection & Removal")
    print("=" * 80)
    
    labels_clean, probs_clean, data_clean, was_cleaned, burn_in_diag = enhanced_burn_in_removal(
        result, data, aggressive=False
    )
    
    # Update result if cleaned
    if was_cleaned:
        result.cluster_labels = labels_clean
        result.cluster_probabilities = probs_clean
        result.n_clusters = len(np.unique(labels_clean))
        data = data_clean
    
    # === STEP 5: Quality assessment ===
    print("\n" + "=" * 80)
    print("📊 STEP 5: Quality Assessment")
    print("=" * 80)
    
    feature_df = pd.DataFrame(data, columns=['composite_regime_signal'])
    
    quality_assessor = ClusterQualityAssessor(
        enable_hardware_optimization=True,
        enable_vectorization=True
    )
    
    try:
        quality_metrics = quality_assessor.assess_quality(
            regime_labels=result.cluster_labels,
            feature_data=feature_df,
            forward_returns=None,
            timestamps=None,
            min_regime_size=10
        )
        
        print(f"\n✅ Quality Assessment Complete:")
        print(f"   Silhouette Score: {quality_metrics.silhouette_score:.4f}" if quality_metrics.silhouette_score else "   Silhouette Score: None")
        print(f"   Davies-Bouldin Index: {quality_metrics.davies_bouldin_score:.4f}" if quality_metrics.davies_bouldin_score else "   Davies-Bouldin Index: None")
        print(f"   Balance Score: {quality_metrics.balance_score:.4f}" if quality_metrics.balance_score else "   Balance Score: None")
        print(f"   Overall Quality: {quality_metrics.quality_score:.4f}" if quality_metrics.quality_score else "   Overall Quality: None")
        
    except Exception as e:
        print(f"\n⚠️ Quality assessment failed: {e}")
        quality_metrics = None
    
    # === STEP 6: Generate report ===
    print("\n" + "=" * 80)
    print("📝 STEP 6: Generating Report")
    print("=" * 80)
    
    markdown_content = []
    markdown_content.append("# Improved MS-DR Clustering Report (Real Data)\n")
    markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    markdown_content.append("## Data Source\n\n")
    markdown_content.append(f"- **Symbol:** {symbol}\n")
    markdown_content.append(f"- **Exchange:** {exchange}\n")
    markdown_content.append(f"- **Timeframe:** {timeframe}\n")
    markdown_content.append(f"- **Samples:** {len(df)}\n")
    markdown_content.append(f"- **Date Range:** {df.index.min()} to {df.index.max()}\n\n")
    
    markdown_content.append("## Improvements Applied\n\n")
    markdown_content.append("1. **Enhanced Signal Construction**\n")
    markdown_content.append(f"   - Multi-scale indicators: {signal_diagnostics['n_components']} components\n")
    markdown_content.append(f"   - Adaptive weighting: {len(signal_diagnostics['weights'])} weighted features\n")
    markdown_content.append(f"   - Signal diversity score: {quality['diversity_score']:.3f}\n\n")
    
    markdown_content.append("2. **Enhanced Burn-in Detection**\n")
    markdown_content.append(f"   - Burn-in detected: {burn_in_diag['burn_in_detected']}\n")
    if burn_in_diag['burn_in_detected']:
        markdown_content.append(f"   - Burn-in samples removed: {burn_in_diag['burn_in_samples']}\n")
        markdown_content.append(f"   - Burn-in regime: {burn_in_diag['burn_in_regime']}\n")
    markdown_content.append("\n")
    
    markdown_content.append("3. **MS-DR Configuration**\n")
    markdown_content.append(f"   - Model: AR({config.order}) with {config.method} optimization\n")
    markdown_content.append(f"   - Regime selection: {config.ic_criterion.upper()} criterion ({config.min_regimes}-{config.max_regimes} regimes)\n")
    markdown_content.append(f"   - Max iterations: {config.max_iter}\n\n")
    
    markdown_content.append("---\n\n")
    
    markdown_content.append("## 🎯 Clustering Results\n\n")
    markdown_content.append(f"- **n_clusters:** {result.n_clusters}\n")
    markdown_content.append(f"- **success:** {result.success}\n")
    markdown_content.append(f"- **processing_time:** {result.processing_time:.2f}s\n\n")
    
    markdown_content.append("### Regime Distribution\n\n")
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    markdown_content.append("| Regime ID | Samples | Percentage |\n")
    markdown_content.append("|-----------|---------|------------|\n")
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        markdown_content.append(f"| {regime_id} | {count} | {percentage:.1f}% |\n")
    markdown_content.append("\n")
    
    if quality_metrics:
        markdown_content.append("## 🎨 Quality Metrics\n\n")
        markdown_content.append(f"- **Silhouette Score:** {quality_metrics.silhouette_score:.4f}\n" if quality_metrics.silhouette_score else "- **Silhouette Score:** None\n")
        markdown_content.append(f"- **Davies-Bouldin Index:** {quality_metrics.davies_bouldin_score:.4f}\n" if quality_metrics.davies_bouldin_score else "- **Davies-Bouldin Index:** None\n")
        markdown_content.append(f"- **Balance Score:** {quality_metrics.balance_score:.4f}\n" if quality_metrics.balance_score else "- **Balance Score:** None\n")
        markdown_content.append(f"- **Overall Quality:** {quality_metrics.quality_score:.4f}\n\n" if quality_metrics.quality_score else "- **Overall Quality:** None\n\n")
    
    markdown_content.append("## 🔍 Diagnostics\n\n")
    markdown_content.append("### Signal Quality\n\n")
    for key, value in quality.items():
        markdown_content.append(f"- **{key}:** {value:.4f}\n")
    markdown_content.append("\n")
    
    markdown_content.append("### Burn-in Detection\n\n")
    for key, value in burn_in_diag.items():
        if key != 'recommendation':
            markdown_content.append(f"- **{key}:** {value}\n")
    if 'recommendation' in burn_in_diag:
        markdown_content.append(f"\n**Recommendation:** {burn_in_diag['recommendation']}\n")
    markdown_content.append("\n")
    
    markdown_content.append("---\n")
    markdown_content.append(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Write report
    with open(report_filename, 'w') as f:
        f.writelines(markdown_content)
    
    print(f"\n✅ Report saved to: {report_filename}")
    
    print("\n" + "=" * 80)
    print("✅ IMPROVED MS-DR TEST WITH REAL DATA COMPLETE")
    print("=" * 80)
    
    return result, quality_metrics, signal_diagnostics, burn_in_diag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-DR Clustering Test with Real Data')
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (e.g., 1h, 15m, 5m)')
    parser.add_argument('--limit', type=int, default=2000, help='Maximum number of records')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse dates if provided
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
    
    run_improved_ms_dr_test_real_data(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        limit=args.limit,
        start_date=start_date,
        end_date=end_date
    )

