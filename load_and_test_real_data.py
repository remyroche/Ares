#!/usr/bin/env python3
"""
Load Real ETHUSDT Data and Run Improved MS-DR Test

This script:
1. Loads real 1m klines data from data_cache
2. Resamples to desired timeframe (1h, 15m, etc.)
3. Runs improved MS-DR clustering
4. Generates comprehensive report

Usage:
    python3 load_and_test_real_data.py --timeframe 1h --limit 2000
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor
from improved_ms_dr_signal import create_improved_regime_signal
from improved_ms_dr_test import enhanced_burn_in_removal


def load_and_resample_real_data(symbol: str = "ETHUSDT",
                                 exchange: str = "binance",
                                 target_timeframe: str = "1h",
                                 limit: int = 2000) -> pd.DataFrame:
    """
    Load real 1m data and resample to target timeframe.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        target_timeframe: Target timeframe for resampling
        limit: Maximum number of candles after resampling
        
    Returns:
        Resampled DataFrame
    """
    print("\n" + "=" * 80)
    print("📊 LOADING REAL MARKET DATA")
    print("=" * 80)
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Source: 1m klines")
    print(f"   Target Timeframe: {target_timeframe}")
    print(f"   Limit: {limit} candles")
    
    # Path to 1m data
    data_path = f"data_cache/{exchange}/{symbol.lower()}/klines_{exchange}_{symbol}_1m.parquet"
    
    print(f"\n🔍 Looking for data: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ File not found: {data_path}")
        return None
    
    # Load 1m data
    print(f"📥 Loading 1m data...")
    df_1m = pd.read_parquet(data_path)
    print(f"✅ Loaded {len(df_1m)} 1m candles")
    print(f"   Date range: {df_1m.index.min()} to {df_1m.index.max()}")
    print(f"   Columns: {list(df_1m.columns)}")
    
    # Map timeframe to pandas frequency
    timeframe_map = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '60m': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    freq = timeframe_map.get(target_timeframe, '1H')
    
    # Resample to target timeframe
    print(f"\n🔄 Resampling 1m → {target_timeframe}...")
    df_resampled = df_1m.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Take last N candles
    if len(df_resampled) > limit:
        df_resampled = df_resampled.tail(limit)
    
    print(f"✅ Resampled to {target_timeframe}: {len(df_resampled)} candles")
    print(f"   Date range: {df_resampled.index.min()} to {df_resampled.index.max()}")
    
    return df_resampled


def run_test_with_real_data(timeframe: str = "1h", limit: int = 2000):
    """Run improved MS-DR test with real data."""
    
    print("=" * 80)
    print("🎯 IMPROVED MS-DR CLUSTERING WITH REAL ETHUSDT DATA")
    print("=" * 80)
    
    # Create outcomes directory
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"improved_ms_dr_REAL_ETHUSDT_{timeframe}_{timestamp}.md"
    
    print(f"📝 Report will be saved to: {report_filename}")
    
    # === STEP 1: Load and resample real data ===
    df = load_and_resample_real_data("ETHUSDT", "binance", timeframe, limit)
    
    if df is None or df.empty:
        print("\n❌ Failed to load real data!")
        return None
    
    # === STEP 2: Create improved regime signal ===
    print("\n" + "=" * 80)
    print("🔧 STEP 2: Creating Improved Regime Signal (42 components)")
    print("=" * 80)
    
    try:
        regime_signal, signal_diagnostics = create_improved_regime_signal(
            df,
            use_nonlinear=True,
            use_multiscale=True,
            use_adaptive_weights=True
        )
        
        # Signal quality check
        print("\n📊 Signal Quality Check:")
        quality = signal_diagnostics['signal_quality']
        
        print(f"   ✅ Diversity score: {quality['diversity_score']:.3f}")
        print(f"   ✅ Signal range: {quality['range']:.3f}")
        print(f"   ✅ Autocorrelation (lag 1): {quality['autocorr_lag1']:.3f}")
        print(f"   ✅ Transition rate: {quality['transition_rate']:.3f}")
        
        warnings = []
        if quality['std'] < 0.5:
            warnings.append(f"Low variance (std={quality['std']:.3f})")
        if quality['range'] < 3.0:
            warnings.append(f"Narrow range ({quality['range']:.3f})")
        if quality['transition_rate'] < 0.1:
            warnings.append(f"Low transition rate ({quality['transition_rate']:.3f})")
        
        if warnings:
            print("\n⚠️ Signal quality warnings:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("\n✅ Signal quality is EXCELLENT!")
        
    except Exception as e:
        print(f"\n❌ Failed to create regime signal: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Prepare data for MS-DR
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✅ Prepared data for MS-DR: {data.shape}")
    
    # === STEP 3: Run MS-DR clustering ===
    print("\n" + "=" * 80)
    print("🚀 STEP 3: Running MS-DR Clustering (Powell optimizer, AR(2), BIC)")
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
        
        print(f"\n✅ MS-DR clustering completed!")
        print(f"   Success: {result.success}")
        print(f"   N regimes discovered: {result.n_clusters}")
        print(f"   Processing time: {result.processing_time:.1f}s")
        
    except Exception as e:
        print(f"\n❌ MS-DR clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # === STEP 4: Enhanced burn-in detection ===
    print("\n" + "=" * 80)
    print("🔧 STEP 4: Enhanced Burn-in Detection")
    print("=" * 80)
    
    labels_clean, probs_clean, data_clean, was_cleaned, burn_in_diag = enhanced_burn_in_removal(
        result, data, aggressive=False
    )
    
    # Update result if cleaned
    if was_cleaned:
        print(f"\n🔧 Applied burn-in removal:")
        print(f"   Removed {burn_in_diag['burn_in_samples']} samples")
        result.cluster_labels = labels_clean
        result.cluster_probabilities = probs_clean
        result.n_clusters = len(np.unique(labels_clean))
        data = data_clean
    else:
        print("\n✅ No burn-in removal needed")
    
    # === STEP 5: Quality assessment ===
    print("\n" + "=" * 80)
    print("📊 STEP 5: Quality Assessment")
    print("=" * 80)
    
    # Fix the length mismatch issue
    feature_df = pd.DataFrame(data[:len(result.cluster_labels)], columns=['composite_regime_signal'])
    
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
        
        print(f"\n✅ Quality Assessment:")
        if quality_metrics.silhouette_score:
            print(f"   Silhouette Score: {quality_metrics.silhouette_score:.4f}")
        if quality_metrics.davies_bouldin_score:
            print(f"   Davies-Bouldin Index: {quality_metrics.davies_bouldin_score:.4f}")
        print(f"   Balance Score: {quality_metrics.balance_score:.4f}" if quality_metrics.balance_score else "   Balance Score: None")
        print(f"   Overall Quality: {quality_metrics.quality_score:.4f}" if quality_metrics.quality_score else "   Overall Quality: None")
        
    except Exception as e:
        print(f"\n⚠️ Quality assessment failed: {e}")
        quality_metrics = None
    
    # === STEP 6: Generate report ===
    print("\n" + "=" * 80)
    print("📝 STEP 6: Generating Report")
    print("=" * 80)
    
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    
    markdown_content = []
    markdown_content.append("# Improved MS-DR Clustering - REAL ETHUSDT Data\n")
    markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    markdown_content.append("## 🔥 REAL DATA SOURCE\n\n")
    markdown_content.append(f"- **Symbol:** ETHUSDT\n")
    markdown_content.append(f"- **Exchange:** Binance\n")
    markdown_content.append(f"- **Source Data:** 1m klines (resampled to {timeframe})\n")
    markdown_content.append(f"- **Samples:** {len(df)}\n")
    markdown_content.append(f"- **Date Range:** {df.index.min()} to {df.index.max()}\n")
    markdown_content.append(f"- **Data File:** `historical_data/binance/ethusdt/processed/klines_binance_ETHUSDT_1m.parquet`\n\n")
    
    markdown_content.append("## ✅ Improvements Applied\n\n")
    markdown_content.append("1. **Enhanced Signal Construction**\n")
    markdown_content.append(f"   - Multi-scale indicators: {signal_diagnostics['n_components']} components\n")
    markdown_content.append(f"   - Adaptive weighting: {len(signal_diagnostics['weights'])} features\n")
    markdown_content.append(f"   - Signal diversity score: {quality['diversity_score']:.3f}\n")
    markdown_content.append(f"   - Signal range: {quality['range']:.3f}\n\n")
    
    markdown_content.append("2. **Enhanced Burn-in Detection**\n")
    markdown_content.append(f"   - Burn-in detected: {burn_in_diag['burn_in_detected']}\n")
    markdown_content.append(f"   - Degenerate clustering: {burn_in_diag['is_degenerate']}\n\n")
    
    markdown_content.append("3. **MS-DR Configuration**\n")
    markdown_content.append(f"   - Model: AR({config.order}) with {config.method} optimizer\n")
    markdown_content.append(f"   - Regime selection: {config.ic_criterion.upper()} ({config.min_regimes}-{config.max_regimes} regimes)\n")
    markdown_content.append(f"   - Max iterations: {config.max_iter}\n")
    markdown_content.append(f"   - Processing time: {result.processing_time:.1f}s\n\n")
    
    markdown_content.append("---\n\n")
    
    markdown_content.append("## 🎯 Clustering Results\n\n")
    markdown_content.append(f"- **Regimes Discovered:** {result.n_clusters}\n")
    markdown_content.append(f"- **Success:** {result.success}\n")
    markdown_content.append(f"- **AIC:** {result.aic:.2f}\n" if result.aic else "- **AIC:** N/A\n")
    markdown_content.append(f"- **BIC:** {result.bic:.2f}\n" if result.bic else "- **BIC:** N/A\n")
    markdown_content.append(f"- **Log Likelihood:** {result.log_likelihood:.2f}\n\n" if result.log_likelihood else "- **Log Likelihood:** N/A\n\n")
    
    markdown_content.append("### Regime Distribution\n\n")
    markdown_content.append("| Regime | Samples | Percentage | Status |\n")
    markdown_content.append("|--------|---------|------------|--------|\n")
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        status = "✅" if 10 <= percentage <= 60 else "⚠️"
        markdown_content.append(f"| {regime_id} | {count} | {percentage:.1f}% | {status} |\n")
    markdown_content.append("\n")
    
    # Check for degenerate clustering
    if len(unique) == 1:
        markdown_content.append("**⚠️ WARNING:** Degenerate clustering detected! All samples in one regime.\n\n")
    elif counts.max() / len(result.cluster_labels) > 0.95:
        markdown_content.append("**⚠️ WARNING:** Highly imbalanced clustering (>95% in one regime).\n\n")
    
    if quality_metrics:
        markdown_content.append("## 🎨 Quality Metrics\n\n")
        if quality_metrics.silhouette_score:
            status = "✅" if quality_metrics.silhouette_score > 0.5 else "⚠️"
            markdown_content.append(f"- **Silhouette Score:** {quality_metrics.silhouette_score:.4f} {status}\n")
        if quality_metrics.davies_bouldin_score:
            status = "✅" if quality_metrics.davies_bouldin_score < 1.0 else "⚠️"
            markdown_content.append(f"- **Davies-Bouldin Index:** {quality_metrics.davies_bouldin_score:.4f} {status}\n")
        if quality_metrics.balance_score:
            status = "✅" if quality_metrics.balance_score > 0.5 else "⚠️"
            markdown_content.append(f"- **Balance Score:** {quality_metrics.balance_score:.4f} {status}\n")
        if quality_metrics.quality_score:
            status = "✅" if quality_metrics.quality_score > 0.7 else "⚠️"
            markdown_content.append(f"- **Overall Quality:** {quality_metrics.quality_score:.4f} {status}\n\n")
    
    markdown_content.append("## 🔍 Signal Diagnostics\n\n")
    markdown_content.append("### Signal Quality Metrics\n\n")
    for key, value in quality.items():
        markdown_content.append(f"- **{key}:** {value:.4f}\n")
    markdown_content.append("\n")
    
    markdown_content.append("### Component Diversity\n\n")
    div = signal_diagnostics['diversity_metrics']
    markdown_content.append(f"- **Max Correlation:** {div['max_correlation']:.3f}\n")
    markdown_content.append(f"- **Mean Correlation:** {div['mean_correlation']:.3f}\n")
    markdown_content.append(f"- **Diversity Score:** {div['diversity_score']:.3f}\n\n")
    
    markdown_content.append("## ⚙️ Regime Parameters\n\n")
    if result.regime_params:
        for regime_id, params in result.regime_params.items():
            markdown_content.append(f"### {regime_id}\n\n")
            if isinstance(params, dict):
                for key, val in params.items():
                    if isinstance(val, list):
                        markdown_content.append(f"- **{key}:** {val}\n")
                    else:
                        markdown_content.append(f"- **{key}:** {val:.4f}\n")
            markdown_content.append("\n")
    
    if result.regime_variances is not None:
        markdown_content.append("### Regime Variances\n\n")
        for i, var in enumerate(result.regime_variances):
            markdown_content.append(f"- Regime {i}: {var:.4f}\n")
        markdown_content.append("\n")
    
    markdown_content.append("## 🔄 Transition Matrix\n\n")
    if result.transition_matrix is not None:
        markdown_content.append("| From/To | " + " | ".join([f"Regime {i}" for i in range(result.n_clusters)]) + " |\n")
        markdown_content.append("|---------|" + "|".join(["---------" for _ in range(result.n_clusters + 1)]) + "|\n")
        
        for i, row in enumerate(result.transition_matrix):
            markdown_content.append(f"| Regime {i} | " + " | ".join([f"{val:.4f}" for val in row]) + " |\n")
        markdown_content.append("\n")
        markdown_content.append(f"**Transition Persistence:** {result.transition_persistence:.4f}\n\n")
    
    markdown_content.append("---\n")
    markdown_content.append(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Write report
    with open(report_filename, 'w') as f:
        f.writelines(markdown_content)
    
    print(f"\n✅ Report saved to: {report_filename}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE WITH REAL DATA!")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Data source: Real ETHUSDT {timeframe} (resampled from 1m)")
    print(f"   Samples: {len(df)}")
    print(f"   Regimes discovered: {result.n_clusters}")
    print(f"   Quality score: {quality_metrics.quality_score:.4f}" if quality_metrics and quality_metrics.quality_score else "   Quality score: N/A")
    print(f"   Degenerate: {burn_in_diag['is_degenerate']}")
    print(f"   Burn-in detected: {burn_in_diag['burn_in_detected']}")
    
    # Check regime distribution
    print(f"\n📊 Regime Distribution:")
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        status = "✅" if 10 <= percentage <= 60 else "⚠️"
        print(f"   {status} Regime {regime_id}: {count} samples ({percentage:.1f}%)")
    
    return result, quality_metrics, signal_diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-DR with Real ETHUSDT Data')
    parser.add_argument('--timeframe', type=str, default='1h', help='Target timeframe (1h, 15m, 5m, etc.)')
    parser.add_argument('--limit', type=int, default=2000, help='Number of candles after resampling')
    
    args = parser.parse_args()
    
    run_test_with_real_data(timeframe=args.timeframe, limit=args.limit)

