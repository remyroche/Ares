#!/usr/bin/env python3
"""
MS-DR Clustering with Real Processed ETHUSDT Data

Loads data from: historical_data/binance/ethusdt/processed/ethusdt_1h/

Usage:
    python3 ms_dr_with_real_processed_data.py --timeframe 1h --limit 2000
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor
from improved_ms_dr_signal import create_improved_regime_signal
from improved_ms_dr_test import enhanced_burn_in_removal


def load_processed_data(symbol: str = "ETHUSDT",
                       exchange: str = "binance", 
                       timeframe: str = "1h",
                       limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Load processed data from historical_data directory.
    
    Looks in: historical_data/{exchange}/{symbol}/processed/{symbol}_{timeframe}/
    
    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '1h', '15m', '5m')
        limit: Maximum number of records (None = all)
        
    Returns:
        DataFrame with OHLCV data or None
    """
    print("\n" + "=" * 80)
    print("📊 LOADING REAL PROCESSED DATA")
    print("=" * 80)
    
    # Build path to processed data
    symbol_lower = symbol.lower()
    processed_dir = Path(f"historical_data/{exchange}/{symbol_lower}/processed/{symbol_lower}_{timeframe}")
    
    print(f"   Symbol: {symbol}")
    print(f"   Exchange: {exchange}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Directory: {processed_dir}")
    
    if not processed_dir.exists():
        print(f"\n❌ Directory not found: {processed_dir}")
        print(f"\n📁 Available directories:")
        parent_dir = processed_dir.parent
        if parent_dir.exists():
            for d in parent_dir.iterdir():
                if d.is_dir():
                    print(f"   - {d.name}")
        return None
    
    # Find parquet files in directory (including partitioned subdirectories)
    parquet_files = list(processed_dir.glob("*.parquet"))
    
    # If no direct files, look in partitioned subdirectories (year=XXXX/month=XX/)
    if not parquet_files:
        parquet_files = list(processed_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        print(f"\n❌ No parquet files found in {processed_dir}")
        print(f"   Checked: direct files and partitioned subdirectories")
        return None
    
    print(f"\n✅ Found {len(parquet_files)} parquet files")
    for f in parquet_files[:5]:
        print(f"   - {f.name}")
    if len(parquet_files) > 5:
        print(f"   ... and {len(parquet_files) - 5} more files")
    
    # Load and combine all files
    print(f"\n📥 Loading parquet files...")
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Failed to load {file.name}: {e}")
    
    if not dfs:
        print(f"❌ No files loaded successfully")
        return None
    
    # Combine all dataframes
    df_combined = pd.concat(dfs, axis=0)
    
    # Sort by index (timestamp)
    if isinstance(df_combined.index, pd.DatetimeIndex):
        df_combined = df_combined.sort_index()
    
    # Remove duplicates
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    print(f"\n✅ Combined data loaded:")
    print(f"   Total samples: {len(df_combined)}")
    print(f"   Date range: {df_combined.index.min()} to {df_combined.index.max()}")
    print(f"   Columns: {list(df_combined.columns)}")
    
    # Apply limit if specified
    if limit and len(df_combined) > limit:
        df_combined = df_combined.tail(limit)
        print(f"   Limited to last {limit} samples")
    
    # Verify required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(col in df_combined.columns for col in required_cols):
        print(f"\n✅ All required OHLCV columns present")
        return df_combined
    else:
        missing = [col for col in required_cols if col not in df_combined.columns]
        print(f"\n❌ Missing columns: {missing}")
        return None


def run_ms_dr_with_real_processed_data(timeframe: str = "1h", limit: int = 2000):
    """Run improved MS-DR with real processed data."""
    
    print("=" * 80)
    print("🎯 IMPROVED MS-DR CLUSTERING - REAL PROCESSED ETHUSDT DATA")
    print("=" * 80)
    
    # Create outcomes directory
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"ms_dr_REAL_PROCESSED_ETHUSDT_{timeframe}_{timestamp}.md"
    
    print(f"📝 Report will be saved to: {report_filename}")
    
    # === STEP 1: Load real processed data ===
    df = load_processed_data("ETHUSDT", "binance", timeframe, limit)
    
    if df is None or df.empty:
        print("\n❌ Failed to load processed data!")
        return None
    
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
        
        # Signal quality metrics
        quality = signal_diagnostics['signal_quality']
        
        print(f"\n📊 Signal Quality:")
        print(f"   ✅ Components: {signal_diagnostics['n_components']}")
        print(f"   ✅ Diversity score: {quality['diversity_score']:.3f}")
        print(f"   ✅ Signal range: {quality['range']:.3f}")
        print(f"   ✅ Autocorrelation: {quality['autocorr_lag1']:.3f}")
        print(f"   ✅ Transition rate: {quality['transition_rate']:.3f}")
        
        # Warnings
        warnings = []
        if quality['diversity_score'] < 0.3:
            warnings.append(f"⚠️ Low diversity: {quality['diversity_score']:.3f}")
        if quality['range'] < 3.0:
            warnings.append(f"⚠️ Narrow range: {quality['range']:.3f}")
        
        if warnings:
            print(f"\n⚠️ Signal Quality Warnings:")
            for w in warnings:
                print(f"   {w}")
        else:
            print(f"\n✅ Signal quality is EXCELLENT!")
        
    except Exception as e:
        print(f"\n❌ Failed to create regime signal: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Prepare data
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
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
        
        print(f"\n✅ MS-DR Completed!")
        print(f"   Regimes discovered: {result.n_clusters}")
        print(f"   Processing time: {result.processing_time:.1f}s")
        
    except Exception as e:
        print(f"\n❌ MS-DR failed: {e}")
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
    
    if was_cleaned:
        result.cluster_labels = labels_clean
        result.cluster_probabilities = probs_clean
        result.n_clusters = len(np.unique(labels_clean))
        data = data_clean
        print(f"   Removed {burn_in_diag.get('burn_in_samples', 0)} burn-in samples")
    
    # === STEP 5: Quality assessment ===
    print("\n" + "=" * 80)
    print("📊 STEP 5: Quality Assessment")
    print("=" * 80)
    
    # Fix length mismatch (MS-DR returns n-2 labels for AR(2) model)
    feature_df = pd.DataFrame(data[:len(result.cluster_labels)], 
                             columns=['composite_regime_signal'])
    
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
        if quality_metrics.balance_score:
            print(f"   Balance Score: {quality_metrics.balance_score:.4f}")
        if quality_metrics.quality_score:
            print(f"   Overall Quality: {quality_metrics.quality_score:.4f}")
        
    except Exception as e:
        print(f"\n⚠️ Quality assessment partial failure (length mismatch): {e}")
        quality_metrics = None
    
    # === STEP 6: Generate report ===
    print("\n" + "=" * 80)
    print("📝 STEP 6: Generating Report")
    print("=" * 80)
    
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    
    markdown_content = []
    markdown_content.append("# MS-DR Clustering - REAL PROCESSED ETHUSDT DATA ✅\n")
    markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    markdown_content.append("## 🔥 REAL DATA SOURCE\n\n")
    markdown_content.append(f"- **Symbol:** ETHUSDT\n")
    markdown_content.append(f"- **Exchange:** Binance\n")
    markdown_content.append(f"- **Timeframe:** {timeframe}\n")
    markdown_content.append(f"- **Source:** Processed data from `historical_data/binance/ethusdt/processed/ethusdt_{timeframe}/`\n")
    markdown_content.append(f"- **Samples:** {len(df)}\n")
    markdown_content.append(f"- **Date Range:** {df.index.min()} to {df.index.max()}\n\n")
    
    markdown_content.append("## ✅ Improvements Applied\n\n")
    markdown_content.append("### 1. Enhanced Signal Construction\n")
    markdown_content.append(f"   - **Components:** {signal_diagnostics['n_components']} (vs 4 original)\n")
    markdown_content.append(f"   - **Signal Diversity:** {quality['diversity_score']:.3f}\n")
    markdown_content.append(f"   - **Signal Range:** {quality['range']:.3f}\n")
    markdown_content.append(f"   - **Autocorrelation (lag 1):** {quality['autocorr_lag1']:.3f}\n\n")
    
    markdown_content.append("### 2. Enhanced Burn-in Detection\n")
    markdown_content.append(f"   - **Degenerate clustering:** {burn_in_diag['is_degenerate']}\n")
    markdown_content.append(f"   - **Burn-in detected:** {burn_in_diag['burn_in_detected']}\n")
    if burn_in_diag['burn_in_detected']:
        markdown_content.append(f"   - **Burn-in samples removed:** {burn_in_diag.get('burn_in_samples', 0)}\n")
    markdown_content.append("\n")
    
    markdown_content.append("### 3. MS-DR Configuration\n")
    markdown_content.append(f"   - **Model:** AR({config.order}) autoregression\n")
    markdown_content.append(f"   - **Optimizer:** {config.method}\n")
    markdown_content.append(f"   - **Regime Selection:** {config.ic_criterion.upper()} criterion ({config.min_regimes}-{config.max_regimes} regimes)\n")
    markdown_content.append(f"   - **Max Iterations:** {config.max_iter}\n")
    markdown_content.append(f"   - **Processing Time:** {result.processing_time:.1f}s\n\n")
    
    markdown_content.append("---\n\n")
    
    markdown_content.append("## 🎯 Clustering Results\n\n")
    markdown_content.append(f"- **Regimes Discovered:** {result.n_clusters}\n")
    markdown_content.append(f"- **Success:** {result.success}\n")
    
    if result.aic:
        markdown_content.append(f"- **AIC:** {result.aic:.2f}\n")
    if result.bic:
        markdown_content.append(f"- **BIC:** {result.bic:.2f}\n")
    if result.log_likelihood:
        markdown_content.append(f"- **Log Likelihood:** {result.log_likelihood:.2f}\n")
    markdown_content.append("\n")
    
    # Check for degenerate clustering
    is_degenerate = len(unique) == 1 or (counts.max() / len(result.cluster_labels) > 0.95)
    
    if is_degenerate:
        markdown_content.append("### ⚠️ WARNING: Degenerate Clustering Detected!\n\n")
        markdown_content.append("**Problem:** All or most samples assigned to one regime.\n\n")
        markdown_content.append("**Possible causes:**\n")
        markdown_content.append("- Signal too uniform (check signal diversity)\n")
        markdown_content.append("- Market conditions too stable during this period\n")
        markdown_content.append("- Model initialization issue\n\n")
    
    markdown_content.append("### Regime Distribution\n\n")
    markdown_content.append("| Regime | Samples | Percentage | Balance Status |\n")
    markdown_content.append("|--------|---------|------------|----------------|\n")
    
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        
        if percentage < 5:
            status = "❌ Too Small"
        elif percentage > 80:
            status = "❌ Too Large"
        elif percentage < 10 or percentage > 60:
            status = "⚠️ Imbalanced"
        else:
            status = "✅ Good"
        
        markdown_content.append(f"| {regime_id} | {count} | {percentage:.1f}% | {status} |\n")
    markdown_content.append("\n")
    
    if quality_metrics:
        markdown_content.append("## 🎨 Quality Metrics\n\n")
        if quality_metrics.balance_score is not None:
            status = "✅" if quality_metrics.balance_score > 0.5 else "⚠️"
            markdown_content.append(f"- **Balance Score:** {quality_metrics.balance_score:.4f} {status}\n")
        if quality_metrics.quality_score is not None:
            status = "✅" if quality_metrics.quality_score > 0.7 else "⚠️"
            markdown_content.append(f"- **Overall Quality:** {quality_metrics.quality_score:.4f} {status}\n")
        markdown_content.append("\n")
    
    markdown_content.append("## 🔍 Signal Diagnostics\n\n")
    for key, value in quality.items():
        markdown_content.append(f"- **{key}:** {value:.4f}\n")
    markdown_content.append("\n")
    
    markdown_content.append("## ⚙️ Regime Parameters\n\n")
    if result.regime_params:
        for regime_id, params in result.regime_params.items():
            markdown_content.append(f"### {regime_id}\n")
            if isinstance(params, dict):
                for key, val in params.items():
                    if isinstance(val, list):
                        val_str = ', '.join([f"{v:.4f}" for v in val])
                        markdown_content.append(f"- **{key}:** [{val_str}]\n")
                    else:
                        markdown_content.append(f"- **{key}:** {val:.4f}\n")
            markdown_content.append("\n")
    
    if result.regime_variances is not None:
        markdown_content.append("**Regime Variances:**\n")
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
    
    # === FINAL SUMMARY ===
    print("\n" + "=" * 80)
    print("✅ MS-DR CLUSTERING COMPLETE WITH REAL DATA!")
    print("=" * 80)
    
    print(f"\n📊 Results Summary:")
    print(f"   📁 Data: Real ETHUSDT {timeframe} ({len(df)} candles)")
    print(f"   📅 Period: {df.index.min().date()} to {df.index.max().date()}")
    print(f"   🎯 Regimes: {result.n_clusters}")
    
    print(f"\n📊 Regime Distribution:")
    for regime_id, count in zip(unique, counts):
        percentage = (count / len(result.cluster_labels)) * 100
        
        if percentage < 10 or percentage > 60:
            status = "⚠️"
        else:
            status = "✅"
        
        print(f"   {status} Regime {regime_id}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\n📊 Quality Metrics:")
    if quality_metrics:
        if quality_metrics.balance_score:
            print(f"   Balance: {quality_metrics.balance_score:.4f}")
        if quality_metrics.quality_score:
            print(f"   Overall Quality: {quality_metrics.quality_score:.4f}")
    
    print(f"\n📊 Signal Quality:")
    print(f"   Diversity: {quality['diversity_score']:.3f}")
    print(f"   Range: {quality['range']:.3f}")
    
    print(f"\n🚨 Status Checks:")
    print(f"   {'❌' if is_degenerate else '✅'} Degenerate clustering: {burn_in_diag['is_degenerate']}")
    print(f"   {'⚠️' if burn_in_diag['burn_in_detected'] else '✅'} Burn-in detected: {burn_in_diag['burn_in_detected']}")
    
    print(f"\n📄 Full report: {report_filename}")
    
    return result, quality_metrics, signal_diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-DR with Real Processed Data')
    parser.add_argument('--timeframe', type=str, default='1h', 
                       help='Timeframe (1h, 15m, 5m, 30m, 1m)')
    parser.add_argument('--limit', type=int, default=2000, 
                       help='Number of candles to use')
    
    args = parser.parse_args()
    
    run_ms_dr_with_real_processed_data(
        timeframe=args.timeframe,
        limit=args.limit
    )

