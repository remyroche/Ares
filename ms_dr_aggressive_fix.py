#!/usr/bin/env python3
"""
MS-DR Aggressive Fix for Poor Regime Separation

Problem: Only 2 regimes detected with highly imbalanced distribution (16.9% / 83.1%)

Root Cause Analysis:
1. Signal components still too correlated despite improvements
2. MS-DR optimizer favoring simpler models (BIC penalty on complexity)
3. Need more aggressive non-linear transformations
4. Should use AIC instead of BIC (less penalty for more regimes)

Solutions:
1. Add regime change indicators (transition detection)
2. Use percentile-based transformations for better separation
3. Add market stress indicators
4. Force AIC criterion (favors more regimes)
5. Increase minimum regimes to 3
6. Add explicit regime initialization hints
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor


def create_aggressive_regime_signal(df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
    """
    Create aggressive regime signal with maximum separation.
    
    Key differences from improved version:
    - Percentile-based transformations (non-linear)
    - Regime change detection indicators
    - Market stress/crisis indicators
    - Reduced component correlation via PCA
    - Higher emphasis on volatility regimes
    """
    print("\n" + "=" * 80)
    print("🔨 CREATING AGGRESSIVE REGIME SIGNAL")
    print("=" * 80)
    
    regime_indicators = pd.DataFrame(index=df.index)
    returns = df['close'].pct_change()
    
    # === 1. VOLATILITY REGIMES (40% weight - most important) ===
    print("\n📊 Creating volatility regime indicators...")
    
    # Short-term realized volatility with percentile transform
    vol_20 = returns.rolling(20).std()
    vol_20_pct = vol_20.rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
    regime_indicators['vol_short_pct'] = vol_20_pct * 2 - 1  # Scale to [-1, 1]
    
    # Volatility of volatility (regime instability)
    vol_of_vol = vol_20.rolling(20).std()
    regime_indicators['vol_of_vol'] = (vol_of_vol - vol_of_vol.rolling(100).mean()) / (vol_of_vol.rolling(100).std() + 1e-8)
    
    # Volatility regime changes (transitions)
    vol_change = vol_20.pct_change(10)
    regime_indicators['vol_regime_change'] = np.sign(vol_change) * np.sqrt(np.abs(vol_change))
    
    # Crisis indicator (extreme volatility)
    vol_99th = vol_20.rolling(252).quantile(0.99)
    regime_indicators['crisis_indicator'] = (vol_20 > vol_99th).astype(float)
    
    # === 2. TREND REGIMES (30% weight) ===
    print("📊 Creating trend regime indicators...")
    
    # Multi-timeframe trend alignment
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    sma_100 = df['close'].rolling(100).mean()
    
    # Trend strength (all SMAs aligned)
    trend_alignment = (
        np.sign(df['close'] - sma_20) +
        np.sign(sma_20 - sma_50) +
        np.sign(sma_50 - sma_100)
    ) / 3.0  # -1 (strong bear) to +1 (strong bull)
    regime_indicators['trend_alignment'] = trend_alignment
    
    # Trend acceleration
    price_momentum = df['close'].diff(10)
    momentum_accel = price_momentum.diff(5)
    regime_indicators['trend_accel'] = momentum_accel / (df['close'].rolling(20).std() + 1e-8)
    
    # Trend exhaustion (price far from mean)
    price_zscore = (df['close'] - df['close'].rolling(100).mean()) / (df['close'].rolling(100).std() + 1e-8)
    regime_indicators['trend_exhaustion'] = np.tanh(price_zscore / 2)  # Squash extremes
    
    # === 3. VOLUME REGIMES (20% weight) ===
    print("📊 Creating volume regime indicators...")
    
    # Volume percentile (non-linear)
    volume_pct = df['volume'].rolling(252).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
    regime_indicators['volume_pct'] = volume_pct * 2 - 1
    
    # Volume momentum (increasing/decreasing interest)
    volume_ma_ratio = df['volume'].rolling(20).mean() / (df['volume'].rolling(100).mean() + 1e-8)
    regime_indicators['volume_momentum'] = np.log(volume_ma_ratio + 1e-8)
    
    # Volume-price divergence (distribution days)
    price_change = returns
    volume_change = df['volume'].pct_change()
    regime_indicators['volume_price_div'] = price_change.rolling(10).corr(volume_change) * -1  # Negative corr = divergence
    
    # === 4. MOMENTUM REGIMES (10% weight) ===
    print("📊 Creating momentum regime indicators...")
    
    # RSI percentile-based
    price_diff = df['close'].diff()
    gains = price_diff.where(price_diff > 0, 0).rolling(14).mean()
    losses = -price_diff.where(price_diff < 0, 0).rolling(14).mean()
    rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
    rsi_pct = rsi.rolling(100).apply(lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100)
    regime_indicators['rsi_pct'] = rsi_pct * 2 - 1
    
    # Momentum exhaustion (overbought/oversold extremes)
    rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(float)
    regime_indicators['momentum_extreme'] = rsi_extreme
    
    # Rate of change acceleration
    roc = df['close'].pct_change(10)
    roc_accel = roc.diff(5)
    regime_indicators['roc_accel'] = roc_accel / (returns.rolling(20).std() + 1e-8)
    
    # === 5. REGIME TRANSITION INDICATORS ===
    print("📊 Creating regime transition indicators...")
    
    # Volatility regime changes
    vol_regime_binary = (vol_20 > vol_20.rolling(100).median()).astype(int)
    vol_regime_transitions = vol_regime_binary.diff().abs()
    regime_indicators['vol_regime_trans'] = vol_regime_transitions.rolling(20).sum()
    
    # Price range expansion/contraction
    price_range = (df['high'] - df['low']) / df['close']
    range_expansion = price_range / (price_range.rolling(50).mean() + 1e-8) - 1
    regime_indicators['range_expansion'] = np.tanh(range_expansion)
    
    # Market stress indicator (multiple factors)
    stress_score = (
        (vol_20 / vol_20.rolling(252).median() - 1).clip(-2, 2) +  # Vol spike
        (-returns / returns.rolling(20).std()).clip(-2, 2) +  # Negative returns
        (price_range / price_range.rolling(100).mean() - 1).clip(-2, 2)  # Range expansion
    ) / 3.0
    regime_indicators['market_stress'] = stress_score
    
    # Fill NaN
    regime_indicators = regime_indicators.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Clip extreme values
    for col in regime_indicators.columns:
        regime_indicators[col] = regime_indicators[col].clip(-5, 5)
    
    print(f"\n✅ Created {len(regime_indicators.columns)} aggressive regime indicators")
    
    # === Calculate correlation and apply PCA for decorrelation ===
    print("\n🔍 Analyzing component correlation...")
    corr_matrix = regime_indicators.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    max_corr = corr_matrix.max().max()
    mean_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    print(f"   Max correlation: {max_corr:.3f}")
    print(f"   Mean correlation: {mean_corr:.3f}")
    
    # Apply PCA if correlation too high
    if mean_corr > 0.3 or max_corr > 0.8:
        print(f"\n⚠️ High correlation detected - applying PCA decorrelation...")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        indicators_scaled = scaler.fit_transform(regime_indicators)
        
        # Keep components explaining 95% variance
        pca = PCA(n_components=0.95, random_state=42)
        indicators_pca = pca.fit_transform(indicators_scaled)
        
        print(f"   Reduced {len(regime_indicators.columns)} → {indicators_pca.shape[1]} PCA components")
        print(f"   Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
        
        # Reweight PCA components by explained variance
        weights = pca.explained_variance_ratio_[:indicators_pca.shape[1]]
        regime_signal = (indicators_pca * weights).sum(axis=1)
        
    else:
        print(f"\n✅ Correlation acceptable - using adaptive weighting...")
        
        # Adaptive weights (inverse to correlation)
        avg_corr = corr_matrix.mean(axis=1)
        raw_weights = 1.0 / (1.0 + avg_corr)
        weights = raw_weights / raw_weights.sum()
        
        # Weighted composite
        regime_signal = (regime_indicators * weights.values).sum(axis=1)
    
    # Standardize final signal
    regime_signal = pd.Series(regime_signal, index=df.index)
    regime_signal = (regime_signal - regime_signal.mean()) / (regime_signal.std() + 1e-8)
    regime_signal = regime_signal.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Validate signal quality
    signal_range = regime_signal.max() - regime_signal.min()
    signal_std = regime_signal.std()
    autocorr = regime_signal.autocorr(lag=1)
    
    # Check for regime changes (transitions)
    signal_diff = regime_signal.diff().abs()
    transition_rate = (signal_diff > signal_diff.quantile(0.75)).sum() / len(regime_signal)
    
    diagnostics = {
        'n_components': len(regime_indicators.columns),
        'signal_range': float(signal_range),
        'signal_std': float(signal_std),
        'autocorr_lag1': float(autocorr),
        'transition_rate': float(transition_rate),
        'max_correlation': float(max_corr),
        'mean_correlation': float(mean_corr),
        'diversity_score': float(1.0 - mean_corr)
    }
    
    print(f"\n✅ Aggressive signal created:")
    print(f"   Range: {signal_range:.3f}")
    print(f"   Std: {signal_std:.3f}")
    print(f"   Autocorr: {autocorr:.3f}")
    print(f"   Transition rate: {transition_rate:.3f}")
    print(f"   Diversity: {diagnostics['diversity_score']:.3f}")
    
    return regime_signal, diagnostics


def run_aggressive_ms_dr_test(df: pd.DataFrame, 
                               force_min_regimes: int = 3,
                               use_aic: bool = True):
    """
    Run MS-DR with aggressive settings to force better regime separation.
    
    Args:
        df: Market data DataFrame
        force_min_regimes: Minimum number of regimes (default: 3)
        use_aic: Use AIC instead of BIC (favors more regimes)
    """
    print("\n" + "=" * 80)
    print("🚀 AGGRESSIVE MS-DR CLUSTERING")
    print("=" * 80)
    print(f"   Min regimes: {force_min_regimes}")
    print(f"   Max regimes: 8")
    print(f"   Criterion: {'AIC' if use_aic else 'BIC'} (AIC favors more regimes)")
    
    # Create aggressive signal
    regime_signal, diagnostics = create_aggressive_regime_signal(df)
    
    # Validate signal quality
    if diagnostics['diversity_score'] < 0.3:
        print(f"\n⚠️ WARNING: Low signal diversity ({diagnostics['diversity_score']:.3f})")
        print(f"   This may indicate:")
        print(f"   - Market conditions too uniform during this period")
        print(f"   - Need different indicators")
        print(f"   - Data quality issues")
    
    if diagnostics['signal_range'] < 4.0:
        print(f"\n⚠️ WARNING: Narrow signal range ({diagnostics['signal_range']:.3f})")
        print(f"   May result in poor regime separation")
    
    # Prepare data
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # === AGGRESSIVE MS-DR CONFIGURATION ===
    print("\n🔧 Configuring aggressive MS-DR...")
    
    config = MSDRConfig(
        # FORCE more regimes
        n_regimes=force_min_regimes,
        auto_select_regimes=True,
        min_regimes=force_min_regimes,  # FORCE minimum
        max_regimes=8,  # Try more regimes
        
        # Use AIC (less penalty for complexity than BIC)
        ic_criterion='aic' if use_aic else 'bic',
        
        # Model configuration
        model_type='autoregression',
        order=3,  # AR(3) for richer dynamics
        switching_variance=True,
        
        # Optimization
        method='powell',  # Robust optimizer
        max_iter=5000,  # More iterations
        
        # Other settings
        enable_pca=False,
        random_state=42,
        use_memory_optimization=True,
        use_hardware_acceleration=True,
        show_progress=True
    )
    
    print(f"   Model: AR({config.order}) with {config.method}")
    print(f"   Iterations: {config.max_iter}")
    print(f"   IC: {config.ic_criterion.upper()}")
    print(f"   Regime range: {config.min_regimes}-{config.max_regimes}")
    
    # Run MS-DR
    print("\n⏳ Running MS-DR (this may take 5-10 minutes)...")
    
    try:
        clusterer = MSDRClusterer(config)
        result = clusterer.fit_predict(data)
        
        print(f"\n✅ MS-DR Completed!")
        print(f"   Regimes discovered: {result.n_clusters}")
        print(f"   Processing time: {result.processing_time:.1f}s")
        
        # Check if still degenerate or imbalanced
        unique, counts = np.unique(result.cluster_labels, return_counts=True)
        
        print(f"\n📊 Regime Distribution:")
        max_pct = 0
        for regime_id, count in zip(unique, counts):
            pct = (count / len(result.cluster_labels)) * 100
            max_pct = max(max_pct, pct)
            status = "✅" if 15 <= pct <= 50 else "⚠️" if 10 <= pct <= 60 else "❌"
            print(f"   {status} Regime {regime_id}: {count:4d} ({pct:5.1f}%)")
        
        # Assessment
        is_degenerate = len(unique) == 1
        is_highly_imbalanced = max_pct > 70
        
        print(f"\n🔍 Quality Check:")
        print(f"   {'❌' if is_degenerate else '✅'} Degenerate: {is_degenerate}")
        print(f"   {'❌' if is_highly_imbalanced else '✅'} Highly imbalanced: {is_highly_imbalanced}")
        print(f"   {'❌' if result.n_clusters < force_min_regimes else '✅'} Meets min regimes: {result.n_clusters} >= {force_min_regimes}")
        
        if is_degenerate or is_highly_imbalanced or result.n_clusters < force_min_regimes:
            print(f"\n🚨 STILL PROBLEMATIC - Recommendations:")
            print(f"   1. Try even more aggressive signal (add more indicators)")
            print(f"   2. Use AIC instead of BIC (set use_aic=True)")
            print(f"   3. Increase min_regimes to {force_min_regimes + 1}")
            print(f"   4. Try alternative clustering (HDP-HMM, GMM)")
            print(f"   5. Check if data actually has distinct regimes (visual inspection)")
        else:
            print(f"\n✅ PROBLEM SOLVED!")
            print(f"   Balanced regime distribution achieved")
        
        return result, diagnostics
        
    except Exception as e:
        print(f"\n❌ MS-DR failed: {e}")
        import traceback
        traceback.print_exc()
        return None, diagnostics


def run_full_aggressive_test():
    """Run full test with real processed data and aggressive settings."""
    
    print("=" * 80)
    print("🎯 MS-DR AGGRESSIVE FIX - FORCE BETTER REGIME SEPARATION")
    print("=" * 80)
    
    # Load processed data
    processed_dir = Path("historical_data/binance/ethusdt/processed/ethusdt_1h")
    
    if not processed_dir.exists():
        print(f"\n❌ Processed data not found: {processed_dir}")
        print(f"   Generating synthetic data instead...")
        df = generate_synthetic_multiregime_data()
    else:
        print(f"\n📥 Loading processed data from: {processed_dir}")
        parquet_files = list(processed_dir.glob("**/*.parquet"))
        
        if not parquet_files:
            print(f"❌ No parquet files found")
            print(f"   Generating synthetic data instead...")
            df = generate_synthetic_multiregime_data()
        else:
            print(f"✅ Found {len(parquet_files)} files")
            dfs = []
            for file in parquet_files[:20]:  # Limit files to load
                try:
                    dfs.append(pd.read_parquet(file))
                except:
                    pass
            
            df = pd.concat(dfs, axis=0).sort_index()
            df = df[~df.index.duplicated(keep='first')]
            df = df.tail(2000)
            
            print(f"✅ Loaded {len(df)} candles")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    # Run aggressive test with multiple configurations
    print("\n" + "=" * 80)
    print("🧪 TESTING MULTIPLE CONFIGURATIONS")
    print("=" * 80)
    
    configs_to_test = [
        {'name': 'Aggressive AIC, min=3', 'force_min_regimes': 3, 'use_aic': True},
        {'name': 'Aggressive AIC, min=4', 'force_min_regimes': 4, 'use_aic': True},
        {'name': 'Aggressive BIC, min=3', 'force_min_regimes': 3, 'use_aic': False},
    ]
    
    best_result = None
    best_balance = 0
    best_config = None
    
    for i, config_params in enumerate(configs_to_test):
        print(f"\n{'='*80}")
        print(f"📊 Configuration {i+1}/{len(configs_to_test)}: {config_params['name']}")
        print(f"{'='*80}")
        
        result, diag = run_aggressive_ms_dr_test(
            df,
            force_min_regimes=config_params['force_min_regimes'],
            use_aic=config_params['use_aic']
        )
        
        if result:
            # Calculate balance score
            unique, counts = np.unique(result.cluster_labels, return_counts=True)
            max_pct = (counts.max() / len(result.cluster_labels)) * 100
            min_pct = (counts.min() / len(result.cluster_labels)) * 100
            
            # Balance: 1 - normalized difference from ideal equal distribution
            ideal_pct = 100 / len(unique)
            balance = 1.0 - abs(max_pct - min_pct) / 100.0
            
            print(f"\n   Balance score: {balance:.3f}")
            
            if balance > best_balance and len(unique) >= config_params['force_min_regimes']:
                best_result = result
                best_balance = balance
                best_config = config_params
                print(f"   ⭐ NEW BEST CONFIGURATION!")
    
    # Generate report for best result
    if best_result:
        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION: {best_config['name']}")
        print(f"{'='*80}")
        
        generate_aggressive_report(df, best_result, diag, best_config)
    
    return best_result


def generate_synthetic_multiregime_data(n_samples: int = 2000):
    """Generate synthetic data with 4 clear regimes."""
    print("\n🔨 Generating synthetic data with 4 distinct regimes...")
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='1h')
    base_price = 3000.0
    
    # 4 regimes with very distinct characteristics
    regime_lengths = [500, 500, 500, 500]
    regime_params = [
        {'volatility': 0.01, 'trend': 0.0015, 'volume': 2.0, 'name': 'Strong Bull'},
        {'volatility': 0.06, 'trend': -0.001, 'volume': 0.6, 'name': 'Bear/Crisis'},
        {'volatility': 0.015, 'trend': 0.0, 'volume': 1.0, 'name': 'Sideways'},
        {'volatility': 0.025, 'trend': 0.0008, 'volume': 1.5, 'name': 'Moderate Bull'}
    ]
    
    prices = [base_price]
    volumes = []
    regime_idx = 0
    regime_counter = 0
    
    for i in range(n_samples):
        if regime_counter >= regime_lengths[regime_idx]:
            regime_idx = (regime_idx + 1) % 4
            regime_counter = 0
            print(f"   Switching to {regime_params[regime_idx]['name']} at candle {i}")
        
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
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': volumes
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"✅ Generated {len(df)} candles with 4 regimes")
    return df


def generate_aggressive_report(df, result, diagnostics, config_params):
    """Generate comprehensive report for aggressive test."""
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = outcomes_dir / f"ms_dr_AGGRESSIVE_FIX_{timestamp}.md"
    
    unique, counts = np.unique(result.cluster_labels, return_counts=True)
    
    content = []
    content.append("# MS-DR Aggressive Fix Report\n")
    content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    content.append("## 🎯 Best Configuration\n\n")
    content.append(f"- **Name:** {config_params['name']}\n")
    content.append(f"- **Min Regimes:** {config_params['force_min_regimes']}\n")
    content.append(f"- **Criterion:** {'AIC' if config_params['use_aic'] else 'BIC'}\n\n")
    
    content.append("## 🎯 Results\n\n")
    content.append(f"- **Regimes Discovered:** {result.n_clusters}\n")
    content.append(f"- **Processing Time:** {result.processing_time:.1f}s\n\n")
    
    content.append("### Regime Distribution\n\n")
    content.append("| Regime | Samples | Percentage | Status |\n")
    content.append("|--------|---------|------------|--------|\n")
    
    for regime_id, count in zip(unique, counts):
        pct = (count / len(result.cluster_labels)) * 100
        if pct < 10:
            status = "❌ Too Small"
        elif pct > 60:
            status = "❌ Dominates"
        elif pct < 15 or pct > 50:
            status = "⚠️ Imbalanced"
        else:
            status = "✅ Good"
        
        content.append(f"| {regime_id} | {count} | {pct:.1f}% | {status} |\n")
    
    content.append("\n")
    
    # Assessment
    max_pct = (counts.max() / len(result.cluster_labels)) * 100
    
    if len(unique) == 1:
        content.append("### ❌ STILL DEGENERATE\n\n")
        content.append("All samples in one regime. Try alternative clustering method (HDP-HMM, GMM).\n\n")
    elif max_pct > 70:
        content.append("### ⚠️ STILL HIGHLY IMBALANCED\n\n")
        content.append(f"One regime dominates with {max_pct:.1f}%. Consider:\n")
        content.append("- Using AIC instead of BIC\n")
        content.append("- Increasing min_regimes\n")
        content.append("- Adding more discriminative indicators\n\n")
    else:
        content.append("### ✅ IMPROVED BALANCE\n\n")
        content.append(f"Max regime: {max_pct:.1f}%, Min regime: {(counts.min() / len(result.cluster_labels)) * 100:.1f}%\n\n")
    
    content.append("## 📊 Signal Quality\n\n")
    for key, val in diagnostics.items():
        content.append(f"- **{key}:** {val:.4f}\n")
    content.append("\n")
    
    content.append("---\n")
    content.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    with open(report_file, 'w') as f:
        f.writelines(content)
    
    print(f"\n📄 Report: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MS-DR Aggressive Fix')
    parser.add_argument('--min-regimes', type=int, default=3, help='Minimum regimes to force')
    parser.add_argument('--use-aic', action='store_true', help='Use AIC instead of BIC')
    
    args = parser.parse_args()
    
    # Generate synthetic data with 4 clear regimes for testing
    df = generate_synthetic_multiregime_data(n_samples=2000)
    
    # Run aggressive test
    run_aggressive_ms_dr_test(
        df,
        force_min_regimes=args.min_regimes,
        use_aic=args.use_aic
    )


