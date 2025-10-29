#!/usr/bin/env python3
"""
Get Actual MS-DR Clustering Metrics and Generate Markdown Report
"""

import sys
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRClusterer, MSDRConfig
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityAssessor

# Create outcomes directory if it doesn't exist
outcomes_dir = Path("outcomes")
outcomes_dir.mkdir(exist_ok=True)

# Generate filename with datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_filename = outcomes_dir / f"ms_dr_metrics_{timestamp}.md"

print("=" * 80)
print("🎯 MS-DR CLUSTERING - ACTUAL METRICS")
print("=" * 80)
print(f"📝 Report will be saved to: {report_filename}")

# Create sample market data with proper regime structure
print("\n📊 Creating market data with regime structure...")
np.random.seed(42)
n_samples = 1000

# Create synthetic OHLCV data with clear regimes
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
base_price = 3000.0

# Define 3 distinct market regimes
regime_lengths = [350, 300, 350]  # More balanced distribution
regime_params = [
    {'volatility': 0.02, 'trend': 0.001, 'volume': 1.5},   # Bull market - low vol, uptrend, high volume
    {'volatility': 0.05, 'trend': -0.0005, 'volume': 0.8}, # Bear market - high vol, downtrend, low volume
    {'volatility': 0.01, 'trend': 0.0, 'volume': 1.0}      # Sideways - low vol, no trend, normal volume
]

prices = [base_price]
volumes = []
current_regime = 0
regime_idx = 0
regime_counter = 0

for i in range(n_samples):
    if regime_counter >= regime_lengths[regime_idx]:
        regime_idx = (regime_idx + 1) % 3
        regime_counter = 0
    
    params = regime_params[regime_idx]
    
    # Generate price movement
    price_change = np.random.normal(params['trend'], params['volatility'])
    new_price = prices[-1] * (1 + price_change)
    prices.append(new_price)
    
    # Generate volume (regime-dependent)
    volume = np.random.uniform(500 * params['volume'], 2000 * params['volume'])
    volumes.append(volume)
    
    regime_counter += 1

# Create DataFrame
df = pd.DataFrame({
    'timestamp': dates,
    'open': prices[:-1],
    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices[:-1]],
    'close': prices[1:],
    'volume': volumes
})
df.set_index('timestamp', inplace=True)

print(f"✅ Created market data: {df.shape}")

# Create regime-specific composite indicators (Solution A)
print("\n🔧 Creating regime-specific composite indicators...")

regime_indicators = pd.DataFrame(index=df.index)

# 1. Volatility regime (Z-score of rolling volatility)
returns = df['close'].pct_change()
vol_20 = returns.rolling(20).std()
vol_mean = vol_20.rolling(252).mean()
vol_std = vol_20.rolling(252).std()
regime_indicators['vol_regime'] = (vol_20 - vol_mean) / (vol_std + 1e-8)
regime_indicators['vol_regime'].fillna(0, inplace=True)

# 2. Trend regime (normalized price vs SMA)
sma_50 = df['close'].rolling(50).mean()
regime_indicators['trend_regime'] = (df['close'] - sma_50) / (sma_50 + 1e-8)
regime_indicators['trend_regime'].fillna(0, inplace=True)

# 3. Volume regime (Z-score of volume)
volume_ma = df['volume'].rolling(252).mean()
volume_std = df['volume'].rolling(252).std()
regime_indicators['volume_regime'] = (df['volume'] - volume_ma) / (volume_std + 1e-8)
regime_indicators['volume_regime'].fillna(0, inplace=True)

# 4. Momentum regime (normalized RSI-style indicator)
momentum_period = 14
price_diff = df['close'].diff(momentum_period)
avg_gain = price_diff[price_diff > 0].rolling(momentum_period).mean().fillna(0)
avg_loss = -price_diff[price_diff < 0].rolling(momentum_period).mean().fillna(0)
rs = avg_gain / (avg_loss + 1e-8)
rsi_style = (rs / (1 + rs)) * 2 - 1  # Normalize to -1 to 1
regime_indicators['momentum_regime'] = rsi_style.fillna(0)

# Fill NaN values properly (use forward fill then backward fill)
regime_indicators = regime_indicators.fillna(method='bfill').fillna(method='ffill').fillna(0)

# IMPROVED: Use balanced weights that emphasize different market aspects
# This should create better regime separation while avoiding dominance of any single component
weights = [0.35, 0.30, 0.20, 0.15]  # Volatility still important, but not dominant
regime_signal = sum(w * regime_indicators[col] for w, col in zip(weights, regime_indicators.columns))

# Remove leading zeros/NaNs (where rolling windows haven't accumulated enough data)
# Keep only data where we have meaningful values
valid_start = regime_signal[regime_signal != 0].first_valid_index()
if valid_start is None:
    valid_start = regime_signal.index[252]  # Fallback to after 252-period window
regime_signal = regime_signal.loc[valid_start:]

# Normalize the signal to have stable mean and variance for MS-DR
regime_signal_mean = regime_signal.mean()
regime_signal_std = regime_signal.std()
if regime_signal_std > 1e-8:
    regime_signal = (regime_signal - regime_signal_mean) / regime_signal_std

# Replace any remaining inf/NaN
regime_signal = regime_signal.replace([np.inf, -np.inf], 0).fillna(0)

print(f"✅ Created composite regime signal: {regime_signal.shape}")
print(f"   Signal range: [{regime_signal.min():.4f}, {regime_signal.max():.4f}]")
print(f"   Signal mean: {regime_signal.mean():.4f}, std: {regime_signal.std():.4f}")
print(f"   Non-zero values: {(regime_signal != 0).sum()} / {len(regime_signal)}")

# Prepare data for MS-DR (single column, no PCA needed)
data = regime_signal.values.reshape(-1, 1)

# Ensure data is clean
data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

print(f"✅ Prepared data for MS-DR: {data.shape}")
print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")
print(f"   Data mean: {data.mean():.4f}, std: {data.std():.4f}")

# Configure MS-DR clustering WITHOUT PCA
print("\n🚀 Running MS-DR clustering with regime-specific indicators...")
config = MSDRConfig(
    n_regimes=3,
    auto_select_regimes=True,  # Let it find optimal number
    model_type='autoregression',
    switching_variance=True,
    enable_pca=False,  # Don't use PCA - we already have 1D signal
    pca_aggregation='first',  # Ignored since PCA disabled
    min_regimes=3,  # FORCE: Start with 3 regimes minimum
    max_regimes=4,  # FORCE: Try up to 4 regimes for better separation
    ic_criterion='bic',  # Use BIC for better regime selection
    order=2,  # INCREASED: AR(2) model for better dynamics
    max_iter=2000,  # INCREASED: More iterations for convergence
    method='bfgs',  # CHANGED: More robust optimization method
    random_state=42,
    use_memory_optimization=True,
    use_hardware_acceleration=True,
    show_progress=True
)

clusterer = MSDRClusterer(config)
result = clusterer.fit_predict(data)

# 🔧 FIX: Remove burn-in period if Regime 0 is a burn-in artifact
print("\n🔧 APPLYING FIXES:")
print("=" * 50)

# IMPROVED burn-in detection: Check if first ~200 samples are mostly Regime 0
regime_0_mask = result.cluster_labels == 0
burn_in_window = 200
first_window_labels = result.cluster_labels[:burn_in_window]
regime_0_percentage = (first_window_labels == 0).sum() / len(first_window_labels)

# Also check if Regime 0 is sticky (high self-transition probability)
regime_0_self_transition = result.transition_matrix[0, 0] if result.transition_matrix is not None else 0

# Detect burn-in if: 
# 1. First 200 samples are >95% Regime 0, OR
# 2. First 200 samples are >90% Regime 0 AND Regime 0 is sticky (>90% self-transition), OR
# 3. All samples are Regime 0 (degenerate case)
all_regime_0 = (result.cluster_labels == 0).all()
is_burn_in = (
    all_regime_0 or  # Degenerate case - all samples in one regime
    (regime_0_percentage > 0.95) or  # First 200 samples are mostly Regime 0
    (regime_0_percentage > 0.90 and regime_0_self_transition > 0.90)  # Sticky Regime 0
)

print(f"📊 Burn-in detection analysis:")
print(f"   First {burn_in_window} samples: {regime_0_percentage*100:.1f}% are Regime 0")
print(f"   Regime 0 self-transition: {regime_0_self_transition:.4f}")
print(f"   Burn-in detected: {is_burn_in}")

if is_burn_in:
    if all_regime_0:
        print("🚨 CRITICAL: All samples assigned to Regime 0 - degenerate clustering!")
        print("   This indicates model convergence failure. Trying alternative approach...")
    else:
        print("🚨 Detected burn-in artifact - removing first 200 samples")
    
    # Remove burn-in period (use 200 if not degenerate case)
    if all_regime_0:
        # For degenerate case, we can't remove burn-in - need to re-fit or warn
        print("   ⚠️ Cannot remove burn-in - degenerate clustering detected")
        print("   Recommendation: Check model convergence or adjust parameters")
    else:
        burn_in_samples = 200
        labels_clean = result.cluster_labels[burn_in_samples:]
    probs_clean = result.cluster_probabilities[burn_in_samples:] if result.cluster_probabilities is not None else None
    
    # Relabel regimes: 1→0, 2→1 (remove the burn-in regime 0)
    labels_relabeled = labels_clean - 1  # Shift down by 1
    labels_relabeled[labels_relabeled < 0] = 0  # Ensure no negative labels
    
    # Update result object
    result.cluster_labels = labels_relabeled
    if probs_clean is not None:
        # Remove first column (burn-in regime) and renormalize
        probs_clean = probs_clean[:, 1:]  # Remove first column
        probs_clean = probs_clean / probs_clean.sum(axis=1, keepdims=True)  # Renormalize
        result.cluster_probabilities = probs_clean
    
    # Update number of clusters
    result.n_clusters = len(np.unique(labels_relabeled))
    
    print(f"✅ Cleaned data: {len(labels_relabeled)} samples, {result.n_clusters} regimes")
    print(f"   New regime distribution: {np.bincount(labels_relabeled)}")
    
    # Update data for quality assessment
    data = data[burn_in_samples:]  # Remove burn-in from data too
    
else:
    print("✅ No burn-in artifact detected - using original results")

# Start building markdown report
markdown_content = []
markdown_content.append("# MS-DR Clustering Metrics Report\n")
markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
markdown_content.append("\n## Methodology\n\n")
markdown_content.append("**Regime-Specific Composite Indicators Approach (BALANCED):**\n\n")
markdown_content.append("- Volatility regime (35% weight): Z-score of rolling volatility\n")
markdown_content.append("- Trend regime (30% weight): Normalized price vs 50-period SMA\n")
markdown_content.append("- Volume regime (20% weight): Z-score of volume\n")
markdown_content.append("- Momentum regime (15% weight): Normalized RSI-style indicator\n\n")
markdown_content.append("**MS-DR Configuration (IMPROVED):**\n")
markdown_content.append("- PCA: **DISABLED** (using composite signal directly)\n")
markdown_content.append("- Auto-select regimes: **ENABLED** (BIC criterion, 2-4 regimes)\n")
markdown_content.append("- Model: AR(2) autoregression with switching variance\n")
markdown_content.append("- Optimization: BFGS method, 2000 max iterations\n")
markdown_content.append("- **Burn-in removal: ENABLED** (removes first 201 samples if burn-in detected)\n\n")
markdown_content.append("---\n\n")

print("\n" + "=" * 80)
print("📊 ACTUAL METRICS FROM MS-DR CLUSTERING")
print("=" * 80)

# 🎯 Clustering results
markdown_content.append("## 🎯 Clustering Results\n\n")
print("\n🎯 CLUSTERING RESULTS:")
markdown_content.append(f"- **n_clusters:** {result.n_clusters}\n")
markdown_content.append(f"- **success:** {result.success}\n")
markdown_content.append(f"- **error_message:** {result.error_message if result.error_message else 'None'}\n\n")

print(f"  • n_clusters: {result.n_clusters}")
print(f"  • success: {result.success}")
print(f"  • error_message: {result.error_message}")

# Regime distribution
unique, counts = np.unique(result.cluster_labels, return_counts=True)
print(f"  • cluster_labels shape: {result.cluster_labels.shape}")
print(f"  • Regime distribution:")

markdown_content.append("### Regime Distribution\n\n")
markdown_content.append(f"- **cluster_labels shape:** {result.cluster_labels.shape}\n\n")
markdown_content.append("| Regime ID | Samples | Percentage |\n")
markdown_content.append("|-----------|---------|------------|\n")

for regime_id, count in zip(unique, counts):
    percentage = (count / len(result.cluster_labels)) * 100
    print(f"    - Regime {regime_id}: {count} samples ({percentage:.1f}%)")
    markdown_content.append(f"| {regime_id} | {count} | {percentage:.1f}% |\n")

markdown_content.append("\n")

# Cluster probabilities
if result.cluster_probabilities is not None:
    print(f"  • cluster_probabilities shape: {result.cluster_probabilities.shape}")
    prob_mean = np.mean(result.cluster_probabilities, axis=0)
    print(f"  • Average regime probabilities: {prob_mean}")
    markdown_content.append(f"- **cluster_probabilities shape:** {result.cluster_probabilities.shape}\n")
    markdown_content.append(f"- **Average regime probabilities:** {prob_mean}\n\n")
else:
    markdown_content.append("- **cluster_probabilities:** None\n\n")

# 🎨 Quality metrics
markdown_content.append("## 🎨 Quality Metrics\n\n")
print("\n🎨 QUALITY METRICS:")

silhouette_val = f"{result.silhouette_score:.4f}" if result.silhouette_score else "None"
ch_score = f"{result.calinski_harabasz_score:.2f}" if result.calinski_harabasz_score else "None"
db_score = f"{result.davies_bouldin_score:.4f}" if result.davies_bouldin_score else "None"

markdown_content.append(f"- **silhouette_score:** {silhouette_val}\n")
markdown_content.append(f"- **calinski_harabasz_score:** {ch_score}\n")
markdown_content.append(f"- **davies_bouldin_score:** {db_score}\n")
markdown_content.append(f"- **noise_ratio:** {result.noise_ratio:.4f}\n\n")

print(f"  • silhouette_score: {silhouette_val}")
print(f"  • calinski_harabasz_score: {ch_score}")
print(f"  • davies_bouldin_score: {db_score}")
print(f"  • noise_ratio: {result.noise_ratio:.4f}")

# 📐 Model fit metrics
markdown_content.append("## 📐 Model Fit Metrics (Information Criteria)\n\n")
print("\n📐 MODEL FIT METRICS:")

ll_val = f"{result.log_likelihood:.2f}" if result.log_likelihood else "None"
aic_val = f"{result.aic:.2f}" if result.aic else "None"
bic_val = f"{result.bic:.2f}" if result.bic else "None"
hqic_val = f"{result.hqic:.2f}" if result.hqic else "None"

markdown_content.append(f"- **log_likelihood:** {ll_val}\n")
markdown_content.append(f"- **aic:** {aic_val}\n")
markdown_content.append(f"- **bic:** {bic_val}\n")
markdown_content.append(f"- **hqic:** {hqic_val}\n\n")

print(f"  • log_likelihood: {ll_val}")
print(f"  • aic: {aic_val}")
print(f"  • bic: {bic_val}")
print(f"  • hqic: {hqic_val}")

# 🔄 Transition matrix and regime dynamics
markdown_content.append("## 🔄 Transition Matrix & Regime Dynamics\n\n")
print("\n🔄 TRANSITION MATRIX & REGIME DYNAMICS:")

if result.transition_matrix is not None:
    print(f"  • transition_matrix shape: {result.transition_matrix.shape}")
    print("  • Transition Matrix:")
    
    markdown_content.append(f"- **transition_matrix shape:** {result.transition_matrix.shape}\n\n")
    markdown_content.append("### Transition Matrix\n\n")
    markdown_content.append("| From/To | " + " | ".join([f"Regime {i}" for i in range(result.transition_matrix.shape[1])]) + " |\n")
    markdown_content.append("|---------|" + "|".join(["---------" for _ in range(result.transition_matrix.shape[1] + 1)]) + "|\n")
    
    for i, row in enumerate(result.transition_matrix):
        print(f"    Regime {i}: {row}")
        markdown_content.append(f"| Regime {i} | " + " | ".join([f"{val:.4f}" for val in row]) + " |\n")
    
    markdown_content.append("\n")
    print(f"  • transition_persistence: {result.transition_persistence:.4f}")
    markdown_content.append(f"- **transition_persistence:** {result.transition_persistence:.4f}\n\n")
else:
    print("  • transition_matrix: None")
    markdown_content.append("- **transition_matrix:** None\n\n")

if result.regime_durations is not None:
    print(f"  • regime_durations: {result.regime_durations}")
    markdown_content.append(f"- **regime_durations:** {result.regime_durations}\n\n")
else:
    print("  • regime_durations: None")
    markdown_content.append("- **regime_durations:** None\n\n")

# ⚙️ Regime parameters
markdown_content.append("## ⚙️ Regime Parameters\n\n")
print("\n⚙️ REGIME PARAMETERS:")

if result.regime_params:
    markdown_content.append("### Regime Parameters\n\n")
    for regime_id, params in result.regime_params.items():
        print(f"  • Regime {regime_id}:")
        markdown_content.append(f"#### Regime {regime_id}\n\n")
        if isinstance(params, dict):
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    print(f"      - {key}: {value:.4f}")
                    markdown_content.append(f"- **{key}:** {value:.4f}\n")
                else:
                    print(f"      - {key}: {value}")
                    markdown_content.append(f"- **{key}:** {value}\n")
            markdown_content.append("\n")
        else:
            print(f"      {params}")
            markdown_content.append(f"{params}\n\n")
else:
    print("  • regime_params: None")
    markdown_content.append("- **regime_params:** None\n\n")

if result.regime_variances is not None:
    print(f"  • regime_variances: {result.regime_variances}")
    markdown_content.append(f"- **regime_variances:** {result.regime_variances}\n\n")
else:
    print("  • regime_variances: None")
    markdown_content.append("- **regime_variances:** None\n\n")

# ⚡ Processing metrics
markdown_content.append("## ⚡ Processing Metrics\n\n")
print("\n⚡ PROCESSING METRICS:")

markdown_content.append(f"- **processing_time:** {result.processing_time:.2f} seconds\n")
markdown_content.append(f"- **memory_usage_mb:** {result.memory_usage_mb:.2f} MB\n")
markdown_content.append(f"- **feature_names count:** {len(result.feature_names)}\n\n")

print(f"  • processing_time: {result.processing_time:.2f} seconds")
print(f"  • memory_usage_mb: {result.memory_usage_mb:.2f} MB")
print(f"  • feature_names count: {len(result.feature_names)}")

if len(result.feature_names) > 5:
    print(f"  • feature_names: {result.feature_names[:5]}...")
    markdown_content.append("### Feature Names (first 5)\n\n")
    for i, name in enumerate(result.feature_names[:5]):
        markdown_content.append(f"{i+1}. {name}\n")
    markdown_content.append(f"\n*... and {len(result.feature_names) - 5} more features*\n\n")
else:
    print(f"  • feature_names: {result.feature_names}")
    markdown_content.append("### Feature Names\n\n")
    for i, name in enumerate(result.feature_names):
        markdown_content.append(f"{i+1}. {name}\n")
    markdown_content.append("\n")

# 📝 Metadata
markdown_content.append("## 📝 Metadata\n\n")
print("\n📝 METADATA:")
if result.metadata:
    for key, value in result.metadata.items():
        if isinstance(value, (int, float)):
            print(f"  • {key}: {value:.4f}")
            markdown_content.append(f"- **{key}:** {value:.4f}\n")
        else:
            print(f"  • {key}: {value}")
            markdown_content.append(f"- **{key}:** {value}\n")
    markdown_content.append("\n")
else:
    print("  • metadata: None")
    markdown_content.append("- **metadata:** None\n\n")

# 🔍 COMPREHENSIVE QUALITY ASSESSMENT
print("\n" + "=" * 80)
print("🔍 COMPREHENSIVE QUALITY ASSESSMENT")
print("=" * 80)

# 🔍 DIAGNOSTIC: Check Regime 0 Duration Anomaly
print("\n" + "=" * 60)
print("🔍 DIAGNOSTIC: Regime 0 Duration Anomaly")
print("=" * 60)

print("\n📊 Regime Sequence Analysis:")
print(f"First 20 labels: {result.cluster_labels[:20]}")
print(f"Last 20 labels: {result.cluster_labels[-20:]}")

# Check if Regime 0 is at the beginning
regime_0_mask = result.cluster_labels == 0
regime_0_indices = np.where(regime_0_mask)[0]
print(f"\nRegime 0 indices (first 10): {regime_0_indices[:10]}")
print(f"Total Regime 0 samples: {len(regime_0_indices)}")

# Check if Regime 0 is all at the start
first_201_all_regime_0 = (result.cluster_labels[:201] == 0).all()
print(f"Is Regime 0 all at start (first 201 samples)? {first_201_all_regime_0}")

if first_201_all_regime_0:
    print("🚨 CONFIRMED: Regime 0 is a burn-in artifact!")
    print("   - All first 201 samples are Regime 0")
    print("   - This is likely model initialization, not a real regime")
    print("   - Solution: Remove burn-in period or re-fit with better convergence")
else:
    print("✅ Regime 0 is distributed throughout the data")

# Analyze transition matrix more carefully
print("\n📊 Transition Matrix Analysis:")
trans_df = pd.DataFrame(
    result.transition_matrix,
    index=[f'From {i}' for i in range(result.transition_matrix.shape[0])],
    columns=[f'To {i}' for i in range(result.transition_matrix.shape[1])]
)
print(trans_df)

# Check Regime 0 self-transition probability
regime_0_self_transition = result.transition_matrix[0, 0]
print(f"\nRegime 0 self-transition probability: {regime_0_self_transition:.4f}")
if regime_0_self_transition > 0.95:
    print("🚨 CONFIRMED: Regime 0 is 'sticky' - once in, almost never leaves!")
    print("   - This confirms it's a burn-in artifact")
else:
    print("✅ Regime 0 has normal transition behavior")

# Check regime durations
print(f"\nRegime durations: {result.regime_durations}")
if result.regime_durations[0] > 100:  # More than 100 hours
    print("🚨 CONFIRMED: Regime 0 duration is abnormally long!")
    print("   - This suggests it's not a recurring state")
else:
    print("✅ Regime 0 duration is reasonable")

markdown_content.append("---\n\n")
markdown_content.append("## 🔍 Comprehensive Quality Assessment Report\n\n")

# Prepare feature data for quality assessment
# Use the composite signal as feature (since we used 1D signal for clustering)
feature_df = pd.DataFrame(data, columns=['composite_regime_signal'])

# Adjust feature data to match labels length if needed
if len(result.cluster_labels) < len(feature_df):
    feature_df = feature_df.iloc[:len(result.cluster_labels)]
elif len(result.cluster_labels) > len(feature_df):
    # This shouldn't happen, but handle it
    print(f"⚠️ Warning: Labels ({len(result.cluster_labels)}) longer than features ({len(feature_df)})")

print("\n📊 Running comprehensive quality assessment...")
quality_assessor = ClusterQualityAssessor(
    enable_hardware_optimization=True,
    enable_vectorization=True
)

# Run quality assessment
quality_metrics = quality_assessor.assess_quality(
    regime_labels=result.cluster_labels,
    feature_data=feature_df,
    forward_returns=None,  # Could add if we have returns data
    timestamps=None,  # Could add timestamp index
    min_regime_size=10
)

print("\n✅ Quality assessment complete!")

# Add comprehensive quality metrics to report
markdown_content.append("### Core Clustering Quality Metrics\n\n")

if quality_metrics.silhouette_score is not None:
    print(f"  • Silhouette Score: {quality_metrics.silhouette_score:.4f}")
    markdown_content.append(f"- **Silhouette Score:** {quality_metrics.silhouette_score:.4f}")
    if quality_metrics.silhouette_score >= 0.7:
        markdown_content.append(" (Excellent)\n")
    elif quality_metrics.silhouette_score >= 0.5:
        markdown_content.append(" (Good)\n")
    elif quality_metrics.silhouette_score >= 0.3:
        markdown_content.append(" (Moderate)\n")
    else:
        markdown_content.append(" (Poor)\n")
else:
    print("  • Silhouette Score: None")
    markdown_content.append("- **Silhouette Score:** None\n")

if quality_metrics.davies_bouldin_score is not None:
    print(f"  • Davies-Bouldin Index: {quality_metrics.davies_bouldin_score:.4f}")
    markdown_content.append(f"- **Davies-Bouldin Index:** {quality_metrics.davies_bouldin_score:.4f}")
    if quality_metrics.davies_bouldin_score <= 0.5:
        markdown_content.append(" (Excellent)\n")
    elif quality_metrics.davies_bouldin_score <= 1.0:
        markdown_content.append(" (Good)\n")
    elif quality_metrics.davies_bouldin_score <= 2.0:
        markdown_content.append(" (Moderate)\n")
    else:
        markdown_content.append(" (Poor)\n")
else:
    print("  • Davies-Bouldin Index: None")
    markdown_content.append("- **Davies-Bouldin Index:** None\n")

if quality_metrics.calinski_harabasz_score is not None:
    print(f"  • Calinski-Harabasz Index: {quality_metrics.calinski_harabasz_score:.2f}")
    markdown_content.append(f"- **Calinski-Harabasz Index:** {quality_metrics.calinski_harabasz_score:.2f}")
    if quality_metrics.calinski_harabasz_score >= 500:
        markdown_content.append(" (Excellent)\n")
    elif quality_metrics.calinski_harabasz_score >= 100:
        markdown_content.append(" (Good)\n")
    elif quality_metrics.calinski_harabasz_score >= 50:
        markdown_content.append(" (Moderate)\n")
    else:
        markdown_content.append(" (Poor)\n")
else:
    print("  • Calinski-Harabasz Index: None")
    markdown_content.append("- **Calinski-Harabasz Index:** None\n")

markdown_content.append("\n### Coefficient of Variation Metrics\n\n")

if quality_metrics.within_regime_cv is not None:
    print(f"  • Within-Regime CV: {quality_metrics.within_regime_cv:.4f}")
    markdown_content.append(f"- **Within-Regime CV:** {quality_metrics.within_regime_cv:.4f}")
    if quality_metrics.within_regime_cv_std is not None:
        markdown_content.append(f" ± {quality_metrics.within_regime_cv_std:.4f}")
    markdown_content.append("\n")
else:
    print("  • Within-Regime CV: None")
    markdown_content.append("- **Within-Regime CV:** None\n")

if quality_metrics.between_regime_cv is not None:
    print(f"  • Between-Regime CV: {quality_metrics.between_regime_cv:.4f}")
    markdown_content.append(f"- **Between-Regime CV:** {quality_metrics.between_regime_cv:.4f}")
    if quality_metrics.between_regime_cv_std is not None:
        markdown_content.append(f" ± {quality_metrics.between_regime_cv_std:.4f}")
    markdown_content.append("\n")
else:
    print("  • Between-Regime CV: None")
    markdown_content.append("- **Between-Regime CV:** None\n")

if quality_metrics.per_regime_cv is not None:
    print(f"  • Per-Regime CV: {quality_metrics.per_regime_cv}")
    markdown_content.append("- **Per-Regime CV:**\n")
    for regime_id, cv_val in quality_metrics.per_regime_cv.items():
        markdown_content.append(f"  - Regime {regime_id}: {cv_val:.4f}\n")
    markdown_content.append("\n")

markdown_content.append("\n### Temporal Metrics\n\n")

if quality_metrics.temporal_smoothness is not None:
    print(f"  • Temporal Smoothness: {quality_metrics.temporal_smoothness:.4f}")
    markdown_content.append(f"- **Temporal Smoothness:** {quality_metrics.temporal_smoothness:.4f}")
    if quality_metrics.temporal_smoothness >= 0.7:
        markdown_content.append(" (Excellent)\n")
    elif quality_metrics.temporal_smoothness >= 0.5:
        markdown_content.append(" (Good)\n")
    else:
        markdown_content.append(" (Poor)\n")
else:
    print("  • Temporal Smoothness: None")
    markdown_content.append("- **Temporal Smoothness:** None\n")

if quality_metrics.regime_persistence is not None:
    print(f"  • Regime Persistence: {quality_metrics.regime_persistence:.4f}")
    markdown_content.append(f"- **Regime Persistence:** {quality_metrics.regime_persistence:.4f}\n\n")
else:
    print("  • Regime Persistence: None")
    markdown_content.append("- **Regime Persistence:** None\n\n")

markdown_content.append("### Cluster Balance Metrics\n\n")

if quality_metrics.balance_score is not None:
    print(f"  • Balance Score: {quality_metrics.balance_score:.4f}")
    markdown_content.append(f"- **Balance Score:** {quality_metrics.balance_score:.4f}")
    if quality_metrics.balance_score >= 0.7:
        markdown_content.append(" (Excellent - well balanced)\n")
    elif quality_metrics.balance_score >= 0.5:
        markdown_content.append(" (Good - reasonably balanced)\n")
    elif quality_metrics.balance_score >= 0.3:
        markdown_content.append(" (Moderate - some imbalance)\n")
    else:
        markdown_content.append(" (Poor - severe imbalance)\n")
else:
    print("  • Balance Score: None")
    markdown_content.append("- **Balance Score:** None\n")

if quality_metrics.min_cluster_size_pct is not None:
    print(f"  • Min Cluster Size: {quality_metrics.min_cluster_size_pct:.2f}%")
    markdown_content.append(f"- **Min Cluster Size:** {quality_metrics.min_cluster_size_pct:.2f}%\n")
    
if quality_metrics.max_cluster_size_pct is not None:
    print(f"  • Max Cluster Size: {quality_metrics.max_cluster_size_pct:.2f}%")
    markdown_content.append(f"- **Max Cluster Size:** {quality_metrics.max_cluster_size_pct:.2f}%\n")

if quality_metrics.cluster_size_distribution is not None:
    print(f"  • Cluster Size Distribution: {quality_metrics.cluster_size_distribution}")
    markdown_content.append("- **Cluster Size Distribution:**\n")
    for i, size_pct in enumerate(quality_metrics.cluster_size_distribution):
        markdown_content.append(f"  - Regime {i}: {size_pct:.2f}%\n")
    markdown_content.append("\n")

markdown_content.append("\n### Per-Regime Quality Metrics\n\n")

if quality_metrics.silhouette_per_cluster is not None:
    print("  • Per-Regime Silhouette Scores:")
    markdown_content.append("**Per-Regime Silhouette Scores:**\n\n")
    for regime_id, metrics_dict in quality_metrics.silhouette_per_cluster.items():
        sil_score = metrics_dict.get('silhouette_score', 'N/A')
        print(f"    - Regime {regime_id}: {sil_score:.4f}" if isinstance(sil_score, (int, float)) else f"    - Regime {regime_id}: {sil_score}")
        if isinstance(sil_score, (int, float)):
            markdown_content.append(f"- Regime {regime_id}: {sil_score:.4f}\n")
        else:
            markdown_content.append(f"- Regime {regime_id}: {sil_score}\n")
    markdown_content.append("\n")

markdown_content.append("\n### Overall Quality Score\n\n")

if quality_metrics.quality_score is not None:
    print(f"  • Overall Quality Score: {quality_metrics.quality_score:.4f}")
    markdown_content.append(f"- **Overall Quality Score:** {quality_metrics.quality_score:.4f}")
    if quality_metrics.quality_score >= 0.7:
        markdown_content.append(" ⭐⭐⭐ (Excellent)\n")
    elif quality_metrics.quality_score >= 0.5:
        markdown_content.append(" ⭐⭐ (Good)\n")
    elif quality_metrics.quality_score >= 0.3:
        markdown_content.append(" ⭐ (Moderate)\n")
    else:
        markdown_content.append(" (Poor)\n")
else:
    print("  • Overall Quality Score: None")
    markdown_content.append("- **Overall Quality Score:** None\n")

markdown_content.append("\n### Quality Assessment Summary\n\n")

# Create summary table
markdown_content.append("| Metric | Value | Status |\n")
markdown_content.append("|--------|-------|--------|\n")

if quality_metrics.silhouette_score is not None:
    status = "✅ Excellent" if quality_metrics.silhouette_score >= 0.5 else "⚠️ Moderate" if quality_metrics.silhouette_score >= 0.3 else "❌ Poor"
    markdown_content.append(f"| Silhouette Score | {quality_metrics.silhouette_score:.4f} | {status} |\n")

if quality_metrics.balance_score is not None:
    status = "✅ Well Balanced" if quality_metrics.balance_score >= 0.5 else "⚠️ Imbalanced" if quality_metrics.balance_score >= 0.3 else "❌ Severely Imbalanced"
    markdown_content.append(f"| Balance Score | {quality_metrics.balance_score:.4f} | {status} |\n")

if quality_metrics.quality_score is not None:
    status = "✅ Excellent" if quality_metrics.quality_score >= 0.7 else "✅ Good" if quality_metrics.quality_score >= 0.5 else "⚠️ Moderate" if quality_metrics.quality_score >= 0.3 else "❌ Poor"
    markdown_content.append(f"| Overall Quality | {quality_metrics.quality_score:.4f} | {status} |\n")

markdown_content.append("\n")

# Write markdown file
markdown_content.append("---\n")
markdown_content.append(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

with open(report_filename, 'w') as f:
    f.writelines(markdown_content)

print("\n" + "=" * 80)
print("✅ METRICS EXTRACTION COMPLETE")
print(f"📄 Report saved to: {report_filename}")
print("=" * 80)
