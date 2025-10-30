#!/usr/bin/env python3
"""
Improved MS-DR Clustering Test with Enhanced Signal and Burn-in Detection

Addresses Problems:
1. Degenerate clustering - uses improved signal with better separation
2. Burn-in detection - enhanced logic with multiple strategies
3. Signal uniformity - validates signal quality before clustering
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
from improved_ms_dr_signal import create_improved_regime_signal


def enhanced_burn_in_removal(result, data: np.ndarray, aggressive: bool = False) -> tuple:
    """
    Enhanced burn-in detection and removal.
    
    Strategies:
    1. Check if all samples are in one regime (degenerate case)
    2. Check if first N samples are dominated by one regime
    3. Check if a regime has very high self-transition probability
    4. Check regime duration anomalies
    
    Args:
        result: MSDRResult object
        data: Original data array
        aggressive: If True, use more aggressive burn-in removal
        
    Returns:
        Tuple of (cleaned_labels, cleaned_probabilities, cleaned_data, was_cleaned, diagnostics)
    """
    print("\n" + "=" * 80)
    print("🔧 ENHANCED BURN-IN DETECTION")
    print("=" * 80)
    
    diagnostics = {}
    n_samples = len(result.cluster_labels)
    unique_regimes = np.unique(result.cluster_labels)
    n_regimes = len(unique_regimes)
    
    # Get regime counts
    regime_counts = np.bincount(result.cluster_labels)
    diagnostics['regime_counts'] = regime_counts.tolist()
    diagnostics['n_regimes'] = n_regimes
    
    print(f"\n📊 Initial Analysis:")
    print(f"   Total samples: {n_samples}")
    print(f"   Discovered regimes: {n_regimes}")
    print(f"   Regime distribution: {regime_counts}")
    
    # === STRATEGY 1: Check for degenerate clustering (all samples in one regime) ===
    is_degenerate = (n_regimes == 1) or (regime_counts.max() == n_samples)
    
    if is_degenerate:
        print("\n🚨 CRITICAL: Degenerate clustering detected!")
        print("   All samples assigned to a single regime")
        print("   This indicates:")
        print("   - Signal may be too uniform")
        print("   - Model initialization failed")
        print("   - Insufficient regime separation in data")
        
        diagnostics['is_degenerate'] = True
        diagnostics['recommendation'] = "Re-run with different signal construction or initialization"
        
        # Cannot clean degenerate case
        return (result.cluster_labels, result.cluster_probabilities, data, False, diagnostics)
    
    diagnostics['is_degenerate'] = False
    
    # === STRATEGY 2: Check for burn-in artifact (first N samples dominated by one regime) ===
    burn_in_windows = [50, 100, 200] if aggressive else [100, 200]
    burn_in_detected = False
    burn_in_samples = 0
    burn_in_regime = None
    
    for window in burn_in_windows:
        if window >= n_samples:
            continue
        
        first_window = result.cluster_labels[:window]
        regime_pcts = np.bincount(first_window, minlength=n_regimes) / window
        dominant_regime = np.argmax(regime_pcts)
        dominant_pct = regime_pcts[dominant_regime]
        
        print(f"\n   Window {window}: Regime {dominant_regime} = {dominant_pct*100:.1f}%")
        
        # Burn-in if >90% in first window (aggressive) or >95% (normal)
        threshold = 0.90 if aggressive else 0.95
        
        if dominant_pct > threshold:
            burn_in_detected = True
            burn_in_samples = window
            burn_in_regime = dominant_regime
            print(f"   ✅ Burn-in artifact detected in first {window} samples")
            break
    
    # === STRATEGY 3: Check transition matrix for sticky regimes ===
    if result.transition_matrix is not None and not burn_in_detected:
        transition_matrix = result.transition_matrix
        self_transitions = np.diag(transition_matrix)
        
        print(f"\n   Self-transition probabilities: {self_transitions}")
        
        # Check if any regime is "sticky" (high self-transition)
        sticky_threshold = 0.95 if aggressive else 0.98
        sticky_regimes = np.where(self_transitions > sticky_threshold)[0]
        
        if len(sticky_regimes) > 0:
            for regime_id in sticky_regimes:
                regime_count = regime_counts[regime_id] if regime_id < len(regime_counts) else 0
                regime_pct = regime_count / n_samples
                
                print(f"   Regime {regime_id}: sticky (self-trans={self_transitions[regime_id]:.3f}), {regime_pct*100:.1f}% of data")
                
                # If sticky regime is dominant AND at the start, consider it burn-in
                if regime_pct > 0.5:
                    # Check if this regime is at the start
                    first_100 = result.cluster_labels[:100]
                    regime_in_first_100 = (first_100 == regime_id).sum() / len(first_100)
                    
                    if regime_in_first_100 > 0.7:
                        burn_in_detected = True
                        burn_in_regime = regime_id
                        burn_in_samples = 100
                        print(f"   ✅ Sticky regime {regime_id} detected as burn-in artifact")
                        break
    
    # === STRATEGY 4: Check regime durations for anomalies ===
    if result.regime_durations is not None and not burn_in_detected:
        regime_durations = result.regime_durations
        print(f"\n   Regime durations: {regime_durations}")
        
        # Check if any regime has abnormally long duration
        max_duration = np.max(regime_durations)
        if max_duration > n_samples * 0.8:  # More than 80% of data
            long_regime = np.argmax(regime_durations)
            print(f"   Regime {long_regime} has abnormally long duration: {max_duration}")
            
            # Check if it's at the start
            first_100 = result.cluster_labels[:100]
            regime_in_first_100 = (first_100 == long_regime).sum() / len(first_100)
            
            if regime_in_first_100 > 0.7:
                burn_in_detected = True
                burn_in_regime = long_regime
                burn_in_samples = 100
                print(f"   ✅ Long-duration regime {long_regime} detected as burn-in artifact")
    
    # === APPLY BURN-IN REMOVAL ===
    if burn_in_detected and burn_in_samples > 0:
        print(f"\n✅ Removing burn-in: first {burn_in_samples} samples (Regime {burn_in_regime})")
        
        # Remove burn-in samples
        labels_clean = result.cluster_labels[burn_in_samples:]
        data_clean = data[burn_in_samples:]
        
        if result.cluster_probabilities is not None:
            probs_clean = result.cluster_probabilities[burn_in_samples:]
        else:
            probs_clean = None
        
        # Update regime labels
        print(f"   Cleaned data: {len(labels_clean)} samples")
        print(f"   New regime distribution: {np.bincount(labels_clean)}")
        
        diagnostics['burn_in_detected'] = True
        diagnostics['burn_in_samples'] = burn_in_samples
        diagnostics['burn_in_regime'] = int(burn_in_regime)
        
        return (labels_clean, probs_clean, data_clean, True, diagnostics)
    
    else:
        print("\n✅ No burn-in artifact detected")
        diagnostics['burn_in_detected'] = False
        
        return (result.cluster_labels, result.cluster_probabilities, data, False, diagnostics)


def run_improved_ms_dr_test():
    """Run MS-DR test with improved signal and burn-in detection."""
    
    print("=" * 80)
    print("🎯 IMPROVED MS-DR CLUSTERING TEST")
    print("=" * 80)
    
    # Create outcomes directory
    outcomes_dir = Path("outcomes")
    outcomes_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = outcomes_dir / f"improved_ms_dr_metrics_{timestamp}.md"
    
    print(f"📝 Report will be saved to: {report_filename}")
    
    # === STEP 1: Create market data ===
    print("\n" + "=" * 80)
    print("📊 STEP 1: Creating Market Data")
    print("=" * 80)
    
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    base_price = 3000.0
    
    # Create 3 distinct market regimes
    regime_lengths = [350, 300, 350]
    regime_params = [
        {'volatility': 0.02, 'trend': 0.001, 'volume': 1.5},   # Bull
        {'volatility': 0.05, 'trend': -0.0005, 'volume': 0.8}, # Bear
        {'volatility': 0.01, 'trend': 0.0, 'volume': 1.0}      # Sideways
    ]
    
    prices = [base_price]
    volumes = []
    regime_idx = 0
    regime_counter = 0
    
    for i in range(n_samples):
        if regime_counter >= regime_lengths[regime_idx]:
            regime_idx = (regime_idx + 1) % 3
            regime_counter = 0
        
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
    
    print(f"✅ Created market data: {df.shape}")
    
    # === STEP 2: Create improved regime signal ===
    print("\n" + "=" * 80)
    print("🔧 STEP 2: Creating Improved Regime Signal")
    print("=" * 80)
    
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
        print("   Consider adjusting signal construction parameters")
    else:
        print("✅ Signal quality is good")
    
    # Prepare data for MS-DR
    data = regime_signal.values.reshape(-1, 1)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n✅ Prepared data for MS-DR: {data.shape}")
    print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")
    print(f"   Data mean: {data.mean():.4f}, std: {data.std():.4f}")
    
    # === STEP 3: Run MS-DR clustering with improved configuration ===
    print("\n" + "=" * 80)
    print("🚀 STEP 3: Running MS-DR Clustering")
    print("=" * 80)
    
    config = MSDRConfig(
        n_regimes=3,
        auto_select_regimes=True,
        model_type='autoregression',
        switching_variance=True,
        enable_pca=False,  # Already have 1D signal
        min_regimes=2,  # Allow fewer regimes if data doesn't support more
        max_regimes=5,  # Try more regimes
        ic_criterion='bic',
        order=2,
        max_iter=3000,  # More iterations
        method='powell',  # Try Powell instead of BFGS (more robust)
        random_state=42,
        use_memory_optimization=True,
        use_hardware_acceleration=True,
        show_progress=True
    )
    
    clusterer = MSDRClusterer(config)
    result = clusterer.fit_predict(data)
    
    print(f"\n✅ MS-DR clustering completed")
    print(f"   Success: {result.success}")
    print(f"   N clusters: {result.n_clusters}")
    print(f"   Processing time: {result.processing_time:.2f}s")
    
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
        data = data_clean  # Use cleaned data for quality assessment
    
    # === STEP 5: Quality assessment ===
    print("\n" + "=" * 80)
    print("📊 STEP 5: Quality Assessment")
    print("=" * 80)
    
    feature_df = pd.DataFrame(data, columns=['composite_regime_signal'])
    
    quality_assessor = ClusterQualityAssessor(
        enable_hardware_optimization=True,
        enable_vectorization=True
    )
    
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
    
    # === STEP 6: Generate report ===
    print("\n" + "=" * 80)
    print("📝 STEP 6: Generating Report")
    print("=" * 80)
    
    markdown_content = []
    markdown_content.append("# Improved MS-DR Clustering Report\n")
    markdown_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
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
    
    markdown_content.append("3. **Improved MS-DR Configuration**\n")
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
    print("✅ IMPROVED MS-DR TEST COMPLETE")
    print("=" * 80)
    
    return result, quality_metrics, signal_diagnostics, burn_in_diag


if __name__ == "__main__":
    run_improved_ms_dr_test()

