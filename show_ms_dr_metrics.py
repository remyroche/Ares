#!/usr/bin/env python3
"""
Display Comprehensive MS-DR Clustering Metrics

This script shows all available metrics from MSDRResult object.
"""

import sys
sys.path.insert(0, 'src')

from src.training.steps.market_analysis.ms_dr_clustering.ms_dr_clusterer import MSDRResult
import numpy as np

print("=" * 80)
print("📊 MS-DR CLUSTERING - AVAILABLE METRICS REFERENCE")
print("=" * 80)

print("\n" + "=" * 80)
print("🎯 CLUSTERING RESULTS")
print("=" * 80)
metrics = [
    ("result.n_clusters", "Number of regimes discovered"),
    ("result.cluster_labels", "Regime assignment for each sample (ndarray)"),
    ("result.cluster_probabilities", "Smoothed probabilities for each regime (ndarray)"),
    ("result.success", "Whether clustering succeeded (bool)"),
    ("result.error_message", "Error message if failed (str or None)"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("🎨 QUALITY METRICS")
print("=" * 80)
metrics = [
    ("result.silhouette_score", "Silhouette coefficient (higher is better, range: -1 to 1)"),
    ("result.calinski_harabasz_score", "Calinski-Harabasz index (higher is better)"),
    ("result.davies_bouldin_score", "Davies-Bouldin index (lower is better)"),
    ("result.noise_ratio", "Ratio of noise/uncertain samples (float)"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("📐 MODEL FIT METRICS (Information Criteria)")
print("=" * 80)
metrics = [
    ("result.log_likelihood", "Log-likelihood of the fitted model"),
    ("result.aic", "Akaike Information Criterion (lower is better)"),
    ("result.bic", "Bayesian Information Criterion (lower is better)"),
    ("result.hqic", "Hannan-Quinn Information Criterion (lower is better)"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("🔄 TRANSITION MATRIX & REGIME DYNAMICS")
print("=" * 80)
metrics = [
    ("result.transition_matrix", "Transition probability matrix (n_regimes x n_regimes)"),
    ("result.transition_persistence", "Average self-transition probability (higher = more persistent)"),
    ("result.regime_durations", "Average duration of each regime (ndarray)"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("⚙️ REGIME PARAMETERS")
print("=" * 80)
metrics = [
    ("result.regime_params", "Statistical parameters for each regime (Dict)"),
    ("result.regime_variances", "Variance of each regime (ndarray)"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("⚡ PROCESSING METRICS")
print("=" * 80)
metrics = [
    ("result.processing_time", "Time taken to fit the model (seconds)"),
    ("result.memory_usage_mb", "Memory used during processing (MB)"),
    ("result.feature_names", "Names of features used (List[str])"),
]

for metric, desc in metrics:
    print(f"  • {metric:40} - {desc}")

print("\n" + "=" * 80)
print("📝 METADATA")
print("=" * 80)
print("  • result.metadata              - Additional metadata dictionary (Dict[str, Any])")

print("\n" + "=" * 80)
print("📊 HOW TO EXTRACT METRICS FROM YOUR RUN")
print("=" * 80)

example_code = """
# Example: After running clusterer.fit_predict(data)
result = clusterer.fit_predict(feature_data)

# Basic clustering info
print(f"Found {result.n_clusters} regimes")
print(f"Success: {result.success}")

# Quality assessment
print(f"Silhouette Score: {result.silhouette_score:.4f}")
print(f"Davies-Bouldin Index: {result.davies_bouldin_score:.4f}")

# Model fit
print(f"AIC: {result.aic:.2f}")
print(f"BIC: {result.bic:.2f}")

# Regime distribution
unique, counts = np.unique(result.cluster_labels, return_counts=True)
for regime_id, count in zip(unique, counts):
    print(f"Regime {regime_id}: {count} samples ({count/len(result.cluster_labels)*100:.1f}%)")

# Transition matrix
if result.transition_matrix is not None:
    print("Transition Matrix:")
    print(result.transition_matrix)
    print(f"Transition Persistence: {result.transition_persistence:.4f}")

# Regime parameters
if result.regime_params:
    for regime_id, params in result.regime_params.items():
        print(f"Regime {regime_id}: {params}")
"""

print(example_code)

print("\n" + "=" * 80)
print("✅ METRICS REFERENCE COMPLETE")
print("=" * 80)
print("\n💡 Tip: Run your clustering, then access any of these metrics from the result object!")

