#!/usr/bin/env python3
"""
HDP-HMM Iterative Parameter Tuning
Systematically explore parameter space to find optimal trade-off:

α ∈ [1, 1.9]    - Controls regime distribution (balance)
κ ∈ [5, 35]     - Controls regime persistence (separation & temporal)
γ ∈ [3, 6]      - Controls regime distinctness

Evaluates all trials and saves comprehensive metrics to CSV.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Union
from pathlib import Path
import sys
import itertools

# Removed: from src.utils.ml_common.optimization.grid_utils import create_linspace_grid
# We use itertools.product directly to support different step counts per parameter.

print("=" * 80)
print("HDP-HMM Iterative Parameter Tuning (Full Grid Search)")
print("=" * 80)

# Import modules
try:
    from src.training.steps.market_analysis.hdp_hmm_clustering.hdp_hmm_clusterer import (
        HDPHMMClusterer, HDPHMMConfig, HMM_AVAILABLE, HMM_LIBRARY
    )
    # Import the assessor to access the full metrics object if needed
    from src.training.steps.market_analysis.clusters.cluster_quality_assessor import ClusterQualityMetrics
    from src.utils.data.klines_parquet import KlinesParquetManager
    from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
    print(f"✅ Modules imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

if not HMM_AVAILABLE:
    print("⚠️ HMM libraries (pyhsmm) not available. Skipping tuning.")
    sys.exit(0)

# --- Helper Functions ---

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) or isinstance(v, np.ndarray):
            # Convert lists/arrays to a string representation for CSV
            try:
                items.append((new_key, str(list(v))))
            except Exception:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

# --- Data Loading and Preparation ---

print(f"\n📊 Loading 180 days of data...")
try:
    klines_manager = KlinesParquetManager(data_dir="historical_data", exchange="binance")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    df = klines_manager.read_data(
        symbol="ETHUSDT", interval="1h",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    if df is None or df.empty:
        raise ValueError("No data")
    
    print(f"   ✅ Loaded: {df.shape}")
    
    # Generate features
    print("   🔄 Generating features...")
    regime_integrator = RegimeFeatureIntegration()
    
    feature_chunks = []
    # Use a larger, more realistic chunk for feature generation
    for i in range(0, len(df) - 50 + 1, 10):
        chunk = df.iloc[i:i+50]
        if len(chunk) >= 48: # Ensure enough data for rolling ops
            try:
                regime_features = regime_integrator._generate_regime_features(chunk)
                chunk_df = pd.DataFrame([regime_features]) if isinstance(regime_features, dict) else regime_features
                feature_chunks.append(chunk_df)
            except Exception as e:
                # print(f"Warning: Skipping chunk due to error: {e}")
                continue
    
    if not feature_chunks:
        raise ValueError("No feature chunks were generated. Check feature generation logic and data length.")

    feature_df = pd.concat(feature_chunks, ignore_index=True).fillna(0)
    
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            except:
                feature_df[col] = pd.Categorical(feature_df[col]).codes
    feature_df = feature_df.fillna(0)
    
    # Two-scale normalization
    print("   🔧 Applying two-scale normalization (12h + 48h)...")
    feature_df_normalized = pd.DataFrame()
    
    for col in feature_df.columns:
        # Short-term (12h)
        mean_12h = feature_df[col].rolling(12, min_periods=5).mean()
        std_12h = feature_df[col].rolling(12, min_periods=5).std()
        feature_df_normalized[f'{col}_short'] = (feature_df[col] - mean_12h) / (std_12h + 1e-8)
        
        # Long-term (48h)
        mean_48h = feature_df[col].rolling(48, min_periods=10).mean()
        std_48h = feature_df[col].rolling(48, min_periods=10).std()
        feature_df_normalized[f'{col}_long'] = (feature_df[col] - mean_48h) / (std_48h + 1e-8)
    
    # Add robustness for division by zero
    feature_df_normalized = feature_df_normalized.fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"   ✅ Ready: {feature_df_normalized.shape}")
    
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# --- Parameter Grid Definition ---

print(f"\n🔧 Generating Parameter Grid...")

# 1. New Parameter Ranges with specified steps:
alpha_values = np.linspace(1.0, 1.9, 9)  # 9 steps
kappa_values = np.linspace(5.0, 35.0, 15) # 15 steps
gamma_values = np.linspace(3.0, 6.0, 6)   # 6 steps

total_combinations = len(alpha_values) * len(kappa_values) * len(gamma_values)

print(f"   α: 9 steps from [1.0, 1.9]")
print(f"   κ: 15 steps from [5.0, 35.0]")
print(f"   γ: 6 steps from [3.0, 6.0]")
print(f"   Total combinations: {total_combinations} (9*15*6)")
print(f"   Estimated time: ~{total_combinations*3/60:.1f} minutes (3s per run)")

# Create the list of configs from the product of all parameter values
test_configs = list(itertools.product(alpha_values, kappa_values, gamma_values))

# --- Grid Search Execution ---

print(f"\n🔍 Running full grid search ({len(test_configs)} combinations)...")

# Create outcomes directory
from pathlib import Path
outcomes_dir = Path("outcomes")
outcomes_dir.mkdir(exist_ok=True)

results = []

for i, (alpha, kappa, gamma) in enumerate(test_configs, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}/{len(test_configs)}: α={alpha:.3f}, κ={kappa:.3f}, γ={gamma:.3f}")
    print(f"{'='*60}")
    
    # Force garbage collection to prevent memory/semaphore leaks
    import gc
    gc.collect()
    
    try:
        config = HDPHMMConfig(
            alpha=alpha,
            kappa=kappa,
            gamma=gamma,
            n_iterations=75,
            n_burnin=15,
            max_states=15,
            kmeans_n_clusters=7,
            pca_components=20,
            use_gpu_acceleration=True,
            use_kmeans_warmstart=True,
            enable_advanced_diagnostics=True,
            convergence_check=True,
            convergence_patience=5,
            random_state=789,
            show_progress=False  # Disable progress bar for cleaner output
        )
        
        start_time = datetime.now()
        clusterer = HDPHMMClusterer(config)
        
        # Run the clusterer and get the result object
        # Pass data as numpy array for consistency and skip redundant validation
        result = clusterer.fit_predict(feature_df_normalized.values, validate=False)
        elapsed = (datetime.now() - start_time).total_seconds()

        if not result.success:
            print(f"   ❌ Failed: {result.error_message}")
            result_dict = {'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'error': result.error_message, 'success': False}
            results.append(result_dict)
            continue

        # --- Metric Collection ---
        # 2. Tie it to clusters/cluster_quality_assessor.py
        
        # Start with base parameters and HMM-specific results
        result_dict = {
            'alpha': alpha,
            'kappa': kappa,
            'gamma': gamma,
            'n_clusters': result.n_clusters,
            'silhouette_score': result.silhouette_score, # Top-level metric
            'davies_bouldin_score': result.davies_bouldin_score, # Top-level
            'calinski_harabasz_score': result.calinski_harabasz_score, # Top-level
            'log_likelihood': result.log_likelihood,
            'transition_persistence': result.transition_persistence,
            'runtime': elapsed,
            'success': True,
            'error': None
        }

        # Extract and flatten the detailed quality assessment from the assessor
        if result.quality_assessment:
            flat_quality_metrics = flatten_dict(result.quality_assessment, parent_key='qa')
            result_dict.update(flat_quality_metrics)
        else:
            print(f"   ⚠️ WARNING: quality_assessment is None, using only top-level metrics")
        
        results.append(result_dict)
        # --- End of new metric collection ---

        # Quick feedback - use top-level metrics first, then qa_ if available
        print(f"   Clusters: {result_dict.get('n_clusters', 'N/A')}")
        print(f"   Silhouette: {result_dict.get('silhouette_score', 0.0):.4f}")
        
        # Try both top-level and qa_ prefixed names
        temporal = result_dict.get('qa_temporal_smoothness') or result_dict.get('temporal_smoothness', 0.0) or 0.0
        balance = result_dict.get('qa_balance_score') or result_dict.get('balance_score', 0.0) or 0.0
        between_cv = result_dict.get('qa_between_regime_cv') or result_dict.get('between_regime_cv', 0.0) or 0.0
        within_cv = result_dict.get('qa_within_regime_cv') or result_dict.get('within_regime_cv', 1.0) or 1.0
        cv_ratio_feat = between_cv / (within_cv + 1e-9)
        cv_ratio_econ = result_dict.get('qa_economic_cv_metrics_economic_cv_ratio_mean_return', 0.0) or 0.0
        
        print(f"   Temporal: {temporal:.4f} {'✅' if temporal >= 0.70 else '⚠️'}")
        print(f"   Balance:  {balance:.4f} {'✅' if balance >= 0.40 else '⚠️'}")
        print(f"   CV Ratio (Feat): {cv_ratio_feat:.4f} {'✅' if cv_ratio_feat >= 1.0 else '⚠️'}")
        print(f"   CV Ratio (Econ): {cv_ratio_econ:.4f} {'✅' if cv_ratio_econ >= 1.0 else '⚠️'}")
        print(f"   Runtime: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"   ❌ CRITICAL FAILURE: {e}")
        result_dict = {'alpha': alpha, 'kappa': kappa, 'gamma': gamma, 'error': str(e), 'success': False}
        results.append(result_dict)
        continue
    
    # Periodic checkpoint save (every 50 tests)
    if i % 50 == 0 and results:
        try:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_path = outcomes_dir / f"hdp_hmm_checkpoint_{i}_tests.csv"
            checkpoint_df.to_csv(checkpoint_path, index=False)
            print(f"\n💾 Checkpoint saved: {checkpoint_path} ({len(results)} tests)")
        except Exception as e:
            print(f"\n⚠️ Checkpoint save failed: {e}")

# --- Results Analysis and CSV Saving ---

print(f"\n{'='*80}")
print("📊 TUNING RESULTS")
print(f"{'='*80}")

if not results:
    print("❌ No successful runs")
    sys.exit(1)

results_df = pd.DataFrame(results)

# Calculate a simple composite score for ranking (example)
# This uses the new flattened column names
def calculate_composite(row):
    try:
        sil = row.get('silhouette_score', 0.0)
        bal = row.get('qa_balance_score', 0.0)
        temp = row.get('qa_temporal_smoothness', 0.0)
        
        cv_ratio_feat = row.get('qa_between_regime_cv', 0.0) / (row.get('qa_within_regime_cv', 1.0) + 1e-9)
        cv_ratio_econ = row.get('qa_economic_cv_metrics_economic_cv_ratio_mean_return', 0.0)
        
        # Give weight to both feature and economic separation
        cv_score = (cv_ratio_feat + cv_ratio_econ) / 2.0
        
        # Composite: 30% Sil, 30% Balance, 20% Temporal, 20% CV
        composite_score = (sil * 0.3) + (bal * 0.3) + (temp * 0.2) + (np.tanh(cv_score) * 0.2)
        
        # Penalize if not successful
        if not row.get('success', False):
            return -1.0
            
        return composite_score
    except:
        return 0.0

results_df['composite_score'] = results_df.apply(calculate_composite, axis=1)
results_df = results_df.sort_values('composite_score', ascending=False)

# Display results table (top 20)
print(f"\n📊 Top 20 Results (sorted by composite score):")
display_cols = [
    'alpha', 'kappa', 'gamma', 'n_clusters', 'composite_score', 
    'silhouette_score', 'qa_balance_score', 'qa_temporal_smoothness', 'runtime'
]
# Filter for columns that actually exist in the dataframe
display_cols = [col for col in display_cols if col in results_df.columns]
print(f"\n{results_df[display_cols].head(20).to_string(index=False)}")

# Save detailed results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 2. Create a CSV file with all the data, stored in outcomes/
csv_path = outcomes_dir / f"hdp_hmm_full_tuning_results_{timestamp}.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n💾 Comprehensive CSV results saved to: {csv_path}")

# Generate simple markdown report
report_path = outcomes_dir / f"hdp_hmm_tuning_report_{timestamp}.md"

try:
    best_composite = results_df.iloc[0]
    
    report = f"""# HDP-HMM Iterative Tuning Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Configurations Tested**: {len(results_df)}

## Parameter Ranges Explored

- **α (alpha)**: 9 steps from [1.0, 1.9]
- **κ (kappa)**: 15 steps from [5.0, 35.0]
- **γ (gamma)**: 6 steps from [3.0, 6.0]

## 🏆 Best Configuration (Composite Score)

**Parameters:**
- α = {best_composite['alpha']:.4f}
- κ = {best_composite['kappa']:.4f}
- γ = {best_composite['gamma']:.4f}

**Metrics:**
- Composite Score: {best_composite.get('composite_score', 0.0):.4f}
- Clusters: {best_composite.get('n_clusters', 'N/A')}
- Silhouette: {best_composite.get('silhouette_score', 0.0):.4f}
- Balance: {best_composite.get('qa_balance_score', 0.0):.4f}
- Temporal Smoothness: {best_composite.get('qa_temporal_smoothness', 0.0):.4f}
- Runtime: {best_composite.get('runtime', 0.0):.2f}s

## Summary Statistics

- Total Successful Runs: {results_df['success'].sum() if 'success' in results_df else len(results_df)}
- Average Clusters: {results_df['n_clusters'].mean() if 'n_clusters' in results_df else 'N/A'}
- Average Runtime: {results_df['runtime'].mean() if 'runtime' in results_df else 'N/A':.2f}s

## Recommendations

Based on the tuning results:
1. The best parameter combination shows good balance between cluster quality and temporal coherence
2. Review the CSV file for detailed metrics across all configurations
3. Consider the top 5-10 configurations for further validation

---
*Full results available in: {csv_path.name}*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {report_path}")
    
except Exception as e:
    print(f"⚠️ Failed to generate report: {e}")

print(f"\n{'='*80}")
print("✅ TUNING COMPLETE!")
print(f"{'='*80}")
print(f"📊 Results: {csv_path}")
print(f"📄 Report: {report_path}")
print(f"\n🏆 Best Configuration:")
print(f"   α={best_composite['alpha']:.4f}, κ={best_composite['kappa']:.4f}, γ={best_composite['gamma']:.4f}")
print(f"   Composite Score: {best_composite.get('composite_score', 0.0):.4f}")
print(f"{'='*80}")
