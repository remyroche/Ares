
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath("."))

from src.training.steps.labeling.meta_labeling_hpo_sample_weighted import select_features_hierarchical
from sklearn.metrics import roc_auc_score
from src.utils.tprint import tprint_info, tprint_warning, tprint_error


def test_auc_calculation():
    print("\n--- Testing AUC Calculation Edge Cases ---")
    
    # Case 1: Single class in y_true
    y_true_single = np.array([1, 1, 1, 1, 1])
    y_score = np.array([0.6, 0.7, 0.8, 0.9, 0.6])
    print("Case 1: Single class in y_true (all 1s)")
    try:
        auc = roc_auc_score(y_true_single, y_score)
        print(f"Result: {auc}")
    except Exception as e:
        print(f"Caught expected exception: {e}")

    # Case 2: Uniform predictions
    y_true_mixed = np.array([0, 1, 0, 1, 0])
    y_score_uniform = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    print("\nCase 2: Uniform predictions (all 0.5)")
    try:
        auc = roc_auc_score(y_true_mixed, y_score_uniform)
        print(f"Result: {auc}")
    except Exception as e:
        print(f"Caught expected exception: {e}")

    # Case 3: Nan in input
    y_score_nan = np.array([0.5, np.nan, 0.5, 0.5, 0.5])
    print("\nCase 3: NaN in predictions")
    try:
        auc = roc_auc_score(y_true_mixed, y_score_nan)
        print(f"Result: {auc}")
    except Exception as e:
        print(f"Caught expected exception: {e}")

def test_feature_selection():
    print("\n--- Testing Hierarchical Feature Selection ---")
    
    # Generate synthetic correlated data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Create base factors
    factors = np.random.randn(n_samples, 5)
    
    # Create features as linear combinations of factors + noise
    df_dict = {}
    quality_scores = {}
    
    # Cluster 1: Derived from factor 0 (High correlation)
    for i in range(20):
        name = f"feat_cluster1_{i}"
        df_dict[name] = factors[:, 0] + np.random.randn(n_samples) * 0.1
        quality_scores[name] = 0.5 + np.random.rand() * 0.1
        
    # Cluster 2: Derived from factor 1
    for i in range(20):
        name = f"feat_cluster2_{i}"
        df_dict[name] = factors[:, 1] + np.random.randn(n_samples) * 0.1
        quality_scores[name] = 0.4 + np.random.rand() * 0.1

    # Random noise features
    for i in range(60):
        name = f"feat_noise_{i}"
        df_dict[name] = np.random.randn(n_samples)
        quality_scores[name] = 0.1 + np.random.rand() * 0.05
        
    df_features = pd.DataFrame(df_dict)
    
    # We deliberately ask for MORE features than distinct clusters
    target_n = 50 
    print(f"Generated {df_features.shape[1]} features. Target selection: {target_n}")
    print("Expecting significantly fewer selected features due to clustering.")
    
    selected_df = select_features_hierarchical(
        df_features=df_features,
        quality_scores=quality_scores,
        target_n=target_n
    )
    
    print(f"\nFinal Selected Count: {len(selected_df.columns)}")
    print(f"Columns: {selected_df.columns.tolist()[:5]} ...")


if __name__ == "__main__":
    test_auc_calculation()
    test_feature_selection()
