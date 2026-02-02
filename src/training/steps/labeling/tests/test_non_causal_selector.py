import pytest
import numpy as np
import pandas as pd
from src.training.steps.labeling.non_causal_feature_selector import NonCausalFeatureSelector, create_technical_feature_patterns

def test_non_causal_selector_default():
    """Test default behavior with known causal parents."""
    selector = NonCausalFeatureSelector(verbose=False)

    # Default causal parents should be identified
    causal_parents = selector.identify_causal_parents()
    assert 'volume' in causal_parents
    assert 'volatility' in causal_parents

    # Check simple filtering
    all_features = ['volume', 'volatility', 'rsi', 'macd', 'random_feature']
    non_causal = selector.filter_causal_features(all_features)

    assert 'rsi' in non_causal
    assert 'macd' in non_causal
    assert 'random_feature' in non_causal
    assert 'volume' not in non_causal
    assert 'volatility' not in non_causal

def test_pc_algorithm_mapping():
    """Test mapping of PC algorithm results to feature names."""
    # Mock PC results with strong causal link
    # Feature 0 causes Feature 1 (or vice versa, link is strong)
    strength_matrix = np.array([
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    pc_results = {
        'causal_strength': strength_matrix
    }

    feature_names = ['feature_A', 'feature_B', 'feature_C']

    selector = NonCausalFeatureSelector(
        pc_algorithm_results=pc_results,
        verbose=False
    )

    # Pass feature names to identify
    parents = selector.identify_causal_parents(feature_names=feature_names)

    # feature_A (index 0) is strong parent/source
    assert 'feature_A' in parents
    assert 'feature_B' not in parents # It is target/child in this matrix config (0->1)

def test_pc_algorithm_mapping_via_filter():
    """Test mapping when called via filter_causal_features."""
    strength_matrix = np.array([
        [0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

    pc_results = {
        'causal_strength': strength_matrix
    }

    feature_names = ['feature_A', 'feature_B', 'feature_C']

    selector = NonCausalFeatureSelector(
        pc_algorithm_results=pc_results,
        verbose=False
    )

    # This should internally call identify_causal_parents with feature_names
    non_causal = selector.filter_causal_features(feature_names)

    assert 'feature_A' not in non_causal # Should be excluded as parent
    assert 'feature_B' in non_causal # Allowed
    assert 'feature_C' in non_causal

def test_leakage_exclusion():
    """Test exclusion of leakage patterns."""
    selector = NonCausalFeatureSelector(verbose=False)

    features = [
        'normal_feature',
        'target_1d',
        'label_class',
        'future_return',
        'bin_1',
        'ret_5d'
    ]

    # identify_causal_parents will load defaults which includes leakage patterns in exclude_patterns
    non_causal = selector.filter_causal_features(features)

    assert 'normal_feature' in non_causal
    assert 'target_1d' not in non_causal
    assert 'label_class' not in non_causal
    assert 'future_return' not in non_causal
    assert 'bin_1' not in non_causal
    assert 'ret_5d' not in non_causal

def test_technical_prioritization():
    """Test technical feature prioritization."""
    selector = NonCausalFeatureSelector(max_features=2, verbose=False)

    features = ['random_noise', 'rsi_14', 'macd_signal']

    # rsi and macd should be prioritized over random_noise
    prioritized = selector.prioritize_technical_features(features)

    assert len(prioritized) == 2
    assert 'rsi_14' in prioritized
    assert 'macd_signal' in prioritized
    assert 'random_noise' not in prioritized

def test_feature_importance_boost():
    """Test that feature importance boosts score."""
    selector = NonCausalFeatureSelector(max_features=1, verbose=False)

    features = ['feat_A', 'feat_B']
    importance = {'feat_A': 0.5, 'feat_B': 0.1}

    # feat_A should win
    prioritized = selector.prioritize_technical_features(features, feature_importance=importance)
    assert prioritized == ['feat_A']

    # Now verify importance overrides technical if high enough?
    # Technical gets +1.0. Importance adds raw value.
    # If feat_A is technical (score 1) and B is not (0) but has importance 2.0.

    selector = NonCausalFeatureSelector(max_features=1, verbose=False)
    features = ['rsi', 'important_random']
    importance = {'rsi': 0.0, 'important_random': 2.0}

    prioritized = selector.prioritize_technical_features(features, feature_importance=importance)
    assert prioritized == ['important_random']

def test_pc_algorithm_missing_names_inference():
    """Test inference of names from graph keys when feature_names are missing."""
    pc_results = {
        'graph': {'A': {'parents': []}, 'B': {'parents': ['A']}},
        'causal_strength': np.array([[0, 1], [0, 0]])
    }
    selector = NonCausalFeatureSelector(pc_algorithm_results=pc_results, verbose=False)
    parents = selector.identify_causal_parents() # No feature_names passed
    assert 'A' in parents

def test_pc_algorithm_dimension_mismatch_safety():
    """Test safety bounds when dimensions mismatch."""
    pc_results = {
        'causal_strength': np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]) # 3x3
    }
    feature_names = ['X', 'Y'] # Only 2 names
    selector = NonCausalFeatureSelector(pc_algorithm_results=pc_results, verbose=False)
    parents = selector.identify_causal_parents(feature_names=feature_names)
    assert 'X' in parents
    assert 'Y' in parents

def test_token_matching_logic():
    """Test robust token matching logic (avoid substring false positives)."""
    # 'ad' is Accumulation/Distribution
    causal_graph = {'target': ['ad', 'volume']}
    selector = NonCausalFeatureSelector(causal_graph=causal_graph, verbose=False)

    features = [
        'spread',          # Should NOT match 'ad'
        'ad',              # Should match 'ad'
        'ad_ratio',        # Should match 'ad' (token)
        'my_ad_indicator', # Should match 'ad' (token)
        'volume',          # Should match 'volume'
        'BTC_volume',      # Should match 'volume' (token)
        'ETH_volume_rsi',  # Should match 'volume' (token)
        'random_feature'   # Should NOT match
    ]

    filtered = selector.filter_causal_features(features)

    assert 'spread' in filtered
    assert 'ad' not in filtered
    assert 'ad_ratio' not in filtered
    assert 'my_ad_indicator' not in filtered
    assert 'volume' not in filtered
    assert 'BTC_volume' not in filtered
    assert 'ETH_volume_rsi' not in filtered
    assert 'random_feature' in filtered

if __name__ == "__main__":
    # verification run
    try:
        test_non_causal_selector_default()
        test_pc_algorithm_mapping()
        test_pc_algorithm_mapping_via_filter()
        test_leakage_exclusion()
        test_technical_prioritization()
        test_feature_importance_boost()
        test_pc_algorithm_missing_names_inference()
        test_pc_algorithm_dimension_mismatch_safety()
        test_token_matching_logic()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
