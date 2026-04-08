#!/usr/bin/env python3
"""Test script for params_store.load_inference_candidate_mask_params_per_bucket()"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extreme_price_movements.offline_optimisers.params_store import (
    load_inference_candidate_mask_params_per_bucket,
)


def test_default_params():
    """Test with default parameters (top_n=4, ranking_metric='composite_score')."""
    print("\n" + "="*60)
    print("TEST 1: Default parameters (top_n=4, ranking_metric='composite_score')")
    print("="*60)
    
    strategies = load_inference_candidate_mask_params_per_bucket()
    
    print(f"\nLoaded {len(strategies)} strategies")
    print(f"Expected: 4 strategies")
    
    if len(strategies) > 0:
        print("\nFirst strategy:")
        print(f"  strategy_id: {strategies[0].get('strategy_id', 'N/A')[:80]}...")
        print(f"  trade_side: {strategies[0].get('trade_side', 'N/A')}")
        print(f"  canonical_key: {strategies[0].get('base_event_trigger', 'N/A')[:80]}...")
    
    assert len(strategies) <= 4, f"Expected ≤ 4 strategies, got {len(strategies)}"
    print("\n✓ Test 1 PASSED")


def test_custom_ranking_metric():
    """Test with custom ranking metric."""
    print("\n" + "="*60)
    print("TEST 2: Custom ranking metric (learnability_step_c_score)")
    print("="*60)
    
    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=4, 
        ranking_metric="learnability_step_c_score"
    )
    
    print(f"\nLoaded {len(strategies)} strategies")
    print(f"Expected: 4 strategies")
    
    assert len(strategies) <= 4, f"Expected ≤ 4 strategies, got {len(strategies)}"
    print("\n✓ Test 2 PASSED")


def test_fallback_metric():
    """Test fallback when metric doesn't exist."""
    print("\n" + "="*60)
    print("TEST 3: Fallback when metric doesn't exist")
    print("="*60)
    
    strategies = load_inference_candidate_mask_params_per_bucket(
        top_n=4, 
        ranking_metric="nonexistent_metric"
    )
    
    print(f"\nLoaded {len(strategies)} strategies")
    print(f"Expected: 4 strategies (fallback to first 4 rows)")
    
    assert len(strategies) <= 4, f"Expected ≤ 4 strategies, got {len(strategies)}"
    print("\n✓ Test 3 PASSED")


def test_top_n_parameter():
    """Test custom top_n parameter."""
    print("\n" + "="*60)
    print("TEST 4: Custom top_n parameter (top_n=2)")
    print("="*60)
    
    strategies = load_inference_candidate_mask_params_per_bucket(top_n=2)
    
    print(f"\nLoaded {len(strategies)} strategies")
    print(f"Expected: 2 strategies")
    
    assert len(strategies) <= 2, f"Expected ≤ 2 strategies, got {len(strategies)}"
    print("\n✓ Test 4 PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING params_store.load_inference_candidate_mask_params_per_bucket()")
    print("="*60)
    
    try:
        test_default_params()
        test_custom_ranking_metric()
        test_fallback_metric()
        test_top_n_parameter()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
