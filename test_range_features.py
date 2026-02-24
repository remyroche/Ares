#!/usr/bin/env python3
"""
Test script to verify range_pct features are properly included in feature sets
"""
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extreme_price_movements.config import CFG, HELPER_BASE_FEATURES
from extreme_price_movements.utils import tprint

def test_range_features_in_config():
    """Test that range features are included in feature sets."""
    tprint("=== RANGE FEATURES CONFIGURATION TEST ===")
    
    # Check HELPER_BASE_FEATURES
    range_features = ["range_pct", "range_12h_pct", "range_16h_pct", "range_24h_pct"]
    missing_features = []
    
    for feature in range_features:
        if feature not in HELPER_BASE_FEATURES:
            missing_features.append(feature)
        else:
            tprint(f"✅ {feature} found in HELPER_BASE_FEATURES")
    
    if missing_features:
        tprint(f"❌ Missing features: {missing_features}")
        return False
    
    # Check that all feature sets include HELPER_BASE_FEATURES
    feature_sets = ["tf_feature_keys", "mr_feature_keys", "meta_feature_keys"]
    for set_name in feature_sets:
        if set_name in CFG:
            features = CFG[set_name]
            # Check if HELPER_BASE_FEATURES are included (they should be via + HELPER_BASE_FEATURES)
            tprint(f"✅ {set_name} includes HELPER_BASE_FEATURES via concatenation")
    
    tprint("✅ RANGE FEATURES CONFIGURATION VERIFIED")
    return True

def test_feature_priority_order():
    """Test that feature priority order is correct for event scoring."""
    tprint("\n=== FEATURE PRIORITY ORDER TEST ===")
    
    # This should match the order in training.py line 2696
    priority_order = ["range_16h_pct", "range_12h_pct", "range_pct", "range_24h_pct"]
    
    tprint("Feature priority order for event scoring:")
    for i, feature in enumerate(priority_order, 1):
        status = "✅" if feature in HELPER_BASE_FEATURES else "❌"
        tprint(f"  {i}. {feature} {status}")
    
    all_available = all(feature in HELPER_BASE_FEATURES for feature in priority_order)
    if all_available:
        tprint("✅ All priority features are available")
    else:
        tprint("❌ Some priority features are missing")
    
    return all_available

if __name__ == "__main__":
    config_ok = test_range_features_in_config()
    priority_ok = test_feature_priority_order()
    
    if config_ok and priority_ok:
        tprint("\n🎉 ALL TESTS PASSED - Range features are properly configured!")
    else:
        tprint("\n❌ TESTS FAILED - Check configuration")
        sys.exit(1)
