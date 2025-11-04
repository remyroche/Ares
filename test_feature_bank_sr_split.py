#!/usr/bin/env python3
"""
Test script to verify the SR feature split works with the feature bank system.

This tests:
1. Default behavior (only pre-created SR features)
2. Manual registration of custom SR features
3. Feature bank category handling
"""

import pandas as pd
import numpy as np
from src.feature_generation.core.feature_bank import FeatureBank
from src.feature_generation.core.feature_generator import FeatureCategory
from src.feature_generation.categories.custom_support_resistance import create_default_custom_sr_generators

def test_default_sr_features():
    """Test that default feature bank only has pre-created SR features."""
    print("\n" + "="*80)
    print("TEST 1: Default Feature Bank (Custom SR Disabled)")
    print("="*80)
    
    # Create feature bank with defaults
    bank = FeatureBank()
    
    # Check SR generators
    sr_generators = bank.get_generators_by_category(FeatureCategory.SUPPORT_RESISTANCE)
    print(f"✓ Found {len(sr_generators)} standard SR generators")
    
    # Check for custom SR generators (should be empty by default)
    custom_sr_generators = bank.get_generators_by_category(FeatureCategory.CUSTOM_SUPPORT_RESISTANCE)
    print(f"✓ Found {len(custom_sr_generators)} custom SR generators (expected 0)")
    
    # Verify SR generator names
    sr_names = [gen.config.name for gen in sr_generators[:5]]
    print(f"✓ Sample SR generator names: {sr_names[:3]}...")
    
    assert len(custom_sr_generators) == 0, "Custom SR should be disabled by default"
    print("✅ Test 1 PASSED: Default behavior correct\n")

def test_manual_custom_sr_registration():
    """Test manual registration of custom SR features."""
    print("="*80)
    print("TEST 2: Manual Custom SR Registration")
    print("="*80)
    
    # Create feature bank with defaults
    bank = FeatureBank()
    
    # Manually register custom SR generators
    custom_generators = create_default_custom_sr_generators()
    print(f"✓ Created {len(custom_generators)} custom SR generators")
    
    for gen in custom_generators:
        bank.register_generator(gen)
    
    # Verify registration
    custom_sr_generators = bank.get_generators_by_category(FeatureCategory.CUSTOM_SUPPORT_RESISTANCE)
    print(f"✓ Registered {len(custom_sr_generators)} custom SR generators")
    
    # Check generator types
    custom_names = [gen.config.name for gen in custom_sr_generators[:5]]
    print(f"✓ Sample custom SR names: {custom_names[:3]}...")
    
    assert len(custom_sr_generators) > 0, "Custom SR should be registered"
    assert 'sr_strength' in str(custom_names), "Should have sr_strength generator"
    print("✅ Test 2 PASSED: Manual registration works\n")

def test_feature_generation():
    """Test that feature generation works for both types."""
    print("="*80)
    print("TEST 3: Feature Generation")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 + np.random.randn(100).cumsum(),
        'high': 102 + np.random.randn(100).cumsum(),
        'low': 98 + np.random.randn(100).cumsum(),
        'volume': 1000000 + np.random.randint(-100000, 100000, 100)
    })
    
    # Test standard SR features
    bank = FeatureBank()
    sr_generators = bank.get_generators_by_category(FeatureCategory.SUPPORT_RESISTANCE)
    
    if sr_generators:
        result = sr_generators[0].generate(data)
        print(f"✓ Standard SR feature generated: {len(result.data)} values")
        assert len(result.data) > 0, "Should generate feature values"
    
    # Test custom SR features
    custom_generators = create_default_custom_sr_generators()
    if custom_generators:
        result = custom_generators[0].generate(data)
        print(f"✓ Custom SR feature generated: {len(result.data)} values")
        assert len(result.data) > 0, "Should generate feature values"
    
    print("✅ Test 3 PASSED: Feature generation works\n")

def test_category_creator_mapping():
    """Test that the category creator mapping includes custom SR."""
    print("="*80)
    print("TEST 4: Category Creator Mapping")
    print("="*80)
    
    bank = FeatureBank()
    
    # Try to create generators for CUSTOM_SUPPORT_RESISTANCE category
    try:
        custom_gens = bank._create_default_generators_for_category(FeatureCategory.CUSTOM_SUPPORT_RESISTANCE)
        print(f"✓ Category creator returned {len(custom_gens)} generators")
        assert len(custom_gens) > 0, "Should create custom SR generators"
        print("✅ Test 4 PASSED: Category creator mapping works\n")
    except Exception as e:
        print(f"❌ Test 4 FAILED: {e}\n")
        raise

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SR FEATURE SPLIT - FEATURE BANK INTEGRATION TEST")
    print("="*80)
    
    try:
        test_default_sr_features()
        test_manual_custom_sr_registration()
        test_feature_generation()
        test_category_creator_mapping()
        
        print("="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nSummary:")
        print("  ✓ Default behavior: Custom SR disabled")
        print("  ✓ Manual registration: Works correctly")
        print("  ✓ Feature generation: Both types work")
        print("  ✓ Category mapping: Properly configured")
        print("\n✅ The SR feature split works correctly with the feature bank system!")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

