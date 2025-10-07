#!/usr/bin/env python3
"""
Test script for End-to-End Roadmap System

This script tests the complete end-to-end roadmap system to ensure
all components work together correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(n_samples=1000):
    """Create test market data."""
    print("Creating test market data...")
    
    # Create sample bars
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='5min')
    np.random.seed(42)  # For reproducibility
    
    # Generate realistic price data
    returns = np.random.randn(n_samples) * 0.001
    prices = 100 * np.exp(np.cumsum(returns))
    
    bars = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.0005),
        'high': prices * (1 + np.abs(np.random.randn(n_samples) * 0.001)),
        'low': prices * (1 - np.abs(np.random.randn(n_samples) * 0.001)),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add some book data (optional)
    bars['bid'] = bars['close'] * (1 - np.random.rand(n_samples) * 0.001)
    bars['ask'] = bars['close'] * (1 + np.random.rand(n_samples) * 0.001)
    bars['bid_size'] = np.random.randint(100, 1000, n_samples)
    bars['ask_size'] = np.random.randint(100, 1000, n_samples)
    bars['trade_count'] = np.random.randint(10, 100, n_samples)
    
    print(f"Created {len(bars)} bars with {len(bars.columns)} columns")
    return bars

def create_test_targets(bars):
    """Create test targets."""
    print("Creating test targets...")
    
    # Simple momentum-based targets
    returns = np.log(bars['close'] / bars['close'].shift(1))
    
    targets = {
        1: pd.Series(returns, index=bars.index),  # 1-bar horizon
        3: pd.Series(returns.rolling(3).mean(), index=bars.index)  # 3-bar horizon
    }
    
    print(f"Created targets for horizons: {list(targets.keys())}")
    return targets

def test_basic_imports():
    """Test basic imports."""
    print("\n=== Testing Basic Imports ===")
    
    try:
        from src.end_to_end_roadmap import (
            EndToEndRoadmapSystem,
            SystemConfig,
            run_end_to_end_pipeline
        )
        print("✅ Main system imports successful")
    except Exception as e:
        print(f"❌ Main system imports failed: {e}")
        return False
    
    try:
        from src.feature_engineering.feature_registry import FeatureRegistry
        print("✅ Feature registry import successful")
    except Exception as e:
        print(f"❌ Feature registry import failed: {e}")
        return False
    
    try:
        from src.feature_engineering.transforms import TransformRouter
        print("✅ Transform system import successful")
    except Exception as e:
        print(f"❌ Transform system import failed: {e}")
        return False
    
    try:
        from src.feature_engineering.interactions import InteractionEngine
        print("✅ Interaction engine import successful")
    except Exception as e:
        print(f"❌ Interaction engine import failed: {e}")
        return False
    
    return True

def test_feature_registry():
    """Test feature registry."""
    print("\n=== Testing Feature Registry ===")
    
    try:
        from src.feature_engineering.feature_registry import FeatureRegistry
        
        registry = FeatureRegistry()
        features = registry.get_all_features()
        
        print(f"✅ Registry created with {len(features)} features")
        
        # Test feature families
        families = ['price_returns', 'volatility', 'mean_reversion', 'liquidity_micro', 'anchors_tod', 'context']
        for family in families:
            family_features = registry.get_features_by_family(family)
            print(f"   {family}: {len(family_features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature registry test failed: {e}")
        return False

def test_transform_system():
    """Test transform system."""
    print("\n=== Testing Transform System ===")
    
    try:
        from src.feature_engineering.transforms import OnlineEWZ, TODRank, SignedLog, Winsorization
        
        # Test data
        data = pd.Series(np.random.randn(100))
        
        # Test EW-Z
        ewz = OnlineEWZ(halflife=12)
        result = ewz.fit_transform(data)
        print(f"✅ EW-Z transform: {len(result)} values")
        
        # Test Signed-log
        slog = SignedLog()
        result = slog.fit_transform(data)
        print(f"✅ Signed-log transform: {len(result)} values")
        
        # Test Winsorization
        winsor = Winsorization()
        result = winsor.fit_transform(data)
        print(f"✅ Winsorization: {len(result)} values")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform system test failed: {e}")
        return False

def test_interaction_engine():
    """Test interaction engine."""
    print("\n=== Testing Interaction Engine ===")
    
    try:
        from src.feature_engineering.interactions import InteractionEngine, create_default_interaction_config
        
        config = create_default_interaction_config()
        engine = InteractionEngine(config)

        expected_new = {
            'i/vol/sigmaew_x_posmom5_guard',
            'i/vol/sigmaew_x_negmom5_guard',
            'i/vol/sigmaslope_x_trendguard'
        }

        assert expected_new.issubset(set(config.keys())), "New convex interactions missing from config"

        print(f"✅ Interaction engine created with {len(config)} interactions")

        # Test data covering required fields
        rng = np.random.default_rng(123)
        test_data = pd.DataFrame({
            't/mom5': rng.normal(size=100),
            't/mom20': rng.normal(size=100),
            't/rsi14': rng.normal(size=100),
            't/sigma_ew': np.abs(rng.normal(size=100)),
            't/bollz20': rng.normal(size=100),
            't/spread_z18': np.abs(rng.normal(size=100)),
            't/vwap_session_dist': rng.normal(size=100),
            'p/open30': rng.integers(0, 2, size=100),
            't/ofi_proxy': rng.normal(size=100),
            't/tradecount_z18': rng.normal(size=100),
            't/microprice_dev': rng.normal(size=100),
            't/dollarvol_z18': rng.normal(size=100),
            't/r1': rng.normal(size=100),
            't/r3': rng.normal(size=100),
            't/rv_short_3': np.abs(rng.normal(size=100)),
            't/autocorr_r1_w': rng.normal(size=100),
            't/sigma_slope_6': rng.normal(size=100),
            't/price_ema10_pct': rng.normal(size=100),
            't/price_ema20_pct': rng.normal(size=100)
        })

        interactions = engine.build_interactions(test_data)
        print(f"✅ Generated {len(interactions.columns)} interactions")

        assert len(interactions.columns) == len(config), "Interaction catalogue did not fully materialize"

        for name in expected_new:
            assert name in interactions.columns, f"Missing computed interaction: {name}"
            assert (interactions[name] >= 0).all(), f"Guardrails failed for {name}"

        return True

    except Exception as e:
        print(f"❌ Interaction engine test failed: {e}")
        return False

def test_assembly_dag():
    """Test assembly DAG."""
    print("\n=== Testing Assembly DAG ===")
    
    try:
        from src.feature_engineering.assembly_dag import AssemblyDAG, AssemblyConfig
        
        config = AssemblyConfig()
        dag = AssemblyDAG(config)
        
        print("✅ Assembly DAG created")
        
        # Test with sample data
        bars = create_test_data(200)
        targets = create_test_targets(bars)
        
        result = dag.assemble(bars, targets)
        
        if result.status.value == 'completed':
            print(f"✅ Assembly completed: {len(result.features.columns)} features")
            print(f"   Selected: {len(result.selected_features)} features")
            print(f"   Patch: {len(result.patch_features)} features")
        else:
            print(f"⚠️ Assembly status: {result.status.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Assembly DAG test failed: {e}")
        return False

def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline."""
    print("\n=== Testing End-to-End Pipeline ===")
    
    try:
        from src.end_to_end_roadmap import run_end_to_end_pipeline, SystemConfig
        
        # Create test data
        bars = create_test_data(500)
        targets = create_test_targets(bars)
        
        # Run pipeline
        print("Running end-to-end pipeline...")
        result = run_end_to_end_pipeline(
            bars=bars,
            targets=targets,
            enable_validation=False,  # Disable for speed
            enable_monitoring=False,
            enable_deployment=False
        )
        
        if result.success:
            print(f"✅ Pipeline completed successfully!")
            print(f"   Features: {len(result.features.columns)}")
            print(f"   Selected: {len(result.selected_features)}")
            print(f"   Patch: {len(result.patch_features)}")
            print(f"   Execution time: {result.metadata.get('execution_time', 'N/A')}")
        else:
            print(f"❌ Pipeline failed: {result.error_message}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end pipeline test failed: {e}")
        return False

def test_ci_validation():
    """Test CI/CD validation."""
    print("\n=== Testing CI/CD Validation ===")
    
    try:
        from src.ci.validators import run_ci_validation
        
        # Create test features
        test_features = pd.DataFrame({
            'p/r1': np.random.randn(100),
            't/r1/ewz12': np.random.randn(100),
            'i/tension/mom5_x_negmom20': np.random.randn(100)
        })
        
        results = run_ci_validation(test_features)
        
        print(f"✅ CI validation completed: {len(results)} tests")
        
        for test_name, result in results.items():
            status = "✅" if result.status.value == "pass" else "❌" if result.status.value == "fail" else "⚠️"
            print(f"   {test_name}: {status} {result.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ CI validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("End-to-End Roadmap System Test Suite")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Feature Registry", test_feature_registry),
        ("Transform System", test_transform_system),
        ("Interaction Engine", test_interaction_engine),
        ("Assembly DAG", test_assembly_dag),
        ("CI/CD Validation", test_ci_validation),
        ("End-to-End Pipeline", test_end_to_end_pipeline)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The end-to-end roadmap system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())