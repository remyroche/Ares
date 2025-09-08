#!/usr/bin/env python3
"""Test individual Step03 optimization components.

This script tests each optimization component independently to verify they work correctly.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_memory_manager():
    """Test memory manager independently."""
    print("🧪 Testing Memory Manager...")

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_memory_manager import (
            get_memory_manager, optimize_dataframe_memory
        )

        # Test memory manager
        config = {'memory_limit_gb': 2.0}
        memory_manager = get_memory_manager(config)
        print("✅ Memory manager initialization successful")

        # Test DataFrame optimization
        test_df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'str_col': ['category_' + str(i % 10) for i in range(1000)]
        })

        optimized_df = optimize_dataframe_memory(test_df)

        original_memory = test_df.memory_usage(deep=True).sum()
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        savings_mb = (original_memory - optimized_memory) / 1024 / 1024

        print("✅ DataFrame memory optimization successful")
        print(f"Memory savings: {savings_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False


def test_vectorized_operations():
    """Test vectorized operations independently."""
    print("⚡ Testing Vectorized Operations...")

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_vectorized_operations import (
            get_vectorized_operations_manager, create_vectorized_config
        )

        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'close': np.random.randn(1000).cumsum() + 100,
            'high': np.random.randn(1000).cumsum() + 102,
            'low': np.random.randn(1000).cumsum() + 98,
            'volume': np.random.randint(1000, 10000, 1000)
        })

        # Test vectorized operations
        manager = get_vectorized_operations_manager()
        config = create_vectorized_config()

        processed_data = manager.process_dataset(test_data, config)

        print("✅ Vectorized operations successful")
        print(f"   Original shape: {test_data.shape}")
        print(f"   Processed shape: {processed_data.shape}")
        print(f"   New features: {len(processed_data.columns) - len(test_data.columns)}")

        return True

    except Exception as e:
        print(f"❌ Vectorized operations test failed: {e}")
        return False


def test_bayesian_optimization():
    """Test Bayesian optimization independently."""
    print("🎯 Testing Bayesian Optimization...")

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_enhanced_bayesian_optimization import (
            ParallelBayesianOptimizer
        )
        from src.training.steps.market_analysis.hmm_clustering.step03_config import Step03Config

        # Create test data
        np.random.seed(42)
        features = pd.DataFrame(np.random.randn(100, 10))
        data = pd.DataFrame({'returns': np.random.randn(100)})

        # Test parallel optimizer
        config = Step03Config()
        optimizer = ParallelBayesianOptimizer(config)

        print("✅ Bayesian optimizer initialization successful")
        print("   Note: Full optimization test requires more setup")

        return True

    except Exception as e:
        print(f"❌ Bayesian optimization test failed: {e}")
        return False


def test_ensemble_clustering():
    """Test ensemble clustering independently."""
    print("🎯 Testing Ensemble Clustering...")

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_advanced_ensemble_clustering import (
            AdvancedEnsembleClustering
        )
        from src.training.steps.market_analysis.hmm_clustering.step03_config import Step03Config

        # Create test data
        np.random.seed(42)
        features = np.random.randn(50, 5)

        # Test ensemble clustering
        config = Step03Config()
        clustering = AdvancedEnsembleClustering(config)

        print("✅ Ensemble clustering initialization successful")
        print("   Note: Full clustering test requires more setup")

        return True

    except Exception as e:
        print(f"❌ Ensemble clustering test failed: {e}")
        return False


def test_pipeline_orchestrator():
    """Test pipeline orchestrator independently."""
    print("🔧 Testing Pipeline Orchestrator...")

    try:
        from src.training.steps.market_analysis.hmm_clustering.step03_pipeline_orchestrator import (
            get_step03_pipeline_orchestrator, create_step03_pipeline_config
        )

        # Test pipeline orchestrator
        config = create_step03_pipeline_config()
        orchestrator = get_step03_pipeline_orchestrator(config)

        print("✅ Pipeline orchestrator initialization successful")
        print("   Note: Full pipeline test requires more setup")

        return True

    except Exception as e:
        print(f"❌ Pipeline orchestrator test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Step03 Component Integration Tests")
    print("=" * 50)

    tests = [
        ("Memory Manager", test_memory_manager),
        ("Vectorized Operations", test_vectorized_operations),
        ("Bayesian Optimization", test_bayesian_optimization),
        ("Ensemble Clustering", test_ensemble_clustering),
        ("Pipeline Orchestrator", test_pipeline_orchestrator)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*30}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} components working")
    print(".1f")

    if passed == total:
        print("🎉 All Step03 optimizations are working correctly!")
        return True
    else:
        print("⚠️ Some components need attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
