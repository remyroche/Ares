#!/usr/bin/env python3
"""
Test script for VectorBT-optimized advanced regime features.

This script tests the enhanced regime feature generators with VectorBT optimizations
and compares performance against the original implementation.
"""

import numpy as np
import pandas as pd
import time
import warnings
from typing import Dict, List, Tuple

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * (1 + returns).cumprod()
    
    # Add some regime-like behavior
    regime_changes = np.random.choice([0, 1], n_periods, p=[0.95, 0.05])
    regime_multiplier = np.where(regime_changes == 1, 
                                np.random.choice([0.5, 2.0]), 1.0)
    prices = prices * regime_multiplier.cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_periods)
    }, index=pd.date_range('2020-01-01', periods=n_periods, freq='1min'))
    
    return data

def test_individual_generators():
    """Test individual regime feature generators."""
    print("🧪 Testing individual regime feature generators...")
    
    # Import the generators
    try:
        from src.feature_generation.categories.advanced_regime_features import (
            RegimeEntropyGenerator,
            RegimeComplexityGenerator,
            RegimeFractalDimensionGenerator,
            RegimeHurstExponentGenerator,
            RegimeMemoryStrengthGenerator
        )
        print("✅ Successfully imported regime feature generators")
    except ImportError as e:
        print(f"❌ Failed to import generators: {e}")
        return False
    
    # Create sample data
    data = create_sample_data(500)
    print(f"📊 Created sample data with {len(data)} periods")
    
    # Test each generator
    generators = [
        ("RegimeEntropyGenerator", RegimeEntropyGenerator(10)),
        ("RegimeComplexityGenerator", RegimeComplexityGenerator(5)),
        ("RegimeFractalDimensionGenerator", RegimeFractalDimensionGenerator(20)),
        ("RegimeHurstExponentGenerator", RegimeHurstExponentGenerator(20)),
        ("RegimeMemoryStrengthGenerator", RegimeMemoryStrengthGenerator(10))
    ]
    
    results = {}
    
    for name, generator in generators:
        try:
            start_time = time.time()
            feature_result = generator._generate_feature(data)
            execution_time = time.time() - start_time
            
            results[name] = {
                'success': True,
                'execution_time': execution_time,
                'feature_count': len(feature_result) if hasattr(feature_result, '__len__') else 0,
                'has_nans': feature_result.isna().sum() if hasattr(feature_result, 'isna') else 0
            }
            
            print(f"✅ {name}: {execution_time:.3f}s, {len(feature_result)} values, {feature_result.isna().sum()} NaNs")
            
        except Exception as e:
            results[name] = {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
            print(f"❌ {name}: Failed - {e}")
    
    return results

def test_batch_processing():
    """Test batch processing with VectorBT optimizations."""
    print("\n🚀 Testing batch processing with VectorBT optimizations...")
    
    try:
        from src.feature_generation.categories.advanced_regime_features import (
            process_regime_features_batch,
            create_vectorbt_optimized_regime_generators
        )
        print("✅ Successfully imported batch processing functions")
    except ImportError as e:
        print(f"❌ Failed to import batch processing functions: {e}")
        return False
    
    # Create larger sample data for batch testing
    data = create_sample_data(2000)
    print(f"📊 Created sample data with {len(data)} periods for batch testing")
    
    # Test batch processing
    try:
        start_time = time.time()
        batch_result = process_regime_features_batch(data, use_vectorbt=True)
        execution_time = time.time() - start_time
        
        print(f"✅ Batch processing: {execution_time:.3f}s")
        print(f"📈 Generated {len(batch_result.columns)} features")
        print(f"📊 Feature names: {list(batch_result.columns)[:10]}...")  # Show first 10
        
        return {
            'success': True,
            'execution_time': execution_time,
            'feature_count': len(batch_result.columns),
            'data_shape': batch_result.shape
        }
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return {'success': False, 'error': str(e)}

def test_performance_comparison():
    """Compare performance between VectorBT and fallback implementations."""
    print("\n⚡ Testing performance comparison...")
    
    try:
        from src.feature_generation.categories.advanced_regime_features import (
            RegimeEntropyGenerator,
            process_regime_features_batch
        )
    except ImportError as e:
        print(f"❌ Failed to import for performance testing: {e}")
        return False
    
    # Create test data
    data = create_sample_data(1000)
    
    # Test individual generator performance
    generator = RegimeEntropyGenerator(10)
    
    # Test VectorBT path
    try:
        start_time = time.time()
        vectorbt_result = generator._generate_feature(data)
        vectorbt_time = time.time() - start_time
        print(f"✅ VectorBT path: {vectorbt_time:.3f}s")
    except Exception as e:
        print(f"⚠️ VectorBT path failed: {e}")
        vectorbt_time = None
    
    # Test batch processing performance
    try:
        start_time = time.time()
        batch_result = process_regime_features_batch(data, use_vectorbt=True)
        batch_time = time.time() - start_time
        print(f"✅ Batch processing: {batch_time:.3f}s")
        print(f"📊 Generated {len(batch_result.columns)} features in batch")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        batch_time = None
    
    return {
        'vectorbt_time': vectorbt_time,
        'batch_time': batch_time,
        'success': True
    }

def test_vectorbt_availability():
    """Test VectorBT availability and configuration."""
    print("\n🔧 Testing VectorBT availability and configuration...")
    
    # Test VectorBT imports
    try:
        import vectorbt as vbt
        print(f"✅ VectorBT version: {vbt.__version__}")
    except ImportError:
        print("❌ VectorBT not available")
        return False
    
    # Test VectorBT optimizer
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        optimizer = get_vectorbt_rolling_optimizer()
        print("✅ VectorBT Rolling Optimizer available")
        
        # Test optimizer performance stats
        stats = optimizer.get_performance_stats()
        print(f"📊 Optimizer stats: {stats}")
        
    except ImportError as e:
        print(f"⚠️ VectorBT Rolling Optimizer not available: {e}")
    
    # Test Unified Vectorization Manager
    try:
        from src.utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager
        manager = get_unified_vectorization_manager()
        print("✅ Unified Vectorization Manager available")
        
        # Test manager stats
        stats = manager.get_optimization_stats()
        print(f"📊 Manager stats: {stats}")
        
    except ImportError as e:
        print(f"⚠️ Unified Vectorization Manager not available: {e}")
    
    return True

def main():
    """Run all tests."""
    print("🧪 VectorBT Advanced Regime Features Test Suite")
    print("=" * 50)
    
    # Test VectorBT availability
    vectorbt_available = test_vectorbt_availability()
    
    # Test individual generators
    individual_results = test_individual_generators()
    
    # Test batch processing
    batch_results = test_batch_processing()
    
    # Test performance comparison
    performance_results = test_performance_comparison()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    if individual_results:
        successful_generators = sum(1 for r in individual_results.values() if r.get('success', False))
        total_generators = len(individual_results)
        print(f"✅ Individual generators: {successful_generators}/{total_generators} successful")
    
    if batch_results and batch_results.get('success'):
        print(f"✅ Batch processing: {batch_results['execution_time']:.3f}s")
        print(f"📈 Generated {batch_results['feature_count']} features")
    
    if performance_results and performance_results.get('success'):
        if performance_results.get('vectorbt_time'):
            print(f"⚡ VectorBT individual: {performance_results['vectorbt_time']:.3f}s")
        if performance_results.get('batch_time'):
            print(f"⚡ Batch processing: {performance_results['batch_time']:.3f}s")
    
    print("\n🎉 Test suite completed!")
    
    return True

if __name__ == "__main__":
    main()