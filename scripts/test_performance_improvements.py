#!/usr/bin/env python3
"""
Quick test script to verify performance improvements.
"""

import time
import sys

# Add the project root to the path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

def test_feature_generation_speed():
    """Test feature generation speed after optimizations."""
    print("🧪 Testing feature generation speed...")
    
    # Apply optimizations
    from scripts.optimize_feature_generation_speed import apply_quick_performance_fixes
    apply_quick_performance_fixes()
    
    # Test importing feature generation components
    start_time = time.time()
    
    try:
        print("📦 Importing feature generation components...")
        from src.feature_generation.core.feature_bank import FeatureBank
        
        # Initialize with reduced settings
        print("🏦 Initializing FeatureBank...")
        feature_bank = FeatureBank()
        
        init_time = time.time() - start_time
        print(f"✅ FeatureBank initialized in {init_time:.2f}s")
        
        # Test feature generation with small dataset
        import pandas as pd
        import numpy as np
        
        print("📊 Creating test data...")
        test_data = pd.DataFrame({
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 110,
            'low': np.random.randn(100) * 10 + 90,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        print("🚀 Testing feature generation...")
        feature_start = time.time()
        
        # Generate a small set of features
        features = feature_bank.generate_features(
            test_data, 
            categories=['returns', 'momentum'],
            max_features_per_category=5
        )
        
        feature_time = time.time() - feature_start
        total_time = time.time() - start_time
        
        print(f"✅ Generated {len(features.columns)} features in {feature_time:.2f}s")
        print(f"📊 Total time: {total_time:.2f}s")
        print(f"🎯 Performance: {len(features.columns)/feature_time:.1f} features/second")
        
        return {
            'init_time': init_time,
            'feature_time': feature_time,
            'total_time': total_time,
            'features_generated': len(features.columns),
            'features_per_second': len(features.columns)/feature_time
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

if __name__ == "__main__":
    results = test_feature_generation_speed()
    
    if results:
        print("\n🎉 Performance Test Results:")
        for key, value in results.items():
            print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")
