#!/usr/bin/env python3
"""
Test script for Layer 2 optimizations
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append('/Users/remyroche/Documents/Ares')

def test_optimizations():
    """Test all optimized functions."""
    print("🚀 Testing Layer 2 Optimizations...")
    
    try:
        # Import optimized functions
        from src.training.steps.labeling.optimized_layer2_functions import (
            vectorized_feature_selection,
            batch_model_training,
            vectorized_geometry_search,
            jit_feature_engineering,
            benchmark_optimizations
        )
        
        print("✅ Successfully imported optimized functions")
        
        # Run benchmark
        print("\n📊 Running benchmark...")
        benchmark_optimizations()
        
        print("\n🎉 All optimizations working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimizations()
    sys.exit(0 if success else 1)
