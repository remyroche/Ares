#!/usr/bin/env python3
"""
Performance optimization script for Ares feature generation.
This script implements quick fixes to reduce initialization overhead and memory pressure.
"""

import gc
import logging
import os

def configure_environment_for_speed():
    """Configure environment variables for maximum performance."""
    # Reduce memory pressure by lowering thresholds
    os.environ['PYTHONOPTIMIZE'] = '2'  # Enable Python optimizations
    
    # Configure NumPy for better performance
    os.environ['NUMPY_NUM_THREADS'] = '4'  # Limit threads for M1
    
    # Configure PyTorch for M1 if available
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Reduce logging overhead during feature generation
    logging.getLogger('src.utils.hardware.m1_memory_optimizer').setLevel(logging.WARNING)
    logging.getLogger('src.utils.hardware.unified_hardware_manager').setLevel(logging.WARNING)
    
    print("✅ Environment configured for speed")

def optimize_memory_usage():
    """Force garbage collection and optimize memory."""
    # Clear any existing caches
    gc.collect()
    
    # Clear module caches that might be bloated
    import sys
    modules_to_clear = [mod for mod in sys.modules.keys() 
                       if mod.startswith('src.feature_generation') and 'cache' in mod]
    
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    
    print(f"✅ Cleared {len(modules_to_clear)} cached modules")
    gc.collect()

def configure_feature_generation_settings():
    """Configure feature generation for optimal speed."""
    settings = {
        # Reduce feature generation overhead
        'max_features_per_category': 50,  # Limit features to reduce memory
        'enable_aggressive_caching': False,  # Disable caching to reduce memory pressure
        'memory_limit_mb': 4096,  # Lower memory limit
        'batch_size': 500,  # Smaller batches for better memory management
        'enable_gpu': False,  # Disable GPU to reduce memory overhead on M1
        'enable_parallel': True,  # Keep parallel processing
        'max_workers': 4,  # Optimize for M1 cores
    }
    
    return settings

def apply_quick_performance_fixes():
    """Apply all quick performance fixes."""
    print("🚀 Applying quick performance fixes...")
    
    # Configure environment
    configure_environment_for_speed()
    
    # Optimize memory
    optimize_memory_usage()
    
    # Get optimized settings
    settings = configure_feature_generation_settings()
    
    print("✅ Performance fixes applied")
    return settings

if __name__ == "__main__":
    settings = apply_quick_performance_fixes()
    print("\n📊 Optimized Settings:")
    for key, value in settings.items():
        print(f"   {key}: {value}")
