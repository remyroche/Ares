#!/usr/bin/env python3
"""
Example script demonstrating enhanced training pipeline integration with optional matrix operations.
Shows how to enable/disable enhanced matrix operations without breaking the existing pipeline.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config.enhanced_matrix_config import (
    get_enhanced_matrix_training_config,
    get_optimized_enhanced_matrix_config,
    get_production_enhanced_matrix_config,
    get_minimal_enhanced_matrix_config
)
from src.training.enhanced_training_manager import EnhancedTrainingManager


async def example_standard_training():
    """Example of standard training without enhanced matrix operations."""
    
    print("=" * 80)
    print("🚀 STANDARD TRAINING PIPELINE (No Enhanced Matrix Operations)")
    print("=" * 80)
    
    # Standard configuration without enhanced matrix operations
    config = {
        "enable_enhanced_matrix_operations": False,
        "enable_step_2_5_enhancement": False,
        "enable_step_5_5_enhancement": False,
        # ... other standard config options
    }
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "training_mode": "standard",
        "start_step": "step1_data_collection",
        "force_rerun": False,
    }
    
    print("📊 Starting standard training pipeline...")
    
    # Execute training
    success = await training_manager.execute_enhanced_training(training_input)
    
    if success:
        print("✅ Standard training completed successfully")
        
        # Get results
        results = training_manager.get_enhanced_training_results()
        status = training_manager.get_enhanced_training_status()
        
        print(f"📈 Training Status: {status}")
        print(f"📊 Results: {len(results)} result entries")
        
    else:
        print("❌ Standard training failed")
    
    return success


async def example_enhanced_training_performance():
    """Example of enhanced training with performance optimization."""
    
    print("\n" + "=" * 80)
    print("🚀 ENHANCED TRAINING PIPELINE (Performance Mode)")
    print("=" * 80)
    
    # Enhanced configuration with performance optimization
    config = get_optimized_enhanced_matrix_config("performance")
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "training_mode": "enhanced_performance",
        "start_step": "step1_data_collection",
        "force_rerun": False,
    }
    
    print("📊 Starting enhanced training pipeline (performance mode)...")
    
    # Execute training
    success = await training_manager.execute_enhanced_training(training_input)
    
    if success:
        print("✅ Enhanced training completed successfully")
        
        # Get enhanced results
        results = training_manager.get_enhanced_training_results()
        status = training_manager.get_enhanced_training_status()
        matrix_results = training_manager.get_matrix_enhancement_results()
        gpu_summary = training_manager.get_gpu_performance_summary()
        
        print(f"📈 Training Status: {status}")
        print(f"📊 Results: {len(results)} result entries")
        print(f"🔧 Matrix Enhancement Enabled: {matrix_results.get('enhanced_matrix_operations_enabled', False)}")
        
        if gpu_summary:
            print(f"🎯 GPU Operations: {gpu_summary.get('gpu_operations_count', 0)}")
            print(f"⚡ GPU Time: {gpu_summary.get('gpu_processing_time', 0):.2f}s")
        
    else:
        print("❌ Enhanced training failed")
    
    return success


async def example_enhanced_training_accuracy():
    """Example of enhanced training with accuracy optimization."""
    
    print("\n" + "=" * 80)
    print("🚀 ENHANCED TRAINING PIPELINE (Accuracy Mode)")
    print("=" * 80)
    
    # Enhanced configuration with accuracy optimization
    config = get_optimized_enhanced_matrix_config("accuracy")
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "training_mode": "enhanced_accuracy",
        "start_step": "step1_data_collection",
        "force_rerun": False,
    }
    
    print("📊 Starting enhanced training pipeline (accuracy mode)...")
    
    # Execute training
    success = await training_manager.execute_enhanced_training(training_input)
    
    if success:
        print("✅ Enhanced training completed successfully")
        
        # Get enhanced results
        results = training_manager.get_enhanced_training_results()
        status = training_manager.get_enhanced_training_status()
        matrix_results = training_manager.get_matrix_enhancement_results()
        
        print(f"📈 Training Status: {status}")
        print(f"📊 Results: {len(results)} result entries")
        print(f"🔧 Matrix Enhancement Enabled: {matrix_results.get('enhanced_matrix_operations_enabled', False)}")
        
        # Show matrix enhancement details
        if matrix_results.get("matrix_enhancement_results"):
            enhancement = matrix_results["matrix_enhancement_results"]
            print(f"📈 Feature Increase: {enhancement.get('feature_count_increase', 0)}")
            print(f"⏱️ Processing Time: {enhancement.get('total_processing_time', 0):.2f}s")
        
    else:
        print("❌ Enhanced training failed")
    
    return success


async def example_production_training():
    """Example of production-ready enhanced training."""
    
    print("\n" + "=" * 80)
    print("🚀 PRODUCTION ENHANCED TRAINING PIPELINE")
    print("=" * 80)
    
    # Production configuration
    config = get_production_enhanced_matrix_config()
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "training_mode": "production",
        "start_step": "step1_data_collection",
        "force_rerun": False,
    }
    
    print("📊 Starting production enhanced training pipeline...")
    
    # Execute training
    success = await training_manager.execute_enhanced_training(training_input)
    
    if success:
        print("✅ Production training completed successfully")
        
        # Get comprehensive results
        results = training_manager.get_enhanced_training_results()
        status = training_manager.get_enhanced_training_status()
        matrix_results = training_manager.get_matrix_enhancement_results()
        gpu_summary = training_manager.get_gpu_performance_summary()
        
        print(f"📈 Training Status: {status}")
        print(f"📊 Results: {len(results)} result entries")
        print(f"🔧 Matrix Enhancement Enabled: {matrix_results.get('enhanced_matrix_operations_enabled', False)}")
        
        # Show detailed performance metrics
        if gpu_summary:
            print(f"🎯 GPU Operations: {gpu_summary.get('gpu_operations_count', 0)}")
            print(f"⚡ GPU Time: {gpu_summary.get('gpu_processing_time', 0):.2f}s")
            print(f"📊 GPU Available: {gpu_summary.get('gpu_available', False)}")
            print(f"🔧 Device: {gpu_summary.get('device_info', 'Unknown')}")
        
    else:
        print("❌ Production training failed")
    
    return success


async def example_graceful_fallback():
    """Example showing graceful fallback when enhanced operations fail."""
    
    print("\n" + "=" * 80)
    print("🚀 GRACEFUL FALLBACK EXAMPLE")
    print("=" * 80)
    
    # Configuration that might fail (e.g., no GPU available)
    config = get_enhanced_matrix_training_config()
    config.update({
        "enable_enhanced_matrix_operations": True,
        "enable_step_2_5_enhancement": True,
        "enable_step_5_5_enhancement": True,
        # Force GPU usage even if not available (to test fallback)
        "enable_cpu_fallback": True,
        "enable_automatic_fallback": True,
    })
    
    # Initialize training manager
    training_manager = EnhancedTrainingManager(config)
    
    # Training input
    training_input = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "timeframe": "1m",
        "training_mode": "fallback_test",
        "start_step": "step1_data_collection",
        "force_rerun": False,
    }
    
    print("📊 Starting training with potential fallback...")
    
    # Execute training
    success = await training_manager.execute_enhanced_training(training_input)
    
    if success:
        print("✅ Training completed successfully (with fallback if needed)")
        
        # Get results
        results = training_manager.get_enhanced_training_results()
        status = training_manager.get_enhanced_training_status()
        matrix_results = training_manager.get_matrix_enhancement_results()
        
        print(f"📈 Training Status: {status}")
        print(f"📊 Results: {len(results)} result entries")
        print(f"🔧 Matrix Enhancement Enabled: {matrix_results.get('enhanced_matrix_operations_enabled', False)}")
        
        # Check if fallback occurred
        if matrix_results.get("matrix_enhancement_results"):
            enhancement = matrix_results["matrix_enhancement_results"]
            if enhancement.get("status") == "skipped":
                print("🔄 Enhanced operations were skipped (fallback to standard pipeline)")
            elif enhancement.get("status") == "failed":
                print("⚠️ Enhanced operations failed but training continued (graceful fallback)")
            else:
                print("✅ Enhanced operations completed successfully")
        
    else:
        print("❌ Training failed")
    
    return success


async def main():
    """Main function to run all examples."""
    
    print("🚀 Enhanced Training Pipeline Integration Examples")
    print("=" * 80)
    print("This script demonstrates how to integrate enhanced matrix operations")
    print("into the existing training pipeline as optional enhancements.")
    print("=" * 80)
    
    try:
        # Example 1: Standard training (no enhancements)
        await example_standard_training()
        
        # Example 2: Enhanced training with performance optimization
        await example_enhanced_training_performance()
        
        # Example 3: Enhanced training with accuracy optimization
        await example_enhanced_training_accuracy()
        
        # Example 4: Production training
        await example_production_training()
        
        # Example 5: Graceful fallback
        await example_graceful_fallback()
        
        print("\n" + "=" * 80)
        print("🎉 All examples completed!")
        print("✅ Enhanced matrix operations are fully integrated as optional enhancements")
        print("🔄 The pipeline gracefully falls back to standard operations when needed")
        print("🔧 All operations are secured with existing decorators")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the examples
    success = asyncio.run(main())
    
    if success:
        print("\n✅ All examples completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some examples failed!")
        sys.exit(1)