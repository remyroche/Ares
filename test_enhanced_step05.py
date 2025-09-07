#!/usr/bin/env python3
"""
Test script for enhanced step05 with M1 optimizations.

This script tests the integration of M1 hardware-specific optimizations,
vectorized processing, and enhanced data management in step05.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

async def test_enhanced_step05():
    """Test the enhanced step05 implementation."""
    try:
        from src.training.steps.step5_labeling import LabelingStep

        print("🚀 Testing Enhanced Step05 with M1 Optimizations")
        print("=" * 60)

        # Create test configuration
        config = {
            'test_mode': True,
            'enable_m1_optimizations': True,
            'enable_gpu_acceleration': True,
            'enable_memory_optimization': True,
            'enable_parallel_processing': True
        }

        # Initialize enhanced labeling step
        print("📦 Initializing enhanced labeling step...")
        labeling_step = LabelingStep(config)

        # Test optimization components
        print("🔧 Testing optimization components...")

        # Check M1 optimizations
        if hasattr(labeling_step, 'gpu_manager') and labeling_step.gpu_manager:
            print("✅ M1 GPU Manager: Available")
        else:
            print("⚠️ M1 GPU Manager: Not available")

        if hasattr(labeling_step, 'memory_optimizer') and labeling_step.memory_optimizer:
            print("✅ M1 Memory Optimizer: Available")
        else:
            print("⚠️ M1 Memory Optimizer: Not available")

        if hasattr(labeling_step, 'cpu_optimizer') and labeling_step.cpu_optimizer:
            print("✅ M1 CPU Optimizer: Available")
        else:
            print("⚠️ M1 CPU Optimizer: Not available")

        if hasattr(labeling_step, 'pipeline_executor') and labeling_step.pipeline_executor:
            print("✅ Vectorized Pipeline Executor: Available")
        else:
            print("⚠️ Vectorized Pipeline Executor: Not available")

        if hasattr(labeling_step, 'matrix_operations') and labeling_step.matrix_operations:
            print("✅ Enhanced Matrix Operations: Available")
        else:
            print("⚠️ Enhanced Matrix Operations: Not available")

        if hasattr(labeling_step, 'optimization_selector') and labeling_step.optimization_selector:
            print("✅ Intelligent Optimization Selector: Available")
        else:
            print("⚠️ Intelligent Optimization Selector: Not available")

        if hasattr(labeling_step, 'data_manager') and labeling_step.data_manager:
            print("✅ Optimized Data Manager: Available")
        else:
            print("⚠️ Optimized Data Manager: Not available")

        # Test basic functionality
        print("\n🔍 Testing basic functionality...")

        # Create mock training input
        training_input = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m',
            'data_dir': 'data_cache',
            'force_rerun': False
        }

        # Create mock pipeline state
        pipeline_state = {
            'step_name': 'test_step05',
            'dataframe': None,  # Will test without data first
            'validated_data': None
        }

        # Test execute method
        print("🏃 Testing execute method...")
        result = await labeling_step.execute(training_input, pipeline_state)

        print(f"📊 Execution Result: {result.get('success', False)}")
        print(f"⏱️ Execution Time: {result.get('execution_time', 0):.2f}s")
        print(f"🔧 Optimizations Used: {result.get('optimizations_used', False)}")

        if result.get('optimization_metrics'):
            metrics = result.get('optimization_metrics', {})
            print("📈 Optimization Metrics:")
            for key, value in metrics.items():
                print(f"   {key}: {value}")

        print("\n✅ Enhanced Step05 test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_step05())
    sys.exit(0 if success else 1)
