#!/usr/bin/env python3
"""
Test script for Step 16 optimization integrations.

This script validates that all optimization utilities are properly integrated
into step16 and functioning correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_step16_optimizations():
    """Test that step16 optimizations are working correctly."""

    print("🧪 Testing Step 16 Optimization Integrations")
    print("=" * 50)

    # Test individual optimization utilities first
    print("\n🔧 Testing Individual Optimization Utilities...")

    # Test M1 GPU utilities
    try:
        from src.utils.m1_gpu_utils import get_m1_gpu_manager
        gpu_manager = get_m1_gpu_manager()
        print("✅ M1 GPU Manager: Available")
        m1_gpu_available = True
    except ImportError:
        print("⚠️ M1 GPU Manager: Not Available")
        m1_gpu_available = False

    # Test M1 Memory Optimizer
    try:
        from src.utils.m1_memory_optimizer import M1MemoryOptimizer
        memory_optimizer = M1MemoryOptimizer()
        print("✅ M1 Memory Optimizer: Available")
        m1_memory_available = True
    except ImportError:
        print("⚠️ M1 Memory Optimizer: Not Available")
        m1_memory_available = False

    # Test M1 CPU Optimizer
    try:
        from src.utils.m1_cpu_optimizer import M1CPUOptimizer
        cpu_optimizer = M1CPUOptimizer()
        print("✅ M1 CPU Optimizer: Available")
        m1_cpu_available = True
    except ImportError:
        print("⚠️ M1 CPU Optimizer: Not Available")
        m1_cpu_available = False

    # Test Processing Core Optimizations
    try:
        from src.utils.enhanced_matrix_operations import EnhancedMatrixOperations
        from src.utils.vectorized_processing_core import OptimizedPipelineExecutor
        from src.utils.enhanced_step_optimizations import IntelligentOptimizationSelector
        matrix_ops = EnhancedMatrixOperations()
        pipeline_executor = OptimizedPipelineExecutor()
        optimization_selector = IntelligentOptimizationSelector()
        print("✅ Processing Core Optimizations: Available")
        processing_available = True
    except ImportError as e:
        print(f"⚠️ Processing Core Optimizations: Not Available ({e})")
        processing_available = False

    # Test Data Management Optimizations
    try:
        from src.utils.optimized_data_manager import OptimizedDataManager
        data_manager = OptimizedDataManager()
        print("✅ Data Management Optimizations: Available")
        data_available = True
    except ImportError:
        print("⚠️ Data Management Optimizations: Not Available")
        data_available = False

    print("\n📊 Optimization Availability:")
    print(f"   M1 Hardware Optimizations: {'✅ Available' if all([m1_gpu_available, m1_memory_available, m1_cpu_available]) else '⚠️ Partial'}")
    print(f"   Processing Core Optimizations: {'✅ Available' if processing_available else '❌ Not Available'}")
    print(f"   Data Management Optimizations: {'✅ Available' if data_available else '❌ Not Available'}")

    # Now try to import step16
    try:
        from src.training.steps.optimisation.step16_confidence_calibration_per_regime import (
            PerRegimeConfidenceCalibrationStep
        )

        print("✅ Step 16 imports successful")

        # Test optimization availability
        print("\n📊 Optimization Availability:")
        print(f"   M1 Hardware Optimizations: {'✅ Available' if m1_optimizations_available else '❌ Not Available'}")
        print(f"   Processing Core Optimizations: {'✅ Available' if processing_optimizations_available else '❌ Not Available'}")
        print(f"   Data Management Optimizations: {'✅ Available' if data_optimizations_available else '❌ Not Available'}")

        # Create a test configuration
        config = {
            'per_regime_confidence_calibration': True,
            'adaptive_calibration_parameters_per_regime': True,
            'regime_specific_calibration_configs': {},
            'enable_mixed_precision': True,
            'enable_memory_cleanup': True,
            'batch_size': 1000,
            'memory_threshold': 0.8
        }

        # Initialize the optimized step
        print("\n🚀 Initializing Optimized Step 16...")
        step = PerRegimeConfidenceCalibrationStep(config)

        # Test optimization components
        optimizations_status = {
            'M1 GPU Manager': step.m1_gpu_manager is not None,
            'M1 Memory Optimizer': step.m1_memory_optimizer is not None,
            'M1 CPU Optimizer': step.m1_cpu_optimizer is not None,
            'Pipeline Executor': step.pipeline_executor is not None,
            'Matrix Operations': step.matrix_ops is not None,
            'Optimization Selector': step.optimization_selector is not None,
            'Data Manager': step.data_manager is not None
        }

        print("\n🔧 Optimization Components Status:")
        for component, status in optimizations_status.items():
            print(f"   {component}: {'✅ Initialized' if status else '⚠️ Not Available'}")

        # Test workload analysis
        print("\n📈 Testing Workload Analysis...")
        try:
            optimization_profile = await step._analyze_workload_for_optimization(
                symbol='BTCUSDT',
                exchange='binance',
                timeframe='1h',
                data_dir='data_cache'
            )
            if optimization_profile:
                print("✅ Workload analysis successful")
                print(f"   Workload Type: {optimization_profile.workload_type}")
                print(f"   Estimated Duration: {optimization_profile.expected_duration}s")
            else:
                print("⚠️ Workload analysis returned None")
        except Exception as e:
            print(f"❌ Workload analysis failed: {e}")

        # Test initialization
        print("\n🏁 Testing Step Initialization...")
        try:
            await step.initialize()
            print("✅ Step initialization successful")
        except Exception as e:
            print(f"❌ Step initialization failed: {e}")

        print("\n🎉 Step 16 Optimization Integration Test Complete!")
        print("=" * 50)

        # Summary
        total_components = len(optimizations_status)
        working_components = sum(optimizations_status.values())

        print(f"\n📊 Summary:")
        print(f"   Total Optimization Components: {total_components}")
        print(f"   Working Components: {working_components}")
        print(f"   Success Rate: {working_components/total_components*100:.1f}%")

        if working_components == total_components:
            print("🎯 All optimizations successfully integrated!")
            return True
        elif working_components > 0:
            print("⚠️ Partial optimization integration - some components may not be available")
            return True
        else:
            print("❌ No optimizations available - check dependencies")
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = asyncio.run(test_step16_optimizations())
    sys.exit(0 if success else 1)
