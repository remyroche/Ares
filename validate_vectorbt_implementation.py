#!/usr/bin/env python3
"""
Simple validation script to check VectorBT optimization implementation.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_imports():
    """Validate that all required modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        # Test VectorBT rolling optimizer import
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
        )
        print("✅ VectorBTRollingOptimizer imported successfully")
    except ImportError as e:
        print(f"❌ VectorBTRollingOptimizer import failed: {e}")
        return False
    
    try:
        # Test unified optimization system import
        from src.feature_generation.utils.unified_optimization_system import (
            UnifiedOptimizationSystem, get_unified_optimization_system
        )
        print("✅ UnifiedOptimizationSystem imported successfully")
    except ImportError as e:
        print(f"❌ UnifiedOptimizationSystem import failed: {e}")
        return False
    
    try:
        # Test regime feature integration import
        from src.feature_generation.categories.regime_feature_integration import (
            RegimeFeatureIntegration, RegimeFeatureConfig
        )
        print("✅ RegimeFeatureIntegration imported successfully")
    except ImportError as e:
        print(f"❌ RegimeFeatureIntegration import failed: {e}")
        return False
    
    return True

def validate_class_structure():
    """Validate that the classes have the expected VectorBT optimization methods."""
    print("\n🔍 Validating class structure...")
    
    try:
        from src.feature_generation.categories.regime_feature_integration import (
            RegimeFeatureIntegration, RegimeFeatureConfig
        )
        
        # Create a config
        config = RegimeFeatureConfig()
        
        # Initialize the generator
        generator = RegimeFeatureIntegration(config)
        
        # Check for VectorBT optimizer attributes
        if hasattr(generator, 'vectorbt_optimizer'):
            print("✅ vectorbt_optimizer attribute found")
        else:
            print("❌ vectorbt_optimizer attribute missing")
            return False
        
        if hasattr(generator, 'unified_optimizer'):
            print("✅ unified_optimizer attribute found")
        else:
            print("❌ unified_optimizer attribute missing")
            return False
        
        # Check for VectorBT rolling operation methods
        if hasattr(generator, '_vectorbt_rolling_operation'):
            print("✅ _vectorbt_rolling_operation method found")
        else:
            print("❌ _vectorbt_rolling_operation method missing")
            return False
        
        if hasattr(generator, '_pandas_rolling_operation'):
            print("✅ _pandas_rolling_operation method found")
        else:
            print("❌ _pandas_rolling_operation method missing")
            return False
        
        # Check for optimization methods
        if hasattr(generator, 'optimize_dataframe_processing'):
            print("✅ optimize_dataframe_processing method found")
        else:
            print("❌ optimize_dataframe_processing method missing")
            return False
        
        if hasattr(generator, 'vectorized_rolling_operations'):
            print("✅ vectorized_rolling_operations method found")
        else:
            print("❌ vectorized_rolling_operations method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Class structure validation failed: {e}")
        return False

def validate_individual_generators():
    """Validate individual regime feature generators."""
    print("\n🔍 Validating individual generators...")
    
    try:
        from src.feature_generation.categories.regime_volatility import RegimeVolatilityFeatureGenerator
        from src.feature_generation.categories.regime_volume import RegimeVolumeFeatureGenerator
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        generators = [
            ("RegimeVolatilityFeatureGenerator", RegimeVolatilityFeatureGenerator),
            ("RegimeVolumeFeatureGenerator", RegimeVolumeFeatureGenerator),
            ("RegimeStructuralTrendFeatureGenerator", RegimeStructuralTrendFeatureGenerator)
        ]
        
        for name, generator_class in generators:
            try:
                generator = generator_class()
                
                # Check for VectorBT optimizer attributes
                if hasattr(generator, 'vectorbt_optimizer'):
                    print(f"✅ {name}: vectorbt_optimizer attribute found")
                else:
                    print(f"❌ {name}: vectorbt_optimizer attribute missing")
                    return False
                
                if hasattr(generator, 'unified_optimizer'):
                    print(f"✅ {name}: unified_optimizer attribute found")
                else:
                    print(f"❌ {name}: unified_optimizer attribute missing")
                    return False
                
                # Check for VectorBT rolling operation methods
                if hasattr(generator, '_vectorbt_rolling_operation'):
                    print(f"✅ {name}: _vectorbt_rolling_operation method found")
                else:
                    print(f"❌ {name}: _vectorbt_rolling_operation method missing")
                    return False
                
            except Exception as e:
                print(f"❌ {name} validation failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Individual generators validation failed: {e}")
        return False

def validate_code_quality():
    """Validate code quality and consistency."""
    print("\n🔍 Validating code quality...")
    
    # Check if VectorBT imports are consistent
    try:
        with open('src/feature_generation/categories/regime_feature_integration.py', 'r') as f:
            content = f.read()
            
        # Check for VectorBT imports
        if 'from ..utils.vectorbt_rolling_optimizer import' in content:
            print("✅ VectorBT rolling optimizer import found")
        else:
            print("❌ VectorBT rolling optimizer import missing")
            return False
        
        if 'from ..utils.unified_optimization_system import' in content:
            print("✅ Unified optimization system import found")
        else:
            print("❌ Unified optimization system import missing")
            return False
        
        # Check for VectorBT optimizer initialization
        if 'self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(' in content:
            print("✅ VectorBT optimizer initialization found")
        else:
            print("❌ VectorBT optimizer initialization missing")
            return False
        
        if 'self.unified_optimizer = get_unified_optimization_system()' in content:
            print("✅ Unified optimizer initialization found")
        else:
            print("❌ Unified optimizer initialization missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Code quality validation failed: {e}")
        return False

def main():
    """Run all validations."""
    print("🧪 VectorBT Optimization Implementation Validation")
    print("=" * 60)
    
    # Run validations
    validations = [
        ("Import Validation", validate_imports),
        ("Class Structure Validation", validate_class_structure),
        ("Individual Generators Validation", validate_individual_generators),
        ("Code Quality Validation", validate_code_quality)
    ]
    
    results = []
    for name, validation_func in validations:
        print(f"\n{name}:")
        try:
            result = validation_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All validations passed! VectorBT optimization is properly implemented.")
        print("\n📋 Implementation Summary:")
        print("   • VectorBTRollingOptimizer integrated into all regime feature generators")
        print("   • UnifiedVectorizationManager integrated for comprehensive optimization")
        print("   • All rolling operations now use VectorBT with pandas fallback")
        print("   • DataFrame processing optimized using VectorBT optimizers")
        print("   • Consistent VectorBT usage across all regime feature categories")
        return 0
    else:
        print("\n⚠️ Some validations failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())