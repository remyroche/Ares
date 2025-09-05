#!/usr/bin/env python3
import numpy as np
import pandas as pd

"""
Setup Script for Step06 Enhanced Validation Framework

This script sets up the step06 validation framework by:
1. Installing required dependencies
2. Setting up proper import paths
3. Validating the installation
4. Running basic tests
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements_step06_validation.txt"
    
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            return False
    else:
        print("⚠️ Requirements file not found, skipping installation")
        return True


def setup_import_paths():
    """Set up proper import paths."""
    print("🔧 Setting up import paths...")
    
    # Add src directory to Python path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    
    if src_dir.exists():
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        print(f"✅ Added {src_dir} to Python path")
    else:
        print(f"⚠️ Source directory not found: {src_dir}")
    
    # Add training steps directory
    training_steps_dir = src_dir / "training" / "steps"
    if training_steps_dir.exists():
        if str(training_steps_dir) not in sys.path:
            sys.path.insert(0, str(training_steps_dir))
        print(f"✅ Added {training_steps_dir} to Python path")
    else:
        print(f"⚠️ Training steps directory not found: {training_steps_dir}")


def validate_imports():
    """Validate that all imports work correctly."""
    print("🔍 Validating imports...")
    
    import_tests = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("logging", "Logging"),
        ("asyncio", "Async support"),
        ("json", "JSON handling"),
        ("datetime", "Date/time handling"),
        ("pathlib", "Path handling"),
        ("typing", "Type hints"),
        ("dataclasses", "Data classes"),
        ("enum", "Enumerations"),
        ("threading", "Threading"),
        ("contextlib", "Context management")
    ]
    
    failed_imports = []
    
    for module_name, description in import_tests:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name} ({description})")
        except ImportError as e:
            print(f"❌ {module_name} ({description}): {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"⚠️ Failed to import: {failed_imports}")
        return False
    else:
        print("✅ All core imports successful")
        return True


def test_step06_validation_framework():
    """Test the step06 validation framework."""
    print("🧪 Testing step06 validation framework...")
    
    try:
        # Test validation framework import
        from src.training.steps.step06_enhanced_validation_framework import (
            step06_function_validator,
            step06_function_tracker,
            ValidationLevel,
            FunctionStatus
        )
        print("✅ Step06 validation framework imported successfully")
        
        # Test decorators
        @step06_function_validator(function_type="test", validation_level=ValidationLevel.BASIC)
        def test_function():
            return "test_result"
        
        result = test_function()
        if result == "test_result":
            print("✅ Step06 function validator decorator working")
        else:
            print("❌ Step06 function validator decorator failed")
            return False
        
        # Test validation levels
        levels = [ValidationLevel.BASIC, ValidationLevel.DETAILED, ValidationLevel.COMPREHENSIVE]
        print(f"✅ Validation levels available: {[level.value for level in levels]}")
        
        # Test function status
        statuses = [FunctionStatus.PENDING, FunctionStatus.IN_PROGRESS, FunctionStatus.COMPLETED, FunctionStatus.FAILED]
        print(f"✅ Function statuses available: {[status.value for status in statuses]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import step06 validation framework: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing step06 validation framework: {e}")
        return False


def test_step06_components():
    """Test step06 components."""
    print("🧪 Testing step06 components...")
    
    components_to_test = [
        ("src.training.steps.market_analysis.step06_feature_engineering", "FeatureInteractionEngine"),
        ("src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling", "OptimizedTripleBarrierLabeling"),
        ("src.training.steps.data_collection.feature_engineering.step06_feature_engineering", "FeatureEngineeringStep")
    ]
    
    successful_components = 0
    
    for module_path, class_name in components_to_test:
        try:
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
            successful_components += 1
        except ImportError as e:
            print(f"❌ Failed to import {class_name}: {e}")
        except AttributeError as e:
            print(f"❌ {class_name} not found in module: {e}")
        except Exception as e:
            print(f"❌ Error testing {class_name}: {e}")
    
    print(f"📊 Component test results: {successful_components}/{len(components_to_test)} successful")
    return successful_components == len(components_to_test)


def create_test_data():
    """Create test data for validation."""
    print("📊 Creating test data...")
    
    try:
        
        # Generate test data
        np.random.seed(42)
        n_samples = 1000
        
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="1min")
        
        data = pd.DataFrame({
            "open": np.random.uniform(100, 110, n_samples),
            "high": np.random.uniform(105, 115, n_samples),
            "low": np.random.uniform(95, 105, n_samples),
            "close": np.random.uniform(100, 110, n_samples),
            "volume": np.random.uniform(1000, 10000, n_samples)
        }, index=dates)
        
        # Ensure OHLC consistency
        data["high"] = np.maximum(data["high"], np.maximum(data["open"], data["close"]))
        data["low"] = np.minimum(data["low"], np.minimum(data["open"], data["close"]))
        
        # Save test data
        test_data_path = Path(__file__).parent / "test_data_step06.parquet"
        data.to_parquet(test_data_path)
        
        print(f"✅ Test data created: {test_data_path}")
        print(f"   Shape: {data.shape}")
        print(f"   Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create test data: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Step06 Enhanced Validation Framework")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Setup import paths
    setup_import_paths()
    
    # Validate imports
    if not validate_imports():
        print("❌ Setup failed at import validation")
        return False
    
    # Test validation framework
    if not test_step06_validation_framework():
        print("❌ Setup failed at validation framework test")
        return False
    
    # Test components
    if not test_step06_components():
        print("❌ Setup failed at component test")
        return False
    
    # Create test data
    if not create_test_data():
        print("❌ Setup failed at test data creation")
        return False
    
    print("=" * 60)
    print("🎉 Step06 Enhanced Validation Framework setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the comprehensive validation test: python test_step06_comprehensive_validation.py")
    print("2. Check the validation reports in: step06_validation_reports/")
    print("3. Review the summary document: STEP06_COMPREHENSIVE_VALIDATION_SUMMARY.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)