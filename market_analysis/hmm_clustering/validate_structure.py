#!/usr/bin/env python3
"""
Structure validation script for Enhanced HMM Clustering.

This script validates the structure and basic functionality without
requiring external dependencies.
"""

import os
import sys
from pathlib import Path

def validate_file_structure():
    """Validate that all required files exist."""
    print("Validating file structure...")
    
    required_files = [
        "enhanced_hmm_clustering.py",
        "config.py", 
        "example_usage.py",
        "integration_example.py",
        "test_implementation.py",
        "README.md",
        "__init__.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
        else:
            print(f"✓ {file}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    print("✓ All required files present")
    return True

def validate_imports():
    """Validate that imports work correctly."""
    print("\nValidating imports...")
    
    try:
        # Test basic Python imports
        import json
        import logging
        import datetime
        from dataclasses import dataclass
        from enum import Enum
        print("✓ Basic Python imports working")
        
        # Test module structure
        sys.path.append(str(Path(__file__).parent))
        
        # Test config module
        try:
            from config import HMMClusteringConfig, HMMClusteringConfigFactory
            print("✓ Config module imports working")
        except Exception as e:
            print(f"✗ Config module import failed: {e}")
            return False
        
        # Test enhanced_hmm_clustering module (without external deps)
        try:
            # Read the file and check for basic structure
            with open("enhanced_hmm_clustering.py", "r") as f:
                content = f.read()
                
            required_classes = [
                "class HMMClusteringConfig",
                "class HMMClusteringResult", 
                "class EnhancedHMMClustering",
                "class RegimeType"
            ]
            
            for class_def in required_classes:
                if class_def in content:
                    print(f"✓ {class_def} found")
                else:
                    print(f"✗ {class_def} not found")
                    return False
                    
        except Exception as e:
            print(f"✗ Enhanced HMM clustering module validation failed: {e}")
            return False
        
        print("✓ All imports validated")
        return True
        
    except Exception as e:
        print(f"✗ Import validation failed: {e}")
        return False

def validate_configuration_system():
    """Validate configuration system."""
    print("\nValidating configuration system...")
    
    try:
        from config import (
            HMMClusteringConfig,
            HMMClusteringConfigFactory,
            ConfigValidator,
            get_config_by_name,
            create_custom_config
        )
        
        # Test basic configuration creation
        config = HMMClusteringConfig()
        assert config.n_components > 0
        assert len(config.lookback_windows) > 0
        print("✓ Basic configuration creation working")
        
        # Test factory methods
        crypto_config = HMMClusteringConfigFactory.create_crypto_config()
        assert crypto_config.n_components > 0
        print("✓ Configuration factory working")
        
        # Test preset retrieval
        preset_config = get_config_by_name("crypto_btc_1h")
        assert preset_config is not None
        print("✓ Preset configuration retrieval working")
        
        # Test custom configuration
        custom_config = create_custom_config(n_components=5)
        assert custom_config.n_components == 5
        print("✓ Custom configuration creation working")
        
        # Test validation
        validator = ConfigValidator()
        warnings = validator.validate_config(config)
        print(f"✓ Configuration validation working (warnings: {len(warnings)})")
        
        print("✓ Configuration system validated")
        return True
        
    except Exception as e:
        print(f"✗ Configuration system validation failed: {e}")
        return False

def validate_code_structure():
    """Validate code structure and patterns."""
    print("\nValidating code structure...")
    
    try:
        # Check enhanced_hmm_clustering.py
        with open("enhanced_hmm_clustering.py", "r") as f:
            content = f.read()
        
        # Check for required methods
        required_methods = [
            "def load_market_data",
            "def engineer_features", 
            "def select_features",
            "def fit_hmm_model",
            "def predict_regimes",
            "def save_model",
            "def load_model"
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✓ {method} found")
            else:
                print(f"✗ {method} not found")
                return False
        
        # Check for technical indicator methods
        indicator_methods = [
            "def _calculate_rsi",
            "def _calculate_macd",
            "def _calculate_bollinger_bands",
            "def _calculate_atr",
            "def _calculate_stochastic"
        ]
        
        for method in indicator_methods:
            if method in content:
                print(f"✓ {method} found")
            else:
                print(f"✗ {method} not found")
                return False
        
        # Check for common utilities integration
        utility_imports = [
            "from src.utils.common_operations",
            "from src.utils.common_utilities",
            "from src.utils.math_validation",
            "from src.utils.data.klines_parquet",
            "from src.utils.serialization_utils"
        ]
        
        for import_line in utility_imports:
            if import_line in content:
                print(f"✓ {import_line} found")
            else:
                print(f"✗ {import_line} not found")
                return False
        
        print("✓ Code structure validated")
        return True
        
    except Exception as e:
        print(f"✗ Code structure validation failed: {e}")
        return False

def validate_documentation():
    """Validate documentation completeness."""
    print("\nValidating documentation...")
    
    try:
        # Check README.md
        if Path("README.md").exists():
            with open("README.md", "r") as f:
                readme_content = f.read()
            
            required_sections = [
                "# Enhanced HMM Clustering",
                "## Features",
                "## Installation", 
                "## Quick Start",
                "## Configuration Options",
                "## Examples"
            ]
            
            for section in required_sections:
                if section in readme_content:
                    print(f"✓ {section} found in README")
                else:
                    print(f"✗ {section} not found in README")
                    return False
        
        # Check docstrings in main module
        with open("enhanced_hmm_clustering.py", "r") as f:
            content = f.read()
        
        if '"""' in content and 'class EnhancedHMMClustering' in content:
            print("✓ Main class has docstring")
        else:
            print("✗ Main class missing docstring")
            return False
        
        print("✓ Documentation validated")
        return True
        
    except Exception as e:
        print(f"✗ Documentation validation failed: {e}")
        return False

def run_validation():
    """Run all validation checks."""
    print("Enhanced HMM Clustering Structure Validation")
    print("=" * 50)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports),
        ("Configuration System", validate_configuration_system),
        ("Code Structure", validate_code_structure),
        ("Documentation", validate_documentation)
    ]
    
    passed = 0
    total = len(validations)
    
    for validation_name, validation_func in validations:
        print(f"\n--- {validation_name} ---")
        try:
            if validation_func():
                passed += 1
                print(f"✓ {validation_name} validation passed")
            else:
                print(f"✗ {validation_name} validation failed")
        except Exception as e:
            print(f"✗ {validation_name} validation failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 All validations passed! The HMM clustering structure is correct.")
        return True
    else:
        print(f"❌ {total - passed} validations failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)