#!/usr/bin/env python3
"""
Simple test script for basic imports only
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports without loading the entire system."""
    print("Testing basic imports...")
    
    try:
        # Test only the core imports we need
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            MarkovRegressionAdapter,
            MarkovRegressionConfig,
            MarkovRegressionResult,
            ParameterMapper,
            MarkovRegressionDiagnostics,
            create_enhanced_markov_regression_adapter
        )
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run basic import test."""
    print("🧪 Running basic import test...\n")
    
    result = test_basic_imports()
    
    if result:
        print("🎉 Basic import test passed!")
        return 0
    else:
        print("❌ Basic import test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)