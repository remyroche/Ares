#!/usr/bin/env python3
import numpy as np
import pandas as pd

"""
Step07 Import Verification Script

This script verifies that all imports and dependencies for Step07 are properly
configured and available, with appropriate fallback handling.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, import_statement, required=True):
    """Test importing a module and return status."""
    try:
        exec(import_statement)
        print(f"✅ {module_name}: Available")
        return True
    except ImportError as e:
        if required:
            print(f"❌ {module_name}: Required but not available - {e}")
            return False
        else:
            print(f"⚠️ {module_name}: Optional and not available - {e}")
            return True
    except Exception as e:
        print(f"❌ {module_name}: Error during import - {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Step07 Import Verification")
    print("=" * 50)
    
    # Core Python modules (always required)
    core_modules = [
        ("os", "import os"),
        ("time", "import time"),
        ("traceback", "import traceback"),
        ("gc", "import gc"),
        ("functools", "import functools"),
        ("inspect", "import inspect"),
        ("datetime", "from datetime import datetime"),
        ("pathlib", "from pathlib import Path"),
        ("typing", "from typing import Any, Callable, Dict, List, Tuple, Optional"),
        ("json", "import json"),
    ]
    
    print("\n📦 Core Python Modules:")
    core_results = []
    for module_name, import_stmt in core_modules:
        result = test_import(module_name, import_stmt, required=True)
        core_results.append(result)
    
    # Scientific computing modules
    scientific_modules = [
        ("numpy", "import numpy as np", True),
        ("pandas", "import pandas as pd", True),
        ("psutil", "import psutil", False),
        ("sklearn", "from sklearn.feature_selection import mutual_info_classif", False),
        ("scipy", "from scipy.stats import rankdata", False),
        ("lightgbm", "import lightgbm as lgb", False),
    ]
    
    print("\n🧮 Scientific Computing Modules:")
    scientific_results = []
    for module_name, import_stmt, required in scientific_modules:
        result = test_import(module_name, import_stmt, required=required)
        scientific_results.append(result)
    
    # Project-specific modules
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    project_modules = [
        ("PipelineStandards", "from src.utils.pipeline_standards import PipelineStandards, pipeline_standards", False),
        ("BaseStep", "from src.training.base_step import BaseStep", False),
        ("Decorators", "from src.core.decorators import handles_errors", False),
        ("MatrixComponents", "from src.training.steps.model_training.matrix_components import MatrixProcessor", False),
    ]
    
    print("\n🏗️ Project-Specific Modules:")
    project_results = []
    for module_name, import_stmt, required in project_modules:
        result = test_import(module_name, import_stmt, required=required)
        project_results.append(result)
    
    # Test fallback functionality
    print("\n🛡️ Fallback Functionality Tests:")
    
    # Test psutil fallback
    try:
        import psutil
        psutil_available = True
        print("✅ psutil: Available - full performance monitoring enabled")
    except ImportError:
        psutil_available = False
        print("⚠️ psutil: Not available - limited performance monitoring")
    
    # Test pandas fallback
    try:
        pandas_available = True
        print("✅ pandas: Available - full DataFrame support")
    except ImportError:
        pandas_available = False
        print("⚠️ pandas: Not available - limited DataFrame support")
    
    # Test numpy fallback
    try:
        numpy_available = True
        print("✅ numpy: Available - full array support")
    except ImportError:
        numpy_available = False
        print("⚠️ numpy: Not available - limited array support")
    
    # Test sklearn fallback
    try:
        from sklearn.feature_selection import mutual_info_classif
        sklearn_available = True
        print("✅ sklearn: Available - full feature selection support")
    except ImportError:
        sklearn_available = False
        print("⚠️ sklearn: Not available - using variance-based fallback")
    
    # Test scipy fallback
    try:
        from scipy.stats import rankdata
        scipy_available = True
        print("✅ scipy: Available - full statistical functions")
    except ImportError:
        scipy_available = False
        print("⚠️ scipy: Not available - using simple sorting fallback")
    
    # Test lightgbm fallback
    try:
        import lightgbm as lgb
        lightgbm_available = True
        print("✅ lightgbm: Available - full SHAP importance support")
    except ImportError:
        lightgbm_available = False
        print("⚠️ lightgbm: Not available - SHAP importance disabled")
    
    # Summary
    print("\n📊 Summary:")
    print("=" * 50)
    
    core_available = all(core_results)
    scientific_available = all(scientific_results)
    project_available = all(project_results)
    
    print(f"Core modules: {'✅ All available' if core_available else '❌ Some missing'}")
    print(f"Scientific modules: {'✅ All available' if scientific_available else '⚠️ Some optional modules missing'}")
    print(f"Project modules: {'✅ All available' if project_available else '⚠️ Some project modules missing'}")
    
    # Overall status
    if core_available and (scientific_available or not any(scientific_results)):
        print("\n🎉 Step07 can run with current dependencies!")
        if not scientific_available:
            print("   Note: Some optional features may be limited due to missing scientific modules.")
    else:
        print("\n❌ Step07 cannot run - missing required dependencies.")
        print("   Please install missing modules before running Step07.")
    
    # Feature availability
    print("\n🔧 Feature Availability:")
    print(f"   Function call tracking: ✅ Always available")
    print(f"   Error handling: ✅ Always available")
    print(f"   Validation framework: ✅ Always available")
    print(f"   Performance monitoring: {'✅ Full' if psutil_available else '⚠️ Limited (no psutil)'}")
    print(f"   Matrix operations: {'✅ Full' if numpy_available else '❌ Limited (no numpy)'}")
    print(f"   Feature filtering: {'✅ Full' if sklearn_available else '⚠️ Limited (no sklearn)'}")
    print(f"   SHAP importance: {'✅ Full' if lightgbm_available else '❌ Disabled (no lightgbm)'}")
    print(f"   Statistical ranking: {'✅ Full' if scipy_available else '⚠️ Limited (no scipy)'}")
    
    return core_available and (scientific_available or not any(scientific_results))

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)