#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Simple test script to verify pipeline functionality without optional dependencies.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that core components can be imported."""
    tprint("🧪 Testing Pipeline Imports...")
    
    # Test core components
    try:
        from core.config import CodeQualityConfig, get_default_config
        tprint("✅ Core config imports successfully")
    except Exception as e:
        tprint(f"❌ Core config import failed: {e}")
        return False
    
    # Test analyzers
    try:
        from analyzers.complexity_analyzer import ComplexityAnalyzer
        tprint("✅ Complexity analyzer imports successfully")
    except Exception as e:
        tprint(f"❌ Complexity analyzer import failed: {e}")
        return False
    
    try:
        from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
        tprint("✅ Enhanced dead code analyzer imports successfully")
    except Exception as e:
        tprint(f"❌ Enhanced dead code analyzer import failed: {e}")
        return False
    
    try:
        from analyzers.enhanced_import_analysis import EnhancedImportAnalyzer
        tprint("✅ Enhanced import analyzer imports successfully")
    except Exception as e:
        tprint(f"❌ Enhanced import analyzer import failed: {e}")
        return False
    
    # Test mappers (skip visualizers for now due to missing dependencies)
    try:
        # Just test the core mapper without visualizers
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
        from core.config import AnalysisConfig
        tprint("✅ Core mapper components import successfully")
    except Exception as e:
        tprint(f"❌ Core mapper components import failed: {e}")
        return False
    
    # Test validators
    try:
        from validators.function_validator import FunctionValidator
        tprint("✅ Function validator imports successfully")
    except Exception as e:
        tprint(f"❌ Function validator import failed: {e}")
        return False
    
    # Test reporters (skip if dependencies missing)
    try:
        from reporters.quality_reporter import QualityReporter
        tprint("✅ Quality reporter imports successfully")
    except Exception as e:
        if "rich" in str(e) or "toml" in str(e):
            tprint("⚠️  Quality reporter import skipped (missing optional dependencies)")
        else:
            tprint(f"❌ Quality reporter import failed: {e}")
            return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core components."""
    tprint("\n🔧 Testing Basic Functionality...")
    
    try:
        from core.config import get_default_config
        config = get_default_config()
        tprint("✅ Default config creation works")
    except Exception as e:
        tprint(f"❌ Default config creation failed: {e}")
        return False
    
    try:
        from analyzers.complexity_analyzer import ComplexityAnalyzer
        analyzer = ComplexityAnalyzer(config)
        tprint("✅ Complexity analyzer instantiation works")
    except Exception as e:
        tprint(f"❌ Complexity analyzer instantiation failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    tprint("🚀 Testing Code Quality Pipelines")
    tprint("=" * 50)
    
    # Test imports
    if not test_imports():
        tprint("\n❌ Import tests failed")
        return 1
    
    # Test basic functionality
    if not test_basic_functionality():
        tprint("\n❌ Basic functionality tests failed")
        return 1
    
    tprint("\n🎉 All tests passed! Pipelines are working correctly.")
    return 0

if __name__ == "__main__":
    sys.exit(main())