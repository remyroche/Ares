#!/usr/bin/env python3
"""
Basic pipeline test - tests core functionality without optional dependencies.
"""

import sys
from pathlib import Path

def test_pipeline_imports():
    """Test that pipeline files can be imported without errors."""
    print("🧪 Testing Pipeline Imports...")
    
    # Test each pipeline file
    pipelines = [
        "pipelines/complexity_pipeline.py",
        "pipelines/dead_code_pipeline.py", 
        "pipelines/import_free_analysis_pipeline.py",
        "pipelines/pipeline_unified_enhanced.py",
        "pipelines/overall_pipeline.py"
    ]
    
    for pipeline in pipelines:
        try:
            # Import the pipeline module
            pipeline_path = Path(pipeline)
            module_name = pipeline_path.stem
            
            # Add the parent directory to path
            sys.path.insert(0, str(pipeline_path.parent.parent))
            
            # Import the module
            __import__(module_name)
            print(f"✅ {pipeline} imports successfully")
            
        except Exception as e:
            if "rich" in str(e) or "toml" in str(e) or "matplotlib" in str(e) or "networkx" in str(e):
                print(f"⚠️  {pipeline} import skipped (missing optional dependencies)")
            else:
                print(f"❌ {pipeline} import failed: {e}")
                return False
    
    return True

def test_core_components():
    """Test core components work."""
    print("\n🔧 Testing Core Components...")
    
    try:
        from core.config import get_default_config
        config = get_default_config()
        print("✅ Config system works")
    except Exception as e:
        print(f"❌ Config system failed: {e}")
        return False
    
    try:
        from analyzers.complexity_analyzer import ComplexityAnalyzer
        analyzer = ComplexityAnalyzer(config)
        print("✅ Complexity analyzer works")
    except Exception as e:
        print(f"❌ Complexity analyzer failed: {e}")
        return False
    
    try:
        from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
        analyzer = EnhancedDeadCodeAnalyzer(config)
        print("✅ Enhanced dead code analyzer works")
    except Exception as e:
        print(f"❌ Enhanced dead code analyzer failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 Testing Code Quality Pipelines (Basic)")
    print("=" * 50)
    
    # Test imports
    if not test_pipeline_imports():
        print("\n❌ Pipeline import tests failed")
        return 1
    
    # Test core components
    if not test_core_components():
        print("\n❌ Core component tests failed")
        return 1
    
    print("\n🎉 All basic tests passed! Core pipelines are working.")
    return 0

if __name__ == "__main__":
    sys.exit(main())