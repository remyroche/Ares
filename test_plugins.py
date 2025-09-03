#!/usr/bin/env python3
"""
Test script for the plugin system and progress tracking.
"""

import sys

# Add the current directory to Python path
sys.path.insert(0, ".")

def test_plugin_system():
    """Test the plugin system functionality."""
    print("🧪 Testing Plugin System...")

    try:
        from code_quality.core.plugins import (
            PluginManager,
        )
        from code_quality.plugins.black_fixer import BlackFixer
        from code_quality.plugins.flake8_analyzer import Flake8Analyzer
        from code_quality.plugins.isort_fixer import IsortFixer

        # Create plugin manager
        pm=PluginManager()
        print("✅ PluginManager created successfully")

        # Test plugin registration
        black_plugin=BlackFixer({"max_line_length": 88, "aggressive": False})
        isort_plugin=IsortFixer({"max_line_length": 88, "aggressive": False})
        flake8_plugin=Flake8Analyzer({"max_line_length": 88})

        pm.register_plugin("black", black_plugin)
        pm.register_plugin("isort", isort_plugin)
        pm.register_plugin("flake8", flake8_plugin)
        print("✅ Plugins registered successfully")

        # Test plugin discovery
        fixers=pm.get_fixers()
        analyzers=pm.get_analyzers()
        print(f"✅ Found {len(fixers)} fixers and {len(analyzers)} analyzers")

        # Test plugin capabilities
        test_file="test_file.py"
        available_fixers = pm.get_available_fixers(test_file)
        available_analyzers=pm.get_available_analyzers(test_file)
        print(f"✅ Available fixers for {test_file}: {[f.get_name() for f in available_fixers]}")
        print(f"✅ Available analyzers for {test_file}: {[a.get_name() for a in available_analyzers]}")

        # Test plugin info
        plugin_list=pm.list_plugins()
        print(f"✅ Plugin list: {[p['name'] for p in plugin_list]}")

        print("✅ Plugin system test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Plugin system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_progress_tracking():
    """Test the progress tracking functionality."""
    print("\n🧪 Testing Progress Tracking...")

    try:
        from code_quality.utils.progress import CodeQualityProgress, ProgressManager

        # Test progress manager
        pm=ProgressManager()
        print("✅ ProgressManager created successfully")

        # Test code quality progress
        CodeQualityProgress()
        print("✅ CodeQualityProgress created successfully")

        # Test progress tracking
        test_files=["file1.py", "file2.py", "file3.py"]

        def test_operation(file_path):
            return {"file": file_path, "success": True, "message": "Test operation"}

        results=pm.track_file_operation(test_files, "Test Operation", test_operation)
        print(f"✅ Progress tracking completed: {len(results)} results")

        print("✅ Progress tracking test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Progress tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_fixer():
    """Test the auto fixer with plugin system."""
    print("\n🧪 Testing Auto Fixer...")

    try:
        from code_quality.core.config import get_default_config
        from code_quality.fixers.auto_fixer import AutoFixer

        # Get default config
        config=get_default_config()
        print("✅ Configuration loaded successfully")

        # Create auto fixer
        fixer=AutoFixer(config)
        print("✅ AutoFixer created successfully")

        # Test plugin registration
        plugins=fixer.plugin_manager.list_plugins()
        print(f"✅ AutoFixer has {len(plugins)} plugins registered")

        print("✅ Auto fixer test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Auto fixer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Code Quality Tools Tests...\n")

    tests=[
        ("Plugin System", test_plugin_system),
        ("Progress Tracking", test_progress_tracking),
        ("Auto Fixer", test_auto_fixer),
    ]

    results=[]
    for test_name, test_func in tests:
        try:
            result=test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)

    passed=0
    total = len(results)

    for test_name, result in results:
        status="✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")

    if passed== total:
        print("🎉 All tests passed! The plugin system and progress tracking are working correctly.")
        return 0
    print("⚠️ Some tests failed. Please check the output above for details.")
    return 1

if __name__== "__main__":
    sys.exit(main())
