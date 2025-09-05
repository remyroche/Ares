#!/usr/bin/env python3
"""
Test script for Enhanced Import Analysis Pipeline

This script tests the enhanced import and undefined variable analyzer
and its integration with the pipeline system.
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.enhanced_import_analysis import (
    EnhancedImportAndUndefinedAnalyzer,
    IssueSeverity,
    IssueType
)
from pipelines.pipeline_enhanced_import_analysis import EnhancedImportAnalysisPipeline
from pipelines.base_pipeline import PipelineConfig


def create_test_files() -> str:
    """Create test files with various import and undefined variable issues."""
    temp_dir = tempfile.mkdtemp()
    
    # Test file 1: Import issues
    test_file_1 = Path(temp_dir) / "test_imports.py"
    test_file_1.write_text("""
import os
import sys
import os  # Duplicate import
from . import something  # Relative import
from module import *  # Wildcard import
import pandas, numpy  # Multiple imports on one line

def test_function():
    print("Hello world")
""")
    
    # Test file 2: Undefined variable issues
    test_file_2 = Path(temp_dir) / "test_undefined.py"
    test_file_2.write_text("""
import os
import sys

def test_function():
    # This should be detected as undefined
    print(undefined_variable)
    
    # This should be detected as undefined
    result = some_function()
    
    # This should be fine (builtin)
    length = len("hello")
    
    # This should be fine (imported)
    current_dir = os.getcwd()

class TestClass:
    def method(self):
        # This should be detected as undefined
        return undefined_attribute
""")
    
    # Test file 3: Mixed issues
    test_file_3 = Path(temp_dir) / "test_mixed.py"
    test_file_3.write_text("""
import os
import os  # Duplicate
from . import utils  # Relative
import pandas as pd
import numpy as np

def complex_function():
    # Undefined variable
    data = process_data()
    
    # Should be fine
    df = pd.DataFrame()
    
    # Undefined variable
    result = calculate_result(data)
    
    return result
""")
    
    return temp_dir


def test_analyzer_direct():
    """Test the analyzer directly."""
    print("="*60)
    print("Testing Enhanced Import Analyzer Directly")
    print("="*60)
    
    # Create test files
    test_dir = create_test_files()
    
    try:
        # Initialize analyzer
        analyzer = EnhancedImportAndUndefinedAnalyzer(
            project_root=test_dir,
            config={
                'ignore_patterns': ['__pycache__', '.git'],
                'min_severity': IssueSeverity.LOW
            }
        )
        
        # Run analysis
        results = analyzer.run_comprehensive_analysis(test_dir)
        
        # Print results
        print(f"Analysis completed in {results['summary']['total_execution_time']:.2f}s")
        print(f"Files analyzed: {results['summary']['total_files']}")
        print(f"Import issues: {results['summary']['import_issues']}")
        print(f"Undefined issues: {results['summary']['undefined_issues']}")
        print(f"Total issues: {results['summary']['total_issues']}")
        
        # Show detailed results
        for file_path, file_results in results['files'].items():
            print(f"\nFile: {file_path}")
            
            # Import issues
            import_issues = file_results['import_analysis'].issues
            if import_issues:
                print(f"  Import issues ({len(import_issues)}):")
                for issue in import_issues:
                    print(f"    Line {issue.line}: {issue.message} [{issue.severity.value}]")
            
            # Undefined issues
            undefined_issues = file_results['undefined_analysis'].issues
            if undefined_issues:
                print(f"  Undefined issues ({len(undefined_issues)}):")
                for issue in undefined_issues:
                    print(f"    Line {issue.line}: {issue.message} [{issue.severity.value}]")
        
        # Test high-priority issues
        high_priority = analyzer.get_high_priority_issues()
        if high_priority:
            print(f"\nHigh-priority issues ({len(high_priority)}):")
            for issue in high_priority:
                print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        
        # Test statistics
        stats = analyzer.get_issue_statistics()
        print(f"\nStatistics:")
        print(f"  Import issues: {stats['import_issues']['total']}")
        print(f"  Undefined issues: {stats['undefined_issues']['total']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct analyzer test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)


def test_pipeline_integration():
    """Test the pipeline integration."""
    print("\n" + "="*60)
    print("Testing Pipeline Integration")
    print("="*60)
    
    # Create test files
    test_dir = create_test_files()
    
    try:
        # Create pipeline configuration
        config = PipelineConfig(
            project_root=Path(test_dir),
            output_dir=Path(test_dir) / "reports",
            log_level="INFO",
            verbose=True
        )
        
        # Initialize pipeline
        pipeline = EnhancedImportAnalysisPipeline(
            project_root=test_dir,
            config=config
        )
        
        # Run pipeline
        results = pipeline.run_pipeline(test_dir)
        
        # Print results
        summary = results.get("enhanced_summary", {})
        print(f"Pipeline completed in {summary.get('total_execution_time', 0):.2f}s")
        print(f"Files analyzed: {summary.get('total_files', 0)}")
        print(f"Import issues: {summary.get('import_issues', 0)}")
        print(f"Undefined issues: {summary.get('undefined_issues', 0)}")
        print(f"Total issues: {summary.get('total_issues', 0)}")
        
        # Show recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. [{rec.get('priority', 'low').upper()}] {rec.get('message', '')}")
        
        # Test high-priority issues
        high_priority = results.get("high_priority_issues", [])
        if high_priority:
            print(f"\nHigh-priority issues ({len(high_priority)}):")
            for issue in high_priority:
                print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        
        # Test statistics
        stats = results.get("statistics", {})
        print(f"\nStatistics:")
        print(f"  Import issues: {stats.get('import_issues', {}).get('total', 0)}")
        print(f"  Undefined issues: {stats.get('undefined_issues', {}).get('total', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)


def test_plugin_system():
    """Test the plugin system integration."""
    print("\n" + "="*60)
    print("Testing Plugin System Integration")
    print("="*60)
    
    try:
        # Import plugin
        from plugins.production.enhanced_import_analyzer_plugin import EnhancedImportAnalyzerPlugin
        
        # Create test files
        test_dir = create_test_files()
        
        # Initialize plugin
        plugin = EnhancedImportAnalyzerPlugin()
        
        # Create plugin context
        from plugins.plugin_registry import PluginContext
        context = PluginContext(
            project_root=test_dir,
            target_files=[],
            configuration={
                'ignore_patterns': ['__pycache__', '.git'],
                'max_issues_per_file': 100
            },
            cache_dir=None,
            output_dir=Path(test_dir) / "reports",
            parallel_execution=False,
            max_workers=1,
            timeout=300,
            dry_run=False,
            verbose=True
        )
        
        # Initialize plugin
        if not plugin.initialize(context):
            print("❌ Plugin initialization failed")
            return False
        
        # Execute plugin
        result = plugin.execute(context)
        
        # Print results
        print(f"Plugin execution: {'✅ Success' if result.success else '❌ Failed'}")
        print(f"Message: {result.message}")
        
        if result.success:
            data = result.data
            print(f"Files analyzed: {data.get('files_analyzed', 0)}")
            print(f"Total issues: {data.get('total_issues', 0)}")
            print(f"Import issues: {data.get('import_issues', 0)}")
            print(f"Undefined issues: {data.get('undefined_issues', 0)}")
            
            # Show high-priority issues
            high_priority = data.get("high_priority_issues", [])
            if high_priority:
                print(f"\nHigh-priority issues ({len(high_priority)}):")
                for issue in high_priority:
                    print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        
        # Cleanup
        plugin.cleanup()
        
        return result.success
        
    except Exception as e:
        print(f"❌ Plugin system test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        import shutil
        if 'test_dir' in locals():
            shutil.rmtree(test_dir)


def test_accuracy_improvements():
    """Test the accuracy improvements over the original checker."""
    print("\n" + "="*60)
    print("Testing Accuracy Improvements")
    print("="*60)
    
    # Create test files with common false positive patterns
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test file with patterns that should NOT be flagged
        test_file = Path(temp_dir) / "test_accuracy.py"
        test_file.write_text("""
import os
import sys
from typing import List, Dict

class TestClass:
    def __init__(self):
        self.config = {}
        self.settings = {}
        self.logger = None
    
    def method(self, args, kwargs):
        # These should NOT be flagged as undefined
        data = self.config
        result = self.settings
        self.logger.info("test")
        
        # Builtin functions should NOT be flagged
        length = len("hello")
        items = list(range(10))
        
        # Exception handling should NOT be flagged
        try:
            risky_operation()
        except Exception as e:
            print(f"Error: {e}")
        
        # Lambda parameters should NOT be flagged
        func = lambda x: x * 2
        
        return data

def test_function():
    # Common patterns that should NOT be flagged
    config = {}
    settings = {}
    logger = None
    data = {}
    result = {}
    response = {}
    request = {}
    context = {}
    
    return config
""")
        
        # Initialize analyzer
        analyzer = EnhancedImportAndUndefinedAnalyzer(
            project_root=temp_dir,
            config={
                'ignore_patterns': ['__pycache__', '.git'],
                'min_severity': IssueSeverity.LOW
            }
        )
        
        # Run analysis
        results = analyzer.run_comprehensive_analysis(temp_dir)
        
        # Check results
        total_issues = results['summary']['total_issues']
        undefined_issues = results['summary']['undefined_issues']
        
        print(f"Total issues found: {total_issues}")
        print(f"Undefined issues found: {undefined_issues}")
        
        # The original checker would have found many false positives
        # The enhanced checker should find significantly fewer
        if total_issues == 0:
            print("✅ No false positives detected - excellent accuracy!")
            return True
        elif total_issues <= 2:
            print("✅ Very few issues detected - good accuracy!")
            return True
        else:
            print(f"⚠️  {total_issues} issues detected - may need further tuning")
            
            # Show what was detected
            for file_path, file_results in results['files'].items():
                undefined_issues = file_results['undefined_analysis'].issues
                if undefined_issues:
                    print(f"\nUndefined issues in {file_path}:")
                    for issue in undefined_issues:
                        print(f"  Line {issue.line}: {issue.name} - {issue.message}")
            
            return total_issues <= 5  # Allow some issues but not too many
        
    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        return False
    
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Enhanced Import Analysis Pipeline Tests")
    print("="*60)
    
    tests = [
        ("Direct Analyzer Test", test_analyzer_direct),
        ("Pipeline Integration Test", test_pipeline_integration),
        ("Plugin System Test", test_plugin_system),
        ("Accuracy Improvements Test", test_accuracy_improvements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced import analysis is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())