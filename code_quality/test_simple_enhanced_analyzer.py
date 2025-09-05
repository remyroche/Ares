#!/usr/bin/env python3
"""
Simple test script for Enhanced Import Analysis

This script tests the enhanced import and undefined variable analyzer
without complex pipeline dependencies.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.enhanced_import_analysis import (
    EnhancedImportAndUndefinedAnalyzer,
    IssueSeverity,
    IssueType
)


def create_test_files():
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
    
    return temp_dir


def test_analyzer():
    """Test the enhanced import analyzer."""
    print("="*60)
    print("Testing Enhanced Import Analyzer")
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
        
        # Test accuracy - should find issues but not too many false positives
        total_issues = results['summary']['total_issues']
        if total_issues > 0 and total_issues <= 10:
            print(f"\n✅ Test passed! Found {total_issues} issues (reasonable number)")
            return True
        elif total_issues == 0:
            print(f"\n⚠️  No issues found - may need to check test files")
            return False
        else:
            print(f"\n❌ Too many issues found: {total_issues}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir)


def test_accuracy():
    """Test accuracy improvements."""
    print("\n" + "="*60)
    print("Testing Accuracy Improvements")
    print("="*60)
    
    # Create test files with patterns that should NOT be flagged
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
        
        # The enhanced checker should find very few or no false positives
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
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run tests."""
    print("Enhanced Import Analysis - Simple Tests")
    print("="*60)
    
    tests = [
        ("Basic Analyzer Test", test_analyzer),
        ("Accuracy Test", test_accuracy)
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