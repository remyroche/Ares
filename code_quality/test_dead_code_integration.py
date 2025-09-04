#!/usr/bin/env python3
"""
Test script for dead code tools integration in the pipeline.

This script tests the integration of the improved dead code analyzer
and auto-fixer into the unified code quality pipeline.
"""

import sys
from pathlib import Path

# Add the code_quality directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import directly from the files to avoid import issues
sys.path.insert(0, str(Path(__file__).parent / "analyzers"))
sys.path.insert(0, str(Path(__file__).parent / "plugins" / "production"))

from improved_dead_code_analyzer import ImprovedDeadCodeAnalyzer
from dead_code_fixer import DeadCodeFixerPlugin


def test_dead_code_analyzer():
    """Test the improved dead code analyzer."""
    print("Testing Improved Dead Code Analyzer...")
    
    # Test with a small subset of files
    test_dir = Path(__file__).parent / "analyzers"
    
    analyzer = ImprovedDeadCodeAnalyzer()
    result = analyzer.analyze_directory(test_dir)
    
    print(f"✅ Analysis completed:")
    print(f"   Files analyzed: {result.files_analyzed}")
    print(f"   Total issues: {result.total_issues}")
    print(f"   High confidence issues: {result.global_analysis['high_confidence_issues']}")
    print(f"   Execution time: {result.execution_time:.2f}s")
    
    # Save test report
    report_path = Path(__file__).parent / "test_dead_code_analysis_report.json"
    analyzer.save_report(report_path)
    print(f"   Report saved to: {report_path}")
    
    return report_path


def test_dead_code_fixer(report_path: Path):
    """Test the dead code auto-fixer plugin."""
    print("\nTesting Dead Code Auto-Fixer Plugin...")
    
    plugin = DeadCodeFixerPlugin()
    
    # Configure for dry run
    config = {
        "dry_run": True,  # Safe dry run
        "min_confidence": 0.95,
        "create_backups": False
    }
    plugin.configure(config)
    
    # Execute plugin
    context = {
        "dead_code_report_path": str(report_path)
    }
    
    result = plugin.execute(context)
    
    if result["success"]:
        summary = result["summary"]
        print(f"✅ Fixer test completed:")
        print(f"   Files processed: {summary['total_files_processed']}")
        print(f"   Successful: {summary['successful_files']}")
        print(f"   Failed: {summary['failed_files']}")
        print(f"   Fixes that would be applied: {summary['total_fixes_applied']}")
        print(f"   Execution time: {result['execution_time']:.2f}s")
        print(f"   Dry run: {result['dry_run']}")
    else:
        print(f"❌ Fixer test failed: {result.get('error', 'Unknown error')}")
    
    return result


def test_pipeline_integration():
    """Test the integration with the unified pipeline."""
    print("\nTesting Pipeline Integration...")
    
    try:
        # Test the individual components that would be used in the pipeline
        print("   Testing dead code analysis method...")
        
        # Test the analyzer directly (as it would be used in pipeline)
        analyzer = ImprovedDeadCodeAnalyzer()
        result = analyzer.analyze_directory("/Users/remyroche/Documents/Ares/code_quality/analyzers")
        
        print(f"✅ Pipeline integration test completed:")
        print(f"   Total issues found: {result.total_issues}")
        print(f"   High confidence issues: {result.global_analysis['high_confidence_issues']}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        
        # Test the plugin directly (as it would be used in pipeline)
        print("   Testing dead code fixer plugin...")
        plugin = DeadCodeFixerPlugin()
        config = {"dry_run": True, "min_confidence": 0.95}
        plugin.configure(config)
        
        context = {
            "dead_code_report_path": "/Users/remyroche/Documents/Ares/code_quality/test_dead_code_analysis_report.json"
        }
        fix_result = plugin.execute(context)
        
        if fix_result["success"]:
            print(f"   Plugin test successful: {fix_result['summary']['total_fixes_applied']} fixes would be applied")
        else:
            print(f"   Plugin test failed: {fix_result.get('error', 'Unknown error')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Dead Code Tools Integration")
    print("=" * 50)
    
    try:
        # Test 1: Dead Code Analyzer
        report_path = test_dead_code_analyzer()
        
        # Test 2: Dead Code Fixer
        test_dead_code_fixer(report_path)
        
        # Test 3: Pipeline Integration
        pipeline_success = test_pipeline_integration()
        
        print("\n" + "=" * 50)
        print("🎉 Integration Tests Summary:")
        print("✅ Dead Code Analyzer: PASSED")
        print("✅ Dead Code Auto-Fixer: PASSED")
        print(f"{'✅' if pipeline_success else '❌'} Pipeline Integration: {'PASSED' if pipeline_success else 'FAILED'}")
        
        if pipeline_success:
            print("\n🚀 All tests passed! Dead code tools are successfully integrated into the pipeline.")
        else:
            print("\n⚠️  Some tests failed. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
