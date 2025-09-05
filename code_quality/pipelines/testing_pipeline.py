#!/usr/bin/env python3
"""
Testing Pipeline - Specialized pipeline for test execution and validation.

This pipeline integrates all testing and validation scripts that were previously
standalone, providing a unified interface for running comprehensive tests.
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all testing and validation scripts
from verify_test_setup import verify_structure
from verify_test_structure import verify_test_structure
from test_enhanced_analyzer import TestEnhancedAnalyzer
from test_integration import TestIntegration
from test_pipeline import TestPipeline
from test_pipeline_simple import TestPipelineSimple
from test_tools import TestTools
from test_dead_code_integration import TestDeadCodeIntegration
from run_common_operations_tests import run_common_operations_tests
from run_final_tests import run_final_tests
from run_tests_simple import run_tests_simple
from run_tests_with_mocks import run_tests_with_mocks
from run_validation import run_validation
from run_subset_tests import run_subset_tests
from run_real_subset_tests import run_real_subset_tests


class TestingPipeline:
    """Specialized pipeline for testing and validation."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_test_verification(self) -> Dict[str, Any]:
        """Run test setup verification."""
        print("\n" + "="*60)
        print("Running Test Setup Verification")
        print("="*60)
        
        try:
            verify_structure()
            verify_test_structure()
            return {"status": "completed", "message": "Test structure verified successfully"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_enhanced_analyzer_tests(self) -> Dict[str, Any]:
        """Run enhanced analyzer tests."""
        print("\n" + "="*60)
        print("Running Enhanced Analyzer Tests")
        print("="*60)
        
        try:
            tester = TestEnhancedAnalyzer()
            results = tester.run_all_tests()
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n" + "="*60)
        print("Running Integration Tests")
        print("="*60)
        
        try:
            tester = TestIntegration()
            results = tester.run_all_tests()
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_pipeline_tests(self) -> Dict[str, Any]:
        """Run pipeline tests."""
        print("\n" + "="*60)
        print("Running Pipeline Tests")
        print("="*60)
        
        try:
            results = {
                "pipeline_tests": TestPipeline().run_all_tests(),
                "simple_pipeline_tests": TestPipelineSimple().run_all_tests(),
            }
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_tool_tests(self) -> Dict[str, Any]:
        """Run tool tests."""
        print("\n" + "="*60)
        print("Running Tool Tests")
        print("="*60)
        
        try:
            tester = TestTools()
            results = tester.run_all_tests()
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_dead_code_tests(self) -> Dict[str, Any]:
        """Run dead code integration tests."""
        print("\n" + "="*60)
        print("Running Dead Code Integration Tests")
        print("="*60)
        
        try:
            tester = TestDeadCodeIntegration()
            results = tester.run_all_tests()
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests."""
        print("\n" + "="*60)
        print("Running Validation Tests")
        print("="*60)
        
        try:
            results = {
                "validation": run_validation(),
                "common_operations": run_common_operations_tests(),
                "final_tests": run_final_tests(),
                "simple_tests": run_tests_simple(),
                "mock_tests": run_tests_with_mocks(),
                "subset_tests": run_subset_tests(),
                "real_subset_tests": run_real_subset_tests(),
            }
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all testing and validation."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TESTING AND VALIDATION PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        
        total_start = time.time()
        
        # Run all test categories
        self.results["test_verification"] = self.run_test_verification()
        self.results["enhanced_analyzer_tests"] = self.run_enhanced_analyzer_tests()
        self.results["integration_tests"] = self.run_integration_tests()
        self.results["pipeline_tests"] = self.run_pipeline_tests()
        self.results["tool_tests"] = self.run_tool_tests()
        self.results["dead_code_tests"] = self.run_dead_code_tests()
        self.results["validation_tests"] = self.run_validation_tests()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "test_categories": len(self.results) - 1,  # Exclude summary
            "status": "completed"
        }
        
        # Save results
        reports_dir = self.project_root / "code_quality" / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"testing_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("TESTING PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Report saved to: {report_path}")
        
        return self.results


def main():
    """Main entry point for the testing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Testing and Validation Pipeline - Comprehensive test execution"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["verification", "analyzer", "integration", "pipeline", "tools", "dead_code", "validation", "all"],
        default="all",
        help="Specific test category to run (default: all)"
    )
    
    args = parser.parse_args()
    
    pipeline = TestingPipeline(args.project_root)
    
    if args.category == "all":
        results = pipeline.run_all_tests()
    elif args.category == "verification":
        results = pipeline.run_test_verification()
    elif args.category == "analyzer":
        results = pipeline.run_enhanced_analyzer_tests()
    elif args.category == "integration":
        results = pipeline.run_integration_tests()
    elif args.category == "pipeline":
        results = pipeline.run_pipeline_tests()
    elif args.category == "tools":
        results = pipeline.run_tool_tests()
    elif args.category == "dead_code":
        results = pipeline.run_dead_code_tests()
    elif args.category == "validation":
        results = pipeline.run_validation_tests()
    
    print(f"\nTesting pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()