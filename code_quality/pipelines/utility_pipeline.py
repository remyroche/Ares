#!/usr/bin/env python3
"""
Utility Pipeline - Specialized pipeline for utility scripts and tools.

This pipeline integrates all utility scripts that were previously standalone,
providing a unified interface for running various utility functions.
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all utility scripts
from quick_start import QuickStart
from debug_analyzer import DebugAnalyzer
from merge_conflict_detector import MergeConflictDetector
from comprehensive_import_fixer import ComprehensiveImportFixer
from auto_fix_dead_code import AutoFixDeadCode
from targeted_import_fixer import TargetedImportFixer
from comprehensive_code_review import CodeQualityReviewer
from enhanced_validator import EnhancedValidator
from function_validator import FunctionValidator
from function_validator_wrapper import FunctionValidatorWrapper
from map_code_interactions import CodeInteractionMapper
from enhanced_map_code_interactions import EnhancedCodeInteractionMapper
from visualize_interactions import InteractionVisualizer
from dead_code_analysis import DeadCodeAnalysis
from extract_non_pandas_tests import ExtractNonPandasTests
from example_usage import ExampleUsage
from example_usage_extended import ExampleUsageExtended
from example_validation_usage import ExampleValidationUsage


class UtilityPipeline:
    """Specialized pipeline for utility scripts and tools."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def run_quick_start(self) -> Dict[str, Any]:
        """Run quick start utility."""
        print("\n" + "="*60)
        print("Running Quick Start Utility")
        print("="*60)
        
        try:
            quick_start = QuickStart()
            results = quick_start.run()
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_debug_analysis(self) -> Dict[str, Any]:
        """Run debug analysis utility."""
        print("\n" + "="*60)
        print("Running Debug Analysis Utility")
        print("="*60)
        
        try:
            debugger = DebugAnalyzer()
            results = debugger.analyze_project(str(self.project_root))
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_merge_conflict_detection(self) -> Dict[str, Any]:
        """Run merge conflict detection utility."""
        print("\n" + "="*60)
        print("Running Merge Conflict Detection Utility")
        print("="*60)
        
        try:
            detector = MergeConflictDetector()
            conflicts = detector.detect_conflicts(str(self.project_root))
            return {"status": "completed", "conflicts_found": len(conflicts), "conflicts": conflicts}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_import_fixes(self) -> Dict[str, Any]:
        """Run import fixing utilities."""
        print("\n" + "="*60)
        print("Running Import Fixing Utilities")
        print("="*60)
        
        try:
            results = {}
            
            # Comprehensive import fixer
            comprehensive_fixer = ComprehensiveImportFixer()
            results["comprehensive_fixes"] = comprehensive_fixer.fix_all_imports(str(self.project_root))
            
            # Targeted import fixer (if report exists)
            report_files = list(self.project_root.glob("**/import_analysis_report*.json"))
            if report_files:
                latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                targeted_fixer = TargetedImportFixer(str(self.project_root), str(latest_report))
                targeted_fixer.load_issues()
                results["targeted_fixes"] = targeted_fixer.fix_issues()
            
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_dead_code_utilities(self) -> Dict[str, Any]:
        """Run dead code utilities."""
        print("\n" + "="*60)
        print("Running Dead Code Utilities")
        print("="*60)
        
        try:
            results = {}
            
            # Auto fix dead code
            auto_fixer = AutoFixDeadCode()
            results["auto_fixes"] = auto_fixer.auto_fix_dead_code(str(self.project_root))
            
            # Dead code analysis
            analyzer = DeadCodeAnalysis()
            results["analysis"] = analyzer.analyze_project(str(self.project_root))
            
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_code_review_utilities(self) -> Dict[str, Any]:
        """Run code review utilities."""
        print("\n" + "="*60)
        print("Running Code Review Utilities")
        print("="*60)
        
        try:
            results = {}
            
            # Comprehensive code review
            reviewer = CodeQualityReviewer()
            results["comprehensive_review"] = reviewer.review_project(str(self.project_root))
            
            # Enhanced validator
            validator = EnhancedValidator()
            results["enhanced_validation"] = validator.validate_project(str(self.project_root))
            
            # Function validator
            func_validator = FunctionValidator()
            results["function_validation"] = func_validator.validate_functions(str(self.project_root))
            
            # Function validator wrapper
            wrapper = FunctionValidatorWrapper()
            results["wrapper_validation"] = wrapper.run_validation(str(self.project_root))
            
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_interaction_utilities(self) -> Dict[str, Any]:
        """Run code interaction utilities."""
        print("\n" + "="*60)
        print("Running Code Interaction Utilities")
        print("="*60)
        
        try:
            results = {}
            
            # Code interaction mapper
            mapper = CodeInteractionMapper()
            results["interaction_mapping"] = mapper.map_interactions(str(self.project_root))
            
            # Enhanced code interaction mapper
            enhanced_mapper = EnhancedCodeInteractionMapper()
            results["enhanced_mapping"] = enhanced_mapper.map_interactions(str(self.project_root))
            
            # Interaction visualizer
            visualizer = InteractionVisualizer()
            results["visualization"] = visualizer.visualize_interactions(str(self.project_root))
            
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_example_utilities(self) -> Dict[str, Any]:
        """Run example utilities."""
        print("\n" + "="*60)
        print("Running Example Utilities")
        print("="*60)
        
        try:
            results = {}
            
            # Example usage
            example = ExampleUsage()
            results["basic_example"] = example.run_example()
            
            # Extended example usage
            extended_example = ExampleUsageExtended()
            results["extended_example"] = extended_example.run_extended_example()
            
            # Validation example
            validation_example = ExampleValidationUsage()
            results["validation_example"] = validation_example.run_validation_example()
            
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_test_extraction_utilities(self) -> Dict[str, Any]:
        """Run test extraction utilities."""
        print("\n" + "="*60)
        print("Running Test Extraction Utilities")
        print("="*60)
        
        try:
            extractor = ExtractNonPandasTests()
            results = extractor.extract_tests(str(self.project_root))
            return {"status": "completed", "results": results}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_utilities(self) -> Dict[str, Any]:
        """Run all utility scripts."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE UTILITY PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        
        total_start = time.time()
        
        # Run all utility categories
        self.results["quick_start"] = self.run_quick_start()
        self.results["debug_analysis"] = self.run_debug_analysis()
        self.results["merge_conflict_detection"] = self.run_merge_conflict_detection()
        self.results["import_fixes"] = self.run_import_fixes()
        self.results["dead_code_utilities"] = self.run_dead_code_utilities()
        self.results["code_review_utilities"] = self.run_code_review_utilities()
        self.results["interaction_utilities"] = self.run_interaction_utilities()
        self.results["example_utilities"] = self.run_example_utilities()
        self.results["test_extraction_utilities"] = self.run_test_extraction_utilities()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "utility_categories": len(self.results) - 1,  # Exclude summary
            "status": "completed"
        }
        
        # Save results
        reports_dir = self.project_root / "code_quality" / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"utility_pipeline_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("UTILITY PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Report saved to: {report_path}")
        
        return self.results


def main():
    """Main entry point for the utility pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Utility Pipeline - Comprehensive utility script execution"
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
        choices=["quick_start", "debug", "merge_conflicts", "import_fixes", "dead_code", 
                "code_review", "interactions", "examples", "test_extraction", "all"],
        default="all",
        help="Specific utility category to run (default: all)"
    )
    
    args = parser.parse_args()
    
    pipeline = UtilityPipeline(args.project_root)
    
    if args.category == "all":
        results = pipeline.run_all_utilities()
    elif args.category == "quick_start":
        results = pipeline.run_quick_start()
    elif args.category == "debug":
        results = pipeline.run_debug_analysis()
    elif args.category == "merge_conflicts":
        results = pipeline.run_merge_conflict_detection()
    elif args.category == "import_fixes":
        results = pipeline.run_import_fixes()
    elif args.category == "dead_code":
        results = pipeline.run_dead_code_utilities()
    elif args.category == "code_review":
        results = pipeline.run_code_review_utilities()
    elif args.category == "interactions":
        results = pipeline.run_interaction_utilities()
    elif args.category == "examples":
        results = pipeline.run_example_utilities()
    elif args.category == "test_extraction":
        results = pipeline.run_test_extraction_utilities()
    
    print(f"\nUtility pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()