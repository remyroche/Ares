#!/usr/bin/env python3
"""
Enhanced Import Analysis Pipeline

This pipeline integrates the enhanced import and undefined variable analyzer
with the existing code_quality pipeline infrastructure. It provides:

1. Enhanced Import Analysis - duplicate imports, wildcard imports, relative imports
2. Advanced Undefined Variable Detection - sophisticated analysis with reduced false positives
3. Issue Classification - severity levels and issue type categorization
4. Pipeline Integration - works with existing code_quality pipelines
5. Plugin Support - extensible through the plugin architecture
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .base_pipeline import BasePipeline, PipelineConfig
except ImportError:
    from base_pipeline import BasePipeline, PipelineConfig
from analyzers.enhanced_import_analysis import (
    EnhancedImportAndUndefinedAnalyzer, 
    IssueSeverity, 
    IssueType
)


class EnhancedImportAnalysisPipeline(BasePipeline):
    """
    Enhanced pipeline for import and undefined variable analysis.
    
    This pipeline provides comprehensive analysis of import issues and undefined variables
    with improved accuracy and detailed reporting.
    """

    def __init__(self, project_root: str = "/workspace/src", config: Optional[PipelineConfig] = None):
        """Initialize the enhanced import analysis pipeline."""
        super().__init__(project_root, config)
        
        # Initialize the enhanced analyzer
        analyzer_config = {
            'ignore_patterns': self.config.ignore_patterns if hasattr(self.config, 'ignore_patterns') else [
                '__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'
            ],
            'max_issues_per_file': 100,
            'min_severity': IssueSeverity.LOW
        }
        
        self.analyzer = EnhancedImportAndUndefinedAnalyzer(
            project_root=str(self.project_root),
            config=analyzer_config
        )
        
        # Pipeline-specific results
        self.results.update({
            "import_analysis": {},
            "undefined_analysis": {},
            "enhanced_summary": {},
            "recommendations": []
        })

    def run_import_analysis(self, target_path: str = None) -> Dict[str, Any]:
        """Run enhanced import analysis."""
        self.logger.info("Starting enhanced import analysis")
        
        if target_path is None:
            target_path = str(self.project_root)
        
        start_time = time.time()
        
        try:
            # Run comprehensive analysis
            results = self.analyzer.run_comprehensive_analysis(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            import_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "files": results.get("files", {}),
                "total_issues": results.get("summary", {}).get("total_issues", 0),
                "import_issues": results.get("summary", {}).get("import_issues", 0),
                "undefined_issues": results.get("summary", {}).get("undefined_issues", 0),
            }
            
            # Generate recommendations
            recommendations = []
            total_issues = import_results["total_issues"]
            
            if total_issues > 0:
                if import_results["import_issues"] > 0:
                    recommendations.append({
                        "priority": "medium",
                        "category": "imports",
                        "message": f"Review {import_results['import_issues']} import issues"
                    })
                
                if import_results["undefined_issues"] > 0:
                    recommendations.append({
                        "priority": "high",
                        "category": "undefined_variables",
                        "message": f"Fix {import_results['undefined_issues']} undefined variable issues"
                    })
            
            import_results["recommendations"] = recommendations
            
            self.results["import_analysis"] = import_results
            
            # Update metrics
            self.metrics["files_processed"] += results.get("summary", {}).get("total_files", 0)
            self.metrics["issues_found"] += total_issues
            
            # Log results
            self.logger.info(f"Import analysis completed in {execution_time:.2f}s")
            self.logger.info(f"Total files analyzed: {results.get('summary', {}).get('total_files', 0)}")
            self.logger.info(f"Total issues found: {total_issues}")
            
            return import_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["import_analysis"] = error_result
            self.logger.error(f"Import analysis failed: {e}")
            return error_result

    def run_undefined_analysis(self, target_path: str = None) -> Dict[str, Any]:
        """Run enhanced undefined variable analysis."""
        self.logger.info("Starting enhanced undefined variable analysis")
        
        if target_path is None:
            target_path = str(self.project_root)
        
        start_time = time.time()
        
        try:
            # Run comprehensive analysis
            results = self.analyzer.run_comprehensive_analysis(target_path)
            
            execution_time = time.time() - start_time
            
            # Process results
            undefined_results = {
                "status": "success",
                "execution_time": execution_time,
                "target_path": target_path,
                "summary": results.get("summary", {}),
                "files": results.get("files", {}),
                "total_errors": results.get("summary", {}).get("undefined_issues", 0),
                "files_with_errors": results.get("summary", {}).get("files_with_undefined_issues", 0),
            }
            
            # Generate recommendations
            recommendations = []
            total_errors = undefined_results["total_errors"]
            
            if total_errors > 0:
                recommendations.append({
                    "priority": "high",
                    "category": "undefined_variables",
                    "message": f"Fix {total_errors} undefined variable/name issues"
                })
            
            undefined_results["recommendations"] = recommendations
            
            self.results["undefined_analysis"] = undefined_results
            
            # Update metrics
            self.metrics["files_processed"] += results.get("summary", {}).get("total_files", 0)
            self.metrics["issues_found"] += total_errors
            
            # Log results
            self.logger.info(f"Undefined variable analysis completed in {execution_time:.2f}s")
            self.logger.info(f"Total undefined issues: {total_errors}")
            
            return undefined_results
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "target_path": target_path
            }
            self.results["undefined_analysis"] = error_result
            self.logger.error(f"Undefined variable analysis failed: {e}")
            return error_result

    def run_comprehensive_analysis(self, target_path: str = None) -> Dict[str, Any]:
        """Run comprehensive import and undefined variable analysis."""
        self.logger.info("Starting comprehensive enhanced analysis")
        
        if target_path is None:
            target_path = str(self.project_root)
        
        print("="*70)
        print("ENHANCED IMPORT AND UNDEFINED VARIABLE ANALYSIS PIPELINE")
        print("="*70)
        print(f"Target: {target_path}")
        print(f"Timestamp: {self.timestamp}")
        print()
        
        start_time = time.time()
        
        # Run comprehensive analysis
        results = self.analyzer.run_comprehensive_analysis(target_path)
        
        total_time = time.time() - start_time
        
        # Generate overall summary
        overall_summary = {
            "timestamp": self.timestamp,
            "target_path": target_path,
            "total_execution_time": total_time,
            "import_issues": results.get("summary", {}).get("import_issues", 0),
            "undefined_issues": results.get("summary", {}).get("undefined_issues", 0),
            "total_issues": results.get("summary", {}).get("total_issues", 0),
            "files_with_import_issues": results.get("summary", {}).get("files_with_import_issues", 0),
            "files_with_undefined_issues": results.get("summary", {}).get("files_with_undefined_issues", 0),
            "total_files": results.get("summary", {}).get("total_files", 0),
        }
        
        # Combine recommendations
        all_recommendations = results.get("summary", {}).get("recommendations", [])
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 2))
        
        overall_summary["recommendations"] = all_recommendations
        
        self.results["enhanced_summary"] = overall_summary
        self.results["files"] = results.get("files", {})
        
        # Update metrics
        self.metrics["files_processed"] += overall_summary["total_files"]
        self.metrics["issues_found"] += overall_summary["total_issues"]
        self.metrics["execution_count"] += 1
        self.metrics["total_execution_time"] += total_time
        self.metrics["successful_executions"] += 1
        
        # Print final summary
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*70)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Files analyzed: {overall_summary['total_files']}")
        print(f"Import issues: {overall_summary['import_issues']}")
        print(f"Undefined variable issues: {overall_summary['undefined_issues']}")
        print(f"Total issues: {overall_summary['total_issues']}")
        
        if all_recommendations:
            print("\n📋 Recommendations:")
            for i, rec in enumerate(all_recommendations, 1):
                priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec.get("priority", "low"), "⚪")
                print(f"  {i}. {priority_emoji} [{rec.get('priority', 'low').upper()}] {rec.get('message', '')}")
        
        # Log results
        self.logger.info(f"Comprehensive analysis completed in {total_time:.2f}s")
        self.logger.info(f"Total issues found: {overall_summary['total_issues']}")
        
        return self.results

    def save_enhanced_report(self, output_file: str = None) -> str:
        """Save the enhanced analysis results to a JSON report file."""
        if output_file is None:
            output_file = self.reports_dir / f"enhanced_import_analysis_report_{self.timestamp}.json"
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self._make_serializable(self.results)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Enhanced report saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced report: {e}")
            raise

    def get_high_priority_issues(self) -> List[Dict[str, Any]]:
        """Get a list of high-priority issues that need immediate attention."""
        return self.analyzer.get_high_priority_issues()

    def get_issue_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about found issues."""
        return self.analyzer.get_issue_statistics()

    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (IssueSeverity, IssueType)):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

    def run_pipeline(self, target_path: str = None) -> Dict[str, Any]:
        """Run the complete enhanced import analysis pipeline."""
        self.logger.info("Starting enhanced import analysis pipeline")
        
        # Run comprehensive analysis
        results = self.run_comprehensive_analysis(target_path)
        
        # Save report
        report_path = self.save_enhanced_report()
        results["report_path"] = report_path
        
        # Get high-priority issues
        high_priority = self.get_high_priority_issues()
        results["high_priority_issues"] = high_priority
        
        # Get statistics
        stats = self.get_issue_statistics()
        results["statistics"] = stats
        
        self.logger.info("Enhanced import analysis pipeline completed")
        return results


def main():
    """Command-line interface for the enhanced import analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Import Analysis Pipeline"
    )
    parser.add_argument("--target", "-t", 
                       help="Path to Python file or directory to analyze (default: /workspace/src)")
    parser.add_argument("--output", "-o", 
                       help="Output file for JSON report")
    parser.add_argument("--project-root", default="/workspace/src",
                       help="Project root directory (default: /workspace/src)")
    parser.add_argument("--min-severity", choices=['low', 'medium', 'high', 'critical'], default='low',
                       help="Minimum severity level to report (default: low)")
    parser.add_argument("--max-issues-per-file", type=int, default=100,
                       help="Maximum issues to report per file (default: 100)")
    parser.add_argument("--ignore-patterns", nargs='+', 
                       default=['__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache'],
                       help="Directory patterns to ignore")
    parser.add_argument("--stats", action="store_true",
                       help="Show detailed statistics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        project_root=Path(args.project_root),
        output_dir=Path("/workspace/code_quality/reports"),
        log_level="DEBUG" if args.verbose else "INFO",
        verbose=args.verbose
    )
    
    # Initialize pipeline
    pipeline = EnhancedImportAnalysisPipeline(
        project_root=args.project_root,
        config=config
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(args.target)
    
    # Print high-priority issues
    high_priority = results.get("high_priority_issues", [])
    if high_priority:
        print(f"\n🚨 {len(high_priority)} high-priority issues found:")
        for issue in high_priority:
            print(f"  - {issue['file']}:{issue['line']} - {issue['message']}")
    
    # Show detailed statistics if requested
    if args.stats:
        stats = results.get("statistics", {})
        print(f"\n📊 Detailed Statistics:")
        print(f"Import Issues:")
        print(f"  Total: {stats.get('import_issues', {}).get('total', 0)}")
        print(f"  Files affected: {stats.get('import_issues', {}).get('files_affected', 0)}")
        if stats.get('import_issues', {}).get('by_type'):
            print(f"  By type:")
            for issue_type, count in stats['import_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        if stats.get('import_issues', {}).get('by_severity'):
            print(f"  By severity:")
            for severity, count in stats['import_issues']['by_severity'].items():
                print(f"    {severity}: {count}")
        
        print(f"Undefined Issues:")
        print(f"  Total: {stats.get('undefined_issues', {}).get('total', 0)}")
        print(f"  Files affected: {stats.get('undefined_issues', {}).get('files_affected', 0)}")
        if stats.get('undefined_issues', {}).get('by_type'):
            print(f"  By type:")
            for issue_type, count in stats['undefined_issues']['by_type'].items():
                print(f"    {issue_type}: {count}")
        if stats.get('undefined_issues', {}).get('by_severity'):
            print(f"  By severity:")
            for severity, count in stats['undefined_issues']['by_severity'].items():
                print(f"    {severity}: {count}")
    
    # Exit with appropriate code
    summary = results.get("enhanced_summary", {})
    total_issues = summary.get("total_issues", 0)
    
    if total_issues == 0:
        print(f"\n✅ All checks passed!")
        return 0
    elif total_issues <= 10:
        print(f"\n⚠️  Found {total_issues} issues that need attention.")
        return 1
    else:
        print(f"\n❌ Found {total_issues} issues that require immediate attention!")
        return 2


if __name__ == "__main__":
    sys.exit(main())