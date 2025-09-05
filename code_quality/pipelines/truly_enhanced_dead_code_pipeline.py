#!/usr/bin/env python3
"""
Truly Enhanced Dead Code Analysis Pipeline

This pipeline runs only the truly enhanced dead code analysis with advanced filtering
to significantly reduce false positives and provide high-confidence results.
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the truly enhanced analyzer
from analyzers.truly_enhanced_dead_code_analyzer import TrulyEnhancedDeadCodeAnalyzer, TrulyEnhancedDeadCodeReport
from core.config import AnalysisConfig


class TrulyEnhancedDeadCodePipeline:
    """Pipeline for truly enhanced dead code analysis with advanced filtering."""
    
    def __init__(self, project_root: str = "/workspace", enable_plugins: bool = False):
        self.project_root = Path(project_root)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.enable_plugins = enable_plugins
        
        # Create reports directory
        self.reports_dir = Path("code_quality/reports/dead_code")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the truly enhanced analyzer
        self.config = AnalysisConfig()
        self.enhanced_analyzer = TrulyEnhancedDeadCodeAnalyzer(self.config)
        
        print(f"✅ Initialized Truly Enhanced Dead Code Analyzer")
        print(f"📊 Advanced filtering enabled with multi-tool consensus")
        print(f"🎯 False positive reduction techniques active")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run the truly enhanced dead code analysis."""
        print("\n" + "="*80)
        print("TRULY ENHANCED DEAD CODE ANALYSIS PIPELINE")
        print("="*80)
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Advanced filtering: ENABLED")
        print()
        
        start_time = time.time()
        
        try:
            # Run the truly enhanced analysis
            print("🔍 Running Truly Enhanced Dead Code Analysis...")
            print("   - Multi-tool consensus filtering")
            print("   - Dynamic usage detection")
            print("   - Call graph validation")
            print("   - Context-aware filtering")
            print("   - Advanced confidence scoring")
            print()
            
            report = self.enhanced_analyzer.analyze_directory(str(self.project_root))
            
            # Save the enhanced report
            report_path = self.reports_dir / f"truly_enhanced_dead_code_{self.timestamp}.json"
            self._save_enhanced_report(report, report_path)
            
            # Generate summary
            self._print_analysis_summary(report)
            
            execution_time = time.time() - start_time
            
            return {
                "status": "completed",
                "execution_time": execution_time,
                "report_path": str(report_path),
                "total_issues": report.total_issues,
                "high_confidence_issues": report.high_confidence_issues,
                "medium_confidence_issues": report.medium_confidence_issues,
                "low_confidence_issues": report.low_confidence_issues,
                "false_positives_filtered": report.false_positives_filtered,
                "consensus_issues": report.consensus_issues,
                "filtering_effectiveness": report.filtering_stats.get("filtering_effectiveness", 0),
                "report": report
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _save_enhanced_report(self, report: TrulyEnhancedDeadCodeReport, report_path: Path) -> None:
        """Save the enhanced report to JSON."""
        # Convert report to dictionary
        report_dict = {
            "timestamp": self.timestamp,
            "analysis_type": "truly_enhanced_dead_code",
            "project_root": str(self.project_root),
            "total_issues": report.total_issues,
            "high_confidence_issues": report.high_confidence_issues,
            "medium_confidence_issues": report.medium_confidence_issues,
            "low_confidence_issues": report.low_confidence_issues,
            "false_positives_filtered": report.false_positives_filtered,
            "consensus_issues": report.consensus_issues,
            "dynamic_usage_issues": report.dynamic_usage_issues,
            "call_graph_verified_issues": report.call_graph_verified_issues,
            "filtering_stats": report.filtering_stats,
            "tool_agreement_matrix": report.tool_agreement_matrix,
            "issues_by_type": report.issues_by_type,
            "confidence_distribution": report.confidence_distribution,
            "potential_savings": report.potential_savings,
            "impact_analysis": report.impact_analysis,
            "results": {
                "issues": [
                    {
                        "file": issue.file_path,
                        "line": issue.line_number,
                        "type": issue.issue_type,
                        "description": issue.description,
                        "confidence": issue.confidence,
                        "severity": issue.severity,
                        "code_snippet": issue.code_snippet,
                        "tool_source": issue.tool_source,
                        "consensus_count": issue.consensus_count,
                        "dynamic_usage_detected": issue.dynamic_usage_detected,
                        "call_graph_verified": issue.call_graph_verified,
                        "context_score": issue.context_score,
                        "false_positive_risk": issue.false_positive_risk,
                        "function_name": issue.function_name,
                        "class_name": issue.class_name,
                        "module_type": issue.module_type,
                        "is_public_api": issue.is_public_api,
                        "has_docstring": issue.has_docstring,
                        "decorators": issue.decorators,
                        "filtering_reasons": issue.filtering_reasons,
                        "original_confidence": issue.original_confidence
                    }
                    for issue in self._get_all_issues(report)
                ]
            }
        }
        
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"📄 Enhanced report saved to: {report_path}")
    
    def _get_all_issues(self, report: TrulyEnhancedDeadCodeReport) -> List:
        """Get all issues from the report."""
        all_issues = []
        for issues in report.issues_by_file.values():
            all_issues.extend(issues)
        return all_issues
    
    def _print_analysis_summary(self, report: TrulyEnhancedDeadCodeReport) -> None:
        """Print a comprehensive analysis summary."""
        print("\n" + "="*80)
        print("TRULY ENHANCED DEAD CODE ANALYSIS RESULTS")
        print("="*80)
        
        # Overall statistics
        print(f"📊 Total Issues Found: {report.total_issues}")
        print(f"🎯 High Confidence Issues (>80%): {report.high_confidence_issues}")
        print(f"⚖️  Medium Confidence Issues (60-80%): {report.medium_confidence_issues}")
        print(f"⚠️  Low Confidence Issues (<60%): {report.low_confidence_issues}")
        print()
        
        # Show confidence distribution
        if report.total_issues > 0:
            print("📈 CONFIDENCE DISTRIBUTION:")
            high_pct = (report.high_confidence_issues / report.total_issues) * 100
            medium_pct = (report.medium_confidence_issues / report.total_issues) * 100
            low_pct = (report.low_confidence_issues / report.total_issues) * 100
            print(f"   High Confidence:   {report.high_confidence_issues:3d} ({high_pct:5.1f}%)")
            print(f"   Medium Confidence: {report.medium_confidence_issues:3d} ({medium_pct:5.1f}%)")
            print(f"   Low Confidence:    {report.low_confidence_issues:3d} ({low_pct:5.1f}%)")
            print()
        
        # Filtering effectiveness
        print("🔍 FILTERING EFFECTIVENESS:")
        print(f"   False Positives Filtered: {report.false_positives_filtered}")
        print(f"   Filtering Effectiveness: {report.filtering_stats.get('filtering_effectiveness', 0):.1f}%")
        print(f"   Consensus Issues: {report.consensus_issues}")
        print(f"   Dynamic Usage Detected: {report.dynamic_usage_issues}")
        print(f"   Call Graph Verified: {report.call_graph_verified_issues}")
        print()
        
        # Issues by type
        print("📋 ISSUES BY TYPE:")
        for issue_type, count in report.issues_by_type.items():
            print(f"   {issue_type}: {count}")
        print()
        
        # Top files with issues
        print("📁 TOP 10 FILES WITH ISSUES:")
        file_issue_counts = [(file_path, len(issues)) for file_path, issues in report.issues_by_file.items()]
        file_issue_counts.sort(key=lambda x: x[1], reverse=True)
        
        for i, (file_path, count) in enumerate(file_issue_counts[:10], 1):
            print(f"   {i:2d}. {count:3d} issues: {file_path}")
        print()
        
        # Show some example issues with confidence levels
        all_issues = self._get_all_issues(report)
        if all_issues:
            print("🔍 EXAMPLE ISSUES BY CONFIDENCE LEVEL:")
            
            # Show top 5 high confidence issues
            high_conf_issues = [i for i in all_issues if i.confidence > 80][:5]
            if high_conf_issues:
                print("   🎯 HIGH CONFIDENCE ISSUES (>80%):")
                for i, issue in enumerate(high_conf_issues, 1):
                    print(f"      {i}. {issue.confidence:5.1f}% - {issue.function_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
            
            # Show top 5 medium confidence issues
            medium_conf_issues = [i for i in all_issues if 60 <= i.confidence <= 80][:5]
            if medium_conf_issues:
                print("   ⚖️  MEDIUM CONFIDENCE ISSUES (60-80%):")
                for i, issue in enumerate(medium_conf_issues, 1):
                    print(f"      {i}. {issue.confidence:5.1f}% - {issue.function_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
            
            # Show top 5 low confidence issues
            low_conf_issues = [i for i in all_issues if i.confidence < 60][:5]
            if low_conf_issues:
                print("   ⚠️  LOW CONFIDENCE ISSUES (<60%):")
                for i, issue in enumerate(low_conf_issues, 1):
                    print(f"      {i}. {issue.confidence:5.1f}% - {issue.function_name or 'Unknown'} in {Path(issue.file_path).name}:{issue.line_number}")
                    if issue.filtering_reasons:
                        print(f"         Reasons: {', '.join(issue.filtering_reasons[:2])}")
                print()
        
        # Tool agreement matrix
        if report.tool_agreement_matrix:
            print("🤝 TOOL AGREEMENT MATRIX:")
            for tool1, agreements in report.tool_agreement_matrix.items():
                for tool2, count in agreements.items():
                    if tool1 != tool2:
                        print(f"   {tool1} ↔ {tool2}: {count} agreements")
            print()
        
        # Potential savings
        print("💰 POTENTIAL SAVINGS:")
        print(f"   Estimated Lines Removable: {report.potential_savings.get('lines_removable', 0)}")
        print(f"   Files Affected: {report.impact_analysis.get('files_affected', 0)}")
        print()
        
        print("="*80)


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Truly Enhanced Dead Code Analysis Pipeline")
    parser.add_argument("--project-root", default="/workspace", 
                       help="Root directory of the project to analyze")
    parser.add_argument("--enable-plugins", action="store_true",
                       help="Enable plugin-based analysis")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = TrulyEnhancedDeadCodePipeline(
        project_root=args.project_root,
        enable_plugins=args.enable_plugins
    )
    
    results = pipeline.run_analysis()
    
    if results["status"] == "completed":
        print(f"\n✅ Truly Enhanced Dead Code Analysis completed successfully!")
        print(f"⏱️  Execution time: {results['execution_time']:.2f} seconds")
        print(f"📊 Found {results['total_issues']} high-confidence issues")
        print(f"🎯 Filtering effectiveness: {results['filtering_effectiveness']:.1f}%")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()