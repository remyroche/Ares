#!/usr/bin/env python3
"""
Enhanced Dead Code Analysis Runner

This script demonstrates how to use the enhanced dead code analyzer
with multiple tools and comprehensive analysis capabilities.
"""

import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add code_quality to path
sys.path.insert(0, str(Path(__file__).parent))

from analyzers.enhanced_dead_code_analyzer import EnhancedDeadCodeAnalyzer
from core.config import AnalysisConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_analysis_summary(report):
    """Print a summary of the analysis results."""
    print("\n" + "="*60)
    print("📊 ENHANCED DEAD CODE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"🔍 Total Issues Found: {report.total_issues}")
    print(f"🎯 False Positives Filtered: {report.false_positives_filtered}")
    print(f"📈 Call Graph: {report.call_graph.number_of_nodes()} nodes, {report.call_graph.number_of_edges()} edges")
    print(f"🔗 Dependency Graph: {report.dependency_graph.number_of_nodes()} nodes, {report.dependency_graph.number_of_edges()} edges")
    
    print(f"\n📋 Issues by Type:")
    for issue_type, count in report.issues_by_type.items():
        print(f"   {issue_type}: {count}")
    
    print(f"\n⚠️  Issues by Severity:")
    for severity, issues in report.issues_by_severity.items():
        print(f"   {severity}: {len(issues)}")
    
    print(f"\n🛠️  Issues by Tool:")
    for tool, issues in report.issues_by_tool.items():
        print(f"   {tool}: {len(issues)}")
    
    print(f"\n📊 Confidence Distribution:")
    for confidence, count in report.confidence_distribution.items():
        print(f"   {confidence}: {count}")

def print_detailed_results(report, max_files: int = 10):
    """Print detailed results for each file."""
    print(f"\n📄 DETAILED RESULTS (showing first {max_files} files):")
    print("-" * 60)
    
    file_count = 0
    for file_path, issues in report.issues_by_file.items():
        if file_count >= max_files:
            break
            
        print(f"\n📁 {file_path} ({len(issues)} issues):")
        
        for issue in issues[:5]:  # Show first 5 issues per file
            print(f"   Line {issue.line_number}: {issue.description}")
            print(f"      Tool: {issue.tool_source}, Confidence: {issue.confidence}%, Severity: {issue.severity}")
            
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more issues")
            
        file_count += 1

def generate_recommendations(report):
    """Generate actionable recommendations based on the analysis."""
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    # High confidence issues
    high_confidence = [issue for issues in report.issues_by_severity.get('high', []) 
                      for issue in issues if issue.confidence >= 95]
    if high_confidence:
        print(f"🎯 High Priority ({len(high_confidence)} issues):")
        print("   - These issues have high confidence and should be addressed first")
        print("   - Consider removing unused functions and imports")
    
    # Tool-specific recommendations
    if 'DeadCodeRemover' in report.issues_by_tool:
        deadcode_count = len(report.issues_by_tool['DeadCodeRemover'])
        print(f"🔧 DeadCodeRemover found {deadcode_count} issues:")
        print("   - These are likely safe to remove")
        print("   - Review each issue before removal")
    
    if 'PyCG' in report.issues_by_tool:
        pycg_count = len(report.issues_by_tool['PyCG'])
        print(f"🔗 PyCG found {pycg_count} unused functions:")
        print("   - Check if functions are used dynamically")
        print("   - Verify they're not part of public APIs")
    
    # Complexity recommendations
    if report.call_graph.number_of_nodes() > 100:
        print(f"📈 Large codebase detected ({report.call_graph.number_of_nodes()} functions):")
        print("   - Consider breaking into smaller modules")
        print("   - Focus on high-traffic areas first")
    
    # Impact analysis
    if report.impact_analysis:
        estimated_hours = report.impact_analysis.get('estimated_time_savings', {}).get('estimated_hours_saved', 0)
        if estimated_hours > 0:
            print(f"⏱️  Estimated time savings: {estimated_hours:.1f} hours")
            print("   - Focus on high-impact removals first")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Dead Code Analysis")
    parser.add_argument("--project-root", default="/workspace",
                       help="Root directory of the project to analyze")
    parser.add_argument("--output-dir", default="/workspace/code_quality/enhanced_analysis_output",
                       help="Output directory for results and visualizations")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--max-files", type=int, default=10,
                       help="Maximum number of files to show in detailed results")
    parser.add_argument("--generate-visualizations", action="store_true",
                       help="Generate visualization files")
    parser.add_argument("--export-json", action="store_true",
                       help="Export results to JSON")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    config = AnalysisConfig()
    analyzer = EnhancedDeadCodeAnalyzer(config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Enhanced Dead Code Analysis")
    print("=" * 60)
    print(f"📁 Project Root: {args.project_root}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"🔧 Available Tools:")
    print(f"   - DeadCodeRemover: {'✅' if analyzer.deadcode_available else '❌'}")
    print(f"   - PyCG: {'✅' if analyzer.pycg_available else '❌'}")
    print(f"   - NetworkX: ✅")
    print(f"   - Enhanced AST: ✅")
    
    try:
        # Run analysis
        print(f"\n🔍 Starting analysis...")
        start_time = datetime.now()
        
        report = analyzer.analyze_directory(args.project_root)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Analysis completed in {duration:.2f} seconds")
        
        # Print results
        print_analysis_summary(report)
        print_detailed_results(report, args.max_files)
        generate_recommendations(report)
        
        # Generate outputs
        if args.generate_visualizations:
            print(f"\n🎨 Generating visualizations...")
            analyzer.generate_visualization(report, output_dir)
            print(f"   Visualizations saved to: {output_dir}")
        
        if args.export_json:
            print(f"\n💾 Exporting results to JSON...")
            json_file = output_dir / f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            analyzer.export_results(report, json_file)
            print(f"   Results exported to: {json_file}")
        
        print(f"\n🎉 Analysis complete! Check {output_dir} for detailed results.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    await main()