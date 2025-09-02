#!/usr/bin/env python3
"""
Generate Unified Quality Report

This script combines all the individual quality reports into a single
comprehensive human-readable text report.
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from simple_quality_orchestrator import SimpleQualityOrchestrator


def generate_unified_report():
    """Generate a unified quality report combining all analyses."""
    
    # Initialize orchestrator
    orchestrator = SimpleQualityOrchestrator()
    
    # Find all quality report JSON files
    report_files = glob.glob("quality_report_*.json")
    
    if not report_files:
        print("No quality report files found. Running fresh analysis...")
        return run_fresh_analysis()
    
    print(f"Found {len(report_files)} quality report files")
    
    # Load and combine all reports
    all_reports = {}
    summary_stats = {
        "total_files_analyzed": 0,
        "total_directories_analyzed": 0,
        "quality_distribution": {},
        "total_size_mb": 0,
        "success_rate": 0,
        "critical_issues": 0,
        "recommendations": set()
    }
    
    for report_file in sorted(report_files):
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            # Categorize the report
            if "directory_path" in report:
                # Directory report
                summary_stats["total_directories_analyzed"] += 1
                all_reports[report_file] = {
                    "type": "directory",
                    "data": report
                }
                
                # Aggregate directory statistics
                summary = report.get("summary", {})
                summary_stats["total_files_analyzed"] += summary.get("total_files", 0)
                summary_stats["total_size_mb"] += summary.get("total_size_mb", 0)
                
                # Aggregate quality distribution
                quality_dist = summary.get("quality_distribution", {})
                for quality, count in quality_dist.items():
                    summary_stats["quality_distribution"][quality] = summary_stats["quality_distribution"].get(quality, 0) + count
                
                # Check for critical issues
                if summary.get("overall_quality") == "critical":
                    summary_stats["critical_issues"] += 1
                
            else:
                # Single file report
                summary_stats["total_files_analyzed"] += 1
                all_reports[report_file] = {
                    "type": "file",
                    "data": report
                }
                
                # Aggregate file statistics
                file_info = report.get("file_info", {})
                summary_stats["total_size_mb"] += file_info.get("size_mb", 0)
                
                # Aggregate quality
                quality = report.get("quality_assessment", {}).get("overall_quality", "unknown")
                summary_stats["quality_distribution"][quality] = summary_stats["quality_distribution"].get(quality, 0) + 1
                
                # Check for critical issues
                if quality == "critical":
                    summary_stats["critical_issues"] += 1
                
                # Collect recommendations
                recommendations = report.get("recommendations", [])
                summary_stats["recommendations"].update(recommendations)
        
        except Exception as e:
            print(f"Error loading {report_file}: {e}")
    
    # Calculate overall success rate
    total_analyses = summary_stats["total_files_analyzed"] + summary_stats["total_directories_analyzed"]
    if total_analyses > 0:
        summary_stats["success_rate"] = (total_analyses - summary_stats["critical_issues"]) / total_analyses
    
    # Generate unified report
    unified_report = create_unified_report(all_reports, summary_stats)
    
    # Save unified report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_quality_report_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(unified_report)
    
    print(f"✅ Unified quality report saved to: {output_file}")
    return output_file


def run_fresh_analysis():
    """Run fresh analysis on key files and directories."""
    print("Running fresh analysis...")
    
    orchestrator = SimpleQualityOrchestrator()
    
    # Analyze key files
    key_files = [
        "trade_data_20250822_092158.json",
        "results/sr_position_analysis.csv",
        "data_quality/unified_quality_orchestrator.py"
    ]
    
    # Analyze key directories
    key_directories = [
        "results",
        "data_quality"
    ]
    
    all_reports = {}
    summary_stats = {
        "total_files_analyzed": 0,
        "total_directories_analyzed": 0,
        "quality_distribution": {},
        "total_size_mb": 0,
        "success_rate": 0,
        "critical_issues": 0,
        "recommendations": set()
    }
    
    # Analyze files
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"Analyzing file: {file_path}")
            report = orchestrator.analyze_file(file_path, f"Key file: {Path(file_path).name}")
            
            if "error" not in report:
                all_reports[file_path] = {
                    "type": "file",
                    "data": report
                }
                
                summary_stats["total_files_analyzed"] += 1
                file_info = report.get("file_info", {})
                summary_stats["total_size_mb"] += file_info.get("size_mb", 0)
                
                quality = report.get("quality_assessment", {}).get("overall_quality", "unknown")
                summary_stats["quality_distribution"][quality] = summary_stats["quality_distribution"].get(quality, 0) + 1
                
                if quality == "critical":
                    summary_stats["critical_issues"] += 1
    
    # Analyze directories
    for dir_path in key_directories:
        if Path(dir_path).is_dir():
            print(f"Analyzing directory: {dir_path}")
            report = orchestrator.analyze_directory(dir_path)
            
            if "error" not in report:
                all_reports[dir_path] = {
                    "type": "directory",
                    "data": report
                }
                
                summary_stats["total_directories_analyzed"] += 1
                summary = report.get("summary", {})
                summary_stats["total_files_analyzed"] += summary.get("total_files", 0)
                summary_stats["total_size_mb"] += summary.get("total_size_mb", 0)
                
                quality_dist = summary.get("quality_distribution", {})
                for quality, count in quality_dist.items():
                    summary_stats["quality_distribution"][quality] = summary_stats["quality_distribution"].get(quality, 0) + count
                
                if summary.get("overall_quality") == "critical":
                    summary_stats["critical_issues"] += 1
    
    # Calculate success rate
    total_analyses = summary_stats["total_files_analyzed"] + summary_stats["total_directories_analyzed"]
    if total_analyses > 0:
        summary_stats["success_rate"] = (total_analyses - summary_stats["critical_issues"]) / total_analyses
    
    # Generate unified report
    unified_report = create_unified_report(all_reports, summary_stats)
    
    # Save unified report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_quality_report_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(unified_report)
    
    print(f"✅ Fresh unified quality report saved to: {output_file}")
    return output_file


def create_unified_report(all_reports, summary_stats):
    """Create the unified report content."""
    lines = []
    
    # Header
    lines.append("=" * 100)
    lines.append("UNIFIED DATA QUALITY ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 50)
    lines.append(f"Total Files Analyzed: {summary_stats['total_files_analyzed']}")
    lines.append(f"Total Directories Analyzed: {summary_stats['total_directories_analyzed']}")
    lines.append(f"Total Data Size: {summary_stats['total_size_mb']:.3f} MB")
    lines.append(f"Overall Success Rate: {summary_stats['success_rate']:.1%}")
    lines.append(f"Critical Issues Found: {summary_stats['critical_issues']}")
    lines.append("")
    
    # Quality Distribution
    if summary_stats["quality_distribution"]:
        lines.append("QUALITY DISTRIBUTION")
        lines.append("-" * 50)
        for quality, count in sorted(summary_stats["quality_distribution"].items(), 
                                   key=lambda x: {"excellent": 0, "good": 1, "acceptable": 2, "poor": 3, "critical": 4}.get(x[0], 5)):
            lines.append(f"• {quality.capitalize()}: {count} files")
        lines.append("")
    
    # Overall Assessment
    lines.append("OVERALL ASSESSMENT")
    lines.append("-" * 50)
    
    if summary_stats["critical_issues"] == 0:
        lines.append("🎉 EXCELLENT: No critical quality issues detected!")
        lines.append("   All analyzed files meet quality standards.")
    elif summary_stats["critical_issues"] <= 2:
        lines.append("✅ GOOD: Minor quality issues detected.")
        lines.append("   Most files are in good condition with few problems.")
    elif summary_stats["critical_issues"] <= 5:
        lines.append("⚠️  ACCEPTABLE: Some quality issues detected.")
        lines.append("   Several files need attention but overall quality is acceptable.")
    elif summary_stats["critical_issues"] <= 10:
        lines.append("❌ POOR: Significant quality issues detected.")
        lines.append("   Many files have problems requiring immediate attention.")
    else:
        lines.append("🚨 CRITICAL: Severe quality issues detected!")
        lines.append("   Extensive problems found across multiple files.")
    
    lines.append("")
    
    # Recommendations
    if summary_stats["recommendations"]:
        lines.append("KEY RECOMMENDATIONS")
        lines.append("-" * 50)
        for rec in sorted(summary_stats["recommendations"])[:10]:  # Top 10 recommendations
            lines.append(f"• {rec}")
        lines.append("")
    
    # Detailed Results
    lines.append("DETAILED ANALYSIS RESULTS")
    lines.append("=" * 100)
    lines.append("")
    
    # Group by type
    file_reports = {k: v for k, v in all_reports.items() if v["type"] == "file"}
    directory_reports = {k: v for k, v in all_reports.items() if v["type"] == "directory"}
    
    # File Reports
    if file_reports:
        lines.append("INDIVIDUAL FILE ANALYSES")
        lines.append("-" * 50)
        lines.append("")
        
        for file_path, report_info in sorted(file_reports.items()):
            report = report_info["data"]
            
            if "error" in report:
                lines.append(f"❌ {Path(file_path).name}: {report['error']}")
            else:
                file_name = report.get("file_name", Path(file_path).name)
                quality = report.get("quality_assessment", {}).get("overall_quality", "unknown")
                size = report.get("file_info", {}).get("size_mb", "unknown")
                context = report.get("context", "")
                
                lines.append(f"📁 {file_name}")
                lines.append(f"   Quality: {quality.upper()}")
                lines.append(f"   Size: {size} MB")
                if context:
                    lines.append(f"   Context: {context}")
                
                # Show issues if any
                issues = report.get("issues", [])
                if issues:
                    lines.append(f"   Issues: {len(issues)} found")
                    for issue in issues[:2]:  # Show first 2 issues
                        lines.append(f"     - {issue}")
                    if len(issues) > 2:
                        lines.append(f"     ... and {len(issues) - 2} more issues")
                
                lines.append("")
    
    # Directory Reports
    if directory_reports:
        lines.append("DIRECTORY ANALYSES")
        lines.append("-" * 50)
        lines.append("")
        
        for dir_path, report_info in sorted(directory_reports.items()):
            report = report_info["data"]
            
            if "error" in report:
                lines.append(f"❌ {Path(dir_path).name}: {report['error']}")
            else:
                dir_name = Path(dir_path).name
                summary = report.get("summary", {})
                quality = summary.get("overall_quality", "unknown")
                total_files = summary.get("total_files", 0)
                success_rate = summary.get("success_rate", 0)
                total_size = summary.get("total_size_mb", 0)
                
                lines.append(f"📂 {dir_name}/")
                lines.append(f"   Overall Quality: {quality.upper()}")
                lines.append(f"   Total Files: {total_files}")
                lines.append(f"   Success Rate: {success_rate:.1%}")
                lines.append(f"   Total Size: {total_size:.3f} MB")
                
                # Show quality distribution
                quality_dist = summary.get("quality_distribution", {})
                if quality_dist:
                    lines.append(f"   Quality Breakdown:")
                    for q, count in quality_dist.items():
                        lines.append(f"     - {q.capitalize()}: {count} files")
                
                # Show file results summary
                file_results = report.get("file_results", {})
                if file_results:
                    lines.append(f"   File Results:")
                    for file_path, result in list(file_results.items())[:5]:  # Show first 5
                        file_name = Path(file_path).name
                        if "error" in result:
                            lines.append(f"     ❌ {file_name}: {result['error']}")
                        else:
                            file_quality = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                            file_size = result.get("file_info", {}).get("size_mb", "unknown")
                            lines.append(f"     ✅ {file_name}: {file_quality.upper()} ({file_size} MB)")
                    
                    if len(file_results) > 5:
                        lines.append(f"     ... and {len(file_results) - 5} more files")
                
                lines.append("")
    
    # Footer
    lines.append("=" * 100)
    lines.append("END OF UNIFIED QUALITY REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Report generated by SimpleQualityOrchestrator")
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    generate_unified_report()