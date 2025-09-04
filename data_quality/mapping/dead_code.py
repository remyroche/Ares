"""
Dead Code Mapping Tools

Advanced dead code detection and mapping utilities that integrate with the
code_quality analyzers to provide comprehensive dead code analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_quality.analyzers.dead_code_analyzer import DeadCodeAnalyzer, DeadCodeReport, DeprecatedCodeIssue
from code_quality.core.config import CodeQualityConfig, get_default_config, load_config


def _load_cq_config(config_path: str | None) -> CodeQualityConfig:
    """Load code quality configuration."""
    if config_path:
        return load_config(config_path)
    return get_default_config()


def map_dead_code(
    path: str,
    config_path: str | None = None,
    include_deprecated: bool = True,
    include_dynamic_imports: bool = True,
    include_conditional: bool = True,
) -> dict[str, Any]:
    """
    Map dead code in a directory and return comprehensive analysis.

    Args:
        path: Path to directory or file to analyze
        config_path: Optional path to configuration file
        include_deprecated: Whether to include deprecated code detection
        include_dynamic_imports: Whether to include dynamic import analysis
        include_conditional: Whether to include conditional dead code detection

    Returns:
        Dictionary with dead code mapping results
    """
    directory = str(Path(path))
    config = _load_cq_config(config_path)

    analyzer = DeadCodeAnalyzer(config)
    
    if Path(path).is_file():
        # Analyze single file
        issues = analyzer.analyze_file(path)
        deprecated_issues = analyzer.detect_deprecated_code(path) if include_deprecated else []
        dynamic_issues = analyzer.detect_dynamic_imports(path) if include_dynamic_imports else []
        conditional_issues = analyzer.detect_conditional_dead_code(path) if include_conditional else []
        
        all_issues = issues + dynamic_issues + conditional_issues
        impact_analysis = analyzer.analyze_removal_impact(all_issues)
        
        return {
            "file_path": path,
            "dead_code_issues": [issue.__dict__ for issue in all_issues],
            "deprecated_issues": [issue.__dict__ for issue in deprecated_issues],
            "impact_analysis": impact_analysis,
            "summary": {
                "total_issues": len(all_issues),
                "deprecated_count": len(deprecated_issues),
                "dynamic_import_count": len(dynamic_issues),
                "conditional_dead_count": len(conditional_issues),
                "high_impact_count": len(impact_analysis.get("high_impact", [])),
                "medium_impact_count": len(impact_analysis.get("medium_impact", [])),
                "low_impact_count": len(impact_analysis.get("low_impact", [])),
            }
        }
    else:
        # Analyze directory
        report = analyzer.analyze_directory(directory)
        
        return {
            "directory_path": directory,
            "dead_code_report": {
                "total_issues": report.total_issues,
                "issues_by_type": report.issues_by_type,
                "issues_by_severity": {
                    severity: [issue.__dict__ for issue in issues]
                    for severity, issues in report.issues_by_severity.items()
                },
                "potential_savings": report.potential_savings,
            },
            "deprecated_issues": [issue.__dict__ for issue in (report.deprecated_issues or [])],
            "impact_analysis": report.impact_analysis or {},
            "summary": {
                "total_issues": report.total_issues,
                "deprecated_count": len(report.deprecated_issues or []),
                "files_affected": len(report.issues_by_file),
                "high_impact_count": len((report.impact_analysis or {}).get("high_impact", [])),
                "medium_impact_count": len((report.impact_analysis or {}).get("medium_impact", [])),
                "low_impact_count": len((report.impact_analysis or {}).get("low_impact", [])),
                "potential_lines_removed": report.potential_savings.get("total_lines", 0),
            }
        }


def export_dead_code_mapping(
    path: str,
    output_file: str,
    config_path: str | None = None,
    format: str = "json",
) -> str:
    """
    Export dead code mapping to a file.

    Args:
        path: Path to directory or file to analyze
        output_file: Output file path
        config_path: Optional path to configuration file
        format: Output format ("json", "csv", "text")

    Returns:
        Path to the exported file
    """
    mapping_data = map_dead_code(path, config_path)
    
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2, default=str)
    elif format.lower() == "csv":
        _export_to_csv(mapping_data, out_path)
    elif format.lower() == "text":
        _export_to_text(mapping_data, out_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return str(out_path)


def _export_to_csv(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export mapping data to CSV format."""
    import csv
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "File", "Line", "Type", "Description", "Confidence", 
            "Severity", "Impact", "Dependencies"
        ])
        
        # Write dead code issues
        for issue in mapping_data.get("dead_code_issues", []):
            writer.writerow([
                issue.get("file_path", ""),
                issue.get("line_number", ""),
                issue.get("issue_type", ""),
                issue.get("description", ""),
                issue.get("confidence", ""),
                issue.get("severity", ""),
                issue.get("removal_impact", ""),
                ",".join(issue.get("dependencies", [])),
            ])
        
        # Write deprecated issues
        for issue in mapping_data.get("deprecated_issues", []):
            writer.writerow([
                issue.get("file_path", ""),
                issue.get("line_number", ""),
                f"deprecated_{issue.get('deprecated_type', '')}",
                issue.get("description", ""),
                "100",  # Deprecated code has high confidence
                issue.get("severity", ""),
                "medium",  # Deprecated code has medium impact
                issue.get("alternative", ""),
            ])


def _export_to_text(mapping_data: dict[str, Any], output_path: Path) -> None:
    """Export mapping data to text format."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("DEAD CODE MAPPING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary
        summary = mapping_data.get("summary", {})
        f.write("SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Issues: {summary.get('total_issues', 0)}\n")
        f.write(f"Deprecated Code: {summary.get('deprecated_count', 0)}\n")
        f.write(f"High Impact: {summary.get('high_impact_count', 0)}\n")
        f.write(f"Medium Impact: {summary.get('medium_impact_count', 0)}\n")
        f.write(f"Low Impact: {summary.get('low_impact_count', 0)}\n")
        f.write(f"Potential Lines Removed: {summary.get('potential_lines_removed', 0)}\n\n")
        
        # Dead code issues
        f.write("DEAD CODE ISSUES\n")
        f.write("-" * 20 + "\n")
        for issue in mapping_data.get("dead_code_issues", []):
            f.write(f"\n{issue.get('file_path', '')}:{issue.get('line_number', '')}\n")
            f.write(f"  Type: {issue.get('issue_type', '')}\n")
            f.write(f"  Description: {issue.get('description', '')}\n")
            f.write(f"  Confidence: {issue.get('confidence', '')}%\n")
            f.write(f"  Severity: {issue.get('severity', '')}\n")
            f.write(f"  Impact: {issue.get('removal_impact', '')}\n")
        
        # Deprecated issues
        f.write("\n\nDEPRECATED CODE ISSUES\n")
        f.write("-" * 25 + "\n")
        for issue in mapping_data.get("deprecated_issues", []):
            f.write(f"\n{issue.get('file_path', '')}:{issue.get('line_number', '')}\n")
            f.write(f"  Type: {issue.get('deprecated_type', '')}\n")
            f.write(f"  Description: {issue.get('description', '')}\n")
            f.write(f"  Reason: {issue.get('deprecation_reason', '')}\n")
            f.write(f"  Removal Version: {issue.get('removal_version', 'N/A')}\n")
            f.write(f"  Alternative: {issue.get('alternative', 'N/A')}\n")


def get_removal_recommendations(
    path: str,
    config_path: str | None = None,
    max_recommendations: int = 10,
) -> list[dict[str, Any]]:
    """
    Get prioritized removal recommendations for dead code.

    Args:
        path: Path to directory or file to analyze
        config_path: Optional path to configuration file
        max_recommendations: Maximum number of recommendations to return

    Returns:
        List of removal recommendations sorted by priority
    """
    mapping_data = map_dead_code(path, config_path)
    impact_analysis = mapping_data.get("impact_analysis", {})
    
    recommendations = []
    
    # High impact recommendations first
    for issue in impact_analysis.get("high_impact", [])[:max_recommendations//2]:
        recommendations.append({
            "priority": "high",
            "file_path": issue.get("file_path", ""),
            "line_number": issue.get("line_number", ""),
            "description": issue.get("description", ""),
            "impact_score": impact_analysis.get("total_impact_score", 0),
            "reason": "High impact dead code - safe to remove with high confidence",
            "action": "Remove immediately"
        })
    
    # Medium impact recommendations
    for issue in impact_analysis.get("medium_impact", [])[:max_recommendations//3]:
        recommendations.append({
            "priority": "medium",
            "file_path": issue.get("file_path", ""),
            "line_number": issue.get("line_number", ""),
            "description": issue.get("description", ""),
            "impact_score": impact_analysis.get("total_impact_score", 0),
            "reason": "Medium impact dead code - review before removal",
            "action": "Review and remove if confirmed unused"
        })
    
    # Deprecated code recommendations
    for issue in mapping_data.get("deprecated_issues", [])[:max_recommendations//4]:
        recommendations.append({
            "priority": "deprecated",
            "file_path": issue.get("file_path", ""),
            "line_number": issue.get("line_number", ""),
            "description": issue.get("description", ""),
            "deprecation_reason": issue.get("deprecation_reason", ""),
            "removal_version": issue.get("removal_version", ""),
            "alternative": issue.get("alternative", ""),
            "reason": "Deprecated code - plan for removal",
            "action": f"Replace with alternative: {issue.get('alternative', 'N/A')}"
        })
    
    return recommendations[:max_recommendations]