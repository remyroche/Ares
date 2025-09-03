"""
Main Quality Reporter - Orchestrates all code quality analysis tools and generates comprehensive reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..analyzers.call_graph_analyzer import CallGraphAnalyzer
from ..analyzers.dependency_analyzer import DependencyAnalyzer
from ..analyzers.linter_analyzer import LinterAnalyzer
from ..analyzers.syntax_validator import SyntaxValidator
from ..core.config import CodeQualityConfig, get_default_config
from ..fixers.auto_fixer import AutoFixer
from ..utils.file_utils import get_directory_stats


class QualityReporter:
    """
    Main quality reporter that orchestrates all analysis tools and generates comprehensive reports.
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None

    def generate_comprehensive_report(self, directory: str,
                                   run_auto_fix: bool = False,
                                   output_dir: str | None = None) -> dict[str, Any]:
        """
        Generate a comprehensive code quality report.

        Args:
            directory: Directory to analyze
            run_auto_fix: Whether to run auto-fixing before analysis
            output_dir: Directory to save reports

        Returns:
            Comprehensive quality report
        """
        self.start_time = time.time()
        print("="*60)
        print("CODE QUALITY COMPREHENSIVE ANALYSIS")
        print("="*60)
        print(f"Analyzing directory: {directory}")
        print(f"Configuration: {self.config.reporting.output_format}")
        print(f"Auto-fix enabled: {run_auto_fix}")

        # Step 1: Run auto-fixer if requested
        if run_auto_fix and self.config.auto_fix.enabled:
            print("\n" + "-"*40)
            print("STEP 1: AUTO-FIXING")
            print("-"*40)
            fixer = AutoFixer(self.config)
            fix_results = fixer.fix_all(directory)
            self.results["auto_fix"] = fix_results
            print("Auto-fixing completed.")

        # Step 2: Syntax validation
        print("\n" + "-"*40)
        print("STEP 2: SYNTAX VALIDATION")
        print("-"*40)
        syntax_validator = SyntaxValidator(self.config)
        syntax_results = syntax_validator.validate_directory(directory)
        self.results["syntax_validation"] = syntax_results
        print("Syntax validation completed.")

        # Step 3: Linter analysis
        print("\n" + "-"*40)
        print("STEP 3: LINTER ANALYSIS")
        print("-"*40)
        linter_analyzer = LinterAnalyzer(self.config)
        linter_results = linter_analyzer.analyze_directory(directory)
        self.results["linter_analysis"] = linter_results
        print("Linter analysis completed.")

        # Step 4: Call graph analysis
        print("\n" + "-"*40)
        print("STEP 4: CALL GRAPH ANALYSIS")
        print("-"*40)
        call_graph_analyzer = CallGraphAnalyzer(self.config)
        call_graph_results = call_graph_analyzer.analyze_directory(directory)
        self.results["call_graph_analysis"] = call_graph_results
        print("Call graph analysis completed.")

        # Step 5: Dependency analysis
        print("\n" + "-"*40)
        print("STEP 5: DEPENDENCY ANALYSIS")
        print("-"*40)
        dependency_analyzer = DependencyAnalyzer(self.config)
        dependency_results = dependency_analyzer.analyze_directory(directory)
        self.results["dependency_analysis"] = dependency_results
        print("Dependency analysis completed.")

        # Step 6: File statistics
        print("\n" + "-"*40)
        print("STEP 6: FILE STATISTICS")
        print("-"*40)
        file_stats = get_directory_stats(directory, self.config.analysis.exclude_patterns)
        self.results["file_statistics"] = file_stats
        print("File statistics completed.")

        # Step 7: Generate comprehensive report
        print("\n" + "-"*40)
        print("STEP 7: GENERATING COMPREHENSIVE REPORT")
        print("-"*40)

        self.end_time = time.time()
        comprehensive_report = self._generate_comprehensive_report(directory)
        self.results["comprehensive_report"] = comprehensive_report

        # Step 8: Save reports
        if output_dir:
            self._save_reports(output_dir)

        # Step 9: Print summary
        self._print_comprehensive_summary()

        return comprehensive_report

    def _generate_comprehensive_report(self, directory: str) -> dict[str, Any]:
        """Generate a comprehensive report combining all analysis results."""
        # Calculate overall quality score
        quality_score = self._calculate_quality_score()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Calculate metrics
        metrics = self._calculate_metrics()

        return {
            "analysis_info": {
                "directory": directory,
                "timestamp": datetime.now().isoformat(),
                "analysis_duration": self.end_time - self.start_time if self.end_time else 0,
                "config_used": {
                    "auto_fix_enabled": self.config.auto_fix.enabled,
                    "linters_used": self.config.analysis.linters,
                    "complexity_threshold": self.config.analysis.complexity_threshold,
                },
            },
            "quality_score": quality_score,
            "metrics": metrics,
            "recommendations": recommendations,
            "summary": {
                "total_files": self.results.get("file_statistics", {}).get("total_files", 0),
                "total_errors": self.results.get("linter_analysis", {}).get("total_issues", 0),
                "syntax_errors": self.results.get("syntax_validation", {}).get("summary", {}).get("total_errors", 0),
                "potential_dead_code": len(self.results.get("call_graph_analysis", {}).get("potential_dead_code", [])),
                "missing_dependencies": len(self.results.get("dependency_analysis", {}).get("missing_dependencies_list", [])),
                "unused_dependencies": len(self.results.get("dependency_analysis", {}).get("unused_dependencies_list", [])),
            },
        }


    def _calculate_quality_score(self) -> dict[str, Any]:
        """Calculate an overall quality score based on all analysis results."""
        scores = {}
        weights = {
            "syntax": 0.25,
            "linting": 0.25,
            "dead_code": 0.20,
            "dependencies": 0.15,
            "style": 0.15,
        }

        # Syntax score (0-100)
        syntax_validation = self.results.get("syntax_validation", {})
        total_files = syntax_validation.get("summary", {}).get("total_files", 1)
        valid_files = syntax_validation.get("summary", {}).get("valid_files", 0)
        syntax_score = (valid_files / total_files) * 100 if total_files > 0 else 100
        scores["syntax"] = syntax_score

        # Linting score (0-100)
        linter_analysis = self.results.get("linter_analysis", {})
        total_issues = linter_analysis.get("total_issues", 0)
        # Normalize to 0-100 scale (fewer issues = higher score)
        linting_score = max(0, 100 - min(total_issues * 2, 100))
        scores["linting"] = linting_score

        # Dead code score (0-100)
        call_graph = self.results.get("call_graph_analysis", {})
        total_functions = call_graph.get("total_nodes", 0)
        dead_code_count = len(call_graph.get("potential_dead_code", []))
        dead_code_score = max(0, 100 - (dead_code_count / max(total_functions, 1)) * 100)
        scores["dead_code"] = dead_code_score

        # Dependencies score (0-100)
        dependency_analysis = self.results.get("dependency_analysis", {})
        total_deps = dependency_analysis.get("total_dependencies", 1)
        missing_deps = len(dependency_analysis.get("missing_dependencies_list", []))
        unused_deps = len(dependency_analysis.get("unused_dependencies_list", []))
        dependency_score = max(0, 100 - ((missing_deps + unused_deps) / max(total_deps, 1)) * 100)
        scores["dependencies"] = dependency_score

        # Style score (0-100)
        style_issues = 0
        for error in syntax_validation.get("syntax_errors", []):
            if error.get("error_type") == "style_issue":
                style_issues += 1
        style_score = max(0, 100 - min(style_issues * 5, 100))
        scores["style"] = style_score

        # Calculate weighted average
        weighted_score = sum(scores[category] * weights[category] for category in scores)

        # Determine grade
        if weighted_score >= 90:
            grade = "A"
        elif weighted_score >= 80:
            grade = "B"
        elif weighted_score >= 70:
            grade = "C"
        elif weighted_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return {
            "overall_score": round(weighted_score, 2),
            "grade": grade,
            "category_scores": scores,
            "weights": weights,
        }

    def _generate_recommendations(self) -> list[dict[str, Any]]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []

        # Syntax recommendations
        syntax_validation = self.results.get("syntax_validation", {})
        if syntax_validation.get("summary", {}).get("total_errors", 0) > 0:
            recommendations.append({
                "category": "syntax",
                "priority": "high",
                "title": "Fix Syntax Errors",
                "description": f"Found {syntax_validation['summary']['total_errors']} syntax errors that prevent code from running.",
                "action": "Run the syntax validator and fix all reported errors.",
                "impact": "Critical - Code cannot execute with syntax errors.",
            })

        # Linting recommendations
        linter_analysis = self.results.get("linter_analysis", {})
        if linter_analysis.get("total_issues", 0) > 0:
            recommendations.append({
                "category": "quality",
                "priority": "medium",
                "title": "Address Linting Issues",
                "description": f"Found {linter_analysis['total_issues']} linting issues affecting code quality.",
                "action": "Review and fix linting issues, especially high-priority ones.",
                "impact": "Medium - Affects code maintainability and style consistency.",
            })

        # Dead code recommendations
        call_graph = self.results.get("call_graph_analysis", {})
        dead_code_count = len(call_graph.get("potential_dead_code", []))
        if dead_code_count > 0:
            recommendations.append({
                "category": "maintenance",
                "priority": "low",
                "title": "Remove Dead Code",
                "description": f"Found {dead_code_count} potentially unused functions/methods.",
                "action": "Review dead code candidates and remove if confirmed unused.",
                "impact": "Low - Reduces codebase size and improves maintainability.",
            })

        # Dependency recommendations
        dependency_analysis = self.results.get("dependency_analysis", {})
        missing_deps = len(dependency_analysis.get("missing_dependencies_list", []))
        unused_deps = len(dependency_analysis.get("unused_dependencies_list", []))

        if missing_deps > 0:
            recommendations.append({
                "category": "dependencies",
                "priority": "high",
                "title": "Install Missing Dependencies",
                "description": f"Found {missing_deps} missing dependencies that may cause runtime errors.",
                "action": "Install missing packages or add them to requirements.txt.",
                "impact": "High - Missing dependencies will cause import errors.",
            })

        if unused_deps > 0:
            recommendations.append({
                "category": "dependencies",
                "priority": "low",
                "title": "Clean Up Unused Dependencies",
                "description": f"Found {unused_deps} unused dependencies in requirements files.",
                "action": "Remove unused dependencies to reduce package bloat.",
                "impact": "Low - Reduces package size and potential security vulnerabilities.",
            })

        # Style recommendations
        style_issues = 0
        for error in syntax_validation.get("syntax_errors", []):
            if error.get("error_type") == "style_issue":
                style_issues += 1

        if style_issues > 0:
            recommendations.append({
                "category": "style",
                "priority": "low",
                "title": "Improve Code Style",
                "description": f"Found {style_issues} style issues affecting code consistency.",
                "action": "Run auto-fixers (black, isort) to standardize code style.",
                "impact": "Low - Improves code readability and consistency.",
            })

        return recommendations

    def _calculate_metrics(self) -> dict[str, Any]:
        """Calculate various code quality metrics."""
        metrics = {}

        # File metrics
        file_stats = self.results.get("file_statistics", {})
        metrics["files"] = {
            "total": file_stats.get("total_files", 0),
            "valid": file_stats.get("valid_files", 0),
            "invalid": file_stats.get("invalid_files", 0),
            "total_lines": file_stats.get("total_lines", 0),
            "total_functions": file_stats.get("total_functions", 0),
            "total_classes": file_stats.get("total_classes", 0),
        }

        # Complexity metrics
        call_graph = self.results.get("call_graph_analysis", {})
        metrics["complexity"] = {
            "total_nodes": call_graph.get("total_nodes", 0),
            "total_edges": call_graph.get("total_edges", 0),
            "circular_dependencies": len(call_graph.get("circular_dependencies", [])),
            "graph_density": call_graph.get("graph_metrics", {}).get("density", 0),
        }

        # Quality metrics
        linter_analysis = self.results.get("linter_analysis", {})
        metrics["quality"] = {
            "total_issues": linter_analysis.get("total_issues", 0),
            "errors": linter_analysis.get("total_errors", 0),
            "warnings": linter_analysis.get("total_warnings", 0),
            "issues_per_file": linter_analysis.get("total_issues", 0) / max(file_stats.get("total_files", 1), 1),
        }

        # Dependency metrics
        dependency_analysis = self.results.get("dependency_analysis", {})
        metrics["dependencies"] = {
            "total": dependency_analysis.get("total_dependencies", 0),
            "installed": dependency_analysis.get("installed_dependencies", 0),
            "missing": dependency_analysis.get("missing_dependencies", 0),
            "unused": dependency_analysis.get("unused_dependencies", 0),
            "used": dependency_analysis.get("used_dependencies", 0),
        }

        return metrics

    def _save_reports(self, output_dir: str) -> None:
        """Save all reports to the specified output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving reports to: {output_path}")

        # Save comprehensive report
        comprehensive_file = output_path / "comprehensive_quality_report.json"
        with open(comprehensive_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Comprehensive report saved: {comprehensive_file}")

        # Save individual reports
        if "syntax_validation" in self.results:
            syntax_file = output_path / "syntax_validation_report.json"
            with open(syntax_file, "w") as f:
                json.dump(self.results["syntax_validation"], f, indent=2)
            print(f"Syntax validation report saved: {syntax_file}")

        if "linter_analysis" in self.results:
            linter_file = output_path / "linter_analysis_report.json"
            with open(linter_file, "w") as f:
                json.dump(self.results["linter_analysis"], f, indent=2)
            print(f"Linter analysis report saved: {linter_file}")

        if "call_graph_analysis" in self.results:
            call_graph_file = output_path / "call_graph_analysis_report.json"
            with open(call_graph_file, "w") as f:
                json.dump(self.results["call_graph_analysis"], f, indent=2)
            print(f"Call graph analysis report saved: {call_graph_file}")

        if "dependency_analysis" in self.results:
            dependency_file = output_path / "dependency_analysis_report.json"
            with open(dependency_file, "w") as f:
                json.dump(self.results["dependency_analysis"], f, indent=2)
            print(f"Dependency analysis report saved: {dependency_file}")

        # Generate HTML report if requested
        if "html" in self.config.reporting.output_format:
            html_file = output_path / "quality_report.html"
            self._generate_html_report(html_file)
            print(f"HTML report saved: {html_file}")

    def _generate_html_report(self, output_path: str) -> None:
        """Generate an HTML version of the quality report."""
        html_content = self._get_html_template()

        # Replace placeholders with actual data
        comprehensive = self.results.get("comprehensive_report", {})

        html_content = html_content.replace("{{TIMESTAMP}}", comprehensive.get("analysis_info", {}).get("timestamp", ""))
        html_content = html_content.replace("{{QUALITY_SCORE}}", str(comprehensive.get("quality_score", {}).get("overall_score", 0)))
        html_content = html_content.replace("{{GRADE}}", comprehensive.get("quality_score", {}).get("grade", "N/A"))

        # Add metrics
        metrics = comprehensive.get("metrics", {})
        metrics_html = ""
        for category, values in metrics.items():
            metrics_html += f"<h3>{category.title()}</h3><ul>"
            for key, value in values.items():
                metrics_html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
            metrics_html += "</ul>"

        html_content = html_content.replace("{{METRICS}}", metrics_html)

        # Add recommendations
        recommendations = comprehensive.get("recommendations", [])
        recs_html = ""
        for rec in recommendations:
            recs_html += f"""
            <div class="recommendation {rec['priority']}">
                <h4>{rec['title']}</h4>
                <p><strong>Priority:</strong> {rec['priority'].title()}</p>
                <p><strong>Description:</strong> {rec['description']}</p>
                <p><strong>Action:</strong> {rec['action']}</p>
                <p><strong>Impact:</strong> {rec['impact']}</p>
            </div>
            """

        html_content = html_content.replace("{{RECOMMENDATIONS}}", recs_html)

        # Save HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

    def _get_html_template(self) -> str:
        """Get the HTML template for the quality report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Quality Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }
        .score { font-size: 48px; font-weight: bold; color: #2c3e50; }
        .grade { font-size: 24px; color: #7f8c8d; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }
        .recommendation { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .recommendation.high { border-left: 5px solid #e74c3c; background-color: #fdf2f2; }
        .recommendation.medium { border-left: 5px solid #f39c12; background-color: #fef9e7; }
        .recommendation.low { border-left: 5px solid #27ae60; background-color: #f0f9f0; }
        .timestamp { color: #7f8c8d; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Code Quality Report</h1>
            <div class="score">{{QUALITY_SCORE}}</div>
            <div class="grade">Grade: {{GRADE}}</div>
            <div class="timestamp">Generated: {{TIMESTAMP}}</div>
        </div>

        <h2>Metrics</h2>
        <div class="metrics">
            {{METRICS}}
        </div>

        <h2>Recommendations</h2>
        {{RECOMMENDATIONS}}
    </div>
</body>
</html>
        """

    def _print_comprehensive_summary(self) -> None:
        """Print a comprehensive summary of all analysis results."""
        comprehensive = self.results.get("comprehensive_report", {})

        print("\n" + "="*60)
        print("COMPREHENSIVE QUALITY REPORT SUMMARY")
        print("="*60)

        # Quality score
        quality_score = comprehensive.get("quality_score", {})
        print(f"Overall Quality Score: {quality_score.get('overall_score', 0)}/100")
        print(f"Grade: {quality_score.get('grade', 'N/A')}")

        # Category scores
        print("\nCategory Scores:")
        for category, score in quality_score.get("category_scores", {}).items():
            print(f"  {category.title()}: {score:.1f}/100")

        # Summary metrics
        summary = comprehensive.get("summary", {})
        print("\nSummary:")
        print(f"  Total files: {summary.get('total_files', 0)}")
        print(f"  Total errors: {summary.get('total_errors', 0)}")
        print(f"  Syntax errors: {summary.get('syntax_errors', 0)}")
        print(f"  Potential dead code: {summary.get('potential_dead_code', 0)}")
        print(f"  Missing dependencies: {summary.get('missing_dependencies', 0)}")
        print(f"  Unused dependencies: {summary.get('unused_dependencies', 0)}")

        # Top recommendations
        recommendations = comprehensive.get("recommendations", [])
        if recommendations:
            print("\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
                print(f"     {rec['description']}")

        # Analysis duration
        duration = comprehensive.get("analysis_info", {}).get("analysis_duration", 0)
        print(f"\nAnalysis completed in {duration:.2f} seconds")


def main():
    """Command-line interface for the quality reporter."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate comprehensive code quality report")
    parser.add_argument("--path", required=True, help="Path to directory containing Python files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--auto-fix", action="store_true", help="Run auto-fixing before analysis")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        from ..core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Run comprehensive analysis
    reporter = QualityReporter(config)
    return reporter.generate_comprehensive_report(
        directory=args.path,
        run_auto_fix=args.auto_fix,
        output_dir=args.output,
    )



if __name__ == "__main__":
    main()
