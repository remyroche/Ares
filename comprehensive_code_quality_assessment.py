#!/usr/bin/env python3
"""
Comprehensive Code Quality Assessment Script
Analyzes the repository for code quality issues and generates a detailed report.
"""

import ast
import json
import os
from collections import Counter
from pathlib import Path


class CodeQualityAnalyzer:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.results = {
            "summary": {},
            "syntax_errors": [],
            "style_issues": [],
            "complexity_metrics": {},
            "import_analysis": {},
            "file_metrics": {},
            "recommendations": [],
        }

    def analyze_repository(self):
        """Run comprehensive code quality analysis."""
        print("🔍 Starting comprehensive code quality assessment...")

        # Find all Python files
        python_files = self._find_python_files()
        print(f"📁 Found {len(python_files)} Python files")

        # Analyze syntax
        self._analyze_syntax(python_files)

        # Analyze imports
        self._analyze_imports(python_files)

        # Analyze complexity
        self._analyze_complexity(python_files)

        # Analyze file metrics
        self._analyze_file_metrics(python_files)

        # Generate recommendations
        self._generate_recommendations()

        # Generate summary
        self._generate_summary()

        return self.results

    def _find_python_files(self) -> list[Path]:
        """Find all Python files in the repository."""
        python_files = []
        exclude_patterns = [
            "__pycache__", ".git", "venv", "env", "node_modules",
            ".pytest_cache", "code_quality_env",
        ]

        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]

            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def _analyze_syntax(self, python_files: list[Path]):
        """Analyze syntax errors in Python files."""
        print("🔧 Analyzing syntax errors...")

        syntax_errors = []
        valid_files = 0
        invalid_files = 0

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Try to parse with AST
                ast.parse(content)
                valid_files += 1

            except SyntaxError as e:
                invalid_files += 1
                syntax_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "message": str(e.msg),
                    "type": "syntax_error",
                })
            except Exception as e:
                invalid_files += 1
                syntax_errors.append({
                    "file": str(file_path),
                    "line": 0,
                    "column": 0,
                    "message": str(e),
                    "type": "other_error",
                })

        self.results["syntax_errors"] = syntax_errors
        self.results["summary"]["total_files"] = len(python_files)
        self.results["summary"]["valid_files"] = valid_files
        self.results["summary"]["invalid_files"] = invalid_files
        self.results["summary"]["error_rate"] = (invalid_files / len(python_files)) * 100 if python_files else 0

        print(f"✅ Valid files: {valid_files}")
        print(f"❌ Invalid files: {invalid_files}")
        print(f"📊 Error rate: {self.results['summary']['error_rate']:.2f}%")

    def _analyze_imports(self, python_files: list[Path]):
        """Analyze import statements and dependencies."""
        print("📦 Analyzing imports and dependencies...")

        import_stats = {
            "total_imports": 0,
            "import_errors": 0,
            "unused_imports": 0,
            "import_patterns": Counter(),
            "problematic_imports": [],
        }

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                # Count imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import | ast.ImportFrom):
                        import_stats["total_imports"] += 1

                        if isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            import_stats["import_patterns"][f"from {module} import"] += 1
                        else:
                            import_stats["import_patterns"]["import"] += 1

            except Exception as e:
                import_stats["import_errors"] += 1
                import_stats["problematic_imports"].append({
                    "file": str(file_path),
                    "error": str(e),
                })

        self.results["import_analysis"] = import_stats
        print(f"📊 Total imports: {import_stats['total_imports']}")
        print(f"❌ Import errors: {import_stats['import_errors']}")

    def _analyze_complexity(self, python_files: list[Path]):
        """Analyze code complexity metrics."""
        print("🧮 Analyzing code complexity...")

        complexity_stats = {
            "total_functions": 0,
            "total_classes": 0,
            "high_complexity_functions": 0,
            "complexity_distribution": Counter(),
            "file_complexity": {},
        }

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_complexity = 0

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity_stats["total_functions"] += 1
                        complexity = self._calculate_function_complexity(node)
                        complexity_stats["complexity_distribution"][complexity] += 1
                        file_complexity += complexity

                        if complexity > 10:
                            complexity_stats["high_complexity_functions"] += 1

                    elif isinstance(node, ast.ClassDef):
                        complexity_stats["total_classes"] += 1

                complexity_stats["file_complexity"][str(file_path)] = file_complexity

            except Exception:
                # Skip files with syntax errors
                pass

        self.results["complexity_metrics"] = complexity_stats
        print(f"🔧 Total functions: {complexity_stats['total_functions']}")
        print(f"🏗️ Total classes: {complexity_stats['total_classes']}")
        print(f"⚠️ High complexity functions: {complexity_stats['high_complexity_functions']}")

    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor | ast.AsyncWith | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _analyze_file_metrics(self, python_files: list[Path]):
        """Analyze file-level metrics."""
        print("📏 Analyzing file metrics...")

        file_metrics = {
            "total_lines": 0,
            "total_characters": 0,
            "average_file_size": 0,
            "largest_files": [],
            "file_size_distribution": Counter(),
        }

        file_sizes = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                lines = len(content.splitlines())
                chars = len(content)

                file_metrics["total_lines"] += lines
                file_metrics["total_characters"] += chars
                file_sizes.append((str(file_path), lines, chars))

                # Categorize file sizes
                if lines < 50:
                    file_metrics["file_size_distribution"]["small (<50 lines)"] += 1
                elif lines < 200:
                    file_metrics["file_size_distribution"]["medium (50-200 lines)"] += 1
                elif lines < 500:
                    file_metrics["file_size_distribution"]["large (200-500 lines)"] += 1
                else:
                    file_metrics["file_size_distribution"]["very large (>500 lines)"] += 1

            except Exception:
                pass

        # Calculate averages and find largest files
        if file_sizes:
            file_metrics["average_file_size"] = file_metrics["total_lines"] / len(file_sizes)
            file_sizes.sort(key=lambda x: x[1], reverse=True)
            file_metrics["largest_files"] = file_sizes[:10]

        self.results["file_metrics"] = file_metrics
        print(f"📊 Total lines: {file_metrics['total_lines']:,}")
        print(f"📏 Average file size: {file_metrics['average_file_size']:.1f} lines")

    def _generate_recommendations(self):
        """Generate actionable recommendations."""
        print("💡 Generating recommendations...")

        recommendations = []

        # Syntax error recommendations
        if self.results["summary"]["invalid_files"] > 0:
            recommendations.append({
                "priority": "CRITICAL",
                "category": "Syntax Errors",
                "description": f"Fix {self.results['summary']['invalid_files']} files with syntax errors",
                "action": "Run automated syntax fixes and manually review complex errors",
            })

        # Complexity recommendations
        if self.results["complexity_metrics"]["high_complexity_functions"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Code Complexity",
                "description": f"Refactor {self.results['complexity_metrics']['high_complexity_functions']} high-complexity functions",
                "action": "Break down complex functions into smaller, more manageable pieces",
            })

        # Import recommendations
        if self.results["import_analysis"]["import_errors"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "Import Issues",
                "description": f"Fix {self.results['import_analysis']['import_errors']} import-related errors",
                "action": "Review and fix import statements, ensure all dependencies are available",
            })

        # General recommendations
        recommendations.extend([
            {
                "priority": "MEDIUM",
                "category": "Code Standards",
                "description": "Implement consistent code formatting",
                "action": "Use Black, isort, and flake8 for consistent code style",
            },
            {
                "priority": "MEDIUM",
                "category": "Testing",
                "description": "Add unit tests for critical functions",
                "action": "Implement pytest and ensure high test coverage",
            },
            {
                "priority": "LOW",
                "category": "Documentation",
                "description": "Improve code documentation",
                "action": "Add docstrings and type hints to functions and classes",
            },
        ])

        self.results["recommendations"] = recommendations

    def _generate_summary(self):
        """Generate executive summary."""
        summary = self.results["summary"]

        # Calculate quality score (0-100)
        if summary["total_files"] > 0:
            valid_ratio = summary["valid_files"] / summary["total_files"]
            quality_score = valid_ratio * 100
        else:
            quality_score = 0

        summary["quality_score"] = quality_score

        # Determine overall status
        if quality_score >= 95:
            status = "EXCELLENT"
        elif quality_score >= 85:
            status = "GOOD"
        elif quality_score >= 70:
            status = "FAIR"
        elif quality_score >= 50:
            status = "POOR"
        else:
            status = "CRITICAL"

        summary["overall_status"] = status

    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive report."""
        report_lines = []

        # Header
        report_lines.append("# Comprehensive Code Quality Assessment Report")
        report_lines.append("")
        report_lines.append(f"**Generated**: {Path.cwd()}")
        report_lines.append(f"**Repository**: {self.repo_path}")
        report_lines.append("")

        # Executive Summary
        summary = self.results["summary"]
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append(f"**Overall Status**: {summary['overall_status']}")
        report_lines.append(f"**Quality Score**: {summary['quality_score']:.1f}/100")
        report_lines.append(f"**Total Python Files**: {summary['total_files']:,}")
        report_lines.append(f"**Valid Files**: {summary['valid_files']:,}")
        report_lines.append(f"**Files with Errors**: {summary['invalid_files']:,}")
        report_lines.append(f"**Error Rate**: {summary['error_rate']:.2f}%")
        report_lines.append("")

        # Syntax Analysis
        if self.results["syntax_errors"]:
            report_lines.append("## Syntax Error Analysis")
            report_lines.append("")
            report_lines.append(f"**Critical Issues**: {len(self.results['syntax_errors'])} files have syntax errors")
            report_lines.append("")

            # Group errors by type
            error_types = Counter(error["type"] for error in self.results["syntax_errors"])
            report_lines.append("### Error Distribution by Type")
            for error_type, count in error_types.most_common():
                report_lines.append(f"- **{error_type}**: {count} files")
            report_lines.append("")

            # Show first 10 errors
            report_lines.append("### Sample Syntax Errors")
            for error in self.results["syntax_errors"][:10]:
                report_lines.append(f"- `{error['file']}`: Line {error['line']} - {error['message']}")
            report_lines.append("")

        # Complexity Analysis
        complexity = self.results["complexity_metrics"]
        report_lines.append("## Code Complexity Analysis")
        report_lines.append("")
        report_lines.append(f"**Total Functions**: {complexity['total_functions']:,}")
        report_lines.append(f"**Total Classes**: {complexity['total_classes']:,}")
        report_lines.append(f"**High Complexity Functions (>10)**: {complexity['high_complexity_functions']:,}")
        report_lines.append("")

        if complexity["complexity_distribution"]:
            report_lines.append("### Complexity Distribution")
            for comp, count in sorted(complexity["complexity_distribution"].items()):
                report_lines.append(f"- **Complexity {comp}**: {count} functions")
            report_lines.append("")

        # File Metrics
        metrics = self.results["file_metrics"]
        report_lines.append("## File Metrics")
        report_lines.append("")
        report_lines.append(f"**Total Lines of Code**: {metrics['total_lines']:,}")
        report_lines.append(f"**Total Characters**: {metrics['total_characters']:,}")
        report_lines.append(f"**Average File Size**: {metrics['average_file_size']:.1f} lines")
        report_lines.append("")

        if metrics["file_size_distribution"]:
            report_lines.append("### File Size Distribution")
            for size_cat, count in metrics["file_size_distribution"].items():
                report_lines.append(f"- **{size_cat}**: {count} files")
            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")

        for rec in self.results["recommendations"]:
            report_lines.append(f"### {rec['priority']} Priority: {rec['category']}")
            report_lines.append(f"**Issue**: {rec['description']}")
            report_lines.append(f"**Action**: {rec['action']}")
            report_lines.append("")

        # Next Steps
        report_lines.append("## Next Steps")
        report_lines.append("")
        report_lines.append("1. **Immediate**: Fix critical syntax errors")
        report_lines.append("2. **Short-term**: Address high-priority recommendations")
        report_lines.append("3. **Medium-term**: Implement code quality tools and standards")
        report_lines.append("4. **Long-term**: Establish quality gates and monitoring")
        report_lines.append("")

        report_lines.append("---")
        report_lines.append("*Report generated by Code Quality Assessment Script*")

        report_content = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_content)
            print(f"📄 Report saved to: {output_file}")

        return report_content

def main():
    """Main function to run the code quality assessment."""
    analyzer = CodeQualityAnalyzer()
    results = analyzer.analyze_repository()

    # Generate and display report
    analyzer.generate_report("comprehensive_code_quality_report.md")
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE CODE QUALITY ASSESSMENT COMPLETE")
    print("="*80)
    print(f"Overall Quality Score: {results['summary']['quality_score']:.1f}/100")
    print(f"Status: {results['summary']['overall_status']}")
    print(f"Files with Errors: {results['summary']['invalid_files']}")
    print("="*80)

    # Save detailed results as JSON
    with open("code_quality_assessment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("📁 Detailed results saved to: code_quality_assessment_results.json")

if __name__ == "__main__":
    main()
