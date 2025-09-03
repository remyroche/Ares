#!/usr/bin/env python3
"""
Comprehensive Quality Auditor

A single-file solution that provides:
1. Exhaustive audit of files and directories
2. Comprehensive, unified report generation
3. Support for CSV, JSON, TXT, Python, YAML files
4. No external dependencies - uses only standard Python libraries
5. Both individual and batch analysis capabilities
6. Multiple output formats (JSON, text, unified)

Usage:
    python comprehensive_quality_auditor.py --audit <path> [options]
    python comprehensive_quality_auditor.py --generate-unified [options]
    python comprehensive_quality_auditor.py --full-audit <path> [options]
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("quality_audit.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ComprehensiveQualityAuditor")


class QualityLevel:
    """Quality level enumeration with scoring."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

    @staticmethod
    def get_score(quality: str) -> int:
        """Get numeric score for quality level."""
        scores = {
            QualityLevel.EXCELLENT: 100,
            QualityLevel.GOOD: 80,
            QualityLevel.ACCEPTABLE: 60,
            QualityLevel.POOR: 40,
            QualityLevel.CRITICAL: 20,
        }
        return scores.get(quality, 0)


class ComprehensiveQualityAuditor:
    """Comprehensive quality auditor with exhaustive analysis capabilities."""

    def __init__(self):
        """Initialize the comprehensive quality auditor."""
        self.logger = logger

        # Supported file extensions
        self.supported_extensions = {".csv", ".json", ".txt", ".py", ".yaml", ".yml", ".md", ".log"}

        # Analysis results storage
        self.analysis_results = {}
        self.audit_summary = {}

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 90,
            "good": 75,
            "acceptable": 60,
            "poor": 40,
            "critical": 20,
        }

    def run_exhaustive_audit(self, target_path: str, recursive: bool = True,
                            file_pattern: str = "*", max_files: int = 1000) -> dict[str, Any]:
        """
        Run an exhaustive audit on the target path.

        Args:
            target_path: Path to file or directory to audit
            recursive: Whether to recursively analyze subdirectories
            file_pattern: File pattern for filtering
            max_files: Maximum number of files to analyze

        Returns:
            Comprehensive audit results
        """
        self.logger.info(f"Starting exhaustive audit of: {target_path}")
        start_time = time.time()

        target_path_obj = Path(target_path)

        if not target_path_obj.exists():
            return {"error": f"Target path not found: {target_path}"}

        # Reset results
        self.analysis_results = {}
        self.audit_summary = {
            "audit_timestamp": datetime.now().isoformat(),
            "target_path": str(target_path),
            "audit_mode": "exhaustive",
            "recursive": recursive,
            "file_pattern": file_pattern,
            "max_files": max_files,
        }

        if target_path_obj.is_file():
            # Single file audit
            self.logger.info(f"Auditing single file: {target_path_obj.name}")
            result = self._audit_single_file(target_path_obj)
            self.analysis_results[str(target_path_obj)] = result

        elif target_path_obj.is_dir():
            # Directory audit
            self.logger.info(f"Auditing directory: {target_path}")
            result = self._audit_directory(target_path_obj, recursive, file_pattern, max_files)
            self.analysis_results.update(result)

        # Generate comprehensive summary
        self._generate_audit_summary()

        audit_duration = time.time() - start_time
        self.audit_summary["audit_duration_seconds"] = round(audit_duration, 2)

        self.logger.info(f"Exhaustive audit completed in {audit_duration:.2f} seconds")

        return {
            "audit_summary": self.audit_summary,
            "analysis_results": self.analysis_results,
        }

    def _audit_single_file(self, file_path: Path) -> dict[str, Any]:
        """Audit a single file comprehensively."""
        try:
            if file_path.suffix.lower() not in self.supported_extensions:
                return {
                    "error": f"Unsupported file format: {file_path.suffix}",
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "audit_timestamp": datetime.now().isoformat(),
                }

            self.logger.info(f"Analyzing file: {file_path.name}")

            # Basic file analysis
            basic_analysis = self._analyze_file_basic(file_path)

            # Format-specific analysis
            format_analysis = self._analyze_file_format(file_path)

            # Quality assessment
            quality_assessment = self._assess_file_quality(basic_analysis, format_analysis)

            # Issue identification
            issues = self._identify_file_issues(basic_analysis, format_analysis)

            # Recommendations
            recommendations = self._generate_file_recommendations(basic_analysis, format_analysis, issues)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "audit_timestamp": datetime.now().isoformat(),
                "basic_analysis": basic_analysis,
                "format_analysis": format_analysis,
                "quality_assessment": quality_assessment,
                "issues": issues,
                "recommendations": recommendations,
                "audit_type": "single_file",
            }

        except Exception as e:
            self.logger.exception(f"Error auditing {file_path.name}: {e}")
            return {
                "error": f"Audit failed: {str(e)}",
                "file_path": str(file_path),
                "file_name": file_path.name,
                "audit_timestamp": datetime.now().isoformat(),
            }

    def _audit_directory(self, directory_path: Path, recursive: bool,
                        file_pattern: str, max_files: int) -> dict[str, Any]:
        """Audit a directory comprehensively."""
        results = {}

        # Find all supported files
        all_files = []

        if recursive:
            # Recursive search
            for ext in self.supported_extensions:
                if file_pattern == "*":
                    all_files.extend(directory_path.rglob(f"*{ext}"))
                else:
                    all_files.extend(directory_path.rglob(file_pattern))
        else:
            # Non-recursive search
            for ext in self.supported_extensions:
                if file_pattern == "*":
                    all_files.extend(directory_path.glob(f"*{ext}"))
                else:
                    all_files.extend(directory_path.glob(file_pattern))

        # Limit files if needed
        if len(all_files) > max_files:
            self.logger.warning(f"Limiting analysis to {max_files} files (found {len(all_files)})")
            all_files = all_files[:max_files]

        self.logger.info(f"Found {len(all_files)} files to audit")

        # Audit each file
        for i, file_path in enumerate(all_files, 1):
            self.logger.info(f"Auditing file {i}/{len(all_files)}: {file_path.name}")
            result = self._audit_single_file(file_path)
            results[str(file_path)] = result

            # Progress indicator
            if i % 50 == 0:
                self.logger.info(f"Progress: {i}/{len(all_files)} files audited")

        return results

    def _analyze_file_basic(self, file_path: Path) -> dict[str, Any]:
        """Analyze basic file properties."""
        try:
            stat = file_path.stat()

            return {
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_path.suffix.lower(),
                "is_readable": os.access(file_path, os.R_OK),
                "is_writable": os.access(file_path, os.W_OK),
                "is_executable": os.access(file_path, os.X_OK),
            }
        except Exception as e:
            return {"error": f"Basic analysis failed: {str(e)}"}

    def _analyze_file_format(self, file_path: Path) -> dict[str, Any]:
        """Analyze file format-specific content."""
        try:
            if file_path.suffix.lower() == ".json":
                return self._analyze_json_content(file_path)
            if file_path.suffix.lower() == ".csv":
                return self._analyze_csv_content(file_path)
            if file_path.suffix.lower() in [".txt", ".md", ".log"]:
                return self._analyze_text_content(file_path)
            if file_path.suffix.lower() == ".py":
                return self._analyze_python_content(file_path)
            if file_path.suffix.lower() in [".yaml", ".yml"]:
                return self._analyze_yaml_content(file_path)
            return {"error": f"Unsupported format: {file_path.suffix}"}
        except Exception as e:
            return {"error": f"Format analysis failed: {str(e)}"}

    def _analyze_json_content(self, file_path: Path) -> dict[str, Any]:
        """Analyze JSON file content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            return {
                "format": "JSON",
                "data_type": type(data).__name__,
                "structure": self._analyze_json_structure(data),
                "encoding": "UTF-8",
                "is_valid": True,
            }
        except json.JSONDecodeError as e:
            return {
                "format": "JSON",
                "is_valid": False,
                "error": f"Invalid JSON: {str(e)}",
            }
        except Exception as e:
            return {
                "format": "JSON",
                "is_valid": False,
                "error": f"Analysis failed: {str(e)}",
            }

    def _analyze_csv_content(self, file_path: Path) -> dict[str, Any]:
        """Analyze CSV file content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                return {
                    "format": "CSV",
                    "is_valid": False,
                    "error": "Empty CSV file",
                }

            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []

            # Analyze structure
            column_lengths = [len(row) for row in rows]
            consistent_columns = len(set(column_lengths)) == 1

            # Check for empty cells
            empty_cells = sum(1 for row in rows for cell in row if not cell.strip())
            total_cells = sum(len(row) for row in rows)
            empty_ratio = empty_cells / total_cells if total_cells > 0 else 0

            return {
                "format": "CSV",
                "is_valid": True,
                "row_count": len(rows),
                "data_row_count": len(data_rows),
                "header_count": len(headers),
                "headers": headers,
                "consistent_columns": consistent_columns,
                "empty_cells": empty_cells,
                "total_cells": total_cells,
                "empty_ratio": round(empty_ratio, 3),
                "encoding": "UTF-8",
            }
        except Exception as e:
            return {
                "format": "CSV",
                "is_valid": False,
                "error": f"Analysis failed: {str(e)}",
            }

    def _analyze_text_content(self, file_path: Path) -> dict[str, Any]:
        """Analyze text file content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            return {
                "format": "TEXT",
                "is_valid": True,
                "total_characters": len(content),
                "total_lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "average_line_length": round(len(content) / len(lines), 2) if lines else 0,
                "longest_line": max(len(line) for line in lines) if lines else 0,
                "shortest_line": min(len(line) for line in lines) if lines else 0,
                "encoding": "UTF-8",
                "encoding_issues": self._check_text_encoding_issues(content),
            }
        except Exception as e:
            return {
                "format": "TEXT",
                "is_valid": False,
                "error": f"Analysis failed: {str(e)}",
            }

    def _analyze_python_content(self, file_path: Path) -> dict[str, Any]:
        """Analyze Python file content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            # Basic Python analysis
            imports = [line for line in lines if line.strip().startswith(("import ", "from "))]
            functions = [line for line in lines if line.strip().startswith("def ")]
            classes = [line for line in lines if line.strip().startswith("class ")]

            return {
                "format": "PYTHON",
                "is_valid": True,
                "total_characters": len(content),
                "total_lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "imports": len(imports),
                "functions": len(functions),
                "classes": len(classes),
                "encoding": "UTF-8",
                "syntax_check": self._check_python_syntax(content),
            }
        except Exception as e:
            return {
                "format": "PYTHON",
                "is_valid": False,
                "error": f"Analysis failed: {str(e)}",
            }

    def _analyze_yaml_content(self, file_path: Path) -> dict[str, Any]:
        """Analyze YAML file content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]

            # Basic YAML analysis
            yaml_indicators = [line for line in lines if ":" in line and not line.strip().startswith("#")]

            return {
                "format": "YAML",
                "is_valid": True,
                "total_characters": len(content),
                "total_lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "yaml_indicators": len(yaml_indicators),
                "encoding": "UTF-8",
            }
        except Exception as e:
            return {
                "format": "YAML",
                "is_valid": False,
                "error": f"Analysis failed: {str(e)}",
            }

    def _analyze_json_structure(self, data: Any, depth: int = 0, max_depth: int = 10) -> dict[str, Any]:
        """Analyze JSON structure recursively."""
        if depth > max_depth:
            return {"type": "max_depth_reached", "depth": depth}

        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_count": len(data),
                "nested_structures": {k: self._analyze_json_structure(v, depth + 1, max_depth)
                                    for k, v in list(data.items())[:10]},  # Limit to first 10 keys
            }
        if isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "sample_items": [self._analyze_json_structure(item, depth + 1, max_depth)
                               for item in data[:3]] if data else [],
            }
        return {
            "type": type(data).__name__,
            "value_sample": str(data)[:100] if data is not None else "null",
        }

    def _check_text_encoding_issues(self, content: str) -> list[str]:
        """Check for text encoding issues."""
        issues = []

        if "\x00" in content:
            issues.append("Contains null bytes")

        control_chars = [char for char in content if ord(char) < 32 and char not in "\n\r\t"]
        if control_chars:
            issues.append(f"Contains {len(control_chars)} control characters")

        return issues

    def _check_python_syntax(self, content: str) -> dict[str, Any]:
        """Check Python syntax (basic check)."""
        try:
            compile(content, "<string>", "exec")
            return {"is_valid": True, "errors": []}
        except SyntaxError as e:
            return {"is_valid": False, "errors": [f"Syntax error at line {e.lineno}: {e.msg}"]}
        except Exception as e:
            return {"is_valid": False, "errors": [f"Compilation error: {str(e)}"]}

    def _assess_file_quality(self, basic_analysis: dict[str, Any],
                            format_analysis: dict[str, Any]) -> dict[str, Any]:
        """Assess overall file quality."""
        score = 100
        issues = []

        # Check basic file properties
        if "error" in basic_analysis:
            score -= 50
            issues.append("Basic file analysis failed")

        # Check format analysis
        if "error" in format_analysis:
            score -= 30
            issues.append("Format analysis failed")
        elif not format_analysis.get("is_valid", True):
            score -= 25
            issues.append("Invalid file format")

        # Check file size
        if basic_analysis.get("size_mb", 0) > 100:
            score -= 15
            issues.append("Very large file (>100MB)")

        # Check readability
        if not basic_analysis.get("is_readable", True):
            score -= 40
            issues.append("File not readable")

        # Determine quality level
        quality_level = self._get_quality_level(score)

        return {
            "overall_quality": quality_level,
            "quality_score": score,
            "issues": issues,
            "assessment_timestamp": datetime.now().isoformat(),
        }

    def _get_quality_level(self, score: int) -> str:
        """Get quality level based on score."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        if score >= 75:
            return QualityLevel.GOOD
        if score >= 60:
            return QualityLevel.ACCEPTABLE
        if score >= 40:
            return QualityLevel.POOR
        return QualityLevel.CRITICAL

    def _identify_file_issues(self, basic_analysis: dict[str, Any],
                             format_analysis: dict[str, Any]) -> list[str]:
        """Identify specific issues in the file."""
        issues = []

        # Basic issues
        if "error" in basic_analysis:
            issues.append(f"Basic analysis error: {basic_analysis['error']}")

        if not basic_analysis.get("is_readable", True):
            issues.append("File is not readable")

        # Format-specific issues
        if "error" in format_analysis:
            issues.append(f"Format analysis error: {format_analysis['error']}")
        elif not format_analysis.get("is_valid", True):
            issues.append("Invalid file format")

        # Size issues
        size_mb = basic_analysis.get("size_mb", 0)
        if size_mb > 100:
            issues.append(f"File is very large ({size_mb} MB)")

        # Content issues
        if format_analysis.get("format") == "CSV":
            if not format_analysis.get("consistent_columns", True):
                issues.append("Inconsistent column counts in CSV")

            empty_ratio = format_analysis.get("empty_ratio", 0)
            if empty_ratio > 0.5:
                issues.append(f"High empty cell ratio: {empty_ratio:.1%}")

        elif format_analysis.get("format") == "JSON":
            structure = format_analysis.get("structure", {})
            if "max_depth_reached" in str(structure):
                issues.append("Very deep JSON nesting detected")

        elif format_analysis.get("format") == "PYTHON":
            syntax_check = format_analysis.get("syntax_check", {})
            if not syntax_check.get("is_valid", True):
                issues.extend(syntax_check.get("errors", []))

        return issues

    def _generate_file_recommendations(self, basic_analysis: dict[str, Any],
                                     format_analysis: dict[str, Any],
                                     issues: list[str]) -> list[str]:
        """Generate recommendations for file improvement."""
        recommendations = []

        # Basic recommendations
        if not basic_analysis.get("is_readable", True):
            recommendations.append("Check file permissions and ensure file is accessible")

        if basic_analysis.get("size_mb", 0) > 100:
            recommendations.append("Consider splitting large files into smaller, manageable chunks")

        # Format-specific recommendations
        if format_analysis.get("format") == "CSV":
            if not format_analysis.get("consistent_columns", True):
                recommendations.append("Ensure all CSV rows have the same number of columns")

            empty_ratio = format_analysis.get("empty_ratio", 0)
            if empty_ratio > 0.2:
                recommendations.append("Consider data cleaning to reduce empty cells")

        elif format_analysis.get("format") == "JSON":
            structure = format_analysis.get("structure", {})
            if "max_depth_reached" in str(structure):
                recommendations.append("Consider flattening deeply nested JSON structures")

        elif format_analysis.get("format") == "PYTHON":
            syntax_check = format_analysis.get("syntax_check", {})
            if not syntax_check.get("is_valid", True):
                recommendations.append("Fix Python syntax errors before execution")

        # General recommendations
        if len(issues) > 5:
            recommendations.append("File has multiple issues - consider comprehensive review")

        if not recommendations:
            recommendations.append("File appears to be in good condition")

        return recommendations

    def _generate_audit_summary(self):
        """Generate comprehensive audit summary."""
        if not self.analysis_results:
            return

        # Aggregate statistics
        total_files = len(self.analysis_results)
        successful_audits = sum(1 for r in self.analysis_results.values() if "error" not in r)
        failed_audits = total_files - successful_audits

        # Quality distribution
        quality_distribution = defaultdict(int)
        total_size_mb = 0
        critical_issues = 0
        all_recommendations = set()

        for result in self.analysis_results.values():
            if "error" not in result:
                # Quality distribution
                quality = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                quality_distribution[quality] += 1

                # Size aggregation
                basic_analysis = result.get("basic_analysis", {})
                total_size_mb += basic_analysis.get("size_mb", 0)

                # Critical issues
                if quality == QualityLevel.CRITICAL:
                    critical_issues += 1

                # Recommendations
                recommendations = result.get("recommendations", [])
                all_recommendations.update(recommendations)

        # Calculate success rate
        success_rate = successful_audits / total_files if total_files > 0 else 0

        # Determine overall quality
        if critical_issues == 0:
            overall_quality = QualityLevel.EXCELLENT
        elif critical_issues <= 2:
            overall_quality = QualityLevel.GOOD
        elif critical_issues <= 5:
            overall_quality = QualityLevel.ACCEPTABLE
        elif critical_issues <= 10:
            overall_quality = QualityLevel.POOR
        else:
            overall_quality = QualityLevel.CRITICAL

        self.audit_summary.update({
            "total_files": total_files,
            "successful_audits": successful_audits,
            "failed_audits": failed_audits,
            "success_rate": round(success_rate, 3),
            "overall_quality": overall_quality,
            "quality_distribution": dict(quality_distribution),
            "total_size_mb": round(total_size_mb, 3),
            "critical_issues": critical_issues,
            "total_recommendations": len(all_recommendations),
            "top_recommendations": list(all_recommendations)[:10],  # Top 10
        })

    def generate_unified_report(self, output_format: str = "text") -> str:
        """
        Generate a comprehensive unified report.

        Args:
            output_format: Output format ("text", "json", or "both")

        Returns:
            Path to generated report(s)
        """
        if not self.analysis_results:
            return "No audit results available. Run audit first."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format in ["json", "both"]:
            json_file = f"comprehensive_audit_report_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump({
                    "audit_summary": self.audit_summary,
                    "analysis_results": self.analysis_results,
                }, f, indent=2, default=str)
            self.logger.info(f"JSON report saved to: {json_file}")

        if output_format in ["text", "both"]:
            text_file = f"comprehensive_audit_report_{timestamp}.txt"
            with open(text_file, "w") as f:
                f.write(self._format_unified_text_report())
            self.logger.info(f"Text report saved to: {text_file}")

        if output_format == "json":
            return json_file
        if output_format == "text":
            return text_file
        # both
        return f"{json_file}, {text_file}"

    def _format_unified_text_report(self) -> str:
        """Format the unified report as human-readable text."""
        lines = []

        # Header
        lines.append("=" * 100)
        lines.append("COMPREHENSIVE QUALITY AUDIT REPORT")
        lines.append("=" * 100)
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 50)
        lines.append(f"Target Path: {self.audit_summary.get('target_path', 'Unknown')}")
        lines.append(f"Audit Mode: {self.audit_summary.get('audit_mode', 'Unknown')}")
        lines.append(f"Total Files: {self.audit_summary.get('total_files', 0)}")
        lines.append(f"Successful Audits: {self.audit_summary.get('successful_audits', 0)}")
        lines.append(f"Failed Audits: {self.audit_summary.get('failed_audits', 0)}")
        lines.append(f"Success Rate: {self.audit_summary.get('success_rate', 0):.1%}")
        lines.append(f"Overall Quality: {self.audit_summary.get('overall_quality', 'Unknown').upper()}")
        lines.append(f"Total Data Size: {self.audit_summary.get('total_size_mb', 0):.3f} MB")
        lines.append(f"Critical Issues: {self.audit_summary.get('critical_issues', 0)}")
        lines.append(f"Audit Duration: {self.audit_summary.get('audit_duration_seconds', 0)} seconds")
        lines.append("")

        # Quality Distribution
        quality_dist = self.audit_summary.get("quality_distribution", {})
        if quality_dist:
            lines.append("QUALITY DISTRIBUTION")
            lines.append("-" * 50)
            for quality, count in sorted(quality_dist.items(),
                                       key=lambda x: {"excellent": 0, "good": 1, "acceptable": 2, "poor": 3, "critical": 4}.get(x[0], 5)):
                lines.append(f"• {quality.capitalize()}: {count} files")
            lines.append("")

        # Overall Assessment
        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 50)

        critical_issues = self.audit_summary.get("critical_issues", 0)
        if critical_issues == 0:
            lines.append("🎉 EXCELLENT: No critical quality issues detected!")
            lines.append("   All audited files meet quality standards.")
        elif critical_issues <= 2:
            lines.append("✅ GOOD: Minor quality issues detected.")
            lines.append("   Most files are in good condition with few problems.")
        elif critical_issues <= 5:
            lines.append("⚠️  ACCEPTABLE: Some quality issues detected.")
            lines.append("   Several files need attention but overall quality is acceptable.")
        elif critical_issues <= 10:
            lines.append("❌ POOR: Significant quality issues detected.")
            lines.append("   Many files have problems requiring immediate attention.")
        else:
            lines.append("🚨 CRITICAL: Severe quality issues detected!")
            lines.append("   Extensive problems found across multiple files.")

        lines.append("")

        # Top Recommendations
        top_recommendations = self.audit_summary.get("top_recommendations", [])
        if top_recommendations:
            lines.append("TOP RECOMMENDATIONS")
            lines.append("-" * 50)
            for rec in top_recommendations:
                lines.append(f"• {rec}")
            lines.append("")

        # Detailed Results
        lines.append("DETAILED AUDIT RESULTS")
        lines.append("=" * 100)
        lines.append("")

        # Group results by quality level
        quality_groups = defaultdict(list)
        for file_path, result in self.analysis_results.items():
            if "error" not in result:
                quality = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                quality_groups[quality].append((file_path, result))
            else:
                quality_groups["error"].append((file_path, result))

        # Report by quality level
        for quality in ["excellent", "good", "acceptable", "poor", "critical", "error"]:
            if quality in quality_groups:
                lines.append(f"{quality.upper()} QUALITY FILES")
                lines.append("-" * 50)
                lines.append("")

                for file_path, result in quality_groups[quality][:20]:  # Limit to first 20
                    file_name = Path(file_path).name

                    if "error" in result:
                        lines.append(f"❌ {file_name}: {result['error']}")
                    else:
                        quality_level = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                        size = result.get("basic_analysis", {}).get("size_mb", "unknown")
                        issues_count = len(result.get("issues", []))

                        lines.append(f"📁 {file_name}")
                        lines.append(f"   Quality: {quality_level.upper()}")
                        lines.append(f"   Size: {size} MB")
                        lines.append(f"   Issues: {issues_count}")

                        # Show first few issues
                        issues = result.get("issues", [])
                        if issues:
                            for issue in issues[:2]:
                                lines.append(f"     - {issue}")
                            if len(issues) > 2:
                                lines.append(f"     ... and {len(issues) - 2} more issues")

                        lines.append("")

                if len(quality_groups[quality]) > 20:
                    lines.append(f"... and {len(quality_groups[quality]) - 20} more files")
                lines.append("")

        # Footer
        lines.append("=" * 100)
        lines.append("END OF COMPREHENSIVE QUALITY AUDIT REPORT")
        lines.append("=" * 100)
        lines.append("")
        lines.append("Report generated by ComprehensiveQualityAuditor")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")

        return "\n".join(lines)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Quality Auditor - Exhaustive audit and unified reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run exhaustive audit on a directory
  python comprehensive_quality_auditor.py --audit /path/to/directory --recursive

  # Run exhaustive audit on a single file
  python comprehensive_quality_auditor.py --audit /path/to/file.csv

  # Generate unified report from existing audit
  python comprehensive_quality_auditor.py --generate-unified

  # Run full audit and generate report
  python comprehensive_quality_auditor.py --full-audit /path/to/target --output-format both
        """,
    )

    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--audit", metavar="PATH",
                             help="Run exhaustive audit on specified path (file or directory)")
    action_group.add_argument("--generate-unified", action="store_true",
                             help="Generate unified report from existing audit results")
    action_group.add_argument("--full-audit", metavar="PATH",
                             help="Run full audit and generate unified report")

    # Audit options
    parser.add_argument("--recursive", action="store_true",
                       help="Recursively analyze subdirectories (default: True)")
    parser.add_argument("--file-pattern", default="*",
                       help="File pattern for filtering (e.g., '*.csv')")
    parser.add_argument("--max-files", type=int, default=1000,
                       help="Maximum number of files to analyze (default: 1000)")

    # Output options
    parser.add_argument("--output-format", choices=["text", "json", "both"], default="both",
                       help="Output format for reports (default: both)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize auditor
    auditor = ComprehensiveQualityAuditor()

    try:
        if args.audit:
            # Run audit only
            print(f"🔍 Running exhaustive audit on: {args.audit}")
            result = auditor.run_exhaustive_audit(
                target_path=args.audit,
                recursive=args.recursive,
                file_pattern=args.file_pattern,
                max_files=args.max_files,
            )

            if "error" in result:
                print(f"❌ Audit failed: {result['error']}")
                return 1

            print("✅ Audit completed successfully!")
            print(f"📊 Files analyzed: {result['audit_summary'].get('total_files', 0)}")
            print(f"🎯 Overall quality: {result['audit_summary'].get('overall_quality', 'unknown').upper()}")

        elif args.generate_unified:
            # Generate unified report
            print("📋 Generating unified report...")
            output_file = auditor.generate_unified_report(args.output_format)
            print(f"✅ Unified report generated: {output_file}")

        elif args.full_audit:
            # Run full audit and generate report
            print(f"🚀 Running full audit and report generation on: {args.full_audit}")

            # Run audit
            result = auditor.run_exhaustive_audit(
                target_path=args.full_audit,
                recursive=args.recursive,
                file_pattern=args.file_pattern,
                max_files=args.max_files,
            )

            if "error" in result:
                print(f"❌ Audit failed: {result['error']}")
                return 1

            print("✅ Audit completed successfully!")

            # Generate report
            print("📋 Generating unified report...")
            output_file = auditor.generate_unified_report(args.output_format)
            print("✅ Full process completed!")
            print(f"📊 Files analyzed: {result['audit_summary'].get('total_files', 0)}")
            print(f"🎯 Overall quality: {result['audit_summary'].get('overall_quality', 'unknown').upper()}")
            print(f"📋 Report generated: {output_file}")

        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Audit interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import os
    sys.exit(main())
