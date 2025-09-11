#!/usr/bin/env python3
from src.utils.tprint import tprint

"""
Simple Data Quality Orchestrator

A simplified version that works with basic Python libraries and can analyze
JSON and CSV files without requiring pandas or sklearn.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from centralized_logging import get_logger
import numpy as np
import logging
import time

logger = get_logger(__name__)


class QualityLevel:
    """Quality level enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class SimpleQualityOrchestrator:
    """Simple data quality orchestrator for basic file analysis."""

    def __init__(self):
        """Initialize the simple quality orchestrator."""
        self.logger = logger

        # Supported file extensions
        self.supported_extensions = {".csv", ".json", ".txt", ".py", ".yaml", ".yml"}

    def analyze_file(self, file_path: str, context: str = "") -> dict[str, Any]:
        """
        Analyze a single file for quality issues.

        Args:
            file_path: Path to the file to analyze
            context: Context description for the data

        Returns:
            Dictionary with quality analysis results
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            return {"error": f"File not found: {file_path}"}

        if file_path_obj.suffix.lower() not in self.supported_extensions:
            return {"error": f"Unsupported file format: {file_path_obj.suffix}"}

        self.logger.info(f"Analyzing file: {file_path_obj.name}")

        try:
            if file_path_obj.suffix.lower() == ".json":
                return self._analyze_json_file(file_path_obj, context)
            if file_path_obj.suffix.lower() == ".csv":
                return self._analyze_csv_file(file_path_obj, context)
            if file_path_obj.suffix.lower() in [".txt", ".py", ".yaml", ".yml"]:
                return self._analyze_text_file(file_path_obj, context)
            return {"error": f"Unsupported file format: {file_path_obj.suffix}"}

        except Exception as e:
            self.logger.exception(f"Error analyzing {file_path_obj.name}: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_json_file(self, file_path: Path, context: str) -> dict[str, Any]:
        """Analyze a JSON file for quality issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Basic file info
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Analyze JSON structure
            structure_analysis = self._analyze_json_structure(data)

            # Quality assessment
            quality_score = self._assess_json_quality(data, structure_analysis)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "file_info": {
                    "size_bytes": file_size,
                    "size_mb": round(file_size_mb, 3),
                    "format": "JSON",
                },
                "structure_analysis": structure_analysis,
                "quality_assessment": quality_score,
                "issues": self._identify_json_issues(data, structure_analysis),
                "recommendations": self._generate_json_recommendations(data, structure_analysis),
            }

        except json.JSONDecodeError as e:
            error_details = self._analyze_json_error(e, file_path)
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Invalid JSON: {str(e)}",
                "error_details": error_details,
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except UnicodeDecodeError as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Encoding error: {str(e)}",
                "error_details": f"File encoding issue. Try opening with different encoding (utf-8, latin-1, cp1252).",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except PermissionError as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Permission denied: {str(e)}",
                "error_details": "Check file permissions. File may be read-only or locked by another process.",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Unexpected error: {str(e)}",
                "error_details": "Unexpected error occurred while analyzing JSON file. Check file integrity.",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }

    def _analyze_csv_file(self, file_path: Path, context: str) -> dict[str, Any]:
        """Analyze a CSV file for quality issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                return {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "context": context,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "error": "Empty CSV file",
                    "error_details": "CSV file contains no data rows. Check if the file is corrupted or was not properly saved.",
                    "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
                }

            # Basic file info
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Analyze CSV structure
            structure_analysis = self._analyze_csv_structure(rows)

            # Quality assessment
            quality_score = self._assess_csv_quality(rows, structure_analysis)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "file_info": {
                    "size_bytes": file_size,
                    "size_mb": round(file_size_mb, 3),
                    "format": "CSV",
                },
                "structure_analysis": structure_analysis,
                "quality_assessment": quality_score,
                "issues": self._identify_csv_issues(rows, structure_analysis),
                "recommendations": self._generate_csv_recommendations(rows, structure_analysis),
            }

        except UnicodeDecodeError as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Encoding error: {str(e)}",
                "error_details": "CSV file encoding issue. Try opening with different encoding (utf-8, latin-1, cp1252).",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except csv.Error as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"CSV parsing error: {str(e)}",
                "error_details": "CSV format error. Check for malformed CSV data, incorrect delimiters, or unescaped quotes.",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except PermissionError as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Permission denied: {str(e)}",
                "error_details": "Check file permissions. File may be read-only or locked by another process.",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }
        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"CSV analysis failed: {str(e)}",
                "error_details": "Unexpected error occurred while analyzing CSV file. Check file integrity and format.",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }

    def _analyze_text_file(self, file_path: Path, context: str) -> dict[str, Any]:
        """Analyze a text file for quality issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Basic file info
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            lines = content.split("\n")

            # Analyze text structure
            structure_analysis = self._analyze_text_structure(content, lines)

            # Quality assessment
            quality_score = self._assess_text_quality(content, lines, structure_analysis)

            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "file_info": {
                    "size_bytes": file_size,
                    "size_mb": round(file_size_mb, 3),
                    "format": "TEXT",
                },
                "structure_analysis": structure_analysis,
                "quality_assessment": quality_score,
                "issues": self._identify_text_issues(content, lines, structure_analysis),
                "recommendations": self._generate_text_recommendations(content, lines, structure_analysis),
            }

        except Exception as e:
            return {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "context": context,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": f"Text analysis failed: {str(e)}",
                "quality_assessment": {"overall_quality": QualityLevel.CRITICAL},
            }

    def _analyze_json_structure(self, data: Any) -> dict[str, Any]:
        """Analyze the structure of JSON data."""
        def _analyze_recursive(obj, depth=0, max_depth=10):
            if depth > max_depth:
                return {"type": "max_depth_reached", "depth": depth}

            if isinstance(obj, dict):
                return {
                    "type": "object",
                    "keys": list(obj.keys()),
                    "key_count": len(obj),
                    "nested_structures": {k: _analyze_recursive(v, depth + 1) for k, v in obj.items()},
                }
            if isinstance(obj, list):
                return {
                    "type": "array",
                    "length": len(obj),
                    "sample_items": [_analyze_recursive(item, depth + 1) for item in obj[:3]] if obj else [],
                }
            return {
                "type": type(obj).__name__,
                "value_sample": str(obj)[:100] if obj is not None else "null",
            }

        return _analyze_recursive(data)

    def _analyze_csv_structure(self, rows: list[list[str]]) -> dict[str, Any]:
        """Analyze the structure of CSV data."""
        if not rows:
            return {"error": "No rows found"}

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        # Analyze column consistency
        column_lengths = [len(row) for row in rows]
        consistent_columns = len(set(column_lengths)) == 1

        # Check for empty cells
        empty_cells = 0
        total_cells = sum(len(row) for row in rows)

        for row in rows:
            empty_cells += sum(1 for cell in row if not cell.strip())

        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0

        return {
            "headers": headers,
            "header_count": len(headers),
            "row_count": len(rows),
            "data_row_count": len(data_rows),
            "consistent_columns": consistent_columns,
            "column_lengths": column_lengths,
            "empty_cells": empty_cells,
            "total_cells": total_cells,
            "empty_ratio": round(empty_ratio, 3),
        }

    def _analyze_text_structure(self, content: str, lines: list[str]) -> dict[str, Any]:
        """Analyze the structure of text data."""
        return {
            "total_characters": len(content),
            "total_lines": len(lines),
            "non_empty_lines": len([line for line in lines if line.strip()]),
            "average_line_length": round(len(content) / len(lines), 2) if lines else 0,
            "longest_line": max(len(line) for line in lines) if lines else 0,
            "shortest_line": min(len(line) for line in lines) if lines else 0,
            "encoding_issues": self._check_encoding_issues(content),
        }

    def _check_encoding_issues(self, content: str) -> list[str]:
        """Check for potential encoding issues."""
        issues = []

        # Check for common encoding problems
        if "\x00" in content:
            issues.append("Contains null bytes")

        # Check for control characters
        control_chars = [char for char in content if ord(char) < 32 and char not in "\n\r\t"]
        if control_chars:
            issues.append(f"Contains {len(control_chars)} control characters")

        return issues

    def _assess_json_quality(self, data: Any, structure: dict[str, Any]) -> dict[str, Any]:
        """Assess the quality of JSON data."""
        score = 100
        issues = []

        # Check for empty data
        if not data:
            score -= 50
            issues.append("Empty data structure")

        # Check for very deep nesting
        if "max_depth_reached" in str(structure):
            score -= 20
            issues.append("Very deep nesting detected")

        # Determine quality level
        if score >= 90:
            quality = QualityLevel.EXCELLENT
        elif score >= 75:
            quality = QualityLevel.GOOD
        elif score >= 60:
            quality = QualityLevel.ACCEPTABLE
        elif score >= 40:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL

        return {
            "overall_quality": quality,
            "quality_score": score,
            "issues": issues,
        }

    def _assess_csv_quality(self, rows: list[list[str]], structure: dict[str, Any]) -> dict[str, Any]:
        """Assess the quality of CSV data."""
        score = 100
        issues = []

        if not rows:
            score -= 100
            issues.append("No data rows")
        else:
            # Check column consistency
            if not structure.get("consistent_columns", True):
                score -= 30
                issues.append("Inconsistent column counts")

            # Check for empty data
            if structure.get("data_row_count", 0) == 0:
                score -= 40
                issues.append("No data rows")

            # Check empty cell ratio
            empty_ratio = structure.get("empty_ratio", 0)
            if empty_ratio > 0.5:
                score -= 25
                issues.append(f"High empty cell ratio: {empty_ratio:.1%}")
            elif empty_ratio > 0.2:
                score -= 10
                issues.append(f"Moderate empty cell ratio: {empty_ratio:.1%}")

        # Determine quality level
        if score >= 90:
            quality = QualityLevel.EXCELLENT
        elif score >= 75:
            quality = QualityLevel.GOOD
        elif score >= 60:
            quality = QualityLevel.ACCEPTABLE
        elif score >= 40:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL

        return {
            "overall_quality": quality,
            "quality_score": score,
            "issues": issues,
        }

    def _assess_text_quality(self, content: str, lines: list[str], structure: dict[str, Any]) -> dict[str, Any]:
        """Assess the quality of text data."""
        score = 100
        issues = []

        if not content:
            score -= 100
            issues.append("Empty content")
        else:
            # Check for encoding issues
            if structure.get("encoding_issues"):
                score -= 30
                issues.extend(structure["encoding_issues"])

            # Check for very long lines
            if structure.get("longest_line", 0) > 1000:
                score -= 15
                issues.append("Very long lines detected")

            # Check for empty lines
            empty_lines = len([line for line in lines if not line.strip()])
            if empty_lines > len(lines) * 0.3:
                score -= 20
                issues.append(f"High empty line ratio: {empty_lines}/{len(lines)}")

        # Determine quality level
        if score >= 90:
            quality = QualityLevel.EXCELLENT
        elif score >= 75:
            quality = QualityLevel.GOOD
        elif score >= 60:
            quality = QualityLevel.ACCEPTABLE
        elif score >= 40:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL

        return {
            "overall_quality": quality,
            "quality_score": score,
            "issues": issues,
        }

    def _identify_json_issues(self, data: Any, structure: dict[str, Any]) -> list[str]:
        """Identify specific issues in JSON data."""
        issues = []

        if not data:
            issues.append("Empty JSON data")
            return issues

        # Check for very large structures
        if isinstance(data, dict) and len(data) > 1000:
            issues.append("Very large object (over 1000 keys)")

        if isinstance(data, list) and len(data) > 10000:
            issues.append("Very large array (over 10,000 items)")

        return issues

    def _identify_csv_issues(self, rows: list[list[str]], structure: dict[str, Any]) -> list[str]:
        """Identify specific issues in CSV data."""
        issues = []

        if not rows:
            issues.append("Empty CSV file")
            return issues

        # Check column consistency
        if not structure.get("consistent_columns", True):
            issues.append("Inconsistent column counts across rows")

        # Check for empty headers
        if structure.get("headers"):
            empty_headers = [h for h in structure["headers"] if not h.strip()]
            if empty_headers:
                issues.append(f"Empty column headers: {len(empty_headers)} found")

        # Check for data quality issues
        empty_ratio = structure.get("empty_ratio", 0)
        if empty_ratio > 0.5:
            issues.append(f"High empty cell ratio: {empty_ratio:.1%}")

        return issues

    def _identify_text_issues(self, content: str, lines: list[str], structure: dict[str, Any]) -> list[str]:
        """Identify specific issues in text data."""
        issues = []

        if not content:
            issues.append("Empty text file")
            return issues

        # Check encoding issues
        encoding_issues = structure.get("encoding_issues", [])
        issues.extend(encoding_issues)

        # Check for very long lines
        if structure.get("longest_line", 0) > 1000:
            issues.append("Very long lines detected (over 1000 characters)")

        # Check for empty lines
        empty_lines = len([line for line in lines if not line.strip()])
        if empty_lines > len(lines) * 0.3:
            issues.append(f"High empty line ratio: {empty_lines}/{len(lines)}")

        return issues

    def _generate_json_recommendations(self, data: Any, structure: dict[str, Any]) -> list[str]:
        """Generate recommendations for JSON data."""
        recommendations = []

        if not data:
            recommendations.append("Add meaningful data to the JSON structure")
            return recommendations

        # Check for deep nesting
        if "max_depth_reached" in str(structure):
            recommendations.append("Consider flattening deeply nested structures")

        # Check for large structures
        if isinstance(data, dict) and len(data) > 1000:
            recommendations.append("Consider splitting large objects into smaller, focused structures")

        if isinstance(data, list) and len(data) > 10000:
            recommendations.append("Consider pagination or streaming for large arrays")

        return recommendations

    def _generate_csv_recommendations(self, rows: list[list[str]], structure: dict[str, Any]) -> list[str]:
        """Generate recommendations for CSV data."""
        recommendations = []

        if not rows:
            recommendations.append("Add data rows to the CSV file")
            return recommendations

        # Check column consistency
        if not structure.get("consistent_columns", True):
            recommendations.append("Ensure all rows have the same number of columns")

        # Check for empty headers
        if structure.get("headers"):
            empty_headers = [h for h in structure["headers"] if not h.strip()]
            if empty_headers:
                recommendations.append("Provide meaningful names for all column headers")

        # Check for data quality
        empty_ratio = structure.get("empty_ratio", 0)
        if empty_ratio > 0.2:
            recommendations.append("Consider data cleaning to reduce empty cells")

        return recommendations

    def _generate_text_recommendations(self, content: str, lines: list[str], structure: dict[str, Any]) -> list[str]:
        """Generate recommendations for text data."""
        recommendations = []

        if not content:
            recommendations.append("Add meaningful content to the text file")
            return recommendations

        # Check encoding issues
        if structure.get("encoding_issues"):
            recommendations.append("Fix encoding issues to ensure proper text processing")

        # Check for very long lines
        if structure.get("longest_line", 0) > 1000:
            recommendations.append("Consider breaking very long lines for better readability")

        # Check for empty lines
        empty_lines = len([line for line in lines if not line.strip()])
        if empty_lines > len(lines) * 0.3:
            recommendations.append("Reduce excessive empty lines for better content density")

        return recommendations

    def analyze_directory(self, directory_path: str, file_pattern: str = "*") -> dict[str, Any]:
        """
        Analyze all supported files in a directory.

        Args:
            directory_path: Path to directory to analyze
            file_pattern: Glob pattern for file matching

        Returns:
            Dictionary with directory analysis results
        """
        directory = Path(directory_path)
        if not directory.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        self.logger.info(f"Analyzing directory: {directory_path}")

        # Find all supported files
        data_files = []
        if file_pattern == "*":
            # Find all supported extensions
            for ext in self.supported_extensions:
                data_files.extend(directory.glob(f"*{ext}"))
        else:
            # Use the specific pattern
            data_files.extend(directory.glob(file_pattern))

        if not data_files:
            return {"error": f"No supported data files found in {directory_path}"}

        self.logger.info(f"Found {len(data_files)} data files to analyze")

        # Analyze each file
        results = {}
        summary_stats = {
            "total_files": len(data_files),
            "successful_analyses": 0,
            "failed_analyses": 0,
            "quality_distribution": {},
            "total_size_bytes": 0,
            "total_size_mb": 0,
        }

        for file_path in data_files:
            try:
                # Analyze file
                result = self.analyze_file(str(file_path), f"File: {file_path.name}")

                if "error" not in result:
                    results[str(file_path)] = result
                    summary_stats["successful_analyses"] += 1

                    # Aggregate quality statistics
                    quality = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                    summary_stats["quality_distribution"][quality] = summary_stats["quality_distribution"].get(quality, 0) + 1

                    # Aggregate size statistics
                    file_info = result.get("file_info", {})
                    summary_stats["total_size_bytes"] += file_info.get("size_bytes", 0)
                else:
                    results[str(file_path)] = result
                    summary_stats["failed_analyses"] += 1

            except Exception as e:
                self.logger.exception(f"Failed to analyze {file_path.name}: {e}")
                results[str(file_path)] = {"error": str(e)}
                summary_stats["failed_analyses"] += 1

        # Calculate total size in MB
        summary_stats["total_size_mb"] = round(summary_stats["total_size_bytes"] / (1024 * 1024), 3)

        # Determine overall directory quality
        quality_scores = list(summary_stats["quality_distribution"].keys())
        if "critical" in quality_scores:
            overall_quality = QualityLevel.CRITICAL
        elif "poor" in quality_scores:
            overall_quality = QualityLevel.POOR
        elif "acceptable" in quality_scores:
            overall_quality = QualityLevel.ACCEPTABLE
        elif "good" in quality_scores:
            overall_quality = QualityLevel.GOOD
        else:
            overall_quality = QualityLevel.EXCELLENT

        return {
            "directory_path": str(directory),
            "analysis_timestamp": datetime.now().isoformat(),
            "file_pattern": file_pattern,
            "summary": {
                "overall_quality": overall_quality,
                "total_files": summary_stats["total_files"],
                "successful_analyses": summary_stats["successful_analyses"],
                "failed_analyses": summary_stats["failed_analyses"],
                "success_rate": round(summary_stats["successful_analyses"] / summary_stats["total_files"], 3),
                "quality_distribution": summary_stats["quality_distribution"],
                "total_size_bytes": summary_stats["total_size_bytes"],
                "total_size_mb": summary_stats["total_size_mb"],
            },
            "file_results": results,
        }

    def save_report(self, report: dict[str, Any], filename: str = None) -> str:
        """
        Save the quality report to a file.

        Args:
            report: Quality report to save
            filename: Output filename (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Quality report saved to: {output_path}")
        return str(output_path)

    def generate_text_report(self, report: dict[str, Any], filename: str = None) -> str:
        """
        Generate a human-readable text report.

        Args:
            report: Quality report to convert
            filename: Output filename (optional)

        Returns:
            Path to saved text file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.txt"

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self._format_text_report(report))

        self.logger.info(f"Text report saved to: {output_path}")
        return str(output_path)

    def _format_text_report(self, report: dict[str, Any]) -> str:
        """Format the report as human-readable text."""
        if "error" in report:
            return f"ERROR: {report['error']}"

        # Check if it's a directory report
        if "directory_path" in report:
            return self._format_directory_text_report(report)
        return self._format_file_text_report(report)

    def _format_file_text_report(self, report: dict[str, Any]) -> str:
        """Format a single file report as text."""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA QUALITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # File information
        lines.append("FILE INFORMATION:")
        lines.append("-" * 40)
        lines.append(f"File Name: {report.get('file_name', 'Unknown')}")
        lines.append(f"File Path: {report.get('file_path', 'Unknown')}")
        lines.append(f"Context: {report.get('context', 'None')}")
        lines.append(f"Analysis Time: {report.get('analysis_timestamp', 'Unknown')}")
        lines.append("")

        # File details
        file_info = report.get("file_info", {})
        if file_info:
            lines.append("FILE DETAILS:")
            lines.append("-" * 40)
            lines.append(f"Format: {file_info.get('format', 'Unknown')}")
            lines.append(f"Size: {file_info.get('size_mb', 'Unknown')} MB")
            lines.append("")

        # Quality assessment
        quality_assessment = report.get("quality_assessment", {})
        if quality_assessment:
            lines.append("QUALITY ASSESSMENT:")
            lines.append("-" * 40)
            lines.append(f"Overall Quality: {quality_assessment.get('overall_quality', 'Unknown').upper()}")
            lines.append(f"Quality Score: {quality_assessment.get('quality_score', 'Unknown')}/100")
            lines.append("")

            issues = quality_assessment.get("issues", [])
            if issues:
                lines.append("QUALITY ISSUES:")
                lines.append("-" * 40)
                for issue in issues:
                    lines.append(f"• {issue}")
                lines.append("")

        # Structure analysis
        structure_analysis = report.get("structure_analysis", {})
        if structure_analysis:
            lines.append("STRUCTURE ANALYSIS:")
            lines.append("-" * 40)
            if structure_analysis.get("type") == "object":
                lines.append("Type: JSON Object")
                lines.append(f"Key Count: {structure_analysis.get('key_count', 'Unknown')}")
                lines.append(f"Keys: {', '.join(structure_analysis.get('keys', [])[:10])}")
                if len(structure_analysis.get("keys", [])) > 10:
                    lines.append(f"... and {len(structure_analysis.get('keys', [])) - 10} more keys")
            elif structure_analysis.get("type") == "array":
                lines.append("Type: JSON Array")
                lines.append(f"Length: {structure_analysis.get('length', 'Unknown')}")
            elif structure_analysis.get("type") == "object":
                lines.append("Type: CSV")
                lines.append(f"Rows: {structure_analysis.get('row_count', 'Unknown')}")
                lines.append(f"Columns: {structure_analysis.get('header_count', 'Unknown')}")
                lines.append(f"Empty Cell Ratio: {structure_analysis.get('empty_ratio', 'Unknown')}")
            lines.append("")

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 40)
            for rec in recommendations:
                lines.append(f"• {rec}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_directory_text_report(self, report: dict[str, Any]) -> str:
        """Format a directory report as text."""
        lines = []
        lines.append("=" * 80)
        lines.append("DIRECTORY DATA QUALITY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Directory information
        lines.append("DIRECTORY INFORMATION:")
        lines.append("-" * 40)
        lines.append(f"Directory Path: {report.get('directory_path', 'Unknown')}")
        lines.append(f"Analysis Time: {report.get('analysis_timestamp', 'Unknown')}")
        lines.append(f"File Pattern: {report.get('file_pattern', 'Unknown')}")
        lines.append("")

        # Summary
        summary = report.get("summary", {})
        if summary:
            lines.append("ANALYSIS SUMMARY:")
            lines.append("-" * 40)
            lines.append(f"Overall Quality: {summary.get('overall_quality', 'Unknown').upper()}")
            lines.append(f"Total Files: {summary.get('total_files', 'Unknown')}")
            lines.append(f"Successful Analyses: {summary.get('successful_analyses', 'Unknown')}")
            lines.append(f"Failed Analyses: {summary.get('failed_analyses', 'Unknown')}")
            lines.append(f"Success Rate: {summary.get('success_rate', 'Unknown'):.1%}")
            lines.append(f"Total Size: {summary.get('total_size_mb', 'Unknown')} MB")
            lines.append("")

            # Quality distribution
            quality_dist = summary.get("quality_distribution", {})
            if quality_dist:
                lines.append("QUALITY DISTRIBUTION:")
                lines.append("-" * 40)
                for quality, count in quality_dist.items():
                    lines.append(f"• {quality.capitalize()}: {count} files")
                lines.append("")

        # Individual file results
        file_results = report.get("file_results", {})
        if file_results:
            lines.append("INDIVIDUAL FILE RESULTS:")
            lines.append("-" * 40)

            for file_path, result in file_results.items():
                file_name = Path(file_path).name

                if "error" in result:
                    lines.append(f"❌ {file_name}: {result['error']}")
                else:
                    quality = result.get("quality_assessment", {}).get("overall_quality", "unknown")
                    size = result.get("file_info", {}).get("size_mb", "unknown")
                    lines.append(f"✅ {file_name}: {quality.upper()} ({size} MB)")

                    # Show issues if any
                    issues = result.get("issues", [])
                    if issues:
                        for issue in issues[:2]:  # Show first 2 issues
                            lines.append(f"    - {issue}")
                        if len(issues) > 2:
                            lines.append(f"    ... and {len(issues) - 2} more issues")

                lines.append("")

        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _analyze_json_error(self, error: json.JSONDecodeError, file_path: Path) -> str:
        """Analyze JSON decode error and provide specific details."""
        error_details = []
        
        # Basic error information
        error_details.append(f"JSON parsing failed at line {error.lineno}, column {error.colno}")
        
        # Common JSON error patterns
        error_msg = str(error.msg).lower()
        
        if "expecting" in error_msg:
            if "value" in error_msg:
                error_details.append("❌ Expected a JSON value (string, number, boolean, null, object, or array)")
            elif "property name" in error_msg:
                error_details.append("❌ Expected a property name in JSON object")
            elif "colon" in error_msg:
                error_details.append("❌ Expected a colon (:) after property name")
            elif "comma" in error_msg:
                error_details.append("❌ Expected a comma (,) to separate array elements or object properties")
            elif "end of file" in error_msg:
                error_details.append("❌ Unexpected end of file - JSON may be incomplete")
        
        if "unterminated" in error_msg:
            if "string" in error_msg:
                error_details.append("❌ Unterminated string - missing closing quote")
            elif "comment" in error_msg:
                error_details.append("❌ Unterminated comment - JSON doesn't support comments")
        
        if "invalid" in error_msg:
            if "character" in error_msg:
                error_details.append("❌ Invalid character in JSON - check for special characters or encoding issues")
            elif "escape" in error_msg:
                error_details.append("❌ Invalid escape sequence in string")
        
        if "trailing" in error_msg:
            error_details.append("❌ Trailing comma found - JSON doesn't allow trailing commas")
        
        if "duplicate" in error_msg:
            error_details.append("❌ Duplicate key found in JSON object")
        
        # Try to read the problematic line for context
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if error.lineno <= len(lines):
                    problem_line = lines[error.lineno - 1].strip()
                    error_details.append(f"📄 Problematic line: {problem_line}")
        except:
            pass
        
        return " | ".join(error_details)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Simple Data Quality Orchestrator")
    parser.add_argument("--data_path", required=True, help="Path to data file or directory")
    parser.add_argument("--context", default="", help="Context description for the data")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--text_output", help="Text output file for human-readable report")
    parser.add_argument("--mode", choices=["file", "directory", "auto"], default="auto",
                       help="Analysis mode: file, directory, or auto-detect")
    parser.add_argument("--file_pattern", default="*",
                       help="File pattern for directory analysis (e.g., '*.csv')")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = SimpleQualityOrchestrator()

    # Determine analysis mode
    data_path = Path(args.data_path)
    if args.mode == "auto":
        if data_path.is_file():
            mode = "file"
        elif data_path.is_dir():
            mode = "directory"
        else:
            tprint(f"Path not found: {data_path}")
            return
    else:
        mode = args.mode

    # Perform analysis based on mode
    if mode == "file":
        # Single file analysis
        if not data_path.exists():
            tprint(f"File not found: {data_path}")
            return

        tprint(f"📁 Analyzing single file: {data_path.name}")

        # Analyze file
        report = orchestrator.analyze_file(str(data_path), args.context or f"File: {data_path.name}")

        if "error" in report:
            tprint(f"❌ Analysis failed: {report['error']}")
            return

        # Save JSON report
        if args.output:
            output_file = orchestrator.save_report(report, args.output)
        else:
            output_file = orchestrator.save_report(report)

        # Generate text report
        if args.text_output:
            text_file = orchestrator.generate_text_report(report, args.text_output)
        else:
            text_file = orchestrator.generate_text_report(report)

        # Print summary
        quality_assessment = report.get("quality_assessment", {})
        tprint("\n📊 QUALITY REPORT SUMMARY")
        tprint(f"Overall Quality: {quality_assessment.get('overall_quality', 'unknown').upper()}")
        tprint(f"Quality Score: {quality_assessment.get('quality_score', 'unknown')}/100")
        tprint(f"JSON Report: {output_file}")
        tprint(f"Text Report: {text_file}")

    elif mode == "directory":
        # Directory analysis
        if not data_path.is_dir():
            tprint(f"Path is not a directory: {data_path}")
            return

        tprint(f"📁 Analyzing directory: {data_path}")

        # Analyze directory
        directory_report = orchestrator.analyze_directory(str(data_path), args.file_pattern)

        if "error" in directory_report:
            tprint(f"❌ Directory analysis failed: {directory_report['error']}")
            return

        # Save JSON report
        if args.output:
            output_file = orchestrator.save_report(directory_report, args.output)
        else:
            output_file = orchestrator.save_report(directory_report)

        # Generate text report
        if args.text_output:
            text_file = orchestrator.generate_text_report(directory_report, args.text_output)
        else:
            text_file = orchestrator.generate_text_report(directory_report)

        # Print summary
        summary = directory_report.get("summary", {})
        tprint("\n📊 DIRECTORY QUALITY SUMMARY")
        tprint(f"Overall Quality: {summary.get('overall_quality', 'unknown').upper()}")
        tprint(f"Total Files: {summary.get('total_files', 'unknown')}")
        tprint(f"Success Rate: {summary.get('success_rate', 'unknown'):.1%}")
        tprint(f"JSON Report: {output_file}")
        tprint(f"Text Report: {text_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
