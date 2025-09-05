#!/usr/bin/env python3
"""Text report generator for code analysis results."""

from datetime import datetime
from typing import Dict, Any
import time


class TextReporter:
    """Generates text reports from analysis results."""
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a text summary report."""
        report = []
        report.append("CODE INTERACTION MAPPING SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Add timestamp
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add summary statistics
        if "dead_code" in results:
            dead_code = results["dead_code"]
            report.append("DEAD CODE ANALYSIS")
            report.append("-" * 40)
            report.append(f"Total Issues: {getattr(dead_code, 'total_issues', 0)}")
            report.append(f"Deprecated Issues: {len(getattr(dead_code, 'deprecated_issues', []))}")
            report.append("")
        
        return "\n".join(report)
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed text report."""
        report = []
        report.append(self.generate_summary_report(results))
        
        # Add detailed sections
        if "dead_code" in results:
            report.append(self._generate_dead_code_details(results["dead_code"]))
        
        return "\n".join(report)
    
    def _generate_dead_code_details(self, dead_code: Dict[str, Any]) -> str:
        """Generate detailed dead code analysis."""
        details = []
        details.append("DETAILED DEAD CODE ANALYSIS")
        details.append("-" * 40)
        
        if hasattr(dead_code, 'deprecated_issues') and dead_code.deprecated_issues:
            details.append(f"Deprecated Code ({len(dead_code.deprecated_issues)}):")
            for issue in dead_code.deprecated_issues[:10]:  # Show top 10
                details.append(f"  • {issue.file_path}:{issue.line_number} - {issue.description}")
        
        return "\n".join(details)
