#!/usr/bin/env python3
"""
Code Quality Monitoring Script

This script monitors code quality metrics for the feature generation system,
including duplication detection, complexity analysis, and performance tracking.
"""

import os
import sys
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

class CodeQualityMonitor:
    """Monitor code quality metrics for the feature generation system."""
    
    def __init__(self):
        self.base_dir = Path("src/feature_generation")
        self.metrics = {
            'duplicate_methods': 0,
            'total_lines': 0,
            'total_files': 0,
            'complex_methods': 0,
            'long_methods': 0,
            'large_files': 0,
            'import_issues': 0,
            'documentation_coverage': 0.0
        }
    
    def scan_duplicate_methods(self) -> Dict[str, int]:
        """Scan for duplicate method patterns."""
        print("🔍 Scanning for duplicate methods...")
        
        duplicates = defaultdict(int)
        method_patterns = [
            r'def optimize_dataframe_processing\(',
            r'def vectorized_rolling_operations\('
        ]
        
        for py_file in self.base_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in method_patterns:
                    matches = len(re.findall(pattern, content))
                    if matches > 0:
                        duplicates[f"{py_file.relative_to(self.base_dir)}:{pattern}"] = matches
                        
            except Exception as e:
                print(f"⚠️ Error reading {py_file}: {e}")
        
        self.metrics['duplicate_methods'] = sum(duplicates.values())
        return dict(duplicates)
    
    def analyze_file_complexity(self) -> Dict[str, Any]:
        """Analyze file complexity and size."""
        print("🔍 Analyzing file complexity...")
        
        file_stats = []
        total_lines = 0
        total_files = 0
        complex_methods = 0
        long_methods = 0
        large_files = 0
        
        for py_file in self.base_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_lines = len(lines)
                total_lines += file_lines
                total_files += 1
                
                # Check for large files (>1000 lines)
                if file_lines > 1000:
                    large_files += 1
                
                # Analyze method complexity
                in_method = False
                method_lines = 0
                method_start = 0
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Method definition
                    if re.match(r'def\s+\w+\(', stripped):
                        if in_method:
                            # Previous method ended
                            if method_lines > 50:  # Long method
                                long_methods += 1
                            if method_lines > 20:  # Complex method
                                complex_methods += 1
                        
                        in_method = True
                        method_lines = 0
                        method_start = i
                    
                    elif in_method:
                        if stripped == '' or stripped.startswith('#'):
                            method_lines += 1
                        elif not stripped.startswith(' ') and not stripped.startswith('\t'):
                            # Method ended
                            if method_lines > 50:
                                long_methods += 1
                            if method_lines > 20:
                                complex_methods += 1
                            in_method = False
                            method_lines = 0
                        else:
                            method_lines += 1
                
                # Count methods and classes
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_stats.append({
                    'file': str(py_file.relative_to(self.base_dir)),
                    'lines': file_lines,
                    'methods': len(re.findall(r'def\s+\w+\(', content)),
                    'classes': len(re.findall(r'class\s+\w+', content))
                })
                
            except Exception as e:
                print(f"⚠️ Error analyzing {py_file}: {e}")
        
        self.metrics.update({
            'total_lines': total_lines,
            'total_files': total_files,
            'complex_methods': complex_methods,
            'long_methods': long_methods,
            'large_files': large_files
        })
        
        return {
            'file_stats': file_stats,
            'metrics': self.metrics
        }
    
    def check_import_issues(self) -> List[str]:
        """Check for import issues and circular dependencies."""
        print("🔍 Checking import issues...")
        
        issues = []
        import_pattern = r'^from\s+\.\.?.*import|^import\s+'
        
        for py_file in self.base_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    if re.match(import_pattern, line.strip()):
                        # Check for common import issues
                        if 'from ..' in line and 'import *' in line:
                            issues.append(f"{py_file.relative_to(self.base_dir)}:{i}: Wildcard import")
                        
                        if 'import' in line and 'as' not in line and ',' in line:
                            issues.append(f"{py_file.relative_to(self.base_dir)}:{i}: Multiple imports on one line")
                            
            except Exception as e:
                print(f"⚠️ Error checking imports in {py_file}: {e}")
        
        self.metrics['import_issues'] = len(issues)
        return issues
    
    def check_documentation_coverage(self) -> float:
        """Check documentation coverage."""
        print("🔍 Checking documentation coverage...")
        
        documented_methods = 0
        total_methods = 0
        
        for py_file in self.base_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find method definitions
                method_matches = re.finditer(r'def\s+(\w+)\(', content)
                
                for match in method_matches:
                    method_name = match.group(1)
                    if not method_name.startswith('_'):  # Public methods
                        total_methods += 1
                        
                        # Check if method has docstring
                        start_pos = match.start()
                        method_content = content[start_pos:start_pos + 500]  # Look ahead 500 chars
                        
                        if '"""' in method_content or "'''" in method_content:
                            documented_methods += 1
                            
            except Exception as e:
                print(f"⚠️ Error checking documentation in {py_file}: {e}")
        
        coverage = (documented_methods / total_methods * 100) if total_methods > 0 else 0
        self.metrics['documentation_coverage'] = coverage
        return coverage
    
    def generate_report(self) -> str:
        """Generate a comprehensive code quality report."""
        print("📊 Generating code quality report...")
        
        # Run all analyses
        duplicates = self.scan_duplicate_methods()
        complexity = self.analyze_file_complexity()
        import_issues = self.check_import_issues()
        doc_coverage = self.check_documentation_coverage()
        
        # Generate report
        report = []
        report.append("=" * 60)
        report.append("📊 FEATURE GENERATION CODE QUALITY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary metrics
        report.append("📈 SUMMARY METRICS")
        report.append("-" * 30)
        report.append(f"Total Files: {self.metrics['total_files']:,}")
        report.append(f"Total Lines: {self.metrics['total_lines']:,}")
        report.append(f"Duplicate Methods: {self.metrics['duplicate_methods']}")
        report.append(f"Complex Methods: {self.metrics['complex_methods']}")
        report.append(f"Long Methods: {self.metrics['long_methods']}")
        report.append(f"Large Files (>1000 lines): {self.metrics['large_files']}")
        report.append(f"Import Issues: {self.metrics['import_issues']}")
        report.append(f"Documentation Coverage: {self.metrics['documentation_coverage']:.1f}%")
        report.append("")
        
        # Duplicate methods details
        if duplicates:
            report.append("🔄 DUPLICATE METHODS")
            report.append("-" * 30)
            for location, count in sorted(duplicates.items()):
                report.append(f"{location}: {count}")
            report.append("")
        
        # File complexity details
        if complexity['file_stats']:
            report.append("📁 FILE COMPLEXITY")
            report.append("-" * 30)
            # Sort by lines (largest first)
            sorted_files = sorted(complexity['file_stats'], key=lambda x: x['lines'], reverse=True)
            for file_info in sorted_files[:10]:  # Top 10 largest files
                report.append(f"{file_info['file']}: {file_info['lines']} lines, {file_info['methods']} methods, {file_info['classes']} classes")
            report.append("")
        
        # Import issues details
        if import_issues:
            report.append("⚠️ IMPORT ISSUES")
            report.append("-" * 30)
            for issue in import_issues[:10]:  # First 10 issues
                report.append(issue)
            if len(import_issues) > 10:
                report.append(f"... and {len(import_issues) - 10} more issues")
            report.append("")
        
        # Quality score
        quality_score = self.calculate_quality_score()
        report.append("🎯 QUALITY SCORE")
        report.append("-" * 30)
        report.append(f"Overall Score: {quality_score:.1f}/100")
        
        if quality_score >= 90:
            report.append("Status: 🟢 EXCELLENT")
        elif quality_score >= 80:
            report.append("Status: 🟡 GOOD")
        elif quality_score >= 70:
            report.append("Status: 🟠 FAIR")
        else:
            report.append("Status: 🔴 NEEDS IMPROVEMENT")
        
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if self.metrics['duplicate_methods'] > 0:
            report.append("• Remove remaining duplicate methods")
        
        if self.metrics['complex_methods'] > 10:
            report.append("• Refactor complex methods (break into smaller functions)")
        
        if self.metrics['long_methods'] > 5:
            report.append("• Refactor long methods (reduce line count)")
        
        if self.metrics['large_files'] > 3:
            report.append("• Consider splitting large files")
        
        if self.metrics['import_issues'] > 0:
            report.append("• Fix import issues and avoid wildcard imports")
        
        if self.metrics['documentation_coverage'] < 80:
            report.append("• Improve documentation coverage")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct points for issues
        if self.metrics['duplicate_methods'] > 0:
            score -= min(20, self.metrics['duplicate_methods'] * 2)
        
        if self.metrics['complex_methods'] > 10:
            score -= min(15, (self.metrics['complex_methods'] - 10) * 0.5)
        
        if self.metrics['long_methods'] > 5:
            score -= min(10, (self.metrics['long_methods'] - 5) * 1)
        
        if self.metrics['large_files'] > 3:
            score -= min(10, (self.metrics['large_files'] - 3) * 2)
        
        if self.metrics['import_issues'] > 0:
            score -= min(10, self.metrics['import_issues'] * 0.5)
        
        if self.metrics['documentation_coverage'] < 80:
            score -= (80 - self.metrics['documentation_coverage']) * 0.2
        
        return max(0, score)

def main():
    """Run code quality monitoring."""
    print("🚀 Feature Generation Code Quality Monitor")
    print("=" * 50)
    
    monitor = CodeQualityMonitor()
    report = monitor.generate_report()
    
    print(report)
    
    # Save report to file
    report_file = Path("src/feature_generation/CODE_QUALITY_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Report saved to: {report_file}")
    
    # Return quality score for CI/CD
    quality_score = monitor.calculate_quality_score()
    if quality_score < 70:
        print(f"\n⚠️ Quality score {quality_score:.1f} is below threshold (70)")
        return 1
    else:
        print(f"\n✅ Quality score {quality_score:.1f} meets standards")
        return 0

if __name__ == "__main__":
    sys.exit(main())