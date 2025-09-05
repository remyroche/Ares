#!/usr/bin/env python3
"""
Redundancy Analysis Tool

This tool analyzes the 181 scripts in the code_quality directory to identify:
1. Redundant functionality
2. Duplicate implementations
3. Overlapping features
4. Consolidation opportunities
"""

import os
import sys
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class RedundancyAnalyzer:
    """Analyzes code redundancy and duplication."""
    
    def __init__(self, code_quality_root: str = "/workspace/code_quality"):
        self.code_quality_root = Path(code_quality_root)
        self.script_functions = defaultdict(list)  # function_name -> [(file, line)]
        self.script_classes = defaultdict(list)    # class_name -> [(file, line)]
        self.script_imports = defaultdict(list)    # import -> [(file, line)]
        self.script_keywords = defaultdict(list)   # keyword -> [(file, line)]
        self.file_analysis = {}
        
        # Common patterns to look for
        self.analysis_patterns = [
            "analyze", "analysis", "analyzer",
            "fix", "fixer", "repair",
            "validate", "validator", "check",
            "import", "imports",
            "dead_code", "unused",
            "complexity", "complex",
            "syntax", "syntactic",
            "linter", "lint",
            "static", "static_analysis",
            "type_check", "type_checker",
            "undefined", "undefined_names",
            "circular", "circular_import",
            "dependency", "dependencies",
            "architecture", "architectural",
            "performance", "perf",
            "metrics", "metric",
            "coverage", "test_coverage",
            "documentation", "doc",
            "error_handling", "error",
            "concurrency", "async",
            "data_flow", "dataflow",
            "call_graph", "callgraph",
            "code_smell", "smell",
            "duplication", "duplicate",
            "configuration", "config"
        ]
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for patterns and functionality."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError:
                return {"error": "Syntax error", "patterns": [], "functions": [], "classes": []}
            
            # Extract functions and classes
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # Find patterns in content
            content_lower = content.lower()
            found_patterns = []
            for pattern in self.analysis_patterns:
                if pattern in content_lower:
                    found_patterns.append(pattern)
            
            return {
                "file_path": str(file_path),
                "patterns": found_patterns,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "line_count": len(content.split('\n')),
                "size": len(content)
            }
            
        except Exception as e:
            return {"error": str(e), "patterns": [], "functions": [], "classes": []}
    
    def analyze_all_scripts(self) -> Dict[str, Any]:
        """Analyze all scripts in the code_quality directory."""
        print("🔍 Analyzing script redundancy...")
        
        # Find all Python files
        python_files = []
        for py_file in self.code_quality_root.rglob("*.py"):
            # Skip backup files, test files, and reports
            if (py_file.name.endswith(".backup") or 
                "test" in py_file.name.lower() or
                "reports" in str(py_file) or
                "tests" in str(py_file)):
                continue
            python_files.append(py_file)
        
        print(f"📁 Found {len(python_files)} Python files to analyze")
        
        # Analyze each file
        for i, file_path in enumerate(python_files, 1):
            if i % 20 == 0:
                print(f"   Analyzed {i}/{len(python_files)} files...")
            
            analysis = self.analyze_file(file_path)
            self.file_analysis[str(file_path)] = analysis
            
            # Track functions, classes, and patterns
            for func in analysis.get("functions", []):
                self.script_functions[func].append(str(file_path))  # Just store file path
            
            for cls in analysis.get("classes", []):
                self.script_classes[cls].append(str(file_path))
            
            for imp in analysis.get("imports", []):
                self.script_imports[imp].append(str(file_path))
            
            for pattern in analysis.get("patterns", []):
                self.script_keywords[pattern].append(str(file_path))
        
        return self.generate_redundancy_report()
    
    def generate_redundancy_report(self) -> Dict[str, Any]:
        """Generate comprehensive redundancy report."""
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_files": len(self.file_analysis),
            "redundancy_summary": {},
            "duplicate_functions": {},
            "duplicate_classes": {},
            "pattern_analysis": {},
            "consolidation_opportunities": [],
            "recommendations": []
        }
        
        # Find duplicate functions
        duplicate_functions = {name: files for name, files in self.script_functions.items() if len(files) > 1}
        report["duplicate_functions"] = duplicate_functions
        
        # Find duplicate classes
        duplicate_classes = {name: files for name, files in self.script_classes.items() if len(files) > 1}
        report["duplicate_classes"] = duplicate_classes
        
        # Analyze patterns
        pattern_analysis = {}
        for pattern, files in self.script_keywords.items():
            if len(files) > 1:
                pattern_analysis[pattern] = {
                    "count": len(files),
                    "files": files
                }
        report["pattern_analysis"] = pattern_analysis
        
        # Identify consolidation opportunities
        consolidation_opportunities = []
        
        # Group files by similar patterns
        pattern_groups = defaultdict(list)
        for file_path, analysis in self.file_analysis.items():
            patterns = analysis.get("patterns", [])
            if patterns:
                key = tuple(sorted(patterns))
                pattern_groups[key].append(file_path)
        
        for patterns, files in pattern_groups.items():
            if len(files) > 1:
                consolidation_opportunities.append({
                    "patterns": list(patterns),
                    "files": files,
                    "potential_consolidation": True,
                    "reason": f"Files share patterns: {', '.join(patterns)}"
                })
        
        report["consolidation_opportunities"] = consolidation_opportunities
        
        # Generate recommendations
        recommendations = []
        
        # Function duplication recommendations
        if duplicate_functions:
            recommendations.append({
                "type": "function_duplication",
                "count": len(duplicate_functions),
                "description": f"Found {len(duplicate_functions)} functions with duplicate names across multiple files",
                "action": "Review and consolidate duplicate functions"
            })
        
        # Class duplication recommendations
        if duplicate_classes:
            recommendations.append({
                "type": "class_duplication",
                "count": len(duplicate_classes),
                "description": f"Found {len(duplicate_classes)} classes with duplicate names across multiple files",
                "action": "Review and consolidate duplicate classes"
            })
        
        # Pattern-based recommendations
        high_frequency_patterns = {p: data for p, data in pattern_analysis.items() if data["count"] > 5}
        if high_frequency_patterns:
            recommendations.append({
                "type": "pattern_consolidation",
                "count": len(high_frequency_patterns),
                "description": f"Found {len(high_frequency_patterns)} patterns with high frequency across files",
                "action": "Consider creating shared modules for common functionality"
            })
        
        # Consolidation recommendations
        if consolidation_opportunities:
            recommendations.append({
                "type": "file_consolidation",
                "count": len(consolidation_opportunities),
                "description": f"Found {len(consolidation_opportunities)} groups of files that could be consolidated",
                "action": "Review file groups for consolidation opportunities"
            })
        
        report["recommendations"] = recommendations
        
        # Summary statistics
        report["redundancy_summary"] = {
            "total_files": len(self.file_analysis),
            "duplicate_functions": len(duplicate_functions),
            "duplicate_classes": len(duplicate_classes),
            "high_frequency_patterns": len(high_frequency_patterns),
            "consolidation_groups": len(consolidation_opportunities),
            "total_recommendations": len(recommendations)
        }
        
        return report
    
    def print_redundancy_report(self, report: Dict[str, Any]):
        """Print the redundancy report."""
        print("\n" + "=" * 80)
        print("REDUNDANCY ANALYSIS REPORT")
        print("=" * 80)
        print(f"Analysis time: {report['analysis_timestamp']}")
        print(f"Total files analyzed: {report['total_files']}")
        print()
        
        # Summary
        summary = report["redundancy_summary"]
        print("SUMMARY:")
        print(f"  Duplicate functions: {summary['duplicate_functions']}")
        print(f"  Duplicate classes: {summary['duplicate_classes']}")
        print(f"  High frequency patterns: {summary['high_frequency_patterns']}")
        print(f"  Consolidation groups: {summary['consolidation_groups']}")
        print(f"  Total recommendations: {summary['total_recommendations']}")
        print()
        
        # Duplicate functions
        if report["duplicate_functions"]:
            print("DUPLICATE FUNCTIONS:")
            for func_name, files in list(report["duplicate_functions"].items())[:10]:  # Show first 10
                print(f"  {func_name}: {len(files)} files")
                for file_path in files[:3]:  # Show first 3 files
                    print(f"    - {Path(file_path).name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
            if len(report["duplicate_functions"]) > 10:
                print(f"  ... and {len(report['duplicate_functions']) - 10} more duplicate functions")
            print()
        
        # Duplicate classes
        if report["duplicate_classes"]:
            print("DUPLICATE CLASSES:")
            for class_name, files in list(report["duplicate_classes"].items())[:10]:  # Show first 10
                print(f"  {class_name}: {len(files)} files")
                for file_path in files[:3]:  # Show first 3 files
                    print(f"    - {Path(file_path).name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
            if len(report["duplicate_classes"]) > 10:
                print(f"  ... and {len(report['duplicate_classes']) - 10} more duplicate classes")
            print()
        
        # High frequency patterns
        high_freq_patterns = {p: data for p, data in report["pattern_analysis"].items() if data["count"] > 5}
        if high_freq_patterns:
            print("HIGH FREQUENCY PATTERNS:")
            for pattern, data in sorted(high_freq_patterns.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
                print(f"  {pattern}: {data['count']} files")
            print()
        
        # Consolidation opportunities
        if report["consolidation_opportunities"]:
            print("CONSOLIDATION OPPORTUNITIES:")
            for i, opp in enumerate(report["consolidation_opportunities"][:10], 1):
                print(f"  {i}. {opp['reason']}")
                print(f"     Files: {len(opp['files'])}")
                for file_path in opp['files'][:3]:
                    print(f"       - {Path(file_path).name}")
                if len(opp['files']) > 3:
                    print(f"       ... and {len(opp['files']) - 3} more")
            if len(report["consolidation_opportunities"]) > 10:
                print(f"  ... and {len(report['consolidation_opportunities']) - 10} more opportunities")
            print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec['description']}")
            print(f"     Action: {rec['action']}")
        print()
        
        print("=" * 80)
    
    def save_report(self, report: Dict[str, Any], output_file: str = None) -> str:
        """Save the redundancy report to a file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"redundancy_analysis_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_file


def main():
    """Main entry point for redundancy analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Redundancy Analysis Tool")
    parser.add_argument("--output", "-o", help="Output file for report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RedundancyAnalyzer()
    
    # Run analysis
    report = analyzer.analyze_all_scripts()
    
    # Print report
    analyzer.print_redundancy_report(report)
    
    # Save report
    output_file = analyzer.save_report(report, args.output)
    print(f"📄 Report saved to: {output_file}")


if __name__ == "__main__":
    main()