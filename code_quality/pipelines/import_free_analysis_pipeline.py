#!/usr/bin/env python3
"""
Import-Free Code Analysis Pipeline

Specialized pipeline for code analysis that doesn't rely on imports, including:
- AST-based analysis
- Syntax validation
- Code structure analysis
- Basic metrics calculation
- Pattern detection
"""

import ast
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
from collections import defaultdict, Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ImportFreeAnalyzer:
    """Base class for import-free analysis."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for py_file in self.project_root.rglob("*.py"):
            # Skip common directories to avoid
            if any(skip_dir in str(py_file) for skip_dir in ["__pycache__", ".git", "venv", "env", "node_modules"]):
                continue
            python_files.append(py_file)
        return python_files
    
    def parse_file(self, file_path: Path) -> ast.AST:
        """Parse a Python file and return AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            return None


class SyntaxAnalyzer(ImportFreeAnalyzer):
    """Import-free syntax analyzer."""
    
    def analyze_syntax(self) -> Dict[str, Any]:
        """Analyze syntax without imports."""
        print("\n" + "="*60)
        print("Running Import-Free Syntax Analysis")
        print("="*60)
        
        python_files = self.find_python_files()
        syntax_issues = []
        valid_files = 0
        
        for file_path in python_files:
            try:
                tree = self.parse_file(file_path)
                if tree is not None:
                    valid_files += 1
                    # Basic syntax validation
                    issues = self._check_syntax_issues(tree, file_path)
                    syntax_issues.extend(issues)
            except SyntaxError as e:
                syntax_issues.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "issue": f"Syntax error: {e.msg}",
                    "severity": "error"
                })
            except Exception as e:
                syntax_issues.append({
                    "file": str(file_path),
                    "line": 0,
                    "issue": f"Parse error: {str(e)}",
                    "severity": "error"
                })
        
        return {
            "total_files": len(python_files),
            "valid_files": valid_files,
            "syntax_issues": syntax_issues,
            "issues_by_severity": self._count_by_severity(syntax_issues)
        }
    
    def _check_syntax_issues(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Check for common syntax issues."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for common issues
            if isinstance(node, ast.FunctionDef):
                # Check for functions without docstrings
                if not ast.get_docstring(node):
                    issues.append({
                        "file": str(file_path),
                        "line": node.lineno,
                        "issue": f"Function '{node.name}' missing docstring",
                        "severity": "warning"
                    })
            
            elif isinstance(node, ast.ClassDef):
                # Check for classes without docstrings
                if not ast.get_docstring(node):
                    issues.append({
                        "file": str(file_path),
                        "line": node.lineno,
                        "issue": f"Class '{node.name}' missing docstring",
                        "severity": "warning"
                    })
            
            elif isinstance(node, ast.Import):
                # Check for wildcard imports
                for alias in node.names:
                    if alias.name == "*":
                        issues.append({
                            "file": str(file_path),
                            "line": node.lineno,
                            "issue": "Wildcard import detected",
                            "severity": "warning"
                        })
        
        return issues
    
    def _count_by_severity(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count issues by severity."""
        counts = defaultdict(int)
        for issue in issues:
            counts[issue["severity"]] += 1
        return dict(counts)


class StructureAnalyzer(ImportFreeAnalyzer):
    """Import-free code structure analyzer."""
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze code structure without imports."""
        print("\n" + "="*60)
        print("Running Import-Free Structure Analysis")
        print("="*60)
        
        python_files = self.find_python_files()
        structure_metrics = {
            "total_files": len(python_files),
            "total_functions": 0,
            "total_classes": 0,
            "total_lines": 0,
            "average_function_length": 0,
            "average_class_length": 0,
            "complexity_distribution": defaultdict(int),
            "file_metrics": []
        }
        
        function_lengths = []
        class_lengths = []
        
        for file_path in python_files:
            try:
                tree = self.parse_file(file_path)
                if tree is None:
                    continue
                
                file_metrics = self._analyze_file_structure(tree, file_path)
                structure_metrics["file_metrics"].append(file_metrics)
                
                structure_metrics["total_functions"] += file_metrics["functions"]
                structure_metrics["total_classes"] += file_metrics["classes"]
                structure_metrics["total_lines"] += file_metrics["lines"]
                
                function_lengths.extend(file_metrics["function_lengths"])
                class_lengths.extend(file_metrics["class_lengths"])
                
                # Complexity distribution
                for complexity in file_metrics["complexities"]:
                    structure_metrics["complexity_distribution"][complexity] += 1
                
            except Exception as e:
                print(f"Warning: Could not analyze structure of {file_path}: {e}")
        
        # Calculate averages
        if function_lengths:
            structure_metrics["average_function_length"] = sum(function_lengths) / len(function_lengths)
        if class_lengths:
            structure_metrics["average_class_length"] = sum(class_lengths) / len(class_lengths)
        
        structure_metrics["complexity_distribution"] = dict(structure_metrics["complexity_distribution"])
        
        return structure_metrics
    
    def _analyze_file_structure(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Analyze structure of a single file."""
        functions = 0
        classes = 0
        lines = 0
        function_lengths = []
        class_lengths = []
        complexities = []
        
        # Count lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
        except:
            lines = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions += 1
                # Calculate function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_length = node.end_lineno - node.lineno + 1
                    function_lengths.append(func_length)
                
                # Calculate basic complexity
                complexity = self._calculate_complexity(node)
                complexities.append(complexity)
            
            elif isinstance(node, ast.ClassDef):
                classes += 1
                # Calculate class length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    class_length = node.end_lineno - node.lineno + 1
                    class_lengths.append(class_length)
        
        return {
            "file": str(file_path),
            "functions": functions,
            "classes": classes,
            "lines": lines,
            "function_lengths": function_lengths,
            "class_lengths": class_lengths,
            "complexities": complexities
        }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate basic cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity


class PatternAnalyzer(ImportFreeAnalyzer):
    """Import-free pattern analyzer."""
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze code patterns without imports."""
        print("\n" + "="*60)
        print("Running Import-Free Pattern Analysis")
        print("="*60)
        
        python_files = self.find_python_files()
        patterns = {
            "total_files": len(python_files),
            "pattern_counts": defaultdict(int),
            "pattern_issues": [],
            "file_patterns": []
        }
        
        for file_path in python_files:
            try:
                tree = self.parse_file(file_path)
                if tree is None:
                    continue
                
                file_patterns = self._analyze_file_patterns(tree, file_path)
                patterns["file_patterns"].append(file_patterns)
                
                # Aggregate pattern counts
                for pattern, count in file_patterns["patterns"].items():
                    patterns["pattern_counts"][pattern] += count
                
                # Collect pattern issues
                patterns["pattern_issues"].extend(file_patterns["issues"])
                
            except Exception as e:
                print(f"Warning: Could not analyze patterns in {file_path}: {e}")
        
        patterns["pattern_counts"] = dict(patterns["pattern_counts"])
        
        return patterns
    
    def _analyze_file_patterns(self, tree: ast.AST, file_path: Path) -> Dict[str, Any]:
        """Analyze patterns in a single file."""
        patterns = defaultdict(int)
        issues = []
        
        for node in ast.walk(tree):
            # Pattern detection
            if isinstance(node, ast.ListComp):
                patterns["list_comprehensions"] += 1
            
            elif isinstance(node, ast.DictComp):
                patterns["dict_comprehensions"] += 1
            
            elif isinstance(node, ast.SetComp):
                patterns["set_comprehensions"] += 1
            
            elif isinstance(node, ast.GeneratorExp):
                patterns["generator_expressions"] += 1
            
            elif isinstance(node, ast.Lambda):
                patterns["lambda_functions"] += 1
            
            elif isinstance(node, ast.DecoratorList):
                patterns["decorators"] += len(node.decorators)
            
            elif isinstance(node, ast.AsyncFunctionDef):
                patterns["async_functions"] += 1
            
            elif isinstance(node, ast.ClassDef):
                # Check for common class patterns
                if any(base.id == "Exception" for base in node.bases if isinstance(base, ast.Name)):
                    patterns["exception_classes"] += 1
                
                # Check for abstract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and any(
                        isinstance(dec, ast.Name) and dec.id == "abstractmethod"
                        for dec in item.decorator_list
                    ):
                        patterns["abstract_methods"] += 1
            
            # Issue detection
            if isinstance(node, ast.FunctionDef):
                # Check for long parameter lists
                if len(node.args.args) > 5:
                    issues.append({
                        "file": str(file_path),
                        "line": node.lineno,
                        "issue": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        "severity": "warning"
                    })
                
                # Check for long functions
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    func_length = node.end_lineno - node.lineno + 1
                    if func_length > 50:
                        issues.append({
                            "file": str(file_path),
                            "line": node.lineno,
                            "issue": f"Function '{node.name}' is too long ({func_length} lines)",
                            "severity": "warning"
                        })
        
        return {
            "file": str(file_path),
            "patterns": dict(patterns),
            "issues": issues
        }


class ImportFreeAnalysisPipeline:
    """Specialized pipeline for import-free code analysis."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
        # Initialize analyzers
        self.syntax_analyzer = SyntaxAnalyzer(str(self.project_root))
        self.structure_analyzer = StructureAnalyzer(str(self.project_root))
        self.pattern_analyzer = PatternAnalyzer(str(self.project_root))
        
        # Setup reports directory
        self.reports_dir = self.project_root / "code_quality" / "reports" / "import_free_analysis"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def run_syntax_analysis(self) -> Dict[str, Any]:
        """Run import-free syntax analysis."""
        try:
            results = self.syntax_analyzer.analyze_syntax()
            
            # Save report
            report_path = self.reports_dir / f"syntax_analysis_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_files": results["total_files"],
                "valid_files": results["valid_files"],
                "syntax_issues": len(results["syntax_issues"]),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_structure_analysis(self) -> Dict[str, Any]:
        """Run import-free structure analysis."""
        try:
            results = self.structure_analyzer.analyze_structure()
            
            # Save report
            report_path = self.reports_dir / f"structure_analysis_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_functions": results["total_functions"],
                "total_classes": results["total_classes"],
                "total_lines": results["total_lines"],
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_pattern_analysis(self) -> Dict[str, Any]:
        """Run import-free pattern analysis."""
        try:
            results = self.pattern_analyzer.analyze_patterns()
            
            # Save report
            report_path = self.reports_dir / f"pattern_analysis_{self.timestamp}.json"
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)
            
            return {
                "status": "completed",
                "report_path": str(report_path),
                "total_files": results["total_files"],
                "pattern_types": len(results["pattern_counts"]),
                "pattern_issues": len(results["pattern_issues"]),
                "results": results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_import_free_analysis(self) -> Dict[str, Any]:
        """Run comprehensive import-free analysis."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE IMPORT-FREE ANALYSIS PIPELINE")
        print(f"{'='*80}")
        print(f"Project root: {self.project_root}")
        print(f"Timestamp: {self.timestamp}")
        print("Note: This analysis runs without external imports for maximum compatibility")
        
        total_start = time.time()
        
        # Run all import-free analyses
        self.results["syntax_analysis"] = self.run_syntax_analysis()
        self.results["structure_analysis"] = self.run_structure_analysis()
        self.results["pattern_analysis"] = self.run_pattern_analysis()
        
        # Generate summary
        total_time = time.time() - total_start
        self.results["summary"] = {
            "timestamp": self.timestamp,
            "project_root": str(self.project_root),
            "total_execution_time": total_time,
            "analysis_categories": len(self.results) - 1,  # Exclude summary
            "import_free": True,
            "status": "completed"
        }
        
        # Save comprehensive report
        report_path = self.reports_dir / f"import_free_analysis_{self.timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("IMPORT-FREE ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Reports saved to: {self.reports_dir}")
        print(f"Main report: {report_path}")
        
        return self.results


def main():
    """Main entry point for the import-free analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Import-Free Analysis Pipeline - Code analysis without external dependencies"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["syntax", "structure", "patterns"],
        default="syntax",
        help="Type of import-free analysis to perform (default: syntax)"
    )
    
    args = parser.parse_args()
    
    pipeline = ImportFreeAnalysisPipeline(project_root=args.project_root)
    
    if args.analysis_type == "all":
        results = pipeline.run_all_import_free_analysis()
    elif args.analysis_type == "syntax":
        results = pipeline.run_syntax_analysis()
    elif args.analysis_type == "structure":
        results = pipeline.run_structure_analysis()
    elif args.analysis_type == "patterns":
        results = pipeline.run_pattern_analysis()
    
    print(f"\nImport-free analysis pipeline completed with status: {results.get('status', 'unknown')}")


if __name__ == "__main__":
    main()