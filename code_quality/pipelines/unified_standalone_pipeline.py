#!/usr/bin/env python3
"""
Unified Standalone Pipeline

This pipeline can run completely independently without any external imports or dependencies.
It provides basic code quality analysis using only Python standard library.

Features:
- No external dependencies required
- Basic syntax validation
- Simple import analysis
- File structure analysis
- Basic code metrics
- Standalone execution
"""

import os
import sys
import ast
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    file_path: str
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    syntax_errors: List[str]
    import_issues: List[str]
    complexity_score: int
    analysis_time: float


class StandaloneCodeAnalyzer:
    """Standalone code analyzer using only Python standard library."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = []
        self.total_files = 0
        self.total_errors = 0
        
    def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a single Python file."""
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            lines = content.split('\n')
            line_count = len(lines)
            
            # Parse AST
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                return AnalysisResult(
                    file_path=str(file_path),
                    line_count=line_count,
                    function_count=0,
                    class_count=0,
                    import_count=0,
                    syntax_errors=[f"Syntax error at line {e.lineno}: {e.msg}"],
                    import_issues=[],
                    complexity_score=0,
                    analysis_time=time.time() - start_time
                )
            
            # Count functions and classes
            function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            
            # Analyze imports
            imports = []
            import_issues = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # Check for common import issues
            if len(imports) > 20:
                import_issues.append("Too many imports (>20)")
            
            # Simple complexity calculation
            complexity_score = function_count + class_count + len([node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.While, ast.Try))])
            
            return AnalysisResult(
                file_path=str(file_path),
                line_count=line_count,
                function_count=function_count,
                class_count=class_count,
                import_count=len(imports),
                syntax_errors=[],
                import_issues=import_issues,
                complexity_score=complexity_score,
                analysis_time=time.time() - start_time
            )
            
        except Exception as e:
            return AnalysisResult(
                file_path=str(file_path),
                line_count=0,
                function_count=0,
                class_count=0,
                import_count=0,
                syntax_errors=[f"Error reading file: {str(e)}"],
                import_issues=[],
                complexity_score=0,
                analysis_time=time.time() - start_time
            )
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze the entire project."""
        print("🔍 Standalone Code Quality Analysis")
        print("=" * 50)
        print(f"Project root: {self.project_root}")
        print()
        
        # Find all Python files
        python_files = self.find_python_files()
        self.total_files = len(python_files)
        
        print(f"📁 Found {self.total_files} Python files")
        print()
        
        # Analyze each file
        for i, file_path in enumerate(python_files, 1):
            print(f"📄 Analyzing {i}/{self.total_files}: {file_path.relative_to(self.project_root)}")
            
            result = self.analyze_file(file_path)
            self.results.append(result)
            
            # Count errors
            self.total_errors += len(result.syntax_errors) + len(result.import_issues)
            
            if result.syntax_errors or result.import_issues:
                print(f"   ⚠️  Found {len(result.syntax_errors)} syntax errors, {len(result.import_issues)} import issues")
            else:
                print(f"   ✅ OK ({result.line_count} lines, {result.function_count} functions, {result.class_count} classes)")
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate analysis summary."""
        if not self.results:
            return {"error": "No files analyzed"}
        
        total_lines = sum(r.line_count for r in self.results)
        total_functions = sum(r.function_count for r in self.results)
        total_classes = sum(r.class_count for r in self.results)
        total_imports = sum(r.import_count for r in self.results)
        total_syntax_errors = sum(len(r.syntax_errors) for r in self.results)
        total_import_issues = sum(len(r.import_issues) for r in self.results)
        avg_complexity = sum(r.complexity_score for r in self.results) / len(self.results)
        total_analysis_time = sum(r.analysis_time for r in self.results)
        
        summary = {
            "project_root": str(self.project_root),
            "analysis_timestamp": datetime.now().isoformat(),
            "total_files": self.total_files,
            "total_lines": total_lines,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_imports": total_imports,
            "total_syntax_errors": total_syntax_errors,
            "total_import_issues": total_import_issues,
            "total_issues": total_syntax_errors + total_import_issues,
            "average_complexity": round(avg_complexity, 2),
            "total_analysis_time": round(total_analysis_time, 2),
            "files_with_issues": len([r for r in self.results if r.syntax_errors or r.import_issues]),
            "files_clean": len([r for r in self.results if not r.syntax_errors and not r.import_issues])
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print analysis summary."""
        print("\n" + "=" * 50)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Project: {summary['project_root']}")
        print(f"Analysis time: {summary['analysis_timestamp']}")
        print()
        print(f"📁 Files analyzed: {summary['total_files']}")
        print(f"📝 Total lines: {summary['total_lines']:,}")
        print(f"🔧 Functions: {summary['total_functions']}")
        print(f"🏗️  Classes: {summary['total_classes']}")
        print(f"📦 Imports: {summary['total_imports']}")
        print()
        print(f"❌ Syntax errors: {summary['total_syntax_errors']}")
        print(f"⚠️  Import issues: {summary['total_import_issues']}")
        print(f"🚨 Total issues: {summary['total_issues']}")
        print()
        print(f"📈 Average complexity: {summary['average_complexity']}")
        print(f"⏱️  Analysis time: {summary['total_analysis_time']:.2f}s")
        print()
        print(f"✅ Clean files: {summary['files_clean']}")
        print(f"⚠️  Files with issues: {summary['files_with_issues']}")
        
        if summary['total_issues'] > 0:
            print(f"\n🎯 Issues per file: {summary['total_issues'] / summary['total_files']:.2f}")
        
        print("\n" + "=" * 50)
    
    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"standalone_analysis_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {
            "summary": self.generate_summary(),
            "files": [asdict(result) for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return output_file


def main():
    """Main entry point for standalone pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Standalone Pipeline - No external dependencies required",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze current directory
  python unified_standalone_pipeline.py
  
  # Analyze specific directory
  python unified_standalone_pipeline.py --project-root /path/to/project
  
  # Save results to file
  python unified_standalone_pipeline.py --output results.json
  
  # Verbose output
  python unified_standalone_pipeline.py --verbose
        """
    )
    
    parser.add_argument("--project-root", "-p", 
                       default=".",
                       help="Project root directory to analyze (default: current directory)")
    parser.add_argument("--output", "-o",
                       help="Output file for results (JSON format)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StandaloneCodeAnalyzer(args.project_root)
    
    # Run analysis
    summary = analyzer.analyze_project()
    
    # Print summary
    analyzer.print_summary(summary)
    
    # Save results
    if args.output:
        output_file = analyzer.save_results(args.output)
        print(f"\n📄 Results saved to: {output_file}")
    elif args.verbose:
        output_file = analyzer.save_results()
        print(f"\n📄 Results saved to: {output_file}")
    
    # Exit with error code if issues found
    if summary.get('total_issues', 0) > 0:
        print(f"\n⚠️  Found {summary['total_issues']} issues. Consider running full analysis pipeline.")
        sys.exit(1)
    else:
        print("\n✅ No issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()