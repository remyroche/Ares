#!/usr/bin/env python3
"""
Unified Code Quality Pipeline - Enhanced Standalone Version

A standalone version that provides comprehensive code quality analysis
without complex import dependencies.
"""

import ast
import json
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedEnhancedPipeline:
    """Enhanced unified pipeline with comprehensive analysis capabilities."""

    def __init__(self, project_root: str, output_dir: str = None):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            "project_root": str(self.project_root),
            "timestamp": datetime.now().isoformat(),
            "analysis_results": {},
            "summary": {},
            "files_analyzed": 0,
            "total_issues": 0,
            "execution_time": 0.0
        }
        
        # Analysis tools
        self.tools = {
            "syntax_analysis": self._analyze_syntax,
            "import_analysis": self._analyze_imports,
            "complexity_analysis": self._analyze_complexity,
            "dead_code_analysis": self._analyze_dead_code,
            "style_analysis": self._analyze_style,
            "security_analysis": self._analyze_security,
        }

    def run_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on the project."""
        start_time = time.time()
        
        print("🚀 Starting Enhanced Unified Code Quality Pipeline")
        print("=" * 80)
        print(f"📁 Project: {self.project_root}")
        print(f"📊 Output: {self.output_dir}")
        print("=" * 80)
        
        # Find Python files
        python_files = self._find_python_files()
        self.results["files_analyzed"] = len(python_files)
        
        print(f"📁 Found {len(python_files)} Python files to analyze")
        
        # Run each analysis tool
        for tool_name, tool_func in self.tools.items():
            print(f"\n🔧 Running {tool_name.replace('_', ' ').title()}...")
            try:
                result = tool_func(python_files)
                self.results["analysis_results"][tool_name] = result
                print(f"   ✅ {tool_name}: {result.get('issues_found', 0)} issues found")
            except Exception as e:
                print(f"   ❌ {tool_name}: Error - {e}")
                self.results["analysis_results"][tool_name] = {"error": str(e)}
        
        # Generate summary
        self._generate_summary()
        
        # Calculate execution time
        end_time = time.time()
        self.results["execution_time"] = end_time - start_time
        
        # Save results
        self._save_results()
        
        print("\n" + "=" * 80)
        print("🎉 ENHANCED UNIFIED PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"📊 Total Issues Found: {self.results['total_issues']}")
        print(f"⏱️  Execution Time: {self.results['execution_time']:.2f} seconds")
        print(f"📁 Files Analyzed: {self.results['files_analyzed']}")
        print(f"💾 Reports saved to: {self.output_dir}")
        
        return self.results

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        exclude_patterns = [
            "*/__pycache__/*",
            "*/.*/*",
            "*/venv/*",
            "*/env/*",
            "*/node_modules/*",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.dylib",
        ]
        
        for file_path in self.project_root.rglob("*.py"):
            should_exclude = False
            for pattern in exclude_patterns:
                if file_path.match(pattern):
                    should_exclude = True
                    break
            
            if not should_exclude:
                python_files.append(file_path)
        
        return python_files

    def _analyze_syntax(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze syntax issues."""
        issues = []
        files_with_errors = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                files_with_errors += 1
                issues.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "message": str(e.msg),
                    "severity": "error",
                    "type": "syntax_error"
                })
            except Exception as e:
                files_with_errors += 1
                issues.append({
                    "file": str(file_path),
                    "line": 0,
                    "message": f"Parse error: {str(e)}",
                    "severity": "error",
                    "type": "parse_error"
                })
        
        return {
            "issues_found": len(issues),
            "files_with_errors": files_with_errors,
            "issues": issues
        }

    def _analyze_imports(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze import issues."""
        issues = []
        import_stats = {
            "total_imports": 0,
            "unused_imports": 0,
            "circular_imports": 0
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                # Find imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.asname or alias.name)
                            import_stats["total_imports"] += 1
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            for alias in node.names:
                                imports.append(alias.asname or alias.name)
                                import_stats["total_imports"] += 1
                
                # Check for unused imports (simple heuristic)
                lines = content.split('\n')
                for import_name in imports:
                    if import_name.startswith('_'):
                        continue
                    
                    usage_count = 0
                    for line in lines:
                        if import_name in line and not line.strip().startswith(('import', 'from')):
                            usage_count += 1
                    
                    if usage_count == 0:
                        import_stats["unused_imports"] += 1
                        issues.append({
                            "file": str(file_path),
                            "line": 0,
                            "message": f"Unused import: {import_name}",
                            "severity": "warning",
                            "type": "unused_import"
                        })
                        
            except Exception as e:
                logger.warning(f"Error analyzing imports in {file_path}: {e}")
        
        return {
            "issues_found": len(issues),
            "import_stats": import_stats,
            "issues": issues
        }

    def _analyze_complexity(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze code complexity."""
        issues = []
        complexity_stats = {
            "high_complexity_functions": 0,
            "long_functions": 0,
            "deep_nesting": 0
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check function length
                        if len(node.body) > 50:
                            complexity_stats["long_functions"] += 1
                            issues.append({
                                "file": str(file_path),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' is too long ({len(node.body)} lines)",
                                "severity": "warning",
                                "type": "long_function"
                            })
                        
                        # Check nesting depth
                        max_depth = self._get_nesting_depth(node)
                        if max_depth > 4:
                            complexity_stats["deep_nesting"] += 1
                            issues.append({
                                "file": str(file_path),
                                "line": node.lineno,
                                "message": f"Function '{node.name}' has deep nesting (depth: {max_depth})",
                                "severity": "warning",
                                "type": "deep_nesting"
                            })
                            
            except Exception as e:
                logger.warning(f"Error analyzing complexity in {file_path}: {e}")
        
        return {
            "issues_found": len(issues),
            "complexity_stats": complexity_stats,
            "issues": issues
        }

    def _analyze_dead_code(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze for dead code."""
        issues = []
        dead_code_stats = {
            "unused_functions": 0,
            "unused_classes": 0,
            "unreachable_code": 0
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_'):
                            # Simple heuristic: check if function is called
                            lines = content.split('\n')
                            usage_count = 0
                            for line in lines:
                                if f"{node.name}(" in line and line.strip() != lines[node.lineno - 1].strip():
                                    usage_count += 1
                            
                            if usage_count == 0:
                                dead_code_stats["unused_functions"] += 1
                                issues.append({
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "message": f"Function '{node.name}' appears to be unused",
                                    "severity": "warning",
                                    "type": "unused_function"
                                })
                    
                    elif isinstance(node, ast.ClassDef):
                        if not node.name.startswith('_'):
                            # Simple heuristic: check if class is instantiated
                            lines = content.split('\n')
                            usage_count = 0
                            for line in lines:
                                if f"{node.name}(" in line and line.strip() != lines[node.lineno - 1].strip():
                                    usage_count += 1
                            
                            if usage_count == 0:
                                dead_code_stats["unused_classes"] += 1
                                issues.append({
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "message": f"Class '{node.name}' appears to be unused",
                                    "severity": "warning",
                                    "type": "unused_class"
                                })
                                
            except Exception as e:
                logger.warning(f"Error analyzing dead code in {file_path}: {e}")
        
        return {
            "issues_found": len(issues),
            "dead_code_stats": dead_code_stats,
            "issues": issues
        }

    def _analyze_style(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze code style issues."""
        issues = []
        style_stats = {
            "long_lines": 0,
            "missing_docstrings": 0,
            "trailing_whitespace": 0
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines, 1):
                    # Check line length
                    if len(line.rstrip()) > 120:
                        style_stats["long_lines"] += 1
                        issues.append({
                            "file": str(file_path),
                            "line": i,
                            "message": f"Line too long ({len(line.rstrip())} characters)",
                            "severity": "warning",
                            "type": "long_line"
                        })
                    
                    # Check trailing whitespace
                    if line.rstrip() != line.rstrip(' \t'):
                        style_stats["trailing_whitespace"] += 1
                        issues.append({
                            "file": str(file_path),
                            "line": i,
                            "message": "Trailing whitespace",
                            "severity": "info",
                            "type": "trailing_whitespace"
                        })
                
                # Check for missing docstrings in functions and classes
                try:
                    content = ''.join(lines)
                    tree = ast.parse(content, filename=str(file_path))
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            if not node.name.startswith('_') and not ast.get_docstring(node):
                                style_stats["missing_docstrings"] += 1
                                issues.append({
                                    "file": str(file_path),
                                    "line": node.lineno,
                                    "message": f"{'Function' if isinstance(node, ast.FunctionDef) else 'Class'} '{node.name}' missing docstring",
                                    "severity": "info",
                                    "type": "missing_docstring"
                                })
                except:
                    pass  # Skip if parsing fails
                    
            except Exception as e:
                logger.warning(f"Error analyzing style in {file_path}: {e}")
        
        return {
            "issues_found": len(issues),
            "style_stats": style_stats,
            "issues": issues
        }

    def _analyze_security(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analyze security issues."""
        issues = []
        security_stats = {
            "hardcoded_secrets": 0,
            "unsafe_functions": 0,
            "sql_injection_risks": 0
        }
        
        # Common security patterns to check
        unsafe_patterns = [
            ("eval(", "Use of eval() is dangerous"),
            ("exec(", "Use of exec() is dangerous"),
            ("subprocess.call", "Use subprocess.run() instead"),
            ("os.system", "Use subprocess.run() instead"),
        ]
        
        secret_patterns = [
            ("password", "Potential hardcoded password"),
            ("secret", "Potential hardcoded secret"),
            ("api_key", "Potential hardcoded API key"),
            ("token", "Potential hardcoded token"),
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    
                    # Check for unsafe functions
                    for pattern, message in unsafe_patterns:
                        if pattern in line_lower:
                            security_stats["unsafe_functions"] += 1
                            issues.append({
                                "file": str(file_path),
                                "line": i,
                                "message": message,
                                "severity": "error",
                                "type": "unsafe_function"
                            })
                    
                    # Check for potential secrets
                    for pattern, message in secret_patterns:
                        if pattern in line_lower and ('=' in line or ':' in line):
                            security_stats["hardcoded_secrets"] += 1
                            issues.append({
                                "file": str(file_path),
                                "line": i,
                                "message": message,
                                "severity": "warning",
                                "type": "potential_secret"
                            })
                            
            except Exception as e:
                logger.warning(f"Error analyzing security in {file_path}: {e}")
        
        return {
            "issues_found": len(issues),
            "security_stats": security_stats,
            "issues": issues
        }

    def _get_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth of a node."""
        max_depth = 0
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                depth = 1 + self._get_nesting_depth(child)
                max_depth = max(max_depth, depth)
            else:
                depth = self._get_nesting_depth(child)
                max_depth = max(max_depth, depth)
        
        return max_depth

    def _generate_summary(self) -> None:
        """Generate analysis summary."""
        total_issues = 0
        issues_by_severity = {"error": 0, "warning": 0, "info": 0}
        issues_by_type = {}
        
        for tool_name, result in self.results["analysis_results"].items():
            if "issues" in result:
                for issue in result["issues"]:
                    total_issues += 1
                    severity = issue.get("severity", "info")
                    issue_type = issue.get("type", "unknown")
                    
                    issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
                    issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        
        self.results["total_issues"] = total_issues
        self.results["summary"] = {
            "total_issues": total_issues,
            "issues_by_severity": issues_by_severity,
            "issues_by_type": issues_by_type,
            "tools_executed": len(self.results["analysis_results"]),
            "files_analyzed": self.results["files_analyzed"],
            "execution_time": self.results["execution_time"]
        }

    def _save_results(self) -> None:
        """Save analysis results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = self.output_dir / f"enhanced_pipeline_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        summary_file = self.output_dir / f"enhanced_pipeline_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("ENHANCED UNIFIED PIPELINE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Project: {self.results['project_root']}\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Files Analyzed: {self.results['files_analyzed']}\n")
            f.write(f"Total Issues: {self.results['total_issues']}\n")
            f.write(f"Execution Time: {self.results['execution_time']:.2f} seconds\n\n")
            
            summary = self.results['summary']
            f.write("Issues by Severity:\n")
            for severity, count in summary['issues_by_severity'].items():
                f.write(f"  {severity.title()}: {count}\n")
            
            f.write("\nIssues by Type:\n")
            for issue_type, count in summary['issues_by_type'].items():
                f.write(f"  {issue_type.replace('_', ' ').title()}: {count}\n")
        
        print(f"📄 JSON report: {json_file}")
        print(f"📄 Summary report: {summary_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Unified Code Quality Pipeline")
    parser.add_argument("--project-root", required=True, help="Root directory of the project to analyze")
    parser.add_argument("--output-dir", help="Output directory for reports")
    
    args = parser.parse_args()
    
    pipeline = UnifiedEnhancedPipeline(args.project_root, args.output_dir)
    results = pipeline.run_analysis()
    
    return results


if __name__ == "__main__":
    main()