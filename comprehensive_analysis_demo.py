#!/usr/bin/env python3
"""
Comprehensive Professional Analysis Demo
This demonstrates the concept working with basic Python tools
"""

import os
import sys
import json
import time
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import argparse

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_analysis_demo.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ComprehensiveAnalysisDemo")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    file_path: str
    directory: str
    analyzer_name: str
    category: str
    issues_found: int
    issues_fixed: int
    details: Dict[str, Any]
    processing_time: float
    status: str


@dataclass
class DirectorySummary:
    """Summary of analysis for a directory."""
    directory: str
    total_files: int
    files_analyzed: int
    total_issues: int
    total_fixed: int
    analyzers_run: List[str]
    categories_covered: List[str]
    processing_time: float


@dataclass
class GlobalMetrics:
    """Global metrics across all analysis."""
    total_directories: int
    total_files: int
    total_analyzers_run: int
    total_issues_found: int
    total_issues_fixed: int
    total_processing_time: float
    success_rate: float
    categories_covered: List[str]
    top_issues: List[tuple]


class BasicSyntaxAnalyzer:
    """Basic syntax analyzer using Python's built-in ast module."""
    
    def __init__(self):
        self.name = "basic_syntax"
        self.category = "syntax"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Basic analysis
            issues = []
            
            # Check for syntax issues
            try:
                compile(content, file_path, 'exec')
            except SyntaxError as e:
                issues.append(f"Syntax error at line {e.lineno}: {e.msg}")
            
            # Count nodes
            node_count = len(ast.walk(tree))
            
            # Check for common patterns
            function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            class_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            import_count = len([n for n in ast.walk(tree) if isinstance(n, ast.Import)])
            import_from_count = len([n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)])
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": len(issues),
                "issues_fixed": 0,
                "details": {
                    "syntax_errors": issues,
                    "node_count": node_count,
                    "function_count": function_count,
                    "class_count": class_count,
                    "import_count": import_count + import_from_count,
                    "ast_valid": True
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }


class AdvancedComplexityAnalyzer:
    """Advanced complexity and code quality analyzer."""
    
    def __init__(self):
        self.name = "advanced_complexity"
        self.category = "complexity"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Calculate advanced complexity metrics
            complexity_score = 0
            function_complexities = []
            function_details = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity_score += 1
                elif isinstance(node, ast.FunctionDef):
                    func_complexity = 1
                    nested_structures = 0
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                            func_complexity += 1
                            nested_structures += 1
                    
                    function_complexities.append(func_complexity)
                    function_details.append({
                        "name": node.name,
                        "complexity": func_complexity,
                        "nested_structures": nested_structures,
                        "line": node.lineno,
                        "has_docstring": ast.get_docstring(node) is not None
                    })
            
            # Calculate metrics
            avg_complexity = sum(function_complexities) / len(function_complexities) if function_complexities else 0
            high_complexity_functions = len([c for c in function_complexities if c > 10])
            medium_complexity_functions = len([c for c in function_complexities if 5 < c <= 10])
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": high_complexity_functions + medium_complexity_functions,
                "issues_fixed": 0,
                "details": {
                    "overall_complexity": complexity_score,
                    "function_complexities": function_complexities,
                    "function_details": function_details,
                    "average_function_complexity": avg_complexity,
                    "high_complexity_functions": high_complexity_functions,
                    "medium_complexity_functions": medium_complexity_functions,
                    "total_functions": len(function_complexities)
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }


class DeadCodeAnalyzer:
    """Dead code detection analyzer."""
    
    def __init__(self):
        self.name = "dead_code"
        self.category = "dead_code"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Detect dead code patterns
            dead_code_issues = []
            unused_imports = []
            unused_variables = []
            unreachable_code = []
            
            # Check for unused imports
            import_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_nodes.append(node)
            
            # Check for unused variables and functions
            defined_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_names.add(target.id)
                elif isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Load):
                        used_names.add(node.id)
            
            # Find unused definitions
            unused_definitions = defined_names - used_names
            for name in unused_definitions:
                if not name.startswith('_'):  # Skip intentionally unused names
                    unused_variables.append(f"Unused definition: {name}")
            
            # Check for unreachable code (code after return/raise/break/continue)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    unreachable_lines = self._find_unreachable_code(node)
                    if unreachable_lines:
                        unreachable_code.extend(unreachable_lines)
            
            # Combine all dead code issues
            dead_code_issues = unused_variables + unreachable_code
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": len(dead_code_issues),
                "issues_fixed": 0,
                "details": {
                    "dead_code_issues": dead_code_issues,
                    "unused_variables": unused_variables,
                    "unreachable_code": unreachable_code,
                    "total_definitions": len(defined_names),
                    "total_used": len(used_names),
                    "unused_count": len(unused_definitions)
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }
    
    def _find_unreachable_code(self, func_node: ast.FunctionDef) -> List[str]:
        """Find unreachable code in a function."""
        unreachable = []
        
        for i, stmt in enumerate(func_node.body):
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                # Check if there are statements after this
                if i + 1 < len(func_node.body):
                    unreachable.append(f"Unreachable code after {type(stmt).__name__} at line {stmt.lineno}")
        
        return unreachable


class FunctionCallQualityAnalyzer:
    """Analyzes function call quality and interactions."""
    
    def __init__(self):
        self.name = "function_call_quality"
        self.category = "function_calls"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze function calls and interactions
            function_calls = []
            call_graph = {}
            function_interactions = []
            quality_issues = []
            
            # Build call graph and analyze function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Extract function name
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        function_calls.append(func_name)
                        
                        # Find which function this call is in
                        current_function = self._find_parent_function(node, tree)
                        if current_function:
                            if current_function not in call_graph:
                                call_graph[current_function] = []
                            call_graph[current_function].append(func_name)
            
            # Analyze function call patterns
            call_frequency = Counter(function_calls)
            high_frequency_calls = {name: count for name, count in call_frequency.items() if count > 5}
            
            # Check for potential issues
            for func_name, count in high_frequency_calls.items():
                if count > 10:
                    quality_issues.append(f"High frequency function call: {func_name} called {count} times")
            
            # Analyze function interactions
            for caller, callees in call_graph.items():
                if len(callees) > 8:
                    quality_issues.append(f"Function {caller} has many dependencies ({len(callees)} calls)")
                
                # Check for circular dependencies (simplified)
                if caller in callees:
                    quality_issues.append(f"Potential self-reference in {caller}")
            
            # Calculate metrics
            total_calls = len(function_calls)
            unique_calls = len(set(function_calls))
            avg_calls_per_function = total_calls / len(call_graph) if call_graph else 0
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": len(quality_issues),
                "issues_fixed": 0,
                "details": {
                    "quality_issues": quality_issues,
                    "call_graph": call_graph,
                    "function_calls": function_calls,
                    "call_frequency": dict(call_frequency),
                    "high_frequency_calls": high_frequency_calls,
                    "total_calls": total_calls,
                    "unique_calls": unique_calls,
                    "avg_calls_per_function": avg_calls_per_function,
                    "functions_with_many_calls": len([f for f in call_graph.values() if len(f) > 8])
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }
    
    def _find_parent_function(self, node: ast.Call, tree: ast.AST) -> Optional[str]:
        """Find the parent function of a call node."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.FunctionDef):
                for child in ast.walk(parent):
                    if child is node:
                        return parent.name
        return None


class DependencyAnalyzer:
    """Analyzes code dependencies and import patterns."""
    
    def __init__(self):
        self.name = "dependency_analyzer"
        self.category = "dependencies"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze dependencies and imports
            imports = []
            import_issues = []
            dependency_metrics = {}
            
            # Extract all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        if alias.name == "*":
                            imports.append(f"{module}.*")
                        else:
                            imports.append(f"{module}.{alias.name}")
            
            # Analyze import patterns
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports:
                if imp.startswith('__'):
                    continue
                elif imp.startswith('.') or imp.startswith('src.') or imp.startswith('test_'):
                    local_imports.append(imp)
                elif imp in ['os', 'sys', 'json', 'time', 'pathlib', 'typing', 'collections', 'dataclasses', 'argparse', 'logging', 'ast', 'inspect', 'datetime', 'math']:
                    stdlib_imports.append(imp)
                else:
                    third_party_imports.append(imp)
            
            # Check for potential issues
            if len(imports) > 20:
                import_issues.append(f"Many imports ({len(imports)}) - consider consolidating")
            
            if len(third_party_imports) > 10:
                import_issues.append(f"Many third-party dependencies ({len(third_party_imports)}) - potential maintenance burden")
            
            # Check for unused imports (simplified)
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
            
            potentially_unused_imports = []
            for imp in imports:
                if '.' in imp:
                    base_name = imp.split('.')[-1]
                    if base_name not in used_names and base_name not in ['*']:
                        potentially_unused_imports.append(imp)
            
            if potentially_unused_imports:
                import_issues.append(f"Potentially unused imports: {', '.join(potentially_unused_imports[:5])}")
            
            # Calculate metrics
            total_imports = len(imports)
            unique_imports = len(set(imports))
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": len(import_issues),
                "issues_fixed": 0,
                "details": {
                    "import_issues": import_issues,
                    "all_imports": imports,
                    "stdlib_imports": stdlib_imports,
                    "third_party_imports": third_party_imports,
                    "local_imports": local_imports,
                    "potentially_unused_imports": potentially_unused_imports,
                    "total_imports": total_imports,
                    "unique_imports": unique_imports,
                    "import_diversity": len(set(imports)) / len(imports) if imports else 0
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }


class BasicStyleAnalyzer:
    """Basic code style analyzer - focuses on auto-fixable issues only."""
    
    def __init__(self):
        self.name = "basic_style"
        self.category = "style"
    
    def can_analyze(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            issues = []
            auto_fixable_issues = []
            
            # Only check for issues that can be auto-fixed
            for i, line in enumerate(lines, 1):
                # Check for trailing whitespace (auto-fixable)
                if line.rstrip() != line.rstrip('\n'):
                    auto_fixable_issues.append(f"Line {i}: Trailing whitespace")
            
            # Only include auto-fixable issues
            issues = auto_fixable_issues
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "issues_found": len(issues),
                "issues_fixed": 0,
                "details": {
                    "auto_fixable_style_issues": issues,
                    "line_count": len(lines),
                    "note": "Only auto-fixable style issues are reported"
                },
                "processing_time": processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "status": "error",
                "issues_found": 1,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "processing_time": processing_time
            }


class ComprehensiveAnalysisDemo:
    """Demo of comprehensive analysis using basic Python tools."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        
        # Initialize comprehensive analyzers - focusing on actionable, fixable issues
        self.analyzers = {
            "syntax": BasicSyntaxAnalyzer(),
            "complexity": AdvancedComplexityAnalyzer(),
            "dead_code": DeadCodeAnalyzer(),
            "function_calls": FunctionCallQualityAnalyzer(),
            "dependencies": DependencyAnalyzer()
        }
        
        # Results storage
        self.analysis_results: List[AnalysisResult] = []
        self.directory_summaries: Dict[str, DirectorySummary] = {}
        self.global_metrics = None
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run the comprehensive analysis demo."""
        logger.info("🚀 Starting comprehensive analysis demo...")
        start_time = time.time()
        
        # Find all Python files organized by directory
        python_files_by_dir = self._find_python_files_by_directory()
        logger.info(f"Found Python files in {len(python_files_by_dir)} directories")
        
        # Run all analyzers
        logger.info("🔍 Running analyzers...")
        self._run_analyzers(python_files_by_dir)
        
        # Generate summaries
        self._generate_directory_summaries(python_files_by_dir)
        self._generate_global_metrics(start_time)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Comprehensive analysis demo completed in {total_time:.2f} seconds")
        
        return report
    
    def _find_python_files_by_directory(self) -> Dict[str, List[Path]]:
        """Find all Python files organized by directory."""
        python_files_by_dir = defaultdict(list)
        
        exclude_patterns = [
            "__pycache__", ".git", "venv", "env", "node_modules", 
            ".pytest_cache", "code_quality_env", ".venv"
        ]
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_patterns]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    # Get relative directory from project root
                    rel_dir = str(file_path.parent.relative_to(self.project_root))
                    if rel_dir == ".":
                        rel_dir = "root"
                    python_files_by_dir[rel_dir].append(file_path)
        
        return dict(python_files_by_dir)
    
    def _run_analyzers(self, python_files_by_dir: Dict[str, List[Path]]):
        """Run all analyzers on Python files."""
        logger.info("🔍 Running analyzers...")
        
        for analyzer_name, analyzer in self.analyzers.items():
            logger.info(f"  Running {analyzer_name}...")
            
            for directory, files in python_files_by_dir.items():
                for file_path in files:
                    try:
                        start_time = time.time()
                        
                        # Check if analyzer can handle this file
                        if not analyzer.can_analyze(str(file_path)):
                            continue
                        
                        # Run the analyzer
                        result = analyzer.analyze(str(file_path))
                        
                        processing_time = time.time() - start_time
                        
                        # Record result
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=analyzer_name,
                            category=analyzer.category,
                            issues_found=result.get("issues_found", 0),
                            issues_fixed=result.get("issues_fixed", 0),
                            details=result,
                            processing_time=processing_time,
                            status="success"
                        )
                        self.analysis_results.append(analysis_result)
                        
                    except Exception as e:
                        logger.error(f"Error running {analyzer_name} on {file_path}: {e}")
                        analysis_result = AnalysisResult(
                            file_path=str(file_path),
                            directory=directory,
                            analyzer_name=analyzer_name,
                            category=analyzer.category,
                            issues_found=0,
                            issues_fixed=0,
                            details={"error": str(e)},
                            processing_time=0,
                            status="error"
                        )
                        self.analysis_results.append(analysis_result)
    
    def _generate_directory_summaries(self, python_files_by_dir: Dict[str, List[Path]]):
        """Generate summaries for each directory."""
        logger.info("📊 Generating directory summaries...")
        
        for directory, files in python_files_by_dir.items():
            # Get results for this directory
            dir_results = [r for r in self.analysis_results if r.directory == directory]
            
            # Calculate metrics
            total_files = len(files)
            files_analyzed = len(set(r.file_path for r in dir_results))
            total_issues = sum(r.issues_found for r in dir_results)
            total_fixed = sum(r.issues_fixed for r in dir_results)
            analyzers_run = list(set(r.analyzer_name for r in dir_results))
            categories_covered = list(set(r.category for r in dir_results))
            processing_time = sum(r.processing_time for r in dir_results)
            
            # Create directory summary
            dir_summary = DirectorySummary(
                directory=directory,
                total_files=total_files,
                files_analyzed=files_analyzed,
                total_issues=total_issues,
                total_fixed=total_fixed,
                analyzers_run=analyzers_run,
                categories_covered=categories_covered,
                processing_time=processing_time
            )
            
            self.directory_summaries[directory] = dir_summary
    
    def _generate_global_metrics(self, start_time: float):
        """Generate global metrics across all analysis."""
        logger.info("🌍 Generating global metrics...")
        
        # Calculate totals
        total_directories = len(self.directory_summaries)
        total_files = sum(s.total_files for s in self.directory_summaries.values())
        total_analyzers_run = len(set(r.analyzer_name for r in self.analysis_results))
        total_issues_found = sum(r.issues_found for r in self.analysis_results)
        total_issues_fixed = sum(r.issues_fixed for r in self.analysis_results)
        total_processing_time = time.time() - start_time
        
        # Calculate success rate
        successful_runs = len([r for r in self.analysis_results if r.status == "success"])
        total_runs = len(self.analysis_results)
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        # Get categories covered
        categories_covered = list(set(r.category for r in self.analysis_results))
        
        # Get top issues by category
        issues_by_category = Counter(r.category for r in self.analysis_results if r.issues_found > 0)
        top_issues = issues_by_category.most_common(10)
        
        self.global_metrics = GlobalMetrics(
            total_directories=total_directories,
            total_files=total_files,
            total_analyzers_run=total_analyzers_run,
            total_issues_found=total_issues_found,
            total_issues_fixed=total_issues_fixed,
            total_processing_time=total_processing_time,
            success_rate=success_rate,
            categories_covered=categories_covered,
            top_issues=top_issues
        )
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of all analysis results."""
        logger.info("📋 Generating comprehensive report...")
        
        # Convert dataclasses to dictionaries
        analysis_results_dict = [asdict(r) for r in self.analysis_results]
        directory_summaries_dict = {k: asdict(v) for k, v in self.directory_summaries.items()}
        global_metrics_dict = asdict(self.global_metrics) if self.global_metrics else {}
        
        # Group results by directory and category
        results_by_directory = defaultdict(lambda: defaultdict(list))
        for result in self.analysis_results:
            results_by_directory[result.directory][result.category].append(result)
        
        # Create the comprehensive report
        report = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "project_root": str(self.project_root),
                "analysis_duration": self.global_metrics.total_processing_time if self.global_metrics else 0,
                "note": "This is a DEMO using basic Python tools. Install the full toolkit for professional analysis."
            },
            "global_metrics": global_metrics_dict,
            "directory_summaries": directory_summaries_dict,
            "detailed_results": {
                "by_directory": {
                    directory: {
                        category: [asdict(r) for r in results]
                        for category, results in categories.items()
                    }
                    for directory, categories in results_by_directory.items()
                },
                "by_category": {
                    category: [asdict(r) for r in results]
                    for category, results in self._group_results_by_category().items()
                },
                "by_analyzer": {
                    analyzer: [asdict(r) for r in results]
                    for analyzer, results in self._group_results_by_analyzer().items()
                }
            },
            "summary": {
                "total_analysis_runs": len(self.analysis_results),
                "successful_runs": len([r for r in self.analysis_results if r.status == "success"]),
                "failed_runs": len([r for r in self.analysis_results if r.status == "error"]),
                "categories_analyzed": len(set(r.category for r in self.analysis_results)),
                "analyzers_used": len(set(r.analyzer_name for r in self.analysis_results))
            }
        }
        
        return report
    
    def _group_results_by_category(self) -> Dict[str, List[AnalysisResult]]:
        """Group analysis results by category."""
        grouped = defaultdict(list)
        for result in self.analysis_results:
            grouped[result.category].append(result)
        return dict(grouped)
    
    def _group_results_by_analyzer(self) -> Dict[str, List[AnalysisResult]]:
        """Group analysis results by analyzer."""
        grouped = defaultdict(list)
        for result in self.analysis_results:
            grouped[result.analyzer_name].append(result)
        return dict(grouped)
    
    def save_report(self, report: Dict[str, Any], output_file: str = None) -> str:
        """Save the comprehensive report to a file."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_analysis_demo_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"📄 Comprehensive report saved to: {output_path}")
        return str(output_path)
    
    def generate_text_summary(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text summary."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE PROFESSIONAL ANALYSIS DEMO REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Generated: {report['metadata']['generated_at']}")
        lines.append(f"Project Root: {report['metadata']['project_root']}")
        lines.append(f"Analysis Duration: {report['metadata']['analysis_duration']:.2f} seconds")
        lines.append(f"Note: {report['metadata']['note']}")
        lines.append("")
        
        # Global Metrics
        global_metrics = report['global_metrics']
        lines.append("🌍 GLOBAL METRICS")
        lines.append("-" * 50)
        lines.append(f"Total Directories: {global_metrics['total_directories']}")
        lines.append(f"Total Files: {global_metrics['total_files']}")
        lines.append(f"Total Analyzers Run: {global_metrics['total_analyzers_run']}")
        lines.append(f"Total Issues Found: {global_metrics['total_issues_found']}")
        lines.append(f"Total Issues Fixed: {global_metrics['total_issues_fixed']}")
        lines.append(f"Success Rate: {global_metrics['success_rate']:.1f}%")
        lines.append(f"Categories Covered: {', '.join(global_metrics['categories_covered'])}")
        lines.append("")
        
        # Top Issues
        if global_metrics['top_issues']:
            lines.append("🚨 TOP ISSUES BY CATEGORY")
            lines.append("-" * 50)
            for category, count in global_metrics['top_issues']:
                lines.append(f"• {category}: {count} issues")
            lines.append("")
        
        # Directory Summaries
        lines.append("📁 DIRECTORY ANALYSIS SUMMARIES")
        lines.append("=" * 80)
        lines.append("")
        
        for directory, summary in report['directory_summaries'].items():
            lines.append(f"📂 {directory}/")
            lines.append(f"   Files: {summary['total_files']} (analyzed: {summary['files_analyzed']})")
            lines.append(f"   Issues: {summary['total_issues']} (fixed: {summary['total_fixed']})")
            lines.append(f"   Analyzers: {len(summary['analyzers_run'])}")
            lines.append(f"   Categories: {', '.join(summary['categories_covered'])}")
            lines.append(f"   Processing Time: {summary['processing_time']:.2f}s")
            lines.append("")
        
        # Category Analysis
        lines.append("🔍 ANALYSIS BY CATEGORY")
        lines.append("=" * 80)
        lines.append("")
        
        for category, results in report['detailed_results']['by_category'].items():
            total_issues = sum(r['issues_found'] for r in results)
            total_fixed = sum(r['issues_fixed'] for r in results)
            files_analyzed = len(set(r['file_path'] for r in results))
            
            lines.append(f"📊 {category.upper()}")
            lines.append(f"   Files Analyzed: {files_analyzed}")
            lines.append(f"   Issues Found: {total_issues}")
            lines.append(f"   Issues Fixed: {total_fixed}")
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        lines.append("END OF COMPREHENSIVE PROFESSIONAL ANALYSIS DEMO REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append("💡 This is a DEMO showing the comprehensive analysis concept.")
        lines.append("   To run the full professional analysis, install dependencies and run:")
        lines.append("   python comprehensive_professional_analysis.py")
        
        return "\n".join(lines)


def main():
    """Main function to run the comprehensive analysis demo."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Professional Analysis Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo analysis on current directory
  python comprehensive_analysis_demo.py
  
  # Run demo analysis on specific directory
  python comprehensive_analysis_demo.py --project-root test_analysis_dir
  
  # Custom output file
  python comprehensive_analysis_demo.py --output my_demo_analysis.json
  
  # Verbose logging
  python comprehensive_analysis_demo.py --verbose
        """
    )
    
    parser.add_argument("--project-root", default=".", 
                       help="Project root directory to analyze (default: current)")
    parser.add_argument("--output", help="Output file for the JSON report")
    parser.add_argument("--text-summary", help="Output file for text summary")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveAnalysisDemo(args.project_root)
        
        # Run comprehensive analysis
        report = analyzer.run_comprehensive_analysis()
        
        # Save JSON report
        json_file = analyzer.save_report(report, args.output)
        
        # Generate and save text summary
        text_summary = analyzer.generate_text_summary(report)
        if args.text_summary:
            text_file = args.text_summary
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            text_file = f"comprehensive_analysis_demo_{timestamp}.txt"
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_summary)
        
        print(f"\n📄 Text summary saved to: {text_file}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE PROFESSIONAL ANALYSIS DEMO COMPLETE")
        print("="*80)
        
        global_metrics = report['global_metrics']
        print(f"🌍 Total Files: {global_metrics['total_files']}")
        print(f"🔍 Total Issues Found: {global_metrics['total_issues_found']}")
        print(f"🔧 Total Issues Fixed: {global_metrics['total_issues_fixed']}")
        print(f"✅ Success Rate: {global_metrics['success_rate']:.1f}%")
        print(f"📁 Directories Analyzed: {global_metrics['total_directories']}")
        print(f"⚡ Total Processing Time: {global_metrics['total_processing_time']:.2f}s")
        print("="*80)
        
        print("\n💡 This is a DEMO using basic Python tools.")
        print("   The full professional analysis includes:")
        print("   - Advanced AST analysis with astroid")
        print("   - Type checking with mypy")
        print("   - Security analysis with bandit")
        print("   - Code formatting with black/isort")
        print("   - Advanced linting with flake8/pylint")
        print("   - Complexity analysis with radon/mccabe")
        print("   - Dead code detection with vulture")
        print("   - And much more...")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())