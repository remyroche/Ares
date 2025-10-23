#!/usr/bin/env python3
"""
Import-Free Analysis Pipeline

This pipeline performs code analysis without importing external dependencies,
making it easier to troubleshoot and run in isolated environments. It focuses on:
- Static code analysis without imports
- AST-based pattern detection
- Built-in Python functionality analysis
- Code structure and style analysis
- Basic complexity metrics

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and build AST structures
3. ANALYSIS - Perform import-free analysis
4. PROCESSING - Categorize and prioritize findings
5. AGGREGATION - Combine results across files
6. REPORTING - Generate comprehensive reports
7. CLEANUP - Clean up temporary structures
"""

import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult


class ImportFreeAnalysisPipeline(BasePipeline):
    """Pipeline for import-free code analysis and troubleshooting."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the import-free analysis pipeline."""
        super().__init__(config, "import_free_analysis")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.analysis_results: Dict[Path, Dict[str, Any]] = {}
        self.pattern_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.structure_issues: Dict[Path, List[Dict[str, Any]]] = {}
        self.complexity_metrics: Dict[Path, Dict[str, Any]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for import-free analysis pipeline."""
        return [
            PipelineStage.INITIALIZATION,
            PipelineStage.PREPARATION,
            PipelineStage.ANALYSIS,
            PipelineStage.PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.CLEANUP
        ]
    
    async def execute_stage(self, stage: PipelineStage, context: Dict[str, Any]) -> StageResult:
        """Execute a specific pipeline stage."""
        stage_result = StageResult(
            stage=stage,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )
        
        try:
            if stage == PipelineStage.INITIALIZATION:
                await self._execute_initialization(stage_result, context)
            elif stage == PipelineStage.PREPARATION:
                await self._execute_preparation(stage_result, context)
            elif stage == PipelineStage.ANALYSIS:
                await self._execute_analysis(stage_result, context)
            elif stage == PipelineStage.PROCESSING:
                await self._execute_processing(stage_result, context)
            elif stage == PipelineStage.AGGREGATION:
                await self._execute_aggregation(stage_result, context)
            elif stage == PipelineStage.REPORTING:
                await self._execute_reporting(stage_result, context)
            elif stage == PipelineStage.CLEANUP:
                await self._execute_cleanup(stage_result, context)
            
            return stage_result
            
        except Exception as e:
            stage_result.fail([f"Stage {stage.value} failed: {e}"])
            return stage_result
    
    async def _execute_initialization(self, stage_result: StageResult, context: Dict[str, Any]):
        """Initialize the pipeline and discover Python files."""
        self.logger.info("Initializing import-free analysis pipeline...")
        
        # Discover Python files
        self.python_files = list(self.config.project_root.rglob("*.py"))
        
        # Filter out common directories to ignore
        ignore_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"}
        self.python_files = [
            f for f in self.python_files 
            if not any(part in ignore_dirs for part in f.parts)
        ]
        
        stage_result.complete({
            "files_discovered": len(self.python_files),
            "project_root": str(self.config.project_root),
            "files": [str(f) for f in self.python_files]
        })
        
        self.logger.info(f"Discovered {len(self.python_files)} Python files")
    
    async def _execute_preparation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Parse Python files and build AST structures."""
        self.logger.info("Preparing files for import-free analysis...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                successfully_parsed += 1
                
            except SyntaxError as e:
                parse_errors.append({
                    "file": str(file_path),
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "text": e.text
                })
            except Exception as e:
                parse_errors.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        stage_result.complete({
            "files_parsed": successfully_parsed,
            "parse_errors": parse_errors,
            "total_files": len(self.python_files)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        if parse_errors:
            self.logger.warning(f"Found {len(parse_errors)} parse errors")
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Perform comprehensive import-free analysis."""
        self.logger.info("Performing import-free code analysis...")
        
        analysis_results = {
            "pattern_issues": 0,
            "structure_issues": 0,
            "complexity_issues": 0,
            "files_analyzed": 0
        }
        
        for file_path, tree in self.parsed_files.items():
            # Analyze code patterns
            pattern_issues = self._analyze_code_patterns(file_path, tree)
            self.pattern_issues[file_path] = pattern_issues
            analysis_results["pattern_issues"] += len(pattern_issues)
            
            # Analyze code structure
            structure_issues = self._analyze_code_structure(file_path, tree)
            self.structure_issues[file_path] = structure_issues
            analysis_results["structure_issues"] += len(structure_issues)
            
            # Calculate complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(file_path, tree)
            self.complexity_metrics[file_path] = complexity_metrics
            
            # Store combined analysis results
            self.analysis_results[file_path] = {
                "pattern_issues": pattern_issues,
                "structure_issues": structure_issues,
                "complexity_metrics": complexity_metrics,
                "file_size": file_path.stat().st_size,
                "line_count": len(open(file_path, 'r').readlines())
            }
            
            analysis_results["files_analyzed"] += 1
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "files_analyzed": len(self.parsed_files)
        })
        
        total_issues = analysis_results["pattern_issues"] + analysis_results["structure_issues"]
        self.logger.info(f"Analysis complete: {total_issues} issues found across {analysis_results['files_analyzed']} files")
    
    def _analyze_code_patterns(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze code patterns and anti-patterns."""
        issues = []
        
        class PatternVisitor(ast.NodeVisitor):
            def visit_Compare(self, node):
                # Check for dangerous comparisons
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    if any(isinstance(op, ast.Is) or isinstance(op, ast.IsNot) for op in node.ops):
                        issues.append({
                            "type": "dangerous_string_comparison",
                            "line": node.lineno,
                            "message": "Using 'is' or 'is not' with string literals is dangerous",
                            "severity": "high"
                        })
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Check for multiple assignments
                if len(node.targets) > 1:
                    issues.append({
                        "type": "multiple_assignment",
                        "line": node.lineno,
                        "message": f"Multiple assignment to {len(node.targets)} targets",
                        "severity": "medium"
                    })
                self.generic_visit(node)
            
            def visit_For(self, node):
                # Check for range(len()) pattern
                if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                    if node.iter.func.id == "range" and len(node.iter.args) == 1:
                        if isinstance(node.iter.args[0], ast.Call) and isinstance(node.iter.args[0].func, ast.Name):
                            if node.iter.args[0].func.id == "len":
                                issues.append({
                                    "type": "range_len_pattern",
                                    "line": node.lineno,
                                    "message": "Consider using enumerate() instead of range(len())",
                                    "severity": "low"
                                })
                self.generic_visit(node)
            
            def visit_ListComp(self, node):
                # Check for complex list comprehensions
                if len(node.generators) > 1:
                    issues.append({
                        "type": "complex_listcomp",
                        "line": node.lineno,
                        "message": "Complex list comprehension with multiple generators",
                        "severity": "medium"
                    })
                self.generic_visit(node)
            
            def visit_DictComp(self, node):
                # Check for complex dict comprehensions
                if len(node.generators) > 1:
                    issues.append({
                        "type": "complex_dictcomp",
                        "line": node.lineno,
                        "message": "Complex dict comprehension with multiple generators",
                        "severity": "medium"
                    })
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                # Check for bare except
                if node.type is None:
                    issues.append({
                        "type": "bare_except",
                        "line": node.lineno,
                        "message": "Bare except clause - should specify exception type",
                        "severity": "high"
                    })
                # Check for broad except
                elif isinstance(node.type, ast.Name) and node.type.id in ["Exception", "BaseException"]:
                    issues.append({
                        "type": "broad_except",
                        "line": node.lineno,
                        "message": f"Broad except clause catching {node.type.id}",
                        "severity": "medium"
                    })
                self.generic_visit(node)
        
        visitor = PatternVisitor()
        visitor.visit(tree)
        return issues
    
    def _analyze_code_structure(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze code structure and organization."""
        issues = []
        
        class StructureVisitor(ast.NodeVisitor):
            def __init__(self):
                self.function_count = 0
                self.class_count = 0
                self.max_function_length = 0
                self.max_class_length = 0
                self.current_function_length = 0
                self.current_class_length = 0
                self.nested_depth = 0
                self.max_nested_depth = 0
            
            def visit_FunctionDef(self, node):
                self.function_count += 1
                self.current_function_length = 0
                self.nested_depth += 1
                self.max_nested_depth = max(self.max_nested_depth, self.nested_depth)
                
                # Check function length
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        self.current_function_length += 1
                
                if self.current_function_length > 50:
                    issues.append({
                        "type": "long_function",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' is {self.current_function_length} lines long",
                        "severity": "medium"
                    })
                
                # Check parameter count
                if len(node.args.args) > 7:
                    issues.append({
                        "type": "too_many_parameters",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' has {len(node.args.args)} parameters",
                        "severity": "medium"
                    })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    issues.append({
                        "type": "missing_docstring",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' missing docstring",
                        "severity": "low"
                    })
                
                self.generic_visit(node)
                self.nested_depth -= 1
                self.max_function_length = max(self.max_function_length, self.current_function_length)
            
            def visit_ClassDef(self, node):
                self.class_count += 1
                self.current_class_length = 0
                
                # Check class length
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        self.current_class_length += 1
                
                if self.current_class_length > 200:
                    issues.append({
                        "type": "long_class",
                        "line": node.lineno,
                        "message": f"Class '{node.name}' is {self.current_class_length} lines long",
                        "severity": "medium"
                    })
                
                # Check for missing docstring
                if not ast.get_docstring(node):
                    issues.append({
                        "type": "missing_class_docstring",
                        "line": node.lineno,
                        "message": f"Class '{node.name}' missing docstring",
                        "severity": "low"
                    })
                
                self.generic_visit(node)
                self.max_class_length = max(self.max_class_length, self.current_class_length)
        
        visitor = StructureVisitor()
        visitor.visit(tree)
        
        # Add file-level structure issues
        if visitor.function_count > 20:
            issues.append({
                "type": "too_many_functions",
                "line": 1,
                "message": f"File has {visitor.function_count} functions - consider splitting",
                "severity": "medium"
            })
        
        if visitor.class_count > 10:
            issues.append({
                "type": "too_many_classes",
                "line": 1,
                "message": f"File has {visitor.class_count} classes - consider splitting",
                "severity": "medium"
            })
        
        if visitor.max_nested_depth > 4:
            issues.append({
                "type": "deep_nesting",
                "line": 1,
                "message": f"Maximum nesting depth is {visitor.max_nested_depth}",
                "severity": "medium"
            })
        
        return issues
    
    def _calculate_complexity_metrics(self, file_path: Path, tree: ast.AST) -> Dict[str, Any]:
        """Calculate basic complexity metrics."""
        metrics = {
            "cyclomatic_complexity": 0,
            "function_count": 0,
            "class_count": 0,
            "line_count": 0,
            "statement_count": 0,
            "expression_count": 0
        }
        
        class ComplexityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                metrics["function_count"] += 1
                # Simple cyclomatic complexity calculation
                complexity = 1  # Base complexity
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                metrics["cyclomatic_complexity"] = max(metrics["cyclomatic_complexity"], complexity)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                metrics["class_count"] += 1
                self.generic_visit(node)
            
            def visit_Expr(self, node):
                metrics["expression_count"] += 1
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                metrics["statement_count"] += 1
                self.generic_visit(node)
            
            def visit_Return(self, node):
                metrics["statement_count"] += 1
                self.generic_visit(node)
            
            def visit_If(self, node):
                metrics["statement_count"] += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                metrics["statement_count"] += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                metrics["statement_count"] += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Calculate line count
        try:
            with open(file_path, 'r') as f:
                metrics["line_count"] = len(f.readlines())
        except Exception:
            metrics["line_count"] = 0
        
        return metrics
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process and categorize all findings."""
        self.logger.info("Processing and categorizing findings...")
        
        # Categorize issues by severity
        issue_categories = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Process pattern issues
        for file_path, issues in self.pattern_issues.items():
            for issue in issues:
                severity = issue.get("severity", "medium")
                issue_categories[severity].append({
                    "file": str(file_path),
                    "type": issue["type"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "severity": severity
                })
        
        # Process structure issues
        for file_path, issues in self.structure_issues.items():
            for issue in issues:
                severity = issue.get("severity", "medium")
                issue_categories[severity].append({
                    "file": str(file_path),
                    "type": issue["type"],
                    "line": issue["line"],
                    "message": issue["message"],
                    "severity": severity
                })
        
        # Identify complexity issues
        complexity_issues = []
        for file_path, metrics in self.complexity_metrics.items():
            if metrics["cyclomatic_complexity"] > 10:
                complexity_issues.append({
                    "file": str(file_path),
                    "type": "high_complexity",
                    "line": 1,
                    "message": f"High cyclomatic complexity: {metrics['cyclomatic_complexity']}",
                    "severity": "high"
                })
            
            if metrics["function_count"] > 20:
                complexity_issues.append({
                    "file": str(file_path),
                    "type": "too_many_functions",
                    "line": 1,
                    "message": f"Too many functions: {metrics['function_count']}",
                    "severity": "medium"
                })
        
        issue_categories["high"].extend(complexity_issues)
        
        stage_result.complete({
            "issue_categories": issue_categories,
            "total_issues": sum(len(issues) for issues in issue_categories.values())
        })
        
        total_issues = sum(len(issues) for issues in issue_categories.values())
        self.logger.info(f"Processed {total_issues} issues across all categories")
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating import-free analysis results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "parsed_files": len(self.parsed_files),
            "total_issues": 0,
            "issues_by_severity": {},
            "complexity_stats": {
                "total_functions": 0,
                "total_classes": 0,
                "avg_complexity": 0,
                "max_complexity": 0
            },
            "file_stats": {
                "total_lines": 0,
                "total_statements": 0,
                "avg_file_size": 0
            }
        }
        
        # Aggregate issues by severity
        for severity, issues in context.get("issue_categories", {}).items():
            summary["issues_by_severity"][severity] = len(issues)
            summary["total_issues"] += len(issues)
        
        # Aggregate complexity metrics
        total_complexity = 0
        max_complexity = 0
        total_lines = 0
        total_statements = 0
        
        for metrics in self.complexity_metrics.values():
            summary["complexity_stats"]["total_functions"] += metrics["function_count"]
            summary["complexity_stats"]["total_classes"] += metrics["class_count"]
            total_complexity += metrics["cyclomatic_complexity"]
            max_complexity = max(max_complexity, metrics["cyclomatic_complexity"])
            total_lines += metrics["line_count"]
            total_statements += metrics["statement_count"]
        
        if self.complexity_metrics:
            summary["complexity_stats"]["avg_complexity"] = total_complexity / len(self.complexity_metrics)
        summary["complexity_stats"]["max_complexity"] = max_complexity
        
        summary["file_stats"]["total_lines"] = total_lines
        summary["file_stats"]["total_statements"] = total_statements
        if self.parsed_files:
            summary["file_stats"]["avg_file_size"] = total_lines / len(self.parsed_files)
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "analysis_results": self.analysis_results,
                "pattern_issues": self.pattern_issues,
                "structure_issues": self.structure_issues,
                "complexity_metrics": self.complexity_metrics
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_issues']} total issues, "
                        f"{summary['complexity_stats']['total_functions']} functions, "
                        f"{summary['complexity_stats']['total_classes']} classes")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive import-free analysis reports."""
        self.logger.info("Generating import-free analysis reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate complexity report
        complexity_report = self._generate_complexity_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"import_free_analysis_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"import_free_analysis_detailed_{timestamp}.json"
        complexity_path = self.config.output_dir / f"import_free_analysis_complexity_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(complexity_path, 'w') as f:
            json.dump(complexity_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "complexity": str(complexity_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {complexity_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "import_free_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "import_free_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_complexity_report(self) -> Dict[str, Any]:
        """Generate complexity report."""
        return {
            "pipeline": "import_free_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "complexity_metrics": self.complexity_metrics
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("issues_by_severity", {}).get("critical", 0) > 0:
            recommendations.append("Address critical issues first (dangerous patterns, bare excepts)")
        
        if summary.get("issues_by_severity", {}).get("high", 0) > 0:
            recommendations.append("Fix high-priority issues (complexity, structure problems)")
        
        if summary.get("complexity_stats", {}).get("max_complexity", 0) > 15:
            recommendations.append("Refactor functions with high cyclomatic complexity")
        
        if summary.get("complexity_stats", {}).get("total_functions", 0) > 100:
            recommendations.append("Consider splitting large files with many functions")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.analysis_results.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_import_free_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run import-free analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = ImportFreeAnalysisPipeline(config)
    return await pipeline.run()