#!/usr/bin/env python3
"""
Complexity Analysis Pipeline

This pipeline focuses on comprehensive code complexity analysis including:
- Cyclomatic complexity calculation
- Cognitive complexity analysis
- Maintainability index calculation
- Code size metrics
- Nesting depth analysis
- Function/class complexity distribution

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and extract complexity data
3. ANALYSIS - Calculate complexity metrics
4. PROCESSING - Analyze complexity patterns and thresholds
5. AGGREGATION - Combine results and generate insights
6. REPORTING - Generate complexity reports and recommendations
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


class ComplexityAnalysisPipeline(BasePipeline):
    """Pipeline for comprehensive code complexity analysis."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the complexity analysis pipeline."""
        super().__init__(config, "complexity_analysis")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.complexity_metrics: Dict[Path, Dict[str, Any]] = {}
        self.function_complexity: Dict[Path, List[Dict[str, Any]]] = {}
        self.class_complexity: Dict[Path, List[Dict[str, Any]]] = {}
        self.file_complexity: Dict[Path, Dict[str, Any]] = {}
        self.complexity_distribution: Dict[str, int] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for complexity analysis pipeline."""
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
        self.logger.info("Initializing complexity analysis pipeline...")
        
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
        """Parse files and extract complexity data."""
        self.logger.info("Preparing files for complexity analysis...")
        
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
                    "message": e.msg
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
        """Calculate complexity metrics for all files."""
        self.logger.info("Calculating complexity metrics...")
        
        analysis_results = {
            "files_analyzed": 0,
            "functions_analyzed": 0,
            "classes_analyzed": 0,
            "total_complexity": 0
        }
        
        for file_path, tree in self.parsed_files.items():
            # Calculate file-level complexity
            file_metrics = self._calculate_file_complexity(file_path, tree)
            self.file_complexity[file_path] = file_metrics
            
            # Calculate function complexity
            function_metrics = self._calculate_function_complexity(file_path, tree)
            self.function_complexity[file_path] = function_metrics
            analysis_results["functions_analyzed"] += len(function_metrics)
            
            # Calculate class complexity
            class_metrics = self._calculate_class_complexity(file_path, tree)
            self.class_complexity[file_path] = class_metrics
            analysis_results["classes_analyzed"] += len(class_metrics)
            
            # Store combined metrics
            self.complexity_metrics[file_path] = {
                "file_metrics": file_metrics,
                "function_metrics": function_metrics,
                "class_metrics": class_metrics
            }
            
            analysis_results["files_analyzed"] += 1
            analysis_results["total_complexity"] += file_metrics["total_complexity"]
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "files_analyzed": len(self.parsed_files)
        })
        
        self.logger.info(f"Analysis complete: {analysis_results['functions_analyzed']} functions, "
                        f"{analysis_results['classes_analyzed']} classes analyzed")
    
    def _calculate_file_complexity(self, file_path: Path, tree: ast.AST) -> Dict[str, Any]:
        """Calculate file-level complexity metrics."""
        metrics = {
            "lines_of_code": 0,
            "logical_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "total_complexity": 0,
            "average_complexity": 0,
            "max_complexity": 0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "maintainability_index": 0
        }
        
        # Count lines
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                metrics["lines_of_code"] = len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        metrics["blank_lines"] += 1
                    elif stripped.startswith('#'):
                        metrics["comment_lines"] += 1
                    else:
                        metrics["logical_lines"] += 1
        except Exception:
            pass
        
        # Count functions and classes
        class ComplexityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                metrics["function_count"] += 1
                complexity = self._calculate_cyclomatic_complexity(node)
                metrics["total_complexity"] += complexity
                metrics["max_complexity"] = max(metrics["max_complexity"], complexity)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                metrics["class_count"] += 1
                self.generic_visit(node)
            
            def visit_Import(self, node):
                metrics["import_count"] += len(node.names)
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                metrics["import_count"] += len(node.names)
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        # Calculate average complexity
        if metrics["function_count"] > 0:
            metrics["average_complexity"] = metrics["total_complexity"] / metrics["function_count"]
        
        # Calculate maintainability index (simplified)
        metrics["maintainability_index"] = self._calculate_maintainability_index(metrics)
        
        return metrics
    
    def _calculate_function_complexity(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Calculate complexity for each function."""
        function_metrics = []
        
        class FunctionComplexityVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                complexity = self._calculate_cyclomatic_complexity(node)
                cognitive_complexity = self._calculate_cognitive_complexity(node)
                
                function_metrics.append({
                    "name": node.name,
                    "line": node.lineno,
                    "cyclomatic_complexity": complexity,
                    "cognitive_complexity": cognitive_complexity,
                    "lines_of_code": self._count_function_lines(node),
                    "parameter_count": len(node.args.args),
                    "nested_depth": self._calculate_nested_depth(node),
                    "maintainability_score": self._calculate_function_maintainability(node, complexity)
                })
                
                self.generic_visit(node)
        
        visitor = FunctionComplexityVisitor()
        visitor.visit(tree)
        
        return function_metrics
    
    def _calculate_class_complexity(self, file_path: Path, tree: ast.AST) -> List[Dict[str, Any]]:
        """Calculate complexity for each class."""
        class_metrics = []
        
        class ClassComplexityVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                total_complexity = 0
                max_method_complexity = 0
                
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(child)
                        total_complexity += complexity
                        max_method_complexity = max(max_method_complexity, complexity)
                
                class_metrics.append({
                    "name": node.name,
                    "line": node.lineno,
                    "method_count": method_count,
                    "total_complexity": total_complexity,
                    "average_complexity": total_complexity / method_count if method_count > 0 else 0,
                    "max_method_complexity": max_method_complexity,
                    "inheritance_depth": len(node.bases),
                    "lines_of_code": self._count_class_lines(node)
                })
                
                self.generic_visit(node)
        
        visitor = ClassComplexityVisitor()
        visitor.visit(tree)
        
        return class_metrics
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        class ComplexityVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                nonlocal complexity
                complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                nonlocal complexity
                complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                nonlocal complexity
                complexity += 1
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                nonlocal complexity
                complexity += 1
                self.generic_visit(node)
            
            def visit_BoolOp(self, node):
                nonlocal complexity
                complexity += len(node.values) - 1
                self.generic_visit(node)
            
            def visit_ListComp(self, node):
                nonlocal complexity
                complexity += len(node.generators)
                self.generic_visit(node)
            
            def visit_DictComp(self, node):
                nonlocal complexity
                complexity += len(node.generators)
                self.generic_visit(node)
            
            def visit_SetComp(self, node):
                nonlocal complexity
                complexity += len(node.generators)
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(node)
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        complexity = 0
        nesting_level = 0
        
        class CognitiveComplexityVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                nonlocal complexity, nesting_level
                complexity += 1 + nesting_level
                nesting_level += 1
                self.generic_visit(node)
                nesting_level -= 1
            
            def visit_For(self, node):
                nonlocal complexity, nesting_level
                complexity += 1 + nesting_level
                nesting_level += 1
                self.generic_visit(node)
                nesting_level -= 1
            
            def visit_While(self, node):
                nonlocal complexity, nesting_level
                complexity += 1 + nesting_level
                nesting_level += 1
                self.generic_visit(node)
                nesting_level -= 1
            
            def visit_ExceptHandler(self, node):
                nonlocal complexity, nesting_level
                complexity += 1 + nesting_level
                nesting_level += 1
                self.generic_visit(node)
                nesting_level -= 1
        
        visitor = CognitiveComplexityVisitor()
        visitor.visit(node)
        
        return complexity
    
    def _calculate_nested_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        class NestingVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                nonlocal max_depth, current_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
            
            def visit_For(self, node):
                nonlocal max_depth, current_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
            
            def visit_While(self, node):
                nonlocal max_depth, current_depth
                current_depth += 1
                max_depth = max(max_depth, current_depth)
                self.generic_visit(node)
                current_depth -= 1
        
        visitor = NestingVisitor()
        visitor.visit(node)
        
        return max_depth
    
    def _count_function_lines(self, node: ast.FunctionDef) -> int:
        """Count lines in a function."""
        if not node.body:
            return 0
        
        start_line = node.lineno
        end_line = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else start_line
        
        return end_line - start_line + 1
    
    def _count_class_lines(self, node: ast.ClassDef) -> int:
        """Count lines in a class."""
        if not node.body:
            return 0
        
        start_line = node.lineno
        end_line = node.body[-1].end_lineno if hasattr(node.body[-1], 'end_lineno') else start_line
        
        return end_line - start_line + 1
    
    def _calculate_function_maintainability(self, node: ast.FunctionDef, complexity: int) -> float:
        """Calculate maintainability score for a function (0-100)."""
        score = 100.0
        
        # Penalize high complexity
        if complexity > 10:
            score -= (complexity - 10) * 5
        
        # Penalize long functions
        lines = self._count_function_lines(node)
        if lines > 50:
            score -= (lines - 50) * 0.5
        
        # Penalize many parameters
        if len(node.args.args) > 5:
            score -= (len(node.args.args) - 5) * 2
        
        # Penalize deep nesting
        nesting = self._calculate_nested_depth(node)
        if nesting > 3:
            score -= (nesting - 3) * 3
        
        return max(0.0, score)
    
    def _calculate_maintainability_index(self, metrics: Dict[str, Any]) -> float:
        """Calculate maintainability index for a file (0-100)."""
        score = 100.0
        
        # Penalize high complexity
        if metrics["total_complexity"] > 50:
            score -= (metrics["total_complexity"] - 50) * 0.5
        
        # Penalize many functions
        if metrics["function_count"] > 20:
            score -= (metrics["function_count"] - 20) * 0.5
        
        # Penalize many classes
        if metrics["class_count"] > 10:
            score -= (metrics["class_count"] - 10) * 1
        
        # Penalize long files
        if metrics["lines_of_code"] > 500:
            score -= (metrics["lines_of_code"] - 500) * 0.01
        
        return max(0.0, score)
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process complexity data and identify patterns."""
        self.logger.info("Processing complexity data and identifying patterns...")
        
        # Identify complexity patterns
        patterns = {
            "high_complexity_functions": self._find_high_complexity_functions(),
            "high_complexity_classes": self._find_high_complexity_classes(),
            "long_functions": self._find_long_functions(),
            "deeply_nested_code": self._find_deeply_nested_code(),
            "low_maintainability_files": self._find_low_maintainability_files()
        }
        
        # Calculate complexity distribution
        self._calculate_complexity_distribution()
        
        # Calculate thresholds
        thresholds = {
            "cyclomatic_complexity": {
                "low": 0,
                "medium": 10,
                "high": 20,
                "very_high": 30
            },
            "cognitive_complexity": {
                "low": 0,
                "medium": 15,
                "high": 25,
                "very_high": 35
            },
            "maintainability_index": {
                "excellent": 80,
                "good": 60,
                "fair": 40,
                "poor": 20
            }
        }
        
        stage_result.complete({
            "patterns": patterns,
            "complexity_distribution": self.complexity_distribution,
            "thresholds": thresholds,
            "total_patterns": sum(len(pattern) for pattern in patterns.values())
        })
        
        total_patterns = sum(len(pattern) for pattern in patterns.values())
        self.logger.info(f"Processed complexity data: {total_patterns} patterns identified")
    
    def _find_high_complexity_functions(self) -> List[Dict[str, Any]]:
        """Find functions with high complexity."""
        high_complexity = []
        
        for file_path, functions in self.function_complexity.items():
            for func in functions:
                if func["cyclomatic_complexity"] > 15:
                    high_complexity.append({
                        "file": str(file_path),
                        "function": func["name"],
                        "line": func["line"],
                        "cyclomatic_complexity": func["cyclomatic_complexity"],
                        "cognitive_complexity": func["cognitive_complexity"],
                        "maintainability_score": func["maintainability_score"]
                    })
        
        return sorted(high_complexity, key=lambda x: x["cyclomatic_complexity"], reverse=True)
    
    def _find_high_complexity_classes(self) -> List[Dict[str, Any]]:
        """Find classes with high complexity."""
        high_complexity = []
        
        for file_path, classes in self.class_complexity.items():
            for cls in classes:
                if cls["total_complexity"] > 50:
                    high_complexity.append({
                        "file": str(file_path),
                        "class": cls["name"],
                        "line": cls["line"],
                        "total_complexity": cls["total_complexity"],
                        "average_complexity": cls["average_complexity"],
                        "method_count": cls["method_count"]
                    })
        
        return sorted(high_complexity, key=lambda x: x["total_complexity"], reverse=True)
    
    def _find_long_functions(self) -> List[Dict[str, Any]]:
        """Find functions that are too long."""
        long_functions = []
        
        for file_path, functions in self.function_complexity.items():
            for func in functions:
                if func["lines_of_code"] > 50:
                    long_functions.append({
                        "file": str(file_path),
                        "function": func["name"],
                        "line": func["line"],
                        "lines_of_code": func["lines_of_code"],
                        "cyclomatic_complexity": func["cyclomatic_complexity"]
                    })
        
        return sorted(long_functions, key=lambda x: x["lines_of_code"], reverse=True)
    
    def _find_deeply_nested_code(self) -> List[Dict[str, Any]]:
        """Find deeply nested code."""
        deeply_nested = []
        
        for file_path, functions in self.function_complexity.items():
            for func in functions:
                if func["nested_depth"] > 4:
                    deeply_nested.append({
                        "file": str(file_path),
                        "function": func["name"],
                        "line": func["line"],
                        "nested_depth": func["nested_depth"],
                        "cyclomatic_complexity": func["cyclomatic_complexity"]
                    })
        
        return sorted(deeply_nested, key=lambda x: x["nested_depth"], reverse=True)
    
    def _find_low_maintainability_files(self) -> List[Dict[str, Any]]:
        """Find files with low maintainability."""
        low_maintainability = []
        
        for file_path, metrics in self.file_complexity.items():
            if metrics["maintainability_index"] < 50:
                low_maintainability.append({
                    "file": str(file_path),
                    "maintainability_index": metrics["maintainability_index"],
                    "total_complexity": metrics["total_complexity"],
                    "function_count": metrics["function_count"],
                    "lines_of_code": metrics["lines_of_code"]
                })
        
        return sorted(low_maintainability, key=lambda x: x["maintainability_index"])
    
    def _calculate_complexity_distribution(self):
        """Calculate complexity distribution across the codebase."""
        self.complexity_distribution = {
            "low_complexity": 0,
            "medium_complexity": 0,
            "high_complexity": 0,
            "very_high_complexity": 0
        }
        
        for file_path, functions in self.function_complexity.items():
            for func in functions:
                complexity = func["cyclomatic_complexity"]
                if complexity <= 10:
                    self.complexity_distribution["low_complexity"] += 1
                elif complexity <= 20:
                    self.complexity_distribution["medium_complexity"] += 1
                elif complexity <= 30:
                    self.complexity_distribution["high_complexity"] += 1
                else:
                    self.complexity_distribution["very_high_complexity"] += 1
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating complexity analysis results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "parsed_files": len(self.parsed_files),
            "total_functions": sum(len(funcs) for funcs in self.function_complexity.values()),
            "total_classes": sum(len(classes) for classes in self.class_complexity.values()),
            "total_lines_of_code": sum(metrics["lines_of_code"] for metrics in self.file_complexity.values()),
            "total_complexity": sum(metrics["total_complexity"] for metrics in self.file_complexity.values()),
            "average_complexity": 0,
            "max_complexity": 0,
            "average_maintainability": 0,
            "complexity_distribution": self.complexity_distribution
        }
        
        # Calculate averages
        if summary["total_functions"] > 0:
            summary["average_complexity"] = summary["total_complexity"] / summary["total_functions"]
        
        # Find maximum complexity
        for functions in self.function_complexity.values():
            for func in functions:
                summary["max_complexity"] = max(summary["max_complexity"], func["cyclomatic_complexity"])
        
        # Calculate average maintainability
        if self.file_complexity:
            total_maintainability = sum(metrics["maintainability_index"] for metrics in self.file_complexity.values())
            summary["average_maintainability"] = total_maintainability / len(self.file_complexity)
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "file_complexity": self.file_complexity,
                "function_complexity": self.function_complexity,
                "class_complexity": self.class_complexity,
                "complexity_metrics": self.complexity_metrics
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_functions']} functions, "
                        f"{summary['total_classes']} classes, "
                        f"{summary['total_complexity']} total complexity")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive complexity analysis reports."""
        self.logger.info("Generating complexity analysis reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate recommendations report
        recommendations_report = self._generate_recommendations_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"complexity_analysis_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"complexity_analysis_detailed_{timestamp}.json"
        recommendations_path = self.config.output_dir / f"complexity_analysis_recommendations_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "recommendations": str(recommendations_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {recommendations_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "complexity_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "complexity_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_recommendations_report(self) -> Dict[str, Any]:
        """Generate recommendations report."""
        return {
            "pipeline": "complexity_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "recommendations": {
                "high_complexity_functions": self._find_high_complexity_functions()[:10],
                "high_complexity_classes": self._find_high_complexity_classes()[:10],
                "long_functions": self._find_long_functions()[:10],
                "deeply_nested_code": self._find_deeply_nested_code()[:10],
                "low_maintainability_files": self._find_low_maintainability_files()[:10]
            }
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("average_complexity", 0) > 15:
            recommendations.append("High average complexity - consider refactoring complex functions")
        
        if summary.get("max_complexity", 0) > 30:
            recommendations.append("Very high complexity functions found - prioritize refactoring")
        
        if summary.get("average_maintainability", 0) < 60:
            recommendations.append("Low maintainability - focus on code quality improvements")
        
        if summary.get("complexity_distribution", {}).get("very_high_complexity", 0) > 0:
            recommendations.append("Very high complexity functions detected - immediate refactoring needed")
        
        if summary.get("total_functions", 0) > 1000:
            recommendations.append("Large number of functions - consider modularization")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.complexity_metrics.clear()
        self.function_complexity.clear()
        self.class_complexity.clear()
        self.file_complexity.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_complexity_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run complexity analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = ComplexityAnalysisPipeline(config)
    return await pipeline.run()