#!/usr/bin/env python3
"""
Dead Code Analysis Pipeline

This pipeline focuses on identifying and analyzing dead code including:
- Unused functions and classes
- Unreachable code paths
- Unused variables and imports
- Dead code elimination opportunities
- Code coverage analysis

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and build call graphs
3. ANALYSIS - Identify dead code patterns
4. PROCESSING - Categorize and validate findings
5. AGGREGATION - Combine results and generate insights
6. REPORTING - Generate comprehensive dead code reports
7. CLEANUP - Clean up temporary structures
"""

import ast
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base_pipeline import BasePipeline, PipelineConfig, PipelineStage, StageResult, PipelineStatus, PipelineResult


class DeadCodeAnalysisPipeline(BasePipeline):
    """Pipeline for comprehensive dead code analysis and elimination."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the dead code analysis pipeline."""
        super().__init__(config, "dead_code_analysis")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.defined_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.used_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.dead_functions: Dict[str, List[Dict[str, Any]]] = {}
        self.dead_classes: Dict[str, List[Dict[str, Any]]] = {}
        self.unused_variables: Dict[str, List[Dict[str, Any]]] = {}
        self.unreachable_code: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for dead code analysis pipeline."""
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
        self.logger.info("Initializing dead code analysis pipeline...")
        
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
        """Parse files and build call graphs."""
        self.logger.info("Preparing files and building call graphs...")
        
        parse_errors = []
        successfully_parsed = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                
                # Extract module name
                module_name = self._get_module_name(file_path)
                
                # Build call graph and symbol tables
                self._build_call_graph(file_path, tree, module_name)
                
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
            "total_files": len(self.python_files),
            "modules_analyzed": len(self.defined_symbols),
            "call_graph_size": len(self.call_graph)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        self.logger.info(f"Built call graph with {len(self.call_graph)} symbols")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            relative_path = file_path.relative_to(self.config.project_root)
        except ValueError:
            return str(file_path)
        
        parts = list(relative_path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension
        
        return ".".join(parts)
    
    def _build_call_graph(self, file_path: Path, tree: ast.AST, module_name: str):
        """Build call graph and symbol tables for a file."""
        defined_symbols = set()
        used_symbols = set()
        calls = set()
        
        class CallGraphVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                defined_symbols.add(node.name)
                # Check if it's a main function or has special names
                if node.name in ["main", "__init__", "__call__", "__str__", "__repr__"]:
                    used_symbols.add(node.name)  # These are implicitly used
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                defined_symbols.add(node.name)
                # Check if it's a main class or has special methods
                if node.name in ["Main", "Base", "Abstract"]:
                    used_symbols.add(node.name)  # These might be implicitly used
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Track variable assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_symbols.add(target.id)
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Track symbol usage
                if isinstance(node.ctx, ast.Load):
                    used_symbols.add(node.id)
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Track function calls
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    if isinstance(node.func.value, ast.Name):
                        calls.add(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                # Track attribute access
                if isinstance(node.ctx, ast.Load):
                    if isinstance(node.value, ast.Name):
                        used_symbols.add(f"{node.value.id}.{node.attr}")
                self.generic_visit(node)
        
        visitor = CallGraphVisitor()
        visitor.visit(tree)
        
        # Store symbol information
        self.defined_symbols[module_name] = defined_symbols
        self.used_symbols[module_name] = used_symbols
        
        # Build call graph
        for call in calls:
            self.call_graph[module_name].add(call)
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze dead code patterns."""
        self.logger.info("Analyzing dead code patterns...")
        
        analysis_results = {
            "dead_functions": 0,
            "dead_classes": 0,
            "unused_variables": 0,
            "unreachable_code": 0
        }
        
        # Find dead functions
        self.dead_functions = self._find_dead_functions()
        analysis_results["dead_functions"] = sum(len(funcs) for funcs in self.dead_functions.values())
        
        # Find dead classes
        self.dead_classes = self._find_dead_classes()
        analysis_results["dead_classes"] = sum(len(classes) for classes in self.dead_classes.values())
        
        # Find unused variables
        self.unused_variables = self._find_unused_variables()
        analysis_results["unused_variables"] = sum(len(vars) for vars in self.unused_variables.values())
        
        # Find unreachable code
        self.unreachable_code = self._find_unreachable_code()
        analysis_results["unreachable_code"] = sum(len(code) for code in self.unreachable_code.values())
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "modules_analyzed": len(self.parsed_files)
        })
        
        total_dead_code = sum(analysis_results.values())
        self.logger.info(f"Analysis complete: {total_dead_code} dead code items found")
    
    def _find_dead_functions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused functions."""
        dead_functions = {}
        
        for module_name, defined_symbols in self.defined_symbols.items():
            used_symbols = self.used_symbols[module_name]
            module_dead_functions = []
            
            for symbol in defined_symbols:
                # Check if it's a function (not a class or variable)
                if self._is_function(module_name, symbol):
                    # Check if it's used anywhere
                    is_used = False
                    
                    # Check if used in this module
                    if symbol in used_symbols:
                        is_used = True
                    
                    # Check if used in other modules
                    if not is_used:
                        for other_module, other_used in self.used_symbols.items():
                            if other_module != module_name:
                                # Check for direct usage
                                if symbol in other_used:
                                    is_used = True
                                    break
                                # Check for module.symbol usage
                                if f"{module_name}.{symbol}" in other_used:
                                    is_used = True
                                    break
                    
                    # Check if it's a special function that might be used implicitly
                    if not is_used and symbol.startswith("__") and symbol.endswith("__"):
                        is_used = True  # Magic methods are implicitly used
                    
                    # Check if it's a main function
                    if not is_used and symbol == "main":
                        is_used = True  # Main functions are entry points
                    
                    if not is_used:
                        module_dead_functions.append({
                            "name": symbol,
                            "type": "function",
                            "reason": "not_called"
                        })
            
            if module_dead_functions:
                dead_functions[module_name] = module_dead_functions
        
        return dead_functions
    
    def _find_dead_classes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused classes."""
        dead_classes = {}
        
        for module_name, defined_symbols in self.defined_symbols.items():
            used_symbols = self.used_symbols[module_name]
            module_dead_classes = []
            
            for symbol in defined_symbols:
                # Check if it's a class
                if self._is_class(module_name, symbol):
                    # Check if it's used anywhere
                    is_used = False
                    
                    # Check if used in this module
                    if symbol in used_symbols:
                        is_used = True
                    
                    # Check if used in other modules
                    if not is_used:
                        for other_module, other_used in self.used_symbols.items():
                            if other_module != module_name:
                                # Check for direct usage
                                if symbol in other_used:
                                    is_used = True
                                    break
                                # Check for module.symbol usage
                                if f"{module_name}.{symbol}" in other_used:
                                    is_used = True
                                    break
                    
                    # Check if it's a base class or has special methods
                    if not is_used and self._has_special_methods(module_name, symbol):
                        is_used = True  # Classes with special methods might be used
                    
                    if not is_used:
                        module_dead_classes.append({
                            "name": symbol,
                            "type": "class",
                            "reason": "not_instantiated"
                        })
            
            if module_dead_classes:
                dead_classes[module_name] = module_dead_classes
        
        return dead_classes
    
    def _find_unused_variables(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unused variables."""
        unused_variables = {}
        
        for file_path, tree in self.parsed_files.items():
            module_name = self._get_module_name(file_path)
            module_unused = []
            
            class VariableVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.assigned_vars = set()
                    self.used_vars = set()
                
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.assigned_vars.add(target.id)
                    self.generic_visit(node)
                
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        self.used_vars.add(node.id)
                    self.generic_visit(node)
            
            visitor = VariableVisitor()
            visitor.visit(tree)
            
            # Find unused variables
            for var in visitor.assigned_vars:
                if var not in visitor.used_vars and not var.startswith("_"):
                    module_unused.append({
                        "name": var,
                        "type": "variable",
                        "reason": "not_used"
                    })
            
            if module_unused:
                unused_variables[module_name] = module_unused
        
        return unused_variables
    
    def _find_unreachable_code(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find unreachable code (basic analysis)."""
        unreachable_code = {}
        
        for file_path, tree in self.parsed_files.items():
            module_name = self._get_module_name(file_path)
            module_unreachable = []
            
            class UnreachableVisitor(ast.NodeVisitor):
                def visit_Return(self, node):
                    # Check if there's code after return
                    if hasattr(node, 'sibling') and node.sibling:
                        module_unreachable.append({
                            "type": "unreachable_code",
                            "line": node.lineno,
                            "reason": "code_after_return"
                        })
                    self.generic_visit(node)
                
                def visit_Raise(self, node):
                    # Check if there's code after raise
                    if hasattr(node, 'sibling') and node.sibling:
                        module_unreachable.append({
                            "type": "unreachable_code",
                            "line": node.lineno,
                            "reason": "code_after_raise"
                        })
                    self.generic_visit(node)
            
            visitor = UnreachableVisitor()
            visitor.visit(tree)
            
            if module_unreachable:
                unreachable_code[module_name] = module_unreachable
        
        return unreachable_code
    
    def _is_function(self, module_name: str, symbol: str) -> bool:
        """Check if a symbol is a function."""
        # This is a simplified check - in a real implementation,
        # you'd need to track the AST node types more carefully
        return True  # Simplified for this example
    
    def _is_class(self, module_name: str, symbol: str) -> bool:
        """Check if a symbol is a class."""
        # This is a simplified check - in a real implementation,
        # you'd need to track the AST node types more carefully
        return symbol[0].isupper()  # Simple heuristic: classes start with uppercase
    
    def _has_special_methods(self, module_name: str, symbol: str) -> bool:
        """Check if a class has special methods."""
        # This is a simplified check
        return False  # Simplified for this example
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process and categorize dead code findings."""
        self.logger.info("Processing dead code findings...")
        
        # Categorize by type and severity
        dead_code_categories = {
            "functions": [],
            "classes": [],
            "variables": [],
            "unreachable": []
        }
        
        # Process dead functions
        for module, functions in self.dead_functions.items():
            for func in functions:
                dead_code_categories["functions"].append({
                    "module": module,
                    "name": func["name"],
                    "type": func["type"],
                    "reason": func["reason"],
                    "severity": "high"
                })
        
        # Process dead classes
        for module, classes in self.dead_classes.items():
            for cls in classes:
                dead_code_categories["classes"].append({
                    "module": module,
                    "name": cls["name"],
                    "type": cls["type"],
                    "reason": cls["reason"],
                    "severity": "high"
                })
        
        # Process unused variables
        for module, variables in self.unused_variables.items():
            for var in variables:
                dead_code_categories["variables"].append({
                    "module": module,
                    "name": var["name"],
                    "type": var["type"],
                    "reason": var["reason"],
                    "severity": "medium"
                })
        
        # Process unreachable code
        for module, code in self.unreachable_code.items():
            for unreachable in code:
                dead_code_categories["unreachable"].append({
                    "module": module,
                    "type": unreachable["type"],
                    "line": unreachable["line"],
                    "reason": unreachable["reason"],
                    "severity": "medium"
                })
        
        stage_result.complete({
            "dead_code_categories": dead_code_categories,
            "total_dead_code": sum(len(items) for items in dead_code_categories.values())
        })
        
        total_dead_code = sum(len(items) for items in dead_code_categories.values())
        self.logger.info(f"Processed {total_dead_code} dead code items")
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating dead code analysis results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "parsed_files": len(self.parsed_files),
            "total_dead_code": 0,
            "dead_code_by_type": {},
            "modules_with_dead_code": 0,
            "potential_savings": {
                "lines_of_code": 0,
                "functions": 0,
                "classes": 0
            }
        }
        
        # Aggregate dead code by type
        dead_code_categories = context.get("dead_code_categories", {})
        for category, items in dead_code_categories.items():
            summary["dead_code_by_type"][category] = len(items)
            summary["total_dead_code"] += len(items)
        
        # Count modules with dead code
        modules_with_dead_code = set()
        for category, items in dead_code_categories.items():
            for item in items:
                modules_with_dead_code.add(item["module"])
        summary["modules_with_dead_code"] = len(modules_with_dead_code)
        
        # Estimate potential savings
        summary["potential_savings"]["functions"] = len(dead_code_categories.get("functions", []))
        summary["potential_savings"]["classes"] = len(dead_code_categories.get("classes", []))
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "dead_functions": self.dead_functions,
                "dead_classes": self.dead_classes,
                "unused_variables": self.unused_variables,
                "unreachable_code": self.unreachable_code,
                "call_graph": dict(self.call_graph)
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_dead_code']} total dead code items, "
                        f"{summary['modules_with_dead_code']} modules affected")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive dead code analysis reports."""
        self.logger.info("Generating dead code analysis reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate elimination report
        elimination_report = self._generate_elimination_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"dead_code_analysis_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"dead_code_analysis_detailed_{timestamp}.json"
        elimination_path = self.config.output_dir / f"dead_code_analysis_elimination_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(elimination_path, 'w') as f:
            json.dump(elimination_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "elimination": str(elimination_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {elimination_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "dead_code_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "dead_code_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_elimination_report(self) -> Dict[str, Any]:
        """Generate dead code elimination report."""
        return {
            "pipeline": "dead_code_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "elimination_plan": {
                "dead_functions": self.dead_functions,
                "dead_classes": self.dead_classes,
                "unused_variables": self.unused_variables,
                "unreachable_code": self.unreachable_code
            }
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("dead_code_by_type", {}).get("functions", 0) > 0:
            recommendations.append("Remove unused functions to reduce code complexity")
        
        if summary.get("dead_code_by_type", {}).get("classes", 0) > 0:
            recommendations.append("Remove unused classes to improve maintainability")
        
        if summary.get("dead_code_by_type", {}).get("variables", 0) > 0:
            recommendations.append("Remove unused variables to clean up code")
        
        if summary.get("dead_code_by_type", {}).get("unreachable", 0) > 0:
            recommendations.append("Remove unreachable code to improve readability")
        
        if summary.get("potential_savings", {}).get("functions", 0) > 10:
            recommendations.append("Consider significant refactoring - many unused functions found")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.call_graph.clear()
        self.defined_symbols.clear()
        self.used_symbols.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_dead_code_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run dead code analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = DeadCodeAnalysisPipeline(config)
    return await pipeline.run()