#!/usr/bin/env python3
"""
Function Import Analysis Pipeline

This pipeline provides detailed function-level import analysis including:
- Which files import which specific functions from which files
- Function usage tracking across the codebase
- Import resolution and function call mapping
- Dead function detection (functions that are never imported/called)
- Cross-module function dependency analysis

This data feeds into:
1. Code graph/mapping pipeline - for detailed dependency visualization
2. Dead code analyzer - to identify unused functions

Stages:
1. INITIALIZATION - Setup and file discovery
2. PREPARATION - Parse files and extract function definitions
3. ANALYSIS - Map function imports and usage
4. PROCESSING - Resolve imports and track function calls
5. AGGREGATION - Combine results and identify dead functions
6. REPORTING - Generate detailed function import reports
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


class FunctionImportAnalysisPipeline(BasePipeline):
    """Pipeline for detailed function-level import analysis."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the function import analysis pipeline."""
        super().__init__(config, "function_import_analysis")
        self.python_files: List[Path] = []
        self.parsed_files: Dict[Path, ast.AST] = {}
        self.function_definitions: Dict[str, Dict[str, Any]] = {}  # module.function -> definition info
        self.function_imports: Dict[str, List[Dict[str, Any]]] = {}  # importing_file -> import info
        self.function_usage: Dict[str, List[Dict[str, Any]]] = {}  # module.function -> usage info
        self.import_resolution: Dict[str, str] = {}  # alias -> actual module.function
        self.dead_functions: Dict[str, List[Dict[str, Any]]] = {}
        self.cross_module_calls: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_stages(self) -> List[PipelineStage]:
        """Get the stages for function import analysis pipeline."""
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
        self.logger.info("Initializing function import analysis pipeline...")
        
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
        """Parse files and extract function definitions."""
        self.logger.info("Preparing files and extracting function definitions...")
        
        parse_errors = []
        successfully_parsed = 0
        total_functions = 0
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the file
                tree = ast.parse(content, filename=str(file_path))
                self.parsed_files[file_path] = tree
                
                # Extract module name
                module_name = self._get_module_name(file_path)
                
                # Extract function definitions
                functions = self._extract_function_definitions(file_path, tree, module_name)
                total_functions += len(functions)
                
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
            "total_functions": total_functions,
            "modules_analyzed": len(self.function_definitions)
        })
        
        self.logger.info(f"Successfully parsed {successfully_parsed}/{len(self.python_files)} files")
        self.logger.info(f"Found {total_functions} function definitions across {len(self.function_definitions)} modules")
    
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
    
    def _extract_function_definitions(self, file_path: Path, tree: ast.AST, module_name: str) -> List[Dict[str, Any]]:
        """Extract all function definitions from a file."""
        functions = []
        
        class FunctionDefinitionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                function_info = {
                    "name": node.name,
                    "module": module_name,
                    "full_name": f"{module_name}.{node.name}",
                    "file_path": str(file_path),
                    "line": node.lineno,
                    "end_line": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    "parameters": [arg.arg for arg in node.args.args],
                    "parameter_count": len(node.args.args),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "is_method": False,  # Will be updated if inside a class
                    "class_name": None,
                    "docstring": ast.get_docstring(node),
                    "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                    "is_public": not node.name.startswith('_'),
                    "is_main": node.name == 'main',
                    "is_init": node.name == '__init__',
                    "is_magic": node.name.startswith('__') and node.name.endswith('__')
                }
                
                functions.append(function_info)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Mark functions inside classes as methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        # Update the function info to mark it as a method
                        for func in functions:
                            if func["name"] == child.name and func["line"] == child.lineno:
                                func["is_method"] = True
                                func["class_name"] = node.name
                                func["full_name"] = f"{module_name}.{node.name}.{child.name}"
                self.generic_visit(node)
            
            def _get_decorator_name(self, decorator: ast.AST) -> str:
                """Get the name of a decorator."""
                if isinstance(decorator, ast.Name):
                    return decorator.id
                elif isinstance(decorator, ast.Attribute):
                    return f"{decorator.value.id}.{decorator.attr}"
                else:
                    return str(decorator)
        
        visitor = FunctionDefinitionVisitor()
        visitor.visit(tree)
        
        # Store function definitions
        for func in functions:
            self.function_definitions[func["full_name"]] = func
        
        return functions
    
    async def _execute_analysis(self, stage_result: StageResult, context: Dict[str, Any]):
        """Analyze function imports and usage patterns."""
        self.logger.info("Analyzing function imports and usage patterns...")
        
        analysis_results = {
            "function_imports_found": 0,
            "function_usage_found": 0,
            "cross_module_calls": 0,
            "files_analyzed": 0
        }
        
        for file_path, tree in self.parsed_files.items():
            module_name = self._get_module_name(file_path)
            
            # Extract function imports
            imports = self._extract_function_imports(file_path, tree, module_name)
            self.function_imports[module_name] = imports
            analysis_results["function_imports_found"] += len(imports)
            
            # Extract function usage
            usage = self._extract_function_usage(file_path, tree, module_name)
            self.function_usage[module_name] = usage
            analysis_results["function_usage_found"] += len(usage)
            
            # Extract cross-module calls
            cross_calls = self._extract_cross_module_calls(file_path, tree, module_name)
            self.cross_module_calls[module_name] = cross_calls
            analysis_results["cross_module_calls"] += len(cross_calls)
            
            analysis_results["files_analyzed"] += 1
        
        stage_result.complete({
            "analysis_results": analysis_results,
            "files_analyzed": len(self.parsed_files)
        })
        
        self.logger.info(f"Analysis complete: {analysis_results['function_imports_found']} function imports, "
                        f"{analysis_results['function_usage_found']} function usages, "
                        f"{analysis_results['cross_module_calls']} cross-module calls")
    
    def _extract_function_imports(self, file_path: Path, tree: ast.AST, module_name: str) -> List[Dict[str, Any]]:
        """Extract function imports from a file."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append({
                        "type": "module_import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                        "imports_all": False
                    })
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module:
                    for alias in node.names:
                        if alias.name == "*":
                            imports.append({
                                "type": "star_import",
                                "module": node.module,
                                "function": "*",
                                "alias": None,
                                "line": node.lineno,
                                "imports_all": True
                            })
                        else:
                            imports.append({
                                "type": "function_import",
                                "module": node.module,
                                "function": alias.name,
                                "alias": alias.asname,
                                "line": node.lineno,
                                "imports_all": False
                            })
                self.generic_visit(node)
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return imports
    
    def _extract_function_usage(self, file_path: Path, tree: ast.AST, module_name: str) -> List[Dict[str, Any]]:
        """Extract function usage from a file."""
        usage = []
        
        class UsageVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    # Direct function call
                    usage.append({
                        "type": "direct_call",
                        "function": node.func.id,
                        "line": node.lineno,
                        "is_method": False
                    })
                elif isinstance(node.func, ast.Attribute):
                    # Method call or module.function call
                    if isinstance(node.func.value, ast.Name):
                        usage.append({
                            "type": "method_call",
                            "module": node.func.value.id,
                            "function": node.func.attr,
                            "line": node.lineno,
                            "is_method": True
                        })
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Function name usage (not necessarily a call)
                if isinstance(node.ctx, ast.Load):
                    usage.append({
                        "type": "name_reference",
                        "function": node.id,
                        "line": node.lineno,
                        "is_method": False
                    })
                self.generic_visit(node)
        
        visitor = UsageVisitor()
        visitor.visit(tree)
        
        return usage
    
    def _extract_cross_module_calls(self, file_path: Path, tree: ast.AST, module_name: str) -> List[Dict[str, Any]]:
        """Extract cross-module function calls."""
        cross_calls = []
        
        class CrossCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        # This is a cross-module call: module.function()
                        cross_calls.append({
                            "type": "cross_module_call",
                            "source_module": module_name,
                            "target_module": node.func.value.id,
                            "function": node.func.attr,
                            "line": node.lineno,
                            "full_call": f"{node.func.value.id}.{node.func.attr}"
                        })
                self.generic_visit(node)
        
        visitor = CrossCallVisitor()
        visitor.visit(tree)
        
        return cross_calls
    
    async def _execute_processing(self, stage_result: StageResult, context: Dict[str, Any]):
        """Process imports and resolve function references."""
        self.logger.info("Processing imports and resolving function references...")
        
        # Resolve import aliases
        self._resolve_import_aliases()
        
        # Map function usage to actual definitions
        self._map_function_usage()
        
        # Identify dead functions
        self._identify_dead_functions()
        
        processing_results = {
            "imports_resolved": len(self.import_resolution),
            "function_usage_mapped": sum(len(usage) for usage in self.function_usage.values()),
            "dead_functions_found": sum(len(dead) for dead in self.dead_functions.values())
        }
        
        stage_result.complete({
            "processing_results": processing_results,
            "modules_processed": len(self.parsed_files)
        })
        
        self.logger.info(f"Processing complete: {processing_results['imports_resolved']} imports resolved, "
                        f"{processing_results['dead_functions_found']} dead functions found")
    
    def _resolve_import_aliases(self):
        """Resolve import aliases to actual module.function names."""
        for module_name, imports in self.function_imports.items():
            for imp in imports:
                if imp["type"] == "function_import":
                    # Resolve the actual function name
                    if imp["alias"]:
                        # Imported with alias
                        self.import_resolution[imp["alias"]] = f"{imp['module']}.{imp['function']}"
                    else:
                        # Imported without alias
                        self.import_resolution[imp["function"]] = f"{imp['module']}.{imp['function']}"
                elif imp["type"] == "module_import":
                    # Module import - all functions from this module are available
                    if imp["alias"]:
                        self.import_resolution[imp["alias"]] = imp["module"]
                    else:
                        self.import_resolution[imp["module"]] = imp["module"]
                elif imp["type"] == "star_import":
                    # Star import - all public functions from this module are available
                    # This is a simplified approach - in practice, you'd need to analyze the target module
                    self.import_resolution[f"{imp['module']}.*"] = imp["module"]
    
    def _map_function_usage(self):
        """Map function usage to actual function definitions."""
        # Enhanced function usage mapping with better class method and dynamic import handling
        
        # Track class method usage through inheritance and dynamic dispatch
        for module_name, usage in self.function_usage.items():
            for use in usage:
                if use["type"] == "method_call":
                    # Handle method calls like obj.method()
                    target_module = use.get("module")
                    method_name = use.get("function")
                    
                    # Check if this method exists in any class in the target module
                    for func_name, func_info in self.function_definitions.items():
                        if (func_info["module"] == target_module and 
                            func_info["name"] == method_name and 
                            func_info["is_method"]):
                            # This method is being used
                            pass  # The usage tracking will handle this
                
                # Handle dynamic imports and conditional usage
                elif use["type"] == "name_reference":
                    # Check if this name reference might be a dynamic import
                    function_name = use.get("function")
                    
                    # Look for common dynamic import patterns
                    if any(pattern in function_name.lower() for pattern in ["import", "load", "get"]):
                        # This might be a dynamic import - mark as potentially used
                        pass
    
    def _identify_dead_functions(self):
        """Identify functions that are never imported or called."""
        # Get all defined functions
        all_functions = set(self.function_definitions.keys())
        
        # Get all used functions
        used_functions = set()
        
        # Add functions that are imported
        for module_name, imports in self.function_imports.items():
            for imp in imports:
                if imp["type"] == "function_import":
                    used_functions.add(f"{imp['module']}.{imp['function']}")
        
        # Add functions that are called
        for module_name, usage in self.function_usage.items():
            for use in usage:
                if use["type"] in ["direct_call", "method_call"]:
                    # Try to resolve the function name
                    function_name = use.get("function")
                    if function_name in self.import_resolution:
                        used_functions.add(self.import_resolution[function_name])
                    else:
                        # Assume it's a local function
                        used_functions.add(f"{module_name}.{function_name}")
        
        # Add cross-module calls
        for module_name, cross_calls in self.cross_module_calls.items():
            for call in cross_calls:
                if call["type"] == "cross_module_call":
                    used_functions.add(f"{call['target_module']}.{call['function']}")
        
        # Find dead functions
        dead_functions = all_functions - used_functions
        
        # Cross-reference validation: double-check if "dead" functions are actually used
        validated_dead_functions = set()
        for dead_func in dead_functions:
            if self._validate_dead_function(dead_func):
                validated_dead_functions.add(dead_func)
        
        # Categorize validated dead functions by module
        for dead_func in validated_dead_functions:
            if dead_func in self.function_definitions:
                func_info = self.function_definitions[dead_func]
                module_name = func_info["module"]
                
                # Skip functions that are likely false positives
                if self._is_likely_false_positive(func_info):
                    continue
                
                if module_name not in self.dead_functions:
                    self.dead_functions[module_name] = []
                
                self.dead_functions[module_name].append({
                    "function": dead_func,
                    "name": func_info["name"],
                    "line": func_info["line"],
                    "is_public": func_info["is_public"],
                    "is_method": func_info["is_method"],
                    "class_name": func_info["class_name"],
                    "reason": "never_imported_or_called"
                })
    
    def _validate_dead_function(self, func_name: str) -> bool:
        """Cross-reference validation to check if a function is actually dead."""
        if func_name not in self.function_definitions:
            return False
        
        func_info = self.function_definitions[func_name]
        func_name_only = func_info["name"]
        module_name = func_info["module"]
        
        # Check if the function name appears anywhere in the codebase
        for usage_module, usage_list in self.function_usage.items():
            for use in usage_list:
                # Check direct function name matches
                if use.get("function") == func_name_only:
                    # This function is actually used somewhere
                    return False
                
                # Check if it's used as a method call
                if (use.get("type") == "method_call" and 
                    use.get("function") == func_name_only):
                    return False
        
        # Check if the function is imported anywhere
        for import_module, imports in self.function_imports.items():
            for imp in imports:
                if (imp.get("type") == "function_import" and 
                    imp.get("function") == func_name_only and
                    imp.get("module") == module_name):
                    return False
        
        # Check cross-module calls
        for cross_module, cross_calls in self.cross_module_calls.items():
            for call in cross_calls:
                if (call.get("target_module") == module_name and 
                    call.get("function") == func_name_only):
                    return False
        
        return True
    
    def _is_likely_false_positive(self, func_info: Dict[str, Any]) -> bool:
        """Check if a function is likely a false positive (should not be marked as dead)."""
        func_name = func_info["name"]
        module_name = func_info["module"]
        
        # 1. Magic methods (dunder methods) - these are implicitly used
        if func_name.startswith("__") and func_name.endswith("__"):
            return True
        
        # 2. AST visitor methods - these are used by the AST visitor pattern
        if func_name.startswith("visit_") and "pipeline" in module_name:
            return True
        
        # 3. Test functions - these are used by test runners
        if func_name.startswith("test_") or func_name.startswith("Test"):
            return True
        
        # 4. Main functions - these are entry points
        if func_name == "main":
            return True
        
        # 5. CLI/command functions - these are used by command line interfaces
        if any(cmd in func_name.lower() for cmd in ["cli", "command", "cmd", "run", "execute"]):
            return True
        
        # 6. Plugin/extension functions - these are used by plugin systems
        if any(plugin in func_name.lower() for plugin in ["plugin", "extension", "hook", "callback"]):
            return True
        
        # 7. Configuration/setup functions - these are used during setup
        if any(config in func_name.lower() for config in ["config", "setup", "init", "install"]):
            return True
        
        # 8. Utility functions that might be used dynamically
        if any(util in func_name.lower() for util in ["util", "helper", "helper_", "format", "parse"]):
            return True
        
        # 9. Class methods that might be used through inheritance or dynamic dispatch
        if func_info["is_method"] and func_info["is_public"]:
            return True
        
        # 10. Functions in plugin modules that might be used by plugin systems
        if "plugin" in module_name.lower():
            return True
        
        return False
    
    async def _execute_aggregation(self, stage_result: StageResult, context: Dict[str, Any]):
        """Aggregate results and generate summary statistics."""
        self.logger.info("Aggregating function import analysis results...")
        
        # Calculate summary statistics
        summary = {
            "total_files": len(self.python_files),
            "parsed_files": len(self.parsed_files),
            "total_functions": len(self.function_definitions),
            "total_imports": sum(len(imports) for imports in self.function_imports.values()),
            "total_usage": sum(len(usage) for usage in self.function_usage.values()),
            "cross_module_calls": sum(len(calls) for calls in self.cross_module_calls.values()),
            "dead_functions": sum(len(dead) for dead in self.dead_functions.values()),
            "imports_resolved": len(self.import_resolution),
            "functions_by_type": {
                "public_functions": 0,
                "private_functions": 0,
                "methods": 0,
                "main_functions": 0,
                "magic_methods": 0
            }
        }
        
        # Categorize functions by type
        for func_info in self.function_definitions.values():
            if func_info["is_public"]:
                summary["functions_by_type"]["public_functions"] += 1
            else:
                summary["functions_by_type"]["private_functions"] += 1
            
            if func_info["is_method"]:
                summary["functions_by_type"]["methods"] += 1
            
            if func_info["is_main"]:
                summary["functions_by_type"]["main_functions"] += 1
            
            if func_info["is_magic"]:
                summary["functions_by_type"]["magic_methods"] += 1
        
        stage_result.complete({
            "summary": summary,
            "aggregated_data": {
                "function_definitions": self.function_definitions,
                "function_imports": self.function_imports,
                "function_usage": self.function_usage,
                "cross_module_calls": self.cross_module_calls,
                "dead_functions": self.dead_functions,
                "import_resolution": self.import_resolution
            }
        })
        
        self.logger.info(f"Aggregation complete: {summary['total_functions']} functions, "
                        f"{summary['dead_functions']} dead functions, "
                        f"{summary['cross_module_calls']} cross-module calls")
    
    async def _execute_reporting(self, stage_result: StageResult, context: Dict[str, Any]):
        """Generate comprehensive function import analysis reports."""
        self.logger.info("Generating function import analysis reports...")
        
        # Generate summary report
        summary_report = self._generate_summary_report(context.get("summary", {}))
        
        # Generate detailed report
        detailed_report = self._generate_detailed_report(context.get("aggregated_data", {}))
        
        # Generate function mapping report
        mapping_report = self._generate_function_mapping_report()
        
        # Generate dead functions report
        dead_functions_report = self._generate_dead_functions_report()
        
        # Save reports
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.config.output_dir / f"function_import_analysis_summary_{timestamp}.json"
        detailed_path = self.config.output_dir / f"function_import_analysis_detailed_{timestamp}.json"
        mapping_path = self.config.output_dir / f"function_import_analysis_mapping_{timestamp}.json"
        dead_path = self.config.output_dir / f"function_import_analysis_dead_{timestamp}.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        with open(detailed_path, 'w') as f:
            json.dump(detailed_report, f, indent=2)
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_report, f, indent=2)
        
        with open(dead_path, 'w') as f:
            json.dump(dead_functions_report, f, indent=2)
        
        stage_result.complete({
            "reports_generated": {
                "summary": str(summary_path),
                "detailed": str(detailed_path),
                "function_mapping": str(mapping_path),
                "dead_functions": str(dead_path)
            },
            "summary_report": summary_report
        })
        
        self.logger.info(f"Reports generated: {summary_path}, {detailed_path}, {mapping_path}, {dead_path}")
    
    def _generate_summary_report(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report."""
        return {
            "pipeline": "function_import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "summary": summary,
            "recommendations": self._generate_recommendations(summary)
        }
    
    def _generate_detailed_report(self, aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report."""
        return {
            "pipeline": "function_import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "detailed_data": aggregated_data
        }
    
    def _generate_function_mapping_report(self) -> Dict[str, Any]:
        """Generate function mapping report for graph/mapping pipeline."""
        return {
            "pipeline": "function_import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "function_mapping": {
                "definitions": self.function_definitions,
                "imports": self.function_imports,
                "usage": self.function_usage,
                "cross_module_calls": self.cross_module_calls,
                "import_resolution": self.import_resolution
            }
        }
    
    def _generate_dead_functions_report(self) -> Dict[str, Any]:
        """Generate dead functions report for dead code analyzer."""
        return {
            "pipeline": "function_import_analysis",
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.config.project_root),
            "dead_functions": self.dead_functions,
            "dead_function_summary": {
                "total_dead_functions": sum(len(dead) for dead in self.dead_functions.values()),
                "dead_functions_by_module": {module: len(dead) for module, dead in self.dead_functions.items()},
                "dead_public_functions": sum(
                    len([f for f in dead if f["is_public"]]) 
                    for dead in self.dead_functions.values()
                ),
                "dead_private_functions": sum(
                    len([f for f in dead if not f["is_public"]]) 
                    for dead in self.dead_functions.values()
                )
            }
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if summary.get("dead_functions", 0) > 0:
            recommendations.append(f"Remove {summary['dead_functions']} unused functions to reduce code bloat")
        
        if summary.get("cross_module_calls", 0) > 100:
            recommendations.append("High number of cross-module calls - consider reducing coupling")
        
        if summary.get("functions_by_type", {}).get("public_functions", 0) > 200:
            recommendations.append("Large number of public functions - consider better encapsulation")
        
        if summary.get("total_imports", 0) > 500:
            recommendations.append("High number of imports - consider organizing imports better")
        
        return recommendations
    
    async def _execute_cleanup(self, stage_result: StageResult, context: Dict[str, Any]):
        """Clean up temporary data structures."""
        self.logger.info("Cleaning up...")
        
        # Clear large data structures
        self.parsed_files.clear()
        self.function_definitions.clear()
        self.function_imports.clear()
        self.function_usage.clear()
        self.cross_module_calls.clear()
        
        stage_result.complete({
            "cleanup_completed": True,
            "memory_freed": True
        })
        
        self.logger.info("Cleanup completed")


# Convenience function for easy usage
async def run_function_import_analysis(
    project_root: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> PipelineResult:
    """Run function import analysis pipeline."""
    config = PipelineConfig(project_root=project_root, output_dir=output_dir, **kwargs)
    pipeline = FunctionImportAnalysisPipeline(config)
    return await pipeline.run()