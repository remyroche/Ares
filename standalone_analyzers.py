"""
Standalone analyzers for code quality analysis.
This module contains all the analyzers we created without external dependencies.
"""

import os
import ast
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import json
import logging

# Import minimal modules
from minimal_config import CodeQualityConfig, get_default_config
from minimal_file_utils import find_python_files


class TypeChecker:
    """Comprehensive Python type checking and analysis using AST and basic type inference."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.type_issues = []
        self.type_info = {}
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze types for a single Python file."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Basic AST-based type analysis
            ast_results = self._analyze_ast_types(tree)
            
            # Calculate type coverage
            type_coverage = self._calculate_type_coverage(ast_results)
            
            # Store results
            self.file_stats[str(file_path)] = {
                "type_issues": len(self.type_issues),
                "type_coverage": type_coverage,
                "ast_analysis": ast_results
            }
            
            return {
                "status": "success",
                "issues_found": len(self.type_issues),
                "issues_fixed": 0,
                "type_coverage": type_coverage,
                "details": ast_results
            }
            
        except Exception as e:
            logging.error(f"Error in type validation for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "type_coverage": 0.0,
                "details": {"error": str(e)}
            }
    
    def _analyze_ast_types(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze types using AST parsing."""
        try:
            type_info = []
            missing_type_hints = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Analyze function type hints
                    has_return_type = node.returns is not None
                    has_param_types = any(arg.annotation is not None for arg in node.args.args)
                    
                    type_info.append({
                        "name": node.name,
                        "type_hint": str(node.returns) if node.returns else "Any",
                        "line": node.lineno,
                        "node_type": "function",
                        "has_type_hint": has_return_type or has_param_types
                    })
                    
                    if not has_return_type:
                        missing_type_hints.append(f"Function '{node.name}' missing return type hint")
                    
                    if not has_param_types:
                        missing_type_hints.append(f"Function '{node.name}' missing parameter type hints")
                
                elif isinstance(node, ast.ClassDef):
                    # Analyze class type hints
                    has_bases = len(node.bases) > 0
                    
                    type_info.append({
                        "name": node.name,
                        "type_hint": "object",
                        "line": node.lineno,
                        "node_type": "class",
                        "has_type_hint": has_bases
                    })
                    
                    if not has_bases:
                        missing_type_hints.append(f"Class '{node.name}' missing base class type hints")
            
            return {
                "type_info": type_info,
                "missing_type_hints": missing_type_hints,
                "total_functions": len([info for info in type_info if info["node_type"] == "function"]),
                "total_classes": len([info for info in type_info if info["node_type"] == "class"]),
                "functions_with_types": len([info for info in type_info if info["node_type"] == "function" and info["has_type_hint"]]),
                "classes_with_types": len([info for info in type_info if info["node_type"] == "class" and info["has_type_hint"]])
            }
            
        except Exception as e:
            logging.error(f"Error in AST type analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_type_coverage(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate type coverage percentage."""
        try:
            total_functions = analysis_results.get("total_functions", 0)
            total_classes = analysis_results.get("total_classes", 0)
            
            functions_with_types = analysis_results.get("functions_with_types", 0)
            classes_with_types = analysis_results.get("classes_with_types", 0)
            
            total_items = total_functions + total_classes
            items_with_types = functions_with_types + classes_with_types
            
            if total_items == 0:
                return 100.0
            
            return (items_with_types / total_items) * 100
            
        except Exception as e:
            logging.error(f"Error calculating type coverage: {e}")
            return 0.0
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')


class AdvancedASTAnalyzer:
    """Advanced Python AST analysis with deep code structure insights."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.ast_patterns = []
        self.code_structures = {}
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file using advanced AST analysis."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform comprehensive AST analysis
            structure_analysis = self._analyze_code_structure(tree)
            pattern_analysis = self._detect_patterns(tree)
            quality_analysis = self._assess_code_quality(tree)
            
            # Combine all analysis results
            combined_results = {
                "structure": structure_analysis,
                "patterns": pattern_analysis,
                "quality": quality_analysis
            }
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.ast_patterns),
                "issues_fixed": 0,
                "details": combined_results,
                "quality_score": quality_analysis.get("overall_score", 0)
            }
            
        except Exception as e:
            logging.error(f"Error in advanced AST analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "quality_score": 0
            }
    
    def _analyze_code_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze the overall code structure."""
        structures = []
        function_count = 0
        class_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                structure = self._analyze_function_structure(node)
                structures.append(structure)
            elif isinstance(node, ast.ClassDef):
                class_count += 1
                structure = self._analyze_class_structure(node)
                structures.append(structure)
        
        return {
            "structures": structures,
            "counts": {
                "functions": function_count,
                "classes": class_count
            },
            "total_structures": len(structures)
        }
    
    def _analyze_function_structure(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Analyze the structure of a function."""
        complexity = 1  # Base complexity
        
        # Count control flow statements
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        
        # Analyze function metrics
        metrics = {
            "parameters": len(node.args.args),
            "defaults": len(node.args.defaults),
            "varargs": node.args.vararg is not None,
            "kwargs": node.args.kwarg is not None,
            "decorators": len(node.decorator_list),
            "has_docstring": ast.get_docstring(node) is not None,
            "has_type_hints": node.returns is not None or any(arg.annotation for arg in node.args.args)
        }
        
        return {
            "name": node.name,
            "node_type": "function",
            "line": node.lineno,
            "complexity": complexity,
            "metrics": metrics
        }
    
    def _analyze_class_structure(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze the structure of a class."""
        complexity = 1  # Base complexity
        children = []
        
        # Count methods and analyze inheritance
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                children.append(f"method:{child.name}")
                # Add method complexity
                method_complexity = 1
                for grandchild in ast.walk(child):
                    if isinstance(grandchild, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        method_complexity += 1
                complexity += method_complexity
        
        metrics = {
            "bases": len(node.bases),
            "methods": len([c for c in children if c.startswith("method:")]),
            "has_docstring": ast.get_docstring(node) is not None
        }
        
        return {
            "name": node.name,
            "node_type": "class",
            "line": node.lineno,
            "complexity": complexity,
            "children": children,
            "metrics": metrics
        }
    
    def _detect_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect various code patterns and anti-patterns."""
        patterns = []
        
        # Detect anti-patterns
        for node in ast.walk(tree):
            # Magic numbers
            if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
                if abs(node.n) > 1000 and node.n not in [1000, 10000, 100000]:
                    patterns.append({
                        "type": "anti_pattern",
                        "description": f"Magic number: {node.n}",
                        "line": node.lineno,
                        "severity": "warning"
                    })
            
            # Deep nesting
            if isinstance(node, (ast.If, ast.While, ast.For)):
                nesting_depth = self._calculate_nesting_depth(node)
                if nesting_depth > 4:
                    patterns.append({
                        "type": "anti_pattern",
                        "description": f"Deep nesting ({nesting_depth} levels)",
                        "line": node.lineno,
                        "severity": "warning"
                    })
            
            # Long functions
            if isinstance(node, ast.FunctionDef):
                function_length = len(node.body)
                if function_length > 50:
                    patterns.append({
                        "type": "anti_pattern",
                        "description": f"Long function ({function_length} lines)",
                        "line": node.lineno,
                        "severity": "warning"
                    })
        
        return {
            "patterns": patterns,
            "total_patterns": len(patterns)
        }
    
    def _assess_code_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Assess overall code quality."""
        quality_score = 100.0
        issues = []
        
        # Analyze various quality aspects
        structure_quality = self._assess_structure_quality(tree)
        complexity_quality = self._assess_complexity_quality(tree)
        
        # Calculate overall score
        quality_score = (structure_quality + complexity_quality) / 2
        
        return {
            "overall_score": quality_score,
            "structure_quality": structure_quality,
            "complexity_quality": complexity_quality,
            "issues": issues
        }
    
    def _assess_structure_quality(self, tree: ast.AST) -> float:
        """Assess code structure quality."""
        score = 100.0
        
        # Check for proper organization
        module_structure = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_structure.append("import")
            elif isinstance(node, ast.FunctionDef):
                module_structure.append("function")
            elif isinstance(node, ast.ClassDef):
                module_structure.append("class")
        
        # Penalize mixed organization
        if len(set(module_structure)) > 1:
            score -= 20
        
        return max(0.0, score)
    
    def _assess_complexity_quality(self, tree: ast.AST) -> float:
        """Assess code complexity quality."""
        score = 100.0
        
        total_complexity = 0
        function_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                func_complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        func_complexity += 1
                total_complexity += func_complexity
        
        if function_count > 0:
            avg_complexity = total_complexity / function_count
            if avg_complexity > 10:
                score -= 30
            elif avg_complexity > 5:
                score -= 15
        
        return max(0.0, score)
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate nesting depth of a node."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                max_depth = max(max_depth, self._calculate_nesting_depth(child, current_depth + 1))
        
        return max_depth
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')


class ArchitectureAnalyzer:
    """Comprehensive code architecture analysis."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.architecture_issues = []
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze architecture for a single Python file."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform architecture analysis
            coupling_analysis = self._analyze_coupling(tree)
            cohesion_analysis = self._analyze_cohesion(tree)
            dependency_analysis = self._analyze_dependencies(tree)
            
            # Combine results
            combined_results = {
                "coupling": coupling_analysis,
                "cohesion": cohesion_analysis,
                "dependencies": dependency_analysis
            }
            
            # Calculate overall architecture score
            architecture_score = self._calculate_architecture_score(combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.architecture_issues),
                "issues_fixed": 0,
                "details": combined_results,
                "architecture_score": architecture_score
            }
            
        except Exception as e:
            logging.error(f"Error in architecture analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "architecture_score": 0
            }
    
    def _analyze_coupling(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze coupling between components."""
        external_dependencies = set()
        internal_dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    external_dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    external_dependencies.add(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    internal_dependencies.add(node.func.id)
        
        # Calculate coupling score (higher = more coupled)
        coupling_score = min(100.0, len(external_dependencies) * 5 + len(internal_dependencies) * 2)
        
        return {
            "external_dependencies": list(external_dependencies),
            "internal_dependencies": list(internal_dependencies),
            "total_dependencies": len(external_dependencies) + len(internal_dependencies),
            "coupling_score": coupling_score
        }
    
    def _analyze_cohesion(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze cohesion within components."""
        cohesion_score = 100.0
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Analyze class cohesion
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                if len(methods) > 15:
                    cohesion_score -= 20
                    issues.append(f"Large class '{node.name}' with {len(methods)} methods")
        
        return {
            "cohesion_score": max(0.0, cohesion_score),
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def _analyze_dependencies(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze dependencies and build dependency graph."""
        dependencies = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            dependencies[func_name].append(child.func.id)
        
        return {
            "dependency_graph": dict(dependencies),
            "total_dependencies": sum(len(deps) for deps in dependencies.values())
        }
    
    def _calculate_architecture_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall architecture quality score."""
        score = 100.0
        
        # Coupling penalty (higher coupling = lower score)
        coupling_score = analysis_results.get("coupling", {}).get("coupling_score", 0)
        score -= coupling_score * 0.3
        
        # Cohesion penalty (lower cohesion = lower score)
        cohesion_score = analysis_results.get("cohesion", {}).get("cohesion_score", 100)
        score -= (100 - cohesion_score) * 0.2
        
        return max(0.0, score)
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')


class CodeDuplicationAnalyzer:
    """Code duplication detection and analysis."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.duplication_blocks = []
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for code duplication."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform duplication analysis
            function_duplications = self._analyze_function_duplications(tree, str(file_path))
            class_duplications = self._analyze_class_duplications(tree, str(file_path))
            
            # Combine results
            combined_results = {
                "function_duplications": function_duplications,
                "class_duplications": class_duplications
            }
            
            # Calculate duplication metrics
            duplication_score = self._calculate_duplication_score(combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.duplication_blocks),
                "issues_fixed": 0,
                "details": combined_results,
                "duplication_score": duplication_score
            }
            
        except Exception as e:
            logging.error(f"Error in duplication analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "duplication_score": 0
            }
    
    def _analyze_function_duplications(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze function-level duplications."""
        functions = []
        function_hashes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function content
                func_content = self._extract_function_content(node)
                func_hash = hashlib.md5(func_content.encode()).hexdigest()
                
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "content": func_content,
                    "hash": func_hash,
                    "length": len(func_content.split('\n'))
                })
                
                if func_hash in function_hashes:
                    function_hashes[func_hash].append((file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1)))
                else:
                    function_hashes[func_hash] = [(file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1))]
        
        # Find duplicate functions
        duplicate_functions = []
        for func_hash, locations in function_hashes.items():
            if len(locations) > 1:
                duplicate_functions.append({
                    "content": func_content,
                    "hash_value": func_hash,
                    "locations": [
                        {"file": loc[0], "start_line": loc[1], "end_line": loc[2]}
                        for loc in locations
                    ],
                    "duplicate_count": len(locations)
                })
        
        return {
            "total_functions": len(functions),
            "duplicate_functions": duplicate_functions,
            "duplicate_count": len(duplicate_functions)
        }
    
    def _analyze_class_duplications(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze class-level duplications."""
        classes = []
        class_hashes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class content
                class_content = self._extract_class_content(node)
                class_hash = hashlib.md5(class_content.encode()).hexdigest()
                
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "content": class_content,
                    "hash": class_hash
                })
                
                if class_hash in class_hashes:
                    class_hashes[class_hash].append((file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1)))
                else:
                    class_hashes[class_hash] = [(file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1))]
        
        # Find duplicate classes
        duplicate_classes = []
        for class_hash, locations in class_hashes.items():
            if len(locations) > 1:
                duplicate_classes.append({
                    "content": class_content,
                    "hash_value": class_hash,
                    "locations": [
                        {"file": loc[0], "start_line": loc[1], "end_line": loc[2]}
                        for loc in locations
                    ],
                    "duplicate_count": len(locations)
                })
        
        return {
            "total_classes": len(classes),
            "duplicate_classes": duplicate_classes,
            "duplicate_count": len(duplicate_classes)
        }
    
    def _extract_function_content(self, node: ast.FunctionDef) -> str:
        """Extract the content of a function."""
        try:
            # Create a simplified representation
            content = f"def {node.name}({len(node.args.args)} args):\n"
            content += f"  {len(node.body)} statements\n"
            content += f"  Decorators: {len(node.decorator_list)}\n"
            content += f"  Returns: {node.returns is not None}\n"
            
            return content
        except Exception:
            return f"def {node.name}()"
    
    def _extract_class_content(self, node: ast.ClassDef) -> str:
        """Extract the content of a class."""
        try:
            methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
            
            content = f"class {node.name}({len(node.bases)} bases):\n"
            content += f"  Methods: {len(methods)}\n"
            
            return content
        except Exception:
            return f"class {node.name}"
    
    def _calculate_duplication_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate duplication score (lower is better)."""
        score = 100.0
        
        # Penalize based on duplication counts
        function_dups = analysis_results.get("function_duplications", {}).get("duplicate_count", 0)
        class_dups = analysis_results.get("class_duplications", {}).get("duplicate_count", 0)
        
        total_dups = function_dups + class_dups
        
        # Apply penalties
        score -= function_dups * 15  # Function duplication is most serious
        score -= class_dups * 12     # Class duplication is serious
        
        return max(0.0, score)
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')


class ErrorHandlingAnalyzer:
    """Error handling analysis and quality assessment."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.error_issues = []
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze error handling for a single Python file."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform error handling analysis
            exception_analysis = self._analyze_exception_handling(tree)
            quality_analysis = self._assess_error_handling_quality(tree)
            
            # Combine results
            combined_results = {
                "exceptions": exception_analysis,
                "quality": quality_analysis
            }
            
            # Calculate overall error handling score
            error_handling_score = self._calculate_error_handling_score(combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.error_issues),
                "issues_fixed": 0,
                "details": combined_results,
                "error_handling_score": error_handling_score
            }
            
        except Exception as e:
            logging.error(f"Error in error handling analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "error_handling_score": 0
            }
    
    def _analyze_exception_handling(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze exception handling patterns."""
        try_blocks = []
        except_blocks = []
        bare_excepts = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_blocks.append({
                    "line": node.lineno,
                    "body_length": len(node.body),
                    "handlers": len(node.handlers)
                })
                
                # Analyze except handlers
                for handler in node.handlers:
                    if handler.type is None:
                        bare_excepts.append({
                            "line": handler.lineno,
                            "description": "Bare except clause"
                        })
        
        return {
            "try_blocks": try_blocks,
            "except_blocks": len(bare_excepts),
            "bare_excepts": bare_excepts,
            "total_exception_handlers": len(bare_excepts)
        }
    
    def _assess_error_handling_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Assess overall error handling quality."""
        quality_score = 100.0
        issues = []
        
        # Analyze exception handling coverage
        functions_with_exceptions = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if self._function_has_exception_handling(node):
                    functions_with_exceptions += 1
        
        # Calculate coverage
        exception_coverage = (functions_with_exceptions / total_functions * 100) if total_functions > 0 else 0
        
        # Penalize for low coverage
        if exception_coverage < 50:
            quality_score -= 30
            issues.append("Low exception handling coverage")
        elif exception_coverage < 80:
            quality_score -= 15
            issues.append("Moderate exception handling coverage")
        
        return {
            "quality_score": max(0.0, quality_score),
            "exception_coverage": exception_coverage,
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def _function_has_exception_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function has exception handling."""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def _calculate_error_handling_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall error handling quality score."""
        score = 100.0
        
        # Quality score from analysis
        quality_score = analysis_results.get("quality", {}).get("quality_score", 100)
        score = (score + quality_score) / 2
        
        # Exception coverage bonus
        exception_coverage = analysis_results.get("quality", {}).get("exception_coverage", 0)
        if exception_coverage > 80:
            score += 10
        elif exception_coverage > 60:
            score += 5
        
        return max(0.0, min(100.0, score))
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')


class ConcurrencyAnalyzer:
    """Concurrency analysis and quality assessment."""
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.concurrency_issues = []
        self.file_stats = {}
        
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze concurrency for a single Python file."""
        try:
            file_path = Path(file_path).resolve()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform concurrency analysis
            threading_analysis = self._analyze_threading_patterns(tree)
            async_analysis = self._analyze_async_patterns(tree)
            quality_analysis = self._assess_concurrency_quality(tree)
            
            # Combine results
            combined_results = {
                "threading": threading_analysis,
                "async_patterns": async_analysis,
                "quality": quality_analysis
            }
            
            # Calculate overall concurrency score
            concurrency_score = self._calculate_concurrency_score(combined_results)
            
            # Store results
            self.file_stats[str(file_path)] = combined_results
            
            return {
                "status": "success",
                "issues_found": len(self.concurrency_issues),
                "issues_fixed": 0,
                "details": combined_results,
                "concurrency_score": concurrency_score
            }
            
        except Exception as e:
            logging.error(f"Error in concurrency analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "concurrency_score": 0
            }
    
    def _analyze_threading_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze threading patterns and usage."""
        threading_imports = []
        thread_creations = []
        
        for node in ast.walk(tree):
            # Check for threading imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['threading', 'thread', '_thread']:
                        threading_imports.append({
                            "line": node.lineno,
                            "module": alias.name
                        })
            
            # Check for thread creation
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'Thread':
                    thread_creations.append({
                        "line": node.lineno,
                        "type": "Thread_instantiation",
                        "args": len(node.args)
                    })
        
        return {
            "threading_imports": threading_imports,
            "thread_creations": thread_creations,
            "total_threading_operations": len(thread_creations)
        }
    
    def _analyze_async_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze async/await patterns."""
        async_functions = []
        await_statements = []
        
        for node in ast.walk(tree):
            # Check for async function definitions
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('async'):
                    async_functions.append({
                        "line": node.lineno,
                        "name": node.name,
                        "is_coroutine": True
                    })
            
            # Check for await statements
            elif isinstance(node, ast.Await):
                await_statements.append({
                    "line": node.lineno,
                    "value": str(node.value)
                })
        
        return {
            "async_functions": async_functions,
            "await_statements": await_statements,
            "total_async_operations": len(async_functions) + len(await_statements)
        }
    
    def _assess_concurrency_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Assess overall concurrency quality."""
        quality_score = 100.0
        issues = []
        
        # Check for proper synchronization
        threading_ops = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'Thread':
                    threading_ops += 1
        
        # Penalize for threading without proper patterns
        if threading_ops > 0:
            quality_score -= 20
            issues.append("Threading operations detected")
        
        return {
            "quality_score": max(0.0, quality_score),
            "issues": issues,
            "total_issues": len(issues),
            "threading_operations": threading_ops
        }
    
    def _calculate_concurrency_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall concurrency quality score."""
        score = 100.0
        
        # Quality score from analysis
        quality_score = analysis_results.get("quality", {}).get("quality_score", 100)
        score = (score + quality_score) / 2
        
        # Threading penalties
        threading_ops = analysis_results.get("quality", {}).get("threading_operations", 0)
        if threading_ops > 0:
            score -= 20
        
        return max(0.0, min(100.0, score))
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')