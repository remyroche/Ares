"""
Advanced AST Analyzer - Deep Python code structure analysis, pattern detection, and quality insights.
"""

import os
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import json
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from minimal_config import CodeQualityConfig, get_default_config
from minimal_file_utils import find_python_files


class ASTPattern:
    """Container for AST pattern information."""
    
    def __init__(self, pattern_type: str, description: str, line: int, 
                 severity: str = "info", details: Dict[str, Any] = None):
        self.pattern_type = pattern_type
        self.description = description
        self.line = line
        self.severity = severity
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "line": self.line,
            "severity": self.severity,
            "details": self.details
        }


class CodeStructure:
    """Container for code structure information."""
    
    def __init__(self, name: str, node_type: str, line: int, 
                 complexity: int = 0, children: List[str] = None,
                 metrics: Dict[str, Any] = None):
        self.name = name
        self.node_type = node_type
        self.line = line
        self.complexity = complexity
        self.children = children or []
        self.metrics = metrics or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "node_type": self.node_type,
            "line": self.line,
            "complexity": self.complexity,
            "children": self.children,
            "metrics": self.metrics
        }


class AdvancedASTAnalyzer:
    """
    Advanced Python AST analysis with deep code structure insights.
    
    Features:
    - Deep AST traversal and analysis
    - Code pattern detection
    - Structural complexity metrics
    - Anti-pattern identification
    - Code quality scoring
    - Architecture insights
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.ast_patterns: List[ASTPattern] = []
        self.code_structures: Dict[str, List[CodeStructure]] = {}
        self.pattern_stats: Dict[str, int] = {}
        self.quality_metrics: Dict[str, Dict[str, Any]] = {}
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Python file using advanced AST analysis.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Clear previous results for this file
            self.ast_patterns = [p for p in self.ast_patterns if p.description != str(file_path)]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform comprehensive AST analysis
            structure_analysis = self._analyze_code_structure(tree)
            pattern_analysis = self._detect_patterns(tree)
            quality_analysis = self._assess_code_quality(tree)
            architecture_analysis = self._analyze_architecture(tree)
            
            # Combine all analysis results
            combined_results = {
                "structure": structure_analysis,
                "patterns": pattern_analysis,
                "quality": quality_analysis,
                "architecture": architecture_analysis
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
        module_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                structure = self._analyze_function_structure(node)
                structures.append(structure)
            elif isinstance(node, ast.ClassDef):
                class_count += 1
                structure = self._analyze_class_structure(node)
                structures.append(structure)
            elif isinstance(node, ast.Module):
                module_count += 1
                structure = self._analyze_module_structure(node)
                structures.append(structure)
        
        return {
            "structures": [s.to_dict() for s in structures],
            "counts": {
                "functions": function_count,
                "classes": class_count,
                "modules": module_count
            },
            "total_structures": len(structures)
        }
    
    def _analyze_function_structure(self, node: ast.FunctionDef) -> CodeStructure:
        """Analyze the structure of a function."""
        complexity = 1  # Base complexity
        children = []
        
        # Count control flow statements
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.FunctionDef):
                children.append(f"nested_function:{child.name}")
            elif isinstance(child, ast.ClassDef):
                children.append(f"nested_class:{child.name}")
        
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
        
        return CodeStructure(
            name=node.name,
            node_type="function",
            line=node.lineno,
            complexity=complexity,
            children=children,
            metrics=metrics
        )
    
    def _analyze_class_structure(self, node: ast.ClassDef) -> CodeStructure:
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
            elif isinstance(child, ast.ClassDef):
                children.append(f"nested_class:{child.name}")
        
        metrics = {
            "bases": len(node.bases),
            "methods": len([c for c in children if c.startswith("method:")]),
            "nested_classes": len([c for c in children if c.startswith("nested_class:")]),
            "has_docstring": ast.get_docstring(node) is not None,
            "inheritance_depth": self._calculate_inheritance_depth(node)
        }
        
        return CodeStructure(
            name=node.name,
            node_type="class",
            line=node.lineno,
            complexity=complexity,
            children=children,
            metrics=metrics
        )
    
    def _analyze_module_structure(self, node: ast.Module) -> CodeStructure:
        """Analyze the structure of a module."""
        children = []
        
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                children.append(f"function:{child.name}")
            elif isinstance(child, ast.ClassDef):
                children.append(f"class:{child.name}")
            elif isinstance(child, ast.Import):
                children.append(f"import:{', '.join(alias.name for alias in child.names)}")
            elif isinstance(child, ast.ImportFrom):
                children.append(f"import_from:{child.module}")
        
        metrics = {
            "total_imports": len([c for c in children if c.startswith("import")]),
            "total_functions": len([c for c in children if c.startswith("function:")]),
            "total_classes": len([c for c in children if c.startswith("class:")])
        }
        
        return CodeStructure(
            name="module",
            node_type="module",
            line=1,
            complexity=len(children),
            children=children,
            metrics=metrics
        )
    
    def _detect_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect various code patterns and anti-patterns."""
        patterns = []
        
        # Detect anti-patterns
        patterns.extend(self._detect_anti_patterns(tree))
        
        # Detect design patterns
        patterns.extend(self._detect_design_patterns(tree))
        
        # Detect code smells
        patterns.extend(self._detect_code_smells(tree))
        
        # Count pattern types
        pattern_counts = Counter(p.pattern_type for p in patterns)
        
        return {
            "patterns": [p.to_dict() for p in patterns],
            "pattern_counts": dict(pattern_counts),
            "total_patterns": len(patterns)
        }
    
    def _detect_anti_patterns(self, tree: ast.AST) -> List[ASTPattern]:
        """Detect anti-patterns in the code."""
        anti_patterns = []
        
        for node in ast.walk(tree):
            # Magic numbers
            if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
                if abs(node.n) > 1000 and node.n not in [1000, 10000, 100000]:
                    anti_patterns.append(ASTPattern(
                        "anti_pattern",
                        f"Magic number: {node.n}",
                        node.lineno,
                        "warning",
                        {"value": node.n, "type": "magic_number"}
                    ))
            
            # Deep nesting
            if isinstance(node, (ast.If, ast.While, ast.For)):
                nesting_depth = self._calculate_nesting_depth(node)
                if nesting_depth > 4:
                    anti_patterns.append(ASTPattern(
                        "anti_pattern",
                        f"Deep nesting ({nesting_depth} levels)",
                        node.lineno,
                        "warning",
                        {"nesting_depth": nesting_depth}
                    ))
            
            # Long functions
            if isinstance(node, ast.FunctionDef):
                function_length = len(node.body)
                if function_length > 50:
                    anti_patterns.append(ASTPattern(
                        "anti_pattern",
                        f"Long function ({function_length} lines)",
                        node.lineno,
                        "warning",
                        {"function_length": function_length}
                    ))
            
            # Bare except clauses
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    anti_patterns.append(ASTPattern(
                        "anti_pattern",
                        "Bare except clause",
                        node.lineno,
                        "error",
                        {"type": "bare_except"}
                    ))
        
        return anti_patterns
    
    def _detect_design_patterns(self, tree: ast.AST) -> List[ASTPattern]:
        """Detect design patterns in the code."""
        design_patterns = []
        
        for node in ast.walk(tree):
            # Singleton pattern
            if isinstance(node, ast.ClassDef):
                if self._is_singleton_pattern(node):
                    design_patterns.append(ASTPattern(
                        "design_pattern",
                        "Singleton pattern detected",
                        node.lineno,
                        "info",
                        {"pattern": "singleton"}
                    ))
            
            # Factory pattern
            if isinstance(node, ast.FunctionDef):
                if self._is_factory_pattern(node):
                    design_patterns.append(ASTPattern(
                        "design_pattern",
                        "Factory pattern detected",
                        node.lineno,
                        "info",
                        {"pattern": "factory"}
                    ))
        
        return design_patterns
    
    def _detect_code_smells(self, tree: ast.AST) -> List[ASTPattern]:
        """Detect code smells and quality issues."""
        code_smells = []
        
        for node in ast.walk(tree):
            # Duplicate code detection (simplified)
            if isinstance(node, ast.FunctionDef):
                if self._has_duplicate_logic(node):
                    code_smells.append(ASTPattern(
                        "code_smell",
                        "Potential duplicate logic",
                        node.lineno,
                        "warning",
                        {"type": "duplicate_logic"}
                    ))
            
            # Complex expressions
            if isinstance(node, ast.BinOp):
                if self._is_complex_expression(node):
                    code_smells.append(ASTPattern(
                        "code_smell",
                        "Complex expression",
                        node.lineno,
                        "warning",
                        {"type": "complex_expression"}
                    ))
        
        return code_smells
    
    def _assess_code_quality(self, tree: ast.AST) -> Dict[str, Any]:
        """Assess overall code quality."""
        quality_score = 100
        issues = []
        
        # Analyze various quality aspects
        structure_quality = self._assess_structure_quality(tree)
        complexity_quality = self._assess_complexity_quality(tree)
        naming_quality = self._assess_naming_quality(tree)
        
        # Calculate overall score
        quality_score = (structure_quality + complexity_quality + naming_quality) / 3
        
        return {
            "overall_score": quality_score,
            "structure_quality": structure_quality,
            "complexity_quality": complexity_quality,
            "naming_quality": naming_quality,
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
    
    def _assess_naming_quality(self, tree: ast.AST) -> float:
        """Assess naming convention quality."""
        score = 100.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not self._follows_naming_convention(node.name, "function"):
                    score -= 10
            elif isinstance(node, ast.ClassDef):
                if not self._follows_naming_convention(node.name, "class"):
                    score -= 10
            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    if not self._follows_naming_convention(node.id, "variable"):
                        score -= 5
        
        return max(0.0, score)
    
    def _analyze_architecture(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code architecture and design."""
        architecture_metrics = {
            "coupling": self._calculate_coupling(tree),
            "cohesion": self._calculate_cohesion(tree),
            "abstraction_levels": self._count_abstraction_levels(tree),
            "dependency_graph": self._build_dependency_graph(tree)
        }
        
        return architecture_metrics
    
    def _calculate_coupling(self, tree: ast.AST) -> float:
        """Calculate coupling between components."""
        # Simplified coupling calculation
        imports = 0
        function_calls = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
            elif isinstance(node, ast.Call):
                function_calls += 1
        
        # Higher coupling = more dependencies
        coupling_score = min(100.0, (imports + function_calls) * 2)
        return coupling_score
    
    def _calculate_cohesion(self, tree: ast.AST) -> float:
        """Calculate cohesion within components."""
        # Simplified cohesion calculation
        cohesion_score = 100.0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                if len(methods) > 10:
                    cohesion_score -= 20  # Large classes may have low cohesion
        
        return max(0.0, cohesion_score)
    
    def _count_abstraction_levels(self, tree: ast.AST) -> int:
        """Count levels of abstraction in the code."""
        levels = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                levels = max(levels, self._get_class_depth(node))
        
        return levels
    
    def _build_dependency_graph(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Build a dependency graph of the code."""
        dependencies = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        dependencies[func_name].append(child.func.id)
        
        return dict(dependencies)
    
    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate inheritance depth of a class."""
        depth = 0
        for base in node.bases:
            if isinstance(base, ast.Name):
                depth = max(depth, 1)
        return depth
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate nesting depth of a node."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                max_depth = max(max_depth, self._calculate_nesting_depth(child, current_depth + 1))
        
        return max_depth
    
    def _is_singleton_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the singleton pattern."""
        # Simplified singleton detection
        return (
            len(node.bases) == 0 and
            any(isinstance(child, ast.FunctionDef) and child.name == "__new__" 
                for child in node.body)
        )
    
    def _is_factory_pattern(self, node: ast.FunctionDef) -> bool:
        """Check if a function implements the factory pattern."""
        # Simplified factory detection
        return (
            node.name.lower().startswith(('create', 'make', 'build')) and
            any(isinstance(child, ast.Return) and isinstance(child.value, ast.Call)
                for child in node.body)
        )
    
    def _has_duplicate_logic(self, node: ast.FunctionDef) -> bool:
        """Check if a function has duplicate logic."""
        # Simplified duplicate detection
        return len(node.body) > 20  # Long functions may have duplicate logic
    
    def _is_complex_expression(self, node: ast.BinOp) -> bool:
        """Check if an expression is complex."""
        # Simplified complexity check
        return isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
    
    def _follows_naming_convention(self, name: str, element_type: str) -> bool:
        """Check if a name follows Python naming conventions."""
        if element_type == "class":
            return name[0].isupper() and name.isalnum()
        elif element_type == "function":
            return name.islower() and (name.isalnum() or '_' in name)
        elif element_type == "variable":
            return name.islower() and (name.isalnum() or '_' in name)
        return True
    
    def _get_class_depth(self, node: ast.ClassDef, current_depth: int = 0) -> int:
        """Get the depth of a class in the inheritance hierarchy."""
        max_depth = current_depth
        
        for base in node.bases:
            if isinstance(base, ast.Name):
                max_depth = max(max_depth, current_depth + 1)
        
        return max_depth
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Performing advanced AST analysis on {len(python_files)} Python files...")
        
        # Clear previous results
        self.ast_patterns.clear()
        self.code_structures.clear()
        self.pattern_stats.clear()
        self.quality_metrics.clear()
        self.file_stats.clear()
        
        total_issues = 0
        total_quality_score = 0.0
        successful_files = 0
        
        for file_path in python_files:
            try:
                result = self.analyze_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_quality_score += result["quality_score"]
                    successful_files += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        avg_quality_score = total_quality_score / successful_files if successful_files > 0 else 0.0
        
        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_quality_score": avg_quality_score,
            "file_stats": self.file_stats
        }
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze the given file (alias for analyze_file)."""
        return self.analyze_file(file_path)