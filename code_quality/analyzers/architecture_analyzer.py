"""
Architecture Analyzer - Code architecture analysis, coupling, cohesion, and design pattern detection.
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


class ArchitectureIssue:
    """Container for architecture issue information."""
    
    def __init__(self, issue_type: str, description: str, line: int, 
                 severity: str = "warning", details: Dict[str, Any] = None):
        self.issue_type = issue_type
        self.description = description
        self.line = line
        self.severity = severity
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "line": self.line,
            "severity": self.severity,
            "details": self.details
        }


class ArchitectureMetrics:
    """Container for architecture metrics."""
    
    def __init__(self, coupling_score: float, cohesion_score: float, 
                 abstraction_levels: int, dependency_count: int):
        self.coupling_score = coupling_score
        self.cohesion_score = cohesion_score
        self.abstraction_levels = abstraction_levels
        self.dependency_count = dependency_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "coupling_score": self.coupling_score,
            "cohesion_score": self.cohesion_score,
            "abstraction_levels": self.abstraction_levels,
            "dependency_count": self.dependency_count
        }


class ArchitectureAnalyzer:
    """
    Comprehensive code architecture analysis.
    
    Features:
    - Coupling analysis between modules
    - Cohesion analysis within components
    - Abstraction level detection
    - Dependency graph analysis
    - Design pattern identification
    - Architecture violation detection
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.architecture_issues: List[ArchitectureIssue] = []
        self.architecture_metrics: Dict[str, ArchitectureMetrics] = {}
        self.dependency_graphs: Dict[str, Dict[str, List[str]]] = {}
        self.design_patterns: Dict[str, List[str]] = {}
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze architecture for a single Python file.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Clear previous results for this file
            self.architecture_issues = [issue for issue in self.architecture_issues if issue.description != str(file_path)]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform architecture analysis
            coupling_analysis = self._analyze_coupling(tree)
            cohesion_analysis = self._analyze_cohesion(tree)
            abstraction_analysis = self._analyze_abstraction_levels(tree)
            dependency_analysis = self._analyze_dependencies(tree)
            pattern_analysis = self._detect_design_patterns(tree)
            violation_analysis = self._detect_architecture_violations(tree)
            
            # Combine results
            combined_results = {
                "coupling": coupling_analysis,
                "cohesion": cohesion_analysis,
                "abstraction": abstraction_analysis,
                "dependencies": dependency_analysis,
                "patterns": pattern_analysis,
                "violations": violation_analysis
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
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        internal_dependencies.add(node.func.value.id)
        
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
                
                # Check for mixed responsibilities
                method_names = [m.name for m in methods]
                if any('get' in name for name in method_names) and any('set' in name for name in method_names):
                    if any('process' in name for name in method_names) or any('validate' in name for name in method_names):
                        cohesion_score -= 15
                        issues.append(f"Class '{node.name}' has mixed responsibilities")
            
            elif isinstance(node, ast.FunctionDef):
                # Analyze function cohesion
                if len(node.body) > 30:
                    cohesion_score -= 10
                    issues.append(f"Long function '{node.name}' with {len(node.body)} lines")
                
                # Check for multiple responsibilities
                has_io = any(isinstance(child, ast.Call) for child in node.body)
                has_logic = any(isinstance(child, (ast.If, ast.While, ast.For)) for child in node.body)
                has_data = any(isinstance(child, ast.Assign) for child in node.body)
                
                responsibility_count = sum([has_io, has_logic, has_data])
                if responsibility_count > 2:
                    cohesion_score -= 10
                    issues.append(f"Function '{node.name}' has multiple responsibilities")
        
        return {
            "cohesion_score": max(0.0, cohesion_score),
            "issues": issues,
            "total_issues": len(issues)
        }
    
    def _analyze_abstraction_levels(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze abstraction levels in the code."""
        abstraction_levels = 0
        inheritance_depth = 0
        interface_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Count inheritance levels
                current_depth = self._calculate_inheritance_depth(node)
                inheritance_depth = max(inheritance_depth, current_depth)
                
                # Check for abstract base classes
                if any('abc' in base.id.lower() for base in node.bases if isinstance(base, ast.Name)):
                    interface_count += 1
                
                # Check for abstract methods
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        if any('abstract' in decorator.id.lower() for decorator in child.decorator_list 
                               if isinstance(decorator, ast.Name)):
                            interface_count += 1
        
        abstraction_levels = inheritance_depth + interface_count
        
        return {
            "abstraction_levels": abstraction_levels,
            "inheritance_depth": inheritance_depth,
            "interface_count": interface_count
        }
    
    def _analyze_dependencies(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze dependencies and build dependency graph."""
        dependencies = defaultdict(list)
        dependency_types = defaultdict(int)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            dependencies[func_name].append(child.func.id)
                            dependency_types["function_calls"] += 1
                        elif isinstance(child.func, ast.Attribute):
                            if isinstance(child.func.value, ast.Name):
                                dependencies[func_name].append(f"{child.func.value.id}.{child.func.attr}")
                                dependency_types["method_calls"] += 1
                    
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        if child.id not in ['self', 'cls', 'True', 'False', 'None']:
                            dependencies[func_name].append(f"uses:{child.id}")
                            dependency_types["variable_usage"] += 1
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies(dependencies)
        
        return {
            "dependency_graph": dict(dependencies),
            "dependency_types": dict(dependency_types),
            "circular_dependencies": circular_deps,
            "total_dependencies": sum(len(deps) for deps in dependencies.values())
        }
    
    def _detect_design_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect common design patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Singleton pattern
                if self._is_singleton_pattern(node):
                    patterns.append("Singleton")
                
                # Factory pattern
                if self._is_factory_pattern(node):
                    patterns.append("Factory")
                
                # Observer pattern
                if self._is_observer_pattern(node):
                    patterns.append("Observer")
                
                # Strategy pattern
                if self._is_strategy_pattern(node):
                    patterns.append("Strategy")
            
            elif isinstance(node, ast.FunctionDef):
                # Template method pattern
                if self._is_template_method(node):
                    patterns.append("Template Method")
                
                # Command pattern
                if self._is_command_pattern(node):
                    patterns.append("Command")
        
        return {
            "detected_patterns": patterns,
            "pattern_count": len(patterns)
        }
    
    def _detect_architecture_violations(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect architecture violations."""
        violations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for God Object anti-pattern
                methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
                if len(methods) > 20:
                    violations.append({
                        "type": "god_object",
                        "description": f"Class '{node.name}' has too many methods ({len(methods)})",
                        "line": node.lineno,
                        "severity": "high"
                    })
                
                # Check for violation of Single Responsibility Principle
                if self._violates_single_responsibility(node):
                    violations.append({
                        "type": "single_responsibility",
                        "description": f"Class '{node.name}' violates Single Responsibility Principle",
                        "line": node.lineno,
                        "severity": "medium"
                    })
            
            elif isinstance(node, ast.FunctionDef):
                # Check for function doing too much
                if self._function_does_too_much(node):
                    violations.append({
                        "type": "function_responsibility",
                        "description": f"Function '{node.name}' does too many things",
                        "line": node.lineno,
                        "severity": "medium"
                    })
        
        return {
            "violations": violations,
            "violation_count": len(violations),
            "by_severity": {
                "high": len([v for v in violations if v["severity"] == "high"]),
                "medium": len([v for v in violations if v["severity"] == "medium"]),
                "low": len([v for v in violations if v["severity"] == "low"])
            }
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
        
        # Violation penalties
        violations = analysis_results.get("violations", {})
        high_violations = violations.get("by_severity", {}).get("high", 0)
        medium_violations = violations.get("by_severity", {}).get("medium", 0)
        
        score -= high_violations * 15
        score -= medium_violations * 8
        
        return max(0.0, score)
    
    def _calculate_inheritance_depth(self, node: ast.ClassDef) -> int:
        """Calculate inheritance depth of a class."""
        depth = 0
        for base in node.bases:
            if isinstance(base, ast.Name):
                depth = max(depth, 1)
        return depth
    
    def _detect_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in the dependency graph."""
        # Simplified circular dependency detection
        circular = []
        visited = set()
        
        def dfs(node: str, path: List[str]):
            if node in path:
                cycle_start = path.index(node)
                circular.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for dep in dependencies.get(node, []):
                if dep in dependencies:
                    dfs(dep, path.copy())
            
            path.pop()
        
        for node in dependencies:
            if node not in visited:
                dfs(node, [])
        
        return circular
    
    def _is_singleton_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the singleton pattern."""
        return (
            len(node.bases) == 0 and
            any(isinstance(child, ast.FunctionDef) and child.name == "__new__" 
                for child in node.body)
        )
    
    def _is_factory_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the factory pattern."""
        return any(
            isinstance(child, ast.FunctionDef) and 
            child.name.lower().startswith(('create', 'make', 'build'))
            for child in node.body
        )
    
    def _is_observer_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the observer pattern."""
        return any(
            isinstance(child, ast.FunctionDef) and 
            child.name.lower().startswith(('notify', 'update', 'observe'))
            for child in node.body
        )
    
    def _is_strategy_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the strategy pattern."""
        return any(
            isinstance(child, ast.FunctionDef) and 
            child.name.lower().startswith(('execute', 'algorithm', 'strategy'))
            for child in node.body
        )
    
    def _is_template_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a template method."""
        return (
            node.name.lower().startswith(('template', 'process', 'execute')) and
            any(isinstance(child, ast.Call) for child in node.body)
        )
    
    def _is_command_pattern(self, node: ast.FunctionDef) -> bool:
        """Check if a function implements the command pattern."""
        return (
            node.name.lower().startswith(('execute', 'run', 'command')) and
            len(node.args.args) == 1  # Usually just self
        )
    
    def _violates_single_responsibility(self, node: ast.ClassDef) -> bool:
        """Check if a class violates Single Responsibility Principle."""
        methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
        method_names = [m.name.lower() for m in methods]
        
        # Check for mixed responsibilities
        has_data_ops = any('get' in name or 'set' in name for name in method_names)
        has_business_logic = any('process' in name or 'calculate' in name for name in method_names)
        has_io = any('save' in name or 'load' in name for name in method_names)
        
        responsibility_count = sum([has_data_ops, has_business_logic, has_io])
        return responsibility_count > 1
    
    def _function_does_too_much(self, node: ast.FunctionDef) -> bool:
        """Check if a function does too many things."""
        # Count different types of operations
        has_io = any(isinstance(child, ast.Call) for child in node.body)
        has_logic = any(isinstance(child, (ast.If, ast.While, ast.For)) for child in node.body)
        has_data = any(isinstance(child, ast.Assign) for child in node.body)
        has_returns = any(isinstance(child, ast.Return) for child in node.body)
        
        operation_count = sum([has_io, has_logic, has_data, has_returns])
        return operation_count > 3 or len(node.body) > 25
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze architecture for all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing architecture for {len(python_files)} Python files...")
        
        # Clear previous results
        self.architecture_issues.clear()
        self.architecture_metrics.clear()
        self.dependency_graphs.clear()
        self.design_patterns.clear()
        self.file_stats.clear()
        
        total_issues = 0
        total_architecture_score = 0.0
        successful_files = 0
        
        for file_path in python_files:
            try:
                result = self.analyze_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_architecture_score += result["architecture_score"]
                    successful_files += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        avg_architecture_score = total_architecture_score / successful_files if successful_files > 0 else 0.0
        
        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_architecture_score": avg_architecture_score,
            "file_stats": self.file_stats
        }
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze the given file (alias for analyze_file)."""
        return self.analyze_file(file_path)