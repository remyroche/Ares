#!/usr/bin/env python3
"""Dependency utility functions for code analysis."""

from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict


class DependencyUtils:
    """Utility functions for dependency analysis."""
    
    @staticmethod
    def build_dependency_map(project_root: str) -> Dict[str, Any]:
        """Build a comprehensive dependency map."""
        return {
            'function_definitions': {},
            'function_calls': {},
            'class_definitions': {},
            'class_usage': {},
            'import_statements': {},
            'dynamic_imports': {},
            'string_references': {},
            'decorator_usage': {},
            'reflection_usage': {},
        }
    
    @staticmethod
    def detect_circular_imports(modules: Dict[str, Any]) -> List[List[str]]:
        """Detect circular import dependencies."""
        circular_imports = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(module: str, path: List[str]) -> bool:
            if module in rec_stack:
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                circular_imports.append(cycle)
                return True
            
            if module in visited:
                return False
            
            visited.add(module)
            rec_stack.add(module)
            
            if module in modules:
                for dep in modules[module].get("dependencies", []):
                    if has_cycle(dep, path + [module]):
                        return True
            
            rec_stack.remove(module)
            return False
        
        for module in modules:
            if module not in visited:
                has_cycle(module, [])
        
        return circular_imports
    
    @staticmethod
    def validate_dead_code_findings(dead_code_report, dependency_map: Dict[str, Any]) -> Any:
        """Validate dead code findings against dependency map."""
        validated_report = dead_code_report
        validated_report.false_positives_filtered = 0
        
        # Check deprecated issues
        if hasattr(dead_code_report, 'deprecated_issues') and dead_code_report.deprecated_issues:
            filtered_deprecated = []
            for issue in dead_code_report.deprecated_issues:
                if not DependencyUtils._is_false_positive(issue, dependency_map):
                    filtered_deprecated.append(issue)
                else:
                    validated_report.false_positives_filtered += 1
            dead_code_report.deprecated_issues = filtered_deprecated
        
        return validated_report
    
    @staticmethod
    def _is_false_positive(issue, dependency_map: Dict[str, Any]) -> bool:
        """Check if a dead code issue is a false positive."""
        # Extract function/class name from issue
        issue_name = DependencyUtils._extract_name_from_issue(issue)
        if not issue_name:
            return False
        
        # Check if it's defined in the dependency map
        is_defined = (
            issue_name in dependency_map['function_definitions'] or
            issue_name in dependency_map['class_definitions']
        )
        
        if not is_defined:
            return False
        
        # Check if it's used in actual code
        is_used_in_code = (
            issue_name in dependency_map['function_calls'] or
            issue_name in dependency_map['class_usage']
        )
        
        return is_used_in_code
    
    @staticmethod
    def _extract_name_from_issue(issue) -> str:
        """Extract function/class name from dead code issue."""
        if hasattr(issue, 'description'):
            import re
            patterns = [
                r"'([^']+)' is defined but never used",
                r"'([^']+)' is assigned but never used",
                r"function '([^']+)'",
                r"class '([^']+)'",
            ]
            for pattern in patterns:
                match = re.search(pattern, issue.description)
                if match:
                    return match.group(1)
        return None
