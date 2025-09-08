#!/usr/bin/env python3
"""
Pattern Analyzer

Analyzes usage patterns in Python codebases to identify:
- Common design patterns (Singleton, Factory, Observer, etc.)
- Framework-specific patterns (Django models, Flask routes, etc.)
- Anti-patterns and code smells
- Usage frequency and importance patterns
- Call patterns and dependencies
"""

import ast
import json
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging


@dataclass
class PatternMatch:
    """Represents a pattern match in the code."""
    pattern_name: str
    pattern_type: str
    file_path: str
    line_number: int
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    related_elements: List[str] = field(default_factory=list)


@dataclass
class UsagePattern:
    """Represents a usage pattern analysis result."""
    element_name: str
    element_type: str  # function, class, method, variable
    file_path: str
    usage_count: int
    usage_locations: List[Tuple[str, int]] = field(default_factory=list)
    import_sources: List[str] = field(default_factory=list)
    call_patterns: List[str] = field(default_factory=list)
    importance_score: float = 0.0
    is_public_api: bool = False
    is_framework_hook: bool = False


@dataclass
class PatternAnalysisResult:
    """Complete pattern analysis result."""
    timestamp: str
    project_root: str
    design_patterns: List[PatternMatch] = field(default_factory=list)
    framework_patterns: List[PatternMatch] = field(default_factory=list)
    anti_patterns: List[PatternMatch] = field(default_factory=list)
    usage_patterns: List[UsagePattern] = field(default_factory=list)
    pattern_statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PatternAnalyzer:
    """Analyzes patterns in Python codebases."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.design_patterns = self._initialize_design_patterns()
        self.framework_patterns = self._initialize_framework_patterns()
        self.anti_patterns = self._initialize_anti_patterns()
        self.usage_tracker = defaultdict(list)
        self.import_graph = defaultdict(set)
        self.call_graph = defaultdict(set)
    
    def analyze_patterns(self, project_root: Path, framework_context: Optional[Dict[str, Any]] = None) -> PatternAnalysisResult:
        """Perform comprehensive pattern analysis."""
        from datetime import datetime
        
        self.logger.info(f"Analyzing patterns in {project_root}")
        
        result = PatternAnalysisResult(
            timestamp=datetime.now().isoformat(),
            project_root=str(project_root)
        )
        
        # Analyze all Python files
        for py_file in project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Analyze different pattern types
                file_patterns = self._analyze_file_patterns(tree, py_file, framework_context)
                result.design_patterns.extend(file_patterns["design"])
                result.framework_patterns.extend(file_patterns["framework"])
                result.anti_patterns.extend(file_patterns["anti"])
                
                # Track usage patterns
                usage_patterns = self._analyze_usage_patterns(tree, py_file)
                result.usage_patterns.extend(usage_patterns)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Calculate statistics and recommendations
        result.pattern_statistics = self._calculate_pattern_statistics(result)
        result.recommendations = self._generate_recommendations(result)
        
        self.logger.info(f"Pattern analysis completed: {len(result.design_patterns)} design patterns, {len(result.framework_patterns)} framework patterns")
        
        return result
    
    def _initialize_design_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize design pattern detection rules."""
        return {
            "singleton": {
                "class_patterns": ["_instance", "__new__", "getInstance"],
                "method_patterns": ["if.*_instance.*is None", "cls\\._instance"],
                "confidence": 0.8
            },
            "factory": {
                "function_patterns": ["create_", "make_", "build_", "get_"],
                "class_patterns": ["Factory", "Builder", "Creator"],
                "confidence": 0.7
            },
            "observer": {
                "method_patterns": ["notify", "update", "subscribe", "unsubscribe"],
                "class_patterns": ["Observer", "Subject", "Listener"],
                "confidence": 0.8
            },
            "decorator": {
                "function_patterns": ["@", "wraps", "functools"],
                "class_patterns": ["__call__", "wrapper"],
                "confidence": 0.9
            },
            "strategy": {
                "class_patterns": ["Strategy", "Algorithm", "Policy"],
                "method_patterns": ["execute", "run", "process"],
                "confidence": 0.7
            },
            "command": {
                "class_patterns": ["Command", "Action", "Operation"],
                "method_patterns": ["execute", "undo", "redo"],
                "confidence": 0.8
            },
            "adapter": {
                "class_patterns": ["Adapter", "Wrapper", "Translator"],
                "method_patterns": ["adapt", "convert", "transform"],
                "confidence": 0.7
            },
            "facade": {
                "class_patterns": ["Facade", "Manager", "Controller"],
                "method_patterns": ["simplify", "unify", "coordinate"],
                "confidence": 0.6
            }
        }
    
    def _initialize_framework_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework-specific pattern detection rules."""
        return {
            "django": {
                "model_patterns": ["class.*Model", "models\\.Model", "db_table"],
                "view_patterns": ["class.*View", "def.*view", "render", "HttpResponse"],
                "url_patterns": ["urlpatterns", "path\\(", "re_path\\("],
                "admin_patterns": ["admin\\.site", "ModelAdmin", "list_display"],
                "form_patterns": ["forms\\.Form", "forms\\.ModelForm", "clean_"],
                "middleware_patterns": ["process_request", "process_response"],
                "signal_patterns": ["@receiver", "post_save", "pre_delete"]
            },
            "flask": {
                "route_patterns": ["@app\\.route", "@blueprint\\.route", "def.*\\("],
                "blueprint_patterns": ["Blueprint\\(", "register_blueprint"],
                "template_patterns": ["render_template", "Jinja2"],
                "request_patterns": ["request\\.", "g\\.", "session\\."],
                "error_patterns": ["@app\\.errorhandler", "abort\\("],
                "context_patterns": ["@app\\.before_request", "@app\\.after_request"]
            },
            "fastapi": {
                "router_patterns": ["APIRouter", "@router\\."],
                "dependency_patterns": ["Depends\\(", "def.*\\("],
                "model_patterns": ["BaseModel", "Field\\("],
                "response_patterns": ["Response", "JSONResponse"],
                "middleware_patterns": ["add_middleware", "BaseHTTPMiddleware"]
            },
            "pytest": {
                "fixture_patterns": ["@pytest\\.fixture", "def.*fixture"],
                "test_patterns": ["def test_", "assert "],
                "parametrize_patterns": ["@pytest\\.mark\\.parametrize"],
                "mock_patterns": ["@patch", "Mock\\(", "MagicMock"]
            }
        }
    
    def _initialize_anti_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize anti-pattern detection rules."""
        return {
            "god_object": {
                "class_patterns": ["class.*:"],
                "method_count_threshold": 20,
                "line_count_threshold": 500,
                "confidence": 0.8
            },
            "long_parameter_list": {
                "method_patterns": ["def.*\\("],
                "parameter_count_threshold": 7,
                "confidence": 0.9
            },
            "duplicate_code": {
                "similarity_threshold": 0.8,
                "confidence": 0.7
            },
            "dead_code": {
                "unused_imports": True,
                "unused_variables": True,
                "unreachable_code": True,
                "confidence": 0.9
            },
            "circular_import": {
                "import_patterns": ["import.*", "from.*import"],
                "confidence": 0.8
            },
            "magic_numbers": {
                "number_patterns": ["\\b\\d+\\b"],
                "confidence": 0.6
            },
            "deep_nesting": {
                "nesting_threshold": 4,
                "confidence": 0.7
            }
        }
    
    def _analyze_file_patterns(self, tree: ast.AST, file_path: Path, framework_context: Optional[Dict[str, Any]]) -> Dict[str, List[PatternMatch]]:
        """Analyze patterns in a single file."""
        patterns = {
            "design": [],
            "framework": [],
            "anti": []
        }
        
        # Analyze design patterns
        patterns["design"] = self._detect_design_patterns(tree, file_path)
        
        # Analyze framework patterns
        if framework_context:
            patterns["framework"] = self._detect_framework_patterns(tree, file_path, framework_context)
        
        # Analyze anti-patterns
        patterns["anti"] = self._detect_anti_patterns(tree, file_path)
        
        return patterns
    
    def _detect_design_patterns(self, tree: ast.AST, file_path: Path) -> List[PatternMatch]:
        """Detect design patterns in the AST."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for Singleton pattern
                if self._is_singleton_pattern(node):
                    patterns.append(PatternMatch(
                        pattern_name="singleton",
                        pattern_type="design",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.8,
                        context={"class_name": node.name}
                    ))
                
                # Check for Factory pattern
                if self._is_factory_pattern(node):
                    patterns.append(PatternMatch(
                        pattern_name="factory",
                        pattern_type="design",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.7,
                        context={"class_name": node.name}
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                # Check for Decorator pattern
                if self._is_decorator_pattern(node):
                    patterns.append(PatternMatch(
                        pattern_name="decorator",
                        pattern_type="design",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.9,
                        context={"function_name": node.name}
                    ))
        
        return patterns
    
    def _detect_framework_patterns(self, tree: ast.AST, file_path: Path, framework_context: Dict[str, Any]) -> List[PatternMatch]:
        """Detect framework-specific patterns."""
        patterns = []
        
        # Get primary framework
        primary_framework = framework_context.get("primary_framework", {})
        framework_name = primary_framework.get("framework_name", "")
        
        if framework_name in self.framework_patterns:
            framework_rules = self.framework_patterns[framework_name]
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check for framework-specific class patterns
                    for pattern_name, rules in framework_rules.items():
                        if self._matches_framework_pattern(node, rules):
                            patterns.append(PatternMatch(
                                pattern_name=f"{framework_name}_{pattern_name}",
                                pattern_type="framework",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                confidence=0.8,
                                context={"class_name": node.name, "framework": framework_name}
                            ))
                
                elif isinstance(node, ast.FunctionDef):
                    # Check for framework-specific function patterns
                    for pattern_name, rules in framework_rules.items():
                        if self._matches_framework_pattern(node, rules):
                            patterns.append(PatternMatch(
                                pattern_name=f"{framework_name}_{pattern_name}",
                                pattern_type="framework",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                confidence=0.7,
                                context={"function_name": node.name, "framework": framework_name}
                            ))
        
        return patterns
    
    def _detect_anti_patterns(self, tree: ast.AST, file_path: Path) -> List[PatternMatch]:
        """Detect anti-patterns in the code."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for God Object anti-pattern
                if self._is_god_object(node):
                    patterns.append(PatternMatch(
                        pattern_name="god_object",
                        pattern_type="anti",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.8,
                        context={"class_name": node.name, "method_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)])}
                    ))
            
            elif isinstance(node, ast.FunctionDef):
                # Check for Long Parameter List anti-pattern
                if self._is_long_parameter_list(node):
                    patterns.append(PatternMatch(
                        pattern_name="long_parameter_list",
                        pattern_type="anti",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.9,
                        context={"function_name": node.name, "parameter_count": len(node.args.args)}
                    ))
                
                # Check for Deep Nesting anti-pattern
                if self._is_deep_nesting(node):
                    patterns.append(PatternMatch(
                        pattern_name="deep_nesting",
                        pattern_type="anti",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        confidence=0.7,
                        context={"function_name": node.name}
                    ))
        
        return patterns
    
    def _analyze_usage_patterns(self, tree: ast.AST, file_path: Path) -> List[UsagePattern]:
        """Analyze usage patterns for functions, classes, and methods."""
        usage_patterns = []
        
        # Track function definitions and their usage
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                usage_pattern = self._create_usage_pattern(node, file_path, "function")
                if usage_pattern:
                    usage_patterns.append(usage_pattern)
            
            elif isinstance(node, ast.ClassDef):
                usage_pattern = self._create_usage_pattern(node, file_path, "class")
                if usage_pattern:
                    usage_patterns.append(usage_pattern)
        
        return usage_patterns
    
    def _create_usage_pattern(self, node: Union[ast.FunctionDef, ast.ClassDef], file_path: Path, element_type: str) -> Optional[UsagePattern]:
        """Create a usage pattern for a function or class."""
        # This is a simplified implementation
        # In a full implementation, you'd track actual usage across the codebase
        
        importance_score = self._calculate_importance_score(node, element_type)
        
        return UsagePattern(
            element_name=node.name,
            element_type=element_type,
            file_path=str(file_path),
            usage_count=0,  # Would be calculated from actual usage tracking
            importance_score=importance_score,
            is_public_api=self._is_public_api(node),
            is_framework_hook=self._is_framework_hook(node)
        )
    
    def _calculate_importance_score(self, node: Union[ast.FunctionDef, ast.ClassDef], element_type: str) -> float:
        """Calculate importance score for an element."""
        score = 0.0
        
        # Public API elements are more important
        if self._is_public_api(node):
            score += 0.5
        
        # Framework hooks are important
        if self._is_framework_hook(node):
            score += 0.3
        
        # Elements with documentation are more important
        if self._has_documentation(node):
            score += 0.2
        
        # Size-based scoring
        if element_type == "function":
            if len(node.body) > 10:
                score += 0.1
        elif element_type == "class":
            method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
            if method_count > 5:
                score += 0.1
        
        return min(score, 1.0)
    
    def _is_public_api(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        """Check if an element is part of the public API."""
        # Not starting with underscore
        if not node.name.startswith("_"):
            return True
        
        # Special methods (__init__, __str__, etc.) are considered public
        if node.name.startswith("__") and node.name.endswith("__"):
            return True
        
        return False
    
    def _is_framework_hook(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        """Check if an element is a framework hook."""
        framework_hooks = {
            "__init__", "__call__", "setup", "teardown", "process_request",
            "process_response", "before_request", "after_request", "get",
            "post", "put", "delete", "patch", "head", "options"
        }
        
        return node.name in framework_hooks
    
    def _has_documentation(self, node: Union[ast.FunctionDef, ast.ClassDef]) -> bool:
        """Check if a node has documentation."""
        if not node.body:
            return False
        
        first_stmt = node.body[0]
        return isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant)
    
    # Pattern detection helper methods
    def _is_singleton_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the Singleton pattern."""
        has_instance_var = False
        has_new_method = False
        
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id == "_instance":
                        has_instance_var = True
            
            elif isinstance(child, ast.FunctionDef) and child.name == "__new__":
                has_new_method = True
        
        return has_instance_var and has_new_method
    
    def _is_factory_pattern(self, node: ast.ClassDef) -> bool:
        """Check if a class implements the Factory pattern."""
        factory_keywords = ["Factory", "Builder", "Creator", "Maker"]
        return any(keyword in node.name for keyword in factory_keywords)
    
    def _is_decorator_pattern(self, node: ast.FunctionDef) -> bool:
        """Check if a function implements the Decorator pattern."""
        # Look for functools.wraps usage
        for child in node.body:
            if isinstance(child, ast.ImportFrom):
                if child.module == "functools" and any(alias.name == "wraps" for alias in child.names):
                    return True
        
        return False
    
    def _matches_framework_pattern(self, node: Union[ast.ClassDef, ast.FunctionDef], rules: List[str]) -> bool:
        """Check if a node matches framework pattern rules."""
        # Simplified pattern matching
        node_name = node.name.lower()
        return any(pattern.lower() in node_name for pattern in rules)
    
    def _is_god_object(self, node: ast.ClassDef) -> bool:
        """Check if a class is a God Object anti-pattern."""
        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
        return method_count > 20  # Threshold for God Object
    
    def _is_long_parameter_list(self, node: ast.FunctionDef) -> bool:
        """Check if a function has too many parameters."""
        return len(node.args.args) > 7  # Threshold for long parameter list
    
    def _is_deep_nesting(self, node: ast.FunctionDef) -> bool:
        """Check if a function has deep nesting."""
        max_depth = self._calculate_nesting_depth(node)
        return max_depth > 4  # Threshold for deep nesting
    
    def _calculate_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in an AST node."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
                child_depth = self._calculate_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _calculate_pattern_statistics(self, result: PatternAnalysisResult) -> Dict[str, Any]:
        """Calculate statistics from pattern analysis results."""
        return {
            "total_design_patterns": len(result.design_patterns),
            "total_framework_patterns": len(result.framework_patterns),
            "total_anti_patterns": len(result.anti_patterns),
            "total_usage_patterns": len(result.usage_patterns),
            "pattern_distribution": self._calculate_pattern_distribution(result),
            "anti_pattern_severity": self._calculate_anti_pattern_severity(result)
        }
    
    def _calculate_pattern_distribution(self, result: PatternAnalysisResult) -> Dict[str, int]:
        """Calculate distribution of patterns."""
        distribution = defaultdict(int)
        
        for pattern in result.design_patterns:
            distribution[f"design_{pattern.pattern_name}"] += 1
        
        for pattern in result.framework_patterns:
            distribution[f"framework_{pattern.pattern_name}"] += 1
        
        for pattern in result.anti_patterns:
            distribution[f"anti_{pattern.pattern_name}"] += 1
        
        return dict(distribution)
    
    def _calculate_anti_pattern_severity(self, result: PatternAnalysisResult) -> Dict[str, float]:
        """Calculate severity of anti-patterns."""
        severity = defaultdict(list)
        
        for pattern in result.anti_patterns:
            severity[pattern.pattern_name].append(pattern.confidence)
        
        return {
            pattern_name: sum(confidences) / len(confidences)
            for pattern_name, confidences in severity.items()
        }
    
    def _generate_recommendations(self, result: PatternAnalysisResult) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        # Anti-pattern recommendations
        anti_pattern_counts = Counter(pattern.pattern_name for pattern in result.anti_patterns)
        
        if anti_pattern_counts.get("god_object", 0) > 0:
            recommendations.append("Consider breaking down God Objects into smaller, focused classes")
        
        if anti_pattern_counts.get("long_parameter_list", 0) > 0:
            recommendations.append("Refactor functions with long parameter lists using data classes or configuration objects")
        
        if anti_pattern_counts.get("deep_nesting", 0) > 0:
            recommendations.append("Reduce nesting depth by extracting methods or using early returns")
        
        # Design pattern recommendations
        design_pattern_counts = Counter(pattern.pattern_name for pattern in result.design_patterns)
        
        if design_pattern_counts.get("singleton", 0) > 3:
            recommendations.append("Consider if all Singleton patterns are necessary - they can make testing difficult")
        
        # Framework pattern recommendations
        if result.framework_patterns:
            recommendations.append("Ensure framework patterns follow best practices for the detected framework")
        
        return recommendations


def main():
    """Main entry point for testing the pattern analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pattern Analyzer")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = PatternAnalyzer()
    
    # Analyze patterns
    project_root = Path(args.project_root)
    result = analyzer.analyze_patterns(project_root)
    
    # Print results
    print(f"\nPattern Analysis Results for {project_root}:")
    print(f"Design patterns: {len(result.design_patterns)}")
    print(f"Framework patterns: {len(result.framework_patterns)}")
    print(f"Anti-patterns: {len(result.anti_patterns)}")
    print(f"Usage patterns: {len(result.usage_patterns)}")
    
    if result.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result.__dict__, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()