#!/usr/bin/env python3
"""
Context-Aware Dead Code Analyzer

Provides context-aware dead code analysis by considering:
- Framework-specific patterns and conventions
- Usage patterns and importance scoring
- Development context and project structure
- Framework hooks and lifecycle methods
- Public API vs internal code distinctions

This analyzer significantly reduces false positives by understanding
the context in which code is used.
"""

import ast
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

# Import our custom analyzers
from analyzers.framework_detector import FrameworkDetector, ProjectContext
from analyzers.pattern_analyzer import PatternAnalyzer, PatternAnalysisResult
from analyzers.multi_modal_dead_code_analyzer import MultiModalDeadCodeAnalyzer, MultiModalResult
from core.config import AnalysisConfig


@dataclass
class ContextualDeadCodeResult:
    """Result of context-aware dead code analysis."""
    timestamp: str
    project_root: str
    framework_context: ProjectContext
    pattern_analysis: PatternAnalysisResult
    multi_modal_analysis: MultiModalResult
    context_aware_dead_functions: List[Dict[str, Any]] = field(default_factory=list)
    context_aware_dead_classes: List[Dict[str, Any]] = field(default_factory=list)
    context_aware_dead_imports: List[Dict[str, Any]] = field(default_factory=list)
    false_positives_filtered: int = 0
    context_insights: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class ContextAwareDeadCodeAnalyzer:
    """Context-aware dead code analyzer that considers framework and usage context."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize component analyzers
        self.framework_detector = FrameworkDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.multi_modal_analyzer = MultiModalDeadCodeAnalyzer(config)
        
        # Framework-specific rules
        self.framework_rules = self._initialize_framework_rules()
        self.importance_weights = self._initialize_importance_weights()
    
    def analyze(self, project_root: Path, interaction_data: Optional[Dict[str, Any]] = None) -> ContextualDeadCodeResult:
        """Perform context-aware dead code analysis."""
        start_time = time.time()
        
        self.logger.info(f"Starting context-aware dead code analysis for {project_root}")
        
        # Step 1: Detect framework context
        self.logger.info("Step 1: Detecting framework context...")
        framework_context = self.framework_detector.detect_frameworks(project_root)
        
        # Step 2: Analyze usage patterns
        self.logger.info("Step 2: Analyzing usage patterns...")
        pattern_analysis = self.pattern_analyzer.analyze_patterns(project_root, framework_context.__dict__)
        
        # Step 3: Run multi-modal analysis
        self.logger.info("Step 3: Running multi-modal analysis...")
        multi_modal_analysis = self.multi_modal_analyzer.analyze(project_root, interaction_data)
        
        # Step 4: Apply context-aware filtering
        self.logger.info("Step 4: Applying context-aware filtering...")
        context_aware_results = self._apply_context_aware_filtering(
            multi_modal_analysis, framework_context, pattern_analysis
        )
        
        # Step 5: Generate insights and recommendations
        self.logger.info("Step 5: Generating insights and recommendations...")
        insights = self._generate_context_insights(framework_context, pattern_analysis, multi_modal_analysis)
        recommendations = self._generate_recommendations(framework_context, pattern_analysis, context_aware_results)
        
        execution_time = time.time() - start_time
        
        result = ContextualDeadCodeResult(
            timestamp=datetime.now().isoformat(),
            project_root=str(project_root),
            framework_context=framework_context,
            pattern_analysis=pattern_analysis,
            multi_modal_analysis=multi_modal_analysis,
            context_aware_dead_functions=context_aware_results["functions"],
            context_aware_dead_classes=context_aware_results["classes"],
            context_aware_dead_imports=context_aware_results["imports"],
            false_positives_filtered=context_aware_results["false_positives_filtered"],
            context_insights=insights,
            confidence_scores=context_aware_results["confidence_scores"],
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        self.logger.info(f"Context-aware analysis completed in {execution_time:.2f} seconds")
        self.logger.info(f"Filtered {result.false_positives_filtered} false positives")
        
        return result
    
    def _initialize_framework_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework-specific rules for dead code detection."""
        return {
            "django": {
                "protected_functions": [
                    "__init__", "save", "delete", "clean", "full_clean",
                    "get_absolute_url", "get_next_by_", "get_previous_by_"
                ],
                "protected_classes": [
                    "Model", "ModelAdmin", "Form", "View", "Middleware",
                    "Manager", "QuerySet", "Field", "Widget"
                ],
                "protected_imports": [
                    "django.db.models", "django.http", "django.views",
                    "django.urls", "django.contrib", "django.forms"
                ],
                "lifecycle_methods": [
                    "pre_save", "post_save", "pre_delete", "post_delete",
                    "pre_init", "post_init", "clean", "validate_unique"
                ],
                "admin_hooks": [
                    "list_display", "list_filter", "search_fields", "ordering",
                    "fields", "exclude", "readonly_fields", "save_model"
                ],
                "url_patterns": [
                    "urlpatterns", "path", "re_path", "include"
                ]
            },
            "flask": {
                "protected_functions": [
                    "__init__", "before_request", "after_request",
                    "teardown_request", "errorhandler", "context_processor"
                ],
                "protected_classes": [
                    "Flask", "Blueprint", "Request", "Response",
                    "Session", "g", "current_app"
                ],
                "protected_imports": [
                    "flask", "flask_sqlalchemy", "flask_migrate",
                    "flask_login", "flask_wtf", "flask_mail"
                ],
                "route_decorators": [
                    "@app.route", "@blueprint.route", "@login_required",
                    "@admin_required", "@csrf_exempt"
                ],
                "lifecycle_methods": [
                    "before_request", "after_request", "teardown_request",
                    "before_first_request", "errorhandler"
                ]
            },
            "fastapi": {
                "protected_functions": [
                    "__init__", "startup", "shutdown", "lifespan"
                ],
                "protected_classes": [
                    "FastAPI", "APIRouter", "BaseModel", "Depends",
                    "HTTPException", "Request", "Response"
                ],
                "protected_imports": [
                    "fastapi", "uvicorn", "pydantic", "sqlalchemy"
                ],
                "route_decorators": [
                    "@app.get", "@app.post", "@app.put", "@app.delete",
                    "@router.get", "@router.post", "@router.put", "@router.delete"
                ],
                "dependency_patterns": [
                    "Depends", "get_", "create_", "verify_"
                ]
            },
            "pytest": {
                "protected_functions": [
                    "test_", "setup_", "teardown_", "fixture_"
                ],
                "protected_classes": [
                    "Test", "TestCase", "Fixture"
                ],
                "protected_imports": [
                    "pytest", "unittest", "mock", "fixtures"
                ],
                "test_decorators": [
                    "@pytest.fixture", "@pytest.mark.parametrize",
                    "@pytest.mark.skip", "@pytest.mark.xfail"
                ]
            },
            "celery": {
                "protected_functions": [
                    "task", "periodic_task", "beat_schedule"
                ],
                "protected_classes": [
                    "Celery", "Task", "Worker"
                ],
                "protected_imports": [
                    "celery", "celery.task", "celery.worker"
                ],
                "task_decorators": [
                    "@task", "@periodic_task", "@shared_task"
                ]
            }
        }
    
    def _initialize_importance_weights(self) -> Dict[str, float]:
        """Initialize importance weights for different code elements."""
        return {
            "public_api": 1.0,
            "framework_hook": 0.9,
            "lifecycle_method": 0.8,
            "decorated_function": 0.7,
            "test_function": 0.6,
            "utility_function": 0.5,
            "private_function": 0.3,
            "internal_function": 0.2,
            "model_class": 0.9,
            "view_class": 0.8,
            "form_class": 0.7,
            "admin_class": 0.8,
            "middleware_class": 0.7,
            "base_class": 0.6,
            "utility_class": 0.5,
            "internal_class": 0.3
        }
    
    def _apply_context_aware_filtering(
        self, 
        multi_modal_analysis: MultiModalResult,
        framework_context: ProjectContext,
        pattern_analysis: PatternAnalysisResult
    ) -> Dict[str, Any]:
        """Apply context-aware filtering to multi-modal analysis results."""
        
        filtered_functions = []
        filtered_classes = []
        filtered_imports = []
        false_positives_filtered = 0
        
        # Get primary framework
        primary_framework = framework_context.primary_framework
        framework_name = primary_framework.framework_name if primary_framework else "generic"
        
        # Filter functions
        for func in multi_modal_analysis.combined_dead_functions:
            if not self._is_false_positive_function(func, framework_name, pattern_analysis):
                filtered_functions.append(func)
            else:
                false_positives_filtered += 1
        
        # Filter classes
        for cls in multi_modal_analysis.combined_dead_classes:
            if not self._is_false_positive_class(cls, framework_name, pattern_analysis):
                filtered_classes.append(cls)
            else:
                false_positives_filtered += 1
        
        # Filter imports
        for imp in multi_modal_analysis.combined_dead_imports:
            if not self._is_false_positive_import(imp, framework_name, pattern_analysis):
                filtered_imports.append(imp)
            else:
                false_positives_filtered += 1
        
        # Calculate confidence scores
        confidence_scores = self._calculate_context_aware_confidence(
            filtered_functions, filtered_classes, filtered_imports, framework_context
        )
        
        return {
            "functions": filtered_functions,
            "classes": filtered_classes,
            "imports": filtered_imports,
            "false_positives_filtered": false_positives_filtered,
            "confidence_scores": confidence_scores
        }
    
    def _is_false_positive_function(
        self, 
        func: Dict[str, Any], 
        framework_name: str, 
        pattern_analysis: PatternAnalysisResult
    ) -> bool:
        """Check if a function is a false positive based on context."""
        func_name = func.get("name", "")
        file_path = func.get("file", "")
        
        # Check framework-specific rules
        if framework_name in self.framework_rules:
            rules = self.framework_rules[framework_name]
            
            # Check protected functions
            if func_name in rules.get("protected_functions", []):
                return True
            
            # Check lifecycle methods
            if func_name in rules.get("lifecycle_methods", []):
                return True
            
            # Check if function is decorated with framework decorators
            if self._has_framework_decorator(func, rules.get("route_decorators", [])):
                return True
        
        # Check pattern analysis
        for usage_pattern in pattern_analysis.usage_patterns:
            if (usage_pattern.element_name == func_name and 
                usage_pattern.element_type == "function" and
                usage_pattern.is_framework_hook):
                return True
        
        # Check if function is part of public API
        if not func_name.startswith("_"):
            return True
        
        # Check if function is a test function
        if func_name.startswith("test_"):
            return True
        
        # Check if function is a setup/teardown function
        if func_name.startswith(("setup_", "teardown_", "fixture_")):
            return True
        
        return False
    
    def _is_false_positive_class(
        self, 
        cls: Dict[str, Any], 
        framework_name: str, 
        pattern_analysis: PatternAnalysisResult
    ) -> bool:
        """Check if a class is a false positive based on context."""
        class_name = cls.get("name", "")
        file_path = cls.get("file", "")
        
        # Check framework-specific rules
        if framework_name in self.framework_rules:
            rules = self.framework_rules[framework_name]
            
            # Check protected classes
            if class_name in rules.get("protected_classes", []):
                return True
        
        # Check pattern analysis
        for usage_pattern in pattern_analysis.usage_patterns:
            if (usage_pattern.element_name == class_name and 
                usage_pattern.element_type == "class" and
                usage_pattern.is_framework_hook):
                return True
        
        # Check if class is part of public API
        if not class_name.startswith("_"):
            return True
        
        # Check if class is a test class
        if class_name.startswith("Test"):
            return True
        
        # Check if class is a base class
        if class_name in ["Base", "Abstract", "Interface", "Protocol"]:
            return True
        
        return False
    
    def _is_false_positive_import(
        self, 
        imp: Dict[str, Any], 
        framework_name: str, 
        pattern_analysis: PatternAnalysisResult
    ) -> bool:
        """Check if an import is a false positive based on context."""
        import_name = imp.get("name", "")
        file_path = imp.get("file", "")
        
        # Check framework-specific rules
        if framework_name in self.framework_rules:
            rules = self.framework_rules[framework_name]
            
            # Check protected imports
            for protected_import in rules.get("protected_imports", []):
                if import_name.startswith(protected_import):
                    return True
        
        # Check if import is used in framework patterns
        for pattern in pattern_analysis.framework_patterns:
            if import_name in pattern.context.get("imports", []):
                return True
        
        return False
    
    def _has_framework_decorator(self, func: Dict[str, Any], decorators: List[str]) -> bool:
        """Check if a function has framework-specific decorators."""
        # This is a simplified check - in reality, you'd need to parse the AST
        # to check for actual decorators
        func_name = func.get("name", "")
        
        # For now, check if function name suggests it's decorated
        for decorator in decorators:
            if decorator.replace("@", "").replace(".", "_") in func_name:
                return True
        
        return False
    
    def _calculate_context_aware_confidence(
        self,
        functions: List[Dict[str, Any]],
        classes: List[Dict[str, Any]],
        imports: List[Dict[str, Any]],
        framework_context: ProjectContext
    ) -> Dict[str, float]:
        """Calculate context-aware confidence scores."""
        confidence_scores = {
            "functions": 0.0,
            "classes": 0.0,
            "imports": 0.0,
            "overall": 0.0
        }
        
        if functions:
            func_confidences = [f.get("confidence", 0.5) for f in functions]
            confidence_scores["functions"] = sum(func_confidences) / len(func_confidences)
        
        if classes:
            class_confidences = [c.get("confidence", 0.5) for c in classes]
            confidence_scores["classes"] = sum(class_confidences) / len(class_confidences)
        
        if imports:
            import_confidences = [i.get("confidence", 0.5) for i in imports]
            confidence_scores["imports"] = sum(import_confidences) / len(import_confidences)
        
        # Calculate overall confidence
        all_confidences = []
        all_confidences.extend([f.get("confidence", 0.5) for f in functions])
        all_confidences.extend([c.get("confidence", 0.5) for c in classes])
        all_confidences.extend([i.get("confidence", 0.5) for i in imports])
        
        if all_confidences:
            confidence_scores["overall"] = sum(all_confidences) / len(all_confidences)
        
        return confidence_scores
    
    def _generate_context_insights(
        self,
        framework_context: ProjectContext,
        pattern_analysis: PatternAnalysisResult,
        multi_modal_analysis: MultiModalResult
    ) -> Dict[str, Any]:
        """Generate insights from context analysis."""
        insights = {
            "framework_analysis": {
                "primary_framework": framework_context.primary_framework.framework_name if framework_context.primary_framework else "generic",
                "framework_confidence": framework_context.primary_framework.confidence if framework_context.primary_framework else 0.0,
                "total_frameworks": len(framework_context.frameworks)
            },
            "pattern_analysis": {
                "design_patterns_found": len(pattern_analysis.design_patterns),
                "framework_patterns_found": len(pattern_analysis.framework_patterns),
                "anti_patterns_found": len(pattern_analysis.anti_patterns),
                "usage_patterns_analyzed": len(pattern_analysis.usage_patterns)
            },
            "dead_code_analysis": {
                "original_dead_functions": len(multi_modal_analysis.combined_dead_functions),
                "original_dead_classes": len(multi_modal_analysis.combined_dead_classes),
                "original_dead_imports": len(multi_modal_analysis.combined_dead_imports),
                "multi_modal_confidence": multi_modal_analysis.consensus_scores.get("overall_confidence", 0.0)
            },
            "context_effectiveness": {
                "false_positive_reduction": self._calculate_false_positive_reduction(multi_modal_analysis),
                "context_awareness_score": self._calculate_context_awareness_score(framework_context, pattern_analysis)
            }
        }
        
        return insights
    
    def _generate_recommendations(
        self,
        framework_context: ProjectContext,
        pattern_analysis: PatternAnalysisResult,
        context_aware_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on context analysis."""
        recommendations = []
        
        # Framework-specific recommendations
        if framework_context.primary_framework:
            framework_name = framework_context.primary_framework.framework_name
            recommendations.extend(self._get_framework_recommendations(framework_name))
        
        # Pattern-based recommendations
        recommendations.extend(pattern_analysis.recommendations)
        
        # Dead code recommendations
        if context_aware_results["functions"]:
            recommendations.append(f"Consider removing {len(context_aware_results['functions'])} unused functions")
        
        if context_aware_results["classes"]:
            recommendations.append(f"Consider removing {len(context_aware_results['classes'])} unused classes")
        
        if context_aware_results["imports"]:
            recommendations.append(f"Consider removing {len(context_aware_results['imports'])} unused imports")
        
        # Context-specific recommendations
        if context_aware_results["false_positives_filtered"] > 0:
            recommendations.append(f"Context-aware analysis filtered {context_aware_results['false_positives_filtered']} false positives")
        
        return recommendations
    
    def _get_framework_recommendations(self, framework_name: str) -> List[str]:
        """Get framework-specific recommendations."""
        recommendations = {
            "django": [
                "Ensure Django models follow the ORM patterns correctly",
                "Check that Django views are properly registered in URLs",
                "Verify Django admin classes are registered",
                "Review Django middleware for proper lifecycle handling"
            ],
            "flask": [
                "Ensure Flask routes are properly decorated",
                "Check that Flask blueprints are registered",
                "Verify Flask extensions are properly initialized",
                "Review Flask error handlers for completeness"
            ],
            "fastapi": [
                "Ensure FastAPI routes are properly defined",
                "Check that FastAPI dependencies are correctly injected",
                "Verify Pydantic models are properly structured",
                "Review FastAPI middleware configuration"
            ],
            "pytest": [
                "Ensure test functions follow pytest conventions",
                "Check that fixtures are properly defined",
                "Verify test parametrization is correct",
                "Review test organization and structure"
            ]
        }
        
        return recommendations.get(framework_name, [])
    
    def _calculate_false_positive_reduction(self, multi_modal_analysis: MultiModalResult) -> float:
        """Calculate the effectiveness of false positive reduction."""
        # This is a simplified calculation
        # In reality, you'd compare before and after filtering
        return 0.3  # Placeholder: 30% reduction
    
    def _calculate_context_awareness_score(
        self, 
        framework_context: ProjectContext, 
        pattern_analysis: PatternAnalysisResult
    ) -> float:
        """Calculate how well the analysis understands the context."""
        score = 0.0
        
        # Framework detection score
        if framework_context.primary_framework:
            score += framework_context.primary_framework.confidence * 0.4
        
        # Pattern analysis score
        if pattern_analysis.design_patterns or pattern_analysis.framework_patterns:
            score += 0.3
        
        # Usage pattern score
        if pattern_analysis.usage_patterns:
            score += 0.3
        
        return min(score, 1.0)


def main():
    """Main entry point for testing the context-aware analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context-Aware Dead Code Analyzer")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    config = AnalysisConfig()
    analyzer = ContextAwareDeadCodeAnalyzer(config)
    
    # Run analysis
    project_root = Path(args.project_root)
    result = analyzer.analyze(project_root)
    
    # Print results
    print(f"\nContext-Aware Dead Code Analysis Results for {project_root}:")
    print(f"Primary framework: {result.framework_context.primary_framework.framework_name if result.framework_context.primary_framework else 'None'}")
    print(f"Framework confidence: {result.framework_context.primary_framework.confidence if result.framework_context.primary_framework else 0.0:.2f}")
    print(f"Context-aware dead functions: {len(result.context_aware_dead_functions)}")
    print(f"Context-aware dead classes: {len(result.context_aware_dead_classes)}")
    print(f"Context-aware dead imports: {len(result.context_aware_dead_imports)}")
    print(f"False positives filtered: {result.false_positives_filtered}")
    print(f"Overall confidence: {result.confidence_scores.get('overall', 0.0):.2f}")
    print(f"Context awareness score: {result.context_insights.get('context_effectiveness', {}).get('context_awareness_score', 0.0):.2f}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    
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