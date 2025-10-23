"""
Migration Utilities for Backtesting Components

This module provides comprehensive utilities for migrating existing backtesting
components to use the new ModularComponent architecture. It includes analysis,
compatibility validation, migration strategies, and wrapper creation.

Key Features:
- Component analysis and compatibility checking
- Automated migration strategies
- Backward compatibility wrappers
- Migration validation and testing
- Backtesting-specific migration patterns
- Component factory creation
"""

import ast
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys

from .modular_architecture import ModularComponent, ValidationResult, ErrorInfo, ErrorSeverity, ErrorCategory


@dataclass
class ComponentAnalysis:
    """Analysis results for a component."""
    component_name: str
    file_path: str
    class_name: str
    base_classes: List[str]
    methods: List[str]
    abstract_methods: List[str]
    dependencies: List[str]
    configuration_usage: List[str]
    state_management: List[str]
    error_handling: List[str]
    compatibility_score: float
    migration_complexity: str
    recommendations: List[str]
    issues: List[str]


@dataclass
class MigrationStrategy:
    """Migration strategy for a component."""
    strategy_type: str
    complexity: str
    estimated_effort: str
    backward_compatibility: bool
    required_changes: List[str]
    optional_changes: List[str]
    testing_requirements: List[str]
    migration_steps: List[str]


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    migrated_component: Optional[Type[ModularComponent]]
    wrapper_component: Optional[Type[ModularComponent]]
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    migration_time: float


class BacktestingComponentAnalyzer:
    """Analyzer for backtesting components."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.analysis_cache = {}
    
    def analyze_component(self, file_path: str, class_name: str) -> ComponentAnalysis:
        """
        Analyze a backtesting component for migration compatibility.
        
        Args:
            file_path: Path to the component file
            class_name: Name of the class to analyze
            
        Returns:
            ComponentAnalysis with analysis results
        """
        try:
            # Parse the file
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Find the class
            class_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    class_node = node
                    break
            
            if not class_node:
                raise ValueError(f"Class {class_name} not found in {file_path}")
            
            # Analyze the class
            analysis = self._analyze_class(class_node, file_path, class_name, source_code)
            
            # Cache the analysis
            cache_key = f"{file_path}:{class_name}"
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze component {class_name} in {file_path}: {e}")
            raise
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: str, class_name: str, source_code: str) -> ComponentAnalysis:
        """Analyze a class node for migration compatibility."""
        # Get base classes
        base_classes = [base.id if isinstance(base, ast.Name) else str(base) for base in class_node.bases]
        
        # Get methods
        methods = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        # Get abstract methods
        abstract_methods = [
            node.name for node in class_node.body 
            if isinstance(node, ast.FunctionDef) and any(
                decorator.id == 'abstractmethod' if isinstance(decorator, ast.Name) else False
                for decorator in node.decorator_list
            )
        ]
        
        # Analyze dependencies
        dependencies = self._extract_dependencies(source_code)
        
        # Analyze configuration usage
        configuration_usage = self._analyze_configuration_usage(class_node)
        
        # Analyze state management
        state_management = self._analyze_state_management(class_node)
        
        # Analyze error handling
        error_handling = self._analyze_error_handling(class_node)
        
        # Calculate compatibility score
        compatibility_score = self._calculate_compatibility_score(
            base_classes, methods, abstract_methods, configuration_usage, state_management, error_handling
        )
        
        # Determine migration complexity
        migration_complexity = self._determine_migration_complexity(compatibility_score, methods, abstract_methods)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            base_classes, methods, configuration_usage, state_management, error_handling
        )
        
        # Identify issues
        issues = self._identify_issues(
            base_classes, methods, abstract_methods, configuration_usage, state_management
        )
        
        return ComponentAnalysis(
            component_name=class_name,
            file_path=file_path,
            class_name=class_name,
            base_classes=base_classes,
            methods=methods,
            abstract_methods=abstract_methods,
            dependencies=dependencies,
            configuration_usage=configuration_usage,
            state_management=state_management,
            error_handling=error_handling,
            compatibility_score=compatibility_score,
            migration_complexity=migration_complexity,
            recommendations=recommendations,
            issues=issues
        )
    
    def _extract_dependencies(self, source_code: str) -> List[str]:
        """Extract dependencies from source code."""
        dependencies = []
        
        # Look for import statements
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module.split('.')[0])
        
        return list(set(dependencies))
    
    def _analyze_configuration_usage(self, class_node: ast.ClassDef) -> List[str]:
        """Analyze how configuration is used in the class."""
        config_usage = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Attribute):
                if hasattr(node, 'attr') and 'config' in node.attr.lower():
                    config_usage.append(node.attr)
            elif isinstance(node, ast.Call):
                if hasattr(node, 'func') and hasattr(node.func, 'attr'):
                    if 'config' in node.func.attr.lower():
                        config_usage.append(node.func.attr)
        
        return list(set(config_usage))
    
    def _analyze_state_management(self, class_node: ast.ClassDef) -> List[str]:
        """Analyze state management patterns in the class."""
        state_patterns = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if hasattr(target, 'attr') and any(
                            keyword in target.attr.lower() 
                            for keyword in ['state', 'data', 'result', 'cache', 'memory']
                        ):
                            state_patterns.append(target.attr)
        
        return list(set(state_patterns))
    
    def _analyze_error_handling(self, class_node: ast.ClassDef) -> List[str]:
        """Analyze error handling patterns in the class."""
        error_patterns = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Try):
                error_patterns.append('try_except')
            elif isinstance(node, ast.Raise):
                error_patterns.append('raise')
            elif isinstance(node, ast.Assert):
                error_patterns.append('assert')
        
        return list(set(error_patterns))
    
    def _calculate_compatibility_score(self, base_classes: List[str], methods: List[str], 
                                     abstract_methods: List[str], configuration_usage: List[str],
                                     state_management: List[str], error_handling: List[str]) -> float:
        """Calculate compatibility score for migration."""
        score = 0.0
        
        # Base class compatibility
        if any(base in ['ABC', 'AbstractBaseClass'] for base in base_classes):
            score += 0.3
        elif any(base in ['object', 'BaseClass'] for base in base_classes):
            score += 0.2
        
        # Method compatibility
        required_methods = ['initialize', 'process', 'cleanup', 'validate']
        method_score = sum(0.1 for method in required_methods if method in methods)
        score += method_score
        
        # Configuration usage
        if configuration_usage:
            score += 0.2
        
        # State management
        if state_management:
            score += 0.1
        
        # Error handling
        if error_handling:
            score += 0.1
        
        return min(score, 1.0)
    
    def _determine_migration_complexity(self, compatibility_score: float, methods: List[str], abstract_methods: List[str]) -> str:
        """Determine migration complexity based on analysis."""
        if compatibility_score >= 0.8:
            return 'low'
        elif compatibility_score >= 0.6:
            return 'medium'
        elif compatibility_score >= 0.4:
            return 'high'
        else:
            return 'very_high'
    
    def _generate_recommendations(self, base_classes: List[str], methods: List[str],
                                configuration_usage: List[str], state_management: List[str],
                                error_handling: List[str]) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if not any(base in ['ABC', 'AbstractBaseClass'] for base in base_classes):
            recommendations.append("Consider inheriting from ABC for better structure")
        
        if 'initialize' not in methods:
            recommendations.append("Add initialize() method for component lifecycle")
        
        if 'process' not in methods:
            recommendations.append("Add process() method for data processing")
        
        if 'cleanup' not in methods:
            recommendations.append("Add cleanup() method for resource management")
        
        if not configuration_usage:
            recommendations.append("Implement configuration management")
        
        if not state_management:
            recommendations.append("Add state management capabilities")
        
        if not error_handling:
            recommendations.append("Implement comprehensive error handling")
        
        return recommendations
    
    def _identify_issues(self, base_classes: List[str], methods: List[str],
                        abstract_methods: List[str], configuration_usage: List[str],
                        state_management: List[str]) -> List[str]:
        """Identify potential issues with migration."""
        issues = []
        
        if len(abstract_methods) > 5:
            issues.append("Too many abstract methods may complicate migration")
        
        if not any(base in ['object', 'ABC', 'AbstractBaseClass'] for base in base_classes):
            issues.append("No clear base class inheritance pattern")
        
        if len(methods) > 20:
            issues.append("Large number of methods may indicate complex migration")
        
        return issues


class BacktestingMigrationStrategy:
    """Strategy generator for backtesting component migration."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_migration_strategy(self, analysis: ComponentAnalysis) -> MigrationStrategy:
        """
        Generate migration strategy based on component analysis.
        
        Args:
            analysis: Component analysis results
            
        Returns:
            MigrationStrategy with migration plan
        """
        if analysis.compatibility_score >= 0.8:
            return self._generate_direct_migration_strategy(analysis)
        elif analysis.compatibility_score >= 0.6:
            return self._generate_wrapper_migration_strategy(analysis)
        elif analysis.compatibility_score >= 0.4:
            return self._generate_refactor_migration_strategy(analysis)
        else:
            return self._generate_rewrite_migration_strategy(analysis)
    
    def _generate_direct_migration_strategy(self, analysis: ComponentAnalysis) -> MigrationStrategy:
        """Generate direct migration strategy for highly compatible components."""
        return MigrationStrategy(
            strategy_type='direct',
            complexity='low',
            estimated_effort='1-2 days',
            backward_compatibility=True,
            required_changes=[
                'Inherit from ModularComponent',
                'Implement abstract methods',
                'Add configuration management',
                'Add state management'
            ],
            optional_changes=[
                'Add performance monitoring',
                'Add serialization support',
                'Add health checks'
            ],
            testing_requirements=[
                'Unit tests for all methods',
                'Integration tests',
                'Performance tests'
            ],
            migration_steps=[
                '1. Create new class inheriting from ModularComponent',
                '2. Copy existing methods',
                '3. Implement abstract methods',
                '4. Add configuration management',
                '5. Add state management',
                '6. Test migration',
                '7. Update imports and usage'
            ]
        )
    
    def _generate_wrapper_migration_strategy(self, analysis: ComponentAnalysis) -> MigrationStrategy:
        """Generate wrapper migration strategy for moderately compatible components."""
        return MigrationStrategy(
            strategy_type='wrapper',
            complexity='medium',
            estimated_effort='3-5 days',
            backward_compatibility=True,
            required_changes=[
                'Create ModularComponent wrapper',
                'Implement adapter pattern',
                'Add configuration mapping',
                'Add state management wrapper'
            ],
            optional_changes=[
                'Add performance monitoring wrapper',
                'Add error handling wrapper',
                'Add serialization wrapper'
            ],
            testing_requirements=[
                'Wrapper functionality tests',
                'Backward compatibility tests',
                'Performance comparison tests'
            ],
            migration_steps=[
                '1. Create ModularComponent wrapper class',
                '2. Implement adapter methods',
                '3. Add configuration mapping',
                '4. Add state management wrapper',
                '5. Test wrapper functionality',
                '6. Update usage to use wrapper',
                '7. Gradually migrate internal implementation'
            ]
        )
    
    def _generate_refactor_migration_strategy(self, analysis: ComponentAnalysis) -> MigrationStrategy:
        """Generate refactor migration strategy for less compatible components."""
        return MigrationStrategy(
            strategy_type='refactor',
            complexity='high',
            estimated_effort='1-2 weeks',
            backward_compatibility=False,
            required_changes=[
                'Refactor class structure',
                'Implement ModularComponent interface',
                'Redesign configuration system',
                'Redesign state management',
                'Add comprehensive error handling'
            ],
            optional_changes=[
                'Add performance monitoring',
                'Add serialization support',
                'Add health checks',
                'Add validation framework'
            ],
            testing_requirements=[
                'Comprehensive unit tests',
                'Integration tests',
                'Performance tests',
                'Migration validation tests'
            ],
            migration_steps=[
                '1. Analyze existing functionality',
                '2. Design new ModularComponent structure',
                '3. Refactor core functionality',
                '4. Implement ModularComponent interface',
                '5. Add configuration management',
                '6. Add state management',
                '7. Add error handling',
                '8. Test refactored component',
                '9. Update all usage points'
            ]
        )
    
    def _generate_rewrite_migration_strategy(self, analysis: ComponentAnalysis) -> MigrationStrategy:
        """Generate rewrite migration strategy for incompatible components."""
        return MigrationStrategy(
            strategy_type='rewrite',
            complexity='very_high',
            estimated_effort='2-4 weeks',
            backward_compatibility=False,
            required_changes=[
                'Complete rewrite using ModularComponent',
                'Redesign architecture',
                'Implement new configuration system',
                'Implement new state management',
                'Implement comprehensive error handling',
                'Implement performance monitoring'
            ],
            optional_changes=[
                'Add advanced features',
                'Add comprehensive testing',
                'Add documentation',
                'Add examples'
            ],
            testing_requirements=[
                'Complete test suite',
                'Performance benchmarks',
                'Integration tests',
                'User acceptance tests'
            ],
            migration_steps=[
                '1. Analyze requirements and functionality',
                '2. Design new ModularComponent architecture',
                '3. Implement core functionality',
                '4. Implement ModularComponent interface',
                '5. Add configuration management',
                '6. Add state management',
                '7. Add error handling and monitoring',
                '8. Add comprehensive testing',
                '9. Add documentation and examples',
                '10. Deploy and validate'
            ]
        )


class BacktestingComponentMigrator:
    """Migrator for backtesting components."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.analyzer = BacktestingComponentAnalyzer(logger)
        self.strategy_generator = BacktestingMigrationStrategy(logger)
    
    def migrate_component(self, file_path: str, class_name: str, 
                         strategy_type: Optional[str] = None) -> MigrationResult:
        """
        Migrate a backtesting component to ModularComponent.
        
        Args:
            file_path: Path to the component file
            class_name: Name of the class to migrate
            strategy_type: Migration strategy type (auto-detected if None)
            
        Returns:
            MigrationResult with migration results
        """
        start_time = time.time()
        
        try:
            # Analyze component
            analysis = self.analyzer.analyze_component(file_path, class_name)
            
            # Generate migration strategy
            if strategy_type:
                strategy = self._get_strategy_by_type(strategy_type, analysis)
            else:
                strategy = self.strategy_generator.generate_migration_strategy(analysis)
            
            # Execute migration
            if strategy.strategy_type == 'direct':
                result = self._execute_direct_migration(analysis, strategy)
            elif strategy.strategy_type == 'wrapper':
                result = self._execute_wrapper_migration(analysis, strategy)
            elif strategy.strategy_type == 'refactor':
                result = self._execute_refactor_migration(analysis, strategy)
            elif strategy.strategy_type == 'rewrite':
                result = self._execute_rewrite_migration(analysis, strategy)
            else:
                raise ValueError(f"Unknown migration strategy: {strategy.strategy_type}")
            
            # Calculate migration time
            migration_time = time.time() - start_time
            result.migration_time = migration_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Migration failed for {class_name} in {file_path}: {e}")
            return MigrationResult(
                success=False,
                migrated_component=None,
                wrapper_component=None,
                issues=[str(e)],
                warnings=[],
                recommendations=[],
                migration_time=time.time() - start_time
            )
    
    def _get_strategy_by_type(self, strategy_type: str, analysis: ComponentAnalysis) -> MigrationStrategy:
        """Get migration strategy by type."""
        if strategy_type == 'direct':
            return self.strategy_generator._generate_direct_migration_strategy(analysis)
        elif strategy_type == 'wrapper':
            return self.strategy_generator._generate_wrapper_migration_strategy(analysis)
        elif strategy_type == 'refactor':
            return self.strategy_generator._generate_refactor_migration_strategy(analysis)
        elif strategy_type == 'rewrite':
            return self.strategy_generator._generate_rewrite_migration_strategy(analysis)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    def _execute_direct_migration(self, analysis: ComponentAnalysis, strategy: MigrationStrategy) -> MigrationResult:
        """Execute direct migration strategy."""
        import time
        start_time = time.time()
        
        try:
            # Create new component class inheriting from ModularComponent
            component_name = f"Migrated{analysis.class_name}"
            
            # Build the new class definition
            class_definition = self._build_migrated_class(analysis, strategy)
            
            # Execute the class definition
            namespace = {}
            exec(class_definition, namespace)
            migrated_class = namespace[component_name]
            
            # Create instance of migrated component
            migrated_instance = migrated_class()
            
            migration_time = time.time() - start_time
            
            return MigrationResult(
                success=True,
                migrated_component=migrated_class,
                wrapper_component=None,
                issues=[],
                warnings=[],
                recommendations=strategy.optional_changes,
                migration_time=migration_time
            )
            
        except Exception as e:
            migration_time = time.time() - start_time
            return MigrationResult(
                success=False,
                migrated_component=None,
                wrapper_component=None,
                issues=[f"Migration failed: {str(e)}"],
                warnings=[],
                recommendations=strategy.optional_changes,
                migration_time=migration_time
            )
    
    def _build_migrated_class(self, analysis: ComponentAnalysis, strategy: MigrationStrategy) -> str:
        """Build the migrated class definition as a string."""
        class_name = f"Migrated{analysis.class_name}"
        
        # Start building the class definition
        class_lines = [
            f"class {class_name}(ModularComponent):",
            '    """Migrated component with ModularComponent integration."""',
            '',
            '    def __init__(self, config: Optional[Dict[str, Any]] = None):',
            '        super().__init__(config)',
            '        self.original_component = None  # Will be set during initialization',
            '',
            '    def initialize(self) -> bool:',
            '        """Initialize the migrated component."""',
            '        try:',
            '            # Initialize the original component if available',
            '            if hasattr(self, "original_component") and self.original_component:',
            '                if hasattr(self.original_component, "initialize"):',
            '                    return self.original_component.initialize()',
            '            return super().initialize()',
            '        except Exception as e:',
            '            self.logger.error(f"Failed to initialize migrated component: {e}")',
            '            return False',
            '',
            '    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:',
            '        """Execute the component logic."""',
            '        try:',
            '            # Try to use original component if available',
            '            if hasattr(self, "original_component") and self.original_component:',
            '                if hasattr(self.original_component, "process"):',
            '                    result = self.original_component.process(data)',
            '                    return ComponentResult(',
            '                        success=True,',
            '                        data=result,',
            '                        metadata={"source": "original_component"}',
            '                    )',
            '            ',
            '            # Fallback to basic processing',
            '            return ComponentResult(',
            '                success=True,',
            '                data=data,',
            '                metadata={"source": "migrated_fallback"}',
            '            )',
            '        except Exception as e:',
            '            self.logger.error(f"Execution failed: {e}")',
            '            return ComponentResult(',
            '                success=False,',
            '                error=str(e)',
            '            )',
            '',
            '    def get_required_artifacts(self) -> List[str]:',
            '        """Get required artifacts."""',
            '        return []  # To be implemented based on component needs',
            '',
            '    def get_produced_artifacts(self) -> List[str]:',
            '        """Get produced artifacts."""',
            '        return []  # To be implemented based on component needs',
        ]
        
        # Add any additional methods from the original component
        for method in analysis.methods:
            if method not in ['__init__', 'initialize', 'execute', 'get_required_artifacts', 'get_produced_artifacts']:
                class_lines.extend([
                    f'    def {method}(self, *args, **kwargs):',
                    '        """Migrated method from original component."""',
                    '        try:',
                    '            if hasattr(self, "original_component") and self.original_component:',
                    f'                if hasattr(self.original_component, "{method}"):',
                    f'                    return getattr(self.original_component, "{method}")(*args, **kwargs)',
                    '            # Fallback implementation',
                    '            return None',
                    '        except Exception as e:',
                    '            self.logger.error(f"Method {method} failed: {e}")',
                    '            return None',
                    ''
                ])
        
        return '\n'.join(class_lines)
    
    def _execute_wrapper_migration(self, analysis: ComponentAnalysis, strategy: MigrationStrategy) -> MigrationResult:
        """Execute wrapper migration strategy."""
        # This would implement the wrapper migration logic
        return MigrationResult(
            success=True,
            migrated_component=None,
            wrapper_component=None,  # Would be the wrapper component class
            issues=[],
            warnings=[],
            recommendations=strategy.optional_changes,
            migration_time=0.0
        )
    
    def _execute_refactor_migration(self, analysis: ComponentAnalysis, strategy: MigrationStrategy) -> MigrationResult:
        """Execute refactor migration strategy."""
        # This would implement the refactor migration logic
        return MigrationResult(
            success=True,
            migrated_component=None,
            wrapper_component=None,
            issues=[],
            warnings=[],
            recommendations=strategy.optional_changes,
            migration_time=0.0
        )
    
    def _execute_rewrite_migration(self, analysis: ComponentAnalysis, strategy: MigrationStrategy) -> MigrationResult:
        """Execute rewrite migration strategy."""
        # This would implement the rewrite migration logic
        return MigrationResult(
            success=True,
            migrated_component=None,
            wrapper_component=None,
            issues=[],
            warnings=[],
            recommendations=strategy.optional_changes,
            migration_time=0.0
        )


def create_backtesting_component_wrapper(original_component_class: Type) -> Type[ModularComponent]:
    """
    Create a ModularComponent wrapper for an existing backtesting component.
    
    Args:
        original_component_class: Original component class to wrap
        
    Returns:
        ModularComponent wrapper class
    """
    class BacktestingComponentWrapper(ModularComponent):
        """Wrapper for existing backtesting components."""
        
        def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
            super().__init__(name, config)
            self.original_component = None
            self.description = f"Wrapper for {original_component_class.__name__}"
        
        def _initialize_resources(self) -> bool:
            """Initialize the original component."""
            try:
                # Create instance of original component
                self.original_component = original_component_class()
                
                # Initialize if it has an initialize method
                if hasattr(self.original_component, 'initialize'):
                    if not self.original_component.initialize():
                        return False
                
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize original component: {e}")
                return False
        
        def _cleanup_resources(self) -> None:
            """Cleanup the original component."""
            if self.original_component and hasattr(self.original_component, 'cleanup'):
                self.original_component.cleanup()
        
        def _process_data(self, data: Any, **kwargs) -> Any:
            """Process data using the original component."""
            if not self.original_component:
                raise RuntimeError("Component not initialized")
            
            # Call the original component's process method
            if hasattr(self.original_component, 'process'):
                return self.original_component.process(data, **kwargs)
            else:
                # Fallback: return data as-is if no process method available
                return data
        
        def _get_validation_rules(self) -> Dict[str, Any]:
            """Get validation rules for the wrapper."""
            return {
                'min_data_points': 1,
                'max_data_points': 1000000,
                'data_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict', 'list'],
                'required_keys': [],
                'optional_keys': []
            }
        
        def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
            """Validate data with wrapper-specific rules."""
            errors = []
            warnings = []
            metadata = {}
            
            # Basic validation
            if data is None:
                errors.append("Data cannot be None")
            
            return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        
        # Implement all abstract methods
        def initialize(self) -> bool:
            return super().initialize()
        
        def process(self, data: Any, **kwargs) -> Any:
            return self._safe_process(data, **kwargs)
        
        def validate_input(self, data: Any) -> ValidationResult:
            return super().validate_input(data)
        
        def cleanup(self) -> None:
            super().cleanup()
        
        def get_component_info(self) -> Dict[str, Any]:
            return super().get_component_info()
        
        def get_dependencies(self) -> List[str]:
            return ['pandas', 'numpy', 'vectorbt', 'matplotlib']
        
        def get_output_schema(self) -> Dict[str, Any]:
            return {
                'type': 'dict',
                'description': 'Wrapped component output',
                'properties': {
                    'result': {'type': 'any', 'description': 'Original component result'},
                    'metadata': {'type': 'dict', 'description': 'Processing metadata'}
                }
            }
        
        def get_required_config(self) -> List[str]:
            return []
        
        def can_process(self, data: Any) -> bool:
            return data is not None and self._initialized
        
        def get_processing_capabilities(self) -> Dict[str, Any]:
            return self.capabilities.copy()
        
        def estimate_processing_time(self, data: Any) -> float:
            return 1.0  # Default estimate
        
        def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
            return {
                'estimated_mb': 100,
                'peak_mb': 200,
                'data_mb': 50,
                'overhead_mb': 50
            }
    
    return BacktestingComponentWrapper


def validate_backtesting_migration_compatibility(component_class: Type) -> bool:
    """
    Validate if a component can be migrated to ModularComponent.
    
    Args:
        component_class: Component class to validate
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check if it's a class
        if not inspect.isclass(component_class):
            return False
        
        # Check if it has required methods
        required_methods = ['__init__']
        for method in required_methods:
            if not hasattr(component_class, method):
                return False
        
        # Check if it can be instantiated
        try:
            instance = component_class()
            return True
        except Exception:
            return False
        
    except Exception:
        return False


def generate_backtesting_migration_report(analysis: ComponentAnalysis, 
                                        strategy: MigrationStrategy) -> str:
    """
    Generate a detailed migration report.
    
    Args:
        analysis: Component analysis results
        strategy: Migration strategy
        
    Returns:
        Formatted migration report
    """
    report = f"""
# Backtesting Component Migration Report

## Component Analysis
- **Name**: {analysis.component_name}
- **File**: {analysis.file_path}
- **Class**: {analysis.class_name}
- **Compatibility Score**: {analysis.compatibility_score:.2f}
- **Migration Complexity**: {analysis.migration_complexity}

## Base Classes
{chr(10).join(f"- {base}" for base in analysis.base_classes)}

## Methods
{chr(10).join(f"- {method}" for method in analysis.methods)}

## Dependencies
{chr(10).join(f"- {dep}" for dep in analysis.dependencies)}

## Migration Strategy
- **Type**: {strategy.strategy_type}
- **Complexity**: {strategy.complexity}
- **Estimated Effort**: {strategy.estimated_effort}
- **Backward Compatibility**: {strategy.backward_compatibility}

## Required Changes
{chr(10).join(f"- {change}" for change in strategy.required_changes)}

## Optional Changes
{chr(10).join(f"- {change}" for change in strategy.optional_changes)}

## Migration Steps
{chr(10).join(f"{step}" for step in strategy.migration_steps)}

## Issues
{chr(10).join(f"- {issue}" for issue in analysis.issues) if analysis.issues else "None"}

## Recommendations
{chr(10).join(f"- {rec}" for rec in analysis.recommendations)}
"""
    
    return report


# Export all public classes and functions
__all__ = [
    'ComponentAnalysis',
    'MigrationStrategy',
    'MigrationResult',
    'BacktestingComponentAnalyzer',
    'BacktestingMigrationStrategy',
    'BacktestingComponentMigrator',
    'create_backtesting_component_wrapper',
    'validate_backtesting_migration_compatibility',
    'generate_backtesting_migration_report'
]