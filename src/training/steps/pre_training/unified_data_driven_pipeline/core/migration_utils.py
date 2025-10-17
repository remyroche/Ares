"""
Migration utilities for converting existing components to ModularComponent architecture.

This module provides utilities to help migrate existing component classes to use
the new ModularComponent base class while maintaining backward compatibility.
"""

import logging
import inspect
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass
from datetime import datetime

from .modular_architecture import ModularComponent, ValidationResult, ErrorInfo

logger = logging.getLogger(__name__)

@dataclass
class MigrationReport:
    """Report of component migration analysis."""
    
    component_name: str
    migration_compatibility: float  # 0.0 to 1.0
    required_changes: List[str]
    optional_improvements: List[str]
    breaking_changes: List[str]
    estimated_effort: str  # "LOW", "MEDIUM", "HIGH"
    migration_strategy: str
    warnings: List[str]
    recommendations: List[str]

class ComponentMigrationAnalyzer:
    """Analyzes existing components for ModularComponent migration compatibility."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_component(self, component_class: Type) -> MigrationReport:
        """
        Analyze a component class for migration compatibility.
        
        Args:
            component_class: The component class to analyze.
            
        Returns:
            MigrationReport: Detailed migration analysis.
        """
        self.logger.info(f"Analyzing component: {component_class.__name__}")
        
        # Check inheritance
        inheritance_analysis = self._analyze_inheritance(component_class)
        
        # Check method compatibility
        method_analysis = self._analyze_methods(component_class)
        
        # Check configuration compatibility
        config_analysis = self._analyze_configuration(component_class)
        
        # Check state management
        state_analysis = self._analyze_state_management(component_class)
        
        # Calculate overall compatibility
        compatibility = self._calculate_compatibility(
            inheritance_analysis, method_analysis, config_analysis, state_analysis
        )
        
        # Generate migration strategy
        strategy = self._generate_migration_strategy(component_class, compatibility)
        
        # Estimate effort
        effort = self._estimate_migration_effort(component_class, compatibility)
        
        return MigrationReport(
            component_name=component_class.__name__,
            migration_compatibility=compatibility,
            required_changes=method_analysis['required_changes'],
            optional_improvements=method_analysis['optional_improvements'],
            breaking_changes=inheritance_analysis['breaking_changes'],
            estimated_effort=effort,
            migration_strategy=strategy,
            warnings=method_analysis['warnings'],
            recommendations=method_analysis['recommendations']
        )
    
    def _analyze_inheritance(self, component_class: Type) -> Dict[str, Any]:
        """Analyze component inheritance structure."""
        analysis = {
            'is_abc': False,
            'has_abstract_methods': False,
            'breaking_changes': [],
            'compatibility_score': 0.0
        }
        
        # Check if it's an ABC
        if hasattr(component_class, '__abstractmethods__'):
            analysis['is_abc'] = True
            analysis['has_abstract_methods'] = len(component_class.__abstractmethods__) > 0
        
        # Check inheritance chain
        mro = inspect.getmro(component_class)
        if ModularComponent in mro:
            analysis['breaking_changes'].append("Already inherits from ModularComponent")
            analysis['compatibility_score'] = 1.0
        elif any('Component' in cls.__name__ for cls in mro[1:]):
            analysis['compatibility_score'] = 0.8
        else:
            analysis['compatibility_score'] = 0.5
        
        return analysis
    
    def _analyze_methods(self, component_class: Type) -> Dict[str, Any]:
        """Analyze component methods for ModularComponent compatibility."""
        analysis = {
            'required_changes': [],
            'optional_improvements': [],
            'warnings': [],
            'recommendations': [],
            'compatibility_score': 0.0
        }
        
        # Get all methods
        methods = inspect.getmembers(component_class, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]
        
        # Check for required ModularComponent methods
        required_methods = [
            'initialize', 'process', 'validate_input', 'cleanup',
            'get_component_info', 'get_dependencies', 'get_output_schema',
            'get_required_config', 'can_process', 'get_processing_capabilities',
            'estimate_processing_time', 'get_memory_requirements'
        ]
        
        missing_methods = [method for method in required_methods if method not in method_names]
        if missing_methods:
            analysis['required_changes'].append(f"Missing required methods: {missing_methods}")
        
        # Check for existing methods that can be mapped
        mappable_methods = {
            'process': ['process', 'execute', 'run', 'transform'],
            'initialize': ['initialize', 'setup', 'init'],
            'cleanup': ['cleanup', 'teardown', 'close', 'destroy'],
            'validate_input': ['validate', 'validate_input', 'check_input']
        }
        
        for modular_method, existing_methods in mappable_methods.items():
            if modular_method not in method_names:
                found_mapping = [method for method in existing_methods if method in method_names]
                if found_mapping:
                    analysis['optional_improvements'].append(
                        f"Can map {found_mapping[0]} to {modular_method}"
                    )
        
        # Check for configuration methods
        config_methods = ['get_config', 'set_config', 'update_config', 'validate_config']
        has_config_methods = any(method in method_names for method in config_methods)
        if not has_config_methods:
            analysis['optional_improvements'].append("Add configuration management methods")
        
        # Check for state management methods
        state_methods = ['get_state', 'set_state', 'clear_state']
        has_state_methods = any(method in method_names for method in state_methods)
        if not has_state_methods:
            analysis['optional_improvements'].append("Add state management methods")
        
        # Calculate compatibility score
        total_required = len(required_methods)
        missing_count = len(missing_methods)
        analysis['compatibility_score'] = max(0.0, (total_required - missing_count) / total_required)
        
        return analysis
    
    def _analyze_configuration(self, component_class: Type) -> Dict[str, Any]:
        """Analyze component configuration handling."""
        analysis = {
            'has_config_class': False,
            'has_config_validation': False,
            'compatibility_score': 0.0
        }
        
        # Check for configuration dataclass
        if hasattr(component_class, '__annotations__'):
            annotations = component_class.__annotations__
            if any('Config' in str(annotation) for annotation in annotations.values()):
                analysis['has_config_class'] = True
        
        # Check for configuration validation
        methods = inspect.getmembers(component_class, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]
        if 'validate_config' in method_names or 'validate' in method_names:
            analysis['has_config_validation'] = True
        
        # Calculate score
        score = 0.0
        if analysis['has_config_class']:
            score += 0.5
        if analysis['has_config_validation']:
            score += 0.5
        
        analysis['compatibility_score'] = score
        return analysis
    
    def _analyze_state_management(self, component_class: Type) -> Dict[str, Any]:
        """Analyze component state management."""
        analysis = {
            'has_state_attributes': False,
            'has_state_methods': False,
            'compatibility_score': 0.0
        }
        
        # Check for state attributes
        if hasattr(component_class, '__init__'):
            init_signature = inspect.signature(component_class.__init__)
            params = list(init_signature.parameters.keys())
            if any('state' in param.lower() for param in params):
                analysis['has_state_attributes'] = True
        
        # Check for state methods
        methods = inspect.getmembers(component_class, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]
        state_methods = ['get_state', 'set_state', 'clear_state', 'reset_state']
        if any(method in method_names for method in state_methods):
            analysis['has_state_methods'] = True
        
        # Calculate score
        score = 0.0
        if analysis['has_state_attributes']:
            score += 0.5
        if analysis['has_state_methods']:
            score += 0.5
        
        analysis['compatibility_score'] = score
        return analysis
    
    def _calculate_compatibility(self, inheritance: Dict, methods: Dict, config: Dict, state: Dict) -> float:
        """Calculate overall migration compatibility score."""
        weights = {
            'inheritance': 0.3,
            'methods': 0.4,
            'config': 0.2,
            'state': 0.1
        }
        
        score = (
            inheritance['compatibility_score'] * weights['inheritance'] +
            methods['compatibility_score'] * weights['methods'] +
            config['compatibility_score'] * weights['config'] +
            state['compatibility_score'] * weights['state']
        )
        
        return min(1.0, max(0.0, score))
    
    def _generate_migration_strategy(self, component_class: Type, compatibility: float) -> str:
        """Generate migration strategy based on compatibility score."""
        if compatibility >= 0.9:
            return "Direct inheritance - minimal changes required"
        elif compatibility >= 0.7:
            return "Wrapper approach - create ModularComponent wrapper"
        elif compatibility >= 0.5:
            return "Refactoring approach - significant changes required"
        else:
            return "Complete rewrite - major architectural changes needed"
    
    def _estimate_migration_effort(self, component_class: Type, compatibility: float) -> str:
        """Estimate migration effort based on compatibility score."""
        if compatibility >= 0.9:
            return "LOW"
        elif compatibility >= 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

class ComponentMigrationWrapper:
    """Wrapper to help migrate existing components to ModularComponent."""
    
    def __init__(self, original_component_class: Type, modular_component_class: Type = None):
        self.original_class = original_component_class
        self.modular_class = modular_component_class or ModularComponent
        self.logger = logging.getLogger(__name__)
    
    def create_migrated_component(self, name: str, config: Optional[Dict[str, Any]] = None, 
                                logger: Optional[logging.Logger] = None) -> ModularComponent:
        """
        Create a migrated component instance.
        
        Args:
            name: Component name.
            config: Component configuration.
            logger: Logger instance.
            
        Returns:
            ModularComponent: Migrated component instance.
        """
        # Create the modular component
        modular_component = self.modular_class(name, config, logger)
        
        # Wrap the original component
        original_instance = self.original_class()
        
        # Map methods
        self._map_methods(modular_component, original_instance)
        
        return modular_component
    
    def _map_methods(self, modular_component: ModularComponent, original_instance: Any) -> None:
        """Map original component methods to ModularComponent methods."""
        # Map initialization
        if hasattr(original_instance, 'initialize'):
            def _initialize_resources():
                return original_instance.initialize()
            modular_component._initialize_resources = _initialize_resources
        
        # Map cleanup
        if hasattr(original_instance, 'cleanup'):
            def _cleanup_resources():
                original_instance.cleanup()
            modular_component._cleanup_resources = _cleanup_resources
        
        # Map processing
        if hasattr(original_instance, 'process'):
            def _process_data(data, **kwargs):
                return original_instance.process(data, **kwargs)
            modular_component._process_data = _process_data
        
        # Map validation
        if hasattr(original_instance, 'validate'):
            def _get_validation_rules():
                return getattr(original_instance, 'validation_rules', {})
            modular_component._get_validation_rules = _get_validation_rules

def migrate_base_component_to_modular(component_class: Type, 
                                    modular_component_class: Type = None) -> Type[ModularComponent]:
    """
    Migrate a base component class to inherit from ModularComponent.
    
    Args:
        component_class: The component class to migrate.
        modular_component_class: The ModularComponent class to inherit from.
        
    Returns:
        Type[ModularComponent]: Migrated component class.
    """
    if modular_component_class is None:
        modular_component_class = ModularComponent
    
    class MigratedComponent(modular_component_class):
        """Migrated component that inherits from ModularComponent."""
        
        def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, 
                     logger: Optional[logging.Logger] = None):
            super().__init__(name, config, logger)
            self._original_component = component_class()
        
        def _initialize_resources(self) -> bool:
            """Initialize resources using original component."""
            if hasattr(self._original_component, 'initialize'):
                return self._original_component.initialize()
            return True
        
        def _cleanup_resources(self) -> None:
            """Cleanup resources using original component."""
            if hasattr(self._original_component, 'cleanup'):
                self._original_component.cleanup()
        
        def _process_data(self, data: Any, **kwargs) -> Any:
            """Process data using original component."""
            if hasattr(self._original_component, 'process'):
                return self._original_component.process(data, **kwargs)
            return data
        
        def _get_validation_rules(self) -> Dict[str, Any]:
            """Get validation rules from original component."""
            if hasattr(self._original_component, 'validation_rules'):
                return self._original_component.validation_rules
            return {}
        
        def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
            """Validate data using original component."""
            if hasattr(self._original_component, 'validate'):
                try:
                    result = self._original_component.validate(data)
                    if isinstance(result, bool):
                        return {'errors': [] if result else ['Validation failed'], 'warnings': [], 'metadata': {}}
                    return result
                except Exception as e:
                    return {'errors': [str(e)], 'warnings': [], 'metadata': {}}
            return {'errors': [], 'warnings': [], 'metadata': {}}
    
    return MigratedComponent

def create_component_wrapper(original_component_class: Type) -> Type[ModularComponent]:
    """
    Create a wrapper that makes an existing component compatible with ModularComponent.
    
    Args:
        original_component_class: The original component class to wrap.
        
    Returns:
        Type[ModularComponent]: Wrapped component class.
    """
    class ComponentWrapper(ModularComponent):
        """Wrapper for existing component to make it ModularComponent compatible."""
        
        def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, 
                     logger: Optional[logging.Logger] = None):
            super().__init__(name, config, logger)
            self._wrapped_component = original_component_class()
        
        def _initialize_resources(self) -> bool:
            """Initialize wrapped component."""
            try:
                if hasattr(self._wrapped_component, 'initialize'):
                    return self._wrapped_component.initialize()
                return True
            except Exception as e:
                self.logger.error(f"Failed to initialize wrapped component: {e}")
                return False
        
        def _cleanup_resources(self) -> None:
            """Cleanup wrapped component."""
            try:
                if hasattr(self._wrapped_component, 'cleanup'):
                    self._wrapped_component.cleanup()
            except Exception as e:
                self.logger.error(f"Failed to cleanup wrapped component: {e}")
        
        def _process_data(self, data: Any, **kwargs) -> Any:
            """Process data using wrapped component."""
            try:
                if hasattr(self._wrapped_component, 'process'):
                    return self._wrapped_component.process(data, **kwargs)
                elif hasattr(self._wrapped_component, 'execute'):
                    return self._wrapped_component.execute(data, **kwargs)
                elif hasattr(self._wrapped_component, 'run'):
                    return self._wrapped_component.run(data, **kwargs)
                else:
                    return data
            except Exception as e:
                self.logger.error(f"Failed to process data with wrapped component: {e}")
                raise
        
        def _get_validation_rules(self) -> Dict[str, Any]:
            """Get validation rules from wrapped component."""
            if hasattr(self._wrapped_component, 'validation_rules'):
                return self._wrapped_component.validation_rules
            return {}
        
        def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
            """Validate data using wrapped component."""
            try:
                if hasattr(self._wrapped_component, 'validate'):
                    result = self._wrapped_component.validate(data)
                    if isinstance(result, bool):
                        return {'errors': [] if result else ['Validation failed'], 'warnings': [], 'metadata': {}}
                    return result
                return {'errors': [], 'warnings': [], 'metadata': {}}
            except Exception as e:
                return {'errors': [str(e)], 'warnings': [], 'metadata': {}}
    
    return ComponentWrapper

def validate_migration_compatibility(component_class: Type) -> bool:
    """
    Validate if a component can be migrated to ModularComponent.
    
    Args:
        component_class: The component class to validate.
        
    Returns:
        bool: True if migration is possible, False otherwise.
    """
    analyzer = ComponentMigrationAnalyzer()
    report = analyzer.analyze_component(component_class)
    
    # Consider migration possible if compatibility is above 0.5
    return report.migration_compatibility >= 0.5

def generate_migration_report(component_class: Type) -> MigrationReport:
    """
    Generate a detailed migration report for a component.
    
    Args:
        component_class: The component class to analyze.
        
    Returns:
        MigrationReport: Detailed migration analysis report.
    """
    analyzer = ComponentMigrationAnalyzer()
    return analyzer.analyze_component(component_class)

# Convenience functions for common migration patterns
def create_simple_migration(original_class: Type) -> Type[ModularComponent]:
    """Create a simple migration for basic components."""
    return create_component_wrapper(original_class)

def create_advanced_migration(original_class: Type) -> Type[ModularComponent]:
    """Create an advanced migration with full ModularComponent features."""
    return migrate_base_component_to_modular(original_class)

# Export main functions
__all__ = [
    'ComponentMigrationAnalyzer',
    'ComponentMigrationWrapper',
    'MigrationReport',
    'migrate_base_component_to_modular',
    'create_component_wrapper',
    'validate_migration_compatibility',
    'generate_migration_report',
    'create_simple_migration',
    'create_advanced_migration'
]