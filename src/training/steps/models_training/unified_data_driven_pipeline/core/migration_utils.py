"""
Migration Utilities for Models Training Components

This module provides utilities to migrate existing models training components
to the new ModularComponent architecture. It includes analysis tools,
compatibility validation, and automated migration strategies.

Key Features:
- Component analysis and compatibility checking
- Automated migration wrapper generation
- Migration validation and testing
- Migration report generation
- ML-specific migration patterns
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Type, Callable, Tuple
from dataclasses import dataclass
import traceback
import time

from .modular_architecture import ModularComponent, ErrorInfo, ErrorSeverity, ErrorCategory


@dataclass
class ComponentAnalysis:
    """Analysis results for a component."""
    component_name: str
    component_class: Type
    has_init: bool
    has_process: bool
    has_initialize: bool
    has_cleanup: bool
    has_config: bool
    has_state: bool
    methods: List[str]
    attributes: List[str]
    dependencies: List[str]
    compatibility_score: float
    migration_difficulty: str
    recommendations: List[str]


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    success: bool
    original_component: Type
    migrated_component: Optional[Type]
    errors: List[ErrorInfo]
    warnings: List[str]
    migration_time: float
    compatibility_score: float


class ModelsTrainingMigrationUtils:
    """Utilities for migrating models training components to ModularComponent."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.migration_patterns = self._initialize_migration_patterns()
    
    def _initialize_migration_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize migration patterns for different component types."""
        return {
            'training_pipeline': {
                'base_methods': ['train', 'fit', 'process'],
                'config_keys': ['model_config', 'training_config', 'data_config'],
                'state_keys': ['model', 'metrics', 'history'],
                'ml_specific': True
            },
            'model_trainer': {
                'base_methods': ['train_model', 'fit_model', 'evaluate'],
                'config_keys': ['model_type', 'hyperparameters', 'training_params'],
                'state_keys': ['model_weights', 'training_progress', 'validation_metrics'],
                'ml_specific': True
            },
            'ensemble_trainer': {
                'base_methods': ['train_ensemble', 'fit_ensemble', 'combine_models'],
                'config_keys': ['ensemble_config', 'model_configs', 'combination_method'],
                'state_keys': ['ensemble_models', 'weights', 'performance'],
                'ml_specific': True
            },
            'ml_labeler': {
                'base_methods': ['generate_labels', 'label_data', 'predict_labels'],
                'config_keys': ['labeling_config', 'model_config', 'thresholds'],
                'state_keys': ['labeling_model', 'label_history', 'quality_metrics'],
                'ml_specific': True
            },
            'pre_ml_orchestrator': {
                'base_methods': ['orchestrate', 'prepare_data', 'coordinate'],
                'config_keys': ['orchestration_config', 'pipeline_config'],
                'state_keys': ['pipeline_state', 'coordination_data'],
                'ml_specific': False
            }
        }
    
    def analyze_component(self, component_class: Type) -> ComponentAnalysis:
        """
        Analyze a component for migration compatibility.
        
        Args:
            component_class: Component class to analyze
            
        Returns:
            ComponentAnalysis with detailed analysis results
        """
        try:
            self.logger.info(f"Analyzing component: {component_class.__name__}")
            
            # Get class methods and attributes
            methods = [method for method in dir(component_class) 
                      if not method.startswith('_') and callable(getattr(component_class, method))]
            attributes = [attr for attr in dir(component_class) 
                         if not attr.startswith('_') and not callable(getattr(component_class, attr))]
            
            # Check for common methods
            has_init = hasattr(component_class, '__init__')
            has_process = any(method in methods for method in ['process', 'train', 'fit', 'predict'])
            has_initialize = 'initialize' in methods
            has_cleanup = 'cleanup' in methods
            has_config = any(attr in attributes for attr in ['config', 'configuration', 'params'])
            has_state = any(attr in attributes for attr in ['state', 'model', 'weights', 'history'])
            
            # Analyze dependencies
            dependencies = self._extract_dependencies(component_class)
            
            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(
                has_init, has_process, has_initialize, has_cleanup, has_config, has_state
            )
            
            # Determine migration difficulty
            migration_difficulty = self._determine_migration_difficulty(compatibility_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                component_class, has_init, has_process, has_initialize, 
                has_cleanup, has_config, has_state, compatibility_score
            )
            
            return ComponentAnalysis(
                component_name=component_class.__name__,
                component_class=component_class,
                has_init=has_init,
                has_process=has_process,
                has_initialize=has_initialize,
                has_cleanup=has_cleanup,
                has_config=has_config,
                has_state=has_state,
                methods=methods,
                attributes=attributes,
                dependencies=dependencies,
                compatibility_score=compatibility_score,
                migration_difficulty=migration_difficulty,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Component analysis failed: {e}")
            raise
    
    def _extract_dependencies(self, component_class: Type) -> List[str]:
        """Extract dependencies from component class."""
        dependencies = []
        
        try:
            # Get source code
            source = inspect.getsource(component_class)
            
            # Common ML dependencies
            ml_deps = ['pandas', 'numpy', 'torch', 'sklearn', 'tensorflow', 'keras', 'xgboost', 'lightgbm']
            for dep in ml_deps:
                if f'import {dep}' in source or f'from {dep}' in source:
                    dependencies.append(dep)
            
            # Check for other common imports
            if 'import joblib' in source:
                dependencies.append('joblib')
            if 'import matplotlib' in source:
                dependencies.append('matplotlib')
            if 'import seaborn' in source:
                dependencies.append('seaborn')
                
        except Exception as e:
            self.logger.warning(f"Could not extract dependencies: {e}")
        
        return dependencies
    
    def _calculate_compatibility_score(self, has_init: bool, has_process: bool, 
                                     has_initialize: bool, has_cleanup: bool,
                                     has_config: bool, has_state: bool) -> float:
        """Calculate compatibility score for migration."""
        score = 0.0
        
        if has_init:
            score += 0.2
        if has_process:
            score += 0.3
        if has_initialize:
            score += 0.1
        if has_cleanup:
            score += 0.1
        if has_config:
            score += 0.15
        if has_state:
            score += 0.15
        
        return min(score, 1.0)
    
    def _determine_migration_difficulty(self, compatibility_score: float) -> str:
        """Determine migration difficulty based on compatibility score."""
        if compatibility_score >= 0.8:
            return 'easy'
        elif compatibility_score >= 0.6:
            return 'medium'
        elif compatibility_score >= 0.4:
            return 'hard'
        else:
            return 'very_hard'
    
    def _generate_recommendations(self, component_class: Type, has_init: bool,
                                has_process: bool, has_initialize: bool,
                                has_cleanup: bool, has_config: bool,
                                has_state: bool, compatibility_score: float) -> List[str]:
        """Generate migration recommendations."""
        recommendations = []
        
        if not has_init:
            recommendations.append("Add __init__ method with name, config, and logger parameters")
        
        if not has_process:
            recommendations.append("Add process method or rename existing method to _process_data")
        
        if not has_initialize:
            recommendations.append("Add initialize method or rename existing method to _initialize_resources")
        
        if not has_cleanup:
            recommendations.append("Add cleanup method or rename existing method to _cleanup_resources")
        
        if not has_config:
            recommendations.append("Add configuration management using get_config/update_config")
        
        if not has_state:
            recommendations.append("Add state management using set_state/get_state")
        
        if compatibility_score < 0.5:
            recommendations.append("Consider refactoring component architecture before migration")
        
        return recommendations
    
    def validate_migration_compatibility(self, component_class: Type) -> bool:
        """
        Validate if a component can be migrated to ModularComponent.
        
        Args:
            component_class: Component class to validate
            
        Returns:
            True if component can be migrated, False otherwise
        """
        try:
            analysis = self.analyze_component(component_class)
            
            # Basic requirements
            if not analysis.has_init:
                self.logger.warning(f"Component {analysis.component_name} lacks __init__ method")
                return False
            
            if not analysis.has_process:
                self.logger.warning(f"Component {analysis.component_name} lacks process method")
                return False
            
            # Check if it's already a ModularComponent
            if issubclass(component_class, ModularComponent):
                self.logger.info(f"Component {analysis.component_name} is already a ModularComponent")
                return True
            
            # Minimum compatibility score
            if analysis.compatibility_score < 0.3:
                self.logger.warning(f"Component {analysis.component_name} has low compatibility score: {analysis.compatibility_score}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Migration compatibility validation failed: {e}")
            return False
    
    def create_component_wrapper(self, component_class: Type, 
                               wrapper_name: Optional[str] = None) -> Type[ModularComponent]:
        """
        Create a ModularComponent wrapper for an existing component.
        
        Args:
            component_class: Component class to wrap
            wrapper_name: Name for the wrapper class
            
        Returns:
            ModularComponent wrapper class
        """
        if wrapper_name is None:
            wrapper_name = f"{component_class.__name__}ModularWrapper"
        
        try:
            self.logger.info(f"Creating wrapper for component: {component_class.__name__}")
            
            # Analyze component to determine migration strategy
            analysis = self.analyze_component(component_class)
            
            # Create wrapper class
            wrapper_class = type(wrapper_name, (ModularComponent,), {
                '_original_component_class': component_class,
                '_analysis': analysis,
                '__module__': component_class.__module__
            })
            
            # Add wrapper methods
            wrapper_class.__init__ = self._create_wrapper_init(component_class)
            wrapper_class._initialize_resources = self._create_wrapper_initialize(component_class)
            wrapper_class._cleanup_resources = self._create_wrapper_cleanup(component_class)
            wrapper_class._process_data = self._create_wrapper_process(component_class)
            wrapper_class._get_validation_rules = self._create_wrapper_validation_rules(component_class)
            wrapper_class._validate_component_specific = self._create_wrapper_validate(component_class)
            
            self.logger.info(f"Created wrapper class: {wrapper_name}")
            return wrapper_class
            
        except Exception as e:
            self.logger.error(f"Failed to create wrapper: {e}")
            raise
    
    def _create_wrapper_init(self, original_class: Type) -> Callable:
        """Create wrapper __init__ method."""
        def wrapper_init(self, name: str, config: Optional[Dict[str, Any]] = None, 
                        logger: Optional[logging.Logger] = None):
            # Initialize ModularComponent
            ModularComponent.__init__(self, name, config, logger)
            
            # Initialize original component
            try:
                # Try to initialize original component with config
                if hasattr(original_class, '__init__'):
                    sig = inspect.signature(original_class.__init__)
                    params = list(sig.parameters.keys())
                    
                    if 'config' in params:
                        self._original_component = original_class(config=config)
                    elif 'name' in params:
                        self._original_component = original_class(name=name)
                    else:
                        self._original_component = original_class()
                else:
                    self._original_component = original_class()
                    
            except Exception as e:
                self.logger.warning(f"Failed to initialize original component: {e}")
                self._original_component = None
        
        return wrapper_init
    
    def _create_wrapper_initialize(self, original_class: Type) -> Callable:
        """Create wrapper _initialize_resources method."""
        def wrapper_initialize(self) -> bool:
            try:
                if self._original_component is None:
                    return False
                
                # Call original initialize method if it exists
                if hasattr(self._original_component, 'initialize'):
                    return self._original_component.initialize()
                elif hasattr(self._original_component, 'init'):
                    return self._original_component.init()
                else:
                    return True
                    
            except Exception as e:
                self.logger.error(f"Wrapper initialization failed: {e}")
                return False
        
        return wrapper_initialize
    
    def _create_wrapper_cleanup(self, original_class: Type) -> Callable:
        """Create wrapper _cleanup_resources method."""
        def wrapper_cleanup(self) -> None:
            try:
                if self._original_component is not None:
                    # Call original cleanup method if it exists
                    if hasattr(self._original_component, 'cleanup'):
                        self._original_component.cleanup()
                    elif hasattr(self._original_component, 'close'):
                        self._original_component.close()
                        
            except Exception as e:
                self.logger.warning(f"Wrapper cleanup failed: {e}")
        
        return wrapper_cleanup
    
    def _create_wrapper_process(self, original_class: Type) -> Callable:
        """Create wrapper _process_data method."""
        def wrapper_process(self, data: Any, **kwargs) -> Any:
            try:
                if self._original_component is None:
                    raise RuntimeError("Original component not initialized")
                
                # Call original process method
                if hasattr(self._original_component, 'process'):
                    return self._original_component.process(data, **kwargs)
                elif hasattr(self._original_component, 'train'):
                    return self._original_component.train(data, **kwargs)
                elif hasattr(self._original_component, 'fit'):
                    return self._original_component.fit(data, **kwargs)
                elif hasattr(self._original_component, 'predict'):
                    return self._original_component.predict(data, **kwargs)
                else:
                    raise RuntimeError("No suitable process method found")
                    
            except Exception as e:
                self.logger.error(f"Wrapper processing failed: {e}")
                raise
        
        return wrapper_process
    
    def _create_wrapper_validation_rules(self, original_class: Type) -> Callable:
        """Create wrapper _get_validation_rules method."""
        def wrapper_validation_rules(self) -> Dict[str, Any]:
            # Default validation rules
            return {
                'min_size': 10,
                'max_size': 1000000,
                'data_types': ['pandas.DataFrame', 'numpy.ndarray', 'dict'],
                'required_attributes': []
            }
        
        return wrapper_validation_rules
    
    def _create_wrapper_validate(self, original_class: Type) -> Callable:
        """Create wrapper _validate_component_specific method."""
        def wrapper_validate(self, data: Any) -> Dict[str, Any]:
            errors = []
            warnings = []
            metadata = {}
            
            # Basic validation
            if data is None:
                errors.append("Input data is None")
            
            if hasattr(data, '__len__') and len(data) == 0:
                warnings.append("Input data is empty")
            
            return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
        
        return wrapper_validate
    
    def migrate_component(self, component_class: Type, 
                         migration_strategy: str = 'wrapper') -> MigrationResult:
        """
        Migrate a component to ModularComponent architecture.
        
        Args:
            component_class: Component class to migrate
            migration_strategy: Migration strategy ('wrapper', 'refactor', 'hybrid')
            
        Returns:
            MigrationResult with migration details
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            self.logger.info(f"Migrating component: {component_class.__name__}")
            
            # Validate compatibility
            if not self.validate_migration_compatibility(component_class):
                errors.append(ErrorInfo(
                    message=f"Component {component_class.__name__} is not compatible for migration",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.VALIDATION
                ))
                return MigrationResult(
                    success=False,
                    original_component=component_class,
                    migrated_component=None,
                    errors=errors,
                    warnings=warnings,
                    migration_time=time.time() - start_time,
                    compatibility_score=0.0
                )
            
            # Perform migration based on strategy
            if migration_strategy == 'wrapper':
                migrated_component = self.create_component_wrapper(component_class)
                warnings.append("Using wrapper strategy - some features may be limited")
            elif migration_strategy == 'refactor':
                migrated_component = self._refactor_component(component_class)
                warnings.append("Using refactor strategy - requires manual review")
            elif migration_strategy == 'hybrid':
                migrated_component = self._hybrid_migration(component_class)
                warnings.append("Using hybrid strategy - combines wrapper and refactor")
            else:
                raise ValueError(f"Unknown migration strategy: {migration_strategy}")
            
            migration_time = time.time() - start_time
            
            self.logger.info(f"Migration completed in {migration_time:.2f}s")
            
            return MigrationResult(
                success=True,
                original_component=component_class,
                migrated_component=migrated_component,
                errors=errors,
                warnings=warnings,
                migration_time=migration_time,
                compatibility_score=self.analyze_component(component_class).compatibility_score
            )
            
        except Exception as e:
            errors.append(ErrorInfo(
                message=f"Migration failed: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING
            ))
            
            return MigrationResult(
                success=False,
                original_component=component_class,
                migrated_component=None,
                errors=errors,
                warnings=warnings,
                migration_time=time.time() - start_time,
                compatibility_score=0.0
            )
    
    def _refactor_component(self, component_class: Type) -> Type[ModularComponent]:
        """Refactor component to inherit from ModularComponent."""
        # This would require more complex refactoring
        # For now, return a wrapper
        return self.create_component_wrapper(component_class)
    
    def _hybrid_migration(self, component_class: Type) -> Type[ModularComponent]:
        """Hybrid migration combining wrapper and refactor approaches."""
        # This would combine both approaches
        # For now, return a wrapper
        return self.create_component_wrapper(component_class)
    
    def generate_migration_report(self, components: List[Type]) -> Dict[str, Any]:
        """
        Generate a comprehensive migration report for multiple components.
        
        Args:
            components: List of component classes to analyze
            
        Returns:
            Migration report dictionary
        """
        report = {
            'timestamp': time.time(),
            'total_components': len(components),
            'components': [],
            'summary': {
                'compatible': 0,
                'incompatible': 0,
                'easy_migration': 0,
                'medium_migration': 0,
                'hard_migration': 0,
                'very_hard_migration': 0
            },
            'recommendations': []
        }
        
        for component_class in components:
            try:
                analysis = self.analyze_component(component_class)
                is_compatible = self.validate_migration_compatibility(component_class)
                
                component_report = {
                    'name': analysis.component_name,
                    'compatible': is_compatible,
                    'compatibility_score': analysis.compatibility_score,
                    'migration_difficulty': analysis.migration_difficulty,
                    'recommendations': analysis.recommendations,
                    'methods': analysis.methods,
                    'dependencies': analysis.dependencies
                }
                
                report['components'].append(component_report)
                
                # Update summary
                if is_compatible:
                    report['summary']['compatible'] += 1
                else:
                    report['summary']['incompatible'] += 1
                
                if analysis.migration_difficulty == 'easy':
                    report['summary']['easy_migration'] += 1
                elif analysis.migration_difficulty == 'medium':
                    report['summary']['medium_migration'] += 1
                elif analysis.migration_difficulty == 'hard':
                    report['summary']['hard_migration'] += 1
                else:
                    report['summary']['very_hard_migration'] += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze component {component_class.__name__}: {e}")
                report['components'].append({
                    'name': component_class.__name__,
                    'error': str(e),
                    'compatible': False
                })
        
        # Generate overall recommendations
        if report['summary']['compatible'] > 0:
            report['recommendations'].append("Start with easy migration components")
        
        if report['summary']['incompatible'] > 0:
            report['recommendations'].append("Refactor incompatible components before migration")
        
        if report['summary']['very_hard_migration'] > 0:
            report['recommendations'].append("Consider breaking down very hard components")
        
        return report


# Convenience functions
def analyze_component(component_class: Type) -> ComponentAnalysis:
    """Analyze a component for migration compatibility."""
    utils = ModelsTrainingMigrationUtils()
    return utils.analyze_component(component_class)


def validate_migration_compatibility(component_class: Type) -> bool:
    """Validate if a component can be migrated to ModularComponent."""
    utils = ModelsTrainingMigrationUtils()
    return utils.validate_migration_compatibility(component_class)


def create_component_wrapper(component_class: Type, wrapper_name: Optional[str] = None) -> Type[ModularComponent]:
    """Create a ModularComponent wrapper for an existing component."""
    utils = ModelsTrainingMigrationUtils()
    return utils.create_component_wrapper(component_class, wrapper_name)


def migrate_component(component_class: Type, migration_strategy: str = 'wrapper') -> MigrationResult:
    """Migrate a component to ModularComponent architecture."""
    utils = ModelsTrainingMigrationUtils()
    return utils.migrate_component(component_class, migration_strategy)


def generate_migration_report(components: List[Type]) -> Dict[str, Any]:
    """Generate a comprehensive migration report for multiple components."""
    utils = ModelsTrainingMigrationUtils()
    return utils.generate_migration_report(components)