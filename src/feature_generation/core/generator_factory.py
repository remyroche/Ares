"""
Generator Factory

This module provides a factory pattern for creating feature generators programmatically,
reducing duplicate generator creation code and enabling dynamic feature generation.

Usage:
    factory = GeneratorFactory()
    generator = factory.create_generator('sma', window=20)
    result = generator.generate(data)
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from .vectorized_feature_generator import VectorizedFeatureGenerator
from .vectorbt_optimization_mixin import VectorBTOptimizationMixin
from .optimization_mixin import OptimizationMixin
from .rolling_operations_mixin import RollingOperationsMixin

logger = logging.getLogger(__name__)

@dataclass
class GeneratorConfig:
    """Configuration for generator creation."""
    name: str
    category: FeatureCategory
    generator_class: Type[FeatureGenerator]
    required_columns: List[str]
    optional_columns: List[str] = None
    default_parameters: Dict[str, Any] = None
    description: str = ""
    
    def __post_init__(self):
        if self.optional_columns is None:
            self.optional_columns = []
        if self.default_parameters is None:
            self.default_parameters = {}

class GeneratorFactory:
    """Factory for creating feature generators programmatically."""
    
    def __init__(self):
        self._generators: Dict[str, GeneratorConfig] = {}
        self._custom_generators: Dict[str, Type[FeatureGenerator]] = {}
        self._setup_default_generators()
    
    def _setup_default_generators(self):
        """Setup default generator configurations."""
        # This would be populated with actual generator classes
        # For now, we'll create a basic structure
        pass
    
    def register_generator(self, name: str, generator_class: Type[FeatureGenerator], 
                          config: Optional[GeneratorConfig] = None) -> None:
        """
        Register a generator class with the factory.
        
        Args:
            name: Generator name
            generator_class: Generator class
            config: Optional generator configuration
        """
        if config is None:
            # Create basic config from generator class
            config = GeneratorConfig(
                name=name,
                category=FeatureCategory.CUSTOM,
                generator_class=generator_class,
                required_columns=[],
                description=f"Custom generator: {name}"
            )
        
        self._generators[name] = config
        self._custom_generators[name] = generator_class
        logger.info(f"Registered generator: {name}")
    
    def create_generator(self, name: str, **kwargs) -> Optional[FeatureGenerator]:
        """
        Create a generator instance by name.
        
        Args:
            name: Generator name
            **kwargs: Generator parameters
            
        Returns:
            Generator instance or None if not found
        """
        if name not in self._generators:
            logger.error(f"Generator not found: {name}")
            return None
        
        config = self._generators[name]
        
        try:
            # Create feature config
            feature_config = FeatureConfig(
                name=kwargs.get('name', name),
                category=config.category,
                description=config.description,
                required_columns=config.required_columns,
                optional_columns=config.optional_columns,
                parameters=kwargs
            )
            
            # Create generator instance
            generator = config.generator_class(feature_config)
            
            logger.debug(f"Created generator: {name}")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to create generator {name}: {e}")
            return None
    
    def create_custom_generator(self, name: str, generator_class: Type[FeatureGenerator], 
                               config: FeatureConfig, **kwargs) -> Optional[FeatureGenerator]:
        """
        Create a custom generator instance.
        
        Args:
            name: Generator name
            generator_class: Generator class
            config: Feature configuration
            **kwargs: Additional parameters
            
        Returns:
            Generator instance or None if creation fails
        """
        try:
            # Update config with kwargs
            config.parameters.update(kwargs)
            
            # Create generator instance
            generator = generator_class(config)
            
            logger.debug(f"Created custom generator: {name}")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to create custom generator {name}: {e}")
            return None
    
    def create_vectorized_generator(self, name: str, category: FeatureCategory,
                                   required_columns: List[str], **kwargs) -> Optional[VectorizedFeatureGenerator]:
        """
        Create a vectorized generator instance.
        
        Args:
            name: Generator name
            category: Feature category
            required_columns: Required input columns
            **kwargs: Generator parameters
            
        Returns:
            Vectorized generator instance or None if creation fails
        """
        try:
            # Create feature config
            feature_config = FeatureConfig(
                name=name,
                category=category,
                description=f"Vectorized generator: {name}",
                required_columns=required_columns,
                parameters=kwargs
            )
            
            # Create vectorized generator
            generator = VectorizedFeatureGenerator(feature_config)
            
            logger.debug(f"Created vectorized generator: {name}")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to create vectorized generator {name}: {e}")
            return None
    
    def create_optimized_generator(self, name: str, category: FeatureCategory,
                                  required_columns: List[str], **kwargs) -> Optional[FeatureGenerator]:
        """
        Create an optimized generator with all mixins.
        
        Args:
            name: Generator name
            category: Feature category
            required_columns: Required input columns
            **kwargs: Generator parameters
            
        Returns:
            Optimized generator instance or None if creation fails
        """
        try:
            # Create feature config
            feature_config = FeatureConfig(
                name=name,
                category=category,
                description=f"Optimized generator: {name}",
                required_columns=required_columns,
                parameters=kwargs
            )
            
            # Create optimized generator class
            class OptimizedGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin, 
                                   OptimizationMixin, RollingOperationsMixin):
                def _generate_feature(self, data, **kwargs):
                    # This would be implemented by the specific generator
                    raise NotImplementedError("_generate_feature must be implemented")
            
            # Create generator instance
            generator = OptimizedGenerator(feature_config)
            
            logger.debug(f"Created optimized generator: {name}")
            return generator
            
        except Exception as e:
            logger.error(f"Failed to create optimized generator {name}: {e}")
            return None
    
    def list_available_generators(self) -> List[str]:
        """List all available generator names."""
        return list(self._generators.keys())
    
    def get_generator_config(self, name: str) -> Optional[GeneratorConfig]:
        """Get generator configuration by name."""
        return self._generators.get(name)
    
    def create_batch_generators(self, generator_specs: List[Dict[str, Any]]) -> List[FeatureGenerator]:
        """
        Create multiple generators in batch.
        
        Args:
            generator_specs: List of generator specifications
            
        Returns:
            List of created generators
        """
        generators = []
        
        for spec in generator_specs:
            name = spec.get('name')
            if not name:
                logger.warning("Generator spec missing name, skipping")
                continue
            
            generator = self.create_generator(name, **spec.get('parameters', {}))
            if generator:
                generators.append(generator)
        
        logger.info(f"Created {len(generators)} generators in batch")
        return generators
    
    def create_generator_from_template(self, template_name: str, name: str, **kwargs) -> Optional[FeatureGenerator]:
        """
        Create a generator from a template.
        
        Args:
            template_name: Template generator name
            name: New generator name
            **kwargs: Override parameters
            
        Returns:
            Generator instance or None if creation fails
        """
        if template_name not in self._generators:
            logger.error(f"Template generator not found: {template_name}")
            return None
        
        template_config = self._generators[template_name]
        
        # Create new config based on template
        new_config = GeneratorConfig(
            name=name,
            category=template_config.category,
            generator_class=template_config.generator_class,
            required_columns=template_config.required_columns.copy(),
            optional_columns=template_config.optional_columns.copy() if template_config.optional_columns else [],
            default_parameters=template_config.default_parameters.copy(),
            description=f"Generated from template: {template_name}"
        )
        
        # Update with kwargs
        new_config.default_parameters.update(kwargs)
        
        # Create generator
        return self.create_custom_generator(name, template_config.generator_class, 
                                          FeatureConfig(
                                              name=name,
                                              category=new_config.category,
                                              description=new_config.description,
                                              required_columns=new_config.required_columns,
                                              optional_columns=new_config.optional_columns,
                                              parameters=new_config.default_parameters
                                          ))

# Global factory instance
_global_factory = None

def get_generator_factory() -> GeneratorFactory:
    """Get the global generator factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = GeneratorFactory()
    return _global_factory

def create_generator(name: str, **kwargs) -> Optional[FeatureGenerator]:
    """Convenience function to create a generator."""
    factory = get_generator_factory()
    return factory.create_generator(name, **kwargs)

def register_generator(name: str, generator_class: Type[FeatureGenerator], 
                      config: Optional[GeneratorConfig] = None) -> None:
    """Convenience function to register a generator."""
    factory = get_generator_factory()
    factory.register_generator(name, generator_class, config)