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
from src.utils.tprint import tprint

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory
from .feature_generator import VectorizedFeatureGenerator
from .auto_optimized_feature_generator import AutoOptimizedFeatureGenerator
from .auto_optimization_config import AutoOptimizationConfig, OptimizationLevel
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
    
    def populate_from_feature_bank(self, feature_bank) -> None:
        """
        Populate the generator factory with generators from a FeatureBank.
        
        Args:
            feature_bank: FeatureBank instance to get generators from
        """
        try:
            # Get all generators from the feature bank's registry
            for category in FeatureCategory:
                generators = feature_bank.get_generators_by_category(category)
                for generator in generators:
                    # Register the generator in the factory
                    self.register_generator(
                        generator.config.name,
                        generator.__class__,
                        GeneratorConfig(
                            name=generator.config.name,
                            category=generator.config.category,
                            generator_class=generator.__class__,
                            required_columns=generator.config.required_columns,
                            description=generator.config.description
                        )
                    )
            logger.info(f"Populated generator factory with {len(self._generators)} generators")
        except Exception as e:
            logger.error(f"Failed to populate generator factory: {e}")

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
        logger.debug(f"Registered generator: {name}")

    def create_generator(self, name_or_config: Union[str, FeatureConfig], **kwargs) -> Optional[FeatureGenerator]:
        """
        Create a generator instance by name or from a FeatureConfig.

        Args:
            name_or_config: Generator name string or FeatureConfig object
            **kwargs: Generator parameters (ignored if FeatureConfig provided)

        Returns:
            Generator instance or None if not found
        """
        # Handle FeatureConfig objects
        if isinstance(name_or_config, FeatureConfig):
            try:
                # Extract the generator name from the config
                generator_name = name_or_config.name
                if generator_name not in self._generators:
                    logger.error(f"Generator not found: {generator_name}")
                    return None
                
                config = self._generators[generator_name]
                
                # Create generator instance with the provided FeatureConfig
                generator = config.generator_class(name_or_config)
                
                logger.debug(f"Created generator from config: {generator_name}")
                return generator
                
            except Exception as e:
                logger.error(f"Failed to create generator from config {name_or_config.name}: {e}")
                return None
        
        # Handle string names (original behavior)
        name = name_or_config
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
                    """Generate feature using the configured generator class."""
                    if hasattr(self, 'generator_instance'):
                        return self.generator_instance.generate(data, **kwargs)
                    else:
                        raise RuntimeError("Generator instance not properly initialized")

            # Create generator instance
            generator = OptimizedGenerator(feature_config)
            
            # Store the generator instance for feature generation
            generator.generator_instance = generator

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

    def create_auto_optimized_generator(self, name: str, category: FeatureCategory,
                                      required_columns: List[str],
                                      optimization_level: str = "balanced",
                                      auto_optimization_config: Optional[AutoOptimizationConfig] = None,
                                      **kwargs) -> Optional[AutoOptimizedFeatureGenerator]:
        """
        Create generator with automatic optimization enabled.

        Args:
            name: Generator name
            category: Feature category
            required_columns: Required input columns
            optimization_level: Optimization level ("conservative", "balanced", "aggressive")
            auto_optimization_config: Custom optimization configuration
            **kwargs: Generator parameters

        Returns:
            Auto-optimized generator instance or None if creation fails
        """
        try:
            tprint(f"🔧 Creating auto-optimized generator: {name}")
            tprint(f"📊 Category: {category.value}")
            tprint(f"📊 Optimization level: {optimization_level}")
            tprint(f"📊 Required columns: {required_columns}")

            # Create feature config
            tprint("📝 Creating feature configuration...")
            feature_config = FeatureConfig(
                name=name,
                category=category,
                description=f"Auto-optimized generator: {name}",
                required_columns=required_columns,
                parameters=kwargs
            )
            tprint("✅ Feature configuration created")

            # Create auto-optimization config
            tprint("⚙️ Setting up auto-optimization configuration...")
            if auto_optimization_config is None:
                tprint(f"📝 Creating new auto-optimization config with {optimization_level} level...")
                auto_optimization_config = AutoOptimizationConfig()
                auto_optimization_config.optimization_level = OptimizationLevel(optimization_level)
                tprint("✅ Auto-optimization config created")
            else:
                tprint("✅ Using provided auto-optimization config")

            # Create auto-optimized generator
            tprint("🚀 Creating AutoOptimizedFeatureGenerator...")
            generator = AutoOptimizedFeatureGenerator(feature_config, auto_optimization_config)

            logger.debug(f"Created auto-optimized generator: {name} with {optimization_level} strategy")
            tprint(f"✅ Auto-optimized generator '{name}' created successfully")
            return generator

        except Exception as e:
            tprint(f"❌ Failed to create auto-optimized generator '{name}': {e}")
            logger.error(f"Failed to create auto-optimized generator {name}: {e}")
            return None

    def create_generator_with_auto_optimization(self, name: str,
                                              generator_class: Type[FeatureGenerator],
                                              config: FeatureConfig,
                                              optimization_level: str = "balanced",
                                              **kwargs) -> Optional[FeatureGenerator]:
        """
        Create any generator with auto-optimization enabled.

        Args:
            name: Generator name
            generator_class: Generator class
            config: Feature configuration
            optimization_level: Optimization level
            **kwargs: Additional parameters

        Returns:
            Generator instance with auto-optimization or None if creation fails
        """
        try:
            # Create auto-optimization config
            auto_optimization_config = AutoOptimizationConfig()
            auto_optimization_config.optimization_level = OptimizationLevel(optimization_level)

            # Create generator with auto-optimization
            generator = generator_class(config, auto_optimization_config=auto_optimization_config, **kwargs)

            logger.debug(f"Created {name} with auto-optimization enabled")
            return generator

        except Exception as e:
            logger.error(f"Failed to create {name} with auto-optimization: {e}")
            return None

    def create_batch_auto_optimized_generators(self, generator_specs: List[Dict[str, Any]]) -> List[AutoOptimizedFeatureGenerator]:
        """
        Create multiple auto-optimized generators in batch.

        Args:
            generator_specs: List of generator specifications

        Returns:
            List of created auto-optimized generators
        """
        generators = []

        for spec in generator_specs:
            name = spec.get('name')
            if not name:
                logger.warning("Generator spec missing name, skipping")
                continue

            # Extract auto-optimization settings
            optimization_level = spec.pop('optimization_level', 'balanced')
            auto_optimization_config = spec.pop('auto_optimization_config', None)

            generator = self.create_auto_optimized_generator(
                name=name,
                category=spec.get('category', FeatureCategory.CUSTOM),
                required_columns=spec.get('required_columns', []),
                optimization_level=optimization_level,
                auto_optimization_config=auto_optimization_config,
                **spec.get('parameters', {})
            )

            if generator:
                generators.append(generator)

        logger.info(f"Created {len(generators)} auto-optimized generators in batch")
        return generators

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
