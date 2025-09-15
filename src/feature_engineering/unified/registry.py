"""
Feature Generator Registry System

This module provides a centralized registry for discovering, managing,
and coordinating feature generators across the system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any, Callable
from pathlib import Path
import importlib
import inspect
from dataclasses import dataclass

from .core import FeatureGenerator, FeatureGeneratorConfig, FeatureCategory, FeaturePriority
from ...utils.logger import system_logger


@dataclass
class GeneratorInfo:
    """Information about a registered feature generator."""
    name: str
    generator_class: Type[FeatureGenerator]
    config: FeatureGeneratorConfig
    module_path: str
    is_loaded: bool = False
    instance: Optional[FeatureGenerator] = None


class FeatureRegistry:
    """
    Central registry for feature generators.
    
    Provides discovery, loading, and management of feature generators
    with support for dynamic loading and dependency resolution.
    """
    
    def __init__(self):
        """Initialize the feature registry."""
        self.logger = system_logger.getChild("FeatureRegistry")
        self._generators: Dict[str, GeneratorInfo] = {}
        self._categories: Dict[FeatureCategory, List[str]] = {}
        self._initialized = False
        
        # Initialize category mapping
        for category in FeatureCategory:
            self._categories[category] = []
    
    async def initialize(self) -> bool:
        """Initialize the registry and discover available generators."""
        try:
            self.logger.info("Initializing feature generator registry...")
            
            # Discover generators from various locations
            await self._discover_generators()
            
            # Resolve dependencies
            await self._resolve_dependencies()
            
            self._initialized = True
            self.logger.info(f"Registry initialized with {len(self._generators)} generators")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing registry: {e}")
            return False
    
    async def _discover_generators(self) -> None:
        """Discover available feature generators."""
        discovery_paths = [
            "src.feature_engineering.generators",
            "src.analyst.feature_generators", 
            "src.utils.ml_common.feature_generators",
            "src.training.feature_generators"
        ]
        
        for module_path in discovery_paths:
            try:
                await self._discover_from_module(module_path)
            except Exception as e:
                self.logger.warning(f"Could not discover from {module_path}: {e}")
    
    async def _discover_from_module(self, module_path: str) -> None:
        """Discover generators from a specific module."""
        try:
            module = importlib.import_module(module_path)
            
            # Look for classes that inherit from FeatureGenerator
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, FeatureGenerator) and 
                    obj != FeatureGenerator and 
                    not inspect.isabstract(obj)):
                    
                    # Try to get default config
                    config = self._create_default_config(obj, name)
                    
                    generator_info = GeneratorInfo(
                        name=name,
                        generator_class=obj,
                        config=config,
                        module_path=module_path
                    )
                    
                    self._register_generator_info(generator_info)
                    
        except ImportError:
            # Module doesn't exist, skip silently
            pass
        except Exception as e:
            self.logger.warning(f"Error discovering from {module_path}: {e}")
    
    def _create_default_config(self, generator_class: Type[FeatureGenerator], name: str) -> FeatureGeneratorConfig:
        """Create default configuration for a generator class."""
        # Try to get category from class attributes or name
        category = self._infer_category(name, generator_class)
        
        return FeatureGeneratorConfig(
            name=name,
            category=category,
            priority=FeaturePriority.MEDIUM,
            enabled=True
        )
    
    def _infer_category(self, name: str, generator_class: Type[FeatureGenerator]) -> FeatureCategory:
        """Infer category from generator name or class."""
        name_lower = name.lower()
        
        # Category mapping based on naming patterns
        category_mapping = {
            'technical': FeatureCategory.TECHNICAL_INDICATORS,
            'indicator': FeatureCategory.TECHNICAL_INDICATORS,
            'ta': FeatureCategory.TECHNICAL_INDICATORS,
            'statistical': FeatureCategory.STATISTICAL_FEATURES,
            'stats': FeatureCategory.STATISTICAL_FEATURES,
            'microstructure': FeatureCategory.MICROSTRUCTURE,
            'volatility': FeatureCategory.VOLATILITY,
            'vol': FeatureCategory.VOLATILITY,
            'momentum': FeatureCategory.MOMENTUM,
            'volume': FeatureCategory.VOLUME,
            'time': FeatureCategory.TIME_SERIES,
            'temporal': FeatureCategory.TIME_SERIES,
            'cross': FeatureCategory.CROSS_TIMEFRAME,
            'multi': FeatureCategory.CROSS_TIMEFRAME,
            'meta': FeatureCategory.META_LABELING,
            'pattern': FeatureCategory.PATTERN_RECOGNITION,
            'regime': FeatureCategory.REGIME_DETECTION,
            'liquidity': FeatureCategory.LIQUIDITY
        }
        
        for keyword, category in category_mapping.items():
            if keyword in name_lower:
                return category
                
        return FeatureCategory.CUSTOM
    
    def _register_generator_info(self, info: GeneratorInfo) -> None:
        """Register a generator info object."""
        self._generators[info.name] = info
        self._categories[info.config.category].append(info.name)
        self.logger.debug(f"Registered generator: {info.name} ({info.config.category.value})")
    
    async def _resolve_dependencies(self) -> None:
        """Resolve dependencies between generators."""
        # Simple dependency resolution - can be enhanced
        for name, info in self._generators.items():
            if info.config.dependencies:
                self.logger.debug(f"Generator {name} has dependencies: {info.config.dependencies}")
    
    def register_generator(
        self, 
        name: str, 
        generator_class: Type[FeatureGenerator],
        config: FeatureGeneratorConfig
    ) -> bool:
        """
        Manually register a feature generator.
        
        Args:
            name: Unique name for the generator
            generator_class: Generator class
            config: Configuration for the generator
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if name in self._generators:
                self.logger.warning(f"Generator {name} already registered, updating...")
            
            generator_info = GeneratorInfo(
                name=name,
                generator_class=generator_class,
                config=config,
                module_path="manual"
            )
            
            self._register_generator_info(generator_info)
            self.logger.info(f"Registered generator: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering generator {name}: {e}")
            return False
    
    def get_generator(self, name: str) -> Optional[GeneratorInfo]:
        """Get generator information by name."""
        return self._generators.get(name)
    
    def get_generators_by_category(self, category: FeatureCategory) -> List[GeneratorInfo]:
        """Get all generators in a specific category."""
        generator_names = self._categories.get(category, [])
        return [self._generators[name] for name in generator_names if name in self._generators]
    
    def get_enabled_generators(self) -> List[GeneratorInfo]:
        """Get all enabled generators."""
        return [info for info in self._generators.values() if info.config.enabled]
    
    def get_generators_by_priority(self, priority: FeaturePriority) -> List[GeneratorInfo]:
        """Get generators by priority level."""
        return [info for info in self._generators.values() if info.config.priority == priority]
    
    def list_available_generators(self) -> List[str]:
        """List all available generator names."""
        return list(self._generators.keys())
    
    def get_generator_categories(self) -> Dict[FeatureCategory, List[str]]:
        """Get generators organized by category."""
        return {cat: names.copy() for cat, names in self._categories.items()}
    
    async def create_generator_instance(self, name: str) -> Optional[FeatureGenerator]:
        """
        Create an instance of a generator.
        
        Args:
            name: Name of the generator to create
            
        Returns:
            Generator instance or None if creation failed
        """
        info = self.get_generator(name)
        if not info:
            self.logger.error(f"Generator {name} not found")
            return None
            
        try:
            if info.instance is None:
                info.instance = info.generator_class(info.config)
                info.is_loaded = True
                
            return info.instance
            
        except Exception as e:
            self.logger.error(f"Error creating generator instance {name}: {e}")
            return None
    
    def enable_generator(self, name: str) -> bool:
        """Enable a generator."""
        info = self.get_generator(name)
        if info:
            info.config.enabled = True
            self.logger.info(f"Enabled generator: {name}")
            return True
        return False
    
    def disable_generator(self, name: str) -> bool:
        """Disable a generator."""
        info = self.get_generator(name)
        if info:
            info.config.enabled = False
            self.logger.info(f"Disabled generator: {name}")
            return True
        return False
    
    def update_generator_config(self, name: str, config_updates: Dict[str, Any]) -> bool:
        """Update generator configuration."""
        info = self.get_generator(name)
        if not info:
            return False
            
        try:
            for key, value in config_updates.items():
                if hasattr(info.config, key):
                    setattr(info.config, key, value)
                else:
                    info.config.parameters[key] = value
                    
            self.logger.info(f"Updated config for generator: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating config for {name}: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_generators": len(self._generators),
            "enabled_generators": len(self.get_enabled_generators()),
            "categories": {cat.value: len(names) for cat, names in self._categories.items()},
            "initialized": self._initialized
        }


# Global registry instance
_registry = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    return _registry


def register_feature_generator(
    name: str,
    generator_class: Type[FeatureGenerator],
    config: FeatureGeneratorConfig
) -> bool:
    """Register a feature generator with the global registry."""
    return _registry.register_generator(name, generator_class, config)


def get_feature_generator(name: str) -> Optional[GeneratorInfo]:
    """Get a feature generator from the global registry."""
    return _registry.get_generator(name)


def list_available_generators() -> List[str]:
    """List all available generators in the global registry."""
    return _registry.list_available_generators()