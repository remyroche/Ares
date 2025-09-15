"""
Feature Registry

This module provides the FeatureRegistry class for managing and organizing
feature generators by category and name.
"""

import logging
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .feature_generator import FeatureGenerator, FeatureCategory

logger = logging.getLogger(__name__)

class FeatureRegistry:
    """
    Registry for managing feature generators by category and name.
    
    The FeatureRegistry provides a centralized way to organize and access
    feature generators, making it easy to find generators by category
    or specific feature name.
    """
    
    def __init__(self):
        """Initialize the feature registry."""
        self.logger = logger.getChild('FeatureRegistry')
        
        # Registry storage
        self._generators_by_name: Dict[str, FeatureGenerator] = {}
        self._generators_by_category: Dict[FeatureCategory, List[FeatureGenerator]] = defaultdict(list)
        self._category_names: Set[str] = set()
        
        self.logger.info("FeatureRegistry initialized")
    
    def register(self, generator: FeatureGenerator) -> None:
        """
        Register a feature generator.
        
        Args:
            generator: Feature generator to register
        """
        name = generator.config.name
        category = generator.config.category
        
        # Check for duplicate names
        if name in self._generators_by_name:
            self.logger.warning(f"Overwriting existing generator: {name}")
        
        # Register by name
        self._generators_by_name[name] = generator
        
        # Register by category
        if generator not in self._generators_by_category[category]:
            self._generators_by_category[category].append(generator)
        
        # Track category names
        self._category_names.add(category.value)
        
        self.logger.debug(f"Registered generator: {name} ({category.value})")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a feature generator by name.
        
        Args:
            name: Name of the generator to unregister
            
        Returns:
            True if generator was found and unregistered
        """
        if name not in self._generators_by_name:
            self.logger.warning(f"Generator not found: {name}")
            return False
        
        generator = self._generators_by_name[name]
        category = generator.config.category
        
        # Remove from name registry
        del self._generators_by_name[name]
        
        # Remove from category registry
        if generator in self._generators_by_category[category]:
            self._generators_by_category[category].remove(generator)
        
        # Clean up empty categories
        if not self._generators_by_category[category]:
            del self._generators_by_category[category]
            self._category_names.discard(category.value)
        
        self.logger.debug(f"Unregistered generator: {name}")
        return True
    
    def get_by_name(self, name: str) -> Optional[FeatureGenerator]:
        """
        Get a generator by name.
        
        Args:
            name: Generator name
            
        Returns:
            Generator or None if not found
        """
        return self._generators_by_name.get(name)
    
    def get_by_category(self, category: FeatureCategory) -> List[FeatureGenerator]:
        """
        Get all generators for a category.
        
        Args:
            category: Feature category
            
        Returns:
            List of generators for the category
        """
        return self._generators_by_category.get(category, []).copy()
    
    def get_all(self) -> List[FeatureGenerator]:
        """
        Get all registered generators.
        
        Returns:
            List of all generators
        """
        return list(self._generators_by_name.values())
    
    def list_names(self) -> List[str]:
        """
        List all registered generator names.
        
        Returns:
            List of generator names
        """
        return list(self._generators_by_name.keys())
    
    def list_categories(self) -> List[FeatureCategory]:
        """
        List all categories that have registered generators.
        
        Returns:
            List of categories
        """
        return list(self._generators_by_category.keys())
    
    def list_features(self, category: Optional[FeatureCategory] = None) -> List[str]:
        """
        List feature names, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of feature names
        """
        if category is None:
            return self.list_names()
        
        generators = self.get_by_category(category)
        return [gen.config.name for gen in generators]
    
    def get_category_stats(self) -> Dict[str, int]:
        """
        Get statistics about generators per category.
        
        Returns:
            Dictionary mapping category names to generator counts
        """
        stats = {}
        for category, generators in self._generators_by_category.items():
            stats[category.value] = len(generators)
        return stats
    
    def search_generators(self, 
                         name_pattern: Optional[str] = None,
                         category: Optional[FeatureCategory] = None,
                         description_pattern: Optional[str] = None) -> List[FeatureGenerator]:
        """
        Search for generators based on various criteria.
        
        Args:
            name_pattern: Pattern to match in generator names
            category: Category filter
            description_pattern: Pattern to match in descriptions
            
        Returns:
            List of matching generators
        """
        generators = self.get_all()
        
        # Filter by category
        if category is not None:
            generators = [g for g in generators if g.config.category == category]
        
        # Filter by name pattern
        if name_pattern is not None:
            generators = [g for g in generators if name_pattern.lower() in g.config.name.lower()]
        
        # Filter by description pattern
        if description_pattern is not None:
            generators = [g for g in generators 
                         if description_pattern.lower() in g.config.description.lower()]
        
        return generators
    
    def get_dependencies(self, name: str) -> List[str]:
        """
        Get dependencies for a generator.
        
        Args:
            name: Generator name
            
        Returns:
            List of dependency names
        """
        generator = self.get_by_name(name)
        if generator is None:
            return []
        
        return generator.config.dependencies.copy()
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """
        Validate that all dependencies are satisfied.
        
        Returns:
            Dictionary mapping generator names to missing dependencies
        """
        missing_deps = {}
        
        for name, generator in self._generators_by_name.items():
            missing = []
            for dep in generator.config.dependencies:
                if dep not in self._generators_by_name:
                    missing.append(dep)
            
            if missing:
                missing_deps[name] = missing
        
        return missing_deps
    
    def get_generators_with_dependencies(self, name: str) -> List[FeatureGenerator]:
        """
        Get a generator and all its dependencies in dependency order.
        
        Args:
            name: Generator name
            
        Returns:
            List of generators in dependency order
        """
        generator = self.get_by_name(name)
        if generator is None:
            return []
        
        # Simple dependency resolution (no circular dependency detection)
        resolved = []
        to_resolve = [name]
        resolved_names = set()
        
        while to_resolve:
            current_name = to_resolve.pop(0)
            if current_name in resolved_names:
                continue
            
            current_generator = self.get_by_name(current_name)
            if current_generator is None:
                self.logger.warning(f"Dependency not found: {current_name}")
                continue
            
            # Add dependencies to resolution queue
            for dep in current_generator.config.dependencies:
                if dep not in resolved_names:
                    to_resolve.append(dep)
            
            resolved.append(current_generator)
            resolved_names.add(current_name)
        
        return resolved
    
    def clear(self) -> None:
        """Clear all registered generators."""
        self._generators_by_name.clear()
        self._generators_by_category.clear()
        self._category_names.clear()
        self.logger.info("Registry cleared")
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry summary
        """
        return {
            'total_generators': len(self._generators_by_name),
            'categories': list(self._category_names),
            'category_stats': self.get_category_stats(),
            'missing_dependencies': self.validate_dependencies()
        }