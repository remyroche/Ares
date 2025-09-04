"""
Plugin Registry

Manages registration and discovery of plugins.
"""

import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, Union
from .base_plugin import BasePlugin, PluginCategory, PluginPriority
from .exceptions import PluginError, PluginNotFoundError


class PluginRegistry:
    """
    Registry for managing plugin discovery and registration.
    """
    
    def __init__(self):
        self._plugins: Dict[str, Type[BasePlugin]] = {}
        self._plugin_instances: Dict[str, BasePlugin] = {}
        self._categories: Dict[PluginCategory, Set[str]] = {}
        self._priorities: Dict[PluginPriority, Set[str]] = {}
        self._dependencies: Dict[str, Set[str]] = {}
    
    def register_plugin(self, plugin_class: Type[BasePlugin], 
                       instance: Optional[BasePlugin] = None) -> None:
        """
        Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
            instance: Optional plugin instance
        """
        # Create a temporary instance to get metadata
        temp_instance = plugin_class()
        metadata = temp_instance.get_metadata()
        
        plugin_name = metadata.name
        
        if plugin_name in self._plugins:
            raise PluginError(f"Plugin '{plugin_name}' is already registered")
        
        # Validate plugin
        self._validate_plugin(plugin_class)
        
        # Register plugin
        self._plugins[plugin_name] = plugin_class
        
        if instance:
            self._plugin_instances[plugin_name] = instance
        else:
            self._plugin_instances[plugin_name] = plugin_class()
        
        # Update category and priority mappings
        self._categories.setdefault(metadata.category, set()).add(plugin_name)
        self._priorities.setdefault(metadata.priority, set()).add(plugin_name)
        
        # Build dependency graph
        self._dependencies[plugin_name] = set(metadata.dependencies)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(plugin_name)
        
        # Remove from all mappings
        plugin_class = self._plugins[plugin_name]
        temp_instance = plugin_class()
        metadata = temp_instance.get_metadata()
        
        del self._plugins[plugin_name]
        del self._plugin_instances[plugin_name]
        
        self._categories[metadata.category].discard(plugin_name)
        self._priorities[metadata.priority].discard(plugin_name)
        
        if plugin_name in self._dependencies:
            del self._dependencies[plugin_name]
        
        # Remove from other plugins' dependencies
        for deps in self._dependencies.values():
            deps.discard(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> BasePlugin:
        """
        Get a plugin instance by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            BasePlugin: Plugin instance
            
        Raises:
            PluginNotFoundError: If plugin is not found
        """
        if plugin_name not in self._plugin_instances:
            raise PluginNotFoundError(plugin_name)
        
        return self._plugin_instances[plugin_name]
    
    def get_plugin_class(self, plugin_name: str) -> Type[BasePlugin]:
        """
        Get a plugin class by name.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Type[BasePlugin]: Plugin class
            
        Raises:
            PluginNotFoundError: If plugin is not found
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(plugin_name)
        
        return self._plugins[plugin_name]
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.
        
        Returns:
            List[str]: List of plugin names
        """
        return list(self._plugins.keys())
    
    def list_plugins_by_category(self, category: PluginCategory) -> List[str]:
        """
        List plugins in a specific category.
        
        Args:
            category: Plugin category
            
        Returns:
            List[str]: List of plugin names in the category
        """
        return list(self._categories.get(category, set()))
    
    def list_plugins_by_priority(self, priority: PluginPriority) -> List[str]:
        """
        List plugins with a specific priority.
        
        Args:
            priority: Plugin priority
            
        Returns:
            List[str]: List of plugin names with the priority
        """
        return list(self._priorities.get(priority, set()))
    
    def get_available_plugins(self) -> List[str]:
        """
        Get list of available plugins (dependencies met).
        
        Returns:
            List[str]: List of available plugin names
        """
        available = []
        
        for plugin_name, plugin_instance in self._plugin_instances.items():
            if plugin_instance.is_available():
                available.append(plugin_name)
        
        return available
    
    def get_unavailable_plugins(self) -> Dict[str, List[str]]:
        """
        Get list of unavailable plugins and their missing dependencies.
        
        Returns:
            Dict[str, List[str]]: Mapping of plugin names to missing dependencies
        """
        unavailable = {}
        
        for plugin_name, plugin_instance in self._plugin_instances.items():
            if not plugin_instance.is_available():
                missing_deps = plugin_instance.validate_dependencies()
                unavailable[plugin_name] = missing_deps
        
        return unavailable
    
    def discover_plugins(self, directory: Union[str, Path]) -> int:
        """
        Discover and register plugins in a directory.
        
        Args:
            directory: Directory to search for plugins
            
        Returns:
            int: Number of plugins discovered and registered
        """
        directory = Path(directory)
        discovered_count = 0
        
        if not directory.exists():
            return discovered_count
        
        # Add directory to Python path
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        
        # Find Python files
        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                # Import the module
                module_name = py_file.stem
                module_path = py_file.parent
                
                # Create module path relative to directory
                relative_path = py_file.relative_to(directory)
                module_parts = list(relative_path.parts[:-1]) + [module_name]
                full_module_name = ".".join(module_parts)
                
                module = importlib.import_module(full_module_name)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BasePlugin) and 
                        obj != BasePlugin and 
                        not inspect.isabstract(obj)):
                        
                        try:
                            self.register_plugin(obj)
                            discovered_count += 1
                        except PluginError as e:
                            print(f"Warning: Failed to register plugin {name}: {e}")
                
            except Exception as e:
                print(f"Warning: Failed to import {py_file}: {e}")
        
        return discovered_count
    
    def get_execution_order(self, plugin_names: Optional[List[str]] = None) -> List[str]:
        """
        Get plugins in execution order based on dependencies and priorities.
        
        Args:
            plugin_names: Optional list of plugin names to order.
                         If None, orders all registered plugins.
        
        Returns:
            List[str]: Plugin names in execution order
        """
        if plugin_names is None:
            plugin_names = list(self._plugins.keys())
        
        # Topological sort based on dependencies
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(plugin_name: str):
            if plugin_name in temp_visited:
                raise PluginError(f"Circular dependency detected involving '{plugin_name}'")
            
            if plugin_name in visited:
                return
            
            temp_visited.add(plugin_name)
            
            # Visit dependencies first
            for dep in self._dependencies.get(plugin_name, set()):
                if dep in plugin_names:  # Only consider requested plugins
                    visit(dep)
            
            temp_visited.remove(plugin_name)
            visited.add(plugin_name)
            result.append(plugin_name)
        
        # Visit all plugins
        for plugin_name in plugin_names:
            if plugin_name not in visited:
                visit(plugin_name)
        
        # Sort by priority within each dependency level
        priority_order = [
            PluginPriority.CRITICAL,
            PluginPriority.HIGH,
            PluginPriority.MEDIUM,
            PluginPriority.LOW,
            PluginPriority.OPTIONAL
        ]
        
        def get_priority(plugin_name: str) -> int:
            plugin_instance = self._plugin_instances[plugin_name]
            metadata = plugin_instance.get_metadata()
            return priority_order.index(metadata.priority)
        
        result.sort(key=get_priority)
        
        return result
    
    def _validate_plugin(self, plugin_class: Type[BasePlugin]) -> None:
        """
        Validate a plugin class.
        
        Args:
            plugin_class: Plugin class to validate
            
        Raises:
            PluginError: If plugin is invalid
        """
        # Check if it's a proper subclass
        if not issubclass(plugin_class, BasePlugin):
            raise PluginError(f"Plugin class must inherit from BasePlugin")
        
        # Check if it's abstract
        if inspect.isabstract(plugin_class):
            raise PluginError(f"Plugin class cannot be abstract")
        
        # Try to create an instance to validate
        try:
            instance = plugin_class()
            metadata = instance.get_metadata()
            
            # Validate metadata
            if not metadata.name:
                raise PluginError("Plugin metadata must have a name")
            
            if not metadata.version:
                raise PluginError("Plugin metadata must have a version")
            
        except Exception as e:
            raise PluginError(f"Plugin validation failed: {e}")
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, any]:
        """
        Get detailed information about a plugin.
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Dict[str, any]: Plugin information
        """
        if plugin_name not in self._plugin_instances:
            raise PluginNotFoundError(plugin_name)
        
        plugin_instance = self._plugin_instances[plugin_name]
        metadata = plugin_instance.get_metadata()
        
        return {
            "name": metadata.name,
            "version": metadata.version,
            "description": metadata.description,
            "author": metadata.author,
            "category": metadata.category.value,
            "priority": metadata.priority.value,
            "dependencies": metadata.dependencies,
            "tags": list(metadata.tags),
            "min_python_version": metadata.min_python_version,
            "max_python_version": metadata.max_python_version,
            "required_packages": metadata.required_packages,
            "optional_packages": metadata.optional_packages,
            "is_available": plugin_instance.is_available(),
            "missing_dependencies": plugin_instance.validate_dependencies(),
            "supported_file_types": list(plugin_instance.get_supported_file_types())
        }
    
    def get_registry_summary(self) -> Dict[str, any]:
        """
        Get a summary of the plugin registry.
        
        Returns:
            Dict[str, any]: Registry summary
        """
        total_plugins = len(self._plugins)
        available_plugins = len(self.get_available_plugins())
        unavailable_plugins = len(self.get_unavailable_plugins())
        
        category_counts = {
            category.value: len(plugins)
            for category, plugins in self._categories.items()
        }
        
        priority_counts = {
            priority.value: len(plugins)
            for priority, plugins in self._priorities.items()
        }
        
        return {
            "total_plugins": total_plugins,
            "available_plugins": available_plugins,
            "unavailable_plugins": unavailable_plugins,
            "categories": category_counts,
            "priorities": priority_counts,
            "plugin_names": list(self._plugins.keys())
        }