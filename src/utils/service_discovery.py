"""
Service discovery utilities for automatic service registration.

This module provides comprehensive service discovery functionality that can
automatically scan directories for services and register them with a dependency
injection container.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Any, Optional, Dict, List, Type, Set, Callable
from pathlib import Path
import pkgutil

logger = logging.getLogger(__name__)

# Service decorator registry
_service_registry: Dict[str, Dict[str, Any]] = {}

def service(name: Optional[str] = None, singleton: bool = True, dependencies: Optional[List[str]] = None):
    """
    Decorator to mark a class as a service for automatic discovery.
    
    Args:
        name: Optional service name (defaults to class name)
        singleton: Whether this service should be a singleton
        dependencies: List of dependency service names
    """
    def decorator(cls: Type) -> Type:
        service_name = name or cls.__name__
        _service_registry[service_name] = {
            'class': cls,
            'singleton': singleton,
            'dependencies': dependencies or [],
            'module': cls.__module__
        }
        logger.debug(f"Registered service decorator: {service_name} -> {cls.__name__}")
        return cls
    return decorator

def factory(name: Optional[str] = None, dependencies: Optional[List[str]] = None):
    """
    Decorator to mark a function as a service factory for automatic discovery.
    
    Args:
        name: Optional service name (defaults to function name)
        dependencies: List of dependency service names
    """
    def decorator(func: Callable) -> Callable:
        service_name = name or func.__name__
        _service_registry[service_name] = {
            'factory': func,
            'singleton': True,  # Factories are typically singletons
            'dependencies': dependencies or [],
            'module': func.__module__
        }
        logger.debug(f"Registered factory decorator: {service_name} -> {func.__name__}")
        return func
    return decorator

def discover_and_register_services(container: Any, base_path: str) -> None:
    """
    Discover and register services automatically.

    Args:
        container: The dependency injection container
        base_path: Base path to scan for services
    """
    logger.info(f"🔍 Scanning {base_path} for services to register...")
    
    try:
        # Convert to Path object for easier manipulation
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return
        
        # Discover services from the registry first
        _register_discovered_services(container)
        
        # Then scan for additional services in the file system
        _scan_filesystem_for_services(container, base_path_obj)
        
        logger.info("✅ Service discovery completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Service discovery failed: {e}")
        raise

def _register_discovered_services(container: Any) -> None:
    """Register services that were discovered via decorators."""
    logger.info(f"Registering {len(_service_registry)} discovered services...")
    
    for service_name, service_info in _service_registry.items():
        try:
            if 'class' in service_info:
                # Register class-based service
                service_class = service_info['class']
                singleton = service_info['singleton']
                dependencies = service_info['dependencies']
                
                if singleton:
                    container.register_singleton(
                        service_name, 
                        service_class, 
                        dependencies=dependencies
                    )
                else:
                    container.register_transient(
                        service_name, 
                        service_class, 
                        dependencies=dependencies
                    )
                    
                logger.debug(f"Registered service: {service_name} ({'singleton' if singleton else 'transient'})")
                
            elif 'factory' in service_info:
                # Register factory-based service
                factory_func = service_info['factory']
                dependencies = service_info['dependencies']
                
                # Determine the return type from the factory function
                return_type = inspect.signature(factory_func).return_annotation
                if return_type == inspect.Signature.empty:
                    return_type = Any
                
                container.register_factory(
                    service_name,
                    return_type,
                    factory_func,
                    dependencies=dependencies
                )
                
                logger.debug(f"Registered factory: {service_name}")
                
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")

def _scan_filesystem_for_services(container: Any, base_path: Path) -> None:
    """Scan the filesystem for additional services to register."""
    logger.info("Scanning filesystem for additional services...")
    
    discovered_count = 0
    
    # Walk through all Python files in the base path
    for py_file in base_path.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            # Convert file path to module name
            module_name = _file_path_to_module_name(py_file, base_path)
            
            # Import the module
            module = importlib.import_module(module_name)
            
            # Scan for service classes and functions
            services_found = _scan_module_for_services(module, container)
            discovered_count += services_found
            
        except Exception as e:
            logger.debug(f"Could not scan file {py_file}: {e}")
            continue
    
    logger.info(f"Discovered {discovered_count} additional services from filesystem")

def _file_path_to_module_name(file_path: Path, base_path: Path) -> str:
    """Convert a file path to a module name."""
    # Get relative path from base path
    relative_path = file_path.relative_to(base_path)
    
    # Convert to module name
    parts = list(relative_path.parts)
    parts[-1] = parts[-1][:-3]  # Remove .py extension
    
    return ".".join(parts)

def _scan_module_for_services(module: Any, container: Any) -> int:
    """Scan a module for service classes and functions."""
    services_found = 0
    
    # Look for classes that might be services
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if _is_potential_service_class(obj):
            try:
                service_name = _generate_service_name(name)
                
                # Check if already registered
                if not _is_service_registered(container, service_name):
                    # Register as singleton by default
                    container.register_singleton(service_name, obj)
                    logger.debug(f"Auto-registered service class: {service_name}")
                    services_found += 1
                    
            except Exception as e:
                logger.debug(f"Could not register class {name}: {e}")
    
    # Look for functions that might be factories
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if _is_potential_service_factory(obj):
            try:
                service_name = _generate_service_name(name)
                
                # Check if already registered
                if not _is_service_registered(container, service_name):
                    # Determine return type
                    return_type = inspect.signature(obj).return_annotation
                    if return_type == inspect.Signature.empty:
                        return_type = Any
                    
                    container.register_factory(service_name, return_type, obj)
                    logger.debug(f"Auto-registered service factory: {service_name}")
                    services_found += 1
                    
            except Exception as e:
                logger.debug(f"Could not register function {name}: {e}")
    
    return services_found

def _is_potential_service_class(cls: Type) -> bool:
    """Check if a class looks like it could be a service."""
    # Skip private classes, built-ins, and abstract classes
    if cls.__name__.startswith('_') or cls.__module__ == 'builtins':
        return False
    
    # Skip abstract base classes
    if inspect.isabstract(cls):
        return False
    
    # Must be instantiable (not an abstract class)
    try:
        # Check if we can get constructor signature
        inspect.signature(cls.__init__)
        return True
    except (ValueError, TypeError):
        return False

def _is_potential_service_factory(func: Callable) -> bool:
    """Check if a function looks like it could be a service factory."""
    # Skip private functions
    if func.__name__.startswith('_'):
        return False
    
    # Must be callable and not a method
    if not callable(func) or inspect.ismethod(func):
        return False
    
    # Should have a return type annotation
    sig = inspect.signature(func)
    return sig.return_annotation != inspect.Signature.empty

def _generate_service_name(name: str) -> str:
    """Generate a service name from a class or function name."""
    # Convert CamelCase to snake_case for service names
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _is_service_registered(container: Any, service_name: str) -> bool:
    """Check if a service is already registered in the container."""
    try:
        # Try to get the service - if it exists, it's registered
        container.get(service_name)
        return True
    except (ValueError, KeyError, AttributeError):
        return False

def get_service_registry() -> Dict[str, Dict[str, Any]]:
    """Get the current service registry."""
    return _service_registry.copy()

def clear_service_registry() -> None:
    """Clear the service registry."""
    _service_registry.clear()
    logger.info("Service registry cleared")

def register_service_manually(
    container: Any, 
    name: str, 
    service_type: Type, 
    implementation: Any = None,
    singleton: bool = True,
    dependencies: Optional[List[str]] = None
) -> None:
    """Manually register a service with the container."""
    try:
        if singleton:
            container.register_singleton(name, service_type, implementation, dependencies)
        else:
            container.register_transient(name, service_type, implementation, dependencies)
        
        logger.info(f"Manually registered service: {name}")
        
    except Exception as e:
        logger.error(f"Failed to manually register service {name}: {e}")
        raise

def discover_services_in_package(package_name: str, container: Any) -> None:
    """Discover services in a specific Python package."""
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        logger.info(f"Discovering services in package: {package_name}")
        discover_and_register_services(container, str(package_path))
        
    except Exception as e:
        logger.error(f"Failed to discover services in package {package_name}: {e}")
        raise
