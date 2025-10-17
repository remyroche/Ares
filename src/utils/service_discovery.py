"""
Service discovery utilities for automatic service registration.

This module provides comprehensive service discovery functionality that can
automatically scan directories for services and register them with a dependency
injection container. Includes health checks, dynamic registration, and service
monitoring capabilities.
"""

import os
import sys
import importlib
import inspect
import logging
import asyncio
import time
import threading
from typing import Any, Optional, Dict, List, Type, Set, Callable, Union
from pathlib import Path
import pkgutil
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)

# Service decorator registry
_service_registry: Dict[str, Dict[str, Any]] = {}

class ServiceStatus(Enum):
    """Service status enumeration"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ServiceInfo:
    """Information about a discovered service"""
    name: str
    service_type: Type
    module: str
    singleton: bool = True
    dependencies: List[str] = field(default_factory=list)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    health_check_interval: float = 30.0  # seconds
    error_count: int = 0
    max_errors: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)

class ServiceHealthChecker:
    """Handles health checking for services"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(f"{__name__}.ServiceHealthChecker")
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._services: Dict[str, ServiceInfo] = {}
    
    async def start(self) -> None:
        """Start the health checker"""
        if self._running:
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Service health checker started")
    
    async def stop(self) -> None:
        """Stop the health checker"""
        if not self._running:
            return
        
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Service health checker stopped")
    
    def register_service(self, service_info: ServiceInfo) -> None:
        """Register a service for health checking"""
        self._services[service_info.name] = service_info
        self.logger.debug(f"Registered service for health checking: {service_info.name}")
    
    def unregister_service(self, service_name: str) -> None:
        """Unregister a service from health checking"""
        if service_name in self._services:
            del self._services[service_name]
            self.logger.debug(f"Unregistered service from health checking: {service_name}")
    
    async def check_service_health(self, service_info: ServiceInfo) -> ServiceStatus:
        """Check the health of a specific service"""
        try:
            # Check if service has a health check method
            if hasattr(service_info.service_type, 'health_check'):
                if asyncio.iscoroutinefunction(service_info.service_type.health_check):
                    is_healthy = await service_info.service_type.health_check()
                else:
                    is_healthy = service_info.service_type.health_check()
                
                if is_healthy:
                    service_info.status = ServiceStatus.HEALTHY
                    service_info.error_count = 0
                else:
                    service_info.status = ServiceStatus.UNHEALTHY
                    service_info.error_count += 1
            else:
                # Default health check - try to instantiate if not singleton
                if not service_info.singleton or service_info.instance is None:
                    try:
                        # Try to create instance to check if service is valid
                        if service_info.dependencies:
                            # Skip if dependencies not available
                            service_info.status = ServiceStatus.HEALTHY
                        else:
                            service_info.status = ServiceStatus.HEALTHY
                    except Exception:
                        service_info.status = ServiceStatus.UNHEALTHY
                        service_info.error_count += 1
                else:
                    service_info.status = ServiceStatus.HEALTHY
            
            service_info.last_health_check = datetime.now()
            
            # Mark as error if too many consecutive failures
            if service_info.error_count >= service_info.max_errors:
                service_info.status = ServiceStatus.ERROR
            
            return service_info.status
            
        except Exception as e:
            self.logger.error(f"Health check failed for {service_info.name}: {e}")
            service_info.status = ServiceStatus.ERROR
            service_info.error_count += 1
            service_info.last_health_check = datetime.now()
            return ServiceStatus.ERROR
    
    async def _health_check_loop(self) -> None:
        """Main health check loop"""
        while self._running:
            try:
                for service_info in self._services.values():
                    # Check if it's time for a health check
                    if (service_info.last_health_check is None or 
                        datetime.now() - service_info.last_health_check > timedelta(seconds=service_info.health_check_interval)):
                        
                        await self.check_service_health(service_info)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Back off on errors
    
    def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """Get the current status of a service"""
        service_info = self._services.get(service_name)
        return service_info.status if service_info else None
    
    def get_all_service_statuses(self) -> Dict[str, ServiceStatus]:
        """Get status of all registered services"""
        return {name: info.status for name, info in self._services.items()}

class ServiceRegistry:
    """Enhanced service registry with health checking and dynamic management"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ServiceRegistry")
        self._services: Dict[str, ServiceInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._health_checker = ServiceHealthChecker()
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the service registry"""
        await self._health_checker.start()
        self.logger.info("Service registry started")
    
    async def stop(self) -> None:
        """Stop the service registry"""
        await self._health_checker.stop()
        
        # Clean up instances
        for instance in self._instances.values():
            if hasattr(instance, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(instance.cleanup):
                        await instance.cleanup()
                    else:
                        instance.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up instance: {e}")
        
        self._instances.clear()
        self._services.clear()
        self.logger.info("Service registry stopped")
    
    async def register_service(
        self,
        name: str,
        service_type: Type,
        singleton: bool = True,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service with the registry"""
        async with self._lock:
            service_info = ServiceInfo(
                name=name,
                service_type=service_type,
                module=service_type.__module__,
                singleton=singleton,
                dependencies=dependencies or [],
                metadata=metadata or {}
            )
            
            self._services[name] = service_info
            await self._health_checker.register_service(service_info)
            
            self.logger.info(f"Registered service: {name} ({'singleton' if singleton else 'transient'})")
    
    async def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance"""
        async with self._lock:
            service_info = self._services.get(name)
            if not service_info:
                return None
            
            # Check if service is healthy
            if service_info.status == ServiceStatus.ERROR:
                self.logger.warning(f"Service {name} is in error state")
                return None
            
            # Return existing instance if singleton
            if service_info.singleton and service_info.instance is not None:
                return service_info.instance
            
            # Create new instance
            try:
                instance = service_info.service_type()
                service_info.instance = instance
                self._instances[name] = instance
                
                # Initialize if method exists
                if hasattr(instance, 'initialize'):
                    if asyncio.iscoroutinefunction(instance.initialize):
                        await instance.initialize()
                    else:
                        instance.initialize()
                
                return instance
                
            except Exception as e:
                self.logger.error(f"Failed to create instance for {name}: {e}")
                service_info.status = ServiceStatus.ERROR
                return None
    
    async def unregister_service(self, name: str) -> bool:
        """Unregister a service"""
        async with self._lock:
            if name not in self._services:
                return False
            
            service_info = self._services[name]
            
            # Clean up instance
            if service_info.instance and hasattr(service_info.instance, 'cleanup'):
                try:
                    if asyncio.iscoroutinefunction(service_info.instance.cleanup):
                        await service_info.instance.cleanup()
                    else:
                        service_info.instance.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up service {name}: {e}")
            
            # Remove from registries
            await self._health_checker.unregister_service(name)
            del self._services[name]
            self._instances.pop(name, None)
            
            self.logger.info(f"Unregistered service: {name}")
            return True
    
    def get_service_info(self, name: str) -> Optional[ServiceInfo]:
        """Get service information"""
        return self._services.get(name)
    
    def list_services(self) -> List[str]:
        """List all registered service names"""
        return list(self._services.keys())
    
    def get_healthy_services(self) -> List[str]:
        """Get list of healthy services"""
        return [
            name for name, info in self._services.items()
            if info.status == ServiceStatus.HEALTHY
        ]
    
    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get service status"""
        return self._health_checker.get_service_status(name)
    
    def get_all_statuses(self) -> Dict[str, ServiceStatus]:
        """Get all service statuses"""
        return self._health_checker.get_all_service_statuses()

# Global service registry instance
_global_registry: Optional[ServiceRegistry] = None

def get_global_registry() -> ServiceRegistry:
    """Get the global service registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry

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

async def discover_and_register_services(container: Any, base_path: str, use_health_checks: bool = True) -> ServiceRegistry:
    """
    Discover and register services automatically with enhanced capabilities.

    Args:
        container: The dependency injection container
        base_path: Base path to scan for services
        use_health_checks: Whether to enable health checking

    Returns:
        ServiceRegistry instance for advanced service management
    """
    logger.info(f"🔍 Scanning {base_path} for services to register...")
    
    try:
        # Get or create global registry
        registry = get_global_registry()
        
        # Start the registry if not already started
        if not registry._health_checker._running:
            await registry.start()
        
        # Convert to Path object for easier manipulation
        base_path_obj = Path(base_path)
        
        if not base_path_obj.exists():
            logger.warning(f"Base path does not exist: {base_path}")
            return registry
        
        # Discover services from the decorator registry first
        await _register_discovered_services(container, registry)
        
        # Then scan for additional services in the file system
        await _scan_filesystem_for_services(container, base_path_obj, registry)
        
        logger.info("✅ Service discovery completed successfully")
        logger.info(f"📊 Discovered {len(registry.list_services())} services")
        
        return registry
        
    except Exception as e:
        logger.error(f"❌ Service discovery failed: {e}")
        raise

async def _register_discovered_services(container: Any, registry: ServiceRegistry) -> None:
    """Register services that were discovered via decorators."""
    logger.info(f"Registering {len(_service_registry)} discovered services...")
    
    for service_name, service_info in _service_registry.items():
        try:
            if 'class' in service_info:
                # Register class-based service
                service_class = service_info['class']
                singleton = service_info['singleton']
                dependencies = service_info['dependencies']
                
                # Register with enhanced registry
                await registry.register_service(
                    service_name,
                    service_class,
                    singleton=singleton,
                    dependencies=dependencies,
                    metadata={'discovered_via': 'decorator'}
                )
                
                # Also register with original container for backward compatibility
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
                
                # Create a wrapper class for factory functions
                class FactoryService:
                    def __init__(self):
                        self.factory = factory_func
                        self.dependencies = dependencies
                    
                    async def create_instance(self, **kwargs):
                        return await factory_func(**kwargs) if asyncio.iscoroutinefunction(factory_func) else factory_func(**kwargs)
                
                # Register with enhanced registry
                await registry.register_service(
                    service_name,
                    FactoryService,
                    singleton=True,
                    dependencies=dependencies,
                    metadata={'discovered_via': 'factory', 'original_factory': factory_func.__name__}
                )
                
                # Also register with original container
                container.register_factory(
                    service_name,
                    return_type,
                    factory_func,
                    dependencies=dependencies
                )
                
                logger.debug(f"Registered factory: {service_name}")
                
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")

async def _scan_filesystem_for_services(container: Any, base_path: Path, registry: ServiceRegistry) -> None:
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
            services_found = await _scan_module_for_services(module, container, registry)
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

async def _scan_module_for_services(module: Any, container: Any, registry: ServiceRegistry) -> int:
    """Scan a module for service classes and functions."""
    services_found = 0
    
    # Look for classes that might be services
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if _is_potential_service_class(obj):
            try:
                service_name = _generate_service_name(name)
                
                # Check if already registered
                if not _is_service_registered(container, service_name):
                    # Register with enhanced registry
                    await registry.register_service(
                        service_name,
                        obj,
                        singleton=True,
                        metadata={'discovered_via': 'filesystem_scan', 'module': module.__name__}
                    )
                    
                    # Also register with original container for backward compatibility
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
                    
                    # Create a wrapper class for factory functions
                    class FactoryService:
                        def __init__(self):
                            self.factory = obj
                        
                        async def create_instance(self, **kwargs):
                            return await obj(**kwargs) if asyncio.iscoroutinefunction(obj) else obj(**kwargs)
                    
                    # Register with enhanced registry
                    await registry.register_service(
                        service_name,
                        FactoryService,
                        singleton=True,
                        metadata={'discovered_via': 'filesystem_scan', 'module': module.__name__, 'original_factory': name}
                    )
                    
                    # Also register with original container
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

async def discover_services_in_package(package_name: str, container: Any, use_health_checks: bool = True) -> ServiceRegistry:
    """Discover services in a specific Python package."""
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        logger.info(f"Discovering services in package: {package_name}")
        return await discover_and_register_services(container, str(package_path), use_health_checks)
        
    except Exception as e:
        logger.error(f"Failed to discover services in package {package_name}: {e}")
        raise

async def get_service_health_status() -> Dict[str, Any]:
    """Get health status of all services"""
    registry = get_global_registry()
    statuses = registry.get_all_statuses()
    
    return {
        "total_services": len(statuses),
        "healthy_services": len([s for s in statuses.values() if s == ServiceStatus.HEALTHY]),
        "unhealthy_services": len([s for s in statuses.values() if s == ServiceStatus.UNHEALTHY]),
        "error_services": len([s for s in statuses.values() if s == ServiceStatus.ERROR]),
        "service_statuses": {name: status.value for name, status in statuses.items()}
    }

async def restart_unhealthy_services() -> Dict[str, bool]:
    """Restart all unhealthy services"""
    registry = get_global_registry()
    results = {}
    
    for service_name in registry.list_services():
        service_info = registry.get_service_info(service_name)
        if service_info and service_info.status in [ServiceStatus.UNHEALTHY, ServiceStatus.ERROR]:
            try:
                # Unregister and re-register the service
                await registry.unregister_service(service_name)
                await registry.register_service(
                    service_name,
                    service_info.service_type,
                    service_info.singleton,
                    service_info.dependencies,
                    service_info.metadata
                )
                results[service_name] = True
                logger.info(f"Restarted service: {service_name}")
            except Exception as e:
                logger.error(f"Failed to restart service {service_name}: {e}")
                results[service_name] = False
    
    return results

def create_service_monitor(registry: ServiceRegistry, check_interval: float = 30.0) -> Callable:
    """Create a service monitoring function"""
    async def monitor_services():
        """Monitor services and log status changes"""
        while True:
            try:
                statuses = registry.get_all_statuses()
                unhealthy_services = [
                    name for name, status in statuses.items() 
                    if status in [ServiceStatus.UNHEALTHY, ServiceStatus.ERROR]
                ]
                
                if unhealthy_services:
                    logger.warning(f"Unhealthy services detected: {unhealthy_services}")
                
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in service monitor: {e}")
                await asyncio.sleep(5)
    
    return monitor_services
