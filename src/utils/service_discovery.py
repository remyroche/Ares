"""
Automatic service discovery for dependency injection.
"""

import inspect
import importlib
import pkgutil
from typing import Any, Dict, List, Type, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ServiceDiscovery:
    """
    Automatic service discovery for dependency injection.
    
    This class scans modules and packages to automatically discover
    services that can be registered with the DI container.
    """
    
    def __init__(self, base_package: str = "src"):
        """
        Initialize service discovery.
        
        Args:
            base_package: Base package to scan for services
        """
        self.base_package = base_package
        self.discovered_services: Dict[str, Type] = {}
        self.service_interfaces: Dict[str, Type] = {}
        self.service_implementations: Dict[str, List[Type]] = {}
        
    def discover_services(self, package_path: str | None = None) -> Dict[str, Type]:
        """
        Discover all services in the specified package.
        
        Args:
            package_path: Path to scan (defaults to base_package)
            
        Returns:
            Dictionary mapping service names to service types
        """
        if package_path is None:
            package_path = self.base_package
            
        logger.info(f"Discovering services in package: {package_path}")
        
        try:
            package = importlib.import_module(package_path)
            self._scan_package(package, package_path)
        except ImportError as e:
            logger.warning(f"Could not import package {package_path}: {e}")
            
        logger.info(f"Discovered {len(self.discovered_services)} services")
        return self.discovered_services
    
    def _scan_package(self, package: Any, package_name: str) -> None:
        """Recursively scan a package for services."""
        try:
            # Get package path
            if hasattr(package, '__path__'):
                package_path = package.__path__[0]
            else:
                return
                
            # Scan all modules in the package
            for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
                full_module_name = f"{package_name}.{modname}"
                
                try:
                    if ispkg:
                        # Recursively scan subpackages
                        subpackage = importlib.import_module(full_module_name)
                        self._scan_package(subpackage, full_module_name)
                    else:
                        # Scan individual modules
                        module = importlib.import_module(full_module_name)
                        self._scan_module(module, full_module_name)
                        
                except Exception as e:
                    logger.debug(f"Could not scan module {full_module_name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Error scanning package {package_name}: {e}")
    
    def _scan_module(self, module: Any, module_name: str) -> None:
        """Scan a module for service classes."""
        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_service_class(obj):
                    service_name = self._get_service_name(obj, name)
                    self.discovered_services[service_name] = obj
                    
                    # Check if it's an interface
                    if self._is_interface(obj):
                        self.service_interfaces[service_name] = obj
                    else:
                        # Check for interface implementations
                        interfaces = self._get_implemented_interfaces(obj)
                        for interface in interfaces:
                            if interface not in self.service_implementations:
                                self.service_implementations[interface] = []
                            self.service_implementations[interface].append(obj)
                            
        except Exception as e:
            logger.debug(f"Error scanning module {module_name}: {e}")
    
    def _is_service_class(self, obj: Type) -> bool:
        """Check if a class is a service class."""
        # Skip built-in types and abstract classes
        if obj.__module__ == 'builtins':
            return False
            
        # Check for service indicators
        service_indicators = [
            'Service', 'Manager', 'Handler', 'Controller', 
            'Factory', 'Provider', 'Repository', 'Client'
        ]
        
        class_name = obj.__name__
        return any(indicator in class_name for indicator in service_indicators)
    
    def _is_interface(self, obj: Type) -> bool:
        """Check if a class is an interface."""
        # Check for interface indicators
        interface_indicators = ['I', 'Interface', 'Protocol']
        class_name = obj.__name__
        
        return (class_name.startswith('I') and class_name[1:2].isupper()) or \
               any(indicator in class_name for indicator in interface_indicators)
    
    def _get_service_name(self, obj: Type, class_name: str) -> str:
        """Get the service name for a class."""
        # Remove common prefixes/suffixes
        name = class_name
        if name.startswith('I'):
            name = name[1:]  # Remove 'I' prefix from interfaces
            
        # Convert to snake_case for service names
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    def _get_implemented_interfaces(self, obj: Type) -> List[str]:
        """Get interfaces implemented by a class."""
        interfaces = []
        
        # Check base classes
        for base in obj.__bases__:
            if self._is_interface(base):
                interface_name = self._get_service_name(base, base.__name__)
                interfaces.append(interface_name)
        
        # Check for explicit interface implementations
        if hasattr(obj, '__implements__'):
            for interface in obj.__implements__:
                interface_name = self._get_service_name(interface, interface.__name__)
                interfaces.append(interface_name)
                
        return interfaces
    
    def get_services_by_interface(self, interface_name: str) -> List[Type]:
        """Get all services that implement a specific interface."""
        return self.service_implementations.get(interface_name, [])
    
    def get_interfaces(self) -> Dict[str, Type]:
        """Get all discovered interfaces."""
        return self.service_interfaces.copy()
    
    def get_implementations(self) -> Dict[str, List[Type]]:
        """Get all service implementations grouped by interface."""
        return self.service_implementations.copy()
    
    def auto_register_services(self, container: Any) -> None:
        """
        Automatically register discovered services with a DI container.
        
        Args:
            container: DI container to register services with
        """
        logger.info("Auto-registering discovered services...")
        
        # Register interfaces
        for interface_name, interface_type in self.service_interfaces.items():
            try:
                container.register(interface_type, interface_type)
                logger.debug(f"Registered interface: {interface_name}")
            except Exception as e:
                logger.warning(f"Failed to register interface {interface_name}: {e}")
        
        # Register implementations
        for interface_name, implementations in self.service_implementations.items():
            for implementation in implementations:
                try:
                    service_name = self._get_service_name(implementation, implementation.__name__)
                    container.register(service_name, implementation)
                    logger.debug(f"Registered implementation: {service_name} -> {interface_name}")
                except Exception as e:
                    logger.warning(f"Failed to register implementation {implementation.__name__}: {e}")
        
        # Register standalone services
        for service_name, service_type in self.discovered_services.items():
            if service_name not in self.service_interfaces and \
               not any(service_name in impls for impls in self.service_implementations.values()):
                try:
                    container.register(service_name, service_type)
                    logger.debug(f"Registered standalone service: {service_name}")
                except Exception as e:
                    logger.warning(f"Failed to register standalone service {service_name}: {e}")
        
        logger.info("Auto-registration completed")


# Global service discovery instance
_global_discovery: Optional[ServiceDiscovery] = None


def get_global_discovery() -> ServiceDiscovery:
    """Get the global service discovery instance."""
    global _global_discovery
    if _global_discovery is None:
        _global_discovery = ServiceDiscovery()
    return _global_discovery


def discover_and_register_services(container: Any, package_path: str | None = None) -> None:
    """
    Discover and register services automatically.
    
    Args:
        container: DI container to register services with
        package_path: Package path to scan (optional)
    """
    discovery = get_global_discovery()
    discovery.discover_services(package_path)
    discovery.auto_register_services(container)


def find_service_by_name(service_name: str) -> Optional[Type]:
    """
    Find a service by name from discovered services.
    
    Args:
        service_name: Name of the service to find
        
    Returns:
        Service type if found, None otherwise
    """
    discovery = get_global_discovery()
    return discovery.discovered_services.get(service_name)


def find_services_by_interface(interface_name: str) -> List[Type]:
    """
    Find all services that implement a specific interface.
    
    Args:
        interface_name: Name of the interface
        
    Returns:
        List of service types that implement the interface
    """
    discovery = get_global_discovery()
    return discovery.get_services_by_interface(interface_name)
