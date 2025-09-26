"""
Service discovery utilities for automatic service registration.
"""

import os
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List, Type, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Information about a discovered service"""
    name: str
    service_type: Type
    implementation: Type
    module_path: str
    is_abstract: bool
    dependencies: List[str]
    priority: int = 0


class ServiceDiscovery:
    """Service discovery and registration system"""
    
    def __init__(self, container: Any):
        self.container = container
        self.discovered_services: Dict[str, ServiceInfo] = {}
        self.registration_order: List[str] = []
        
    def discover_and_register_services(self, base_path: str) -> None:
        """
        Discover and register services automatically.

        Args:
            base_path: Base path to scan for services
        """
        logger.info(f"🔍 Scanning {base_path} for services to register...")
        
        try:
            # Convert to Path object
            base_path = Path(base_path)
            if not base_path.exists():
                logger.warning(f"Base path does not exist: {base_path}")
                return
            
            # Discover services
            self._discover_services(base_path)
            
            # Sort services by priority and dependencies
            self._sort_services_by_dependencies()
            
            # Register services in order
            self._register_services()
            
            logger.info(f"✅ Service discovery completed: {len(self.discovered_services)} services registered")
            
        except Exception as e:
            logger.error(f"❌ Service discovery failed: {e}")
            raise
    
    def _discover_services(self, base_path: Path) -> None:
        """Discover services by scanning Python modules"""
        # Get all Python files recursively
        python_files = list(base_path.rglob("*.py"))
        
        for file_path in python_files:
            # Skip __init__.py and test files
            if file_path.name.startswith("__") or "test" in file_path.name.lower():
                continue
                
            try:
                self._scan_module(file_path, base_path)
            except Exception as e:
                logger.warning(f"Failed to scan module {file_path}: {e}")
    
    def _scan_module(self, file_path: Path, base_path: Path) -> None:
        """Scan a single module for services"""
        try:
            # Convert file path to module path
            relative_path = file_path.relative_to(base_path)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_path = ".".join(module_parts)
            
            # Import the module
            module = importlib.import_module(module_path)
            
            # Scan for classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_service_class(obj):
                    service_info = self._create_service_info(name, obj, module_path)
                    if service_info:
                        self.discovered_services[service_info.name] = service_info
                        logger.debug(f"Discovered service: {service_info.name} in {module_path}")
                        
        except ImportError as e:
            logger.debug(f"Could not import module {module_path}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning module {file_path}: {e}")
    
    def _is_service_class(self, cls: Type) -> bool:
        """Check if a class is a service that should be registered"""
        # Skip abstract classes, built-ins, and test classes
        if (inspect.isabstract(cls) or 
            cls.__name__.startswith("_") or 
            "test" in cls.__name__.lower() or
            cls.__module__.startswith("test")):
            return False
        
        # Check for service indicators
        service_indicators = [
            "Service", "Manager", "Handler", "Controller", "Processor",
            "Analyzer", "Predictor", "Optimizer", "Validator", "Monitor"
        ]
        
        # Check class name
        if any(indicator in cls.__name__ for indicator in service_indicators):
            return True
        
        # Check for service-related methods
        service_methods = [
            "initialize", "start", "stop", "process", "handle", "execute",
            "analyze", "predict", "optimize", "validate", "monitor"
        ]
        
        methods = [method for method in dir(cls) if not method.startswith("_")]
        if any(method in methods for method in service_methods):
            return True
        
        # Check for dependency injection patterns
        if hasattr(cls, "__init__"):
            init_signature = inspect.signature(cls.__init__)
            if len(init_signature.parameters) > 1:  # More than just 'self'
                return True
        
        return False
    
    def _create_service_info(self, name: str, cls: Type, module_path: str) -> Optional[ServiceInfo]:
        """Create service information from a class"""
        try:
            # Determine service name
            service_name = self._get_service_name(name, cls)
            
            # Get dependencies from constructor
            dependencies = self._get_dependencies(cls)
            
            # Determine priority based on class name and methods
            priority = self._get_service_priority(cls)
            
            return ServiceInfo(
                name=service_name,
                service_type=cls,
                implementation=cls,
                module_path=module_path,
                is_abstract=inspect.isabstract(cls),
                dependencies=dependencies,
                priority=priority
            )
            
        except Exception as e:
            logger.warning(f"Failed to create service info for {name}: {e}")
            return None
    
    def _get_service_name(self, name: str, cls: Type) -> str:
        """Get the service name for registration"""
        # Use the class name as service name
        return name
    
    def _get_dependencies(self, cls: Type) -> List[str]:
        """Extract dependencies from class constructor"""
        dependencies = []
        
        if hasattr(cls, "__init__"):
            init_signature = inspect.signature(cls.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name == "self":
                    continue
                
                # Get type annotation
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        dependencies.append(param.annotation.__name__)
                    elif hasattr(param.annotation, "__origin__"):
                        # Handle generic types
                        if hasattr(param.annotation.__origin__, "__name__"):
                            dependencies.append(param.annotation.__origin__.__name__)
        
        return dependencies
    
    def _get_service_priority(self, cls: Type) -> int:
        """Determine service registration priority"""
        priority = 0
        
        # Higher priority for core services
        if "Core" in cls.__name__ or "Base" in cls.__name__:
            priority += 100
        
        # Higher priority for managers and handlers
        if "Manager" in cls.__name__ or "Handler" in cls.__name__:
            priority += 50
        
        # Lower priority for processors and analyzers
        if "Processor" in cls.__name__ or "Analyzer" in cls.__name__:
            priority += 25
        
        return priority
    
    def _sort_services_by_dependencies(self) -> None:
        """Sort services by dependencies to ensure proper registration order"""
        # Create dependency graph
        dependency_graph = {}
        for name, service in self.discovered_services.items():
            dependency_graph[name] = set(service.dependencies)
        
        # Topological sort
        visited = set()
        temp_visited = set()
        sorted_services = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                logger.warning(f"Circular dependency detected involving {service_name}")
                return
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            # Visit dependencies first
            for dep in dependency_graph.get(service_name, set()):
                if dep in self.discovered_services:
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            sorted_services.append(service_name)
        
        # Visit all services
        for service_name in self.discovered_services:
            if service_name not in visited:
                visit(service_name)
        
        # Sort by priority within each dependency level
        self.registration_order = sorted(
            sorted_services,
            key=lambda name: self.discovered_services[name].priority,
            reverse=True
        )
    
    def _register_services(self) -> None:
        """Register discovered services with the container"""
        for service_name in self.registration_order:
            service_info = self.discovered_services[service_name]
            
            try:
                # Skip abstract services
                if service_info.is_abstract:
                    logger.debug(f"Skipping abstract service: {service_name}")
                    continue
                
                # Register with container
                self.container.register(
                    service_name=service_name,
                    service_type=service_info.service_type,
                    implementation=service_info.implementation,
                    singleton=True
                )
                
                logger.info(f"✅ Registered service: {service_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register service {service_name}: {e}")


def discover_and_register_services(container: Any, base_path: str) -> None:
    """
    Discover and register services automatically.

    Args:
        container: The dependency injection container
        base_path: Base path to scan for services
    """
    discovery = ServiceDiscovery(container)
    discovery.discover_and_register_services(base_path)
