# src/core/service_registry.py

"""
Service registry for dependency injection container configuration.

This module provides centralized service registration for all trading components, ensuring proper dependency injection throughout the system.
"""

from typing import Any, Dict
from src.core.dependency_injection import DependencyContainer


class ServiceRegistry:
    """
    Service registry for managing dependency injection container configuration.
    """
    
    def __init__(self, container: DependencyContainer):
        self.container = container

    def register_all_services(self, config: Dict[str, Any]) -> None:
        """Register all services in the container based on configuration."""
        try:
            # Register core services based on configuration
            # This is a simplified version - in practice, you would register
            # specific implementations based on the config
            
            # Example: Register services if they exist in config
            if "analyst" in config:
                # Register analyst service
                pass
                
            if "strategist" in config:
                # Register strategist service
                pass
                
            if "tactician" in config:
                # Register tactician service
                pass
                
            if "supervisor" in config:
                # Register supervisor service
                pass
                
        except Exception as e:
            # Log error but don't fail completely
            print(f"Error registering services: {e}")

    def get_registered_services(self) -> Dict[str, Any]:
        """Get all registered services."""
        try:
            return self.container.get_all_services()
        except Exception:
            return {}

    def is_service_registered(self, service_name: str) -> bool:
        """Check if a service is registered."""
        try:
            services = self.container.get_all_services()
            return service_name in services
        except Exception:
            return False


