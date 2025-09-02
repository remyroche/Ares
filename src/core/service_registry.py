from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, Callable


class IAnalyst(ABC):
    """Interface for market analysis components."""
    
    @abstractmethod
    def analyze_market(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and return insights."""
        pass


class IEventBus(ABC):
    """Interface for event bus system."""
    
    @abstractmethod
    def publish(self, event: str, data: Dict[str, Any]) -> None:
        """Publish an event with data."""
        pass
    
    @abstractmethod
    def subscribe(self, event: str, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Subscribe to an event."""
        pass


class IStrategist(ABC):
    """Interface for trading strategy components."""
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on market data."""
        pass


class ISupervisor(ABC):
    """Interface for system supervision components."""
    
    @abstractmethod
    def monitor_system(self) -> Dict[str, Any]:
        """Monitor system health and status."""
        pass
    
    @abstractmethod
    def handle_error(self, error: Exception) -> None:
        """Handle system errors."""
        pass


class ITactician(ABC):
    """Interface for tactical execution components."""
    
    @abstractmethod
    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on a signal."""
        pass


class ServiceRegistry:
    """Central service registry for dependency injection."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._implementations: Dict[str, Type] = {}
    
    def register_service(self, service_name: str, implementation: Type) -> None:
        """Register a service implementation."""
        self._implementations[service_name] = implementation
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Get a service instance, creating it if necessary."""
        if service_name not in self._services:
            if service_name in self._implementations:
                self._services[service_name] = self._implementations[service_name]()
            else:
                return None
        return self._services[service_name]
    
    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._implementations
    
    def list_services(self) -> list:
        """List all registered service names."""
        return list(self._implementations.keys())


# Global service registry instance
service_registry = ServiceRegistry()

