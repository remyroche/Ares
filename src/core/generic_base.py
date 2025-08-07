# src/core/generic_base.py

"""
Generic base classes with proper type constraints for reusable components.
"""

from abc import ABC, abstractmethod
from typing import (
    AsyncContextManager,
    Callable,
    ClassVar,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

from src.types import (
    ConfigDict,
    Monitorable,
    PerformanceMetrics,
    Startable,
    TradingComponent,
)

# Type variables with constraints
ConfigT = TypeVar("ConfigT", bound=ConfigDict)
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
ErrorT = TypeVar("ErrorT", bound=Exception)
ComponentT = TypeVar("ComponentT", bound=TradingComponent)

# Protocol constraints for data processing
@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable data."""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        ...
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create from dictionary."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for validatable data."""
    
    def validate(self) -> bool:
        """Validate the data."""
        ...
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        ...


# Generic base classes
class GenericTradingComponent(Generic[ConfigT], ABC):
    """
    Generic base class for trading components with type-safe configuration.
    """
    
    def __init__(self, config: ConfigT) -> None:
        self._config = config
        self._is_running = False
        self._metrics: PerformanceMetrics = {}
        
    @property
    def config(self) -> ConfigT:
        """Get component configuration."""
        return self._config
    
    @abstractmethod
    async def start(self) -> None:
        """Start the component."""
        self._is_running = True
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the component."""
        self._is_running = False
    
    def is_running(self) -> bool:
        """Check if component is running."""
        return self._is_running
    
    @abstractmethod
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self._metrics
    
    @abstractmethod
    def get_health_status(self) -> dict:
        """Get health status."""
        ...


class GenericDataProcessor(Generic[DataT, ResultT], ABC):
    """
    Generic base class for data processors with input/output type constraints.
    """
    
    @abstractmethod
    async def process(self, data: DataT) -> ResultT:
        """Process input data and return result."""
        ...
    
    @abstractmethod
    def validate_input(self, data: DataT) -> bool:
        """Validate input data."""
        ...
    
    @abstractmethod
    def validate_output(self, result: ResultT) -> bool:
        """Validate output result."""
        ...
    
    async def safe_process(self, data: DataT) -> Optional[ResultT]:
        """Safely process data with validation."""
        if not self.validate_input(data):
            return None
        
        try:
            result = await self.process(data)
            if self.validate_output(result):
                return result
            return None
        except Exception:
            return None


class GenericRepository(Generic[DataT], ABC):
    """
    Generic repository base class with type-safe CRUD operations.
    """
    
    @abstractmethod
    async def create(self, item: DataT) -> str:
        """Create a new item and return its ID."""
        ...
    
    @abstractmethod
    async def read(self, item_id: str) -> Optional[DataT]:
        """Read an item by ID."""
        ...
    
    @abstractmethod
    async def update(self, item_id: str, item: DataT) -> bool:
        """Update an existing item."""
        ...
    
    @abstractmethod
    async def delete(self, item_id: str) -> bool:
        """Delete an item by ID."""
        ...
    
    @abstractmethod
    async def list_all(self) -> List[DataT]:
        """List all items."""
        ...
    
    @abstractmethod
    async def find_by_criteria(self, criteria: dict) -> List[DataT]:
        """Find items matching criteria."""
        ...


class GenericEventHandler(Generic[DataT], ABC):
    """
    Generic event handler with type-safe event data.
    """
    
    event_types: ClassVar[List[str]] = []
    
    @abstractmethod
    async def handle(self, event_type: str, data: DataT) -> None:
        """Handle an event."""
        ...
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the event type."""
        ...
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get handler priority (lower = higher priority)."""
        ...


class GenericModelTrainer(Generic[DataT, ResultT], ABC):
    """
    Generic ML model trainer with type-safe data and results.
    """
    
    @abstractmethod
    async def train(self, training_data: DataT) -> ResultT:
        """Train the model with training data."""
        ...
    
    @abstractmethod
    async def validate(self, validation_data: DataT) -> dict:
        """Validate the trained model."""
        ...
    
    @abstractmethod
    async def save_model(self, path: str) -> bool:
        """Save the trained model."""
        ...
    
    @abstractmethod
    async def load_model(self, path: str) -> bool:
        """Load a trained model."""
        ...
    
    @abstractmethod
    def get_model_metadata(self) -> dict:
        """Get model metadata."""
        ...


class GenericCacheManager(Generic[DataT], ABC):
    """
    Generic cache manager with type-safe cached data.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[DataT]:
        """Get cached value."""
        ...
    
    @abstractmethod
    async def set(self, key: str, value: DataT, ttl_seconds: Optional[int] = None) -> bool:
        """Set cached value with optional TTL."""
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cached value."""
        ...
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached values."""
        ...
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class GenericAsyncContextManager(Generic[DataT], AsyncContextManager[DataT]):
    """
    Generic async context manager with type-safe resource management.
    """
    
    def __init__(self, resource_factory: Callable[[], DataT]):
        self._resource_factory = resource_factory
        self._resource: Optional[DataT] = None
    
    async def __aenter__(self) -> DataT:
        """Enter the async context."""
        self._resource = await self._acquire_resource()
        return self._resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context."""
        if self._resource:
            await self._release_resource(self._resource)
    
    @abstractmethod
    async def _acquire_resource(self) -> DataT:
        """Acquire the resource."""
        ...
    
    @abstractmethod
    async def _release_resource(self, resource: DataT) -> None:
        """Release the resource."""
        ...


# Type-constrained factory base class
class GenericFactory(Generic[DataT], ABC):
    """
    Generic factory with type constraints.
    """
    
    @abstractmethod
    def create(self, **kwargs) -> DataT:
        """Create an instance."""
        ...
    
    @abstractmethod
    def can_create(self, type_name: str) -> bool:
        """Check if factory can create the specified type."""
        ...
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported types."""
        ...


# Error handling with generic constraints
class GenericErrorHandler(Generic[ErrorT], ABC):
    """
    Generic error handler with type-safe error handling.
    """
    
    @abstractmethod
    def can_handle(self, error: Exception) -> bool:
        """Check if this handler can handle the error."""
        ...
    
    @abstractmethod
    async def handle(self, error: ErrorT) -> bool:
        """Handle the error. Return True if handled successfully."""
        ...
    
    @abstractmethod
    def get_recovery_strategy(self, error: ErrorT) -> Optional[str]:
        """Get recovery strategy for the error."""
        ...