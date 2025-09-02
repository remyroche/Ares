"""
Standardized Exception Handling Template for Supervisor Module

This module provides a comprehensive error handling system for the supervisor module,
providing decorators for consistent error handling across all supervisor components.

Features:
- Decorators for automatic error handling
- Specific exception types for supervisor operations
- Recovery mechanisms and retry logic
- Comprehensive logging with context
- Performance monitoring integration
"""

import asyncio
import functools
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

# Type variables
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Create a basic logger if system_logger is not available
try:
    from .logger import system_logger
except ImportError:
    system_logger = logging.getLogger("supervisor_error_handler")

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ErrorContext:
    """Base error context class."""
    
    def __init__(self, **kwargs):
        self.timestamp = kwargs.get('timestamp', datetime.now(timezone.utc))
        self.metadata = kwargs.get('metadata', {})
        for key, value in kwargs.items():
            if key not in ['timestamp', 'metadata']:
                setattr(self, key, value)

class SupervisorErrorCategory(Enum):
    """Enumeration of supervisor error categories."""
    COMPONENT_FAILURE = "component_failure"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    RISK_MANAGEMENT = "risk_management"
    PERFORMANCE_MONITORING = "performance_monitoring"
    MODEL_MANAGEMENT = "model_management"
    EXCHANGE_INTEGRATION = "exchange_integration"
    DATA_PROCESSING = "data_processing"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TIMEOUT = "timeout"
    EXTERNAL_SERVICE = "external_service"
    UNKNOWN = "unknown"

class SupervisorError(Exception):
    """Base exception class for supervisor-related errors."""
    
    def __init__(
        self,
        message: str,
        category: SupervisorErrorCategory = SupervisorErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        retryable: bool = False,
        **kwargs
    ):
        super().__init__(message)
        self.category = category
        self.context = context
        self.severity = severity
        self.retryable = retryable
        self.timestamp = datetime.now(timezone.utc)
        for key, value in kwargs.items():
            setattr(self, key, value)

class ComponentFailureError(SupervisorError):
    """Exception raised when a supervisor component fails."""
    
    def __init__(
        self,
        component_name: str,
        operation: str,
        message: str,
        **kwargs
    ):
        super().__init__(
            f"Component {component_name} failed during {operation}: {message}",
            category=SupervisorErrorCategory.COMPONENT_FAILURE,
            **kwargs
        )
        self.component_name = component_name
        self.operation = operation

class PortfolioManagementError(SupervisorError):
    """Exception raised for portfolio management related errors."""
    
    def __init__(
        self,
        operation: str,
        portfolio_id: Optional[str] = None,
        message: str = "Portfolio management operation failed",
        **kwargs
    ):
        super().__init__(
            f"Portfolio management error in {operation}: {message}",
            category=SupervisorErrorCategory.PORTFOLIO_MANAGEMENT,
            **kwargs
        )
        self.operation = operation
        self.portfolio_id = portfolio_id

class RiskManagementError(SupervisorError):
    """Exception raised for risk management related errors."""
    
    def __init__(
        self,
        risk_type: str,
        operation: str,
        message: str = "Risk management operation failed",
        **kwargs
    ):
        super().__init__(
            f"Risk management error in {risk_type} during {operation}: {message}",
            category=SupervisorErrorCategory.RISK_MANAGEMENT,
            **kwargs
        )
        self.risk_type = risk_type
        self.operation = operation

class PerformanceMonitoringError(SupervisorError):
    """Exception raised for performance monitoring related errors."""
    
    def __init__(
        self,
        metric: str,
        operation: str,
        message: str = "Performance monitoring operation failed",
        **kwargs
    ):
        super().__init__(
            f"Performance monitoring error for {metric} during {operation}: {message}",
            category=SupervisorErrorCategory.PERFORMANCE_MONITORING,
            **kwargs
        )
        self.metric = metric
        self.operation = operation

class ModelManagementError(SupervisorError):
    """Exception raised for model management related errors."""
    
    def __init__(
        self,
        model_id: str,
        operation: str,
        message: str = "Model management operation failed",
        **kwargs
    ):
        super().__init__(
            f"Model management error for {model_id} during {operation}: {message}",
            category=SupervisorErrorCategory.MODEL_MANAGEMENT,
            **kwargs
        )
        self.model_id = model_id
        self.operation = operation

class ExchangeIntegrationError(SupervisorError):
    """Exception raised for exchange integration related errors."""
    
    def __init__(
        self,
        exchange: str,
        operation: str,
        message: str = "Exchange integration operation failed",
        **kwargs
    ):
        super().__init__(
            f"Exchange integration error for {exchange} during {operation}: {message}",
            category=SupervisorErrorCategory.EXCHANGE_INTEGRATION,
            **kwargs
        )
        self.exchange = exchange
        self.operation = operation

@dataclass
class SupervisorErrorContext(ErrorContext):
    """Context information for supervisor errors."""
    
    supervisor_id: str = field(default_factory=str)
    component_name: Optional[str] = None
    operation: Optional[str] = None
    portfolio_id: Optional[str] = None
    model_id: Optional[str] = None
    exchange: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SupervisorErrorHandler:
    """Enhanced error handler specifically for supervisor operations."""
    
    def __init__(self, supervisor_id: str, logger: Optional[logging.Logger] = None):
        self.logger = logger or system_logger
        self.supervisor_id = supervisor_id
        self.error_counts: Dict[SupervisorErrorCategory, int] = {
            category: 0 for category in SupervisorErrorCategory
        }
        self.recovery_strategies: Dict[SupervisorErrorCategory, Callable] = {}
        self.retry_configs: Dict[SupervisorErrorCategory, Dict[str, Any]] = {}
        
    def register_recovery_strategy(
        self,
        category: SupervisorErrorCategory,
        strategy: Callable[[SupervisorError], bool]
    ) -> None:
        """Register a recovery strategy for a specific error category."""
        self.recovery_strategies[category] = strategy
        
    def register_retry_config(
        self,
        category: SupervisorErrorCategory,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_delay: float = 60.0
    ) -> None:
        """Register retry configuration for a specific error category."""
        self.retry_configs[category] = {
            "max_retries": max_retries,
            "backoff_factor": backoff_factor,
            "max_delay": max_delay
        }
        
    def handle_error(
        self,
        error: SupervisorError,
        context: Optional[SupervisorErrorContext] = None
    ) -> bool:
        """Handle a supervisor error with recovery strategies and retry logic."""
        try:
            # Update error counts
            self.error_counts[error.category] += 1
            
            # Create enhanced context
            if context is None:
                context = SupervisorErrorContext()
            context.supervisor_id = self.supervisor_id
            context.timestamp = datetime.now(timezone.utc)
            
            # Log the error
            self.logger.error(
                f"Supervisor error occurred: {error.message}",
                extra={
                    "error_category": error.category.value,
                    "supervisor_id": self.supervisor_id,
                    "context": context.__dict__,
                    "retryable": error.retryable
                }
            )
            
            # Attempt recovery if strategy exists
            if error.category in self.recovery_strategies:
                try:
                    recovery_success = self.recovery_strategies[error.category](error)
                    if recovery_success:
                        self.logger.info(f"Recovery successful for {error.category.value}")
                        return True
                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")
            
            # Handle retry logic if error is retryable
            if error.retryable and error.category in self.retry_configs:
                return self._handle_retry(error, context)
            
            return False
            
        except Exception as handler_error:
            self.logger.error(f"Error in error handler: {handler_error}")
            return False
            
    def _handle_retry(
        self,
        error: SupervisorError,
        context: SupervisorErrorContext
    ) -> bool:
        """Handle retry logic for retryable errors."""
        config = self.retry_configs[error.category]
        max_retries = config["max_retries"]
        backoff_factor = config["backoff_factor"]
        max_delay = config["max_delay"]
        
        for attempt in range(max_retries):
            try:
                delay = min(backoff_factor ** attempt, max_delay)
                time.sleep(delay)
                
                self.logger.info(
                    f"Retry attempt {attempt + 1}/{max_retries} for {error.category.value}"
                )
                
                # Here you would typically retry the original operation
                # For now, we'll just return True to indicate retry was attempted
                return True
                
            except Exception as retry_error:
                self.logger.error(f"Retry attempt {attempt + 1} failed: {retry_error}")
                
        self.logger.error(f"All retry attempts failed for {error.category.value}")
        return False
        
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of error counts and statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            "supervisor_id": self.supervisor_id,
            "total_errors": total_errors,
            "error_counts": self.error_counts,
            "recovery_strategies_registered": len(self.recovery_strategies),
            "retry_configs_registered": len(self.retry_configs),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def handle_supervisor_errors(
    exceptions: tuple = (Exception,),
    default_return: Any = None,
    context: Optional[str] = None,
    category: SupervisorErrorCategory = SupervisorErrorCategory.UNKNOWN,
    retryable: bool = False
):
    """Decorator for handling supervisor errors in functions and methods."""
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                # Create supervisor error context
                error_context = SupervisorErrorContext(
                    operation=context or func.__name__,
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Create appropriate supervisor error
                if isinstance(e, SupervisorError):
                    supervisor_error = e
                else:
                    supervisor_error = SupervisorError(
                        message=str(e),
                        category=category,
                        context=error_context,
                        retryable=retryable
                    )
                
                # Log the error
                system_logger.error(
                    f"Supervisor error in {func.__name__}: {supervisor_error.message}",
                    extra={"error_context": error_context.__dict__}
                )
                
                return default_return
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                # Create supervisor error context
                error_context = SupervisorErrorContext(
                    operation=context or func.__name__,
                    timestamp=datetime.now(timezone.utc)
                )
                
                # Create appropriate supervisor error
                if isinstance(e, SupervisorError):
                    supervisor_error = e
                else:
                    supervisor_error = SupervisorError(
                        message=str(e),
                        category=category,
                        context=error_context,
                        retryable=retryable
                    )
                
                # Log the error
                system_logger.error(
                    f"Supervisor error in {func.__name__}: {supervisor_error.message}",
                    extra={"error_context": error_context.__dict__}
                )
                
                return default_return
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

@contextmanager
def supervisor_error_context(
    operation: str,
    supervisor_id: str,
    **context_kwargs
):
    """Context manager for supervisor operations with error handling."""
    context = SupervisorErrorContext(
        supervisor_id=supervisor_id,
        operation=operation,
        **context_kwargs
    )
    
    try:
        yield context
    except Exception as e:
        # Create supervisor error
        if isinstance(e, SupervisorError):
            supervisor_error = e
        else:
            supervisor_error = SupervisorError(
                message=str(e),
                context=context
            )
        
        # Log the error
        system_logger.error(
            f"Supervisor error in {operation}: {supervisor_error.message}",
            extra={"error_context": context.__dict__}
        )
        
        raise supervisor_error

# Example usage and business logic implementation
class SupervisorComponent:
    """Example supervisor component that demonstrates error handling."""
    
    def __init__(self, component_id: str):
        self.component_id = component_id
        self.error_handler = SupervisorErrorHandler(component_id)
        self.is_running = False
        
        # Register recovery strategies
        self.error_handler.register_recovery_strategy(
            SupervisorErrorCategory.COMPONENT_FAILURE,
            self._restart_component
        )
        
        # Register retry configurations
        self.error_handler.register_retry_config(
            SupervisorErrorCategory.EXCHANGE_INTEGRATION,
            max_retries=5,
            backoff_factor=2.0,
            max_delay=120.0
        )
    
    @handle_supervisor_errors(
        category=SupervisorErrorCategory.COMPONENT_FAILURE,
        retryable=True
    )
    async def start(self) -> bool:
        """Start the supervisor component."""
        try:
            self.logger.info(f"Starting component {self.component_id}")
            # Simulate component startup
            await asyncio.sleep(0.1)
            self.is_running = True
            self.logger.info(f"Component {self.component_id} started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start component {self.component_id}: {e}")
            raise ComponentFailureError(
                component_name=self.component_id,
                operation="start",
                message=str(e)
            )
    
    @handle_supervisor_errors(
        category=SupervisorErrorCategory.PORTFOLIO_MANAGEMENT,
        retryable=False
    )
    async def manage_portfolio(self, portfolio_id: str, action: str) -> bool:
        """Manage portfolio operations with error handling."""
        try:
            self.logger.info(f"Managing portfolio {portfolio_id} with action {action}")
            # Simulate portfolio management
            await asyncio.sleep(0.05)
            
            if action == "rebalance":
                # Simulate rebalancing logic
                self.logger.info(f"Portfolio {portfolio_id} rebalanced successfully")
            elif action == "optimize":
                # Simulate optimization logic
                self.logger.info(f"Portfolio {portfolio_id} optimized successfully")
            else:
                raise ValueError(f"Unknown action: {action}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Portfolio management failed: {e}")
            raise PortfolioManagementError(
                operation=action,
                portfolio_id=portfolio_id,
                message=str(e)
            )
    
    @handle_supervisor_errors(
        category=SupervisorErrorCategory.RISK_MANAGEMENT,
        retryable=True
    )
    async def assess_risk(self, risk_type: str) -> Dict[str, Any]:
        """Assess risk with error handling and retry logic."""
        try:
            self.logger.info(f"Assessing {risk_type} risk")
            # Simulate risk assessment
            await asyncio.sleep(0.03)
            
            risk_score = 0.15  # Simulated risk score
            risk_level = "LOW" if risk_score < 0.3 else "MEDIUM" if risk_score < 0.7 else "HIGH"
            
            return {
                "risk_type": risk_type,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            raise RiskManagementError(
                risk_type=risk_type,
                operation="assess_risk",
                message=str(e)
            )
    
    def _restart_component(self, error: SupervisorError) -> bool:
        """Recovery strategy for component failures."""
        try:
            self.logger.info(f"Attempting to restart component {self.component_id}")
            self.is_running = False
            # Simulate restart delay
            time.sleep(0.1)
            self.is_running = True
            self.logger.info(f"Component {self.component_id} restarted successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restart component: {e}")
            return False
    
    @property
    def logger(self):
        """Get the component's logger."""
        return self.error_handler.logger

# Example usage
async def main():
    """Example usage of the supervisor error handler."""
    # Create a supervisor component
    component = SupervisorComponent("portfolio_supervisor_001")
    
    try:
        # Start the component
        await component.start()
        
        # Perform portfolio management
        await component.manage_portfolio("PORT_001", "rebalance")
        
        # Assess risk
        risk_result = await component.assess_risk("market_volatility")
        print(f"Risk assessment result: {risk_result}")
        
        # Get error summary
        error_summary = component.error_handler.get_error_summary()
        print(f"Error summary: {error_summary}")
        
    except Exception as e:
        print(f"Main execution failed: {e}")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())