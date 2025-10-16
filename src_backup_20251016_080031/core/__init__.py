
# Import dependency injection components (optional)
try:
    from .dependency_injection import (
        ComponentFactory,
        DependencyContainer,
        ModularTradingSystem,
        ServiceRegistration,
    )
    DEPENDENCY_INJECTION_AVAILABLE = True
except ImportError:
    DEPENDENCY_INJECTION_AVAILABLE = False

# Import monitoring and reporting components (optional)
try:
    from .decorators import (
        monitor_step03_functions,
        handle_step03_errors,
        monitor_function_calls,
        handle_errors_enhanced,
        FunctionCallMonitor,
        EnhancedErrorHandler,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from .reporting import (
        Step03ExecutionReporter,
        Step03ExecutionReport,
    )
    REPORTING_AVAILABLE = True
except ImportError:
    REPORTING_AVAILABLE = False

# src/core/__init__.py

# Build __all__ list conditionally
__all__ = []

if DEPENDENCY_INJECTION_AVAILABLE:
    __all__.extend([
        "DependencyContainer",
        "ComponentFactory", 
        "ModularTradingSystem",
        "ServiceRegistration",
    ])

if MONITORING_AVAILABLE:
    __all__.extend([
        "monitor_step03_functions",
        "handle_step03_errors",
        "monitor_function_calls",
        "handle_errors_enhanced",
        "FunctionCallMonitor",
        "EnhancedErrorHandler",
    ])

if REPORTING_AVAILABLE:
    __all__.extend([
        "Step03ExecutionReporter",
        "Step03ExecutionReport",
    ])
