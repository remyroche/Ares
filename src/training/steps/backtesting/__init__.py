# Note: consolidated_backtesting_step has been removed as it was legacy code
from .final_parameters_optimization import FinalParametersOptimizer

# New backtesting steps
from .basic_backtesting_pre import BasicBacktestingPreStep, BasicBacktestingPreConfig, BasicBacktestingPreResults
from .basic_backtesting_post import BasicBacktestingPostStep, BasicBacktestingPostConfig, BasicBacktestingPostResults
from .walk_forward_validation import WalkForwardValidationStep, WalkForwardValidationConfig, WalkForwardValidationResults
from .monte_carlo_simulation import MonteCarloSimulationStep, MonteCarloSimulationConfig, MonteCarloSimulationResults
from .ab_testing import ABTestingStep, ABTestingConfig, ABTestingResults
from .reporting import ReportingStep, ReportingConfig, ReportingResults

# Unified infrastructure
from .unified_data_loader import (
    UnifiedDataLoader, DataLoadingConfig, LoadedData, DataSourceType, DataLoadingMode,
    get_unified_data_loader, load_backtesting_data, cleanup_data_loader
)
from .memory_optimizer import (
    BacktestingMemoryOptimizer, MemoryStats, get_backtesting_memory_optimizer,
    optimize_backtesting_data, cleanup_backtesting_memory, memory_managed_backtesting
)
from .improved_trading_strategies import (
    ImprovedTradingStrategy, StrategyFactory, StrategyConfig, TradingSignal,
    StrategyType, MarketRegime, SignalStrength, TechnicalIndicators,
    create_baseline_strategy, create_optimized_strategy
)

# Modular Component Architecture
from .unified_data_driven_pipeline.core import (
    ModularComponent,
    ExampleModularComponent,
    ValidationLevel,
    ValidationResult,
    ErrorInfo,
    PerformanceMetric,
    MetricType,
    MetricLevel,
    ErrorSeverity,
    ErrorCategory,
    create_modular_component,
    create_backtesting_component,
    validate_backtesting_environment,
    ComponentAnalysis,
    MigrationStrategy,
    MigrationResult,
    BacktestingComponentAnalyzer,
    BacktestingMigrationStrategy,
    BacktestingComponentMigrator,
    create_backtesting_component_wrapper,
    validate_backtesting_migration_compatibility,
    generate_backtesting_migration_report,
    get_backtesting_config_template,
    validate_backtesting_config,
    create_backtesting_component_factory
)

# Component Registry and Orchestration
from .unified_data_driven_pipeline.core.component_registry import (
    ComponentStatus,
    ComponentType,
    ComponentInfo,
    DependencyGraph,
    BacktestingComponentRegistry,
    get_registry,
    register_component,
    get_component,
    initialize_component,
    start_component,
    stop_component,
    cleanup_component,
    get_component_status,
    get_all_components,
    run_health_checks
)

from .unified_data_driven_pipeline.core.component_orchestrator import (
    WorkflowStatus,
    ExecutionMode,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowExecution,
    BacktestingWorkflowOrchestrator,
    get_orchestrator,
    define_workflow,
    execute_workflow,
    get_workflow_status,
    cancel_workflow
)

from .unified_data_driven_pipeline.core.component_monitor import (
    AlertLevel,
    MetricType,
    Alert,
    PerformanceMetric,
    ComponentHealth,
    MonitoringConfig,
    BacktestingComponentMonitor,
    get_monitor,
    start_monitoring,
    stop_monitoring,
    get_component_health,
    get_all_component_health,
    get_performance_metrics,
    get_alerts,
    get_monitoring_dashboard_data
)

# Modular Component Architecture
from .unified_data_driven_pipeline.core import (
    ModularComponent,
    ExampleModularComponent,
    ValidationLevel,
    ValidationResult,
    ErrorInfo,
    PerformanceMetric,
    MetricType,
    MetricLevel,
    ErrorSeverity,
    ErrorCategory,
    create_modular_component,
    create_backtesting_component,
    validate_backtesting_environment,
    ComponentAnalysis,
    MigrationStrategy,
    MigrationResult,
    BacktestingComponentAnalyzer,
    BacktestingMigrationStrategy,
    BacktestingComponentMigrator,
    create_backtesting_component_wrapper,
    validate_backtesting_migration_compatibility,
    generate_backtesting_migration_report,
    get_backtesting_config_template,
    validate_backtesting_config,
    create_backtesting_component_factory
)

# Component Registry and Orchestration
from .unified_data_driven_pipeline.core.component_registry import (
    ComponentStatus,
    ComponentType,
    ComponentInfo,
    DependencyGraph,
    BacktestingComponentRegistry,
    get_registry,
    register_component,
    get_component,
    initialize_component,
    start_component,
    stop_component,
    cleanup_component,
    get_component_status,
    get_all_components,
    run_health_checks
)

from .unified_data_driven_pipeline.core.component_orchestrator import (
    WorkflowStatus,
    ExecutionMode,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowExecution,
    BacktestingWorkflowOrchestrator,
    get_orchestrator,
    define_workflow,
    execute_workflow,
    get_workflow_status,
    cancel_workflow
)

from .unified_data_driven_pipeline.core.component_monitor import (
    AlertLevel,
    MetricType,
    Alert,
    PerformanceMetric,
    ComponentHealth,
    MonitoringConfig,
    BacktestingComponentMonitor,
    get_monitor,
    start_monitoring,
    stop_monitoring,
    get_component_health,
    get_all_component_health,
    get_performance_metrics,
    get_alerts,
    get_monitoring_dashboard_data
)

__all__ = [
    # Note: Original consolidated backtesting components have been removed
    'FinalParametersOptimizer',

    # New backtesting steps
    'BasicBacktestingPreStep',
    'BasicBacktestingPreConfig',
    'BasicBacktestingPreResults',
    'BasicBacktestingPostStep',
    'BasicBacktestingPostConfig',
    'BasicBacktestingPostResults',
    'WalkForwardValidationStep',
    'WalkForwardValidationConfig',
    'WalkForwardValidationResults',
    'MonteCarloSimulationStep',
    'MonteCarloSimulationConfig',
    'MonteCarloSimulationResults',
    'ABTestingStep',
    'ABTestingConfig',
    'ABTestingResults',
    'ReportingStep',
    'ReportingConfig',
    'ReportingResults',

    # Unified infrastructure
    'UnifiedDataLoader',
    'DataLoadingConfig',
    'LoadedData',
    'DataSourceType',
    'DataLoadingMode',
    'get_unified_data_loader',
    'load_backtesting_data',
    'cleanup_data_loader',
    'BacktestingMemoryOptimizer',
    'MemoryStats',
    'get_backtesting_memory_optimizer',
    'optimize_backtesting_data',
    'cleanup_backtesting_memory',
    'memory_managed_backtesting',

    # Improved trading strategies
    'ImprovedTradingStrategy',
    'StrategyFactory',
    'StrategyConfig',
    'TradingSignal',
    'StrategyType',
    'MarketRegime',
    'SignalStrength',
    'TechnicalIndicators',
    'create_baseline_strategy',
    'create_optimized_strategy',

    # Modular Component Architecture
    'ModularComponent',
    'ExampleModularComponent',
    'ValidationLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'MetricType',
    'MetricLevel',
    'ErrorSeverity',
    'ErrorCategory',
    'create_modular_component',
    'create_backtesting_component',
    'validate_backtesting_environment',
    'ComponentAnalysis',
    'MigrationStrategy',
    'MigrationResult',
    'BacktestingComponentAnalyzer',
    'BacktestingMigrationStrategy',
    'BacktestingComponentMigrator',
    'create_backtesting_component_wrapper',
    'validate_backtesting_migration_compatibility',
    'generate_backtesting_migration_report',
    'get_backtesting_config_template',
    'validate_backtesting_config',
    'create_backtesting_component_factory',

    # Component Registry and Orchestration
    'ComponentStatus',
    'ComponentType',
    'ComponentInfo',
    'DependencyGraph',
    'BacktestingComponentRegistry',
    'get_registry',
    'register_component',
    'get_component',
    'initialize_component',
    'start_component',
    'stop_component',
    'cleanup_component',
    'get_component_status',
    'get_all_components',
    'run_health_checks',

    # Workflow Orchestration
    'WorkflowStatus',
    'ExecutionMode',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowExecution',
    'BacktestingWorkflowOrchestrator',
    'get_orchestrator',
    'define_workflow',
    'execute_workflow',
    'get_workflow_status',
    'cancel_workflow',

    # Component Monitoring
    'AlertLevel',
    'Alert',
    'ComponentHealth',
    'MonitoringConfig',
    'BacktestingComponentMonitor',
    'get_monitor',
    'start_monitoring',
    'stop_monitoring',
    'get_component_health',
    'get_all_component_health',
    'get_performance_metrics',
    'get_alerts',
    'get_monitoring_dashboard_data'
]
