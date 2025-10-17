"""
Core Module for Backtesting Modular Components

This module provides the core infrastructure for the backtesting modular component
system, including the base ModularComponent class, migration utilities, and
supporting infrastructure.

Key Components:
- ModularComponent: Base class for all backtesting components
- Migration utilities for converting existing components
- Component analysis and compatibility checking
- Backtesting-specific features and optimizations
"""

from .modular_architecture import (
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
    validate_backtesting_environment
)

from .migration_utils import (
    ComponentAnalysis,
    MigrationStrategy,
    MigrationResult,
    BacktestingComponentAnalyzer,
    BacktestingMigrationStrategy,
    BacktestingComponentMigrator,
    create_backtesting_component_wrapper,
    validate_backtesting_migration_compatibility,
    generate_backtesting_migration_report
)

# Backtesting-specific component types
BACKTESTING_COMPONENT_TYPES = [
    'backtesting_engine',
    'monte_carlo_engine',
    'risk_management',
    'portfolio_manager',
    'strategy_optimizer',
    'performance_analyzer',
    'reporting_engine',
    'data_loader',
    'feature_generator',
    'signal_generator'
]

# Backtesting-specific configuration keys
BACKTESTING_CONFIG_KEYS = [
    'backtesting.initial_capital',
    'backtesting.commission',
    'backtesting.slippage',
    'backtesting.enable_risk_management',
    'risk_management.max_drawdown',
    'risk_management.max_position_size',
    'risk_management.max_correlation',
    'strategy.parameters',
    'strategy.optimization',
    'performance.metrics',
    'performance.monitoring'
]

# Backtesting-specific state keys
BACKTESTING_STATE_KEYS = [
    'portfolio_value',
    'current_position',
    'trade_history',
    'daily_returns',
    'risk_metrics',
    'strategy_parameters',
    'performance_metrics',
    'backtest_results',
    'optimization_results'
]

# Backtesting-specific capabilities
BACKTESTING_CAPABILITIES = {
    'portfolio_tracking': 'Track portfolio state and positions',
    'trade_execution': 'Execute trades based on signals',
    'risk_management': 'Manage risk and position sizing',
    'performance_metrics': 'Calculate performance metrics',
    'strategy_optimization': 'Optimize strategy parameters',
    'monte_carlo_simulation': 'Run Monte Carlo simulations',
    'walk_forward_analysis': 'Perform walk-forward analysis',
    'reporting': 'Generate backtesting reports',
    'data_loading': 'Load and preprocess market data',
    'feature_generation': 'Generate trading features',
    'signal_generation': 'Generate trading signals'
}

# Backtesting-specific validation rules
BACKTESTING_VALIDATION_RULES = {
    'market_data': {
        'required_columns': ['open', 'high', 'low', 'close', 'volume'],
        'min_data_points': 100,
        'max_data_points': 1000000,
        'data_types': ['pandas.DataFrame']
    },
    'strategy_signals': {
        'required_columns': ['signal', 'confidence'],
        'signal_values': ['buy', 'sell', 'hold'],
        'confidence_range': [0.0, 1.0]
    },
    'portfolio_data': {
        'required_keys': ['portfolio_value', 'positions'],
        'min_data_points': 1,
        'data_types': ['dict', 'pandas.DataFrame']
    }
}

# Backtesting-specific error categories
BACKTESTING_ERROR_CATEGORIES = [
    'validation',
    'processing',
    'memory',
    'configuration',
    'dependency',
    'backtesting',
    'risk_management',
    'portfolio',
    'strategy',
    'optimization',
    'monte_carlo',
    'walk_forward',
    'reporting'
]

# Backtesting-specific performance metrics
BACKTESTING_PERFORMANCE_METRICS = [
    'total_return',
    'annualized_return',
    'volatility',
    'sharpe_ratio',
    'sortino_ratio',
    'max_drawdown',
    'calmar_ratio',
    'win_rate',
    'profit_factor',
    'recovery_factor',
    'var',
    'cvar',
    'beta',
    'alpha',
    'information_ratio'
]

# Backtesting-specific health checks
BACKTESTING_HEALTH_CHECKS = [
    'portfolio_health',
    'risk_health',
    'strategy_health',
    'data_health',
    'performance_health',
    'memory_health',
    'configuration_health'
]

# Backtesting-specific serialization keys
BACKTESTING_SERIALIZATION_KEYS = [
    'component_class',
    'name',
    'config',
    'state',
    'performance_stats',
    'initialized',
    'timestamp',
    'version',
    'description',
    'capabilities',
    'backtesting_specific': [
        'portfolio_state',
        'trade_history',
        'performance_metrics',
        'strategy_parameters',
        'risk_metrics'
    ]
]

# Backtesting-specific configuration templates
BACKTESTING_CONFIG_TEMPLATES = {
    'basic_backtesting': {
        'backtesting': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005,
            'enable_risk_management': True
        },
        'risk_management': {
            'max_drawdown': 0.15,
            'max_position_size': 0.1,
            'max_correlation': 0.7
        },
        'performance': {
            'enable_metrics': True,
            'enable_health_checks': True
        }
    },
    'advanced_backtesting': {
        'backtesting': {
            'initial_capital': 1000000.0,
            'commission': 0.0005,
            'slippage': 0.0002,
            'enable_risk_management': True,
            'enable_portfolio_optimization': True
        },
        'risk_management': {
            'max_drawdown': 0.10,
            'max_position_size': 0.05,
            'max_correlation': 0.5,
            'var_confidence': 0.95,
            'enable_stress_testing': True
        },
        'strategy': {
            'optimization_method': 'genetic_algorithm',
            'optimization_parameters': {
                'population_size': 100,
                'generations': 50,
                'mutation_rate': 0.1
            }
        },
        'performance': {
            'enable_metrics': True,
            'enable_health_checks': True,
            'enable_serialization': True,
            'enable_monitoring': True
        }
    },
    'monte_carlo_simulation': {
        'monte_carlo': {
            'num_simulations': 1000,
            'confidence_level': 0.95,
            'enable_parallel': True
        },
        'backtesting': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005
        },
        'risk_management': {
            'max_drawdown': 0.20,
            'max_position_size': 0.15
        }
    }
}

# Backtesting-specific utility functions
def get_backtesting_config_template(template_name: str = 'basic_backtesting') -> Dict[str, Any]:
    """
    Get a backtesting configuration template.
    
    Args:
        template_name: Name of the template to retrieve
        
    Returns:
        Configuration template dictionary
    """
    if template_name not in BACKTESTING_CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(BACKTESTING_CONFIG_TEMPLATES.keys())}")
    
    return BACKTESTING_CONFIG_TEMPLATES[template_name].copy()


def validate_backtesting_config(config: Dict[str, Any]) -> bool:
    """
    Validate a backtesting configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required sections
        required_sections = ['backtesting', 'risk_management', 'performance']
        for section in required_sections:
            if section not in config:
                return False
        
        # Validate backtesting section
        backtesting = config['backtesting']
        if not isinstance(backtesting.get('initial_capital'), (int, float)):
            return False
        if not isinstance(backtesting.get('commission'), (int, float)):
            return False
        
        # Validate risk management section
        risk_mgmt = config['risk_management']
        if not isinstance(risk_mgmt.get('max_drawdown'), (int, float)):
            return False
        if not isinstance(risk_mgmt.get('max_position_size'), (int, float)):
            return False
        
        return True
    except Exception:
        return False


def create_backtesting_component_factory(component_type: str) -> Callable:
    """
    Create a factory function for a specific backtesting component type.
    
    Args:
        component_type: Type of component to create factory for
        
    Returns:
        Factory function
    """
    def factory(name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """Factory function for creating backtesting components."""
        if component_type == 'backtesting_engine':
            from .backtesting_engine import BacktestingEngine
            return BacktestingEngine(name, config, logger)
        elif component_type == 'monte_carlo_engine':
            from .monte_carlo_engine import MonteCarloEngine
            return MonteCarloEngine(name, config, logger)
        elif component_type == 'risk_management':
            from .risk_management import RiskManagement
            return RiskManagement(name, config, logger)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    
    return factory


# Export all public classes, functions, and constants
__all__ = [
    # Core classes
    'ModularComponent',
    'ExampleModularComponent',
    
    # Enums and data classes
    'ValidationLevel',
    'ValidationResult',
    'ErrorInfo',
    'PerformanceMetric',
    'MetricType',
    'MetricLevel',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Migration utilities
    'ComponentAnalysis',
    'MigrationStrategy',
    'MigrationResult',
    'BacktestingComponentAnalyzer',
    'BacktestingMigrationStrategy',
    'BacktestingComponentMigrator',
    'create_backtesting_component_wrapper',
    'validate_backtesting_migration_compatibility',
    'generate_backtesting_migration_report',
    
    # Factory functions
    'create_modular_component',
    'create_backtesting_component',
    'create_backtesting_component_factory',
    
    # Utility functions
    'validate_backtesting_environment',
    'get_backtesting_config_template',
    'validate_backtesting_config',
    
    # Constants
    'BACKTESTING_COMPONENT_TYPES',
    'BACKTESTING_CONFIG_KEYS',
    'BACKTESTING_STATE_KEYS',
    'BACKTESTING_CAPABILITIES',
    'BACKTESTING_VALIDATION_RULES',
    'BACKTESTING_ERROR_CATEGORIES',
    'BACKTESTING_PERFORMANCE_METRICS',
    'BACKTESTING_HEALTH_CHECKS',
    'BACKTESTING_SERIALIZATION_KEYS',
    'BACKTESTING_CONFIG_TEMPLATES'
]