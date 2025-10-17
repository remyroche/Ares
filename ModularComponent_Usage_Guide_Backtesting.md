# ModularComponent Usage Guide - Backtesting

## Overview

The `ModularComponent` abstract class has been fully implemented with comprehensive functionality for creating modular, reusable components in the **backtesting** pipeline. This guide provides detailed usage instructions and examples specifically tailored for trading strategy evaluation workflows.

## ✅ Implementation Status: COMPLETE

The backtesting modular component system is **fully implemented** and ready for use. All core components, utilities, and infrastructure are available:

- ✅ **ModularComponent Architecture**: Complete with backtesting-specific features
- ✅ **Migration Utilities**: Comprehensive tools for converting existing components
- ✅ **Component Registry**: Centralized management and orchestration
- ✅ **Component Orchestrator**: Workflow definition and execution
- ✅ **Component Monitor**: Real-time monitoring and health checks
- ✅ **Core Module Exports**: All components accessible from main backtesting module

## Fully Implemented Abstract Methods

### 1. `initialize() -> bool`
**Purpose**: Initialize the component and its resources.

**Implementation**: 
- Validates configuration using `validate_config()`
- Calls `_initialize_resources()` for component-specific setup
- Sets initialization flag and logs success/failure
- Returns `True` if successful, `False` otherwise

**Override**: Implement `_initialize_resources()` in subclasses for custom initialization logic.

### 2. `process(data: Any, **kwargs) -> Any`
**Purpose**: Process input data with comprehensive error handling.

**Implementation**:
- Checks if component is initialized
- Validates input data using `validate_input()`
- Verifies component can process data using `can_process()`
- Calls `_process_data()` for actual processing
- Handles errors gracefully with proper logging

**Override**: Implement `_process_data()` in subclasses for custom processing logic.

### 3. `validate_input(data: Any) -> ValidationResult`
**Purpose**: Comprehensive input validation with detailed results.

**Implementation**:
- Handles multiple data types (DataFrame, Series, ndarray, list, tuple, dict)
- Uses configurable validation rules from `_get_validation_rules()`
- Performs type-specific validation
- Includes component-specific validation via `_validate_component_specific()`
- Returns detailed ValidationResult with errors, warnings, and metadata

**Override**: Implement `_get_validation_rules()` and `_validate_component_specific()` in subclasses.

### 4. `cleanup() -> None`
**Purpose**: Cleanup resources and reset component state.

**Implementation**:
- Calls `_cleanup_resources()` for component-specific cleanup
- Clears component state
- Resets performance statistics
- Resets initialization flag
- Logs cleanup completion

**Override**: Implement `_cleanup_resources()` in subclasses for custom cleanup logic.

### 5. `get_component_info() -> Dict[str, Any]`
**Purpose**: Get comprehensive component metadata.

**Implementation**:
- Returns component name, type, version, description
- Includes initialization status and configuration
- Lists dependencies and capabilities
- Provides complete component information

**Override**: Override to add component-specific information.

### 6. `get_dependencies() -> List[str]`
**Purpose**: Get list of required dependencies.

**Implementation**:
- Returns default dependencies: `['pandas', 'numpy', 'vectorbt', 'matplotlib']`
- Can be overridden for component-specific dependencies

**Override**: Override to specify actual dependencies.

### 7. `get_output_schema() -> Dict[str, Any]`
**Purpose**: Get expected output schema.

**Implementation**:
- Returns generic schema with type, description, and metadata
- Can be overridden for specific output formats

**Override**: Override to specify actual output schema.

### 8. `get_required_config() -> List[str]`
**Purpose**: Get required configuration parameters.

**Implementation**:
- Returns empty list by default
- Can be overridden for specific configuration requirements

**Override**: Override to specify required configuration keys.

### 9. `can_process(data: Any) -> bool`
**Purpose**: Check if component can process given data.

**Implementation**:
- Validates data is not None
- Checks component is initialized
- Verifies data type compatibility
- Checks memory requirements
- Returns `True` if all checks pass

**Override**: Override for custom processing capability checks.

### 10. `get_processing_capabilities() -> Dict[str, Any]`
**Purpose**: Get component processing capabilities.

**Implementation**:
- Returns supported input/output types
- Indicates parallel processing support
- Specifies memory efficiency
- Lists processing features

**Override**: Override to specify actual capabilities.

### 11. `estimate_processing_time(data: Any) -> float`
**Purpose**: Estimate processing time for given data.

**Implementation**:
- Uses base processing time from config
- Calculates size-based factor
- Applies complexity factor
- Uses performance multiplier
- Returns estimated time in seconds

**Override**: Override for more accurate time estimation.

### 12. `get_memory_requirements(data: Any) -> Dict[str, Any]`
**Purpose**: Get memory requirements for processing data.

**Implementation**:
- Calculates base memory usage
- Handles pandas, numpy, and generic objects
- Applies overhead factor
- Returns estimated and peak memory usage

**Override**: Override for more accurate memory estimation.

## Helper Methods

### Concrete Methods (Available to all subclasses)

- **Configuration Management**: `get_config()`, `update_config()`, `validate_config()`
- **State Management**: `set_state()`, `get_state()`, `clear_state()`, `get_all_state()`
- **Performance Monitoring**: `get_performance_stats()`, `reset_stats()`
- **Lifecycle Management**: `is_initialized()`, `get_status()`
- **Serialization**: `serialize()`, `deserialize()`
- **Safe Processing**: `_safe_process()` - Wraps processing with error handling

### Abstract Helper Methods (Must be overridden)

- `_initialize_resources() -> bool` - Initialize component-specific resources
- `_cleanup_resources() -> None` - Cleanup component-specific resources
- `_process_data(data: Any, **kwargs) -> Any` - Process data with component logic
- `_get_validation_rules() -> Dict[str, Any]` - Get validation rules
- `_validate_component_specific(data: Any) -> Dict[str, Any]` - Component-specific validation

## Example Usage for Backtesting

### 1. Basic Backtesting Engine Component

```python
from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent, create_modular_component
)

class BacktestingEngine(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.strategy_config = self.get_config('strategy', {})
        self.risk_config = self.get_config('risk_management', {})
        self.version = "1.0.0"
        self.description = "Backtesting Engine for Trading Strategies"
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('current_position', 0.0)
            self.set_state('portfolio_value', 100000.0)
            self.set_state('trade_history', [])
            self.set_state('daily_returns', [])
            return True
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        self.set_state('cleaned_up_at', time.time())
        self.set_state('trade_history', [])
        self.set_state('daily_returns', [])
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with component logic."""
        # Get market data
        market_data = data.get('market_data')
        strategy_signals = data.get('strategy_signals')
        
        # Initialize backtesting
        initial_capital = self.get_config('backtesting.initial_capital', 100000.0)
        self.set_state('portfolio_value', initial_capital)
        
        # Run backtesting simulation
        results = self._run_backtest(market_data, strategy_signals)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(results)
        
        # Update state
        self.set_state('backtest_results', results)
        self.set_state('performance_metrics', performance_metrics)
        
        return {
            'results': results,
            'performance_metrics': performance_metrics,
            'portfolio_value': self.get_state('portfolio_value'),
            'total_trades': len(self.get_state('trade_history'))
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_data_points': 100,
            'max_data_points': 1000000,
            'required_keys': ['market_data', 'strategy_signals'],
            'data_types': ['dict'],
            'market_data_columns': ['open', 'high', 'low', 'close', 'volume'],
            'strategy_signals_columns': ['signal', 'confidence']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            # Check required keys
            required_keys = ['market_data', 'strategy_signals']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            # Check market data
            if 'market_data' in data:
                market_data = data['market_data']
                if hasattr(market_data, 'shape'):
                    metadata['market_data_shape'] = market_data.shape
                    
                    if len(market_data) < 100:
                        warnings.append("Insufficient market data for reliable backtesting")
                    
                    # Check required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    if hasattr(market_data, 'columns'):
                        missing_columns = [col for col in required_columns if col not in market_data.columns]
                        if missing_columns:
                            errors.append(f"Missing required columns: {missing_columns}")
            
            # Check strategy signals
            if 'strategy_signals' in data:
                strategy_signals = data['strategy_signals']
                if hasattr(strategy_signals, 'shape'):
                    metadata['strategy_signals_shape'] = strategy_signals.shape
                    
                    if len(strategy_signals) != len(market_data):
                        errors.append("Strategy signals and market data must have same length")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _run_backtest(self, market_data, strategy_signals):
        """Run the backtesting simulation."""
        # Implement backtesting logic
        results = []
        current_position = 0.0
        portfolio_value = self.get_state('portfolio_value')
        
        for i, (date, row) in enumerate(market_data.iterrows()):
            signal = strategy_signals.iloc[i]['signal']
            confidence = strategy_signals.iloc[i]['confidence']
            
            # Execute trade based on signal
            if signal == 'buy' and current_position == 0.0:
                # Buy signal
                current_position = self._calculate_position_size(portfolio_value, confidence)
                portfolio_value -= current_position * row['close']
                self._record_trade(date, 'buy', row['close'], current_position)
                
            elif signal == 'sell' and current_position > 0.0:
                # Sell signal
                portfolio_value += current_position * row['close']
                self._record_trade(date, 'sell', row['close'], current_position)
                current_position = 0.0
            
            # Update portfolio value
            if current_position > 0.0:
                portfolio_value = current_position * row['close']
            
            # Record daily return
            daily_return = (portfolio_value - self.get_state('portfolio_value')) / self.get_state('portfolio_value')
            self._record_daily_return(date, daily_return)
            
            # Update state
            self.set_state('current_position', current_position)
            self.set_state('portfolio_value', portfolio_value)
            
            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'position': current_position,
                'signal': signal,
                'confidence': confidence
            })
        
        return results
    
    def _calculate_position_size(self, portfolio_value, confidence):
        """Calculate position size based on confidence and risk management."""
        max_position = self.risk_config.get('max_position_size', 0.1)
        return portfolio_value * max_position * confidence
    
    def _record_trade(self, date, action, price, quantity):
        """Record a trade in the trade history."""
        trade = {
            'date': date,
            'action': action,
            'price': price,
            'quantity': quantity,
            'value': price * quantity
        }
        trade_history = self.get_state('trade_history', [])
        trade_history.append(trade)
        self.set_state('trade_history', trade_history)
    
    def _record_daily_return(self, date, daily_return):
        """Record daily return."""
        daily_returns = self.get_state('daily_returns', [])
        daily_returns.append({'date': date, 'return': daily_return})
        self.set_state('daily_returns', daily_returns)
    
    def _calculate_performance_metrics(self, results):
        """Calculate performance metrics from backtest results."""
        # Implement performance calculation
        total_return = (self.get_state('portfolio_value') - 100000.0) / 100000.0
        daily_returns = [r['return'] for r in self.get_state('daily_returns', [])]
        
        if daily_returns:
            avg_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        else:
            avg_return = 0
            volatility = 0
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(results),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_max_drawdown(self, results):
        """Calculate maximum drawdown."""
        # Implement max drawdown calculation
        pass
    
    def _calculate_win_rate(self):
        """Calculate win rate from trade history."""
        trade_history = self.get_state('trade_history', [])
        if not trade_history:
            return 0.0
        
        # Implement win rate calculation
        return 0.6  # Placeholder

# Usage
config = {
    'strategy': {
        'parameters': {'lookback': 20, 'threshold': 0.5}
    },
    'risk_management': {
        'max_position_size': 0.1,
        'stop_loss': 0.02
    },
    'backtesting': {
        'initial_capital': 100000.0,
        'start_date': '2020-01-01',
        'end_date': '2023-12-31'
    }
}

component = BacktestingEngine('backtesting_engine', config)

# Initialize
if component.initialize():
    # Process data
    backtest_data = {
        'market_data': market_data,
        'strategy_signals': strategy_signals
    }
    result = component.process(backtest_data)
    # Cleanup
    component.cleanup()
```

### 2. Monte Carlo Engine Component

```python
class MonteCarloEngine(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.mc_config = self.get_config('monte_carlo', {})
        self.simulations = []
    
    def _initialize_resources(self) -> bool:
        """Initialize Monte Carlo engine resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('simulation_count', 0)
            self.set_state('simulation_results', [])
            return True
        except Exception as e:
            self.logger.error(f"Monte Carlo initialization failed: {e}")
            return False
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with Monte Carlo simulation logic."""
        # Get strategy parameters
        strategy_params = data.get('strategy_params')
        market_data = data.get('market_data')
        
        # Run Monte Carlo simulations
        num_simulations = self.mc_config.get('num_simulations', 1000)
        results = []
        
        for i in range(num_simulations):
            simulation_result = self._run_single_simulation(strategy_params, market_data)
            results.append(simulation_result)
            self.set_state('simulation_count', i + 1)
        
        # Calculate statistics
        statistics = self._calculate_simulation_statistics(results)
        
        # Update state
        self.set_state('simulation_results', results)
        self.set_state('statistics', statistics)
        
        return {
            'simulations': results,
            'statistics': statistics,
            'num_simulations': num_simulations
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for Monte Carlo engine."""
        return {
            'min_simulations': 100,
            'max_simulations': 10000,
            'required_keys': ['strategy_params', 'market_data'],
            'data_types': ['dict']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with Monte Carlo specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            if 'strategy_params' in data and 'market_data' in data:
                strategy_params = data['strategy_params']
                market_data = data['market_data']
                
                if len(market_data) < 100:
                    warnings.append("Insufficient market data for reliable Monte Carlo simulation")
                
                metadata['strategy_params_count'] = len(strategy_params)
                metadata['market_data_size'] = len(market_data)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _run_single_simulation(self, strategy_params, market_data):
        """Run a single Monte Carlo simulation."""
        # Implement single simulation logic
        pass
    
    def _calculate_simulation_statistics(self, results):
        """Calculate statistics from simulation results."""
        # Implement statistics calculation
        pass
```

### 3. Risk Management Component

```python
class RiskManagement(ModularComponent):
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(name, config, logger)
        self.risk_config = self.get_config('risk_management', {})
        self.risk_metrics = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize risk management resources."""
        try:
            self.set_state('initialized_at', time.time())
            self.set_state('risk_metrics', {})
            self.set_state('risk_alerts', [])
            return True
        except Exception as e:
            self.logger.error(f"Risk management initialization failed: {e}")
            return False
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data with risk management logic."""
        # Get portfolio data
        portfolio_data = data.get('portfolio_data')
        market_data = data.get('market_data')
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_data, market_data)
        
        # Check risk limits
        risk_alerts = self._check_risk_limits(risk_metrics)
        
        # Update state
        self.set_state('risk_metrics', risk_metrics)
        self.set_state('risk_alerts', risk_alerts)
        
        return {
            'risk_metrics': risk_metrics,
            'risk_alerts': risk_alerts,
            'risk_status': 'safe' if not risk_alerts else 'warning'
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for risk management."""
        return {
            'min_data_points': 50,
            'required_keys': ['portfolio_data', 'market_data'],
            'data_types': ['dict']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with risk management specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            if 'portfolio_data' in data and 'market_data' in data:
                portfolio_data = data['portfolio_data']
                market_data = data['market_data']
                
                if len(portfolio_data) < 50:
                    warnings.append("Insufficient portfolio data for reliable risk assessment")
                
                metadata['portfolio_size'] = len(portfolio_data)
                metadata['market_data_size'] = len(market_data)
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def _calculate_risk_metrics(self, portfolio_data, market_data):
        """Calculate risk metrics."""
        # Implement risk calculation
        pass
    
    def _check_risk_limits(self, risk_metrics):
        """Check risk limits and generate alerts."""
        # Implement risk limit checking
        pass
```

## Key Features for Backtesting

1. **Comprehensive Error Handling**: All methods include proper error handling and logging
2. **Performance Monitoring**: Automatic performance statistics collection
3. **State Management**: Built-in state management for portfolio and trade data
4. **Configuration Management**: Flexible configuration system for strategy parameters
5. **Validation Framework**: Comprehensive input validation for market data
6. **Serialization Support**: Built-in serialization for strategy persistence
7. **Memory Management**: Memory requirement estimation and checking
8. **Lifecycle Management**: Proper initialization and cleanup
9. **Extensibility**: Easy to extend with custom functionality
10. **Documentation**: Comprehensive docstrings and examples
11. **Backtesting-Specific**: Optimized for trading strategy evaluation workflows
12. **Portfolio State Tracking**: Built-in support for portfolio and trade state
13. **Component Registry**: Centralized component management and discovery
14. **Workflow Orchestration**: Complex backtesting workflow execution
15. **Real-time Monitoring**: Health checks and performance tracking
16. **Migration Support**: Easy conversion of existing components
17. **Strategy Checkpointing**: Persistent state management for strategies
18. **Risk Management Integration**: Comprehensive risk monitoring and alerts
19. **Monte Carlo Support**: Specialized support for Monte Carlo simulations
20. **Walk-Forward Analysis**: Built-in support for walk-forward validation

## Best Practices for Backtesting

1. **Always call `initialize()`** before using the component
2. **Implement all abstract helper methods** for proper functionality
3. **Use `_safe_process()`** for automatic error handling and performance tracking
4. **Override validation methods** for component-specific validation
5. **Call `cleanup()`** when done with the component
6. **Use state management** for storing portfolio and trade state
7. **Implement proper error handling** in custom methods
8. **Provide accurate capability information** for better integration
9. **Track portfolio state** using state management
10. **Implement strategy checkpointing** for persistence
11. **Use performance monitoring** for backtesting optimization
12. **Validate market data** thoroughly before processing
13. **Use the component registry** for centralized management
14. **Implement workflow orchestration** for complex processes
15. **Enable real-time monitoring** for production deployments
16. **Use migration utilities** for existing components
17. **Follow naming conventions** for components and workflows
18. **Document component capabilities** clearly
19. **Use configuration templates** for consistent setups
20. **Monitor health metrics** regularly

## Advanced Usage Examples

### Component Registry Management

```python
from src.training.steps.backtesting import (
    get_registry, register_component, ComponentType, 
    initialize_component, start_component, get_component_status
)

# Get the global registry
registry = get_registry()

# Register a custom backtesting component
registry.register_component(
    name='my_custom_engine',
    component_type=ComponentType.BACKTESTING_ENGINE,
    component_class=MyCustomBacktestingEngine,
    dependencies=['data_loader', 'risk_management'],
    metadata={'version': '1.0.0', 'author': 'My Team'}
)

# Initialize and start the component
registry.initialize_component('my_custom_engine')
registry.start_component('my_custom_engine')

# Check component status
status = registry.get_component_status('my_custom_engine')
print(f"Component status: {status['status']}")
print(f"Health: {status['health_status']}")
```

### Workflow Orchestration

```python
from src.training.steps.backtesting import (
    define_workflow, execute_workflow, WorkflowStep, 
    ExecutionMode, get_workflow_status
)

# Define a complex backtesting workflow
workflow = define_workflow(
    name='advanced_backtesting_pipeline',
    description='Complete backtesting pipeline with risk management',
    steps=[
        WorkflowStep('load_market_data', 'data_loader'),
        WorkflowStep('generate_features', 'feature_generator', 
                    dependencies=['load_market_data']),
        WorkflowStep('optimize_parameters', 'parameter_optimizer',
                    dependencies=['generate_features']),
        WorkflowStep('run_backtest', 'backtesting_engine',
                    dependencies=['optimize_parameters']),
        WorkflowStep('risk_analysis', 'risk_analyzer',
                    dependencies=['run_backtest']),
        WorkflowStep('performance_analysis', 'performance_analyzer',
                    dependencies=['risk_analysis']),
        WorkflowStep('generate_report', 'reporting_engine',
                    dependencies=['performance_analysis'])
    ],
    execution_mode=ExecutionMode.PIPELINE,
    max_parallel_workers=4,
    timeout=3600,  # 1 hour
    enable_checkpointing=True,
    enable_monitoring=True
)

# Execute the workflow
workflow_id = execute_workflow(
    workflow, 
    input_data={
        'symbol': 'BTCUSDT',
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'strategy_config': {'lookback': 20, 'threshold': 0.5}
    }
)

# Monitor workflow progress
status = get_workflow_status(workflow_id)
print(f"Workflow status: {status['status']}")
print(f"Completed steps: {status['completed_steps']}")
print(f"Current step: {status['current_step']}")
```

### Real-time Monitoring

```python
from src.training.steps.backtesting import (
    start_monitoring, get_monitoring_dashboard_data,
    get_component_health, get_performance_metrics,
    get_alerts, AlertLevel
)

# Start the monitoring system
start_monitoring()

# Get dashboard data
dashboard = get_monitoring_dashboard_data()
print(f"Total components: {dashboard['components']['total']}")
print(f"Healthy components: {dashboard['components']['healthy']}")
print(f"Critical alerts: {dashboard['alerts']['critical']}")

# Get specific component health
health = get_component_health('backtesting_engine')
if health:
    print(f"Health score: {health.health_score:.2f}")
    print(f"Status: {health.status.value}")

# Get performance metrics
metrics = get_performance_metrics('backtesting_engine', 'success_rate')
if metrics:
    latest_metric = metrics[-1]
    print(f"Latest success rate: {latest_metric.value:.2%}")

# Get alerts
critical_alerts = get_alerts(level=AlertLevel.CRITICAL)
for alert in critical_alerts:
    print(f"CRITICAL: {alert.component_name} - {alert.message}")
```

### Migration from Existing Components

```python
from src.training.steps.backtesting import (
    BacktestingComponentAnalyzer, BacktestingComponentMigrator,
    create_backtesting_component_wrapper
)

# Analyze an existing component
analyzer = BacktestingComponentAnalyzer()
analysis = analyzer.analyze_component(
    'existing_component.py', 
    'ExistingBacktestingComponent'
)

print(f"Compatibility score: {analysis.compatibility_score:.2f}")
print(f"Migration complexity: {analysis.migration_complexity}")
print(f"Recommendations: {analysis.recommendations}")

# Migrate the component
migrator = BacktestingComponentMigrator()
result = migrator.migrate_component(
    'existing_component.py',
    'ExistingBacktestingComponent'
)

if result.success:
    print("Migration successful!")
    print(f"Issues: {result.issues}")
    print(f"Recommendations: {result.recommendations}")
else:
    print(f"Migration failed: {result.issues}")

# Create a wrapper for backward compatibility
WrappedComponent = create_backtesting_component_wrapper(ExistingComponent)
wrapped = WrappedComponent('wrapped_component', config)
```

### Configuration Management

```python
from src.training.steps.backtesting import (
    get_backtesting_config_template, validate_backtesting_config
)

# Get a configuration template
config = get_backtesting_config_template('advanced_backtesting')
print(f"Initial capital: {config['backtesting']['initial_capital']}")
print(f"Commission: {config['backtesting']['commission']}")
print(f"Risk management: {config['risk_management']}")

# Validate configuration
is_valid = validate_backtesting_config(config)
print(f"Configuration valid: {is_valid}")

# Use configuration with component
component = create_backtesting_component('backtesting_engine', 'my_engine', config)
```

### Performance Monitoring and Optimization

```python
from src.training.steps.backtesting import (
    get_performance_metrics, create_performance_visualization
)

# Get performance metrics for analysis
metrics = get_performance_metrics(
    'backtesting_engine',
    start_time=time.time() - 3600,  # Last hour
    end_time=time.time()
)

# Analyze performance trends
success_rates = [m.value for m in metrics if m.name == 'success_rate']
if success_rates:
    avg_success_rate = sum(success_rates) / len(success_rates)
    print(f"Average success rate: {avg_success_rate:.2%}")

# Create performance visualization
create_performance_visualization(
    'backtesting_engine',
    'success_rate',
    'performance_chart.png',
    start_time=time.time() - 86400,  # Last 24 hours
    end_time=time.time()
)
```

This comprehensive system provides everything needed for modern, scalable backtesting workflows with full monitoring, orchestration, and management capabilities.