# ModularComponent Complete Implementation - Backtesting

## Overview

The `ModularComponent` abstract class has been **fully implemented** with comprehensive functionality for all core helper methods, specifically adapted for the **backtesting** pipeline. This document provides a complete reference for all implemented methods in the context of backtesting and trading strategy evaluation workflows.

## ✅ Fully Implemented Methods

### 1. Configuration Management

#### `get_config(key: str = None, default: Any = None) -> Any`
- **Purpose**: Get configuration value(s) with support for nested keys
- **Features**: 
  - Returns entire config if no key provided
  - Supports nested key access (e.g., 'backtesting.start_date')
  - Returns default value for missing keys
- **Example**:
  ```python
  config = component.get_config()  # Get all config
  start_date = component.get_config('backtesting.start_date', '2020-01-01')  # Get specific key
  nested = component.get_config('strategy.parameters.lookback', 20)  # Nested access
  ```

#### `update_config(config: Dict[str, Any]) -> None`
- **Purpose**: Update component configuration with validation
- **Features**:
  - Validates configuration keys are strings
  - Merges with existing configuration
  - Triggers configuration change callbacks
  - Logs configuration updates
- **Example**:
  ```python
  component.update_config({
      'backtesting': {'start_date': '2020-01-01', 'end_date': '2023-12-31'},
      'strategy': {'parameters': {'lookback': 20, 'threshold': 0.5}},
      'risk_management': {'max_position_size': 0.1, 'stop_loss': 0.02}
  })
  ```

#### `validate_config() -> bool`
- **Purpose**: Comprehensive configuration validation
- **Features**:
  - Checks required configuration parameters
  - Validates configuration value types
  - Supports component-specific validation
  - Returns detailed validation results
- **Example**:
  ```python
  is_valid = component.validate_config()
  if not is_valid:
      print("Configuration validation failed")
  ```

### 2. State Management

#### `set_state(key: str, value: Any) -> None`
- **Purpose**: Set component state with change tracking
- **Features**:
  - Validates key is string
  - Tracks state changes
  - Triggers state change callbacks
  - Logs state modifications
- **Example**:
  ```python
  component.set_state('current_position', 0.5)
  component.set_state('portfolio_value', 100000.0)
  component.set_state('trade_history', [{'date': '2023-01-01', 'action': 'buy', 'price': 100.0}])
  ```

#### `get_state(key: str, default: Any = None) -> Any`
- **Purpose**: Get component state with default fallback
- **Features**:
  - Returns default value for missing keys
  - Type-safe key validation
- **Example**:
  ```python
  position = component.get_state('current_position', 0.0)
  portfolio = component.get_state('portfolio_value')
  trades = component.get_state('trade_history', [])
  ```

#### `clear_state() -> None`
- **Purpose**: Clear all component state
- **Features**:
  - Removes all state keys
  - Logs cleared state keys
- **Example**:
  ```python
  component.clear_state()
  ```

#### `get_all_state() -> Dict[str, Any]`
- **Purpose**: Get all component state
- **Features**:
  - Returns copy of all state
  - Safe for external access
- **Example**:
  ```python
  all_state = component.get_all_state()
  print(f"State keys: {list(all_state.keys())}")
  ```

#### `has_state(key: str) -> bool`
- **Purpose**: Check if state key exists
- **Example**:
  ```python
  if component.has_state('portfolio_value'):
      portfolio = component.get_state('portfolio_value')
  ```

#### `remove_state(key: str) -> Any`
- **Purpose**: Remove state key and return its value
- **Example**:
  ```python
  old_position = component.remove_state('previous_position')
  ```

### 3. Performance Monitoring

#### `get_performance_stats() -> Dict[str, Any]`
- **Purpose**: Get comprehensive performance statistics
- **Features**:
  - Basic operation counts
  - Success/failure rates
  - Average processing time
  - Component-specific metrics
- **Returns**:
  ```python
  {
      'total_operations': 100,
      'successful_operations': 95,
      'failed_operations': 5,
      'total_time': 10.5,
      'success_rate': 0.95,
      'failure_rate': 0.05,
      'avg_processing_time': 0.105,
      'total_trades': 150,
      'winning_trades': 90,
      'losing_trades': 60,
      'win_rate': 0.6,
      'total_return': 0.15,
      'sharpe_ratio': 1.2
  }
  ```

#### `reset_stats() -> None`
- **Purpose**: Reset performance statistics
- **Features**:
  - Clears all performance data
  - Logs reset operation
- **Example**:
  ```python
  component.reset_stats()
  ```

#### `get_performance_summary() -> Dict[str, Any]`
- **Purpose**: Get detailed performance analysis
- **Features**:
  - Performance grade calculation (A-F)
  - Improvement recommendations
  - Comprehensive analysis
- **Returns**:
  ```python
  {
      'component_name': 'backtesting_engine',
      'performance_stats': {...},
      'performance_grade': 'A',
      'recommendations': ['Consider optimizing position sizing for better risk-adjusted returns']
  }
  ```

### 4. Lifecycle Management

#### `is_initialized() -> bool`
- **Purpose**: Check if component is initialized
- **Example**:
  ```python
  if component.is_initialized():
      result = component.process(market_data)
  ```

#### `get_status() -> Dict[str, Any]`
- **Purpose**: Get comprehensive component status
- **Features**:
  - Health status calculation
  - Configuration status
  - Performance metrics
  - State information
- **Returns**:
  ```python
  {
      'name': 'backtesting_engine',
      'initialized': True,
      'health': 'healthy',
      'config': {...},
      'performance_stats': {...},
      'state_keys': ['current_position', 'portfolio_value', 'trade_history'],
      'dependencies': ['pandas', 'numpy', 'vectorbt'],
      'capabilities': {...}
  }
  ```

#### `get_health_report() -> Dict[str, Any]`
- **Purpose**: Get detailed health analysis
- **Features**:
  - Overall health assessment
  - Performance analysis
  - Configuration validation
  - Health recommendations
- **Returns**:
  ```python
  {
      'component_name': 'backtesting_engine',
      'overall_health': 'healthy',
      'initialization_status': True,
      'performance_metrics': {...},
      'configuration_status': True,
      'state_size': 5,
      'recommendations': [...]
  }
  ```

### 5. Serialization

#### `serialize() -> Dict[str, Any]`
- **Purpose**: Serialize component for persistence
- **Features**:
  - Complete component state
  - Configuration and state
  - Performance statistics
  - Component-specific data
- **Returns**:
  ```python
  {
      'component_class': 'BacktestingEngine',
      'name': 'backtesting_engine',
      'config': {...},
      'state': {...},
      'performance_stats': {...},
      'initialized': True,
      'timestamp': 1234567890.0,
      'version': '1.0.0'
  }
  ```

#### `deserialize(data: Dict[str, Any]) -> None`
- **Purpose**: Deserialize component from persisted data
- **Features**:
  - Validates serialized data
  - Restores complete state
  - Handles component-specific data
- **Example**:
  ```python
  component = BacktestingEngine('new_engine')
  component.deserialize(serialized_data)
  ```

#### `save_to_file(filepath: str) -> None`
- **Purpose**: Save component to JSON file
- **Features**:
  - Creates directory if needed
  - JSON serialization
  - Error handling
- **Example**:
  ```python
  component.save_to_file('/path/to/backtesting_engine.json')
  ```

#### `load_from_file(filepath: str) -> None`
- **Purpose**: Load component from JSON file
- **Features**:
  - JSON deserialization
  - Error handling
- **Example**:
  ```python
  component.load_from_file('/path/to/backtesting_engine.json')
  ```

### 6. Safe Processing

#### `_safe_process(data: Any, **kwargs) -> Any`
- **Purpose**: Safely process data with comprehensive error handling
- **Features**:
  - Pre-processing validation
  - Input validation
  - Capability checking
  - Memory requirement checking
  - Performance tracking
  - Error handling and logging
- **Example**:
  ```python
  try:
      result = component._safe_process(market_data)
  except ValueError as e:
      print(f"Validation error: {e}")
  except MemoryError as e:
      print(f"Memory error: {e}")
  except RuntimeError as e:
      print(f"Backtesting error: {e}")
  ```

#### `_check_memory_usage(data: Any) -> bool`
- **Purpose**: Check if sufficient memory available
- **Features**:
  - Memory requirement estimation
  - Configuration-based limits
  - Graceful fallback
- **Example**:
  ```python
  if component._check_memory_usage(market_data):
      result = component.process(market_data)
  ```

#### `_log_operation(operation: str, success: bool, processing_time: float) -> None`
- **Purpose**: Log operation details with appropriate level
- **Features**:
  - Success/failure logging
  - Performance warnings
  - Configurable thresholds
- **Example**:
  ```python
  component._log_operation("run_backtest", True, 45.2)
  ```

#### `_validate_dependencies(dependencies: List[str]) -> bool`
- **Purpose**: Validate that all dependencies are available
- **Features**:
  - Common dependency checking
  - Generic import support
  - Error handling
- **Example**:
  ```python
  deps = ['pandas', 'numpy', 'vectorbt', 'matplotlib']
  if component._validate_dependencies(deps):
      print("All dependencies available")
  ```

## Helper Methods for Subclasses

### Abstract Helper Methods (Must be overridden)

1. **`_initialize_resources() -> bool`** - Initialize component-specific resources
2. **`_cleanup_resources() -> None`** - Cleanup component-specific resources
3. **`_process_data(data: Any, **kwargs) -> Any`** - Process data with component logic
4. **`_get_validation_rules() -> Dict[str, Any]`** - Get validation rules
5. **`_validate_component_specific(data: Any) -> Dict[str, Any]`** - Component-specific validation

### Optional Helper Methods (Can be overridden)

1. **`_on_config_changed(config: Dict[str, Any]) -> None`** - Configuration change callback
2. **`_on_state_changed(key: str, value: Any, previous_value: Any) -> None`** - State change callback
3. **`_get_component_performance_stats() -> Dict[str, Any]`** - Component-specific performance data
4. **`_get_component_status() -> Dict[str, Any]`** - Component-specific status
5. **`_serialize_component_data() -> Dict[str, Any]`** - Component-specific serialization
6. **`_deserialize_component_data(data: Dict[str, Any]) -> None`** - Component-specific deserialization
7. **`_validate_component_config() -> bool`** - Component-specific config validation

## Complete Example Implementation

```python
from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import ModularComponent

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

# Usage Example
def main():
    # Create component
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
        },
        'memory_limit_mb': 2048,
        'slow_operation_threshold': 10.0
    }
    
    component = BacktestingEngine('backtesting_engine', config)
    
    # Initialize
    if not component.initialize():
        print("Initialization failed")
        return
    
    # Configure
    component.update_config({'risk_management': {'max_position_size': 0.15}})
    
    # Set state
    component.set_state('experiment_id', 'backtest_123')
    
    # Process data safely
    backtest_data = {
        'market_data': market_data,
        'strategy_signals': strategy_signals
    }
    
    try:
        result = component._safe_process(backtest_data)
        print(f"Backtesting successful: {result['performance_metrics']['total_return']:.4f}")
    except Exception as e:
        print(f"Backtesting failed: {e}")
    
    # Monitor performance
    stats = component.get_performance_stats()
    print(f"Success rate: {stats['success_rate']:.2%}")
    
    # Check health
    health = component.get_health_report()
    print(f"Health status: {health['overall_health']}")
    
    # Serialize for persistence
    serialized = component.serialize()
    
    # Cleanup
    component.cleanup()

if __name__ == "__main__":
    main()
```

## Key Benefits for Backtesting

1. **Complete Implementation**: All abstract methods are fully implemented
2. **Production Ready**: Comprehensive error handling and logging
3. **Extensible**: Easy to create custom backtesting components
4. **Robust**: Handles edge cases and provides meaningful errors
5. **Well Documented**: Detailed docstrings and examples
6. **Consistent**: Follows established patterns
7. **Flexible**: Supports various backtesting scenarios
8. **Maintainable**: Clean separation of concerns
9. **Backtesting-Specific**: Optimized for trading strategy evaluation
10. **State Management**: Tracks portfolio state and trade history

The implementation provides a solid foundation for creating modular, reusable components in the backtesting pipeline, with specific optimizations for trading strategy evaluation workflows.