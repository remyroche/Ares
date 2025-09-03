# Walk-Forward Validation Implementation Guide

## Overview
Walk-forward validation prevents overfitting by training on historical data and testing on unseen future data in a rolling window fashion.

## Implementation Structure

### 1. Basic Walk-Forward Framework
```python
class WalkForwardValidator:
    def __init__(self, config):
        self.train_period_days = 365  # 1 year training
        self.test_period_days = 30    # 1 month testing
        self.step_days = 30           # Roll forward monthly
        self.min_train_samples = 1000
        
    async def run_walk_forward_validation(self, data, model_trainer):
        results = []
        
        # Calculate windows
        windows = self._generate_walk_forward_windows(data)
        
        for window in windows:
            # Train on in-sample data
            model = await model_trainer.train(
                data[window['train_start']:window['train_end']]
            )
            
            # Test on out-of-sample data
            test_results = await self._test_model(
                model,
                data[window['test_start']:window['test_end']]
            )
            
            results.append({
                'window': window,
                'model': model,
                'performance': test_results
            })
            
        return self._analyze_results(results)
```

### 2. Regime-Aware Walk-Forward
```python
class RegimeAwareWalkForward:
    def __init__(self):
        self.regime_validator = RegimeValidator()
        
    async def validate_with_regimes(self, data, regime_labels):
        results_by_regime = {}
        
        for regime in ['bull', 'bear', 'sideways']:
            # Filter data by regime
            regime_data = data[regime_labels == regime]
            
            # Ensure minimum data per regime
            if len(regime_data) < self.min_samples_per_regime:
                continue
                
            # Walk-forward for this regime
            regime_results = await self._walk_forward_for_regime(
                regime_data, 
                regime
            )
            
            results_by_regime[regime] = regime_results
            
        return self._combine_regime_results(results_by_regime)
```

### 3. Adaptive Window Sizing
```python
class AdaptiveWalkForward:
    def __init__(self):
        self.market_volatility_analyzer = MarketVolatilityAnalyzer()
        
    def calculate_optimal_window_size(self, data, current_date):
        # Adjust window based on market conditions
        volatility = self.market_volatility_analyzer.get_volatility(data, current_date)
        
        if volatility > 0.03:  # High volatility
            return {
                'train_days': 180,  # Shorter training window
                'test_days': 14     # Shorter test window
            }
        else:  # Normal volatility
            return {
                'train_days': 365,
                'test_days': 30
            }
```

### 4. Performance Tracking
```python
class WalkForwardPerformanceTracker:
    def __init__(self):
        self.metrics = ['sharpe', 'max_drawdown', 'win_rate', 'profit_factor']
        
    def track_performance(self, walk_forward_results):
        performance_over_time = []
        
        for result in walk_forward_results:
            metrics = {
                'period': result['window'],
                'in_sample_sharpe': result['train_performance']['sharpe'],
                'out_sample_sharpe': result['test_performance']['sharpe'],
                'degradation': self._calculate_degradation(result)
            }
            
            # Flag potential overfitting
            if metrics['degradation'] > 0.5:
                metrics['warning'] = 'potential_overfitting'
                
            performance_over_time.append(metrics)
            
        return self._generate_report(performance_over_time)
```

### 5. Integration with Training Pipeline
```python
# Modify step5_labeling.py to support walk-forward
class EnhancedLabelingStep:
    def __init__(self, config):
        self.walk_forward_enabled = config.get('walk_forward_validation', True)
        self.validator = WalkForwardValidator(config)
        
    async def execute_with_validation(self, data):
        if self.walk_forward_enabled:
            # Split data for walk-forward
            train_end_date = data.index[-1] - pd.Timedelta(days=30)
            
            # Train only on data up to train_end_date
            train_data = data[data.index <= train_end_date]
            test_data = data[data.index > train_end_date]
            
            # Generate labels for training
            train_labels = await self.generate_labels(train_data)
            
            # Validate on test data
            validation_results = await self.validator.validate(
                train_data, 
                train_labels, 
                test_data
            )
            
            return {
                'labels': train_labels,
                'validation': validation_results
            }
```

## Configuration Example
```yaml
walk_forward_config:
  enabled: true
  train_period_days: 365
  test_period_days: 30
  step_days: 30
  min_train_samples: 1000
  
  # Regime-specific settings
  regime_aware: true
  min_samples_per_regime: 500
  
  # Adaptive settings
  adaptive_windows: true
  volatility_threshold: 0.03
  
  # Performance thresholds
  max_acceptable_degradation: 0.3
  min_out_sample_sharpe: 0.5
```

## Key Benefits
1. **Prevents Overfitting**: Tests on truly unseen data
2. **Realistic Performance**: Simulates actual trading conditions
3. **Regime Robustness**: Validates each regime separately
4. **Early Warning**: Detects model degradation early

## Implementation Steps
1. Add walk-forward configuration to config files
2. Modify data splitting in steps 4-5
3. Add validation loops to training steps
4. Create performance tracking dashboard
5. Set up alerts for performance degradation