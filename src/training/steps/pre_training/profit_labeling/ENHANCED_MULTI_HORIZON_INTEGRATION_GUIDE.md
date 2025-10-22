# Enhanced Multi-Horizon Integration Guide

This guide explains how to use the enhanced multi-horizon profit labeler that integrates the enhanced data and labels system with the existing multi-horizon functionality.

## 🎯 Overview

The enhanced multi-horizon profit labeler provides:
- **Drop-in replacement** for existing `MultiHorizonProfitLabeler`
- **Enhanced data cleaning** and quality assessment
- **Trading-aware label definitions** (Analyst: "Should we trade?", Tactician: Direction/magnitude)
- **Label stability monitoring** and leakage detection
- **Full backward compatibility** with existing pipeline
- **No duplication** of existing functionality

## 🚀 Quick Start

### Basic Usage (Drop-in Replacement)

```python
from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabeler, EnhancedMultiHorizonConfig
)

# Create enhanced labeler (drop-in replacement)
config = EnhancedMultiHorizonConfig()
labeler = EnhancedMultiHorizonProfitLabeler(config)

# Use exactly like the original MultiHorizonProfitLabeler
result = await labeler.execute_labeling(
    symbol="ETHUSDT",
    exchange="binance", 
    timeframe="15m",
    data_dir="historical_data",
    regime_data=regime_data
)
```

### Enhanced Configuration

```python
from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import (
    create_trading_optimized_multi_horizon_config,
    create_research_optimized_multi_horizon_config
)

# Trading-optimized configuration
trading_config = create_trading_optimized_multi_horizon_config()
trading_labeler = EnhancedMultiHorizonProfitLabeler(trading_config)

# Research-optimized configuration  
research_config = create_research_optimized_multi_horizon_config()
research_labeler = EnhancedMultiHorizonProfitLabeler(research_config)
```

## 🔧 Configuration Options

### Enhanced Data & Labels Settings

```python
config = EnhancedMultiHorizonConfig(
    # Enhanced data cleaning
    enable_enhanced_data_cleaning=True,
    
    # Enhanced stability monitoring
    enable_enhanced_stability_monitoring=True,
    
    # Trading-aware labels
    enable_trading_aware_labels=True,
    
    # Label definitions
    analyst_horizon_minutes=60,      # Analyst: "Should we trade?"
    tactician_horizon_minutes=30,    # Tactician: Direction/magnitude
    
    # Regime conditioning
    enable_regime_conditioning=True,
    enable_risk_awareness=True,
    
    # Quality thresholds
    min_data_quality_score=0.7,
    min_label_stability_score=0.6
)
```

### Trading Objective Configuration

```python
from src.training.steps.pre_training.profit_labeling.enhanced_data_labels_system import (
    create_trading_optimized_config
)

# Custom enhanced configuration
enhanced_config = create_trading_optimized_config()
enhanced_config.trading_objective.primary_objective = "risk_adjusted_returns"
enhanced_config.trading_objective.max_drawdown_pct = 0.05
enhanced_config.trading_objective.target_sharpe_ratio = 1.5

config = EnhancedMultiHorizonConfig(
    enhanced_config=enhanced_config
)
```

## 📊 Enhanced Features

### 1. Trading-Aware Label Definitions

The enhanced system provides two types of labels:

#### Analyst Labels: "Should we trade?"
- **Purpose**: Binary decision on whether to enter a trade
- **Logic**: 1 if expected PnL > fees + slippage within horizon H, else 0
- **Features**:
  - Considers transaction costs (maker/taker fees, slippage)
  - Accounts for volatility and regime conditions
  - Includes risk management constraints
  - Provides confidence scores

#### Tactician Labels: Direction/Magnitude
- **Purpose**: Direction and strength of trade signal
- **Logic**: 1 if max_favorable_excursion(H) ≥ θ_up and max_adverse_excursion(H) ≤ θ_down
- **Features**:
  - Volatility-scaled thresholds (θ_up = k × σ_t)
  - Regime-specific adjustments
  - Magnitude scoring for signal strength
  - Risk-aware filtering

### 2. Comprehensive Data Cleaning

```python
# Automatic data cleaning pipeline
config = EnhancedMultiHorizonConfig(
    enable_enhanced_data_cleaning=True,
    # The system automatically:
    # - Removes outliers using multiple methods (IQR, Z-score, Isolation Forest)
    # - Handles missing values with advanced imputation
    # - Aligns timestamps across timeframes
    # - De-duplicates overlapping samples
    # - Validates OHLCV relationships
)
```

### 3. Label Stability Monitoring

```python
# Label stability monitoring
config = EnhancedMultiHorizonConfig(
    enable_enhanced_stability_monitoring=True,
    min_label_stability_score=0.6,
    # The system automatically:
    # - Detects label leakage (autocorrelation, mutual information)
    # - Monitors concept drift (KS test, Wasserstein distance)
    # - Checks OOS balance similarity
    # - Recomputes labels on data refresh
)
```

## 🔄 Integration with Existing Pipeline

### Component Integration

```python
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent

class EnhancedMultiHorizonProfitLabelerComponent(BasePreTrainingComponent):
    """Component wrapper for enhanced multi-horizon profit labeler."""
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        
        # Create enhanced configuration
        enhanced_config = EnhancedMultiHorizonConfig()
        if config and config.custom_params:
            for key, value in config.custom_params.items():
                if hasattr(enhanced_config, key):
                    setattr(enhanced_config, key, value)
        
        self.labeler = EnhancedMultiHorizonProfitLabeler(enhanced_config)
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """Execute enhanced multi-horizon labeling as a component."""
        try:
            # Extract parameters from pipeline state
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            exchange = pipeline_state.get('exchange', 'binance')
            timeframe = pipeline_state.get('timeframe', '15m')
            data_dir = pipeline_state.get('data_dir', 'historical_data')
            regime_data = pipeline_state.get('regime_data_splitting_result')
            
            # Execute enhanced labeling
            result = await self.labeler.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                regime_data=regime_data
            )
            
            return ComponentResult(
                success=True,
                artifacts=result,
                metadata={
                    'component_type': 'enhanced_multi_horizon_profit_labeler',
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'enhanced_processing': True
                }
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                error_message=str(e),
                metadata={'component_type': 'enhanced_multi_horizon_profit_labeler'}
            )
```

### Pipeline Integration

```python
# In your training pipeline
from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabelerComponent
)

# Add to pipeline
pipeline.add_component(
    EnhancedMultiHorizonProfitLabelerComponent(),
    name="enhanced_multi_horizon_labeling"
)
```

## 📈 Output Structure

### Enhanced Artifacts

The enhanced labeler produces the same artifact structure as the original, plus additional enhanced metadata:

```python
{
    'multi_horizon_labeling_result': {
        'labeled_data': pd.DataFrame,           # Enhanced labels
        'labels': pd.DataFrame,                 # Backward compatibility
        'confidence_scores': pd.DataFrame,      # Enhanced confidence scores
        'eligibility_masks': pd.DataFrame,      # Enhanced eligibility masks
        'quality_scores': Dict,                 # Enhanced quality scores
        'sample_weights': pd.Series,            # Enhanced sample weights
        'method': 'enhanced_multi_horizon_profit_labeling',
        'enhanced_processing': True,
        'metadata': {
            'symbol': str,
            'exchange': str,
            'timeframe': str,
            'regime_aware': bool,
            'processing_time': float,
            'n_samples': int,
            'n_targets': int,
            'n_horizons': int,
            'target_distribution': Dict,
            'enhanced_metadata': {
                'data_quality': Dict,
                'label_stability': Dict,
                'final_quality': Dict,
                'regime_processing': Dict,
                'balancing_applied': bool
            }
        }
    },
    'labeling_report': {
        'status': 'completed',
        'enhanced_processing': True,
        'enhanced_metrics': {
            'data_quality': Dict,
            'label_stability': Dict,
            'final_quality': Dict,
            'regime_processing': Dict,
            'balancing_applied': bool
        },
        'recommendations': List[str]
    },
    'enhanced_artifacts': {
        'enhanced_labels': pd.DataFrame,
        'enhanced_confidence': pd.DataFrame,
        'enhanced_weights': pd.Series,
        'data_quality_metrics': Dict,
        'label_stability_metrics': Dict,
        'final_quality_metrics': Dict,
        'recommendations': List[str]
    }
}
```

## 🧪 Validation and Testing

### Comprehensive Validation

```python
from src.training.steps.pre_training.profit_labeling.enhanced_labels_validation import (
    run_enhanced_labels_validation
)

# Run comprehensive validation
validation_result = run_enhanced_labels_validation()

# Check validation status
if validation_result['overall_status'] == 'excellent':
    print("Enhanced system is ready for production!")
else:
    print("System needs attention:", validation_result['recommendations'])
```

### Integration Testing

```python
# Test enhanced integration
from src.training.steps.pre_training.profit_labeling.infrastructure_integration import (
    validate_system_integration
)

integration_status = validate_system_integration()
if integration_status['integration_working']:
    print("Enhanced system is properly integrated!")
else:
    print("Integration issues:", integration_status.get('error'))
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Enhanced Processing Fails
```python
# The system automatically falls back to standard processing
# Check logs for specific error messages
tprint_warning("⚠️ Enhanced processing failed, falling back to standard processing")
```

#### 2. Data Quality Issues
```python
# Adjust data quality thresholds
config = EnhancedMultiHorizonConfig(
    min_data_quality_score=0.6,  # Lower threshold
    enable_enhanced_data_cleaning=True
)
```

#### 3. Label Stability Issues
```python
# Adjust stability thresholds
config = EnhancedMultiHorizonConfig(
    min_label_stability_score=0.5,  # Lower threshold
    enable_enhanced_stability_monitoring=True
)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('EnhancedMultiHorizonProfitLabeler').setLevel(logging.DEBUG)

# Run with detailed output
result = await labeler.execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
```

## 📚 Examples

### Complete Example

```python
import asyncio
from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import (
    EnhancedMultiHorizonProfitLabeler, create_trading_optimized_multi_horizon_config
)

async def main():
    # Create enhanced labeler
    config = create_trading_optimized_multi_horizon_config()
    labeler = EnhancedMultiHorizonProfitLabeler(config)
    
    # Execute enhanced labeling
    result = await labeler.execute_labeling(
        symbol="ETHUSDT",
        exchange="binance",
        timeframe="15m",
        data_dir="historical_data",
        regime_data=regime_data
    )
    
    # Access enhanced results
    enhanced_labels = result['multi_horizon_labeling_result']['labeled_data']
    data_quality = result['enhanced_artifacts']['data_quality_metrics']
    label_stability = result['enhanced_artifacts']['label_stability_metrics']
    recommendations = result['enhanced_artifacts']['recommendations']
    
    print(f"Enhanced labels shape: {enhanced_labels.shape}")
    print(f"Data quality: {data_quality.get('quality_level', 'unknown')}")
    print(f"Label stability: {label_stability.get('stability_level', 'unknown')}")
    print(f"Recommendations: {recommendations}")

# Run example
asyncio.run(main())
```

## 🎯 Best Practices

### 1. Configuration Selection
- Use `create_trading_optimized_multi_horizon_config()` for production trading
- Use `create_research_optimized_multi_horizon_config()` for research and experimentation

### 2. Quality Monitoring
- Monitor data quality scores over time
- Track label stability metrics
- Review recommendations regularly

### 3. Performance Optimization
- Enable caching for repeated runs
- Use appropriate quality thresholds
- Monitor processing time and memory usage

### 4. Integration Testing
- Test with different market conditions
- Validate regime-aware processing
- Verify backward compatibility

## 🔄 Migration Guide

### From Original Multi-Horizon Labeler

1. **Import Change**:
   ```python
   # Old
   from src.training.steps.pre_training.multi_horizon_profit_labeler import MultiHorizonProfitLabeler
   
   # New (drop-in replacement)
   from src.training.steps.pre_training.profit_labeling.enhanced_multi_horizon_labeler import EnhancedMultiHorizonProfitLabeler as MultiHorizonProfitLabeler
   ```

2. **Configuration Update**:
   ```python
   # Old
   config = MultiHorizonConfig()
   
   # New (enhanced)
   config = EnhancedMultiHorizonConfig()
   # All original parameters still work
   ```

3. **Usage**:
   ```python
   # Same API, enhanced functionality
   labeler = MultiHorizonProfitLabeler(config)
   result = await labeler.execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
   ```

### Gradual Migration

1. **Phase 1**: Replace imports, test with existing configuration
2. **Phase 2**: Enable enhanced features gradually
3. **Phase 3**: Optimize configuration for your use case
4. **Phase 4**: Monitor and tune based on results

---

**Note**: The enhanced multi-horizon profit labeler is a drop-in replacement that provides significant improvements in data quality, label stability, and trading relevance while maintaining full backward compatibility with existing infrastructure.