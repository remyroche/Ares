# Multi-Timeframe Training Guide

## Overview

Multi-timeframe training is a powerful approach that trains models across different timeframes simultaneously, creating ensemble models that can capture patterns at various time scales. This approach provides:

- **Diversification**: Models trained on different timeframes capture different market patterns
- **Ensemble Benefits**: Combined predictions from multiple timeframes
- **Cross-Validation**: Validation across timeframes ensures robustness
- **Parallel Processing**: Train multiple timeframes simultaneously

## Key Concepts

### **Timeframe Hierarchy**

Different timeframes capture different market dynamics:

- **1h**: Short-term patterns, intraday movements
- **4h**: Medium-term trends, swing trading patterns  
- **1d**: Long-term trends, position trading patterns

### **Ensemble Methods**

The system supports various ensemble combination methods:

- **Weighted Average**: Combine predictions with timeframe-specific weights
- **Voting**: Majority vote across timeframes
- **Stacking**: Meta-learner combines timeframe predictions

### **Cross-Timeframe Validation**

Validates model consistency across timeframes:

- **Correlation Analysis**: Check if timeframes are complementary
- **Consistency Scoring**: Measure prediction consistency
- **Diversification Benefits**: Quantify ensemble advantages

## Usage Examples

### **Basic Multi-Timeframe Training**

```bash
# Train on default timeframes (1h, 4h, 1d)
python ares_launcher.py multi-timeframe --symbol ETHUSDT --exchange BINANCE

# Train on specific timeframes
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d

# Quick test with limited data
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --quick-test
```

### **Advanced Options**

```bash
# Sequential training (no parallel)
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d --sequential

# Disable ensemble creation
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --timeframes 1h,4h,1d --no-ensemble

# Ensemble only (assumes models already trained)
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --ensemble-only --timeframes 1h,4h,1d

# Analyze timeframe correlations
python scripts/run_multi_timeframe_training.py --symbol ETHUSDT --analyze --timeframes 1h,4h,1d

# List all available timeframes and their purposes
python scripts/run_multi_timeframe_training.py --list-timeframes

# View current timeframe configuration
python scripts/show_timeframe_config.py

# Get details about a specific timeframe
python scripts/show_timeframe_config.py 1h
```

## Configuration

### **Multi-Timeframe Configuration**

```python
CONFIG["MULTI_TIMEFRAME_TRAINING"] = {
    "enable_parallel_training": True,      # Train timeframes in parallel
    "enable_ensemble": True,               # Create ensemble models
    "enable_cross_validation": True,       # Perform cross-validation
    "ensemble_method": "weighted_average", # Ensemble combination method
    "validation_split": 0.2,              # Validation data split
    "max_parallel_workers": 3,            # Maximum parallel workers
}
```

### **Global Data Configuration**

Centralized configuration for data settings:

```python
CONFIG["DATA_CONFIG"] = {
    "default_lookback_days": 730,  # Default lookback period for all timeframes (2 years)
}
```

### **Timeframe Definitions and Purposes**

Each timeframe has a clear purpose and configuration (lookback_days is centralized):

```python
CONFIG["TIMEFRAMES"] = {
    "1m": {
        "purpose": "Ultra-short-term scalping and high-frequency trading",
        "trading_style": "scalping",
        "feature_set": "ultra_short_term",
        "optimization_trials": 20, # Fewer trials for speed
        "description": "Captures micro-movements and immediate market reactions"
    },
    "1h": {
        "purpose": "Swing trading and medium-term trend identification",
        "trading_style": "swing_trading",
        "feature_set": "swing",
        "optimization_trials": 40,
        "description": "Primary timeframe for swing trading, captures daily cycles"
    },
    "4h": {
        "purpose": "Medium-term trend analysis and position trading",
        "trading_style": "position_trading",
        "feature_set": "medium_term",
        "optimization_trials": 50,
        "description": "Excellent for trend identification and reducing noise"
    },
    "1d": {
        "purpose": "Long-term trend analysis and position trading",
        "trading_style": "position_trading",
        "feature_set": "long_term",
        "optimization_trials": 50,
        "description": "Primary timeframe for long-term trend identification"
    }
}
```

### **Predefined Timeframe Sets**

Ready-to-use timeframe combinations for different trading styles:

```python
CONFIG["TIMEFRAME_SETS"] = {
    "scalping": {
        "timeframes": ["1m", "5m", "15m"],
        "description": "Ultra-short-term trading with high frequency",
        "use_case": "High-frequency trading and scalping strategies"
    },
    "intraday": {
        "timeframes": ["5m", "15m", "1h"],
        "description": "Intraday trading with multiple confirmation levels",
        "use_case": "Day trading and intraday swing trading"
    },
    "swing": {
        "timeframes": ["1h", "4h", "1d"],
        "description": "Swing trading with trend confirmation",
        "use_case": "Swing trading and medium-term position trading"
    },
    "position": {
        "timeframes": ["4h", "1d", "3d"],
        "description": "Position trading with long-term trend analysis",
        "use_case": "Position trading and long-term trend following"
    },
    "investment": {
        "timeframes": ["1d", "3d", "1w"],
        "description": "Long-term investment and major trend analysis",
        "use_case": "Long-term investment and major market cycle analysis"
    }
}

CONFIG["DEFAULT_TIMEFRAME_SET"] = "swing"  # Default set to use
```

## Training Process

### **Step 1: Individual Timeframe Training**

For each timeframe:

1. **Data Collection**: Download historical data for the timeframe
2. **Feature Engineering**: Generate timeframe-specific features
3. **Model Training**: Train individual models for each timeframe
4. **Validation**: Validate each timeframe model independently

### **Step 2: Ensemble Creation**

1. **Model Combination**: Combine predictions from all timeframes
2. **Weight Optimization**: Optimize ensemble weights
3. **Ensemble Validation**: Validate ensemble performance
4. **Cross-Correlation**: Analyze timeframe correlations

### **Step 3: Cross-Timeframe Validation**

1. **Consistency Check**: Verify predictions are consistent across timeframes
2. **Diversification Analysis**: Measure ensemble diversification benefits
3. **Performance Comparison**: Compare ensemble vs individual models
4. **Recommendation Generation**: Generate trading recommendations

## Output and Reports

### **Training Results**

```json
{
    "symbol": "ETHUSDT",
    "timestamp": "2024-01-15T10:30:00",
    "summary": {
        "total_timeframes": 3,
        "successful_timeframes": 3,
        "success_rate": 1.0,
        "ensemble_created": true,
        "validation_performed": true
    },
    "timeframe_results": {
        "1h": {
            "status": "success",
            "session_id": "ETHUSDT_1h_20240115_103000",
            "efficiency_stats": {...}
        },
        "4h": {
            "status": "success", 
            "session_id": "ETHUSDT_4h_20240115_103000",
            "efficiency_stats": {...}
        },
        "1d": {
            "status": "success",
            "session_id": "ETHUSDT_1d_20240115_103000", 
            "efficiency_stats": {...}
        }
    },
    "ensemble_results": {
        "status": "success",
        "ensemble_model": {...},
        "validation": {...},
        "timeframes_used": ["1h", "4h", "1d"]
    },
    "validation_results": {
        "status": "success",
        "validation_results": {
            "timeframe_correlations": {...},
            "consistency_score": 0.82,
            "diversification_benefit": 0.15,
            "optimal_weights": {...}
        }
    },
    "recommendations": [
        "✅ All timeframes trained successfully",
        "✅ Ensemble model created successfully",
        "✅ Cross-timeframe validation completed"
    ]
}
```

### **Performance Metrics**

- **Individual Timeframe Performance**: Accuracy, Sharpe ratio, max drawdown per timeframe
- **Ensemble Performance**: Combined metrics across timeframes
- **Correlation Analysis**: Timeframe correlation matrix
- **Diversification Benefits**: Risk reduction through ensemble
- **Consistency Score**: Prediction consistency across timeframes

## Best Practices

### **Timeframe Selection**

1. **Complementary Timeframes**: Choose timeframes that capture different patterns
2. **Market Conditions**: Adapt timeframes to market conditions
3. **Trading Style**: Align timeframes with trading strategy

### **Ensemble Configuration**

1. **Weight Optimization**: Use cross-validation to optimize weights
2. **Diversification**: Ensure timeframes provide diversification benefits
3. **Consistency**: Verify predictions are consistent across timeframes

### **Performance Monitoring**

1. **Individual Tracking**: Monitor each timeframe's performance
2. **Ensemble Tracking**: Track ensemble vs individual performance
3. **Correlation Monitoring**: Watch for changing timeframe correlations

## Advanced Features

### **Dynamic Timeframe Selection**

```python
# Automatically select optimal timeframes based on market conditions
CONFIG["MULTI_TIMEFRAME_TRAINING"]["dynamic_selection"] = True
CONFIG["MULTI_TIMEFRAME_TRAINING"]["selection_criteria"] = {
    "volatility_threshold": 0.3,
    "correlation_threshold": 0.7,
    "performance_threshold": 0.6
}
```

### **Adaptive Ensemble Weights**

```python
# Dynamically adjust ensemble weights based on recent performance
CONFIG["MULTI_TIMEFRAME_TRAINING"]["adaptive_weights"] = True
CONFIG["MULTI_TIMEFRAME_TRAINING"]["adaptation_window"] = 30  # days
```

### **Cross-Asset Multi-Timeframe**

```python
# Train multi-timeframe models across multiple assets
CONFIG["MULTI_TIMEFRAME_TRAINING"]["cross_asset"] = True
CONFIG["MULTI_TIMEFRAME_TRAINING"]["assets"] = ["ETHUSDT", "BTCUSDT", "ADAUSDT"]
```

## Troubleshooting

### **Common Issues**

1. **Memory Issues**: Reduce parallel workers or use sequential training
2. **Timeframe Failures**: Check data availability for failed timeframes
3. **Ensemble Creation**: Ensure at least 2 timeframes train successfully
4. **Correlation Issues**: High correlations may reduce ensemble benefits

### **Performance Optimization**

1. **Parallel Processing**: Use parallel training for faster execution
2. **Memory Management**: Monitor memory usage across timeframes
3. **Caching**: Enable feature caching for repeated training
4. **Checkpointing**: Use checkpoints for long training runs

## Future Enhancements

### **Planned Features**

1. **Real-time Ensemble Updates**: Update ensemble weights in real-time
2. **Market Regime Detection**: Adapt timeframes to market regimes
3. **Advanced Ensemble Methods**: Support for more sophisticated ensemble techniques
4. **Cross-Asset Ensembles**: Ensemble models across multiple assets

### **Research Directions**

1. **Optimal Timeframe Selection**: Research optimal timeframe combinations
2. **Dynamic Weighting**: Develop adaptive ensemble weighting schemes
3. **Regime-Specific Ensembles**: Create regime-specific ensemble models
4. **Hierarchical Ensembles**: Multi-level ensemble architectures

## Conclusion

Multi-timeframe training provides a robust approach to trading model development by leveraging the complementary nature of different timeframes. The ensemble approach reduces risk while potentially improving performance through diversification.

Key benefits:
- **Diversification**: Reduces model risk through timeframe diversification
- **Robustness**: Cross-timeframe validation ensures model consistency
- **Performance**: Ensemble models often outperform individual timeframes
- **Flexibility**: Adaptable to different market conditions and trading styles

The system is designed to be efficient, scalable, and maintainable while providing comprehensive analysis and reporting capabilities. 