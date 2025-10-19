# Step Reference Guide

## Overview

This document provides a comprehensive reference for all available steps in the Ares trading system, organized by stage and functionality.

## Step Execution

### Command Line Interface

```bash
# Run a single step
python ares_launcher.py step <step_name> --symbol <SYMBOL> --timeframe <TIMEFRAME> --direction <DIRECTION>

# Run multiple steps
python ares_launcher.py steps <step1,step2,step3> --symbol <SYMBOL> --timeframe <TIMEFRAME>

# Run an entire stage
python ares_launcher.py stage <STAGE_NAME> --symbol <SYMBOL> --timeframe <TIMEFRAME>
```

### Configuration Parameters

All steps accept the following standard configuration parameters:

- `symbol`: Trading symbol (e.g., 'ETHUSDT', 'BTCUSDT')
- `exchange`: Exchange name (e.g., 'binance', 'coinbase')
- `timeframe`: Timeframe (e.g., '1m', '15m', '1h', '1d')
- `direction`: Trading direction ('longs' or 'shorts')
- `execution_mode`: Execution mode ('light' or 'full')

## DATA_COLLECTION Stage

### data_download
**Purpose**: Download raw data from exchanges

**Description**: Downloads historical market data from specified exchanges and processes it into standardized format.

**Input**: Exchange configuration, symbol, timeframe
**Output**: Raw market data (OHLCV)

**Example**:
```bash
python ares_launcher.py step data_download --symbol ETHUSDT --timeframe 15m --exchange binance
```

**Artifacts Generated**:
- `raw_data.parquet`: Raw OHLCV data
- `raw_data.csv`: CSV version (if < 2000 rows)

### data_conversion
**Purpose**: Convert data formats and standardize

**Description**: Converts downloaded data to standardized format and applies initial preprocessing.

**Input**: Raw market data
**Output**: Standardized market data

**Example**:
```bash
python ares_launcher.py step data_conversion --symbol ETHUSDT --timeframe 15m
```

**Artifacts Generated**:
- `standardized_data.parquet`: Standardized market data
- `conversion_metadata.json`: Conversion parameters and statistics

### data_validation
**Purpose**: Validate data quality and integrity

**Description**: Performs comprehensive data quality checks including missing data detection, outlier identification, and consistency validation.

**Input**: Standardized market data
**Output**: Validated data with quality metrics

**Example**:
```bash
python ares_launcher.py step data_validation --symbol ETHUSDT --timeframe 15m
```

**Artifacts Generated**:
- `validated_data.parquet`: Quality-checked market data
- `quality_report.json`: Data quality metrics and issues

## MARKET_ANALYSIS Stage

### sr_detection
**Purpose**: Detect Support/Resistance levels

**Description**: Identifies significant support and resistance levels using advanced algorithms and market structure analysis.

**Input**: Market data, detection parameters
**Output**: Detected S/R levels with strength metrics

**Example**:
```bash
python ares_launcher.py step sr_detection --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `sr_levels.parquet`: Detected S/R levels
- `sr_levels.csv`: CSV version (if < 2000 rows)
- `sr_metrics.json`: Detection statistics and parameters

### sr_clustering
**Purpose**: Generate SR clusters

**Description**: Groups nearby support and resistance levels into clusters for better analysis and trading decisions.

**Input**: Detected S/R levels
**Output**: Clustered S/R levels with cluster metrics

**Example**:
```bash
python ares_launcher.py step sr_clustering --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `sr_clusters.parquet`: Clustered S/R levels
- `sr_clusters.csv`: CSV version (if < 2000 rows)
- `cluster_metrics.json`: Clustering statistics

### sr_parameter_optimization
**Purpose**: Optimize SR detection parameters

**Description**: Optimizes parameters for support and resistance detection algorithms using historical data analysis.

**Input**: Market data, parameter ranges
**Output**: Optimized detection parameters

**Example**:
```bash
python ares_launcher.py step sr_parameter_optimization --symbol ETHUSDT --timeframe 15m
```

**Artifacts Generated**:
- `optimized_parameters.json`: Optimized detection parameters
- `optimization_results.json`: Optimization metrics and history

## PRE_TRAINING Stage

### feature_generation_data_validation_step
**Purpose**: Enhanced data validation for feature generation

**Description**: Performs comprehensive data validation specifically for feature generation pipeline, including data alignment and quality assessment.

**Input**: Market data, feature requirements
**Output**: Validated data ready for feature generation

**Example**:
```bash
python ares_launcher.py step feature_generation_data_validation_step --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `validated_features_data.parquet`: Validated data for feature generation
- `validation_report.json`: Validation metrics and quality assessment

### feature_generation_period_lookback_optimization_step
**Purpose**: Optimize period and lookback parameters

**Description**: Optimizes time periods and lookback windows for feature generation to maximize predictive power.

**Input**: Market data, optimization parameters
**Output**: Optimized period and lookback settings

**Example**:
```bash
python ares_launcher.py step feature_generation_period_lookback_optimization_step --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `optimized_periods.parquet`: Optimized period settings
- `optimized_periods.csv`: CSV version (if < 2000 rows)
- `optimization_metrics.json`: Optimization results and performance metrics

### feature_generation_feature_generation_step
**Purpose**: Generate trading features

**Description**: Creates comprehensive feature sets for machine learning models including technical indicators, price patterns, and market microstructure features.

**Input**: Market data, feature specifications
**Output**: Generated feature dataset

**Example**:
```bash
python ares_launcher.py step feature_generation_feature_generation_step --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `generated_features.parquet`: Feature dataset
- `feature_metadata.json`: Feature descriptions and statistics

### feature_generation_feature_selection_step
**Purpose**: Select optimal features

**Description**: Identifies and selects the most predictive features using statistical and machine learning methods.

**Input**: Generated features, selection criteria
**Output**: Selected feature set

**Example**:
```bash
python ares_launcher.py step feature_generation_feature_selection_step --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `selected_features.parquet`: Selected feature dataset
- `feature_importance.json`: Feature importance scores and rankings

## MODEL_TRAINING Stage

### analyst_models_training
**Purpose**: Train Analyst models

**Description**: Trains individual Analyst models for each market regime using advanced machine learning algorithms and hyperparameter optimization.

**Input**: Features, labels, regime information
**Output**: Trained Analyst models

**Example**:
```bash
python ares_launcher.py step analyst_models_training --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `analyst_models.pkl`: Trained Analyst models
- `training_metrics.json`: Training performance metrics
- `model_metadata.json`: Model specifications and parameters

### tactician_models_training
**Purpose**: Train Tactician models

**Description**: Trains Tactician models for entry timing optimization using 1-minute timeframe data and Analyst model outputs.

**Input**: 1m market data, Analyst predictions, labels
**Output**: Trained Tactician models

**Example**:
```bash
python ares_launcher.py step tactician_models_training --symbol ETHUSDT --timeframe 1m --direction longs
```

**Artifacts Generated**:
- `tactician_models.pkl`: Trained Tactician models
- `tactician_metrics.json`: Training performance metrics
- `entry_timing_analysis.json`: Entry timing optimization results

### analyst_ensemble_training
**Purpose**: Train Analyst ensemble models

**Description**: Creates ensemble models combining multiple Analyst models for improved prediction accuracy and robustness.

**Input**: Individual Analyst models, ensemble methods
**Output**: Trained ensemble models

**Example**:
```bash
python ares_launcher.py step analyst_ensemble_training --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `analyst_ensemble.pkl`: Trained ensemble models
- `ensemble_metrics.json`: Ensemble performance metrics
- `ensemble_weights.json`: Model weights and combination parameters

### tactician_ensemble_training
**Purpose**: Train Tactician ensemble models

**Description**: Creates ensemble models combining multiple Tactician models for optimal entry timing decisions.

**Input**: Individual Tactician models, ensemble methods
**Output**: Trained ensemble models

**Example**:
```bash
python ares_launcher.py step tactician_ensemble_training --symbol ETHUSDT --timeframe 1m --direction longs
```

**Artifacts Generated**:
- `tactician_ensemble.pkl`: Trained ensemble models
- `ensemble_metrics.json`: Ensemble performance metrics
- `timing_optimization.json`: Entry timing optimization results

## BACKTESTING Stage

### final_parameters_optimization
**Purpose**: Optimize final system parameters

**Description**: Optimizes final system parameters after model training using advanced optimization algorithms and cross-validation.

**Input**: Trained models, optimization parameters
**Output**: Optimized system parameters

**Example**:
```bash
python ares_launcher.py step final_parameters_optimization --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `optimized_parameters.parquet`: Optimized system parameters
- `optimized_parameters.csv`: CSV version (if < 2000 rows)
- `optimization_report.json`: Optimization results and performance metrics

### real_parameters_optimization
**Purpose**: Optimize real trading parameters

**Description**: Optimizes parameters for real trading execution including position sizing, risk management, and execution parameters.

**Input**: Trading data, risk parameters
**Output**: Optimized trading parameters

**Example**:
```bash
python ares_launcher.py step real_parameters_optimization --symbol ETHUSDT --timeframe 15m --direction longs
```

**Artifacts Generated**:
- `trading_parameters.parquet`: Optimized trading parameters
- `trading_parameters.csv`: CSV version (if < 2000 rows)
- `risk_metrics.json`: Risk analysis and parameter validation

## Execution Modes

### Light Mode
**Purpose**: Quick execution for testing and development
**Features**:
- Reduced dataset sizes
- Simplified algorithms
- Faster execution
- Lower resource usage

**Usage**:
```bash
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --direction longs --execution-mode light
```

### Full Mode
**Purpose**: Production-ready execution with full capabilities
**Features**:
- Complete dataset processing
- Full algorithm complexity
- Comprehensive analysis
- Maximum accuracy

**Usage**:
```bash
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --direction longs --execution-mode full
```

## Step Dependencies

### Data Flow
```
DATA_COLLECTION → MARKET_ANALYSIS → PRE_TRAINING → MODEL_TRAINING → BACKTESTING
```

### Typical Execution Sequences

#### Complete Pipeline
```bash
# Run entire pipeline
python ares_launcher.py stage DATA_COLLECTION --symbol ETHUSDT --timeframe 15m
python ares_launcher.py stage MARKET_ANALYSIS --symbol ETHUSDT --timeframe 15m
python ares_launcher.py stage PRE_TRAINING --symbol ETHUSDT --timeframe 15m
python ares_launcher.py stage MODEL_TRAINING --symbol ETHUSDT --timeframe 15m
python ares_launcher.py stage BACKTESTING --symbol ETHUSDT --timeframe 15m
```

#### Quick Testing
```bash
# Run key steps for testing
python ares_launcher.py steps data_download,sr_detection,analyst_models_training --symbol ETHUSDT --timeframe 15m --direction longs --execution-mode light
```

#### Model Training Only
```bash
# Train models with existing data
python ares_launcher.py steps analyst_models_training,tactician_models_training --symbol ETHUSDT --timeframe 15m --direction longs
```

## Performance Considerations

### Resource Requirements
- **Memory**: Steps require 2-8GB RAM depending on dataset size
- **Storage**: 1-10GB per symbol/timeframe combination
- **CPU**: Multi-core processing for optimization steps
- **GPU**: Optional acceleration for ML training steps

### Execution Times
- **Data Collection**: 1-5 minutes per symbol
- **Market Analysis**: 2-10 minutes per symbol
- **Pre-Training**: 5-30 minutes per symbol
- **Model Training**: 10-60 minutes per symbol
- **Backtesting**: 5-20 minutes per symbol

### Optimization Tips
- Use light mode for development and testing
- Run steps in parallel for multiple symbols
- Clean up old artifacts regularly
- Use appropriate timeframes for your use case

## Troubleshooting

### Common Issues

#### Step Not Found
```bash
# Check available steps
python ares_launcher.py list-steps

# Verify step registration
python ares_launcher.py step-info <step_name>
```

#### Insufficient Data
```bash
# Check data availability
python ares_launcher.py check-data --symbol ETHUSDT --timeframe 15m

# Run data collection first
python ares_launcher.py step data_download --symbol ETHUSDT --timeframe 15m
```

#### Memory Issues
```bash
# Use light mode for large datasets
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --execution-mode light

# Check system resources
python ares_launcher.py system-info
```

### Debug Mode
```bash
# Enable debug logging
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --debug

# Verbose output
python ares_launcher.py step <step_name> --symbol ETHUSDT --timeframe 15m --verbose
```

This reference guide provides comprehensive information about all available steps in the Ares trading system and how to use them effectively.
