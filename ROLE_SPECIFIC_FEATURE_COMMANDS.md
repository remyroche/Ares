# Role-Specific Feature Engineering Commands

## Overview

You can now call feature engineering scripts specifically for **Analyst** or **Tactician** models, mimicking calls from their respective orchestrators. This ensures the correct timeframe and role-specific configurations are applied.

## 🎯 New Role-Specific Commands

### Analyst-Specific Commands

#### 1. Analyst Feature Lookback Optimization
```bash
# Full execution (1460 days, 100% intensity)
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Light execution (10 days, 5% intensity) - recommended for testing
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

#### 2. Analyst Interactive Feature Generation
```bash
# Full execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Light execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

#### 3. Analyst Final Feature Selection
```bash
# Full execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Light execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_final_feature_selection \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

### Tactician-Specific Commands

#### 1. Tactician Feature Lookback Optimization
```bash
# Full execution (1460 days, 100% intensity)
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Light execution (10 days, 5% intensity) - recommended for testing
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 15m
```

#### 2. Tactician Interactive Feature Generation
```bash
# Full execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Light execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode light --symbol ETHUSDT --timeframe 15m
```

#### 3. Tactician Final Feature Selection
```bash
# Full execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Light execution
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_final_feature_selection \
    --execution-mode light --symbol ETHUSDT --timeframe 15m
```

## 📊 What Each Command Does

### Feature Lookback Optimization
- **Purpose**: Optimize feature lookback periods per regime/cluster
- **Analyst**: 60m timeframe, strategic optimization
- **Tactician**: 15m timeframe, tactical optimization
- **Output**: Optimized lookback periods for each feature

### Interactive Feature Generation
- **Purpose**: Generate interaction, polynomial, and cross-timeframe features
- **Analyst**: 60m base, includes 240m (4h) and 1440m (24h) cross-timeframe
- **Tactician**: 15m base, includes 60m (1h) and 240m (4h) cross-timeframe
- **Output**: Comprehensive feature set with interactions

### Final Feature Selection
- **Purpose**: Multi-stage feature selection (120→100→80→60)
- **Analyst**: Selects features most relevant for strategic decisions
- **Tactician**: Selects features most relevant for entry timing
- **Output**: Final 60 features optimized for the specific role

## 🔗 Complete Analyst Pipeline

Run the full Analyst feature engineering pipeline:

```bash
# 1. Analyst Profit Labeling
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

# 2. Analyst Feature Lookback Optimization
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# 3. Analyst Interactive Feature Generation
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# 4. Analyst Final Feature Selection
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

Or use the orchestrator to run all steps:
```bash
python ares_launcher.py --analyst-pre-ml --symbol ETHUSDT
```

## 🔗 Complete Tactician Pipeline

Run the full Tactician feature engineering pipeline:

```bash
# 1. Tactician Entry Labeling
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m

# 2. Tactician Feature Lookback Optimization
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# 3. Tactician Interactive Feature Generation
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# 4. Tactician Final Feature Selection
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

Or use the orchestrator to run all steps:
```bash
python ares_launcher.py --tactician-pre-ml --symbol ETHUSDT
```

## 🎭 Generic vs. Role-Specific Commands

### Generic Commands (Still Available)
```bash
# These work but don't enforce role-specific behavior
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline final_feature_selection \
    --execution-mode full --symbol ETHUSDT
```

### Role-Specific Commands (NEW - Recommended)
```bash
# These enforce role-specific timeframes and configurations
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

## 🔧 How It Works

The role-specific commands are **aliases** that:
1. Point to the same underlying component
2. Can extract role information from the command name
3. Apply role-specific defaults (timeframe, optimization settings)
4. Mimic the behavior of the orchestrators

Internally, these commands:
- Parse the role from the name (`analyst_*` or `tactician_*`)
- Set appropriate timeframe defaults (60m for Analyst, 15m for Tactician)
- Configure per-regime/cluster optimization settings
- Pass role context to the components

## 💡 Quick Reference Table

| Role | Labeler | Lookback Opt | Feature Gen | Selection | Timeframe |
|------|---------|--------------|-------------|-----------|-----------|
| **Analyst** | `--analyst-labeler` | `analyst_feature_lookback_optimization` | `analyst_interactive_feature_generation` | `analyst_final_feature_selection` | 60m |
| **Tactician** | `--tactician-labeler` | `tactician_feature_lookback_optimization` | `tactician_interactive_feature_generation` | `tactician_final_feature_selection` | 15m |

## 🚀 Recommended Workflow

### For Testing (Light Mode)
```bash
# Analyst
python ares_launcher.py --analyst-labeler --execution-mode light --symbol ETHUSDT --timeframe 60m
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_feature_lookback_optimization --execution-mode light --symbol ETHUSDT --timeframe 60m

# Tactician
python ares_launcher.py --tactician-labeler --execution-mode light --symbol ETHUSDT --timeframe 15m
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_feature_lookback_optimization --execution-mode light --symbol ETHUSDT --timeframe 15m
```

### For Production (Full Mode)
```bash
# Analyst full pipeline
python ares_launcher.py --analyst-pre-ml --symbol ETHUSDT

# Tactician full pipeline
python ares_launcher.py --tactician-pre-ml --symbol ETHUSDT
```

### For Individual Components
```bash
# Run just one step for debugging
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

## 📝 Notes

1. **Timeframe Consistency**: Always use 60m for Analyst and 15m for Tactician
2. **Dependencies**: These commands assume prior steps (labeling, regime splitting) are complete
3. **Orchestrators**: For full pipelines, use `--analyst-pre-ml` or `--tactician-pre-ml`
4. **Testing**: Start with `--execution-mode light` to test quickly
5. **Backward Compatibility**: Generic commands still work for custom use cases