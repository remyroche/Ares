# Multi-Timeframe Training Logging Guide

## Overview

This guide documents the comprehensive logging system implemented for multi-timeframe training, ensuring thorough tracking of the training process, performance metrics, and any issues that arise.

## 🎯 **Logging Components**

### **1. MultiTimeframeEnsemble Logging**
- **Location**: `src/analyst/predictive_ensembles/multi_timeframe_ensemble.py`
- **Logger**: `MultiTimeframeEnsemble_{model_name}_{regime}`
- **Scope**: Individual model training across timeframes

### **2. EnhancedRegimePredictiveEnsembles Logging**
- **Location**: `src/analyst/predictive_ensembles/enhanced_ensemble_orchestrator.py`
- **Logger**: `EnhancedRegimePredictiveEnsembles`
- **Scope**: Orchestration across all regimes and model types

### **3. TwoTierIntegration Logging**
- **Location**: `src/analyst/predictive_ensembles/two_tier_integration.py`
- **Logger**: `TwoTierIntegration`
- **Scope**: Two-tier decision system integration

## 📊 **Logging Levels**

### **INFO Level** - Key Events
- Training start/completion
- Model success/failure
- Performance metrics
- Configuration details

### **DEBUG Level** - Detailed Process
- Individual timeframe training steps
- Prediction calculations
- Meta-learner operations
- Strategy classifications

### **WARNING Level** - Potential Issues
- Missing data for timeframes
- Model training failures
- Fallback operations

### **ERROR Level** - Critical Issues
- Training failures
- Prediction errors
- System errors

## 🔍 **Detailed Logging Features**

### **1. MultiTimeframeEnsemble Logging**

#### **Initialization Logging**
```
🚀 Initializing MultiTimeframeEnsemble for xgboost in BULL_TREND
📊 Active timeframes: ['1m', '5m', '15m', '1h']
⚙️ Configuration: {...}
```

#### **Training Process Logging**
```
🎯 Starting multi-timeframe ensemble training for xgboost in BULL_TREND
📈 Model type: xgboost
⏰ Available timeframes: ['1m', '5m', '15m', '1h']
📊 Data shapes: [('1m', (1000, 15)), ('5m', (500, 15)), ...]

🔄 [1/4] Training 1m timeframe...
🔧 Training xgboost model for 1m
📊 Data shape: (1000, 15)
📈 Data columns: ['open', 'high', 'low', 'close', ...]
📊 Features shape: (1000, 14)
🎯 Target distribution: {'HOLD': 600, 'BUY': 250, 'SELL': 150}
🌳 Training XGBoost model...
🔄 Starting cross-validation...
📊 Fold 1/3: 667 train, 333 validation
📊 Fold 2/3: 667 train, 333 validation
📊 Fold 3/3: 666 train, 334 validation
✅ XGBoost model training completed
✅ 1m training completed in 12.34s
📊 Collecting predictions for 1m...
📈 1m stats: 1000 predictions, avg confidence: 0.823
```

#### **Meta-Learner Training Logging**
```
🧠 Training meta-learner for timeframe combination...
📊 Timeframes: ['1m', '5m', '15m', '1h']
🔧 Preparing meta-learner data...
📊 Found 1000 common timestamps
📊 Meta-learner data prepared: (1000, 25)
📊 Meta features shape: (1000, 24)
🎯 Meta target distribution: {'HOLD': 600, 'BUY': 250, 'SELL': 150}
🔧 Encoding target labels...
🔧 Scaling features...
🌳 Training LightGBM meta-learner...
✅ Meta-learner trained successfully
📊 Meta-learner feature importance: [0.15, 0.12, 0.10, ...]
```

#### **Prediction Logging**
```
🔮 Getting prediction for xgboost in BULL_TREND
📊 Getting prediction for 1m...
📊 1m: BUY (confidence: 0.856)
📊 Getting prediction for 5m...
📊 5m: BUY (confidence: 0.789)
🧠 Combining predictions with meta-learner...
📊 Meta-features: [1.0, 0.0, 0.0, 0.856, 1.0, 0.0, 0.0, 0.789, ...]
🎯 Meta-learner prediction: BUY (confidence: 0.823)
🎯 Final prediction: BUY (confidence: 0.823)
```

### **2. EnhancedRegimePredictiveEnsembles Logging**

#### **Training Orchestration Logging**
```
🚀 Initializing EnhancedRegimePredictiveEnsembles
📊 Active timeframes: ['1m', '5m', '15m', '1h']
🔧 Model types: ['xgboost', 'lstm', 'random_forest']
⚙️ Timeframe set: intraday

🎯 Starting enhanced multi-timeframe ensemble training for ETHUSDT
📊 Available timeframes: ['1m', '5m', '15m', '1h']
📈 Data shapes: [('1m', (1000, 15)), ('5m', (500, 15)), ...]

🔄 [1/5] Training enhanced ensemble for regime: BULL_TREND
🔧 [1/3] Training xgboost for BULL_TREND
✅ xgboost for BULL_TREND trained successfully in 45.67s
💾 Saved BULL_TREND_xgboost to models/ETHUSDT_BULL_TREND_xgboost
🔧 [2/3] Training lstm for BULL_TREND
✅ lstm for BULL_TREND trained successfully in 23.45s
💾 Saved BULL_TREND_lstm to models/ETHUSDT_BULL_TREND_lstm
📊 BULL_TREND summary: 3/3 models successful, time: 89.23s
```

#### **Global Meta-Learner Logging**
```
🧠 Training enhanced global meta-learner...
📊 Collecting predictions from BULL_TREND regime...
📊 Collecting from BULL_TREND_xgboost...
📊 Added 1000 predictions from BULL_TREND_xgboost
📊 Collecting from BULL_TREND_lstm...
📊 Added 1000 predictions from BULL_TREND_lstm
📊 Collected data from 15 trained ensembles
📊 Training global meta-learner with 15000 data points...
```

#### **Final Training Summary**
```
✅ Enhanced multi-timeframe ensemble training completed!
⏱️ Total training time: 456.78s
📊 Training summary:
   - Asset: ETHUSDT
   - Regimes: 5
   - Model types: 3
   - Total ensembles: 15
   - Successful: 14
   - Failed: 1
   - Success rate: 93.3%
   - Meta-learner training time: 12.34s
   - BULL_TREND: 3/3 models (100.0%), time: 89.23s
   - BEAR_TREND: 3/3 models (100.0%), time: 87.45s
   - SIDEWAYS_RANGE: 2/3 models (66.7%), time: 78.90s
```

### **3. TwoTierIntegration Logging**

#### **Initialization Logging**
```
🚀 Initializing TwoTierIntegration
📊 Tier 1 timeframes: ['1m', '5m', '15m', '1h']
⏰ Tier 2 timeframes: ['1m', '5m']
🎯 Direction threshold: 0.7
⚡ Timing threshold: 0.8
🔥 High leverage mode: True
```

#### **Prediction Enhancement Logging**
```
🎯 Enhancing ensemble prediction with two-tier logic...
📊 Base prediction: BUY (confidence: 0.823)
📈 Regime: BULL_TREND
🔧 Processing Tier 1 (direction/strategy)...
📊 Tier 1 result: LONG (BULLISH_TREND)
🎯 Should trade: True
⏰ Processing Tier 2 (precise timing)...
📊 Tier 2 result: timing signal 0.856
⚡ Should execute: True
🎯 Final decision: LONG
📈 Position multiplier: 1.32
🛡️ Risk multiplier: 0.74
📊 Confidence level: 0.840
✅ Two-tier enhancement completed in 0.023s
```

#### **Strategy Classification Logging**
```
🔧 Classifying strategy from regime: BULL_TREND, prediction: BUY
📊 Strategy classified as: BULLISH_TREND
⏰ Getting Tier 2 timing for strategy: BULLISH_TREND
🔧 Simulating timing signal for strategy: BULLISH_TREND
📊 Trend strategy adjustment: +0.05
📊 Final timing signal: 0.800
🔧 Getting strategy-specific timing for: BULLISH_TREND
📊 Strategy timing result: BULLISH_PULLBACK - Looking for bullish pullback entry
⚡ Should execute: True
```

## 📈 **Performance Metrics Logging**

### **Training Performance**
- **Time tracking**: Individual timeframe training times
- **Success rates**: Per model type and regime
- **Data statistics**: Shapes, distributions, missing values
- **Model performance**: Confidence scores, prediction counts

### **Prediction Performance**
- **Timeframe predictions**: Individual predictions per timeframe
- **Confidence tracking**: Confidence scores for each prediction
- **Meta-learner performance**: Feature importance, prediction accuracy
- **Two-tier performance**: Tier 1/2 decisions and timing

### **System Performance**
- **Memory usage**: Data shapes and model sizes
- **Processing time**: Training and prediction durations
- **Error tracking**: Failed models and fallback operations
- **Resource utilization**: Model loading/saving operations

## 🔧 **Logging Configuration**

### **Log Level Control**
```python
# Set log level for detailed debugging
import logging
logging.getLogger("MultiTimeframeEnsemble").setLevel(logging.DEBUG)
logging.getLogger("EnhancedRegimePredictiveEnsembles").setLevel(logging.DEBUG)
logging.getLogger("TwoTierIntegration").setLevel(logging.DEBUG)
```

### **Log Format**
```
[2024-01-15 14:30:45] INFO MultiTimeframeEnsemble_xgboost_BULL_TREND: 🎯 Starting multi-timeframe ensemble training
[2024-01-15 14:30:46] DEBUG MultiTimeframeEnsemble_xgboost_BULL_TREND: 📊 Features shape: (1000, 14)
[2024-01-15 14:30:47] INFO MultiTimeframeEnsemble_xgboost_BULL_TREND: ✅ 1m training completed in 12.34s
```

## 🚨 **Error Tracking**

### **Training Errors**
```
❌ xgboost for BULL_TREND training failed
💥 Error training xgboost model: ValueError: Invalid parameter
⚠️ No data for timeframe 1m, skipping
⚠️ Model doesn't support predict_proba, using default confidence
```

### **Prediction Errors**
```
💥 Error getting prediction: IndexError: list index out of range
💥 Error combining with meta-learner: ValueError: Found input variables with inconsistent numbers of samples
⚠️ No trained model for 1m
⚠️ No valid features for 1m
```

### **System Errors**
```
💥 Error saving model: PermissionError: [Errno 13] Permission denied
💥 Error loading model: FileNotFoundError: [Errno 2] No such file or directory
⚠️ No meta-learner found
⚠️ No enhanced predictions available, using fallback
```

## 📊 **Monitoring Dashboard**

### **Real-time Training Progress**
```
🔄 [2/4] Training 5m timeframe...
📊 5m: 500 predictions, avg confidence: 0.789
🔄 [3/4] Training 15m timeframe...
📊 15m: 200 predictions, avg confidence: 0.756
🔄 [4/4] Training 1h timeframe...
📊 1h: 100 predictions, avg confidence: 0.823
```

### **Performance Summary**
```
📊 Training summary:
   - Total ensembles: 15
   - Successful: 14 (93.3%)
   - Failed: 1 (6.7%)
   - Total time: 456.78s
   - Average per ensemble: 30.45s
```

### **Prediction Quality**
```
📊 Prediction quality:
   - High confidence (>0.8): 8/15 (53.3%)
   - Medium confidence (0.6-0.8): 5/15 (33.3%)
   - Low confidence (<0.6): 2/15 (13.3%)
```

## 🎯 **Best Practices**

### **1. Monitor Key Metrics**
- Track training success rates per regime/model
- Monitor prediction confidence distributions
- Watch for timeframe-specific issues
- Check meta-learner performance

### **2. Debug Issues**
- Use DEBUG level for detailed troubleshooting
- Check individual timeframe training logs
- Monitor meta-learner feature importance
- Verify two-tier decision logic

### **3. Performance Optimization**
- Monitor training times per timeframe
- Track memory usage during training
- Check prediction latency
- Optimize based on log insights

This comprehensive logging system ensures **complete visibility** into the multi-timeframe training process, making it easy to monitor performance, debug issues, and optimize the system! 🚀 