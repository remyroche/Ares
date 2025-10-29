# 🚀 **Unified Training Commands Guide**

## ✅ **New Training Commands Available**

The Ares launcher now supports direct training commands using the new unified training pipeline. These commands provide granular control over model training without needing to run the entire pipeline.

### **🎯 Available Commands**

#### **1. Analyst Base Models Training**
```bash
# Train Analyst base models (LightGBM, CatBoost, Neural Networks)
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT

# With custom timeframe
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --timeframe 15m

# With custom execution mode
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light
```

#### **2. Analyst Ensemble Training**
```bash
# Train Analyst ensemble model
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT

# With custom timeframe
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --timeframe 15m

# With custom execution mode
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode full
```

#### **3. Tactician Base Models Training**
```bash
# Train Tactician base models (LightGBM, CatBoost, Neural Networks)
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT

# With custom timeframe
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT --timeframe 5m

# With custom execution mode
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT --execution-mode light
```

#### **4. Tactician Ensemble Training**
```bash
# Train Tactician ensemble model
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT

# With custom timeframe
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --timeframe 5m

# With custom execution mode
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --execution-mode full
```

---

## 🔧 **Command Parameters**

### **Required Parameters**
- `--symbol`: Trading symbol (default: ETHUSDT)
- `--exchange`: Exchange name (default: binance)
- `--timeframe`: Data timeframe (default: 15m)

### **Optional Parameters**
- `--execution-mode`: Execution mode (full, light, blank) - default: full
- `--data-dir`: Data directory (default: historical_data)
- `--direction`: Direction type (long, short, both) - default: long

### **Execution Modes**
- **`full`**: Complete training with all features and optimizations
- **`light`**: Lightweight training for testing and development
- **`blank`**: Minimal training for validation and testing

---

## 🎯 **Training Pipeline Integration**

### **What Each Command Does**

#### **Analyst Base Models (`--train-analyst-base`)**
- ✅ **Trains individual Analyst models** (LightGBM, CatBoost, Neural Networks)
- ✅ **Role-specific training** for Analyst timeframe (15m)
- ✅ **Optimized feature engineering** and selection
- ✅ **Performance monitoring** and optimization
- ✅ **Hardware optimization** for M1 systems

#### **Analyst Ensemble (`--train-analyst-ensemble`)**
- ✅ **Trains Analyst ensemble model** combining base models
- ✅ **Advanced ensemble methods** (Stacking, Blending, Voting)
- ✅ **Cross-validation** and out-of-fold predictions
- ✅ **Performance optimization** and model selection
- ✅ **Role-specific ensemble strategies**

#### **Tactician Base Models (`--train-tactician-base`)**
- ✅ **Trains individual Tactician models** (LightGBM, CatBoost, Neural Networks)
- ✅ **Role-specific training** for Tactician timeframe (5m)
- ✅ **Analyst-filtered training data** integration
- ✅ **Regime and Analyst features** integration
- ✅ **Enhanced tactical decision making**

#### **Tactician Ensemble (`--train-tactician-ensemble`)**
- ✅ **Trains Tactician ensemble model** combining base models
- ✅ **Multi-model ensemble training** with Analyst integration
- ✅ **Advanced ensemble methods** for tactical decisions
- ✅ **Cross-validation** and performance optimization
- ✅ **Final timing decision production**

---

## 🚀 **Usage Examples**

### **Quick Training (Development)**
```bash
# Quick Analyst base models training
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light

# Quick Tactician ensemble training
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --execution-mode light
```

### **Full Training (Production)**
```bash
# Full Analyst ensemble training
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode full

# Full Tactician base models training
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT --execution-mode full
```

### **Custom Configuration**
```bash
# Custom symbol and timeframe
python src/launcher/ares_launcher.py --train-analyst-base --symbol BTCUSDT --timeframe 1h

# Custom data directory
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --data-dir custom_data

# Custom direction
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --direction short
```

---

## 📊 **Training Pipeline Features**

### **Unified Architecture**
- ✅ **Consistent interface** across all training commands
- ✅ **Role-specific optimization** (Analyst vs Tactician)
- ✅ **Hardware optimization** for M1 systems
- ✅ **Comprehensive error handling** and recovery
- ✅ **Performance monitoring** and metrics

### **Advanced Features**
- ✅ **Memory optimization** with M1-specific optimizations
- ✅ **CPU optimization** for efficient processing
- ✅ **GPU acceleration** where available
- ✅ **Data quality monitoring** and validation
- ✅ **Hardware-aware processing** decisions

### **Production Ready**
- ✅ **Comprehensive logging** with visual indicators
- ✅ **Easy configuration** and customization
- ✅ **Clear error messages** and debugging
- ✅ **Performance metrics** and monitoring
- ✅ **Scalable architecture** for different workloads

---

## 🔍 **Command Compatibility**

### **Legacy Commands (Still Available)**
```bash
# Legacy Analyst training
python src/launcher/ares_launcher.py --analyst-models --symbol ETHUSDT
python src/launcher/ares_launcher.py --analyst-ensemble --symbol ETHUSDT

# Legacy Tactician training
python src/launcher/ares_launcher.py --tactician-models --symbol ETHUSDT
python src/launcher/ares_launcher.py --tactician-ensemble --symbol ETHUSDT
```

### **New Unified Commands (Recommended)**
```bash
# New unified training commands
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT
```

---

## 🎯 **Benefits of New Commands**

### **1. Direct Training Control**
- ✅ **No pipeline overhead** - train models directly
- ✅ **Faster execution** - skip unnecessary pipeline steps
- ✅ **Granular control** - train specific model types
- ✅ **Resource efficient** - optimized for training only

### **2. Unified Interface**
- ✅ **Consistent commands** across all model types
- ✅ **Same parameters** for all training commands
- ✅ **Easy to remember** - logical command structure
- ✅ **Comprehensive help** and documentation

### **3. Production Ready**
- ✅ **Robust error handling** and recovery
- ✅ **Performance monitoring** and optimization
- ✅ **Hardware optimization** for M1 systems
- ✅ **Comprehensive logging** and debugging

### **4. Developer Friendly**
- ✅ **Clear command structure** and naming
- ✅ **Comprehensive help text** and examples
- ✅ **Easy configuration** and customization
- ✅ **Consistent behavior** across all commands

---

## 🚀 **Quick Start Guide**

### **1. Test the Commands**
```bash
# Test Analyst base models training
python src/launcher/ares_launcher.py --train-analyst-base --symbol ETHUSDT --execution-mode light

# Test Tactician ensemble training
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --execution-mode light
```

### **2. Production Training**
```bash
# Full Analyst ensemble training
python src/launcher/ares_launcher.py --train-analyst-ensemble --symbol ETHUSDT --execution-mode full

# Full Tactician base models training
python src/launcher/ares_launcher.py --train-tactician-base --symbol ETHUSDT --execution-mode full
```

### **3. Custom Configuration**
```bash
# Custom symbol and timeframe
python src/launcher/ares_launcher.py --train-analyst-base --symbol BTCUSDT --timeframe 1h --execution-mode full

# Custom data directory and direction
python src/launcher/ares_launcher.py --train-tactician-ensemble --symbol ETHUSDT --data-dir custom_data --direction short
```

---

## 📚 **Additional Resources**

### **Help and Documentation**
```bash
# Get help for specific command
python src/launcher/ares_launcher.py --help

# Get help for training commands
python src/launcher/ares_launcher.py --train-analyst-base --help
```

### **Training Pipeline Documentation**
- **Core Architecture**: `src/training/steps/models_training/core/`
- **Unified Pipeline**: `src/training/steps/models_training/unified_training_pipeline.py`
- **Cleanup Summary**: `src/training/steps/models_training/CLEANUP_SUMMARY.md`

---

## 🎉 **Summary**

The new unified training commands provide:

1. ✅ **Direct training control** - train models without pipeline overhead
2. ✅ **Unified interface** - consistent commands across all model types
3. ✅ **Production ready** - robust error handling and optimization
4. ✅ **Developer friendly** - clear commands and comprehensive help
5. ✅ **Hardware optimized** - M1-specific optimizations throughout

**Ready to use!** 🚀

---

**Commands added on**: December 2024  
**Status**: ✅ Complete and verified  
**Compatibility**: ✅ Backward compatible with existing commands
