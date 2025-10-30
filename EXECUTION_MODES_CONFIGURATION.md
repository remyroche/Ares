# Execution Modes Configuration

**Date:** October 30, 2025  
**Status:** ✅ Fully Configured

---

## 🎯 **Execution Modes Overview**

The system supports 4 execution modes that control training intensity:

| Mode | Purpose | Iterations/Epochs | Use Case |
|------|---------|-------------------|----------|
| **LIGHT** | Quick testing | **10%** | Rapid prototyping, debugging |
| **BLANK** | Fast training | **20%** | Quick validation, testing changes |
| **FULL** | Production training | **100%** | Standard production deployment |
| **PRODUCTION** | Extended training | **200%** | Maximum accuracy, competitions |

---

## 📊 **Model Configuration by Mode**

### Neural Networks (TCN, GRU, Autoencoder)

| Component | LIGHT (10%) | BLANK (20%) | FULL (100%) | PRODUCTION (200%) |
|-----------|-------------|-------------|-------------|-------------------|
| **TCN Epochs** | 5 | 10 | 50 | 100 |
| **Autoencoder Epochs** | 3 | 5 | 25 | 50 |
| **GRU Epochs** | 4 | 8 | 40 | 80 |

### Tree-Based Models (LGBM, CatBoost, ExtraTrees)

| Component | LIGHT (10%) | BLANK (20%) | FULL (100%) | PRODUCTION (200%) |
|-----------|-------------|-------------|-------------|-------------------|
| **LGBM n_estimators** | 100 | 200 | 1000 | 2000 |
| **CatBoost iterations** | 50 | 100 | 500 | 1000 |
| **ExtraTrees n_estimators** | 50 | 100 | 500 | 1000 |

### Cross-Validation

| Component | LIGHT (10%) | BLANK (20%) | FULL (100%) | PRODUCTION (200%) |
|-----------|-------------|-------------|-------------|-------------------|
| **CV Folds** (10k samples) | 1 | 1 | 5 | 5 |
| **CV Folds** (50k samples) | 1 | 2 | 10 | 10 |

---

## ⏱️ **Estimated Training Times**

### Analyst Models (LGBM + TCN + CatBoost)

| Mode | Total Time | Per Model | Use Case |
|------|-----------|-----------|----------|
| **LIGHT** | ~3 min | LGBM: 30s, TCN: 90s, CatBoost: 60s | Quick iteration |
| **BLANK** | ~6 min | LGBM: 60s, TCN: 180s, CatBoost: 120s | Fast testing |
| **FULL** | ~30 min | LGBM: 5m, TCN: 15m, CatBoost: 10m | Production ready |
| **PRODUCTION** | ~60 min | LGBM: 10m, TCN: 30m, CatBoost: 20m | Maximum quality |

### Tactician Models (LGBM + CatBoost + ExtraTrees + GRU)

| Mode | Total Time | Per Model | Use Case |
|------|-----------|-----------|----------|
| **LIGHT** | ~2 min | Each ~30s | Rapid testing |
| **BLANK** | ~4 min | Each ~60s | Quick validation |
| **FULL** | ~20 min | Each ~5m | Production ready |
| **PRODUCTION** | ~40 min | Each ~10m | Maximum quality |

---

## 🚀 **How to Use**

### Setting Execution Mode

**Via Command Line:**
```bash
# Light mode (10% - quick testing)
python ares_launcher.py step05 --execution-mode light

# Blank mode (20% - fast training)
python ares_launcher.py step05 --execution-mode blank

# Full mode (100% - production)
python ares_launcher.py step05 --execution-mode full

# Production mode (200% - maximum accuracy)
python ares_launcher.py step05 --execution-mode production
```

**Via Config:**
```python
config = {
    'execution_mode': 'light',  # or 'blank', 'full', 'production'
    'training_type': 'analyst_base',
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m'
}
```

---

## 📈 **Mode Selection Guide**

### When to Use LIGHT (10%)
- ✅ **Development**: Testing new features
- ✅ **Debugging**: Quick iteration cycles
- ✅ **Prototyping**: Rapid experiments
- ⚠️ **Not for**: Any production deployment

**Expected Results:**
- Training: ~2-5 minutes total
- Accuracy: 60-70% (baseline)
- Purpose: Verify code works

### When to Use BLANK (20%)
- ✅ **Validation**: Testing configuration changes
- ✅ **Quick Checks**: Verify data pipeline
- ✅ **CI/CD**: Automated testing
- ⚠️ **Not for**: Production deployment

**Expected Results:**
- Training: ~5-10 minutes total
- Accuracy: 70-75%
- Purpose: Verify system health

### When to Use FULL (100%)
- ✅ **Production**: Standard deployment
- ✅ **Live Trading**: Real money trading
- ✅ **Backtesting**: Performance evaluation
- ✅ **Default**: Recommended for most use cases

**Expected Results:**
- Training: ~20-30 minutes total
- Accuracy: 78-82%
- Purpose: Production-ready models

### When to Use PRODUCTION (200%)
- ✅ **Competitions**: Maximum accuracy needed
- ✅ **Research**: Academic/competition work
- ✅ **Final Models**: When time is not a constraint
- ⚠️ **Overkill**: Usually not needed for live trading

**Expected Results:**
- Training: ~40-60 minutes total
- Accuracy: 80-85%
- Purpose: Squeeze out last 1-2% accuracy

---

## 🔧 **Implementation Details**

### Dynamic Configuration Calculator

The `DynamicConfigCalculator` class (`dynamic_config_calculator.py`) handles all mode scaling:

```python
# Epochs calculation
if execution_mode == 'light':
    epochs = int(base_epochs * 0.1)  # 10%
elif execution_mode == 'blank':
    epochs = int(base_epochs * 0.2)  # 20%
elif execution_mode == 'full':
    epochs = base_epochs              # 100%
else:  # production
    epochs = int(base_epochs * 2.0)   # 200%
```

### Applied To:
- ✅ Neural network epochs (TCN, GRU, Autoencoder)
- ✅ Tree-based iterations (LGBM, CatBoost, ExtraTrees)
- ✅ Cross-validation folds
- ✅ HPO trials (hyperparameter optimization)
- ✅ Early stopping patience

---

## 📝 **Configuration Files**

### Base Configs (Full Mode Values)

**Analyst Base:**
```yaml
# analyst_base_config.yaml
tcn:
  epochs: 50                    # FULL mode
  autoencoder_epochs: 25        # FULL mode
  batch_size: 64

catboost:
  iterations: 500               # FULL mode
  depth: 6
```

**Tactician Base:**
```yaml
# tactician_base_config.yaml
gru:
  epochs: 40                    # FULL mode
  batch_size: 256

lgbm:
  n_estimators: 1000            # FULL mode

catboost:
  iterations: 500               # FULL mode

extratrees:
  n_estimators: 500             # FULL mode
```

### Dynamic Scaling

The system automatically scales these values based on execution mode:
- **LIGHT**: Multiply by 0.1
- **BLANK**: Multiply by 0.2
- **FULL**: No change (as defined in config)
- **PRODUCTION**: Multiply by 2.0

---

## 💡 **Best Practices**

### Development Workflow
1. **LIGHT mode**: Develop and debug (10%)
2. **BLANK mode**: Validate changes (20%)
3. **FULL mode**: Production deployment (100%)
4. **PRODUCTION mode**: Final optimization (200%)

### Performance Expectations

| Metric | LIGHT | BLANK | FULL | PRODUCTION |
|--------|-------|-------|------|------------|
| **Speed** | 🚀🚀🚀🚀 | 🚀🚀🚀 | 🚀🚀 | 🚀 |
| **Accuracy** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **CPU Usage** | 30-40% | 50-60% | 85-95% | 95-100% |
| **Memory** | Low | Medium | High | Very High |

### Recommended Daily Usage
- **Morning**: LIGHT mode for testing new ideas
- **Afternoon**: BLANK mode for validating changes
- **Evening**: FULL mode for production training
- **Weekend**: PRODUCTION mode for competitions/research

---

## ✅ **Conclusion**

The execution mode system provides flexible control over training intensity:

- **LIGHT (10%)**: Fast iteration for development
- **BLANK (20%)**: Quick validation for testing
- **FULL (100%)**: Production-ready training
- **PRODUCTION (200%)**: Maximum accuracy when needed

Choose the mode that best fits your current goal and time constraints! 🎯

