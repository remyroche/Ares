# Complete New Commands Reference

## 🎯 Summary of ALL New Commands

This document lists every new command available after the refactoring.

---

## 📋 Quick Command List

### Labeling Commands (NEW)
```bash
--analyst-labeler              # Analyst profit labeling (60m)
--tactician-labeler            # Tactician entry labeling (15m)
```

### Analyst Feature Engineering Commands (NEW)
```bash
analyst_feature_lookback_optimization
analyst_interactive_feature_generation
analyst_final_feature_selection
```

### Tactician Feature Engineering Commands (NEW)
```bash
tactician_feature_lookback_optimization
tactician_interactive_feature_generation
tactician_final_feature_selection
```

---

## 🚀 Complete Command Reference

### 1. Labeling Commands

#### Analyst Profit Labeler
```bash
# Shortcut flag
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

# Full command
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_profit_labeler \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

#### Tactician Entry Labeler
```bash
# Shortcut flag
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m

# Full command
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_entry_labeler \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

---

### 2. Analyst Feature Engineering Commands

#### Feature Lookback Optimization
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

#### Interactive Feature Generation
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

#### Final Feature Selection
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

---

### 3. Tactician Feature Engineering Commands

#### Feature Lookback Optimization
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

#### Interactive Feature Generation
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

#### Final Feature Selection
```bash
python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

---

## 📊 Command Comparison Table

| Step | Generic Command | Analyst Command | Tactician Command |
|------|----------------|-----------------|-------------------|
| **Labeling** | `multi_horizon_profit_labeler` | `analyst_profit_labeler` ✨ | `tactician_entry_labeler` ✨ |
| **Lookback Opt** | `feature_lookback_optimization` | `analyst_feature_lookback_optimization` ✨ | `tactician_feature_lookback_optimization` ✨ |
| **Feature Gen** | `interactive_feature_generation` | `analyst_interactive_feature_generation` ✨ | `tactician_interactive_feature_generation` ✨ |
| **Selection** | `final_feature_selection` | `analyst_final_feature_selection` ✨ | `tactician_final_feature_selection` ✨ |

✨ = NEW in this refactoring

---

## 🎭 Execution Modes

All commands support three execution modes:

```bash
--execution-mode full    # 1460 days, 100% intensity (production)
--execution-mode light   # 10 days, 5% intensity (testing)
--execution-mode blank   # 180 days, 10% intensity (validation)
```

---

## 💡 Usage Examples by Scenario

### Scenario 1: Quick Test of Analyst Labeling
```bash
python ares_launcher.py --analyst-labeler --execution-mode light \
    --symbol ETHUSDT --timeframe 60m
```

### Scenario 2: Quick Test of Tactician Labeling
```bash
python ares_launcher.py --tactician-labeler --execution-mode light \
    --symbol ETHUSDT --timeframe 15m
```

### Scenario 3: Full Analyst Feature Engineering
```bash
# All steps in sequence
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline analyst_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

### Scenario 4: Full Tactician Feature Engineering
```bash
# All steps in sequence
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

python ares_launcher.py --mode sub_pipeline \
    --sub_pipeline tactician_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

### Scenario 5: Use Complete Orchestrators (Easiest)
```bash
# Analyst full pipeline
python ares_launcher.py --analyst-pre-ml --symbol ETHUSDT

# Tactician full pipeline
python ares_launcher.py --tactician-pre-ml --symbol ETHUSDT
```

---

## 🔑 Key Points

1. **10 New Commands Total**: 2 labelers + 6 feature engineering (3 per role) + 2 shortcuts
2. **Role-Specific**: Clear separation between Analyst (60m) and Tactician (15m)
3. **Backward Compatible**: Original generic commands still work
4. **Orchestrator Equivalence**: Role-specific commands mimic orchestrator behavior
5. **Direct Access**: Can call any step individually without running full orchestrator

---

## 📝 Cheat Sheet

### Analyst Commands
```bash
--analyst-labeler
analyst_feature_lookback_optimization
analyst_interactive_feature_generation
analyst_final_feature_selection
```
**Timeframe**: Always use `--timeframe 60m`

### Tactician Commands
```bash
--tactician-labeler
tactician_feature_lookback_optimization
tactician_interactive_feature_generation
tactician_final_feature_selection
```
**Timeframe**: Always use `--timeframe 15m`

---

## 🎯 What Changed?

### Before Refactoring
- Only generic commands available
- No clear distinction between Analyst and Tactician
- Had to use orchestrators or manually configure timeframes

### After Refactoring (NOW)
- ✅ Dedicated labelers for each role
- ✅ Role-specific feature engineering commands
- ✅ Clear timeframe enforcement
- ✅ Can call individual steps with role context
- ✅ Shortcut flags for convenience

---

## 🚀 Getting Started

**Simplest way to test everything:**

```bash
# Test Analyst
python ares_launcher.py --analyst-labeler --execution-mode light --symbol ETHUSDT --timeframe 60m

# Test Tactician
python ares_launcher.py --tactician-labeler --execution-mode light --symbol ETHUSDT --timeframe 15m
```

**Run full pipelines:**

```bash
# Analyst
python ares_launcher.py --analyst-pre-ml --symbol ETHUSDT

# Tactician
python ares_launcher.py --tactician-pre-ml --symbol ETHUSDT
```

---

**Total New Commands: 10**
- 2 new labelers
- 6 role-specific feature engineering commands (3 × 2 roles)
- 2 new shortcut flags