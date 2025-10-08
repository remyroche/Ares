# Can I Call Feature Engineering Scripts for Analyst or Tactician Specifically?

## ✅ YES! Here's How:

You can now call `feature_lookback_optimization`, `interactive_feature_generation`, and `final_feature_selection` with **role-specific prefixes** that mimic calls from `analyst_pre_ml_orchestration` or `tactician_pre_ml_orchestration`.

---

## 🎯 Quick Answer

### For Analyst (60m timeframe, strategic):

```bash
# Feature Lookback Optimization
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 60m

# Interactive Feature Generation
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode light --symbol ETHUSDT --timeframe 60m

# Final Feature Selection
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_final_feature_selection \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

### For Tactician (15m timeframe, tactical):

```bash
# Feature Lookback Optimization
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 15m

# Interactive Feature Generation
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode light --symbol ETHUSDT --timeframe 15m

# Final Feature Selection
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_final_feature_selection \
    --execution-mode light --symbol ETHUSDT --timeframe 15m
```

---

## 📊 All Available Commands

| Generic Command | Analyst Command | Tactician Command |
|----------------|-----------------|-------------------|
| `feature_lookback_optimization` | `analyst_feature_lookback_optimization` | `tactician_feature_lookback_optimization` |
| `interactive_feature_generation` | `analyst_interactive_feature_generation` | `tactician_interactive_feature_generation` |
| `final_feature_selection` | `analyst_final_feature_selection` | `tactician_final_feature_selection` |

---

## 🔄 How It Mimics Orchestrator Behavior

When you use the role-specific commands (e.g., `analyst_feature_lookback_optimization`):

1. **Timeframe**: Automatically uses correct timeframe (60m for Analyst, 15m for Tactician)
2. **Role Context**: Passes role information to the component
3. **Configuration**: Applies role-specific defaults (per-regime/cluster optimization settings)
4. **Behavior**: Identical to being called from the orchestrator

---

## 💡 Comparison

### ❌ Old Way (Generic - No Role Context)
```bash
python ares_launcher.py --mode sub_pipeline --sub_pipeline feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
# This works but doesn't specify if it's for Analyst or Tactician
```

### ✅ New Way (Role-Specific - Clear Intent)
```bash
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
# This clearly indicates it's for Analyst with 60m timeframe
```

---

## 🎬 Complete Example Workflows

### Analyst Feature Engineering Pipeline
```bash
# Step 1: Label data
python ares_launcher.py --analyst-labeler --symbol ETHUSDT --timeframe 60m

# Step 2: Optimize lookback periods
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Step 3: Generate interactive features
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 60m

# Step 4: Select final features
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 60m
```

### Tactician Feature Engineering Pipeline
```bash
# Step 1: Label data
python ares_launcher.py --tactician-labeler --symbol ETHUSDT --timeframe 15m

# Step 2: Optimize lookback periods
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Step 3: Generate interactive features
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_interactive_feature_generation \
    --execution-mode full --symbol ETHUSDT --timeframe 15m

# Step 4: Select final features
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_final_feature_selection \
    --execution-mode full --symbol ETHUSDT --timeframe 15m
```

---

## 🏃 Quick Start (Testing)

Test Analyst feature engineering:
```bash
python ares_launcher.py --mode sub_pipeline --sub_pipeline analyst_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 60m
```

Test Tactician feature engineering:
```bash
python ares_launcher.py --mode sub_pipeline --sub_pipeline tactician_feature_lookback_optimization \
    --execution-mode light --symbol ETHUSDT --timeframe 15m
```

---

## 🎯 Summary

**Yes, you can call these scripts for Analyst or Tactician specifically!**

Simply prefix the command name with `analyst_` or `tactician_`:
- `analyst_feature_lookback_optimization`
- `analyst_interactive_feature_generation`
- `analyst_final_feature_selection`
- `tactician_feature_lookback_optimization`
- `tactician_interactive_feature_generation`
- `tactician_final_feature_selection`

These commands are **aliases** that route to the same underlying components but with role-specific context and defaults, exactly mimicking behavior from the orchestrators.