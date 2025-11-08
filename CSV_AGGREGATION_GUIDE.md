# CSV Metrics Aggregation Guide

## Overview

The unified models training step now generates **TWO types of CSV reports** to support both individual training runs and consolidated analysis across all model types.

---

## 📊 Dual CSV Generation System

### 1. **Per-Run CSV** (Timestamped)
**Path:** `reports/{training_type}/{symbol}_{timeframe}_{direction}/{timestamp}/{training_type}_metrics.csv`

**Purpose:** Individual training run snapshot

**Characteristics:**
- Created fresh for each training run
- Contains only models from that specific run
- Timestamped directory prevents overwrites
- Ideal for run-specific analysis and debugging

**Example:**
```
reports/analyst_base/ETHUSDT_15m_long/20250108_143022/analyst_base_metrics.csv
reports/analyst_ensemble/ETHUSDT_15m_long/20250108_144533/analyst_ensemble_metrics.csv
reports/tactician_base/ETHUSDT_15m_long/20250108_150144/tactician_base_metrics.csv
reports/tactician_ensemble/ETHUSDT_15m_long/20250108_151655/tactician_ensemble_metrics.csv
```

---

### 2. **Consolidated CSV** (Aggregated)
**Path:** `reports/{symbol}_{timeframe}_{direction}/all_models_metrics.csv`

**Purpose:** Aggregate ALL models from ALL training runs

**Characteristics:**
- Single file per symbol/timeframe/direction
- Appends new models from each training run
- Automatically expands columns when new metrics appear
- Includes models from all 4 training types
- Thread-safe with file locking (Unix) or atomic writes
- Ideal for comparative analysis and trend tracking

**Example:**
```
reports/ETHUSDT_15m_long/all_models_metrics.csv
```

**Contents after 4 separate runs:**
```csv
model_name,training_type,symbol,timeframe,direction,execution_time_seconds,success,...
lightgbm,analyst_base,ETHUSDT,15m,long,234.56,True,...
tcn,analyst_base,ETHUSDT,15m,long,234.56,True,...
catboost,analyst_base,ETHUSDT,15m,long,234.56,True,...
stacking,analyst_ensemble,ETHUSDT,15m,long,145.23,True,...
lightgbm,tactician_base,ETHUSDT,15m,long,189.34,True,...
tcn,tactician_base,ETHUSDT,15m,long,189.34,True,...
catboost,tactician_base,ETHUSDT,15m,long,189.34,True,...
stacking,tactician_ensemble,ETHUSDT,15m,long,167.89,True,...
```

---

## 🔄 How It Works for Separate Runs

### Scenario: Training Models Separately

**Run 1: Analyst Base**
```bash
python train.py --training_type analyst_base --symbol ETHUSDT --timeframe 15m --direction long
```
**Result:**
- ✅ Creates: `reports/analyst_base/ETHUSDT_15m_long/20250108_143022/analyst_base_metrics.csv` (3 rows: lightgbm, tcn, catboost)
- ✅ Creates: `reports/ETHUSDT_15m_long/all_models_metrics.csv` (3 rows)

**Run 2: Analyst Ensemble**
```bash
python train.py --training_type analyst_ensemble --symbol ETHUSDT --timeframe 15m --direction long
```
**Result:**
- ✅ Creates: `reports/analyst_ensemble/ETHUSDT_15m_long/20250108_144533/analyst_ensemble_metrics.csv` (1 row: stacking)
- ✅ **Appends** to: `reports/ETHUSDT_15m_long/all_models_metrics.csv` (now 4 rows total)

**Run 3: Tactician Base**
```bash
python train.py --training_type tactician_base --symbol ETHUSDT --timeframe 15m --direction long
```
**Result:**
- ✅ Creates: `reports/tactician_base/ETHUSDT_15m_long/20250108_150144/tactician_base_metrics.csv` (3 rows)
- ✅ **Appends** to: `reports/ETHUSDT_15m_long/all_models_metrics.csv` (now 7 rows total)

**Run 4: Tactician Ensemble**
```bash
python train.py --training_type tactician_ensemble --symbol ETHUSDT --timeframe 15m --direction long
```
**Result:**
- ✅ Creates: `reports/tactician_ensemble/ETHUSDT_15m_long/20250108_151655/tactician_ensemble_metrics.csv` (1 row)
- ✅ **Appends** to: `reports/ETHUSDT_15m_long/all_models_metrics.csv` (now 8 rows total)

**Final State:**
```
reports/
├── ETHUSDT_15m_long/
│   └── all_models_metrics.csv              ← 8 rows (all models from all runs)
├── analyst_base/
│   └── ETHUSDT_15m_long/
│       └── 20250108_143022/
│           └── analyst_base_metrics.csv    ← 3 rows
├── analyst_ensemble/
│   └── ETHUSDT_15m_long/
│       └── 20250108_144533/
│           └── analyst_ensemble_metrics.csv ← 1 row
├── tactician_base/
│   └── ETHUSDT_15m_long/
│       └── 20250108_150144/
│           └── tactician_base_metrics.csv  ← 3 rows
└── tactician_ensemble/
    └── ETHUSDT_15m_long/
        └── 20250108_151655/
            └── tactician_ensemble_metrics.csv ← 1 row
```

---

## 🔧 Technical Features

### 1. **Dynamic Column Expansion**
When a new training run introduces metrics not present in previous runs:
- System reads existing CSV headers
- Merges with new columns
- **Rewrites** existing file with expanded headers
- Existing rows get `None` for new columns
- New rows populate all available columns

**Example:**
```
Run 1 (analyst_base): Columns = [model_name, accuracy, r2_score]
Run 2 (analyst_ensemble): Adds columns = [ensemble_diversity, stacking_improvement]

Result: all_models_metrics.csv now has 5 columns
        Run 1 rows have None for new ensemble columns
```

### 2. **Thread-Safe Append (Unix)**
```python
import fcntl
with open(csv_path, 'a') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
    writer.writerow(row)
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock
```
- Prevents corruption from concurrent training runs
- Falls back to atomic writes on Windows

### 3. **Streaming Write (No Memory Buffering)**
```python
# Writes row-by-row, never loads entire CSV into memory
for row in csv_rows:
    writer.writerow(row)  # Immediate write to disk
```
- Handles large datasets efficiently
- Low memory footprint
- Works even with thousands of models

### 4. **Header Management**
```python
if not file_exists:
    writer.writeheader()  # Only write header for new files
else:
    # Skip header, just append data
    pass
```
- Headers written only once
- Append mode preserves existing headers
- Column expansion handles header updates

---

## 📈 Use Cases

### Use Case 1: Sequential Training (Separate Runs)
**Scenario:** Train models one at a time over several hours/days

**Benefit:** Consolidated CSV accumulates results automatically
```python
# Monday: Train analyst models
train(training_type='analyst_base')      # → 3 rows added
train(training_type='analyst_ensemble')  # → 1 row added

# Tuesday: Train tactician models
train(training_type='tactician_base')      # → 3 rows added
train(training_type='tactician_ensemble')  # → 1 row added

# Result: all_models_metrics.csv has 8 rows
```

### Use Case 2: Comparative Analysis
**Scenario:** Compare all models side-by-side

**Benefit:** Single CSV with all metrics for easy filtering/sorting
```python
import pandas as pd
df = pd.read_csv('reports/ETHUSDT_15m_long/all_models_metrics.csv')

# Compare accuracies
print(df[['model_name', 'training_type', 'overall_performance_accuracy']].sort_values('overall_performance_accuracy', ascending=False))

# Filter by model type
analyst_models = df[df['training_type'].str.contains('analyst')]
```

### Use Case 3: Trend Tracking
**Scenario:** Track model performance over time

**Benefit:** Historical record with timestamps
```python
# Retrain weekly and append to same consolidated CSV
# Week 1: 8 rows
# Week 2: 16 rows (8 new + 8 old)
# Week 3: 24 rows (8 new + 16 old)

# Analyze performance trends
df['week'] = pd.to_datetime(df['timestamp']).dt.isocalendar().week
performance_by_week = df.groupby(['week', 'model_name'])['overall_performance_accuracy'].mean()
```

### Use Case 4: Export to Database
**Scenario:** Load metrics into PostgreSQL/MySQL for BI tools

**Benefit:** CSV format is universally compatible
```sql
COPY models_metrics FROM '/path/to/all_models_metrics.csv'
WITH (FORMAT csv, HEADER true);
```

---

## ⚠️ Important Notes

### 1. **Symbol/Timeframe/Direction Specificity**
Each combination gets its own consolidated CSV:
```
reports/ETHUSDT_15m_long/all_models_metrics.csv
reports/ETHUSDT_15m_short/all_models_metrics.csv
reports/BTCUSDT_1h_long/all_models_metrics.csv
```

### 2. **Column Compatibility**
- Same metrics produce same columns
- New metrics add new columns
- Different training types may have different columns
- Missing values filled with `None`

### 3. **File Growth**
Consolidated CSV grows indefinitely:
- Monitor file size for long-running systems
- Implement rotation/archival if needed
- Consider database export for very large histories

### 4. **Concurrent Writes**
- Unix systems: File locking prevents corruption
- Windows: Atomic writes (small race condition window)
- For high-concurrency, consider database backend

---

## 📊 CSV Schema Example

### Fixed Metadata Columns (Always Present)
```csv
model_name              # e.g., lightgbm, tcn, catboost, stacking
training_type           # analyst_base, analyst_ensemble, tactician_base, tactician_ensemble
symbol                  # ETHUSDT, BTCUSDT, etc.
timeframe              # 15m, 1h, 4h, etc.
direction              # long, short, both
execution_time_seconds # 234.56
success                # True/False
models_trained_count   # 3 for base, 1 for ensemble
timestamp              # 2025-01-08T14:30:22
```

### Performance Metrics (200+ possible columns)
```csv
overall_performance_accuracy
overall_performance_precision
overall_performance_r2_score
training_metrics_accuracy
validation_metrics_accuracy
test_metrics_accuracy
data_drift_checks_ks_test_train_val
uncertainty_calibration_brier_score
threshold_optimization_optimal_threshold
hpo_method
hpo_total_trials
...
```

---

## 🚀 Best Practices

### ✅ DO
- Use consolidated CSV for cross-model comparisons
- Use per-run CSV for debugging specific training runs
- Archive old consolidated CSVs periodically
- Export to database for long-term storage
- Filter by `timestamp` for temporal analysis

### ❌ DON'T
- Don't manually edit consolidated CSV (append-only)
- Don't delete while training is running
- Don't rely on row order (may change)
- Don't assume all columns exist in all rows

---

## 🔍 Troubleshooting

### Issue: "Columns don't match between runs"
**Solution:** This is expected. The system automatically expands columns.

### Issue: "Concurrent writes causing corruption"
**Solution:**
- Unix: File locking prevents this
- Windows: Avoid running multiple trainings simultaneously
- Consider staggered runs with delays

### Issue: "CSV file too large"
**Solution:**
```python
# Archive old data
import shutil
shutil.move('all_models_metrics.csv', 'archive/all_models_metrics_2024.csv')
# Next run creates fresh file
```

### Issue: "Missing metrics for some models"
**Solution:** Check per-run CSV for that specific model. Consolidated CSV shows `None` for metrics not available in that run.

---

## 📝 Summary

**Two CSV Files, Two Purposes:**

| Feature | Per-Run CSV | Consolidated CSV |
|---------|-------------|------------------|
| **Location** | Timestamped subdirectory | Symbol-level directory |
| **Scope** | Single training run | All training runs |
| **Rows** | Models from one run | Models from all runs |
| **Updates** | Created once | Appended continuously |
| **Use Case** | Run debugging | Comparative analysis |
| **Streaming** | ✅ Yes | ✅ Yes |
| **Memory** | Low | Low |
| **Thread-Safe** | N/A (write-once) | ✅ Yes (Unix) |

**Perfect for:**
- ✅ Training models separately (analyst, then tactician)
- ✅ Incremental retraining over time
- ✅ Performance tracking and monitoring
- ✅ Easy export to Excel, databases, BI tools
- ✅ Historical analysis and trend detection
