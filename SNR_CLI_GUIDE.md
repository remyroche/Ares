# SNR Diagnostics CLI Guide

## Overview

The SNR Diagnostics CLI provides command-line tools for analyzing label quality, learnability, and model robustness. All commands automatically use the **latest trained model**, **latest features**, and **latest generated labels** from your artifacts directory.

## Installation

No additional installation required! The CLI is ready to use.

## Quick Start

```bash
cd /home/user/Ares

# Run label quality diagnostics
./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m

# Run label learnability diagnostics
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m

# Run model robustness diagnostics
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m
```

## Commands

### 1. Label Quality

**Purpose**: Assess the quality and consistency of your labels

**Command**:
```bash
./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m
```

**What It Analyzes**:

#### Noise Ceiling
- **ICC (Intraclass Correlation)**: Measures label consistency
  - Uses bootstrap ensemble to create pseudo-replicates
  - Computes one-way and two-way ICC
  - Determines theoretical maximum achievable R²

- **Expected Max R²**: The ceiling on model performance
  - Based on label variance decomposition
  - Warns if model exceeds ceiling (potential data leakage!)

#### Aleatoric Uncertainty
- **Aleatoric Fraction**: Irreducible uncertainty (inherent noise)
- **Epistemic Fraction**: Reducible uncertainty (model/knowledge gaps)
- **Calibration Score**: How well predicted uncertainty matches observed errors

**Outputs**:
- `outcomes/label_quality_{symbol}_{timeframe}_{timestamp}/`
  - `label_quality_report_{symbol}_{timeframe}_{timestamp}.csv`
  - `label_quality_report_{symbol}_{timeframe}_{timestamp}.md`
  - `noise_ceiling_analysis.png`
  - `uncertainty_decomposition.png`
  - `uncertainty_calibration.png`

**Interpretation**:

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| ICC | > 0.75 | 0.60-0.75 | 0.40-0.60 | < 0.40 |
| Aleatoric | < 40% | 40-60% | 60-80% | > 80% |

**Warning Signs**:
- ⚠️ Model R² > Expected Max R² → **Data leakage!**
- ⚠️ Aleatoric > 60% → **Noise-limited, improvement unlikely**
- ⚠️ ICC < 0.40 → **Poor label quality**

---

### 2. Label Learnability

**Purpose**: Determine if labels contain learnable signal vs noise

**Command**:
```bash
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m
```

**What It Analyzes**:

#### Core Metrics
- **R²**: Coefficient of determination
- **SNR**: Signal-to-Noise Ratio = R² / (1 - R²)
- **Permutation p-value**: Statistical significance (shuffled labels test)
- **Bootstrap 95% CI**: Confidence interval for R²

#### Naive Baselines
Tests against simple predictors to establish floor:
- **Mean**: Always predict mean of y
- **Median**: Always predict median of y
- **Simple Linear**: Ridge regression with high regularization
- **First Feature Only**: Using only the first feature

**Outputs**:
- `outcomes/label_learnability_{symbol}_{timeframe}_{timestamp}/`
  - `label_learnability_report_{symbol}_{timeframe}_{timestamp}.csv`
  - `label_learnability_report_{symbol}_{timeframe}_{timestamp}.md`

**Interpretation**:

| Metric | Strong Signal | Moderate | Weak |
|--------|--------------|----------|------|
| R² | > 0.40 | 0.10-0.40 | < 0.10 |
| SNR | > 1.0 | 0.3-1.0 | < 0.3 |
| p-value | < 0.01 | 0.01-0.20 | > 0.20 |

**Baseline Comparison**:
- Model improvement > 0.2 → **Strong real signal**
- Model improvement 0.05-0.2 → **Moderate signal**
- Model improvement < 0.05 → **Weak signal, may be fitting noise**

---

### 3. Model Robustness

**Purpose**: Assess model and feature quality through residual analysis

**Command**:
```bash
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m
```

**What It Analyzes**:

#### Bootstrap Confidence Interval
- Detailed 95% CI with 2000 iterations
- CI width analysis (stability)
- Checks if CI includes zero

#### Residual Structure
- **Normality tests**: Shapiro-Wilk, Anderson-Darling
- **Residual predictability**: Can we predict residuals from features?
  - If yes → Model missing patterns
  - If no → Model extracted available signal

#### Residual Autocorrelation
- **Durbin-Watson statistic**: Tests independence
- **ACF(1)**: Lag-1 autocorrelation
- **Temporal structure**: Are errors correlated over time?

#### Model Family Comparison
- Compares your model against:
  - Linear (Ridge, Lasso, ElasticNet)
  - Random Forest (shallow & deep)
  - Gradient Boosting (shallow & deep)
  - MLP (small & large)

**Outputs**:
- `outcomes/model_robustness_{symbol}_{timeframe}_{timestamp}/`
  - `model_robustness_report_{symbol}_{timeframe}_{timestamp}.csv`
  - `model_robustness_report_{symbol}_{timeframe}_{timestamp}.md`
  - `model_family_comparison.png`
  - `temporal_analysis.png`

**Interpretation**:

| Metric | Good | Warning | Problem |
|--------|------|---------|---------|
| CI Width | < 0.1 | 0.1-0.2 | > 0.2 |
| Residual R² | < 0.05 | 0.05-0.10 | > 0.10 |
| Durbin-Watson | 1.5-2.5 | 1.2-1.5 or 2.5-2.8 | < 1.2 or > 2.8 |
| ACF(1) | < 0.1 | 0.1-0.2 | > 0.2 |

**Warning Signs**:
- ⚠️ Wide CI (>0.2) → **Unstable performance**
- ⚠️ Residual R² > 0.1 → **Model missing patterns**
- ⚠️ High ACF(1) > 0.2 → **Missing temporal features**
- ⚠️ Other models >> current → **Try different architecture**

---

## Output Format

### CSV Reports

All commands generate CSV reports with:
- **Category**: Diagnostic category (e.g., "Core Metrics")
- **Metric**: Specific metric name
- **Value**: Numeric value
- **Interpretation**: Human-readable interpretation

Example:
```csv
category,metric,value,interpretation
Core Metrics,R²,0.3245,Moderate signal
Core Metrics,SNR,0.4821,Weak signal
Bootstrap CI,CI Width,0.0823,Stable performance
```

### Markdown Reports

Comprehensive reports with:
- Executive summary
- Detailed metrics tables
- Interpretation guidelines
- Warnings and recommendations
- Links to visualizations

### Visualizations

Automatically generated plots:
- Noise ceiling heatmaps
- Uncertainty pie charts
- Calibration curves
- Residual analysis plots
- Model comparison bar charts
- Temporal autocorrelation plots

---

## Workflow Examples

### Scenario 1: New Model Evaluation

```bash
# Step 1: Check if labels are learnable
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m

# If SNR > 0.5 and p < 0.05, proceed...

# Step 2: Assess label quality
./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m

# If aleatoric < 60%, proceed...

# Step 3: Analyze model robustness
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m

# Review residuals and model comparison
```

### Scenario 2: Debugging Poor Performance

```bash
# Model has low R² - why?

# Step 1: Check learnability
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m
# → If barely beats baselines: Labels may be noisy or features inadequate

# Step 2: Check label quality
./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m
# → If aleatoric > 60%: Inherent noise, limited improvement possible
# → If aleatoric < 40%: Model-limited, try better models/features

# Step 3: Check model robustness
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m
# → If residual R² > 0.1: Model missing patterns
# → If model family comparison shows better options: Try different architecture
```

### Scenario 3: Pre-Production Validation

```bash
# Before deploying, validate everything

# Check for data leakage
./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m
# → Ensure model R² ≤ expected max R²

# Verify statistical significance
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m
# → Ensure p-value < 0.01

# Confirm robustness
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m
# → Ensure narrow CI, no residual structure
```

---

## Advanced Usage

### Custom Artifact Locations

The CLI automatically finds the latest artifacts in:
- `artifacts/labeled_data_{symbol}_{timeframe}_*.parquet`
- `artifacts/{model_type}_model_{symbol}_{timeframe}_*.pkl`

### Running Multiple Symbols

```bash
#!/bin/bash
# Run diagnostics for all symbols

for symbol in BTCUSDT ETHUSDT SOLUSDT; do
  echo "Analyzing $symbol..."
  ./snr_diagnostics label-learnability --symbol $symbol --timeframe 15m
  ./snr_diagnostics model-robustness --symbol $symbol --timeframe 15m
done
```

### Scheduled Diagnostics

Add to cron for regular monitoring:
```bash
# Run daily at 2 AM
0 2 * * * cd /home/user/Ares && ./snr_diagnostics label-quality --symbol BTCUSDT --timeframe 15m
```

---

## Troubleshooting

### "No labeled data found"

**Problem**: CLI can't find labeled data files

**Solutions**:
1. Check if meta-labeling step has run
2. Verify artifact naming: `labeled_data_{symbol}_{timeframe}_*.parquet`
3. Check `artifacts/` directory exists

### "No label column found in data"

**Problem**: Labeled data missing expected columns

**Solutions**:
1. Ensure data has one of: `meta_label`, `label`, or `target` column
2. Re-run meta-labeling step
3. Check data schema

### "Could not load model"

**Problem**: Trained model not found or incompatible

**Solutions**:
1. CLI will train a simple RF model for diagnostics
2. For full model analysis, ensure model is pickled in `artifacts/`
3. Check model naming: `{model_type}_model_{symbol}_{timeframe}_*.pkl`

### Memory Issues

**Problem**: Large datasets cause OOM errors

**Solutions**:
1. Reduce bootstrap iterations (edit CLI defaults)
2. Subsample data before running
3. Run diagnostics on validation set only

---

## Performance

**Typical Runtime** (1000 samples, 50 features):

| Command | Duration |
|---------|----------|
| label-quality | 5-10 min |
| label-learnability | 2-5 min |
| model-robustness | 10-15 min |

**Factors Affecting Speed**:
- Number of samples
- Number of features
- Bootstrap/permutation iterations
- Model complexity

---

## Integration with Pipeline

### Automated Workflow

Add to your training pipeline:

```python
import subprocess

def run_diagnostics(symbol, timeframe):
    """Run all diagnostics after training."""
    commands = [
        f"./snr_diagnostics label-quality --symbol {symbol} --timeframe {timeframe}",
        f"./snr_diagnostics label-learnability --symbol {symbol} --timeframe {timeframe}",
        f"./snr_diagnostics model-robustness --symbol {symbol} --timeframe {timeframe}",
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)
```

### Monitoring

Parse CSV reports for automated alerts:

```python
import pandas as pd

def check_for_issues(csv_path):
    """Check diagnostic reports for warning signs."""
    df = pd.read_csv(csv_path)

    issues = []

    # Check for data leakage
    ceiling_row = df[df['metric'] == 'Ceiling Exceeded']
    if ceiling_row['value'].iloc[0]:
        issues.append("Data leakage detected!")

    # Check for weak signal
    r2_row = df[df['metric'] == 'R²']
    if r2_row['value'].iloc[0] < 0.1:
        issues.append("Weak signal detected!")

    return issues
```

---

## Best Practices

### 1. Run After Every Training

Always run diagnostics after training a new model:
```bash
# After training
./snr_diagnostics label-learnability --symbol BTCUSDT --timeframe 15m
./snr_diagnostics model-robustness --symbol BTCUSDT --timeframe 15m
```

### 2. Version Control Reports

Save reports to git for tracking:
```bash
git add outcomes/label_learnability_*
git commit -m "Add diagnostics for BTCUSDT 15m"
```

### 3. Compare Over Time

Track metric trends:
```python
import pandas as pd
from pathlib import Path

# Load all historical reports
reports = sorted(Path('outcomes').glob('label_learnability_BTCUSDT_15m_*/label_learnability_report_*.csv'))

r2_over_time = []
for report in reports:
    df = pd.read_csv(report)
    r2 = df[df['metric'] == 'R²']['value'].iloc[0]
    timestamp = report.parent.name.split('_')[-1]
    r2_over_time.append({'timestamp': timestamp, 'r2': r2})

# Plot trend
df_trend = pd.DataFrame(r2_over_time)
df_trend.plot(x='timestamp', y='r2')
```

### 4. Set Thresholds

Define acceptable ranges:
```python
THRESHOLDS = {
    'min_r2': 0.15,
    'max_pvalue': 0.05,
    'min_baseline_improvement': 0.05,
    'max_aleatoric_fraction': 0.7,
    'min_icc': 0.4
}

def validate_diagnostics(csv_path):
    """Ensure diagnostics meet thresholds."""
    df = pd.read_csv(csv_path)
    # ... check each threshold ...
```

---

## References

- **Phase 1 Documentation**: `src/utils/ml_common/diagnostics/README.md`
- **Complete Implementation**: `SNR_DIAGNOSTICS_ALL_PHASES_COMPLETE.md`
- **Source Code**: `src/utils/ml_common/diagnostics/snr_cli.py`

---

## Support

For issues or questions:
1. Check interpretation guidelines in reports
2. Review visualization outputs
3. Consult phase documentation
4. Open issue in repository

---

**Version**: 1.0.0
**Last Updated**: 2025-11-18
**CLI Location**: `/home/user/Ares/snr_diagnostics`
