# SNR Diagnostics CLI - Compatibility Guide & Usage Instructions

## Compatibility Review

### Current State of Artifact System

**Investigation Date**: 2025-11-18

#### 1. Artifact Storage Architecture

Ares uses a **dual artifact storage system**:

1. **Legacy Flat Storage** (`artifacts/` directory)
   - Stores JSON metadata files only
   - Example: `feature_generation_labeling_integration_step_labeled_data_ETHUSDT_15m_metadata_long_Analyst_20251029_224702.json`
   - Metadata contains `file_path` pointing to subdirectories
   - Actual data stored in: `artifacts/pre_training/long/Analyst/{step_name}/{filename}.parquet`

2. **Versioned Artifacts** (`versioned_artifacts/` directory)
   - Structure: `{SYMBOL}_{exchange}_{timeframe}_{direction}_{model}/`
   - Example: `ETHUSDT_binance_1h_long_regime_alpha/`
   - Uses HDF5 storage (`store.h5`) with JSON metadata tracking
   - Designed for version control and efficient updates

#### 2. Current Data Availability

**Status**:
- ✅ Metadata files exist in `artifacts/` and `versioned_artifacts/`
- ⚠️ **No actual data files found** (no .parquet, .pkl, or store.h5 files)
- **Reason**: Pipeline steps haven't been run yet to generate actual training data

**Verified Metadata** (from `versioned_artifacts/ETHUSDT_binance_1h_long_regime_alpha/metadata.json`):
- Symbol: ETHUSDT
- Timeframe: 15m (despite "1h" in artifact name)
- Direction: long
- Model: regime_alpha
- Artifacts tracked:
  - `hmm_alpha_training_data_1h` (720 rows, 24 columns including `alpha_target`)
  - `hmm_alpha_regime_stats_1h` (3 rows, regime statistics)
  - `hmm_alpha_feature_pipeline_1h` (feature transformations)

#### 3. hmm_ml_alpha_step Artifacts

**Artifact Names**:
- Training data: `hmm_alpha_training_data_1h`
- Model: `hmm_alpha_model_1h` (LightGBM model)
- Feature pipeline: `hmm_alpha_feature_pipeline_1h`
- Regime statistics: `hmm_alpha_regime_stats_1h`

**Label Column**: `alpha_target` (continuous regression target)

**Features**: 24 columns including:
- Market data: open, high, low, close, volume
- Regime features: regime_label, regime_0-4_prob
- Technical features: hl_range, mean_return_short, return_1h, volatility_short, etc.

#### 4. CLI Compatibility Status

**Current CLI Implementation** (`src/utils/ml_common/diagnostics/snr_cli.py`):

```python
def load_latest_artifacts(self, symbol: str, timeframe: str, model_type: str = 'analyst'):
    # Searches for: artifacts/labeled_data_{symbol}_{timeframe}_*.parquet
    # Loads model from: artifacts/{model_type}_model_{symbol}_{timeframe}_*.pkl
```

**Compatibility Issues**:

| Issue | Current Behavior | Required Fix |
|-------|-----------------|--------------|
| **Flat path assumption** | Searches `artifacts/*.parquet` | Must search subdirectories |
| **Naming mismatch** | Looks for `labeled_data_{symbol}_{timeframe}` | hmm_ml_alpha uses `hmm_alpha_training_data_1h` |
| **No versioned support** | Doesn't use VersionedArtifactStore | Must add versioned artifact loading |
| **Label column name** | Assumes `meta_label` or `label` | hmm_ml_alpha uses `alpha_target` |

---

## Required CLI Updates

### 1. Update Artifact Loading Logic

The CLI needs to be updated to:

1. **Search subdirectories** for parquet files (not just root `artifacts/`)
2. **Support versioned artifacts** by reading from VersionedArtifactStore
3. **Handle multiple label column names**: `meta_label`, `label`, `target`, `alpha_target`
4. **Support hmm_ml_alpha naming**: `hmm_alpha_training_data_1h`, `hmm_alpha_model_1h`

### 2. Proposed Update (High-Level)

```python
def load_latest_artifacts(self, symbol: str, timeframe: str, model_type: str = 'analyst'):
    """Load latest artifacts from either flat or versioned storage."""

    # Try versioned artifacts first
    data, model = self._load_from_versioned_artifacts(symbol, timeframe, model_type)
    if data is not None:
        return data, model

    # Fallback to flat artifacts
    data, model = self._load_from_flat_artifacts(symbol, timeframe, model_type)
    return data, model

def _load_from_versioned_artifacts(self, symbol, timeframe, model_type):
    """Load from versioned artifact store."""
    # Construct store path
    store_path = f"versioned_artifacts/{symbol}_binance_{timeframe}_long_{model_type}"
    if not Path(store_path).exists():
        return None, None

    # Initialize VersionedArtifactStore
    store = VersionedArtifactStore(store_path)

    # Load latest version of training data
    # Use artifact names like: hmm_alpha_training_data_1h
    # Return DataFrame and model

def _load_from_flat_artifacts(self, symbol, timeframe, model_type):
    """Load from flat artifact directory."""
    # Search subdirectories recursively
    pattern = f"**/*{symbol}*{timeframe}*.parquet"
    # Load and return
```

---

## How to Call the CLI (Current State)

### ⚠️ Important Prerequisites

**Before running the CLI, you must**:

1. **Run the pipeline to generate artifacts**:
   ```bash
   # Run meta-labeling step to generate labeled data
   python -m src.training.launchers.your_launcher --step feature_generation_meta_labeling

   # Or run hmm_ml_alpha_step to generate alpha training data
   python -m src.training.launchers.your_launcher --step hmm_ml_alpha
   ```

2. **Verify data files exist**:
   ```bash
   # Check for parquet files
   find artifacts -name "*.parquet" -type f

   # Or check versioned artifacts
   ls -la versioned_artifacts/ETHUSDT_binance_*/store.h5
   ```

3. **If no data files exist**, the CLI will:
   - ✅ Generate a simple Random Forest model for diagnostics
   - ✅ Compute cross-validated predictions
   - ⚠️ But cannot use your actual trained model

---

## CLI Usage Examples (Once Data Exists)

### Basic Usage

```bash
cd /home/user/Ares

# Make CLI executable (one-time)
chmod +x snr_diagnostics

# Run label quality diagnostics
./snr_diagnostics label-quality --symbol ETHUSDT --timeframe 15m

# Run label learnability diagnostics
./snr_diagnostics label-learnability --symbol ETHUSDT --timeframe 15m

# Run model robustness diagnostics
./snr_diagnostics model-robustness --symbol ETHUSDT --timeframe 15m
```

### With Custom Model Type

```bash
# For regime_alpha model
./snr_diagnostics label-quality --symbol ETHUSDT --timeframe 1h --model-type regime_alpha

# For analyst model (default)
./snr_diagnostics label-learnability --symbol ETHUSDT --timeframe 15m --model-type analyst
```

### Check Available Options

```bash
./snr_diagnostics --help
./snr_diagnostics label-quality --help
./snr_diagnostics label-learnability --help
./snr_diagnostics model-robustness --help
```

---

## Expected Outputs

### Directory Structure

After running diagnostics, you'll find outputs in:

```
outcomes/
├── label_quality_ETHUSDT_15m_20251118_120000/
│   ├── label_quality_report_ETHUSDT_15m_20251118_120000.csv
│   ├── label_quality_report_ETHUSDT_15m_20251118_120000.md
│   ├── noise_ceiling_analysis.png
│   ├── uncertainty_decomposition.png
│   └── uncertainty_calibration.png
│
├── label_learnability_ETHUSDT_15m_20251118_120500/
│   ├── label_learnability_report_ETHUSDT_15m_20251118_120500.csv
│   └── label_learnability_report_ETHUSDT_15m_20251118_120500.md
│
└── model_robustness_ETHUSDT_15m_20251118_121000/
    ├── model_robustness_report_ETHUSDT_15m_20251118_121000.csv
    ├── model_robustness_report_ETHUSDT_15m_20251118_121000.md
    ├── model_family_comparison.png
    └── temporal_analysis.png
```

### CSV Report Format

```csv
category,metric,value,interpretation
Core Metrics,R²,0.3245,Moderate signal
Core Metrics,SNR,0.4821,Weak to moderate signal
Core Metrics,Permutation p-value,0.0123,Statistically significant
Bootstrap CI,CI Lower,0.2891,95% confidence interval
Bootstrap CI,CI Upper,0.3599,95% confidence interval
```

---

## Integration with Pipeline

### Automated Diagnostics After Training

Add to your training script:

```python
import subprocess

def run_snr_diagnostics(symbol: str, timeframe: str, model_type: str = 'analyst'):
    """Run all SNR diagnostics after training."""

    commands = [
        f"./snr_diagnostics label-quality --symbol {symbol} --timeframe {timeframe} --model-type {model_type}",
        f"./snr_diagnostics label-learnability --symbol {symbol} --timeframe {timeframe} --model-type {model_type}",
        f"./snr_diagnostics model-robustness --symbol {symbol} --timeframe {timeframe} --model-type {model_type}",
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(f"Success: {result.stdout}")

# Example usage
run_snr_diagnostics("ETHUSDT", "15m", "analyst")
run_snr_diagnostics("ETHUSDT", "1h", "regime_alpha")
```

---

## Troubleshooting

### Error: "No labeled data found"

**Cause**: No parquet files found in artifacts directory

**Solutions**:
1. Run the pipeline to generate data first
2. Check if data is in versioned_artifacts instead
3. Verify symbol/timeframe match exactly

### Error: "No label column found"

**Cause**: Data doesn't have expected label columns

**Solutions**:
1. Check your data has one of: `meta_label`, `label`, `target`, `alpha_target`
2. Manually specify label column (future CLI enhancement)
3. Re-run labeling step

### Error: "Could not load model"

**Expected Behavior**: CLI will train a simple Random Forest for diagnostics

**To Use Your Model**:
1. Ensure model is pickled in artifacts
2. Naming: `{model_type}_model_{symbol}_{timeframe}_*.pkl`
3. Or stored in versioned artifacts as `hmm_alpha_model_1h`

---

## Recommended Next Steps

### For Immediate Use (After Running Pipeline):

1. **Run meta-labeling step** to generate labeled data
2. **Train analyst model** to get model artifacts
3. **Run CLI diagnostics** on generated artifacts
4. **Review reports** in `outcomes/` directory

### For hmm_ml_alpha Integration:

1. **Update CLI** to support versioned artifacts (see Required CLI Updates above)
2. **Add `alpha_target` to label column detection**
3. **Test with hmm_ml_alpha artifacts** once generated
4. **Document model-specific usage** in this guide

---

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| CLI Implementation | ✅ Complete | Fully functional for flat artifacts |
| Flat Artifacts Support | ✅ Ready | Needs pipeline run to generate data |
| Versioned Artifacts Support | ⚠️ Partial | Metadata exists, needs CLI update |
| hmm_ml_alpha Compatibility | ⚠️ Partial | Needs naming convention updates |
| Documentation | ✅ Complete | SNR_CLI_GUIDE.md + this guide |
| Testing | ⏳ Pending | Waiting for artifact generation |

**Current Recommendation**:
1. Run your pipeline to generate actual data files
2. Test CLI with generated artifacts
3. Report any issues for CLI updates
4. I can then update CLI to fully support versioned artifacts

---

**Last Updated**: 2025-11-18
**CLI Version**: 1.0.0
**Ares Branch**: `claude/add-snr-diagnostics-0129tJ8tt6Wd6KGh35VzeNGo`
