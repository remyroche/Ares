# Artifact Storage Architecture

## Overview

The Ares system uses a **format-aware, intelligent routing system** for artifact storage. Different types of data are automatically routed to the most appropriate storage format based on their characteristics and use case.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        BaseStep                              │
│                  (Main Artifact Manager)                     │
│                                                              │
│  _save_artifact() / _get_artifact()                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   ArtifactRouter                             │
│            (Intelligent Format Detection)                    │
│                                                              │
│  Detects format based on:                                   │
│  - Data type (DataFrame, dict, model, etc.)                 │
│  - Artifact name keywords                                   │
│  - Explicit data_category hint                              │
└──────┬──────────┬──────────────┬──────────────┬────────────┘
       │          │               │              │
       │          │               │              │
       ▼          ▼               ▼              ▼
┌──────────┐ ┌─────────┐  ┌─────────────┐ ┌──────────────────┐
│   JSON   │ │ Pickle  │  │   Parquet   │ │  HDF5 Versioned  │
│          │ │         │  │  (Klines)   │ │   (Features)     │
└──────────┘ └─────────┘  └─────────────┘ └──────────────────┘
     │            │              │                   │
     ▼            ▼              ▼                   ▼
serialization  serialization  kline_parquet.py  versioned_artifacts/
  _utils.py     _utils.py
```

## Storage Systems & Responsibilities

### 1. JSON Storage (via serialization_utils.py)

**Purpose**: Human-readable configuration and metadata

**Use Cases**:
- Configuration files
- Metadata dictionaries
- Small data structures
- Parameters and settings
- Human-readable debug data

**Storage Location**: `artifacts/*.json`

**Example**:
```python
# Automatic routing
self._save_artifact(config_dict, "training_config")  # → config.json

# Explicit category
self._save_artifact(params, "model_params", data_category="config")  # → JSON
```

**When to Use**:
- ✅ Small dictionaries and lists
- ✅ Configuration data
- ✅ Data that needs human inspection
- ❌ Large datasets
- ❌ Binary data
- ❌ ML models

---

### 2. Pickle Storage (via serialization_utils.py)

**Purpose**: Complex Python objects and ML models

**Use Cases**:
- ML models (sklearn, xgboost, lightgbm, catboost, etc.)
- Complex Python objects
- Arbitrary data structures
- Non-tabular data
- Preprocessors and transformers

**Storage Location**: `artifacts/*.pkl`

**Example**:
```python
# Automatic routing for ML models
self._save_artifact(trained_model, "xgboost_model")  # → .pkl

# Explicit category
self._save_artifact(model, "estimator", data_category="model")  # → Pickle
```

**When to Use**:
- ✅ ML models
- ✅ Complex objects
- ✅ Non-tabular data
- ✅ Custom Python classes
- ❌ Large DataFrames (use Parquet or HDF5)
- ❌ Data requiring versioning

---

### 3. Parquet Storage (via kline_parquet.py)

**Purpose**: Specialized storage for historical OHLCV/time-series data

**Use Cases**:
- Historical candlestick data (OHLCV)
- Raw market data from exchanges
- Time-series data requiring optimization
- Bulk data imports

**Storage Location**: `historical_data/{exchange}/{symbol}/klines/*.parquet`

**Features**:
- ZSTD compression (best compression ratio)
- Batch management for incremental updates
- Data integrity validation (OHLCV relationships)
- Gap detection and handling
- Memory-efficient chunking

**Example**:
```python
# Automatic routing for OHLCV data
ohlcv_df = pd.DataFrame({
    'timestamp': [...],
    'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]
})
self._save_artifact(ohlcv_df, "historical_klines")  # → Parquet

# Explicit category
self._save_artifact(klines_df, "market_data", data_category="historical")  # → Parquet
```

**When to Use**:
- ✅ Historical OHLCV data
- ✅ Raw market data
- ✅ Time-series with temporal ordering
- ✅ Large bulk imports
- ❌ Feature DataFrames (use versioned_artifacts)
- ❌ Data requiring frequent updates
- ❌ Data needing versioning

---

### 4. HDF5 Versioned Storage (via versioned_artifacts/)

**Purpose**: Versioned storage for ML features and training data

**Use Cases**:
- ML feature DataFrames
- Engineered features with many columns
- Training datasets requiring versioning
- Model predictions and scores
- Data requiring view-based access
- Datasets with frequent column additions

**Storage Location**: `versioned_artifacts/{symbol}_{exchange}_{timeframe}_{direction}_{model}/store.h5`

**Features**:
- Columnar storage (like Parquet) via HDF5
- Version tracking and change logs
- Row-level versioning
- View-based access (lazy loading)
- Column operation tagging
- Efficient subset queries
- Reproducibility support

**Example**:
```python
# Automatic routing for feature DataFrames
features_df = pd.DataFrame({
    'feature_1': [...],
    'feature_2': [...],
    # ... many features
})
self._save_artifact(features_df, "training_features")  # → HDF5

# Automatic routing for predictions
predictions_df = pd.DataFrame({
    'prediction': [...],
    'probability': [...],
    'score': [...]
})
self._save_artifact(predictions_df, "ml_predictions")  # → HDF5

# Explicit category
self._save_artifact(data, "features_v2", data_category="features")  # → HDF5
```

**When to Use**:
- ✅ Feature DataFrames with many columns
- ✅ Training data requiring versioning
- ✅ ML predictions and scores
- ✅ Large datasets with partial access needs
- ✅ Data with frequent column additions
- ❌ Historical raw data (use Parquet)
- ❌ Small datasets (use Pickle)
- ❌ ML models (use Pickle)

---

## Routing Decision Flow

**Hierarchy when in doubt: JSON → Pickle → Parquet → HDF5**
(Prefer simpler formats first, move to more complex as needed)

```
Artifact → ArtifactRouter._detect_format()
│
├─ Explicit data_category provided?
│  ├─ "config" / "metadata" / "parameters" / "hpo" → JSON
│  ├─ "model" → Pickle
│  ├─ "historical" / "klines" / "ohlcv" → Parquet
│  └─ "features" / "predictions" / "training" → HDF5
│
├─ Artifact name contains keywords?
│  ├─ JSON: "config", "metadata", "params", "settings", "parameters",
│  │        "hpo", "hyperparameter", "tuning", "grid", "search" → JSON
│  │
│  ├─ Pickle: "model", "estimator", "classifier", "regressor", "ml",
│  │          "base", "ensemble", "sr", "stacked", "voting", "bagging" → Pickle
│  │
│  ├─ Parquet: "historical", "klines", "ohlcv", "candles",
│  │           "market_data", "raw_data" → Parquet
│  │
│  └─ HDF5: "feature", "prediction", "score", "training", "cluster",
│           "label", "target", "regime", "engineered", "selected" → HDF5
│
└─ Data type + complexity analysis (follows hierarchy)
   ├─ Dict → Check JSON serializability
   │  ├─ Simple (serializable) → JSON
   │  └─ Complex (not serializable) → Pickle
   │
   ├─ List/Tuple → Check size and serializability
   │  ├─ Empty or small & simple → JSON
   │  └─ Large or complex → Pickle
   │
   ├─ DataFrame
   │  ├─ Has OHLCV columns → Parquet
   │  ├─ Large (>10 cols, >100 rows) → HDF5
   │  ├─ Medium with HDF5 keywords → HDF5
   │  └─ Small → Pickle
   │
   ├─ ML model object → Pickle
   ├─ NumPy array (small & simple) → JSON
   ├─ NumPy array (large/complex) → Pickle
   ├─ Scalar types → JSON
   └─ Unknown/Complex → Pickle (safe default)
```

## Usage in BaseStep

### Saving Artifacts

```python
class MyStep(BaseStep):
    async def execute(self, config):
        # Automatic format detection
        self._save_artifact(model, "trained_model")  # → Pickle
        self._save_artifact(config_dict, "model_config")  # → JSON
        self._save_artifact(ohlcv_df, "historical_klines")  # → Parquet
        self._save_artifact(features_df, "training_features")  # → HDF5

        # Explicit category (recommended for clarity)
        self._save_artifact(model, "xgb_model", data_category="model")
        self._save_artifact(params, "params", data_category="config")
        self._save_artifact(klines, "ohlcv", data_category="historical")
        self._save_artifact(features, "features", data_category="features")
```

### Loading Artifacts

```python
class MyStep(BaseStep):
    async def execute(self, config):
        # Automatic format detection
        model = self._get_artifact("trained_model")  # Searches all formats
        config_dict = self._get_artifact("model_config")
        ohlcv_df = self._get_artifact("historical_klines")
        features_df = self._get_artifact("training_features")

        # Explicit category (faster, no search needed)
        model = self._get_artifact("model", data_category="model")
        features = self._get_artifact("features", data_category="features")
```

## File Organization

```
project/
├── artifacts/                    # JSON and Pickle files
│   ├── *.json                   # Configuration, metadata
│   └── *.pkl                    # ML models, complex objects
│
├── historical_data/              # Parquet files (OHLCV)
│   └── {exchange}/
│       └── {symbol}/
│           └── klines/
│               └── *.parquet
│
└── versioned_artifacts/          # HDF5 files (Features)
    └── {symbol}_{exchange}_{timeframe}_{direction}_{model}/
        ├── store.h5             # Main HDF5 file
        ├── metadata.json        # Store metadata
        ├── changelog/           # Change tracking
        └── versions/            # Row version tracking
```

## Best Practices

### 1. Use Explicit Categories When Possible

**Good** (explicit):
```python
self._save_artifact(model, "model", data_category="model")
self._save_artifact(features, "features", data_category="features")
```

**Acceptable** (automatic):
```python
self._save_artifact(model, "trained_xgboost_model")  # Auto-detects as model
self._save_artifact(features, "training_features")   # Auto-detects as features
```

### 2. Use Descriptive Artifact Names

Include keywords that indicate the type:
- `"model_*"`, `"*_estimator"` → Auto-routes to Pickle
- `"*_config"`, `"*_metadata"` → Auto-routes to JSON
- `"historical_*"`, `"*_klines"` → Auto-routes to Parquet
- `"*_features"`, `"*_predictions"` → Auto-routes to HDF5

### 3. Separate Raw Data from Features

- **Raw market data** → Parquet (via kline_parquet.py)
- **Engineered features** → HDF5 (via versioned_artifacts/)

### 4. Choose Storage Based on Usage Pattern

| Access Pattern | Storage |
|----------------|---------|
| Full dataset loads | Parquet or Pickle |
| Partial column access | HDF5 (versioned_artifacts) |
| Time-range queries | HDF5 or Parquet |
| Human inspection | JSON |
| ML models | Pickle |

## Migration Guide

### From Old System to New System

**Old Way** (manual routing):
```python
# Manual decision making
if 'historical' in name:
    use_parquet()
elif isinstance(data, pd.DataFrame):
    use_versioned_store()
else:
    use_pickle()
```

**New Way** (automatic routing):
```python
# Just save, router handles it
self._save_artifact(data, name)

# Or be explicit
self._save_artifact(data, name, data_category="features")
```

## Summary

| Format | Use Case | Location | Module |
|--------|----------|----------|--------|
| **JSON** | Configs, metadata | `artifacts/*.json` | serialization_utils.py |
| **Pickle** | ML models, complex objects | `artifacts/*.pkl` | serialization_utils.py |
| **Parquet** | Historical OHLCV data | `historical_data/` | kline_parquet.py |
| **HDF5** | Features, predictions | `versioned_artifacts/` | versioned_artifacts/ |

The system automatically routes artifacts to the most appropriate storage based on:
1. Explicit `data_category` parameter
2. Artifact name keywords
3. Data type and characteristics

This ensures optimal storage format for each type of data while maintaining a simple, unified API.
