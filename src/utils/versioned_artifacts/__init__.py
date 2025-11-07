"""
Versioned Artifact Management System - HDF5-Based Tabular Data Storage

A specialized storage system for versioned tabular data (DataFrames) used in
machine learning pipelines, providing efficient storage, versioning, and view-based access.

RESPONSIBILITIES:
----------------
1. Feature DataFrame Storage:
   - ML features for training
   - Engineered features with multiple columns
   - Large tabular datasets requiring versioning

2. ML Predictions & Scores:
   - Model predictions
   - Probability scores
   - Evaluation metrics over time

3. Training Data Management:
   - Versioned training datasets
   - Feature selection iterations
   - Dataset snapshots for reproducibility

4. View-Based Access:
   - Lazy loading (load only needed columns/rows)
   - Row/column masking
   - Efficient subset queries without full data load

5. Version Tracking:
   - Change logs for all modifications
   - Row-level versioning
   - Column operation tagging
   - Reproducible data states

STORAGE FORMAT:
--------------
- Uses HDF5 (columnar storage like Parquet)
- Efficient compression and chunking
- Stored in versioned_artifacts/ directory
- Organized by context (symbol/exchange/timeframe/direction/model)

WHEN TO USE:
-----------
- Use this for: Feature DataFrames, training data, ML predictions
- Use serialization_utils for: Configs, models, metadata
- Use kline_parquet.py for: Historical OHLCV data

Main Components:
- VersionedArtifactStore: Main storage container
- ArtifactView: Lightweight reference to rows/columns
- ChangeLog: Comprehensive change tracking
- RowVersionTracker: Row-level version management
- ViewMask: Boolean masks for selection

Usage:
    from src.utils.versioned_artifacts import VersionedArtifactStore

    # Store versioned features
    store = VersionedArtifactStore("versioned_artifacts/ETHUSDT_binance_15m_long_analyst")
    view = store.add_data(features_df, version_name="features_v1")

    # Load specific columns efficiently
    subset = view.select_columns(["feature_1", "feature_2"]).materialize()

    # Query by time range without loading full dataset
    data = store.query_by_index_range(start_time, end_time, columns=["predictions"])
"""

from .store import VersionedArtifactStore
from .view import ArtifactView, CombinedView
from .changelog import ChangeLog, ChangeType
from .row_version_tracker import RowVersionTracker, RowVersion
from .view_mask import ViewMask
from .base_step_adapter import VersionedArtifactAdapter

# Import enhanced methods to automatically patch VersionedArtifactStore
from . import enhanced_methods  # noqa: F401

__all__ = [
    "VersionedArtifactStore",
    "ArtifactView",
    "CombinedView",
    "ChangeLog",
    "ChangeType",
    "RowVersionTracker",
    "RowVersion",
    "ViewMask",
    "VersionedArtifactAdapter",
]

__version__ = "1.0.0"
