"""
Versioned Artifact Management System

An alternative to the traditional artifact management approach, providing:
- Single-file unified storage instead of multiple artifact files
- View-based access with row/column masks instead of full data loads
- Comprehensive change tracking at row and column level
- Row-level versioning without full column rewrites
- Space-efficient storage through shared columns and delta encoding

Main Components:
- VersionedArtifactStore: Main storage container
- ArtifactView: Lightweight reference to rows/columns
- ChangeLog: Comprehensive change tracking
- RowVersionTracker: Row-level version management
- ViewMask: Boolean masks for selection

Usage:
    from src.utils.versioned_artifacts import VersionedArtifactStore

    store = VersionedArtifactStore("versioned_artifacts/ETHUSDT_binance_long")
    view = store.add_data(df, version_name="v1")
    subset = view.select_rows(mask).select_columns(["close", "predictions"])
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
