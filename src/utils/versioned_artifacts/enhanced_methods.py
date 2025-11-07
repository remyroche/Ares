"""
Enhanced methods for VersionedArtifactStore

Additional capabilities:
1. Column tagging and retrieval by operation
2. Multi-timeframe data handling with forward-fill
"""

from typing import Optional, Union, List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from .store import VersionedArtifactStore
from .view import ArtifactView
from .view_mask import ViewMask
from .changelog import ChangeType


# Extend VersionedArtifactStore with new methods
def add_columns_with_tags(
    self,
    columns: Dict[str, np.ndarray],
    version_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None
) -> ArtifactView:
    """
    Add new columns to existing version with metadata tags.

    This allows you to later retrieve columns by operation or tags.

    Args:
        columns: Dict of column_name -> values
        version_name: Version to update (None = current)
        tags: Metadata tags for these columns
        operation_name: Name of the operation adding these columns
                       (e.g., "final_feature_selection", "technical_indicators")

    Returns:
        ArtifactView of updated data

    Example:
        # Add 60 features from final feature selection
        store.add_columns_with_tags(
            columns=feature_dict,
            operation_name="final_feature_selection",
            tags={"feature_type": "selected", "source": "feature_selection_step"}
        )

        # Later, retrieve only these columns
        features = store.get_columns_by_operation("final_feature_selection")
    """
    if version_name is None:
        version_name = self._metadata.get('current_version')

    # First, add columns using existing method
    view = self.add_columns(columns, version_name)

    # Store column metadata with tags
    if 'column_metadata' not in self._metadata['versions'][version_name]:
        self._metadata['versions'][version_name]['column_metadata'] = {}

    column_metadata = self._metadata['versions'][version_name]['column_metadata']

    # Add metadata for each column
    for col_name in columns.keys():
        column_metadata[col_name] = {
            'added_at': pd.Timestamp.now().isoformat(),
            'operation': operation_name,
            'tags': tags or {},
            'size': len(columns[col_name])
        }

    self._save_metadata()

    # Record change with operation name
    self.changelog.record_change(
        change_type=ChangeType.UPDATE_COLUMNS,
        version_name=version_name,
        affected_columns=list(columns.keys()),
        metadata={
            'operation': operation_name,
            'tags': tags or {},
            'num_columns': len(columns)
        },
        description=f"Added {len(columns)} columns from operation '{operation_name}'"
    )

    self.logger.info(f"Added {len(columns)} columns with operation '{operation_name}' to version '{version_name}'")

    return view


def get_columns_by_operation(
    self,
    operation_name: str,
    version_name: Optional[str] = None
) -> List[str]:
    """
    Get all columns that were added by a specific operation.

    Args:
        operation_name: Name of the operation
        version_name: Version to query (None = current)

    Returns:
        List of column names

    Example:
        # Get the 60 features from final_feature_selection
        feature_columns = store.get_columns_by_operation("final_feature_selection")

        # Create view with only these columns
        view = store.get_view().select_columns(feature_columns)
        features_df = view.materialize()
    """
    if version_name is None:
        version_name = self._metadata.get('current_version')

    version_meta = self._metadata['versions'].get(version_name, {})
    column_metadata = version_meta.get('column_metadata', {})

    matching_columns = []
    for col_name, col_meta in column_metadata.items():
        if col_meta.get('operation') == operation_name:
            matching_columns.append(col_name)

    return matching_columns


def get_columns_by_tag(
    self,
    tag_key: str,
    tag_value: Any,
    version_name: Optional[str] = None
) -> List[str]:
    """
    Get all columns with a specific tag value.

    Args:
        tag_key: Tag key to search
        tag_value: Tag value to match
        version_name: Version to query (None = current)

    Returns:
        List of column names

    Example:
        # Get all columns tagged as 'technical_indicators'
        tech_cols = store.get_columns_by_tag("type", "technical_indicators")
    """
    if version_name is None:
        version_name = self._metadata.get('current_version')

    version_meta = self._metadata['versions'].get(version_name, {})
    column_metadata = version_meta.get('column_metadata', {})

    matching_columns = []
    for col_name, col_meta in column_metadata.items():
        tags = col_meta.get('tags', {})
        if tags.get(tag_key) == tag_value:
            matching_columns.append(col_name)

    return matching_columns


def get_view_by_operation(
    self,
    operation_name: str,
    version_name: Optional[str] = None
) -> ArtifactView:
    """
    Get a view containing only columns from a specific operation.

    Args:
        operation_name: Name of the operation
        version_name: Version to query (None = current)

    Returns:
        ArtifactView with only the operation's columns

    Example:
        # Get view with only final feature selection columns
        features_view = store.get_view_by_operation("final_feature_selection")
        features_df = features_view.materialize()
    """
    columns = self.get_columns_by_operation(operation_name, version_name)

    if not columns:
        raise ValueError(f"No columns found for operation '{operation_name}'")

    mask = ViewMask(column_mask=set(columns))
    return self.get_view(version_name, mask)


def add_multi_timeframe_data(
    self,
    base_data: pd.DataFrame,
    higher_tf_data: Dict[str, pd.DataFrame],
    version_name: str,
    forward_fill: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> ArtifactView:
    """
    Add data from multiple timeframes, using lower timeframe as base
    and forward-filling higher timeframe data.

    Args:
        base_data: Base data with lowest timeframe (e.g., 15m)
        higher_tf_data: Dict of {timeframe: DataFrame} for higher timeframes
                       (e.g., {"1h": hourly_df, "4h": four_hour_df})
        version_name: Name for this version
        forward_fill: Whether to forward-fill higher timeframe data
        metadata: Additional metadata

    Returns:
        ArtifactView of combined data

    Example:
        # Base data at 15m
        base_15m = pd.DataFrame({'close': [...], 'volume': [...]}, index=dates_15m)

        # Higher timeframe data
        hourly_data = pd.DataFrame({'trend': [...], 'regime': [...]}, index=dates_1h)
        four_hour_data = pd.DataFrame({'macro_trend': [...]}, index=dates_4h)

        # Combine with forward-fill
        view = store.add_multi_timeframe_data(
            base_data=base_15m,
            higher_tf_data={
                "1h": hourly_data,
                "4h": four_hour_data
            },
            version_name="multi_tf_features"
        )

        # Result: 15m data with 1h and 4h features forward-filled
    """
    # Start with base data
    combined = base_data.copy()

    # Track which columns come from which timeframe
    column_timeframes = {col: 'base' for col in base_data.columns}

    # Add higher timeframe data
    for tf, tf_data in higher_tf_data.items():
        # Ensure datetime index
        if not isinstance(combined.index, pd.DatetimeIndex):
            raise ValueError("Base data must have DatetimeIndex")
        if not isinstance(tf_data.index, pd.DatetimeIndex):
            raise ValueError(f"Data for timeframe '{tf}' must have DatetimeIndex")

        # Rename columns to include timeframe suffix
        tf_data_renamed = tf_data.copy()
        tf_data_renamed.columns = [f"{col}_{tf}" for col in tf_data.columns]

        # Reindex to base timeframe
        if forward_fill:
            # Use reindex with forward-fill method
            tf_aligned = tf_data_renamed.reindex(
                combined.index,
                method='ffill'  # Forward-fill
            )
        else:
            # Just reindex without filling
            tf_aligned = tf_data_renamed.reindex(combined.index)

        # Merge with combined data
        combined = pd.concat([combined, tf_aligned], axis=1)

        # Track timeframe for each column
        for col in tf_data_renamed.columns:
            column_timeframes[col] = tf

    # Add combined data to store
    metadata = metadata or {}
    metadata['timeframes'] = list(higher_tf_data.keys())
    metadata['base_timeframe'] = 'base'
    metadata['forward_filled'] = forward_fill
    metadata['column_timeframes'] = column_timeframes

    view = self.add_data(
        data=combined,
        version_name=version_name,
        metadata=metadata
    )

    self.logger.info(
        f"Added multi-timeframe data: base + {len(higher_tf_data)} higher timeframes "
        f"({', '.join(higher_tf_data.keys())})"
    )

    return view


def get_columns_by_timeframe(
    self,
    timeframe: str,
    version_name: Optional[str] = None
) -> List[str]:
    """
    Get all columns from a specific timeframe.

    Args:
        timeframe: Timeframe identifier (e.g., "1h", "4h", "base")
        version_name: Version to query (None = current)

    Returns:
        List of column names

    Example:
        # Get only 1h columns
        hourly_cols = store.get_columns_by_timeframe("1h")
        hourly_view = store.get_view().select_columns(hourly_cols)
    """
    if version_name is None:
        version_name = self._metadata.get('current_version')

    version_meta = self._metadata['versions'].get(version_name, {})
    column_timeframes = version_meta.get('column_timeframes', {})

    matching_columns = [
        col for col, tf in column_timeframes.items()
        if tf == timeframe
    ]

    return matching_columns


# Monkey-patch the methods onto VersionedArtifactStore
VersionedArtifactStore.add_columns_with_tags = add_columns_with_tags
VersionedArtifactStore.get_columns_by_operation = get_columns_by_operation
VersionedArtifactStore.get_columns_by_tag = get_columns_by_tag
VersionedArtifactStore.get_view_by_operation = get_view_by_operation
VersionedArtifactStore.add_multi_timeframe_data = add_multi_timeframe_data
VersionedArtifactStore.get_columns_by_timeframe = get_columns_by_timeframe
