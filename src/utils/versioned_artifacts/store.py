"""
VersionedArtifactStore - Main unified storage container

Provides single-file storage with versioning, views, and comprehensive
change tracking as an alternative to traditional artifact management.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import json
import logging

from .view import ArtifactView, CombinedView
from .view_mask import ViewMask
from .changelog import ChangeLog, ChangeType
from .row_version_tracker import RowVersionTracker


class VersionedArtifactStore:
    """
    Unified artifact storage with versioning and views.

    Core Features:
    - Single HDF5 file for all data
    - View-based access (no full loads)
    - Row and column level versioning
    - Comprehensive change tracking
    - Space-efficient delta storage

    Storage Structure:
    - store.h5: Main HDF5 file with all data
    - changelog/: Change tracking files
    - versions/: Row version tracking
    - metadata.json: Store-level metadata
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        auto_version: bool = True,
        enable_row_versioning: bool = True,
        compression: str = "gzip",
        compression_level: int = 4
    ):
        """
        Initialize versioned artifact store.

        Args:
            store_path: Path to store directory
            auto_version: Automatically create versions on updates
            enable_row_versioning: Enable row-level version tracking
            compression: Compression algorithm for HDF5
            compression_level: Compression level (1-9)
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.auto_version = auto_version
        self.enable_row_versioning = enable_row_versioning
        self.compression = compression
        self.compression_level = compression_level

        # Core files
        self.h5_file = self.store_path / "store.h5"
        self.metadata_file = self.store_path / "metadata.json"

        # Components
        self.changelog = ChangeLog(self.store_path / "changelog")
        if enable_row_versioning:
            self.row_tracker = RowVersionTracker(self.store_path / "versions")
        else:
            self.row_tracker = None

        # Logger
        self.logger = logging.getLogger(f"VersionedArtifactStore.{self.store_path.name}")

        # Metadata
        self._metadata = self._load_metadata()

        # Initialize HDF5 file if needed
        if not self.h5_file.exists():
            self._init_h5_file()

    def _init_h5_file(self) -> None:
        """Initialize new HDF5 file."""
        with h5py.File(self.h5_file, 'w') as f:
            # Create root groups
            f.create_group('versions')
            f.create_group('_metadata')

            # Store creation metadata
            f.attrs['created_at'] = datetime.now().isoformat()
            f.attrs['store_version'] = '1.0.0'

        self.logger.info(f"Initialized new store at {self.store_path}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load store metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'versions': {},
            'current_version': None,
            'created_at': datetime.now().isoformat()
        }

    def _save_metadata(self) -> None:
        """Save store metadata."""
        self._metadata['updated_at'] = datetime.now().isoformat()

        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)

    def add_data(
        self,
        data: pd.DataFrame,
        version_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArtifactView:
        """
        Add new data to the store.

        Args:
            data: DataFrame to add
            version_name: Name for this version
            metadata: Additional metadata

        Returns:
            ArtifactView referencing the new data
        """
        from src.utils.tprint import tprint

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Extract context information from metadata if available
        meta = metadata or {}
        context_parts = []
        if 'symbol' in meta and 'exchange' in meta:
            context_parts.append(f"{meta['symbol']}/{meta['exchange']}")
        if 'timeframe' in meta:
            context_parts.append(f"[{meta['timeframe']}]")
        if 'direction' in meta and 'model' in meta:
            context_parts.append(f"{meta['direction']}/{meta['model']}")

        context_str = " ".join(context_parts) if context_parts else self.store_path.name
        tprint(f"💾 Adding data to store '{version_name}': {len(data)} rows × {len(data.columns)} cols | {context_str}")

        with h5py.File(self.h5_file, 'a') as f:
            versions_group = f['versions']

            # Check if version already exists
            if version_name in versions_group:
                raise ValueError(f"Version '{version_name}' already exists")

            # Create version group
            version_group = versions_group.create_group(version_name)

            # Store data
            for column in data.columns:
                version_group.create_dataset(
                    column,
                    data=data[column].values,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )

            # Store index
            if isinstance(data.index, pd.DatetimeIndex):
                index_data = data.index.astype(np.int64).values
                version_group.create_dataset(
                    '_index',
                    data=index_data,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
                version_group.attrs['index_type'] = 'datetime'
            else:
                version_group.create_dataset(
                    '_index',
                    data=data.index.values,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )
                version_group.attrs['index_type'] = 'default'

            # Store metadata
            version_group.attrs['created_at'] = datetime.now().isoformat()
            version_group.attrs['num_rows'] = len(data)
            version_group.attrs['num_columns'] = len(data.columns)

        # Update store metadata
        self._metadata['versions'][version_name] = {
            'created_at': datetime.now().isoformat(),
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': list(data.columns),
            **(metadata or {})
        }
        self._metadata['current_version'] = version_name
        self._save_metadata()

        # Record change
        self.changelog.record_change(
            change_type=ChangeType.ADD_DATA,
            version_name=version_name,
            affected_rows=len(data),
            affected_columns=list(data.columns),
            metadata=metadata
        )

        self.logger.info(f"Added data version '{version_name}': {len(data)} rows, {len(data.columns)} columns")
        tprint(f"✅ Successfully added version '{version_name}' to store | {context_str}")

        # Create and return view
        return self.get_view(version_name)

    def get_view(
        self,
        version_name: Optional[str] = None,
        mask: Optional[ViewMask] = None
    ) -> ArtifactView:
        """
        Get a view of the data.

        Args:
            version_name: Version to view (None = current)
            mask: Optional mask to apply

        Returns:
            ArtifactView instance
        """
        if version_name is None:
            version_name = self._metadata.get('current_version')
            if version_name is None:
                raise ValueError("No versions available")

        if version_name not in self._metadata['versions']:
            raise ValueError(f"Version '{version_name}' not found")

        return ArtifactView(
            store=self,
            version_name=version_name,
            mask=mask,
            metadata=self._metadata['versions'][version_name].copy()
        )

    def _load_data_with_mask(
        self,
        version_name: str,
        mask: ViewMask
    ) -> pd.DataFrame:
        """
        Load data with mask applied.

        Args:
            version_name: Version to load
            mask: ViewMask to apply

        Returns:
            Filtered DataFrame
        """
        from src.utils.tprint import tprint

        # Get context from version metadata if available
        version_meta = self._metadata.get('versions', {}).get(version_name, {})
        context_parts = []
        if 'symbol' in version_meta and 'exchange' in version_meta:
            context_parts.append(f"{version_meta['symbol']}/{version_meta['exchange']}")
        if 'timeframe' in version_meta:
            context_parts.append(f"[{version_meta['timeframe']}]")
        if 'direction' in version_meta and 'model' in version_meta:
            context_parts.append(f"{version_meta['direction']}/{version_meta['model']}")

        context_str = " ".join(context_parts) if context_parts else self.store_path.name

        with h5py.File(self.h5_file, 'r') as f:
            version_group = f['versions'][version_name]

            # Load index
            index_data = version_group['_index'][:]
            index_type = version_group.attrs.get('index_type', 'default')

            if index_type == 'datetime':
                index = pd.to_datetime(index_data)
            else:
                index = index_data

            # Determine columns to load
            all_columns = [k for k in version_group.keys() if not k.startswith('_')]
            if mask.column_mask:
                columns_to_load = [c for c in all_columns if c in mask.column_mask]
            else:
                columns_to_load = all_columns

            tprint(f"📂 Loading {len(columns_to_load)}/{len(all_columns)} columns from '{version_name}' | {context_str}")

            # Load data
            data_dict = {}
            for column in columns_to_load:
                data_dict[column] = version_group[column][:]

            # Create DataFrame
            df = pd.DataFrame(data_dict, index=index)

            # Apply row mask
            if mask.row_mask is not None:
                original_len = len(df)
                df = df[mask.row_mask]
                tprint(f"✂️ Row mask applied: {len(df)}/{original_len} rows retained | {context_str}")

            return df

    def update_rows(
        self,
        row_indices: Union[List[int], np.ndarray],
        columns: List[str],
        new_values: Union[pd.DataFrame, Dict[str, np.ndarray]],
        version_name: Optional[str] = None,
        create_new_version: bool = False,
        new_version_name: Optional[str] = None
    ) -> ArtifactView:
        """
        Update specific rows without rewriting entire columns.

        Args:
            row_indices: Rows to update
            columns: Columns to update
            new_values: New values (DataFrame or dict of arrays)
            version_name: Version to update (None = current)
            create_new_version: Create a new version instead of updating
            new_version_name: Name for new version if creating

        Returns:
            ArtifactView of updated data
        """
        if version_name is None:
            version_name = self._metadata.get('current_version')

        if create_new_version:
            if new_version_name is None:
                new_version_name = f"{version_name}_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Load full data
            full_data = self._load_data_with_mask(version_name, ViewMask())

            # Apply updates
            if isinstance(new_values, pd.DataFrame):
                for col in columns:
                    full_data.loc[full_data.index[row_indices], col] = new_values[col].values
            else:
                for col in columns:
                    full_data.loc[full_data.index[row_indices], col] = new_values[col]

            # Add as new version
            return self.add_data(full_data, new_version_name)
        else:
            # Update in place
            with h5py.File(self.h5_file, 'a') as f:
                version_group = f['versions'][version_name]

                for col in columns:
                    dataset = version_group[col]
                    if isinstance(new_values, pd.DataFrame):
                        dataset[row_indices] = new_values[col].values
                    else:
                        dataset[row_indices] = new_values[col]

            # Track row versions if enabled
            if self.enable_row_versioning:
                for idx in row_indices:
                    changes = {}
                    for col in columns:
                        if isinstance(new_values, pd.DataFrame):
                            changes[col] = new_values[col].iloc[idx]
                        else:
                            changes[col] = new_values[col][idx]

                    self.row_tracker.create_version(idx, changes)

            # Record change
            self.changelog.record_change(
                change_type=ChangeType.UPDATE_ROWS,
                version_name=version_name,
                affected_rows=list(row_indices) if len(row_indices) < 100 else len(row_indices),
                affected_columns=columns
            )

            self.logger.info(f"Updated {len(row_indices)} rows in version '{version_name}'")

            return self.get_view(version_name)

    def add_columns(
        self,
        columns: Dict[str, np.ndarray],
        version_name: Optional[str] = None
    ) -> ArtifactView:
        """
        Add new columns to existing version.

        Args:
            columns: Dict of column_name -> values
            version_name: Version to update (None = current)

        Returns:
            ArtifactView of updated data
        """
        from src.utils.tprint import tprint

        if version_name is None:
            version_name = self._metadata.get('current_version')

        # Get context from version metadata if available
        version_meta = self._metadata.get('versions', {}).get(version_name, {})
        context_parts = []
        if 'symbol' in version_meta and 'exchange' in version_meta:
            context_parts.append(f"{version_meta['symbol']}/{version_meta['exchange']}")
        if 'timeframe' in version_meta:
            context_parts.append(f"[{version_meta['timeframe']}]")
        if 'direction' in version_meta and 'model' in version_meta:
            context_parts.append(f"{version_meta['direction']}/{version_meta['model']}")

        context_str = " ".join(context_parts) if context_parts else self.store_path.name
        tprint(f"➕ Adding {len(columns)} columns to version '{version_name}' | {context_str}")

        with h5py.File(self.h5_file, 'a') as f:
            version_group = f['versions'][version_name]

            for col_name, col_data in columns.items():
                if col_name in version_group:
                    raise ValueError(f"Column '{col_name}' already exists")

                version_group.create_dataset(
                    col_name,
                    data=col_data,
                    compression=self.compression,
                    compression_opts=self.compression_level
                )

        # Update metadata
        version_meta = self._metadata['versions'][version_name]
        version_meta['columns'].extend(columns.keys())
        version_meta['num_columns'] = len(version_meta['columns'])
        self._save_metadata()

        # Record change
        self.changelog.record_change(
            change_type=ChangeType.UPDATE_COLUMNS,
            version_name=version_name,
            affected_columns=list(columns.keys())
        )

        self.logger.info(f"Added {len(columns)} columns to version '{version_name}'")
        tprint(f"✅ Added {len(columns)} columns to version '{version_name}' | {context_str}")

        return self.get_view(version_name)

    def combine_views(
        self,
        views: List[ArtifactView],
        strategy: str = "merge",
        **kwargs
    ) -> CombinedView:
        """
        Combine multiple views.

        Args:
            views: List of ArtifactView instances
            strategy: Combination strategy ('merge', 'concat', 'join')
            **kwargs: Additional arguments for combination

        Returns:
            CombinedView instance
        """
        return CombinedView(views, strategy=strategy, **kwargs)

    def list_versions(self) -> List[str]:
        """
        List all available versions.

        Returns:
            List of version names
        """
        return list(self._metadata['versions'].keys())

    def get_version_info(self, version_name: str) -> Dict[str, Any]:
        """
        Get information about a version.

        Args:
            version_name: Version name

        Returns:
            Version metadata
        """
        return self._metadata['versions'].get(version_name, {})

    def get_changelog(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        version_name: Optional[str] = None
    ) -> List:
        """
        Get change log entries.

        Args:
            from_time: Start time filter
            to_time: End time filter
            version_name: Filter by version

        Returns:
            List of change records
        """
        return self.changelog.get_changes(
            from_time=from_time,
            to_time=to_time,
            version_name=version_name
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'store_path': str(self.store_path),
            'num_versions': len(self._metadata['versions']),
            'current_version': self._metadata.get('current_version'),
            'created_at': self._metadata.get('created_at'),
            'h5_file_size_mb': self.h5_file.stat().st_size / (1024 * 1024) if self.h5_file.exists() else 0
        }

        # Add changelog stats
        changelog_stats = self.changelog.get_statistics()
        stats['changelog'] = changelog_stats

        # Add row version stats if enabled
        if self.enable_row_versioning:
            row_stats = self.row_tracker.get_statistics()
            stats['row_versioning'] = row_stats

        return stats

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"VersionedArtifactStore(path='{self.store_path}', versions={stats['num_versions']}, current='{stats['current_version']}')"
