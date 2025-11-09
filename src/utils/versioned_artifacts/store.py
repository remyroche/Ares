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
        compression_level: int = 4,
        chunk_rows: Optional[int] = None,
        chunk_cols: Optional[int] = None
    ):
        """
        Initialize versioned artifact store.

        Args:
            store_path: Path to store directory
            auto_version: Automatically create versions on updates
            enable_row_versioning: Enable row-level version tracking
            compression: Compression algorithm for HDF5
            compression_level: Compression level (1-9)
            chunk_rows: Number of rows per chunk (None = auto)
            chunk_cols: Number of columns per chunk (None = auto, typically 1 for column-wise access)
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.auto_version = auto_version
        self.enable_row_versioning = enable_row_versioning
        self.compression = compression
        self.compression_level = compression_level

        # Chunking strategy
        self.chunk_rows = chunk_rows
        self.chunk_cols = chunk_cols

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
        from src.utils.tprint import tprint
        tprint(f"🐛 DEBUG: Loading metadata from {self.metadata_file}", "INFO")
        self._metadata = self._load_metadata()
        tprint(f"🐛 DEBUG: Loaded metadata with {len(self._metadata.get('versions', {}))} versions", "INFO")
        tprint(f"🐛 DEBUG: Current version: {self._metadata.get('current_version')}", "INFO")

        # Initialize HDF5 file if needed
        from src.utils.tprint import tprint
        tprint(f"🐛 DEBUG: Checking HDF5 file existence: {self.h5_file.exists()}", "INFO")
        if not self.h5_file.exists():
            tprint("🐛 DEBUG: HDF5 file does not exist, initializing new file", "INFO")
            self._init_h5_file()
        else:
            tprint(f"🐛 DEBUG: HDF5 file exists, size: {self.h5_file.stat().st_size} bytes", "INFO")

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
        from src.utils.tprint import tprint
        
        tprint(f"🐛 DEBUG: _load_metadata() checking {self.metadata_file}", "INFO")
        if self.metadata_file.exists():
            tprint(f"🐛 DEBUG: Metadata file exists, size: {self.metadata_file.stat().st_size} bytes", "INFO")
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    tprint(f"🐛 DEBUG: Successfully loaded metadata with {len(metadata.get('versions', {}))} versions", "INFO")
                    return metadata
            except Exception as e:
                tprint(f"🐛 DEBUG: Error loading metadata: {e}", "ERROR")
                # Return default metadata on error
                return {
                    'versions': {},
                    'current_version': None,
                    'created_at': datetime.now().isoformat(),
                    'load_error': str(e)
                }
        else:
            tprint("🐛 DEBUG: Metadata file does not exist, returning default metadata", "INFO")
            return {
                'versions': {},
                'current_version': None,
                'created_at': datetime.now().isoformat()
            }

    def _save_metadata(self) -> None:
        """Save store metadata."""
        from src.utils.tprint import tprint
        
        tprint(f"🐛 DEBUG: _save_metadata() called, saving to {self.metadata_file}", "INFO")
        tprint(f"🐛 DEBUG: Metadata has {len(self._metadata.get('versions', {}))} versions", "INFO")
        tprint(f"🐛 DEBUG: Current version in metadata: {self._metadata.get('current_version')}", "INFO")
        
        self._metadata['updated_at'] = datetime.now().isoformat()

        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
            tprint(f"🐛 DEBUG: Successfully saved metadata to {self.metadata_file}", "INFO")
            tprint(f"🐛 DEBUG: New metadata file size: {self.metadata_file.stat().st_size} bytes", "INFO")
        except Exception as e:
            tprint(f"🐛 DEBUG: Error saving metadata: {e}", "ERROR")
            raise

    def _get_context_string(self, metadata: Optional[Dict[str, Any]] = None,
                           version_name: Optional[str] = None) -> str:
        """
        Extract and format context string from metadata.

        Args:
            metadata: Metadata dict to extract context from
            version_name: Version name to lookup metadata from store

        Returns:
            Formatted context string (e.g., "BTCUSDT/binance [15m] long/analyst")
        """
        # Get metadata from version if not provided
        if metadata is None and version_name:
            metadata = self._metadata.get('versions', {}).get(version_name, {})

        meta = metadata or {}
        context_parts = []

        if 'symbol' in meta and 'exchange' in meta:
            context_parts.append(f"{meta['symbol']}/{meta['exchange']}")
        if 'timeframe' in meta:
            context_parts.append(f"[{meta['timeframe']}]")
        if 'direction' in meta and 'model' in meta:
            context_parts.append(f"{meta['direction']}/{meta['model']}")

        return " ".join(context_parts) if context_parts else self.store_path.name

    def _calculate_chunk_shape(self, num_rows: int, num_cols: int) -> Tuple[int, int]:
        """
        Calculate optimal chunk shape for HDF5 storage.

        Args:
            num_rows: Number of rows in dataset
            num_cols: Number of columns in dataset

        Returns:
            Tuple of (chunk_rows, chunk_cols)
        """
        # Use explicit chunk size if provided
        if self.chunk_rows is not None and self.chunk_cols is not None:
            return (self.chunk_rows, self.chunk_cols)

        # Default strategy: optimize for column-wise access (ML features)
        # Store each column separately for efficient loading
        chunk_cols = self.chunk_cols if self.chunk_cols is not None else 1

        # For rows, use reasonable chunk size based on data size
        if self.chunk_rows is not None:
            chunk_rows = self.chunk_rows
        else:
            # Adaptive chunk size based on dataset size
            if num_rows < 10000:
                chunk_rows = min(1000, num_rows)
            elif num_rows < 100000:
                chunk_rows = 5000
            elif num_rows < 1000000:
                chunk_rows = 10000
            else:
                chunk_rows = 50000

        return (chunk_rows, chunk_cols)

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
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        # Get context string for logging
        from src.utils.tprint import tprint
        context_str = self._get_context_string(metadata=metadata)
        tprint(f"💾 Adding data to store '{version_name}': {len(data)} rows × {len(data.columns)} cols | {context_str}")
        tprint(f"🐛 DEBUG: VersionedArtifactStore.add_data() called", "INFO")
        tprint(f"🐛 DEBUG: Store path: {self.h5_file}", "INFO")
        tprint(f"🐛 DEBUG: Version name: {version_name}", "INFO")
        tprint(f"🐛 DEBUG: Data shape: {data.shape}, columns: {list(data.columns)[:10]}...", "INFO")

        with h5py.File(self.h5_file, 'a') as f:
            versions_group = f['versions']
            tprint(f"🐛 DEBUG: Opened HDF5 file, versions_group exists: {'versions' in f}", "INFO")

            # Check if version already exists
            if version_name in versions_group:
                raise ValueError(f"Version '{version_name}' already exists")

            # Create version group
            tprint(f"🐛 DEBUG: Creating version group '{version_name}'", "INFO")
            version_group = versions_group.create_group(version_name)

            # Calculate optimal chunk shape
            chunk_rows, chunk_cols = self._calculate_chunk_shape(len(data), len(data.columns))
            tprint(f"🐛 DEBUG: Calculated chunk shape: ({chunk_rows}, {chunk_cols})", "INFO")

            # Normalize chunk dimensions for 1D column datasets
            chunk_rows = max(1, chunk_rows)
            chunk_cols = max(1, chunk_cols)

            # Store data with chunking
            for i, column in enumerate(data.columns):
                series = data[column]
                column_data = series.to_numpy()

                if pd.api.types.is_datetime64_any_dtype(series):
                    column_data = series.view(np.int64)
                elif pd.api.types.is_bool_dtype(series):
                    column_data = series.astype(np.int8).to_numpy()
                elif pd.api.types.is_categorical_dtype(series):
                    # Convert categorical to string, handling NaN properly
                    column_data = series.astype(str).replace('nan', '').astype('S256').to_numpy()
                elif pd.api.types.is_string_dtype(series) or column_data.dtype == object:
                    column_data = series.fillna('').astype('string').astype('S256').to_numpy()
                elif pd.api.types.is_float_dtype(series):
                    column_data = series.astype(np.float64).to_numpy()
                elif pd.api.types.is_integer_dtype(series):
                    column_data = series.astype(np.int64).to_numpy()
                elif column_data.dtype == object:
                    column_data = series.astype('category').cat.codes.astype(np.int32).to_numpy()

                if column_data.ndim == 1:
                    chunk_shape = (max(1, min(chunk_rows, column_data.shape[0] or chunk_rows)),)
                else:
                    chunk_shape = (chunk_rows, chunk_cols)

                try:
                    tprint(f"🐛 DEBUG: Storing column {i+1}/{len(data.columns)}: '{column}' (dtype: {column_data.dtype}, shape: {column_data.shape})", "INFO")
                    version_group.create_dataset(
                        column,
                        data=column_data,
                        compression=self.compression,
                        compression_opts=self.compression_level,
                        chunks=chunk_shape
                    )
                except TypeError as err:
                    tprint(f"🐛 DEBUG: Failed to store column '{column}': {err}", "ERROR")
                    raise TypeError(
                        f"Failed to store column '{column}' with dtype {column_data.dtype}: {err}"
                    ) from err

            # Store index with consistent chunking
            tprint(f"🐛 DEBUG: Storing index (type: {type(data.index)})", "INFO")
            index_chunk_shape = (max(1, min(chunk_rows, len(data))),)

            if isinstance(data.index, pd.DatetimeIndex):
                index_data = data.index.astype(np.int64).values
                version_group.create_dataset(
                    '_index',
                    data=index_data,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    chunks=index_chunk_shape
                )
                version_group.attrs['index_type'] = 'datetime'
            else:
                version_group.create_dataset(
                    '_index',
                    data=data.index.values,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    chunks=index_chunk_shape
                )
                version_group.attrs['index_type'] = 'default'

            # Store metadata
            version_group.attrs['created_at'] = datetime.now().isoformat()
            version_group.attrs['num_rows'] = len(data)
            version_group.attrs['num_columns'] = len(data.columns)
            tprint(f"🐛 DEBUG: Stored version metadata", "INFO")

        # Update store metadata
        tprint(f"🐛 DEBUG: Updating store metadata", "INFO")
        self._metadata['versions'][version_name] = {
            'created_at': datetime.now().isoformat(),
            'num_rows': len(data),
            'num_columns': len(data.columns),
            'columns': list(data.columns),
            **(metadata or {})
        }
        self._metadata['current_version'] = version_name
        self._save_metadata()
        tprint(f"🐛 DEBUG: Saved store metadata to {self.metadata_file}", "INFO")

        # Record change
        tprint(f"🐛 DEBUG: Recording change in changelog", "INFO")
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
        tprint(f"🐛 DEBUG: Creating view for version '{version_name}'", "INFO")
        view = self.get_view(version_name)
        tprint(f"🐛 DEBUG: Created view: {view}", "INFO")
        return view

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
        # Get context string for logging
        from src.utils.tprint import tprint
        context_str = self._get_context_string(version_name=version_name)

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

            # Load data efficiently using dict comprehension
            data_dict = {col: version_group[col][:] for col in columns_to_load}

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

        # Get context string for logging
        context_str = self._get_context_string(version_name=version_name)
        tprint(f"➕ Adding {len(columns)} columns to version '{version_name}' | {context_str}")

        with h5py.File(self.h5_file, 'a') as f:
            version_group = f['versions'][version_name]

            # Get chunk size from existing columns or calculate new
            existing_cols = [k for k in version_group.keys() if not k.startswith('_')]
            if existing_cols:
                chunks = version_group[existing_cols[0]].chunks
            else:
                # Calculate chunk size for new columns
                num_rows = len(next(iter(columns.values())))
                chunk_rows, chunk_cols = self._calculate_chunk_shape(num_rows, 1)
                chunks = (chunk_rows, chunk_cols)

            for col_name, col_data in columns.items():
                if col_name in version_group:
                    raise ValueError(f"Column '{col_name}' already exists")

                version_group.create_dataset(
                    col_name,
                    data=col_data,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    chunks=chunks
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
        from src.utils.tprint import tprint
        
        tprint(f"🐛 DEBUG: list_versions() called", "INFO")
        tprint(f"🐛 DEBUG: Metadata file exists: {self.metadata_file.exists()}", "INFO")
        tprint(f"🐛 DEBUG: HDF5 file exists: {self.h5_file.exists()}", "INFO")
        
        # Reload metadata to ensure we have the latest
        tprint("🐛 DEBUG: Reloading metadata to get latest versions", "INFO")
        self._metadata = self._load_metadata()
        
        versions = list(self._metadata['versions'].keys())
        tprint(f"🐛 DEBUG: Found {len(versions)} versions in metadata: {versions}", "INFO")
        
        # Also check HDF5 file directly
        if self.h5_file.exists():
            try:
                with h5py.File(self.h5_file, 'r') as f:
                    if 'versions' in f:
                        h5_versions = list(f['versions'].keys())
                        tprint(f"🐛 DEBUG: Found {len(h5_versions)} versions in HDF5: {h5_versions}", "INFO")
                        
                        # Check for discrepancies
                        if set(versions) != set(h5_versions):
                            tprint(f"🐛 DEBUG: MISMATCH between metadata and HDF5!", "ERROR")
                            tprint(f"🐛 DEBUG: Metadata only: {set(versions) - set(h5_versions)}", "ERROR")
                            tprint(f"🐛 DEBUG: HDF5 only: {set(h5_versions) - set(versions)}", "ERROR")
                    else:
                        tprint("🐛 DEBUG: No 'versions' group found in HDF5 file!", "ERROR")
            except Exception as e:
                tprint(f"🐛 DEBUG: Error reading HDF5 file: {e}", "ERROR")
        
        return versions

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

    def query_by_index_range(
        self,
        start_idx: Any,
        end_idx: Any,
        version_name: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Efficiently query data by index range without loading full dataset.

        Args:
            start_idx: Start index value (inclusive)
            end_idx: End index value (inclusive)
            version_name: Version to query (None = current)
            columns: Specific columns to load (None = all)

        Returns:
            Filtered DataFrame

        Example:
            # Query time range for datetime index
            data = store.query_by_index_range(
                start_idx=pd.Timestamp('2024-01-01'),
                end_idx=pd.Timestamp('2024-01-31'),
                columns=['close', 'volume']
            )
        """
        from src.utils.tprint import tprint

        if version_name is None:
            version_name = self._metadata.get('current_version')

        context_str = self._get_context_string(version_name=version_name)
        tprint(f"🔍 Querying index range [{start_idx} to {end_idx}] from '{version_name}' | {context_str}")

        with h5py.File(self.h5_file, 'r') as f:
            version_group = f['versions'][version_name]

            # Load index
            index_data = version_group['_index'][:]
            index_type = version_group.attrs.get('index_type', 'default')

            if index_type == 'datetime':
                index = pd.to_datetime(index_data)
                # Convert query values to timestamps for comparison
                if isinstance(start_idx, (str, pd.Timestamp)):
                    start_idx = pd.Timestamp(start_idx)
                if isinstance(end_idx, (str, pd.Timestamp)):
                    end_idx = pd.Timestamp(end_idx)
            else:
                index = index_data

            # Find matching indices efficiently
            mask = (index >= start_idx) & (index <= end_idx)
            matching_indices = np.where(mask)[0]

            if len(matching_indices) == 0:
                tprint(f"⚠️ No data found in range [{start_idx} to {end_idx}] | {context_str}")
                return pd.DataFrame()

            # Determine columns to load
            all_columns = [k for k in version_group.keys() if not k.startswith('_')]
            columns_to_load = columns if columns else all_columns

            tprint(f"📂 Loading {len(columns_to_load)} columns, {len(matching_indices)} rows | {context_str}")

            # Load only matched rows for selected columns (efficient slicing)
            data_dict = {}
            for col in columns_to_load:
                # HDF5 supports fancy indexing for efficient row selection
                data_dict[col] = version_group[col][matching_indices]

            # Create DataFrame with filtered index
            result = pd.DataFrame(data_dict, index=index[mask])

            tprint(f"✅ Query returned {len(result)} rows | {context_str}")
            return result

    def add_columns_batch(
        self,
        column_groups: List[Dict[str, np.ndarray]],
        version_name: Optional[str] = None
    ) -> ArtifactView:
        """
        Batch add multiple column groups efficiently.

        This reduces HDF5 file open/close overhead by grouping all column
        additions into a single transaction.

        Args:
            column_groups: List of dicts, each containing column_name -> values
            version_name: Version to update (None = current)

        Returns:
            ArtifactView of updated data

        Example:
            # Add multiple feature groups in one batch
            store.add_columns_batch([
                {"feature_1": values1, "feature_2": values2},  # Group 1
                {"feature_3": values3, "feature_4": values4},  # Group 2
                {"feature_5": values5}                          # Group 3
            ])
        """
        from src.utils.tprint import tprint

        if version_name is None:
            version_name = self._metadata.get('current_version')

        # Flatten all columns
        all_columns = {}
        for group in column_groups:
            all_columns.update(group)

        context_str = self._get_context_string(version_name=version_name)
        tprint(f"📦 Batch adding {len(all_columns)} columns in {len(column_groups)} groups | {context_str}")

        # Add all columns in single HDF5 operation
        with h5py.File(self.h5_file, 'a') as f:
            version_group = f['versions'][version_name]

            # Get chunk size from first column in version
            existing_cols = [k for k in version_group.keys() if not k.startswith('_')]
            if existing_cols:
                first_col_chunks = version_group[existing_cols[0]].chunks
            else:
                # Calculate chunk size if no existing columns
                num_rows = len(next(iter(all_columns.values())))
                chunk_rows, chunk_cols = self._calculate_chunk_shape(num_rows, 1)
                first_col_chunks = (chunk_rows, chunk_cols)

            for col_name, col_data in all_columns.items():
                if col_name in version_group:
                    raise ValueError(f"Column '{col_name}' already exists")

                version_group.create_dataset(
                    col_name,
                    data=col_data,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                    chunks=first_col_chunks
                )

        # Update metadata
        version_meta = self._metadata['versions'][version_name]
        version_meta['columns'].extend(all_columns.keys())
        version_meta['num_columns'] = len(version_meta['columns'])
        self._save_metadata()

        # Record change
        self.changelog.record_change(
            change_type=ChangeType.UPDATE_COLUMNS,
            version_name=version_name,
            affected_columns=list(all_columns.keys()),
            metadata={'batch_groups': len(column_groups)}
        )

        self.logger.info(f"Batch added {len(all_columns)} columns to version '{version_name}'")
        tprint(f"✅ Batch added {len(all_columns)} columns | {context_str}")

        return self.get_view(version_name)

    def replace_column(
        self,
        column_name: str,
        new_values: np.ndarray,
        version_name: Optional[str] = None
    ) -> ArtifactView:
        """
        Replace entire column with new values.

        Args:
            column_name: Name of column to replace
            new_values: New values for the column
            version_name: Version to update (None = current)

        Returns:
            ArtifactView of updated data
        """
        from src.utils.tprint import tprint

        if version_name is None:
            version_name = self._metadata.get('current_version')

        context_str = self._get_context_string(version_name=version_name)
        tprint(f"🔄 Replacing column '{column_name}' in version '{version_name}' | {context_str}")

        with h5py.File(self.h5_file, 'a') as f:
            version_group = f['versions'][version_name]

            if column_name not in version_group:
                raise ValueError(f"Column '{column_name}' does not exist")

            # Delete old column and create new one
            del version_group[column_name]

            # Get chunk size from another column
            remaining_cols = [k for k in version_group.keys() if not k.startswith('_')]
            if remaining_cols:
                chunks = version_group[remaining_cols[0]].chunks
            else:
                chunk_rows, chunk_cols = self._calculate_chunk_shape(len(new_values), 1)
                chunks = (chunk_rows, chunk_cols)

            version_group.create_dataset(
                column_name,
                data=new_values,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=chunks
            )

        # Record change
        self.changelog.record_change(
            change_type=ChangeType.UPDATE_COLUMNS,
            version_name=version_name,
            affected_columns=[column_name],
            metadata={'operation': 'replace'}
        )

        self.logger.info(f"Replaced column '{column_name}' in version '{version_name}'")
        tprint(f"✅ Replaced column '{column_name}' | {context_str}")

        return self.get_view(version_name)

    def replace_rows(
        self,
        row_indices: Union[List[int], np.ndarray],
        new_data: pd.DataFrame,
        version_name: Optional[str] = None
    ) -> ArtifactView:
        """
        Replace entire rows with new data.

        Args:
            row_indices: Indices of rows to replace
            new_data: DataFrame with new values (must have same columns as version)
            version_name: Version to update (None = current)

        Returns:
            ArtifactView of updated data
        """
        from src.utils.tprint import tprint

        if version_name is None:
            version_name = self._metadata.get('current_version')

        context_str = self._get_context_string(version_name=version_name)
        tprint(f"🔄 Replacing {len(row_indices)} rows in version '{version_name}' | {context_str}")

        # Update all columns for specified rows
        with h5py.File(self.h5_file, 'a') as f:
            version_group = f['versions'][version_name]

            for col in new_data.columns:
                if col not in version_group:
                    raise ValueError(f"Column '{col}' not found in version")

                dataset = version_group[col]
                dataset[row_indices] = new_data[col].values

        # Record change
        self.changelog.record_change(
            change_type=ChangeType.UPDATE_ROWS,
            version_name=version_name,
            affected_rows=list(row_indices) if len(row_indices) < 100 else len(row_indices),
            affected_columns=list(new_data.columns),
            metadata={'operation': 'replace'}
        )

        self.logger.info(f"Replaced {len(row_indices)} rows in version '{version_name}'")
        tprint(f"✅ Replaced {len(row_indices)} rows | {context_str}")

        return self.get_view(version_name)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Dictionary with statistics
        """
        from src.utils.tprint import tprint
        
        tprint(f"🐛 DEBUG: get_statistics() called", "INFO")
        
        # Reload metadata to ensure we have the latest
        tprint("🐛 DEBUG: Reloading metadata for statistics", "INFO")
        self._metadata = self._load_metadata()
        
        stats = {
            'store_path': str(self.store_path),
            'num_versions': len(self._metadata['versions']),
            'current_version': self._metadata.get('current_version'),
            'created_at': self._metadata.get('created_at'),
            'h5_file_size_mb': self.h5_file.stat().st_size / (1024 * 1024) if self.h5_file.exists() else 0,
            'chunking': {
                'chunk_rows': self.chunk_rows or 'auto',
                'chunk_cols': self.chunk_cols or 'auto'
            }
        }
        
        tprint(f"🐛 DEBUG: Statistics - num_versions: {stats['num_versions']}", "INFO")
        tprint(f"🐛 DEBUG: Statistics - current_version: {stats['current_version']}", "INFO")
        tprint(f"🐛 DEBUG: Statistics - h5_file_size_mb: {stats['h5_file_size_mb']}", "INFO")

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
