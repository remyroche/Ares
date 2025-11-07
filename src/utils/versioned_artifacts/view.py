"""
ArtifactView - Lightweight reference to artifact data

Provides lazy evaluation and efficient access to subsets of data
without loading the full dataset into memory.
"""

from typing import Optional, Union, List, Dict, Any, Callable
import pandas as pd
import numpy as np
from pathlib import Path

from .view_mask import ViewMask


class ArtifactView:
    """
    Lightweight reference to artifact data with lazy evaluation.

    Instead of loading full data, maintains a reference and only loads
    data when explicitly accessed. Supports chaining operations to build
    complex queries before materializing results.

    Features:
    - Lazy evaluation (data loaded only when needed)
    - Method chaining for queries
    - Memory-efficient subsetting
    - Integration with ViewMask
    """

    def __init__(
        self,
        store: 'VersionedArtifactStore',  # Forward reference
        version_name: str,
        mask: Optional[ViewMask] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize artifact view.

        Args:
            store: Parent VersionedArtifactStore instance
            version_name: Version this view references
            mask: ViewMask for row/column selection
            metadata: Additional metadata
        """
        self._store = store
        self._version_name = version_name
        self._mask = mask or ViewMask()
        self._metadata = metadata or {}
        self._cached_data = None
        self._operations = []  # Queue of operations to apply

    @property
    def version_name(self) -> str:
        """Get version name."""
        return self._version_name

    @property
    def mask(self) -> ViewMask:
        """Get current mask."""
        return self._mask

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata."""
        return self._metadata

    def select_rows(
        self,
        indices: Union[np.ndarray, List[int], slice, Callable]
    ) -> 'ArtifactView':
        """
        Select specific rows.

        Args:
            indices: Row indices, boolean mask, slice, or callable

        Returns:
            New ArtifactView with row selection applied
        """
        if callable(indices):
            # Callable - will be applied when data is loaded
            new_view = self._copy()
            new_view._operations.append(('select_rows', indices))
            return new_view
        else:
            # Static selection - update mask
            new_mask = self._mask.select_rows(indices)
            return ArtifactView(
                store=self._store,
                version_name=self._version_name,
                mask=new_mask,
                metadata=self._metadata.copy()
            )

    def select_columns(self, columns: Union[List[str], Callable]) -> 'ArtifactView':
        """
        Select specific columns.

        Args:
            columns: Column names or callable

        Returns:
            New ArtifactView with column selection applied
        """
        if callable(columns):
            # Callable - will be applied when data is loaded
            new_view = self._copy()
            new_view._operations.append(('select_columns', columns))
            return new_view
        else:
            # Static selection - update mask
            new_mask = self._mask.select_columns(columns)
            return ArtifactView(
                store=self._store,
                version_name=self._version_name,
                mask=new_mask,
                metadata=self._metadata.copy()
            )

    def filter(self, condition: Callable[[pd.DataFrame], np.ndarray]) -> 'ArtifactView':
        """
        Filter rows using a callable condition.

        Args:
            condition: Function that takes DataFrame and returns boolean mask

        Returns:
            New ArtifactView with filter applied
        """
        new_view = self._copy()
        new_view._operations.append(('filter', condition))
        return new_view

    def transform(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> 'ArtifactView':
        """
        Transform data using a callable.

        Args:
            func: Function that takes DataFrame and returns transformed DataFrame

        Returns:
            New ArtifactView with transformation queued
        """
        new_view = self._copy()
        new_view._operations.append(('transform', func))
        return new_view

    def with_metadata(self, **kwargs) -> 'ArtifactView':
        """
        Add metadata to view.

        Args:
            **kwargs: Metadata key-value pairs

        Returns:
            New ArtifactView with updated metadata
        """
        new_metadata = self._metadata.copy()
        new_metadata.update(kwargs)

        return ArtifactView(
            store=self._store,
            version_name=self._version_name,
            mask=self._mask,
            metadata=new_metadata
        )

    def _copy(self) -> 'ArtifactView':
        """Create a copy of this view."""
        new_view = ArtifactView(
            store=self._store,
            version_name=self._version_name,
            mask=self._mask,
            metadata=self._metadata.copy()
        )
        new_view._operations = self._operations.copy()
        new_view._cached_data = self._cached_data
        return new_view

    def materialize(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load and materialize the data.

        This is when actual data loading happens. All queued operations
        are applied in order.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Materialized DataFrame
        """
        # Check cache
        if use_cache and self._cached_data is not None:
            return self._cached_data

        # Load data from store with current mask
        data = self._store._load_data_with_mask(self._version_name, self._mask)

        # Apply queued operations
        for op_type, op_func in self._operations:
            if op_type == 'select_rows':
                row_mask = op_func(data)
                data = data[row_mask]
            elif op_type == 'select_columns':
                columns = op_func(data)
                data = data[columns]
            elif op_type == 'filter':
                row_mask = op_func(data)
                data = data[row_mask]
            elif op_type == 'transform':
                data = op_func(data)

        # Cache if configured
        if use_cache:
            self._cached_data = data

        return data

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert view to pandas DataFrame.

        Alias for materialize().

        Returns:
            Materialized DataFrame
        """
        return self.materialize()

    def to_numpy(self) -> np.ndarray:
        """
        Convert view to numpy array.

        Returns:
            Numpy array of values
        """
        return self.materialize().values

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Get first n rows.

        Args:
            n: Number of rows

        Returns:
            DataFrame with first n rows
        """
        return self.materialize().head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Get last n rows.

        Args:
            n: Number of rows

        Returns:
            DataFrame with last n rows
        """
        return self.materialize().tail(n)

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None) -> pd.DataFrame:
        """
        Get random sample.

        Args:
            n: Number of rows
            frac: Fraction of rows

        Returns:
            DataFrame with sampled rows
        """
        return self.materialize().sample(n=n, frac=frac)

    @property
    def shape(self) -> tuple:
        """
        Get shape of the view.

        This requires materializing the data.

        Returns:
            Tuple of (rows, columns)
        """
        return self.materialize().shape

    @property
    def columns(self) -> List[str]:
        """
        Get column names.

        This requires materializing the data.

        Returns:
            List of column names
        """
        return list(self.materialize().columns)

    def __len__(self) -> int:
        """Get number of rows."""
        return len(self.materialize())

    def __getitem__(self, key):
        """
        Access data using indexing.

        Args:
            key: Index, slice, or column name(s)

        Returns:
            Selected data
        """
        return self.materialize()[key]

    def describe(self) -> pd.DataFrame:
        """
        Get statistical description.

        Returns:
            DataFrame with statistics
        """
        return self.materialize().describe()

    def info(self) -> None:
        """Print information about the view."""
        print(f"ArtifactView: {self._version_name}")
        print(f"Mask: {self._mask}")
        print(f"Pending operations: {len(self._operations)}")
        print(f"Cached: {self._cached_data is not None}")

        if self._metadata:
            print("\nMetadata:")
            for key, value in self._metadata.items():
                print(f"  {key}: {value}")

    def clear_cache(self) -> None:
        """Clear cached data."""
        self._cached_data = None

    def persist(self, path: Union[str, Path]) -> None:
        """
        Persist materialized data to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        data = self.materialize()

        if path.suffix == '.parquet':
            data.to_parquet(path, compression='snappy')
        elif path.suffix == '.csv':
            data.to_csv(path, index=True)
        elif path.suffix == '.pkl':
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def __repr__(self) -> str:
        """String representation."""
        mask_info = str(self._mask)
        ops_info = f"{len(self._operations)} pending ops" if self._operations else "no pending ops"
        cache_info = "cached" if self._cached_data is not None else "not cached"

        return f"ArtifactView(version='{self._version_name}', {mask_info}, {ops_info}, {cache_info})"


class CombinedView(ArtifactView):
    """
    View that combines multiple ArtifactViews.

    Supports different combination strategies:
    - merge: Merge on index (like pd.merge)
    - concat: Concatenate rows (like pd.concat)
    - join: Join on index (like pd.join)
    """

    def __init__(
        self,
        views: List[ArtifactView],
        strategy: str = "merge",
        **kwargs
    ):
        """
        Initialize combined view.

        Args:
            views: List of ArtifactView instances to combine
            strategy: Combination strategy ('merge', 'concat', 'join')
            **kwargs: Additional arguments for combination
        """
        if not views:
            raise ValueError("At least one view required")

        # Use first view's store for compatibility
        super().__init__(
            store=views[0]._store,
            version_name=f"combined_{'_'.join([v.version_name for v in views])}",
            mask=None,
            metadata={'views': [v.version_name for v in views]}
        )

        self._views = views
        self._strategy = strategy
        self._kwargs = kwargs

    def materialize(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Materialize combined data.

        Returns:
            Combined DataFrame
        """
        if use_cache and self._cached_data is not None:
            return self._cached_data

        # Materialize all views
        dataframes = [view.materialize(use_cache=use_cache) for view in self._views]

        # Combine based on strategy
        if self._strategy == "merge":
            result = dataframes[0]
            for df in dataframes[1:]:
                result = pd.merge(
                    result, df,
                    left_index=True, right_index=True,
                    **self._kwargs
                )
        elif self._strategy == "concat":
            result = pd.concat(dataframes, **self._kwargs)
        elif self._strategy == "join":
            result = dataframes[0]
            for df in dataframes[1:]:
                result = result.join(df, **self._kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

        if use_cache:
            self._cached_data = result

        return result
