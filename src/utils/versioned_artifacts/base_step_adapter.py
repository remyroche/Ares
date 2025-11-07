"""
BaseStep Adapter for VersionedArtifactStore

Provides compatibility layer between VersionedArtifactStore and the
existing BaseStep artifact management interface.
"""

from typing import Optional, Any, Dict
from pathlib import Path
import pandas as pd

from .store import VersionedArtifactStore
from .view import ArtifactView


class VersionedArtifactAdapter:
    """
    Adapter to use VersionedArtifactStore with BaseStep.

    Provides the same interface as ArtifactManager but uses
    VersionedArtifactStore internally.

    Usage:
        # In BaseStep.__init__
        self.versioned_store = VersionedArtifactAdapter(
            store_dir="versioned_artifacts",
            symbol=symbol,
            exchange=exchange
        )

        # Use like normal artifact manager
        self.versioned_store.save(data, "predictions")
        data = self.versioned_store.get("predictions")
    """

    def __init__(
        self,
        store_dir: str = "versioned_artifacts",
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        direction: str = "long",
        model: str = "analyst"
    ):
        """
        Initialize adapter.

        Args:
            store_dir: Base directory for versioned stores
            symbol: Trading symbol
            exchange: Exchange name
            direction: Trading direction
            model: Model type
        """
        self.store_dir = Path(store_dir)
        self.symbol = symbol
        self.exchange = exchange
        self.direction = direction
        self.model = model

        # Create store path with context
        if symbol and exchange:
            store_name = f"{symbol}_{exchange}_{direction}_{model}"
        else:
            store_name = "default"

        store_path = self.store_dir / store_name

        # Initialize store
        self.store = VersionedArtifactStore(
            store_path=store_path,
            auto_version=True,
            enable_row_versioning=True
        )

        # Track artifact name to version name mapping
        self._artifact_versions: Dict[str, str] = {}

    def save(
        self,
        data: Any,
        artifact_name: str,
        artifact_type: str = "data",
        compression: str = "auto",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save artifact (compatible with ArtifactManager.save).

        Args:
            data: Data to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            compression: Compression method (ignored, uses store settings)
            metadata: Additional metadata

        Returns:
            Path or identifier for saved artifact
        """
        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            else:
                # For non-DataFrame data, wrap in a simple structure
                data = pd.DataFrame({'value': [data]})

        # Generate version name
        from datetime import datetime
        version_name = f"{artifact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Add to store
        view = self.store.add_data(
            data=data,
            version_name=version_name,
            metadata={
                'artifact_name': artifact_name,
                'artifact_type': artifact_type,
                **(metadata or {})
            }
        )

        # Track mapping
        self._artifact_versions[artifact_name] = version_name

        return str(self.store.store_path / f"{version_name}.h5")

    def get_artifact(
        self,
        artifact_name: str,
        artifact_type: str = "data",
        return_path: bool = False
    ) -> Any:
        """
        Retrieve artifact (compatible with ArtifactManager.get_artifact).

        Args:
            artifact_name: Name of artifact to retrieve
            artifact_type: Type of artifact
            return_path: Whether to return path along with data

        Returns:
            Retrieved data or (data, path) tuple
        """
        # Get version name for this artifact
        version_name = self._artifact_versions.get(artifact_name)

        if version_name is None:
            # Try to find latest version with this artifact name
            versions = self.store.list_versions()
            matching = [v for v in versions if artifact_name in v]

            if not matching:
                if return_path:
                    return None, None
                return None

            # Use most recent matching version
            version_name = sorted(matching)[-1]
            self._artifact_versions[artifact_name] = version_name

        # Get view and materialize
        try:
            view = self.store.get_view(version_name)
            data = view.materialize()

            # Handle unwrapping for single-value artifacts
            if 'value' in data.columns and len(data.columns) == 1:
                if len(data) == 1:
                    data = data['value'].iloc[0]

            if return_path:
                path = str(self.store.store_path / f"{version_name}.h5")
                return data, path
            else:
                return data

        except Exception as e:
            if return_path:
                return None, None
            return None

    def save_artifact(self, *args, **kwargs):
        """Alias for save()."""
        return self.save(*args, **kwargs)

    def update_rows(
        self,
        artifact_name: str,
        row_indices: list,
        columns: list,
        new_values: Any
    ) -> str:
        """
        Update specific rows in an artifact.

        Args:
            artifact_name: Name of artifact
            row_indices: Rows to update
            columns: Columns to update
            new_values: New values

        Returns:
            Path or identifier
        """
        version_name = self._artifact_versions.get(artifact_name)
        if not version_name:
            raise ValueError(f"Artifact '{artifact_name}' not found")

        view = self.store.update_rows(
            row_indices=row_indices,
            columns=columns,
            new_values=new_values,
            version_name=version_name
        )

        return str(self.store.store_path / f"{version_name}.h5")

    def get_view(
        self,
        artifact_name: str,
        row_mask: Optional[Any] = None,
        columns: Optional[list] = None
    ) -> ArtifactView:
        """
        Get a view of an artifact.

        Args:
            artifact_name: Name of artifact
            row_mask: Row selection mask
            columns: Columns to include

        Returns:
            ArtifactView instance
        """
        from .view_mask import ViewMask

        version_name = self._artifact_versions.get(artifact_name)
        if not version_name:
            raise ValueError(f"Artifact '{artifact_name}' not found")

        # Create mask
        mask = ViewMask(row_mask=row_mask, column_mask=set(columns) if columns else None)

        return self.store.get_view(version_name, mask)

    def combine_artifacts(
        self,
        artifact_names: list,
        strategy: str = "merge"
    ) -> pd.DataFrame:
        """
        Combine multiple artifacts.

        Args:
            artifact_names: List of artifact names
            strategy: Combination strategy

        Returns:
            Combined DataFrame
        """
        views = []
        for name in artifact_names:
            version_name = self._artifact_versions.get(name)
            if version_name:
                views.append(self.store.get_view(version_name))

        if not views:
            raise ValueError("No valid artifacts found")

        combined = self.store.combine_views(views, strategy=strategy)
        return combined.materialize()

    def get_changelog(self, artifact_name: Optional[str] = None):
        """
        Get change log for artifact(s).

        Args:
            artifact_name: Specific artifact name (None = all)

        Returns:
            List of change records
        """
        if artifact_name:
            version_name = self._artifact_versions.get(artifact_name)
            return self.store.get_changelog(version_name=version_name)
        else:
            return self.store.get_changelog()

    def get_statistics(self) -> Dict:
        """Get store statistics."""
        return self.store.get_statistics()

    def __repr__(self) -> str:
        """String representation."""
        return f"VersionedArtifactAdapter(symbol={self.symbol}, exchange={self.exchange}, artifacts={len(self._artifact_versions)})"
