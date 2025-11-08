"""
BaseStep Adapter for VersionedArtifactStore

Provides compatibility layer between VersionedArtifactStore and the
existing BaseStep artifact management interface.
"""

from typing import Optional, Any, Dict, List
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
        timeframe: str = "15m",
        direction: str = "long",
        model: str = "analyst"
    ):
        """
        Initialize adapter with full context separation.

        Args:
            store_dir: Base directory for versioned stores
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe for data (default: '15m')
            direction: Trading direction - 'long' or 'short' (default: 'long')
            model: Model type - 'analyst' or 'tactician' (default: 'analyst')
        """
        self.store_dir = Path(store_dir)
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        self.model = model

        # Create store path with full context separation
        # Format: {symbol}_{exchange}_{timeframe}_{direction}_{model}
        if symbol and exchange:
            store_name = f"{symbol}_{exchange}_{timeframe}_{direction}_{model}"
        else:
            store_name = f"default_{timeframe}_{direction}_{model}"

        from src.utils.tprint import tprint
        store_path = self.store_dir / store_name
        tprint(f"🐛 DEBUG: VersionedArtifactAdapter initializing store at {store_path}", "INFO")

        # Initialize store with context metadata
        self.store = VersionedArtifactStore(
            store_path=store_path,
            auto_version=True,
            enable_row_versioning=True
        )
        tprint(f"🐛 DEBUG: VersionedArtifactStore initialized", "INFO")

        # Store context in store metadata for reference
        if hasattr(self.store, '_metadata'):
            tprint("🐛 DEBUG: Storing context in store metadata", "INFO")
            self.store._metadata['context'] = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction,
                'model': model
            }
            self.store._save_metadata()
            tprint("🐛 DEBUG: Context saved to store metadata", "INFO")

        # Track artifact name to version name mapping
        self._artifact_versions: Dict[str, str] = {}
        tprint("🐛 DEBUG: VersionedArtifactAdapter initialization complete", "INFO")

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
        from datetime import datetime
        from src.utils.tprint import tprint

        # Log context information
        from src.utils.tprint import tprint
        context_str = f"{self.symbol}/{self.exchange} [{self.timeframe}] {self.direction}/{self.model}"
        tprint(f"💾 Saving '{artifact_name}' to versioned store: {context_str}")

        # Convert to DataFrame if needed
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            else:
                # For non-DataFrame data, wrap in a simple structure
                data = pd.DataFrame({'value': [data]})

        # Generate version name
        version_name = f"{artifact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare metadata with full context
        artifact_metadata = {
            'artifact_name': artifact_name,
            'artifact_type': artifact_type,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'direction': self.direction,
            'model': self.model,
            'created_at': datetime.now().isoformat(),
            **(metadata or {})
        }

        # Add to store
        tprint(f"🐛 DEBUG: Adding data to store with version_name={version_name}", "INFO")
        view = self.store.add_data(
            data=data,
            version_name=version_name,
            metadata=artifact_metadata
        )
        tprint(f"🐛 DEBUG: Store.add_data() returned view: {view}", "INFO")

        # Track mapping
        self._artifact_versions[artifact_name] = version_name
        tprint(f"🐛 DEBUG: Tracked artifact mapping: {artifact_name} -> {version_name}", "INFO")

        artifact_path = str(self.store.store_path / f"{version_name}.h5")
        tprint(f"✅ Saved '{artifact_name}' [{context_str}] to {artifact_path}")

        # Verify the save was successful
        tprint("🐛 DEBUG: Verifying save was successful...", "INFO")
        versions = self.store.list_versions()
        tprint(f"🐛 DEBUG: Store now has {len(versions)} versions: {versions}", "INFO")
        
        return artifact_path

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
        from src.utils.tprint import tprint

        # Log context information
        from src.utils.tprint import tprint
        context_str = f"{self.symbol}/{self.exchange} [{self.timeframe}] {self.direction}/{self.model}"
        tprint(f"📂 Retrieving '{artifact_name}' from versioned store: {context_str}")

        # Get version name for this artifact
        version_name = self._artifact_versions.get(artifact_name)
        tprint(f"🐛 DEBUG: Looking up artifact '{artifact_name}', found version_name: {version_name}", "INFO")

        if version_name is None:
            # Try to find latest version with this artifact name
            tprint("🐛 DEBUG: Artifact not in cache, searching store versions...", "INFO")
            versions = self.store.list_versions()
            tprint(f"🐛 DEBUG: Store has {len(versions)} versions: {versions}", "INFO")
            matching = [v for v in versions if artifact_name in v]
            tprint(f"🐛 DEBUG: Found {len(matching)} matching versions: {matching}", "INFO")

            if not matching:
                tprint(f"⚠️ Artifact '{artifact_name}' not found in versioned store [{context_str}]")
                if return_path:
                    return None, None
                return None

            # Use most recent matching version
            version_name = sorted(matching)[-1]
            self._artifact_versions[artifact_name] = version_name
            tprint(f"🐛 DEBUG: Selected latest version: {version_name}", "INFO")

        # Get view and materialize
        try:
            view = self.store.get_view(version_name)
            data = view.materialize()

            # Handle unwrapping for single-value artifacts
            if 'value' in data.columns and len(data.columns) == 1:
                if len(data) == 1:
                    data = data['value'].iloc[0]

            tprint(f"✅ Retrieved '{artifact_name}' [{context_str}] (version: {version_name})")

            if return_path:
                path = str(self.store.store_path / f"{version_name}.h5")
                return data, path
            else:
                return data

        except Exception as e:
            tprint(f"❌ Failed to retrieve '{artifact_name}' [{context_str}]: {e}")
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
        from src.utils.tprint import tprint

        from src.utils.tprint import tprint
        context_str = f"{self.symbol}/{self.exchange} [{self.timeframe}] {self.direction}/{self.model}"
        tprint(f"🔄 Updating {len(row_indices)} rows in '{artifact_name}' [{context_str}]")

        version_name = self._artifact_versions.get(artifact_name)
        if not version_name:
            raise ValueError(f"Artifact '{artifact_name}' not found")

        view = self.store.update_rows(
            row_indices=row_indices,
            columns=columns,
            new_values=new_values,
            version_name=version_name
        )

        tprint(f"✅ Updated {len(row_indices)} rows in '{artifact_name}' [{context_str}]")
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
        from src.utils.tprint import tprint

        from src.utils.tprint import tprint
        context_str = f"{self.symbol}/{self.exchange} [{self.timeframe}] {self.direction}/{self.model}"
        tprint(f"🔗 Combining {len(artifact_names)} artifacts [{context_str}] using strategy '{strategy}'")

        views = []
        for name in artifact_names:
            version_name = self._artifact_versions.get(name)
            if version_name:
                views.append(self.store.get_view(version_name))

        if not views:
            raise ValueError("No valid artifacts found")

        combined = self.store.combine_views(views, strategy=strategy)
        result = combined.materialize()

        tprint(f"✅ Combined {len(artifact_names)} artifacts into DataFrame with shape {result.shape} [{context_str}]")
        return result

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
        from src.utils.tprint import tprint
        tprint("🐛 DEBUG: VersionedArtifactAdapter.get_statistics() called", "INFO")
        stats = self.store.get_statistics()
        tprint(f"🐛 DEBUG: Store statistics: {stats}", "INFO")
        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VersionedArtifactAdapter("
            f"symbol={self.symbol}, "
            f"exchange={self.exchange}, "
            f"timeframe={self.timeframe}, "
            f"direction={self.direction}, "
            f"model={self.model}, "
            f"artifacts={len(self._artifact_versions)}"
            f")"
        )

    def list_all_versions(self, artifact_type: Optional[str] = None) -> List[str]:
        """
        List all versions across all stores.
        
        Args:
            artifact_type: Optional filter for artifact type (e.g., 'clusters', 'regimes')
                           If provided, will filter stores based on artifact name patterns
        
        Returns:
            List of all version names from all stores
        """
        from typing import List
        from src.utils.tprint import tprint
        
        tprint(f"🐛 DEBUG: VersionedArtifactAdapter.list_all_versions() called with artifact_type={artifact_type}", "INFO")
        all_versions = []
        
        # Get all store directories
        store_dirs = [d for d in self.store.store_path.parent.iterdir() if d.is_dir()]
        tprint(f"🐛 DEBUG: Found {len(store_dirs)} store directories: {store_dirs}", "INFO")
        
        for store_dir in store_dirs:
            # For clusters/regimes/labels, we include all stores regardless of context
            if artifact_type is None or artifact_type.lower() in ['clusters', 'regimes', 'labels']:
                store_path = self.store.store_path.parent / store_dir
                if store_path.exists():
                    # Create a temporary store to list versions
                    temp_store = VersionedArtifactStore(
                        store_path=store_path,
                        auto_version=True,
                        enable_row_versioning=True
                    )
                    versions = temp_store.list_versions()
                    tprint(f"🐛 DEBUG: Store {store_dir} has {len(versions)} versions: {versions}", "INFO")
                    all_versions.extend(versions)
                else:
                    tprint(f"🐛 DEBUG: Skipping store {store_dir} (doesn't match artifact type: {artifact_type})", "INFO")
        
        tprint(f"🐛 DEBUG: Total versions across all stores: {len(all_versions)}", "INFO")
        return all_versions
