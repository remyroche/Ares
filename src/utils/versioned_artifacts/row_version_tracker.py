"""
RowVersionTracker - Row-level version management

Tracks version history at the row level, enabling:
- Row-level rollback without affecting entire columns
- Delta encoding for efficient storage
- Version chains for tracking row evolution
"""

from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from pathlib import Path
import pickle


@dataclass
class RowVersion:
    """
    Version information for a single row.

    Attributes:
        row_index: Index of the row
        version_id: Unique version identifier
        timestamp: When this version was created
        previous_version: Previous version ID (for version chain)
        changes: Dictionary of column -> value changes
        metadata: Additional metadata
    """
    row_index: int
    version_id: str
    timestamp: datetime
    previous_version: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RowVersionTracker:
    """
    Track and manage row-level versions.

    Maintains version history for each row, enabling:
    - Efficient row updates without full column rewrites
    - Row-level rollback
    - Version chain traversal
    - Delta-based storage

    Storage format:
    - versions.pkl: Main version data
    - deltas/: Delta files for efficient storage
    """

    def __init__(self, tracker_dir: Path):
        """
        Initialize row version tracker.

        Args:
            tracker_dir: Directory for version tracking data
        """
        self.tracker_dir = Path(tracker_dir)
        self.tracker_dir.mkdir(parents=True, exist_ok=True)

        self.versions_file = self.tracker_dir / "versions.pkl"
        self.deltas_dir = self.tracker_dir / "deltas"
        self.deltas_dir.mkdir(parents=True, exist_ok=True)

        # Row index -> list of RowVersion (ordered by timestamp)
        self._row_versions: Dict[int, List[RowVersion]] = {}

        # Version ID -> RowVersion for quick lookup
        self._version_lookup: Dict[str, RowVersion] = {}

        # Current version ID for each row
        self._current_versions: Dict[int, str] = {}

        # Load existing versions
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version data from disk."""
        if self.versions_file.exists():
            with open(self.versions_file, 'rb') as f:
                data = pickle.load(f)
                self._row_versions = data.get('row_versions', {})
                self._version_lookup = data.get('version_lookup', {})
                self._current_versions = data.get('current_versions', {})

    def _save_versions(self) -> None:
        """Save version data to disk."""
        data = {
            'row_versions': self._row_versions,
            'version_lookup': self._version_lookup,
            'current_versions': self._current_versions
        }

        with open(self.versions_file, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def create_version(
        self,
        row_index: int,
        changes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> RowVersion:
        """
        Create a new version for a row.

        Args:
            row_index: Row index
            changes: Dictionary of column -> new value
            metadata: Additional metadata

        Returns:
            RowVersion instance
        """
        import uuid

        # Get previous version
        previous_version = self._current_versions.get(row_index)

        # Create new version
        version = RowVersion(
            row_index=row_index,
            version_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            previous_version=previous_version,
            changes=changes,
            metadata=metadata or {}
        )

        # Add to tracking structures
        if row_index not in self._row_versions:
            self._row_versions[row_index] = []

        self._row_versions[row_index].append(version)
        self._version_lookup[version.version_id] = version
        self._current_versions[row_index] = version.version_id

        # Save to disk
        self._save_versions()

        return version

    def get_current_version(self, row_index: int) -> Optional[RowVersion]:
        """
        Get current version for a row.

        Args:
            row_index: Row index

        Returns:
            Current RowVersion or None
        """
        version_id = self._current_versions.get(row_index)
        if version_id:
            return self._version_lookup.get(version_id)
        return None

    def get_version_at_time(
        self,
        row_index: int,
        timestamp: datetime
    ) -> Optional[RowVersion]:
        """
        Get row version at a specific time.

        Args:
            row_index: Row index
            timestamp: Target timestamp

        Returns:
            RowVersion at that time or None
        """
        versions = self._row_versions.get(row_index, [])

        # Find latest version before or at timestamp
        for version in reversed(versions):
            if version.timestamp <= timestamp:
                return version

        return None

    def get_version_history(self, row_index: int) -> List[RowVersion]:
        """
        Get complete version history for a row.

        Args:
            row_index: Row index

        Returns:
            List of RowVersion instances (oldest to newest)
        """
        return self._row_versions.get(row_index, []).copy()

    def rollback_row(
        self,
        row_index: int,
        version_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[RowVersion]:
        """
        Rollback a row to a previous version.

        Args:
            row_index: Row index
            version_id: Specific version to rollback to
            timestamp: Rollback to version at this time

        Returns:
            RowVersion after rollback or None
        """
        if version_id:
            target_version = self._version_lookup.get(version_id)
        elif timestamp:
            target_version = self.get_version_at_time(row_index, timestamp)
        else:
            # Rollback to previous version
            current_version = self.get_current_version(row_index)
            if not current_version or not current_version.previous_version:
                return None
            target_version = self._version_lookup.get(current_version.previous_version)

        if not target_version:
            return None

        # Update current version pointer
        self._current_versions[row_index] = target_version.version_id
        self._save_versions()

        return target_version

    def get_row_value(
        self,
        row_index: int,
        column: str,
        version_id: Optional[str] = None
    ) -> Optional[Any]:
        """
        Get value for a specific row and column at a version.

        Args:
            row_index: Row index
            column: Column name
            version_id: Version to query (None = current)

        Returns:
            Value or None
        """
        if version_id:
            version = self._version_lookup.get(version_id)
        else:
            version = self.get_current_version(row_index)

        if not version:
            return None

        # Check if this version has the value
        if column in version.changes:
            return version.changes[column]

        # Walk back through version chain
        while version.previous_version:
            version = self._version_lookup.get(version.previous_version)
            if version and column in version.changes:
                return version.changes[column]

        return None

    def reconstruct_row(
        self,
        row_index: int,
        columns: List[str],
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct complete row values at a version.

        Args:
            row_index: Row index
            columns: Columns to reconstruct
            version_id: Version to query (None = current)

        Returns:
            Dictionary of column -> value
        """
        result = {}

        for column in columns:
            value = self.get_row_value(row_index, column, version_id)
            if value is not None:
                result[column] = value

        return result

    def get_row_changes(
        self,
        row_index: int,
        from_version: Optional[str] = None,
        to_version: Optional[str] = None
    ) -> List[Tuple[str, RowVersion]]:
        """
        Get all changes for a row between versions.

        Args:
            row_index: Row index
            from_version: Starting version (None = earliest)
            to_version: Ending version (None = current)

        Returns:
            List of (change_type, RowVersion) tuples
        """
        versions = self._row_versions.get(row_index, [])

        # Find version indices
        start_idx = 0
        end_idx = len(versions)

        if from_version:
            for i, v in enumerate(versions):
                if v.version_id == from_version:
                    start_idx = i
                    break

        if to_version:
            for i, v in enumerate(versions):
                if v.version_id == to_version:
                    end_idx = i + 1
                    break

        # Get changes in range
        changes = []
        for version in versions[start_idx:end_idx]:
            for column in version.changes:
                changes.append((column, version))

        return changes

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get version tracking statistics.

        Returns:
            Dictionary with statistics
        """
        total_rows = len(self._row_versions)
        total_versions = len(self._version_lookup)

        # Calculate average versions per row
        avg_versions = total_versions / total_rows if total_rows > 0 else 0

        # Find most versioned rows
        most_versioned = sorted(
            [(idx, len(versions)) for idx, versions in self._row_versions.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_rows_tracked': total_rows,
            'total_versions': total_versions,
            'average_versions_per_row': avg_versions,
            'most_versioned_rows': most_versioned
        }

    def compact_versions(
        self,
        row_index: int,
        keep_last_n: int = 10
    ) -> int:
        """
        Compact version history for a row by removing old versions.

        Args:
            row_index: Row index
            keep_last_n: Number of recent versions to keep

        Returns:
            Number of versions removed
        """
        versions = self._row_versions.get(row_index, [])

        if len(versions) <= keep_last_n:
            return 0

        # Keep last N versions
        to_remove = versions[:-keep_last_n]
        self._row_versions[row_index] = versions[-keep_last_n:]

        # Remove from lookup
        for version in to_remove:
            self._version_lookup.pop(version.version_id, None)

        # Update version chain
        if self._row_versions[row_index]:
            self._row_versions[row_index][0].previous_version = None

        self._save_versions()

        return len(to_remove)

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"RowVersionTracker(dir='{self.tracker_dir}', rows={stats['total_rows_tracked']}, versions={stats['total_versions']})"
