"""
ChangeLog - Comprehensive change tracking system

Records all modifications to the artifact store with full metadata,
enabling audit trails, time-travel queries, and change analysis.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json
import sqlite3
from enum import Enum


class ChangeType(Enum):
    """Types of changes that can be recorded."""
    ADD_DATA = "add_data"
    UPDATE_ROWS = "update_rows"
    UPDATE_COLUMNS = "update_columns"
    DELETE_ROWS = "delete_rows"
    DELETE_COLUMNS = "delete_columns"
    CREATE_VIEW = "create_view"
    MERGE_VIEWS = "merge_views"
    CREATE_VERSION = "create_version"
    ROLLBACK = "rollback"


@dataclass
class ChangeRecord:
    """
    Record of a single change operation.

    Attributes:
        change_id: Unique identifier for this change
        change_type: Type of change operation
        timestamp: When the change occurred
        version_name: Version this change belongs to
        affected_rows: Row indices or count affected
        affected_columns: Column names affected
        metadata: Additional change metadata
        user: User who made the change (optional)
        description: Human-readable description
    """
    change_id: str
    change_type: ChangeType
    timestamp: datetime
    version_name: str
    affected_rows: Optional[Union[List[int], int]] = None
    affected_columns: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    user: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['change_type'] = self.change_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'ChangeRecord':
        """Create from dictionary."""
        data = data.copy()
        data['change_type'] = ChangeType(data['change_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ChangeLog:
    """
    Comprehensive change tracking system.

    Maintains both JSONL files for append-only logging and SQLite
    database for efficient querying.

    Features:
    - Append-only JSONL logs (one per day)
    - SQLite database for fast queries
    - Time-travel queries
    - Change diffs
    - Audit trail export
    """

    def __init__(self, changelog_dir: Union[str, Path]):
        """
        Initialize change log.

        Args:
            changelog_dir: Directory for change logs
        """
        self.changelog_dir = Path(changelog_dir)
        self.changelog_dir.mkdir(parents=True, exist_ok=True)

        # SQLite database for queries
        self.db_path = self.changelog_dir / "audit_trail.db"
        self._init_database()

        # Current day's JSONL file
        self._current_jsonl_file = None
        self._current_jsonl_date = None

    def _init_database(self) -> None:
        """Initialize SQLite database for change tracking."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS changes (
                change_id TEXT PRIMARY KEY,
                change_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                version_name TEXT NOT NULL,
                affected_rows TEXT,
                affected_columns TEXT,
                metadata TEXT,
                user TEXT,
                description TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON changes(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_version ON changes(version_name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_change_type ON changes(change_type)
        """)

        conn.commit()
        conn.close()

    def _get_jsonl_file(self) -> Path:
        """Get current day's JSONL file."""
        today = datetime.now().date()

        if self._current_jsonl_date != today:
            self._current_jsonl_date = today
            self._current_jsonl_file = self.changelog_dir / f"changes_{today.isoformat()}.jsonl"

        return self._current_jsonl_file

    def record_change(
        self,
        change_type: ChangeType,
        version_name: str,
        affected_rows: Optional[Union[List[int], int]] = None,
        affected_columns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        description: Optional[str] = None
    ) -> ChangeRecord:
        """
        Record a change operation.

        Args:
            change_type: Type of change
            version_name: Version this change belongs to
            affected_rows: Rows affected by this change
            affected_columns: Columns affected by this change
            metadata: Additional metadata
            user: User who made the change
            description: Human-readable description

        Returns:
            ChangeRecord instance
        """
        import uuid

        change_record = ChangeRecord(
            change_id=str(uuid.uuid4()),
            change_type=change_type,
            timestamp=datetime.now(),
            version_name=version_name,
            affected_rows=affected_rows,
            affected_columns=affected_columns,
            metadata=metadata or {},
            user=user,
            description=description
        )

        # Write to JSONL file
        self._write_to_jsonl(change_record)

        # Write to database
        self._write_to_database(change_record)

        return change_record

    def _write_to_jsonl(self, record: ChangeRecord) -> None:
        """Write change record to JSONL file."""
        jsonl_file = self._get_jsonl_file()

        with open(jsonl_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')

    def _write_to_database(self, record: ChangeRecord) -> None:
        """Write change record to SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO changes (
                change_id, change_type, timestamp, version_name,
                affected_rows, affected_columns, metadata, user, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.change_id,
            record.change_type.value,
            record.timestamp.isoformat(),
            record.version_name,
            json.dumps(record.affected_rows) if record.affected_rows else None,
            json.dumps(record.affected_columns) if record.affected_columns else None,
            json.dumps(record.metadata),
            record.user,
            record.description
        ))

        conn.commit()
        conn.close()

    def get_changes(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        version_name: Optional[str] = None,
        change_types: Optional[List[ChangeType]] = None,
        limit: Optional[int] = None
    ) -> List[ChangeRecord]:
        """
        Query changes with filters.

        Args:
            from_time: Start time filter
            to_time: End time filter
            version_name: Filter by version
            change_types: Filter by change types
            limit: Maximum number of results

        Returns:
            List of ChangeRecord instances
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM changes WHERE 1=1"
        params = []

        if from_time:
            query += " AND timestamp >= ?"
            params.append(from_time.isoformat())

        if to_time:
            query += " AND timestamp <= ?"
            params.append(to_time.isoformat())

        if version_name:
            query += " AND version_name = ?"
            params.append(version_name)

        if change_types:
            placeholders = ','.join(['?'] * len(change_types))
            query += f" AND change_type IN ({placeholders})"
            params.extend([ct.value for ct in change_types])

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        # Convert to ChangeRecord instances
        records = []
        for row in rows:
            records.append(ChangeRecord(
                change_id=row[0],
                change_type=ChangeType(row[1]),
                timestamp=datetime.fromisoformat(row[2]),
                version_name=row[3],
                affected_rows=json.loads(row[4]) if row[4] else None,
                affected_columns=json.loads(row[5]) if row[5] else None,
                metadata=json.loads(row[6]) if row[6] else {},
                user=row[7],
                description=row[8]
            ))

        return records

    def get_changes_for_rows(
        self,
        row_indices: List[int],
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[ChangeRecord]:
        """
        Get all changes affecting specific rows.

        Args:
            row_indices: Row indices to check
            from_time: Start time filter
            to_time: End time filter

        Returns:
            List of ChangeRecord instances
        """
        all_changes = self.get_changes(from_time=from_time, to_time=to_time)

        # Filter changes affecting these rows
        relevant_changes = []
        row_set = set(row_indices)

        for change in all_changes:
            if change.affected_rows is None:
                # Affects all rows
                relevant_changes.append(change)
            elif isinstance(change.affected_rows, int):
                # Count of rows affected (can't determine specific rows)
                relevant_changes.append(change)
            elif isinstance(change.affected_rows, list):
                # Check if any affected rows match
                if row_set.intersection(set(change.affected_rows)):
                    relevant_changes.append(change)

        return relevant_changes

    def get_version_history(self, version_name: str) -> List[ChangeRecord]:
        """
        Get all changes for a specific version.

        Args:
            version_name: Version to query

        Returns:
            List of ChangeRecord instances
        """
        return self.get_changes(version_name=version_name)

    def export_to_csv(self, output_path: Union[str, Path]) -> None:
        """
        Export change log to CSV.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM changes ORDER BY timestamp")
        rows = cursor.fetchall()

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'change_id', 'change_type', 'timestamp', 'version_name',
                'affected_rows', 'affected_columns', 'metadata', 'user', 'description'
            ])
            writer.writerows(rows)

        conn.close()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get change log statistics.

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Total changes
        cursor.execute("SELECT COUNT(*) FROM changes")
        total_changes = cursor.fetchone()[0]

        # Changes by type
        cursor.execute("""
            SELECT change_type, COUNT(*)
            FROM changes
            GROUP BY change_type
        """)
        changes_by_type = dict(cursor.fetchall())

        # Changes by version
        cursor.execute("""
            SELECT version_name, COUNT(*)
            FROM changes
            GROUP BY version_name
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        top_versions = dict(cursor.fetchall())

        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM changes")
        time_range = cursor.fetchone()

        conn.close()

        return {
            'total_changes': total_changes,
            'changes_by_type': changes_by_type,
            'top_versions': top_versions,
            'time_range': {
                'earliest': time_range[0],
                'latest': time_range[1]
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return f"ChangeLog(dir='{self.changelog_dir}', total_changes={stats['total_changes']})"
