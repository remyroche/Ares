"""
ViewMask - Boolean masks for row/column selection

Provides composable boolean masks for efficient data selection without
loading the full dataset.
"""

from typing import Optional, Union, List, Set
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ViewMask:
    """
    Boolean mask for row and column selection.

    Provides efficient selection of subsets of data without loading
    the full dataset. Masks can be combined using boolean operations.

    Attributes:
        row_mask: Boolean array for row selection (None = all rows)
        column_mask: Set of column names for column selection (None = all columns)
        name: Optional name for this mask
        metadata: Additional metadata
    """

    row_mask: Optional[np.ndarray] = None
    column_mask: Optional[Set[str]] = None
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize mask data."""
        if self.column_mask is not None and not isinstance(self.column_mask, set):
            self.column_mask = set(self.column_mask)

    @property
    def num_rows(self) -> Optional[int]:
        """Get number of selected rows."""
        if self.row_mask is None:
            return None
        return int(np.sum(self.row_mask))

    @property
    def num_columns(self) -> Optional[int]:
        """Get number of selected columns."""
        if self.column_mask is None:
            return None
        return len(self.column_mask)

    @property
    def total_rows(self) -> Optional[int]:
        """Get total number of rows in mask."""
        if self.row_mask is None:
            return None
        return len(self.row_mask)

    def select_rows(self, indices: Union[np.ndarray, List[int], slice]) -> 'ViewMask':
        """
        Create a new mask with selected rows.

        Args:
            indices: Row indices, boolean mask, or slice

        Returns:
            New ViewMask with selected rows
        """
        if self.row_mask is None:
            # No existing row mask, create new one
            if isinstance(indices, slice):
                raise ValueError("Cannot use slice without knowing total rows")

            if isinstance(indices, (list, np.ndarray)):
                if len(indices) > 0 and isinstance(indices[0], bool):
                    # Boolean mask
                    new_row_mask = np.array(indices, dtype=bool)
                else:
                    # Integer indices - need total rows
                    raise ValueError("Cannot use integer indices without knowing total rows")
            else:
                raise ValueError(f"Unsupported indices type: {type(indices)}")
        else:
            # Apply selection to existing mask
            if isinstance(indices, slice):
                new_row_mask = self.row_mask.copy()
                new_row_mask[indices] = False
                new_row_mask[indices] = self.row_mask[indices]
            elif isinstance(indices, (list, np.ndarray)):
                indices_arr = np.array(indices)
                if indices_arr.dtype == bool:
                    # Boolean mask - AND with existing
                    if len(indices_arr) != len(self.row_mask):
                        raise ValueError(f"Boolean mask length {len(indices_arr)} doesn't match existing mask length {len(self.row_mask)}")
                    new_row_mask = self.row_mask & indices_arr
                else:
                    # Integer indices
                    new_row_mask = np.zeros_like(self.row_mask)
                    new_row_mask[indices_arr] = self.row_mask[indices_arr]
            else:
                raise ValueError(f"Unsupported indices type: {type(indices)}")

        return ViewMask(
            row_mask=new_row_mask,
            column_mask=self.column_mask.copy() if self.column_mask else None,
            name=f"{self.name}_rows" if self.name else None,
            metadata=self.metadata.copy()
        )

    def select_columns(self, columns: Union[List[str], Set[str]]) -> 'ViewMask':
        """
        Create a new mask with selected columns.

        Args:
            columns: Column names to select

        Returns:
            New ViewMask with selected columns
        """
        columns_set = set(columns) if not isinstance(columns, set) else columns

        if self.column_mask is None:
            new_column_mask = columns_set
        else:
            # Intersect with existing column mask
            new_column_mask = self.column_mask & columns_set

        return ViewMask(
            row_mask=self.row_mask.copy() if self.row_mask is not None else None,
            column_mask=new_column_mask,
            name=f"{self.name}_cols" if self.name else None,
            metadata=self.metadata.copy()
        )

    def __and__(self, other: 'ViewMask') -> 'ViewMask':
        """
        Combine masks with AND operation.

        Args:
            other: Another ViewMask

        Returns:
            New ViewMask with AND of both masks
        """
        # Combine row masks
        if self.row_mask is None:
            new_row_mask = other.row_mask.copy() if other.row_mask is not None else None
        elif other.row_mask is None:
            new_row_mask = self.row_mask.copy()
        else:
            if len(self.row_mask) != len(other.row_mask):
                raise ValueError(f"Row mask lengths don't match: {len(self.row_mask)} vs {len(other.row_mask)}")
            new_row_mask = self.row_mask & other.row_mask

        # Combine column masks
        if self.column_mask is None:
            new_column_mask = other.column_mask.copy() if other.column_mask else None
        elif other.column_mask is None:
            new_column_mask = self.column_mask.copy()
        else:
            new_column_mask = self.column_mask & other.column_mask

        # Combine metadata
        new_metadata = self.metadata.copy()
        new_metadata.update(other.metadata)

        return ViewMask(
            row_mask=new_row_mask,
            column_mask=new_column_mask,
            name=f"{self.name}_AND_{other.name}" if self.name and other.name else None,
            metadata=new_metadata
        )

    def __or__(self, other: 'ViewMask') -> 'ViewMask':
        """
        Combine masks with OR operation.

        Args:
            other: Another ViewMask

        Returns:
            New ViewMask with OR of both masks
        """
        # Combine row masks
        if self.row_mask is None or other.row_mask is None:
            # If either is None (all rows), result is all rows
            new_row_mask = None
        else:
            if len(self.row_mask) != len(other.row_mask):
                raise ValueError(f"Row mask lengths don't match: {len(self.row_mask)} vs {len(other.row_mask)}")
            new_row_mask = self.row_mask | other.row_mask

        # Combine column masks
        if self.column_mask is None or other.column_mask is None:
            # If either is None (all columns), result is all columns
            new_column_mask = None
        else:
            new_column_mask = self.column_mask | other.column_mask

        # Combine metadata
        new_metadata = self.metadata.copy()
        new_metadata.update(other.metadata)

        return ViewMask(
            row_mask=new_row_mask,
            column_mask=new_column_mask,
            name=f"{self.name}_OR_{other.name}" if self.name and other.name else None,
            metadata=new_metadata
        )

    def __invert__(self) -> 'ViewMask':
        """
        Invert the mask (NOT operation).

        Returns:
            New ViewMask with inverted row mask
        """
        new_row_mask = ~self.row_mask if self.row_mask is not None else None

        return ViewMask(
            row_mask=new_row_mask,
            column_mask=self.column_mask.copy() if self.column_mask else None,
            name=f"NOT_{self.name}" if self.name else None,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> dict:
        """
        Convert mask to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            'row_mask': self.row_mask.tolist() if self.row_mask is not None else None,
            'column_mask': list(self.column_mask) if self.column_mask else None,
            'name': self.name,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ViewMask':
        """
        Create mask from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ViewMask instance
        """
        row_mask = np.array(data['row_mask'], dtype=bool) if data.get('row_mask') else None
        column_mask = set(data['column_mask']) if data.get('column_mask') else None

        return cls(
            row_mask=row_mask,
            column_mask=column_mask,
            name=data.get('name'),
            metadata=data.get('metadata', {})
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save mask to file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ViewMask':
        """
        Load mask from file.

        Args:
            path: File path to load from

        Returns:
            ViewMask instance
        """
        with open(path, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation."""
        row_info = f"{self.num_rows}/{self.total_rows} rows" if self.row_mask is not None else "all rows"
        col_info = f"{self.num_columns} columns" if self.column_mask else "all columns"
        name_info = f"'{self.name}'" if self.name else "unnamed"

        return f"ViewMask({name_info}, {row_info}, {col_info})"
