"""
Temporal Data Splitting for Versioned Artifacts

Provides clean interfaces for splitting data into training/validation/test periods
with proper embargo handling to prevent data leakage in ML pipelines.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

from .view import ArtifactView


@dataclass
class TemporalPeriod:
    """Defines a temporal period with start/end dates and embargo."""

    start: datetime
    end: datetime
    embargo_days: int = 0
    name: str = ""

    def __post_init__(self):
        """Validate period."""
        if self.start >= self.end:
            raise ValueError(f"Period start ({self.start}) must be before end ({self.end})")

    @property
    def effective_end(self) -> datetime:
        """Get effective end date after applying embargo."""
        return self.end - timedelta(days=self.embargo_days)

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this period (excluding embargo)."""
        return self.start <= timestamp <= self.effective_end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start': self.start.isoformat(),
            'end': self.end.isoformat(),
            'embargo_days': self.embargo_days,
            'name': self.name
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalPeriod':
        """Create from dictionary."""
        return cls(
            start=datetime.fromisoformat(data['start']),
            end=datetime.fromisoformat(data['end']),
            embargo_days=data.get('embargo_days', 0),
            name=data.get('name', '')
        )


@dataclass
class TemporalSplitConfig:
    """Configuration for train/validation/test splits."""

    training: TemporalPeriod
    validation: TemporalPeriod
    test: TemporalPeriod

    def __post_init__(self):
        """Validate no overlaps between periods."""
        # Check training doesn't overlap with validation
        if self.training.effective_end >= self.validation.start:
            raise ValueError(
                f"Training period (ends {self.training.effective_end}) "
                f"overlaps with validation period (starts {self.validation.start}). "
                f"Increase training embargo or adjust dates."
            )

        # Check validation doesn't overlap with test
        if self.validation.effective_end >= self.test.start:
            raise ValueError(
                f"Validation period (ends {self.validation.effective_end}) "
                f"overlaps with test period (starts {self.test.start}). "
                f"Increase validation embargo or adjust dates."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'training': self.training.to_dict(),
            'validation': self.validation.to_dict(),
            'test': self.test.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalSplitConfig':
        """Create from dictionary."""
        return cls(
            training=TemporalPeriod.from_dict(data['training']),
            validation=TemporalPeriod.from_dict(data['validation']),
            test=TemporalPeriod.from_dict(data['test'])
        )

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'TemporalSplitConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_from_data(
        cls,
        data_start: datetime,
        data_end: datetime,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        test_pct: float = 0.2,
        embargo_days: int = 30
    ) -> 'TemporalSplitConfig':
        """
        Create split config from data range with specified proportions.

        Args:
            data_start: Start of available data
            data_end: End of available data
            train_pct: Proportion for training (default: 0.6)
            val_pct: Proportion for validation (default: 0.2)
            test_pct: Proportion for testing (default: 0.2)
            embargo_days: Days of embargo between periods (default: 30)

        Returns:
            TemporalSplitConfig with calculated periods
        """
        if not np.isclose(train_pct + val_pct + test_pct, 1.0):
            raise ValueError("Percentages must sum to 1.0")

        total_days = (data_end - data_start).days

        # Calculate period durations (accounting for embargos)
        train_days = int(total_days * train_pct)
        val_days = int(total_days * val_pct)
        test_days = total_days - train_days - val_days - 2 * embargo_days

        # Calculate period boundaries
        train_start = data_start
        train_end = train_start + timedelta(days=train_days)

        val_start = train_end + timedelta(days=embargo_days)
        val_end = val_start + timedelta(days=val_days)

        test_start = val_end + timedelta(days=embargo_days)
        test_end = data_end

        return cls(
            training=TemporalPeriod(train_start, train_end, embargo_days, "training"),
            validation=TemporalPeriod(val_start, val_end, embargo_days, "validation"),
            test=TemporalPeriod(test_start, test_end, 0, "test")
        )


class TemporalViewFilter:
    """Helper for filtering artifact views by temporal period."""

    @staticmethod
    def filter_by_period(
        view: ArtifactView,
        period: TemporalPeriod
    ) -> ArtifactView:
        """
        Filter view to only include data within specified temporal period.

        Args:
            view: Artifact view to filter
            period: Temporal period to filter to

        Returns:
            Filtered artifact view
        """
        def period_filter(df: pd.DataFrame) -> np.ndarray:
            """Filter function that checks if index falls within period."""
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    "DataFrame must have DatetimeIndex for temporal filtering. "
                    f"Got index type: {type(df.index)}"
                )

            # Apply period boundaries
            mask = (df.index >= period.start) & (df.index <= period.effective_end)
            return mask

        return view.filter(period_filter)

    @staticmethod
    def get_training_view(
        view: ArtifactView,
        config: TemporalSplitConfig
    ) -> ArtifactView:
        """Get view filtered to training period."""
        return TemporalViewFilter.filter_by_period(view, config.training)

    @staticmethod
    def get_validation_view(
        view: ArtifactView,
        config: TemporalSplitConfig
    ) -> ArtifactView:
        """Get view filtered to validation period."""
        return TemporalViewFilter.filter_by_period(view, config.validation)

    @staticmethod
    def get_test_view(
        view: ArtifactView,
        config: TemporalSplitConfig
    ) -> ArtifactView:
        """Get view filtered to test period."""
        return TemporalViewFilter.filter_by_period(view, config.test)


def create_temporal_split_config_for_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_start: Optional[datetime] = None,
    data_end: Optional[datetime] = None,
    config_path: Optional[Path] = None
) -> TemporalSplitConfig:
    """
    Create or load temporal split configuration for a trading pair.

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '15m')
        data_start: Start of available data (required if creating new config)
        data_end: End of available data (required if creating new config)
        config_path: Path to save/load config (default: auto-generated)

    Returns:
        TemporalSplitConfig for this trading pair
    """
    if config_path is None:
        config_dir = Path("config/temporal_splits")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{symbol}_{exchange}_{timeframe}.json"

    # Try to load existing config
    if config_path.exists():
        return TemporalSplitConfig.load(config_path)

    # Create new config if data range provided
    if data_start is None or data_end is None:
        raise ValueError(
            f"Temporal split config not found at {config_path} and no data range provided. "
            f"Please provide data_start and data_end to create new config."
        )

    config = TemporalSplitConfig.create_from_data(
        data_start=data_start,
        data_end=data_end,
        train_pct=0.6,
        val_pct=0.2,
        test_pct=0.2,
        embargo_days=30
    )

    # Save for future use
    config.save(config_path)

    return config


# Convenience function for BaseStep integration
def get_data_for_purpose(
    view: ArtifactView,
    purpose: str,
    config: TemporalSplitConfig
) -> ArtifactView:
    """
    Get data view filtered for specific purpose (training/validation/test).

    This is the main function that steps should use to ensure proper
    temporal separation of data.

    Args:
        view: Source artifact view
        purpose: One of 'training', 'validation', 'test'
        config: Temporal split configuration

    Returns:
        Filtered view for the specified purpose

    Example:
        >>> # In model training step
        >>> training_view = get_data_for_purpose(view, 'training', config)
        >>> training_data = training_view.materialize()
        >>>
        >>> # In parameter optimization step
        >>> validation_view = get_data_for_purpose(view, 'validation', config)
        >>> validation_data = validation_view.materialize()
        >>>
        >>> # In final backtesting step
        >>> test_view = get_data_for_purpose(view, 'test', config)
        >>> test_data = test_view.materialize()
    """
    purpose = purpose.lower()

    if purpose == 'training':
        return TemporalViewFilter.get_training_view(view, config)
    elif purpose == 'validation':
        return TemporalViewFilter.get_validation_view(view, config)
    elif purpose == 'test':
        return TemporalViewFilter.get_test_view(view, config)
    else:
        raise ValueError(
            f"Invalid purpose: {purpose}. Must be 'training', 'validation', or 'test'"
        )
