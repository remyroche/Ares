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
        embargo_days: int = 1  # Changed from 30 to 1 day
    ) -> 'TemporalSplitConfig':
        """
        Create split config from data range with specified proportions.

        Args:
            data_start: Start of available data
            data_end: End of available data
            train_pct: Proportion for training (default: 0.6)
            val_pct: Proportion for validation (default: 0.2)
            test_pct: Proportion for testing (default: 0.2)
            embargo_days: Days of embargo between periods (default: 1)

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


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward validation with train and validation periods."""

    fold_num: int
    training: TemporalPeriod
    validation: TemporalPeriod

    def __post_init__(self):
        """Validate no overlap between training and validation."""
        if self.training.effective_end >= self.validation.start:
            raise ValueError(
                f"Fold {self.fold_num}: Training period (ends {self.training.effective_end}) "
                f"overlaps with validation period (starts {self.validation.start}). "
                f"Increase embargo or adjust dates."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fold_num': self.fold_num,
            'training': self.training.to_dict(),
            'validation': self.validation.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalkForwardFold':
        """Create from dictionary."""
        return cls(
            fold_num=data['fold_num'],
            training=TemporalPeriod.from_dict(data['training']),
            validation=TemporalPeriod.from_dict(data['validation'])
        )


@dataclass
class WalkForwardSplitConfig:
    """Configuration for walk-forward cross-validation with multiple train/val folds and held-out test."""

    folds: list  # List[WalkForwardFold]
    test: TemporalPeriod
    strategy: str = 'expanding'  # 'expanding' or 'rolling'

    def __post_init__(self):
        """Validate fold sequence and no overlap with test period."""
        if not self.folds:
            raise ValueError("Must have at least one fold")

        # Validate last fold doesn't overlap with test
        last_fold = self.folds[-1]
        if last_fold.validation.effective_end >= self.test.start:
            raise ValueError(
                f"Last validation fold (ends {last_fold.validation.effective_end}) "
                f"overlaps with test period (starts {self.test.start}). "
                f"Increase embargo or adjust dates."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'folds': [fold.to_dict() for fold in self.folds],
            'test': self.test.to_dict(),
            'strategy': self.strategy
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WalkForwardSplitConfig':
        """Create from dictionary."""
        return cls(
            folds=[WalkForwardFold.from_dict(f) for f in data['folds']],
            test=TemporalPeriod.from_dict(data['test']),
            strategy=data.get('strategy', 'expanding')
        )

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'WalkForwardSplitConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_expanding_window(
        cls,
        data_start: datetime,
        data_end: datetime,
        n_folds: int = 3,
        val_pct_per_fold: float = 0.10,
        final_test_pct: float = 0.15,
        min_train_pct: float = 0.55,
        embargo_days: int = 1
    ) -> 'WalkForwardSplitConfig':
        """
        Create expanding window walk-forward configuration.

        With expanding window, each fold uses progressively more training data:
        - Fold 1: Train on 0-55%, validate on 55-65%
        - Fold 2: Train on 0-65%, validate on 65-75%
        - Fold 3: Train on 0-75%, validate on 75-85%
        - Test: 85-100%

        Args:
            data_start: Start of available data
            data_end: End of available data
            n_folds: Number of train/val pairs (default: 3)
            val_pct_per_fold: Validation percentage per fold (default: 0.10)
            final_test_pct: Percentage for final held-out test (default: 0.15)
            min_train_pct: Minimum training percentage for first fold (default: 0.55)
            embargo_days: Days of embargo between periods (default: 1)

        Returns:
            WalkForwardSplitConfig with expanding window folds
        """
        total_days = (data_end - data_start).days

        # Calculate fold boundaries
        folds = []
        current_pct = min_train_pct

        for i in range(n_folds):
            # Training period: from start to current_pct
            train_start = data_start
            train_days = int(total_days * current_pct)
            train_end = train_start + timedelta(days=train_days)

            # Validation period: next val_pct_per_fold
            val_start = train_end + timedelta(days=embargo_days)
            val_days = int(total_days * val_pct_per_fold)
            val_end = val_start + timedelta(days=val_days)

            fold = WalkForwardFold(
                fold_num=i + 1,
                training=TemporalPeriod(train_start, train_end, embargo_days, f"training_fold_{i+1}"),
                validation=TemporalPeriod(val_start, val_end, embargo_days, f"validation_fold_{i+1}")
            )
            folds.append(fold)

            # Expand window for next fold
            current_pct += val_pct_per_fold

        # Final test period
        test_start_pct = min_train_pct + (n_folds * val_pct_per_fold)
        test_start_days = int(total_days * test_start_pct)
        test_start = data_start + timedelta(days=test_start_days + embargo_days)
        test_end = data_end

        test_period = TemporalPeriod(test_start, test_end, 0, "test")

        return cls(
            folds=folds,
            test=test_period,
            strategy='expanding'
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


def create_walkforward_split_config_for_pipeline(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_start: Optional[datetime] = None,
    data_end: Optional[datetime] = None,
    n_folds: int = 3,
    val_pct_per_fold: float = 0.10,
    final_test_pct: float = 0.15,
    min_train_pct: float = 0.55,
    embargo_days: int = 1,
    config_path: Optional[Path] = None
) -> WalkForwardSplitConfig:
    """
    Create or load walk-forward split configuration for a trading pair.

    This creates an expanding window configuration with multiple train/val folds
    and a final held-out test period:
    - Fold 1: Train 0-55%, Val 55-65%
    - Fold 2: Train 0-65%, Val 65-75%
    - Fold 3: Train 0-75%, Val 75-85%
    - Test: 85-100%

    Args:
        symbol: Trading symbol (e.g., 'ETHUSDT')
        exchange: Exchange name (e.g., 'binance')
        timeframe: Timeframe (e.g., '15m')
        data_start: Start of available data (required if creating new config)
        data_end: End of available data (required if creating new config)
        n_folds: Number of train/val pairs (default: 3)
        val_pct_per_fold: Validation percentage per fold (default: 0.10)
        final_test_pct: Percentage for final test (default: 0.15)
        min_train_pct: Starting training percentage (default: 0.55)
        embargo_days: Days of embargo between periods (default: 1)
        config_path: Path to save/load config (default: auto-generated)

    Returns:
        WalkForwardSplitConfig for this trading pair
    """
    if config_path is None:
        config_dir = Path("config/temporal_splits")
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{symbol}_{exchange}_{timeframe}_walkforward.json"

    # Try to load existing config
    if config_path.exists():
        return WalkForwardSplitConfig.load(config_path)

    # Create new config if data range provided
    if data_start is None or data_end is None:
        raise ValueError(
            f"Walk-forward config not found at {config_path} and no data range provided. "
            f"Please provide data_start and data_end to create new config."
        )

    config = WalkForwardSplitConfig.create_expanding_window(
        data_start=data_start,
        data_end=data_end,
        n_folds=n_folds,
        val_pct_per_fold=val_pct_per_fold,
        final_test_pct=final_test_pct,
        min_train_pct=min_train_pct,
        embargo_days=embargo_days
    )

    # Save for future use
    config.save(config_path)

    return config


# Convenience function for BaseStep integration
def get_data_for_purpose(
    view: ArtifactView,
    purpose: str = 'training',  # Default to 'training' for backward compatibility
    config: Optional[TemporalSplitConfig] = None
) -> ArtifactView:
    """
    Get data view filtered for specific purpose (training/validation/test).

    This is the main function that steps should use to ensure proper
    temporal separation of data.

    Args:
        view: Source artifact view
        purpose: One of 'training' (default), 'validation', 'test', or 'all'
        config: Temporal split configuration (if None, returns full view)

    Returns:
        Filtered view for the specified purpose, or full view if config is None

    Example:
        >>> # Backward compatible - no config means training data (full view)
        >>> training_view = get_data_for_purpose(view)  # Returns full view
        >>> training_data = training_view.materialize()
        >>>
        >>> # With temporal config
        >>> config = create_temporal_split_config_for_pipeline(...)
        >>>
        >>> # In model training step (default)
        >>> training_view = get_data_for_purpose(view, 'training', config)
        >>>
        >>> # In parameter optimization step
        >>> validation_view = get_data_for_purpose(view, 'validation', config)
        >>>
        >>> # In backtesting step
        >>> test_view = get_data_for_purpose(view, 'test', config)
        >>>
        >>> # For monte carlo (all periods)
        >>> all_view = get_data_for_purpose(view, 'all', config)
    """
    # Backward compatibility: if no config provided, return full view
    if config is None:
        return view

    purpose = purpose.lower()

    if purpose == 'training':
        return TemporalViewFilter.get_training_view(view, config)
    elif purpose == 'validation':
        return TemporalViewFilter.get_validation_view(view, config)
    elif purpose == 'test':
        return TemporalViewFilter.get_test_view(view, config)
    elif purpose == 'all':
        # Return full view (no filtering) for monte carlo across all periods
        return view
    else:
        raise ValueError(
            f"Invalid purpose: {purpose}. Must be 'training', 'validation', 'test', or 'all'"
        )
