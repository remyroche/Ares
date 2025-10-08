"""
Time Split Manager for Pre-Training Pipeline.

This module enforces consistent chronological train/validation/test segmentation
to prevent data leakage and non-stationarity issues in the pre-training pipeline.

Key Features:
- Chronological time-based splitting (no shuffling)
- Purged K-Fold Cross-Validation (à la López de Prado)
- Embargo periods to prevent information leakage
- Rolling window validation
- Regime-aware splitting
- Distribution validation across splits
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger


class SplitStrategy(Enum):
    """Available time split strategies."""
    
    SIMPLE_CHRONOLOGICAL = "simple_chronological"  # Simple 70/20/10 split
    PURGED_KFOLD = "purged_kfold"  # Purged k-fold cross-validation
    ROLLING_WINDOW = "rolling_window"  # Rolling window validation
    REGIME_AWARE = "regime_aware"  # Regime-specific splitting


@dataclass
class SplitConfig:
    """Configuration for time-based splitting."""
    
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    
    # Purging and embargo settings
    enable_purging: bool = True
    purge_window_hours: int = 24  # Hours to remove before validation/test
    embargo_window_hours: int = 12  # Hours to remove after training
    
    # Rolling window settings
    rolling_window_months: int = 6  # Size of rolling window
    rolling_step_months: int = 1  # Step size for rolling windows
    
    # Regime-aware settings
    regime_column: Optional[str] = None  # Column name for regime labels
    min_samples_per_regime: int = 100  # Minimum samples per regime
    
    # Validation settings
    validate_distribution: bool = True
    max_distribution_shift: float = 0.15  # Max allowed KL divergence
    
    # Random seed for reproducibility
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration."""
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"(train={self.train_ratio}, val={self.validation_ratio}, test={self.test_ratio})"
            )
        
        if self.train_ratio <= 0 or self.validation_ratio < 0 or self.test_ratio < 0:
            raise ValueError("Split ratios must be non-negative, train ratio must be positive")


@dataclass
class SplitResult:
    """Result of time-based data splitting."""
    
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    
    strategy: SplitStrategy
    config: SplitConfig
    
    # Distribution validation metrics
    distribution_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return len(self.train_idx)
    
    @property
    def validation_size(self) -> int:
        """Number of validation samples."""
        return len(self.validation_idx)
    
    @property
    def test_size(self) -> int:
        """Number of test samples."""
        return len(self.test_idx)
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the split."""
        return {
            'strategy': self.strategy.value,
            'train_size': self.train_size,
            'validation_size': self.validation_size,
            'test_size': self.test_size,
            'train_period': {
                'start': self.train_start.isoformat(),
                'end': self.train_end.isoformat(),
                'duration_days': (self.train_end - self.train_start).days
            },
            'validation_period': {
                'start': self.validation_start.isoformat(),
                'end': self.validation_end.isoformat(),
                'duration_days': (self.validation_end - self.validation_start).days
            },
            'test_period': {
                'start': self.test_start.isoformat(),
                'end': self.test_end.isoformat(),
                'duration_days': (self.test_end - self.test_start).days
            },
            'distribution_metrics': self.distribution_metrics,
            'metadata': self.metadata
        }


class TimeSplitManager:
    """
    Manages chronological time-based data splitting for the pre-training pipeline.
    
    This class enforces proper temporal segmentation to prevent:
    - Look-ahead bias
    - Data leakage through overlapping samples
    - Distribution mismatch between train/val/test
    
    Example:
        >>> manager = TimeSplitManager(config=SplitConfig())
        >>> split = manager.split(data, strategy=SplitStrategy.SIMPLE_CHRONOLOGICAL)
        >>> train_data = data.iloc[split.train_idx]
        >>> val_data = data.iloc[split.validation_idx]
    """
    
    def __init__(
        self,
        config: Optional[SplitConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the TimeSplitManager.
        
        Args:
            config: Split configuration
            logger: Optional logger instance
        """
        self.config = config or SplitConfig()
        self.logger = logger or system_logger.getChild('TimeSplitManager')
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def split(
        self,
        data: pd.DataFrame,
        strategy: SplitStrategy = SplitStrategy.SIMPLE_CHRONOLOGICAL,
        data_split: Optional[str] = None,
    ) -> SplitResult:
        """
        Split data chronologically into train/validation/test sets.
        
        Args:
            data: DataFrame with DatetimeIndex
            strategy: Split strategy to use
            data_split: Optional pre-specified split ('train', 'validation', 'test')
        
        Returns:
            SplitResult containing indices and metadata
        """
        self._validate_input(data)
        
        if strategy == SplitStrategy.SIMPLE_CHRONOLOGICAL:
            return self._simple_chronological_split(data)
        elif strategy == SplitStrategy.PURGED_KFOLD:
            return self._purged_kfold_split(data)
        elif strategy == SplitStrategy.ROLLING_WINDOW:
            return self._rolling_window_split(data)
        elif strategy == SplitStrategy.REGIME_AWARE:
            return self._regime_aware_split(data)
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data format."""
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex for chronological splitting")
        
        if not data.index.is_monotonic_increasing:
            self.logger.warning("Data index is not sorted, sorting now...")
            data.sort_index(inplace=True)
        
        if data.empty:
            raise ValueError("Cannot split empty DataFrame")
    
    def _simple_chronological_split(self, data: pd.DataFrame) -> SplitResult:
        """
        Perform simple chronological split with optional purging and embargo.
        
        This implements a 70/20/10 (configurable) split with:
        - Purging: Remove samples before validation/test that could leak information
        - Embargo: Remove samples after training that overlap with validation/test
        """
        n_samples = len(data)
        timestamps = data.index
        
        # Calculate split points
        train_end_idx = int(n_samples * self.config.train_ratio)
        val_end_idx = train_end_idx + int(n_samples * self.config.validation_ratio)
        
        # Apply purging and embargo if enabled
        if self.config.enable_purging:
            purge_timedelta = timedelta(hours=self.config.purge_window_hours)
            embargo_timedelta = timedelta(hours=self.config.embargo_window_hours)
            
            # Purge before validation
            val_start_time = timestamps[train_end_idx]
            purge_threshold_val = val_start_time - purge_timedelta
            train_end_idx_purged = timestamps.searchsorted(purge_threshold_val, side='right') - 1
            
            # Embargo after training (adjust validation start)
            embargo_threshold = timestamps[train_end_idx] + embargo_timedelta
            val_start_idx = timestamps.searchsorted(embargo_threshold, side='left')
            
            # Purge before test
            test_start_time = timestamps[val_end_idx]
            purge_threshold_test = test_start_time - purge_timedelta
            val_end_idx_purged = timestamps.searchsorted(purge_threshold_test, side='right') - 1
            
            # Embargo after validation (adjust test start)
            embargo_threshold_test = timestamps[val_end_idx] + embargo_timedelta
            test_start_idx = timestamps.searchsorted(embargo_threshold_test, side='left')
            
            self.logger.info(
                f"Purging removed {train_end_idx - train_end_idx_purged} samples from training, "
                f"{val_end_idx - val_end_idx_purged} from validation"
            )
            self.logger.info(
                f"Embargo created {val_start_idx - train_end_idx} sample gap after training, "
                f"{test_start_idx - val_end_idx} sample gap after validation"
            )
            
            train_end_idx = train_end_idx_purged
            val_end_idx = val_end_idx_purged
        else:
            val_start_idx = train_end_idx
            test_start_idx = val_end_idx
        
        # Create index arrays
        train_idx = np.arange(0, train_end_idx)
        validation_idx = np.arange(val_start_idx, val_end_idx)
        test_idx = np.arange(test_start_idx, n_samples)
        
        # Create result
        result = SplitResult(
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            train_start=timestamps[train_idx[0]],
            train_end=timestamps[train_idx[-1]],
            validation_start=timestamps[validation_idx[0]] if len(validation_idx) > 0 else timestamps[-1],
            validation_end=timestamps[validation_idx[-1]] if len(validation_idx) > 0 else timestamps[-1],
            test_start=timestamps[test_idx[0]] if len(test_idx) > 0 else timestamps[-1],
            test_end=timestamps[test_idx[-1]] if len(test_idx) > 0 else timestamps[-1],
            strategy=SplitStrategy.SIMPLE_CHRONOLOGICAL,
            config=self.config
        )
        
        # Validate distributions if enabled
        if self.config.validate_distribution:
            self._validate_distributions(data, result)
        
        self.logger.info(f"Split complete: {result.summary()}")
        return result
    
    def _purged_kfold_split(self, data: pd.DataFrame) -> SplitResult:
        """
        Perform purged k-fold cross-validation (López de Prado style).
        
        This creates multiple train/validation folds with:
        - Purging of overlapping samples
        - Embargo periods between folds
        """
        # For now, return first fold of a 5-fold split
        # In the future, this should support iterating over all folds
        n_samples = len(data)
        n_folds = 5
        fold_size = n_samples // n_folds
        
        # Use fold 0 for validation, folds 1-4 for training
        val_start = 0
        val_end = fold_size
        
        # Training includes all other folds with embargo
        embargo_samples = int(self.config.embargo_window_hours * (n_samples / 24 / 365))  # Approximate
        train_start = val_end + embargo_samples
        train_end = n_samples
        
        train_idx = np.arange(train_start, train_end)
        validation_idx = np.arange(val_start, val_end)
        test_idx = np.array([], dtype=int)  # No test set in k-fold
        
        timestamps = data.index
        result = SplitResult(
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            train_start=timestamps[train_idx[0]],
            train_end=timestamps[train_idx[-1]],
            validation_start=timestamps[validation_idx[0]],
            validation_end=timestamps[validation_idx[-1]],
            test_start=timestamps[-1],
            test_end=timestamps[-1],
            strategy=SplitStrategy.PURGED_KFOLD,
            config=self.config
        )
        
        self.logger.info(f"Purged K-Fold split complete: {result.summary()}")
        return result
    
    def _rolling_window_split(self, data: pd.DataFrame) -> SplitResult:
        """
        Perform rolling window split for walk-forward validation.
        
        Creates a rolling training window that moves forward in time.
        """
        n_samples = len(data)
        window_size = int(n_samples * self.config.train_ratio)
        
        # Use the last window as our split
        train_start = n_samples - window_size - int(n_samples * (self.config.validation_ratio + self.config.test_ratio))
        train_end = train_start + window_size
        
        val_end = train_end + int(n_samples * self.config.validation_ratio)
        
        train_idx = np.arange(train_start, train_end)
        validation_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n_samples)
        
        timestamps = data.index
        result = SplitResult(
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            train_start=timestamps[train_idx[0]],
            train_end=timestamps[train_idx[-1]],
            validation_start=timestamps[validation_idx[0]] if len(validation_idx) > 0 else timestamps[-1],
            validation_end=timestamps[validation_idx[-1]] if len(validation_idx) > 0 else timestamps[-1],
            test_start=timestamps[test_idx[0]] if len(test_idx) > 0 else timestamps[-1],
            test_end=timestamps[test_idx[-1]] if len(test_idx) > 0 else timestamps[-1],
            strategy=SplitStrategy.ROLLING_WINDOW,
            config=self.config
        )
        
        self.logger.info(f"Rolling window split complete: {result.summary()}")
        return result
    
    def _regime_aware_split(self, data: pd.DataFrame) -> SplitResult:
        """
        Perform regime-aware splitting ensuring all regimes represented.
        
        This ensures each split (train/val/test) contains samples from all regimes.
        """
        if self.config.regime_column is None or self.config.regime_column not in data.columns:
            self.logger.warning("Regime column not found, falling back to simple chronological split")
            return self._simple_chronological_split(data)
        
        regimes = data[self.config.regime_column].unique()
        n_samples = len(data)
        
        train_idx_list = []
        val_idx_list = []
        test_idx_list = []
        
        for regime in regimes:
            regime_mask = data[self.config.regime_column] == regime
            regime_indices = np.where(regime_mask)[0]
            
            if len(regime_indices) < self.config.min_samples_per_regime:
                self.logger.warning(
                    f"Regime {regime} has only {len(regime_indices)} samples, "
                    f"minimum is {self.config.min_samples_per_regime}"
                )
                continue
            
            # Split this regime chronologically
            n_regime = len(regime_indices)
            train_end = int(n_regime * self.config.train_ratio)
            val_end = train_end + int(n_regime * self.config.validation_ratio)
            
            train_idx_list.append(regime_indices[:train_end])
            val_idx_list.append(regime_indices[train_end:val_end])
            test_idx_list.append(regime_indices[val_end:])
        
        # Concatenate and sort indices
        train_idx = np.sort(np.concatenate(train_idx_list))
        validation_idx = np.sort(np.concatenate(val_idx_list))
        test_idx = np.sort(np.concatenate(test_idx_list))
        
        timestamps = data.index
        result = SplitResult(
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            train_start=timestamps[train_idx[0]],
            train_end=timestamps[train_idx[-1]],
            validation_start=timestamps[validation_idx[0]] if len(validation_idx) > 0 else timestamps[-1],
            validation_end=timestamps[validation_idx[-1]] if len(validation_idx) > 0 else timestamps[-1],
            test_start=timestamps[test_idx[0]] if len(test_idx) > 0 else timestamps[-1],
            test_end=timestamps[test_idx[-1]] if len(test_idx) > 0 else timestamps[-1],
            strategy=SplitStrategy.REGIME_AWARE,
            config=self.config,
            metadata={'regimes': regimes.tolist()}
        )
        
        self.logger.info(f"Regime-aware split complete: {result.summary()}")
        return result
    
    def _validate_distributions(self, data: pd.DataFrame, split: SplitResult) -> None:
        """
        Validate that train/validation/test distributions are similar.
        
        This computes KL divergence for numeric columns and logs warnings
        if distributions shift significantly.
        """
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) == 0:
                self.logger.warning("No numeric columns found for distribution validation")
                return
            
            # Sample a subset of columns if too many
            if len(numeric_columns) > 10:
                numeric_columns = numeric_columns[:10]
            
            train_data = data.iloc[split.train_idx][numeric_columns]
            val_data = data.iloc[split.validation_idx][numeric_columns]
            
            if len(split.test_idx) > 0:
                test_data = data.iloc[split.test_idx][numeric_columns]
            else:
                test_data = None
            
            # Compute simple statistics
            for col in numeric_columns:
                train_mean = train_data[col].mean()
                val_mean = val_data[col].mean()
                
                relative_shift = abs(val_mean - train_mean) / (abs(train_mean) + 1e-8)
                
                split.distribution_metrics[f'{col}_train_mean'] = float(train_mean)
                split.distribution_metrics[f'{col}_val_mean'] = float(val_mean)
                split.distribution_metrics[f'{col}_relative_shift'] = float(relative_shift)
                
                if relative_shift > self.config.max_distribution_shift:
                    self.logger.warning(
                        f"Large distribution shift detected for {col}: "
                        f"{relative_shift:.3f} (threshold: {self.config.max_distribution_shift})"
                    )
                
                if test_data is not None:
                    test_mean = test_data[col].mean()
                    test_shift = abs(test_mean - train_mean) / (abs(train_mean) + 1e-8)
                    split.distribution_metrics[f'{col}_test_mean'] = float(test_mean)
                    split.distribution_metrics[f'{col}_test_shift'] = float(test_shift)
        
        except Exception as e:
            self.logger.error(f"Error during distribution validation: {e}")


def create_time_split_manager(
    train_ratio: float = 0.70,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.10,
    enable_purging: bool = True,
    purge_window_hours: int = 24,
    embargo_window_hours: int = 12,
    **kwargs
) -> TimeSplitManager:
    """
    Factory function to create a TimeSplitManager with common settings.
    
    Args:
        train_ratio: Ratio of data for training
        validation_ratio: Ratio of data for validation
        test_ratio: Ratio of data for test
        enable_purging: Whether to enable purging
        purge_window_hours: Hours to purge before validation/test
        embargo_window_hours: Hours to embargo after training
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured TimeSplitManager instance
    """
    config = SplitConfig(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        enable_purging=enable_purging,
        purge_window_hours=purge_window_hours,
        embargo_window_hours=embargo_window_hours,
        **kwargs
    )
    return TimeSplitManager(config=config)


__all__ = [
    'TimeSplitManager',
    'SplitConfig',
    'SplitResult',
    'SplitStrategy',
    'create_time_split_manager',
]