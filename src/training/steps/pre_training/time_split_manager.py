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
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_mean, safe_std, validate_finite, optimize_dataframe_dtypes,
    calculate_data_quality_metrics, get_dataframe_info, parallel_map
)
from src.utils.math_validation import safe_correlation, validate_correlation_matrix
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations


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
        Initialize the TimeSplitManager with enhanced utilities.

        Args:
            config: Split configuration
            logger: Optional logger instance
        """
        tprint_info("🔧 Initializing TimeSplitManager with enhanced utilities")
        self.config = config or SplitConfig()
        self.logger = logger or system_logger.getChild('TimeSplitManager')

        # Initialize hardware optimizers for performance
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        self.matrix_ops = UnifiedMatrixOperations()

        # Set random seed using enhanced utilities
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        tprint_success("✅ TimeSplitManager initialized with enhanced utilities")
    
    def split(
        self,
        data: pd.DataFrame,
        strategy: SplitStrategy = SplitStrategy.SIMPLE_CHRONOLOGICAL,
        data_split: Optional[str] = None,
    ) -> SplitResult:
        """
        Split data chronologically into train/validation/test sets with enhanced utilities.

        Args:
            data: DataFrame with DatetimeIndex
            strategy: Split strategy to use
            data_split: Optional pre-specified split ('train', 'validation', 'test')

        Returns:
            SplitResult containing indices and metadata
        """
        tprint_info(f"🔀 Starting time-based split with strategy: {strategy.value}")

        try:
            # Optimize data before processing
            optimized_data = optimize_dataframe_dtypes(data)
            self._validate_input(optimized_data)

            # Track memory usage
            initial_memory = self.memory_optimizer.memory_pressure if self.memory_optimizer else 0.0

            # Execute split based on strategy
            if strategy == SplitStrategy.SIMPLE_CHRONOLOGICAL:
                result = self._simple_chronological_split(optimized_data)
            elif strategy == SplitStrategy.PURGED_KFOLD:
                result = self._purged_kfold_split(optimized_data)
            elif strategy == SplitStrategy.ROLLING_WINDOW:
                result = self._rolling_window_split(optimized_data)
            elif strategy == SplitStrategy.REGIME_AWARE:
                result = self._regime_aware_split(optimized_data)
            else:
                raise ValueError(f"Unknown split strategy: {strategy}")

            # Add memory usage to result metadata
            final_memory = self.memory_optimizer.memory_pressure if self.memory_optimizer else 0.0
            result.metadata.update({
                'memory_usage_before': initial_memory,
                'memory_usage_after': final_memory,
                'memory_delta': final_memory - initial_memory,
                'optimization_applied': True
            })

            tprint_success(f"✅ Time-based split completed: {result.summary()['train_size']} train, {result.summary()['validation_size']} val, {result.summary()['test_size']} test samples")
            return result

        except Exception as e:
            tprint_error(f"❌ Time-based split failed: {e}")
            self.logger.error(f"Time split failed: {e}")
            raise
    
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
    
    def _purged_kfold_split(self, data: pd.DataFrame, n_folds: int = 5) -> SplitResult:
        """
        Perform purged k-fold cross-validation (López de Prado style).
        
        NOTE: This returns only the first fold. Use purged_kfold_splits() to get all folds.
        
        This creates multiple train/validation folds with:
        - Purging of overlapping samples
        - Embargo periods between folds
        
        Args:
            data: DataFrame with DatetimeIndex
            n_folds: Number of folds (default 5)
            
        Returns:
            First fold as SplitResult
        """
        # Get all folds and return the first one
        all_folds = list(self.purged_kfold_splits(data, n_folds))
        
        if not all_folds:
            raise ValueError("Failed to generate purged k-fold splits")
        
        return all_folds[0]
    
    def purged_kfold_splits(self, data: pd.DataFrame, n_folds: int = 5) -> List[SplitResult]:
        """
        Generate all folds for purged k-fold cross-validation (López de Prado style).
        
        This implements proper purged k-fold CV with:
        - Chronological fold ordering
        - Purging of samples before validation to prevent look-ahead
        - Embargo of samples after validation to prevent label leakage
        - Each fold uses non-overlapping validation set
        
        Args:
            data: DataFrame with DatetimeIndex
            n_folds: Number of folds for cross-validation
            
        Returns:
            List of SplitResult, one for each fold
            
        Example:
            >>> manager = TimeSplitManager()
            >>> for fold_idx, split in enumerate(manager.purged_kfold_splits(data, n_folds=5)):
            ...     train_data = data.iloc[split.train_idx]
            ...     val_data = data.iloc[split.validation_idx]
            ...     # Train and validate model on this fold
        """
        tprint_info(f"🔀 Generating {n_folds} purged k-fold splits")
        
        n_samples = len(data)
        timestamps = data.index
        fold_size = n_samples // n_folds
        
        if fold_size < 100:
            tprint_warning(f"⚠️ Fold size is very small ({fold_size} samples), consider reducing n_folds")
        
        purge_delta = timedelta(hours=self.config.purge_window_hours)
        embargo_delta = timedelta(hours=self.config.embargo_window_hours)
        
        all_folds = []
        
        for fold_idx in range(n_folds):
            # Define validation fold boundaries
            val_start_idx = fold_idx * fold_size
            val_end_idx = min((fold_idx + 1) * fold_size, n_samples)
            
            if val_start_idx >= val_end_idx:
                tprint_warning(f"⚠️ Skipping fold {fold_idx}: invalid boundaries")
                continue
            
            validation_idx = np.arange(val_start_idx, val_end_idx)
            
            # Get timestamps for validation period
            val_start_time = timestamps[val_start_idx]
            val_end_time = timestamps[val_end_idx - 1]
            
            # Calculate purge and embargo boundaries
            purge_threshold = val_start_time - purge_delta
            embargo_threshold = val_end_time + embargo_delta
            
            # Training: all samples NOT in validation, purge, or embargo zones
            # Purge zone: [purge_threshold, val_start_time)
            # Validation zone: [val_start_time, val_end_time]
            # Embargo zone: (val_end_time, embargo_threshold]
            
            train_mask = np.ones(n_samples, dtype=bool)
            
            # Remove validation samples
            train_mask[val_start_idx:val_end_idx] = False
            
            # Remove purge zone (samples before validation within purge window)
            if self.config.enable_purging:
                purge_start_idx = timestamps.searchsorted(purge_threshold, side='left')
                train_mask[purge_start_idx:val_start_idx] = False
                
                # Remove embargo zone (samples after validation within embargo window)
                embargo_end_idx = timestamps.searchsorted(embargo_threshold, side='right')
                train_mask[val_end_idx:embargo_end_idx] = False
                
                purged_samples = (val_start_idx - purge_start_idx)
                embargoed_samples = (embargo_end_idx - val_end_idx)
                
                self.logger.debug(
                    f"Fold {fold_idx}: purged {purged_samples} samples, "
                    f"embargoed {embargoed_samples} samples"
                )
            
            train_idx = np.where(train_mask)[0]
            
            # No test set in k-fold CV
            test_idx = np.array([], dtype=int)
            
            if len(train_idx) == 0:
                tprint_warning(f"⚠️ Fold {fold_idx} has no training samples after purging/embargo")
                continue
            
            # Create SplitResult for this fold
            fold_result = SplitResult(
                train_idx=train_idx,
                validation_idx=validation_idx,
                test_idx=test_idx,
                train_start=timestamps[train_idx[0]],
                train_end=timestamps[train_idx[-1]],
                validation_start=timestamps[validation_idx[0]],
                validation_end=timestamps[validation_idx[-1]],
                test_start=timestamps[-1],  # No test set
                test_end=timestamps[-1],
                strategy=SplitStrategy.PURGED_KFOLD,
                config=self.config,
                metadata={
                    'fold_index': fold_idx,
                    'total_folds': n_folds,
                    'purge_window_hours': self.config.purge_window_hours,
                    'embargo_window_hours': self.config.embargo_window_hours,
                    'purging_enabled': self.config.enable_purging
                }
            )
            
            # Validate distributions if enabled
            if self.config.validate_distribution:
                self._validate_distributions(data, fold_result)
            
            all_folds.append(fold_result)
            
            self.logger.info(
                f"Fold {fold_idx}/{n_folds}: train={len(train_idx)}, "
                f"val={len(validation_idx)} samples"
            )
        
        tprint_success(f"✅ Generated {len(all_folds)} purged k-fold splits")
        
        if len(all_folds) < n_folds:
            tprint_warning(
                f"⚠️ Only generated {len(all_folds)}/{n_folds} folds "
                f"(some folds had insufficient data)"
            )
        
        return all_folds
    
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
        Perform regime-aware splitting that maintains chronological ordering.
        
        Strategy:
        1. First, perform chronological split
        2. Then validate that all regimes are represented in each split
        3. Log warnings if any regime is missing from a split
        
        This approach maintains temporal integrity while being aware of regime distribution.
        If strict regime representation is required across all splits and this cannot be
        achieved chronologically, consider using stratified sampling or adjusting split ratios.
        """
        if self.config.regime_column is None or self.config.regime_column not in data.columns:
            self.logger.warning("Regime column not found, falling back to simple chronological split")
            return self._simple_chronological_split(data)
        
        tprint_info(f"🎯 Performing regime-aware chronological split using column: {self.config.regime_column}")
        
        # First, do standard chronological split
        result = self._simple_chronological_split(data)
        
        # Analyze regime distribution across splits
        regime_col = data[self.config.regime_column]
        all_regimes = set(regime_col.unique())
        
        regime_distribution = {}
        
        for split_name, indices in [
            ('train', result.train_idx),
            ('validation', result.validation_idx),
            ('test', result.test_idx)
        ]:
            if len(indices) == 0:
                regime_distribution[split_name] = {
                    'regimes': set(),
                    'counts': {},
                    'missing_regimes': all_regimes
                }
                continue
            
            split_regimes = regime_col.iloc[indices]
            regime_counts = split_regimes.value_counts().to_dict()
            present_regimes = set(regime_counts.keys())
            missing_regimes = all_regimes - present_regimes
            
            regime_distribution[split_name] = {
                'regimes': present_regimes,
                'counts': regime_counts,
                'missing_regimes': missing_regimes
            }
            
            # Log regime representation
            self.logger.info(f"{split_name.capitalize()} split regime distribution:")
            for regime, count in regime_counts.items():
                pct = 100.0 * count / len(indices)
                self.logger.info(f"  - Regime {regime}: {count} samples ({pct:.1f}%)")
            
            # Warn about missing regimes
            if missing_regimes:
                tprint_warning(
                    f"⚠️ {split_name.capitalize()} split is missing regimes: {missing_regimes}"
                )
                self.logger.warning(
                    f"{split_name.capitalize()} split missing regimes: {missing_regimes}. "
                    f"This may indicate regime transitions at split boundaries."
                )
            
            # Warn about under-represented regimes
            for regime, count in regime_counts.items():
                if count < self.config.min_samples_per_regime:
                    tprint_warning(
                        f"⚠️ Regime {regime} in {split_name} has only {count} samples "
                        f"(minimum: {self.config.min_samples_per_regime})"
                    )
                    self.logger.warning(
                        f"Regime {regime} in {split_name} split has insufficient samples: "
                        f"{count} < {self.config.min_samples_per_regime}"
                    )
        
        # Add regime distribution to metadata
        result.metadata.update({
            'regime_column': self.config.regime_column,
            'all_regimes': list(all_regimes),
            'regime_distribution': {
                split_name: {
                    'present_regimes': list(dist['regimes']),
                    'missing_regimes': list(dist['missing_regimes']),
                    'counts': dist['counts']
                }
                for split_name, dist in regime_distribution.items()
            },
            'all_regimes_represented': all(
                len(dist['missing_regimes']) == 0 
                for dist in regime_distribution.values()
            )
        })
        
        # Summary
        if result.metadata['all_regimes_represented']:
            tprint_success("✅ All regimes represented in all splits")
        else:
            tprint_warning("⚠️ Some regimes missing from some splits (see logs for details)")
        
        self.logger.info(f"Regime-aware chronological split complete: {result.summary()}")
        return result
    
    def _select_validation_columns(
        self, 
        data: pd.DataFrame, 
        numeric_columns: pd.Index, 
        max_cols: int = 10
    ) -> pd.Index:
        """
        Intelligently select columns for distribution validation.
        
        Prioritizes:
        1. High-variance columns (more informative about distribution shifts)
        2. Columns with diverse value ranges
        3. Representative sample across different feature types
        
        Args:
            data: Full DataFrame
            numeric_columns: Index of numeric column names
            max_cols: Maximum number of columns to select
            
        Returns:
            Index of selected column names
        """
        try:
            # Calculate variance for each column (using safe operations)
            variances = {}
            for col in numeric_columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    var = safe_std(col_data) ** 2  # Variance
                    if np.isfinite(var) and var > 0:
                        variances[col] = var
            
            if not variances:
                # Fallback to uniform sampling if variance calculation fails
                selected_indices = np.linspace(0, len(numeric_columns)-1, max_cols, dtype=int)
                return numeric_columns[selected_indices]
            
            # Sort columns by variance (descending)
            sorted_cols = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            
            # Take top max_cols by variance
            selected_cols = [col for col, _ in sorted_cols[:max_cols]]
            
            self.logger.info(
                f"Selected {len(selected_cols)} high-variance columns for distribution validation"
            )
            
            return pd.Index(selected_cols)
            
        except Exception as e:
            tprint_warning(f"⚠️ Column selection failed: {e}, using uniform sampling")
            self.logger.exception("Column selection failed, falling back to uniform sampling:")
            # Fallback to uniform sampling
            selected_indices = np.linspace(0, len(numeric_columns)-1, max_cols, dtype=int)
            return numeric_columns[selected_indices]
    
    def _validate_distributions(self, data: pd.DataFrame, split: SplitResult) -> None:
        """
        Validate that train/validation/test distributions are similar using enhanced utilities.

        This computes comprehensive distribution statistics and correlation analysis
        for numeric columns and logs warnings if distributions shift significantly.
        """
        tprint_info("🧪 Validating distribution consistency across splits")

        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) == 0:
                tprint_warning("⚠️ No numeric columns found for distribution validation")
                self.logger.warning("No numeric columns found for distribution validation")
                return

            # Intelligently select columns if too many
            max_validation_columns = 10
            if len(numeric_columns) > max_validation_columns:
                tprint_info(f"📊 Selecting {max_validation_columns} most informative columns from {len(numeric_columns)} for distribution validation")
                numeric_columns = self._select_validation_columns(data, numeric_columns, max_validation_columns)
                tprint_info(f"✅ Selected columns: {list(numeric_columns)}")

            # Extract data for each split using matrix operations
            train_data = data.iloc[split.train_idx][numeric_columns]
            val_data = data.iloc[split.validation_idx][numeric_columns]

            if len(split.test_idx) > 0:
                test_data = data.iloc[split.test_idx][numeric_columns]
            else:
                test_data = None

            # Optimize data types for computation
            train_data = optimize_dataframe_dtypes(train_data)
            val_data = optimize_dataframe_dtypes(val_data)
            if test_data is not None:
                test_data = optimize_dataframe_dtypes(test_data)

            # Compute comprehensive statistics using parallel processing
            def compute_column_stats(col_data: pd.DataFrame) -> Dict[str, float]:
                """Compute statistics for a single column."""
                col = col_data.columns[0]
                series = col_data[col]

                # Use safe operations for all calculations
                stats = {
                    'mean': safe_mean(series),
                    'std': safe_std(series),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'median': float(series.median()),
                    'skewness': float(series.skew()) if len(series) > 2 else 0.0,
                    'kurtosis': float(series.kurtosis()) if len(series) > 3 else 0.0,
                    'missing_ratio': float(series.isnull().sum() / len(series)),
                }

                # Validate all statistics are finite
                for key, value in stats.items():
                    if not np.isfinite(value):
                        stats[key] = 0.0

                return stats

            # Use parallel processing for column statistics
            tprint_info("🔄 Computing distribution statistics in parallel")
            column_stats = parallel_map(
                lambda col: (col, compute_column_stats(train_data[[col]])),
                numeric_columns.tolist(),
                max_workers=4
            )

            # Process results and compute distribution metrics
            for col, train_stats in column_stats:
                val_stats = compute_column_stats(val_data[[col]])

                # Calculate relative shifts using safe operations
                train_mean = train_stats['mean']
                val_mean = val_stats['mean']

                relative_shift = safe_divide(abs(val_mean - train_mean), abs(train_mean) + 1e-8)

                # Store comprehensive statistics
                split.distribution_metrics.update({
                    f'{col}_train_mean': train_mean,
                    f'{col}_train_std': train_stats['std'],
                    f'{col}_val_mean': val_mean,
                    f'{col}_val_std': val_stats['std'],
                    f'{col}_relative_shift': relative_shift,
                })

                # Check for significant distribution shifts
                if relative_shift > self.config.max_distribution_shift:
                    tprint_warning(
                        f"⚠️ Large distribution shift detected for {col}: "
                        f"{relative_shift:.3f} (threshold: {self.config.max_distribution_shift})"
                    )
                    self.logger.warning(
                        f"Large distribution shift detected for {col}: "
                        f"{relative_shift:.3f} (threshold: {self.config.max_distribution_shift})"
                    )

                # Compute test statistics if available
                if test_data is not None:
                    test_stats = compute_column_stats(test_data[[col]])
                    test_mean = test_stats['mean']
                    test_shift = safe_divide(abs(test_mean - train_mean), abs(train_mean) + 1e-8)

                    split.distribution_metrics.update({
                        f'{col}_test_mean': test_mean,
                        f'{col}_test_std': test_stats['std'],
                        f'{col}_test_shift': test_shift,
                    })

            # Compute correlation matrices using enhanced matrix operations
            try:
                tprint_info("🔗 Computing correlation analysis across splits")
                if len(numeric_columns) > 1:
                    # Compute correlation matrices for each split
                    train_corr = self.matrix_ops.compute_correlation_matrix(train_data.values)
                    val_corr = self.matrix_ops.compute_correlation_matrix(val_data.values)

                    # Validate correlation matrices
                    if validate_correlation_matrix(train_corr) and validate_correlation_matrix(val_corr):
                        # Compute correlation difference
                        corr_diff = np.abs(train_corr - val_corr)
                        mean_corr_shift = safe_mean(corr_diff)

                        split.distribution_metrics['mean_correlation_shift'] = mean_corr_shift

                        if mean_corr_shift > 0.2:  # Threshold for significant correlation shift
                            tprint_warning(f"⚠️ Significant correlation shift detected: {mean_corr_shift:.3f}")

                        # Store correlation matrices if not too large
                        if train_corr.shape[0] <= 20:
                            split.distribution_metrics['train_correlation_matrix'] = train_corr.tolist()
                            split.distribution_metrics['validation_correlation_matrix'] = val_corr.tolist()

            except Exception as e:
                tprint_warning(f"⚠️ Correlation analysis failed: {e}")
                self.logger.exception("Correlation analysis failed with exception:")

            tprint_success(f"✅ Distribution validation completed for {len(numeric_columns)} columns")

        except Exception as e:
            tprint_error(f"❌ Distribution validation failed: {e}")
            self.logger.exception("Distribution validation failed with exception:")


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