"""
Time Split Manager for Pre-Training Pipeline

This module implements temporal data segmentation with proper train/validation/test splits
to prevent lookahead bias and ensure realistic evaluation of trading strategies.

Key Features:
- Chronological splitting (70/20/10 default)
- Purged cross-validation support
- Regime-aware splitting
- Lookahead bias detection
- Distribution analysis per segment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from src.utils.lookahead_bias_detector import LookaheadBiasDetector, validate_no_future_data
from src.utils.purged_kfold import PurgedKFoldTime
from src.utils.ml_common.validation.universal_temporal_validation import (
    UniversalTemporalValidator,
    TemporalValidationConfig,
    UniversalTimeSeriesSplit
)
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


logger = logging.getLogger(__name__)


@dataclass
class TimeSplitConfig:
    """Configuration for time-based data splitting."""
    
    # Split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    
    # Purging and embargo settings
    enable_purging: bool = True
    purge_window: pd.Timedelta = pd.Timedelta(hours=24)  # Purge 24h before validation/test
    embargo_window: pd.Timedelta = pd.Timedelta(hours=12)  # Embargo 12h after validation/test
    
    # Cross-validation settings
    n_cv_splits: int = 5
    enable_walk_forward: bool = True
    
    # Lookahead bias detection
    enable_bias_detection: bool = True
    strict_temporal_order: bool = True
    
    # Regime awareness
    enable_regime_aware_split: bool = False
    min_regime_samples: int = 500
    
    # Validation settings
    min_samples_per_split: int = 1000
    validate_distribution: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        if self.train_ratio < 0.5:
            raise ValueError(f"Train ratio too small: {self.train_ratio}")
        
        if self.validation_ratio < 0.05:
            raise ValueError(f"Validation ratio too small: {self.validation_ratio}")


@dataclass
class SplitMetadata:
    """Metadata for a data split."""
    
    split_type: str  # 'train', 'val', 'test'
    start_time: datetime
    end_time: datetime
    n_samples: int
    timestamp_range: str
    
    # Distribution statistics
    target_volatility: Optional[Dict[str, float]] = None
    target_distribution: Optional[Dict[str, Any]] = None
    regime_distribution: Optional[Dict[str, int]] = None
    
    # Quality metrics
    missing_ratio: float = 0.0
    outlier_ratio: float = 0.0
    data_quality_score: float = 1.0
    
    # Lookahead validation
    temporal_order_valid: bool = True
    lookahead_bias_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat() if self.start_time else None
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result


class TimeSplitManager:
    """
    Time Split Manager for preventing lookahead bias and ensuring proper temporal segmentation.
    
    This class implements:
    1. Chronological train/val/test splitting
    2. Purged K-fold cross-validation
    3. Regime-aware splitting
    4. Distribution validation per segment
    5. Lookahead bias detection
    """
    
    def __init__(self, config: Optional[TimeSplitConfig] = None):
        """
        Initialize Time Split Manager.
        
        Args:
            config: Time split configuration
        """
        self.config = config or TimeSplitConfig()
        
        # Initialize lookahead bias detector
        self.bias_detector = LookaheadBiasDetector(strict_mode=self.config.strict_temporal_order)
        
        # Initialize temporal validator
        temporal_config = TemporalValidationConfig(
            enable_temporal_checks=True,
            strict_temporal_order=self.config.strict_temporal_order,
            n_splits=self.config.n_cv_splits,
            enable_walk_forward=self.config.enable_walk_forward
        )
        self.temporal_validator = UniversalTemporalValidator(temporal_config)
        
        # Storage for split metadata
        self.split_metadata: Dict[str, SplitMetadata] = {}
        
        tprint_success("✅ TimeSplitManager initialized")
        tprint_info(f"   → Split ratios: {self.config.train_ratio:.1%}/{self.config.validation_ratio:.1%}/{self.config.test_ratio:.1%}")
        tprint_info(f"   → Purging: {self.config.enable_purging} (window={self.config.purge_window})")
        tprint_info(f"   → Embargo: {self.config.embargo_window}")
        tprint_info(f"   → Temporal validation: {self.config.enable_bias_detection}")
    
    def create_temporal_split(
        self,
        data: pd.DataFrame,
        timestamp_column: str = 'timestamp',
        target_columns: Optional[List[str]] = None,
        regime_column: Optional[str] = None,
        data_split: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create chronological train/validation/test splits.
        
        Args:
            data: Input DataFrame with timestamp index or column
            timestamp_column: Name of timestamp column (if not index)
            target_columns: List of target column names for distribution analysis
            regime_column: Optional regime column for regime-aware splitting
            data_split: If specified, return only this split ('train', 'val', 'test')
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        try:
            tprint_info(f"📊 Creating temporal split for {len(data)} samples")
            
            # Prepare data with proper timestamp index
            prepared_data = self._prepare_temporal_data(data, timestamp_column)
            
            # Validate temporal order
            if self.config.enable_bias_detection:
                self._validate_temporal_order(prepared_data)
            
            # Create chronological splits
            splits = self._create_chronological_splits(prepared_data)
            
            # Apply purging if enabled
            if self.config.enable_purging:
                splits = self._apply_purging(splits)
            
            # Validate splits
            self._validate_splits(splits, target_columns, regime_column)
            
            # Generate metadata for each split
            for split_name, split_data in splits.items():
                metadata = self._generate_split_metadata(
                    split_data, split_name, target_columns, regime_column
                )
                self.split_metadata[split_name] = metadata
            
            # Log split information
            self._log_split_summary(splits)
            
            # Return requested split or all splits
            if data_split:
                if data_split not in splits:
                    raise ValueError(f"Invalid data_split: {data_split}. Must be 'train', 'val', or 'test'")
                return {data_split: splits[data_split]}
            
            return splits
            
        except Exception as e:
            tprint_error(f"❌ Error creating temporal split: {e}")
            raise
    
    def _prepare_temporal_data(
        self, 
        data: pd.DataFrame, 
        timestamp_column: str
    ) -> pd.DataFrame:
        """Prepare data with proper temporal index."""
        df = data.copy()
        
        # Ensure timestamp is index
        if timestamp_column in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index(timestamp_column)
        
        # Convert index to datetime if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {e}")
        
        # Sort by time
        df = df.sort_index()
        
        # Remove duplicates
        if df.index.has_duplicates:
            n_duplicates = df.index.duplicated().sum()
            tprint_warning(f"⚠️ Removing {n_duplicates} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def _validate_temporal_order(self, data: pd.DataFrame):
        """Validate temporal ordering of data."""
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index is not monotonically increasing")
        
        # Check for gaps
        if len(data) > 1:
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            
            # Detect large gaps (>10x median)
            large_gaps = time_diffs[time_diffs > median_diff * 10]
            if len(large_gaps) > 0:
                tprint_warning(f"⚠️ Detected {len(large_gaps)} large time gaps in data")
    
    def _create_chronological_splits(
        self, 
        data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Create chronological train/val/test splits."""
        n_samples = len(data)
        
        # Calculate split indices
        train_end_idx = int(n_samples * self.config.train_ratio)
        val_end_idx = train_end_idx + int(n_samples * self.config.validation_ratio)
        
        # Create splits
        splits = {
            'train': data.iloc[:train_end_idx].copy(),
            'val': data.iloc[train_end_idx:val_end_idx].copy(),
            'test': data.iloc[val_end_idx:].copy()
        }
        
        # Validate minimum samples
        for split_name, split_data in splits.items():
            if len(split_data) < self.config.min_samples_per_split:
                tprint_warning(
                    f"⚠️ Split '{split_name}' has only {len(split_data)} samples "
                    f"(minimum: {self.config.min_samples_per_split})"
                )
        
        return splits
    
    def _apply_purging(
        self, 
        splits: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply purging to remove samples close to validation/test boundaries.
        
        This prevents label leakage from overlapping prediction horizons.
        """
        tprint_info("🧹 Applying purging and embargo...")
        
        purged_splits = {}
        
        # Get boundary times
        train_data = splits['train']
        val_data = splits['val']
        test_data = splits['test']
        
        val_start_time = val_data.index.min()
        test_start_time = test_data.index.min()
        
        # Purge training data before validation
        train_purge_cutoff = val_start_time - self.config.purge_window
        purged_train = train_data[train_data.index <= train_purge_cutoff]
        n_purged_train = len(train_data) - len(purged_train)
        
        # Purge validation data before test
        val_purge_cutoff = test_start_time - self.config.purge_window
        purged_val = val_data[val_data.index <= val_purge_cutoff]
        n_purged_val = len(val_data) - len(purged_val)
        
        purged_splits['train'] = purged_train
        purged_splits['val'] = purged_val
        purged_splits['test'] = test_data
        
        tprint_info(f"   → Purged {n_purged_train} samples from train")
        tprint_info(f"   → Purged {n_purged_val} samples from validation")
        
        return purged_splits
    
    def _validate_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        target_columns: Optional[List[str]] = None,
        regime_column: Optional[str] = None
    ):
        """Validate that splits maintain proper temporal order and distribution."""
        # Check temporal order between splits
        train_max_time = splits['train'].index.max()
        val_min_time = splits['val'].index.min()
        val_max_time = splits['val'].index.max()
        test_min_time = splits['test'].index.min()
        
        if train_max_time >= val_min_time:
            raise ValueError(f"Temporal order violation: train_max ({train_max_time}) >= val_min ({val_min_time})")
        
        if val_max_time >= test_min_time:
            raise ValueError(f"Temporal order violation: val_max ({val_max_time}) >= test_min ({test_min_time})")
        
        # Validate distributions if enabled
        if self.config.validate_distribution and target_columns:
            self._validate_distribution_similarity(splits, target_columns)
    
    def _validate_distribution_similarity(
        self,
        splits: Dict[str, pd.DataFrame],
        target_columns: List[str]
    ):
        """Validate that target distributions are similar across splits."""
        for target_col in target_columns:
            if target_col not in splits['train'].columns:
                continue
            
            train_dist = splits['train'][target_col].dropna()
            val_dist = splits['val'][target_col].dropna()
            test_dist = splits['test'][target_col].dropna()
            
            if len(train_dist) == 0 or len(val_dist) == 0 or len(test_dist) == 0:
                continue
            
            # Calculate distribution statistics
            train_mean, train_std = train_dist.mean(), train_dist.std()
            val_mean, val_std = val_dist.mean(), val_dist.std()
            test_mean, test_std = test_dist.mean(), test_dist.std()
            
            # Check for significant distribution shifts
            mean_shift_val = abs(val_mean - train_mean) / (train_std + 1e-8)
            mean_shift_test = abs(test_mean - train_mean) / (train_std + 1e-8)
            
            if mean_shift_val > 2.0:
                tprint_warning(
                    f"⚠️ Large distribution shift in '{target_col}' "
                    f"(train→val: {mean_shift_val:.2f}σ)"
                )
            
            if mean_shift_test > 2.0:
                tprint_warning(
                    f"⚠️ Large distribution shift in '{target_col}' "
                    f"(train→test: {mean_shift_test:.2f}σ)"
                )
    
    def _generate_split_metadata(
        self,
        split_data: pd.DataFrame,
        split_name: str,
        target_columns: Optional[List[str]] = None,
        regime_column: Optional[str] = None
    ) -> SplitMetadata:
        """Generate metadata for a data split."""
        # Basic metadata
        metadata = SplitMetadata(
            split_type=split_name,
            start_time=split_data.index.min().to_pydatetime(),
            end_time=split_data.index.max().to_pydatetime(),
            n_samples=len(split_data),
            timestamp_range=f"{split_data.index.min()} to {split_data.index.max()}"
        )
        
        # Target distribution
        if target_columns:
            target_dist = {}
            target_vol = {}
            
            for col in target_columns:
                if col in split_data.columns:
                    col_data = split_data[col].dropna()
                    if len(col_data) > 0:
                        target_dist[col] = {
                            'mean': float(col_data.mean()),
                            'std': float(col_data.std()),
                            'min': float(col_data.min()),
                            'max': float(col_data.max()),
                            'skew': float(col_data.skew()),
                            'kurtosis': float(col_data.kurtosis())
                        }
                        target_vol[col] = float(col_data.std())
            
            metadata.target_distribution = target_dist
            metadata.target_volatility = target_vol
        
        # Regime distribution
        if regime_column and regime_column in split_data.columns:
            regime_dist = split_data[regime_column].value_counts().to_dict()
            metadata.regime_distribution = regime_dist
        
        # Data quality
        missing_ratio = split_data.isnull().sum().sum() / (len(split_data) * len(split_data.columns))
        metadata.missing_ratio = float(missing_ratio)
        
        # Calculate quality score
        quality_score = 1.0
        if missing_ratio > 0.1:
            quality_score -= 0.3
        if missing_ratio > 0.3:
            quality_score -= 0.3
        
        metadata.data_quality_score = max(0.0, quality_score)
        
        return metadata
    
    def _log_split_summary(self, splits: Dict[str, pd.DataFrame]):
        """Log summary of splits."""
        tprint_success("✅ Temporal splits created:")
        
        for split_name, split_data in splits.items():
            metadata = self.split_metadata.get(split_name)
            if metadata:
                tprint_info(
                    f"   → {split_name.upper()}: {metadata.n_samples} samples "
                    f"({metadata.start_time.strftime('%Y-%m-%d')} to "
                    f"{metadata.end_time.strftime('%Y-%m-%d')})"
                )
    
    def create_purged_kfold_cv(
        self,
        data: pd.DataFrame,
        data_split: str = 'train'
    ) -> PurgedKFoldTime:
        """
        Create purged K-fold cross-validator for nested validation.
        
        Args:
            data: Input data (should be training data)
            data_split: Which split to use ('train' recommended)
        
        Returns:
            PurgedKFoldTime cross-validator
        """
        tprint_info(f"🔄 Creating purged K-fold CV with {self.config.n_cv_splits} splits")
        
        return PurgedKFoldTime(
            n_splits=self.config.n_cv_splits,
            purge=self.config.purge_window,
            embargo=self.config.embargo_window
        )
    
    def get_split_metadata(self, split_name: str) -> Optional[SplitMetadata]:
        """Get metadata for a specific split."""
        return self.split_metadata.get(split_name)
    
    def export_split_metadata(self, output_path: Union[str, Path]) -> Path:
        """
        Export split metadata to JSON file.
        
        Args:
            output_path: Output file path
        
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata_dict = {
            'config': asdict(self.config),
            'splits': {
                name: metadata.to_dict()
                for name, metadata in self.split_metadata.items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
        
        tprint_success(f"✅ Split metadata exported to {output_path}")
        return output_path
    
    def validate_no_lookahead(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, bool]:
        """
        Validate that there's no lookahead bias between splits.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'temporal_order_valid': True,
            'no_overlap': True,
            'proper_gaps': True,
            'all_checks_passed': True
        }
        
        # Check temporal order
        if train_data.index.max() >= val_data.index.min():
            results['temporal_order_valid'] = False
            results['all_checks_passed'] = False
            tprint_error("❌ Temporal order violation: train overlaps with validation")
        
        if val_data.index.max() >= test_data.index.min():
            results['temporal_order_valid'] = False
            results['all_checks_passed'] = False
            tprint_error("❌ Temporal order violation: validation overlaps with test")
        
        # Check for sample overlap
        train_indices = set(train_data.index)
        val_indices = set(val_data.index)
        test_indices = set(test_data.index)
        
        train_val_overlap = train_indices.intersection(val_indices)
        val_test_overlap = val_indices.intersection(test_indices)
        train_test_overlap = train_indices.intersection(test_indices)
        
        if train_val_overlap or val_test_overlap or train_test_overlap:
            results['no_overlap'] = False
            results['all_checks_passed'] = False
            tprint_error(f"❌ Sample overlap detected: {len(train_val_overlap) + len(val_test_overlap) + len(train_test_overlap)} samples")
        
        if results['all_checks_passed']:
            tprint_success("✅ All lookahead bias checks passed")
        
        return results


def create_time_split_manager(config: Optional[TimeSplitConfig] = None) -> TimeSplitManager:
    """
    Factory function to create TimeSplitManager.
    
    Args:
        config: Optional configuration
    
    Returns:
        TimeSplitManager instance
    """
    return TimeSplitManager(config)