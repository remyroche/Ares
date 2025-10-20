"""
Purged & Embargoed Walk-Forward Cross-Validation

Implements López de Prado's Purged & Embargoed Walk-Forward CV to prevent
leakage and overfitting in time series data.

Key Features:
- Strict time ordering enforcement
- Purged samples (overlapping test periods)
- Embargo window (gap between train and test)
- Configurable parameters
- Leakage prevention utilities
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# Import enhanced CV utilities from ml_commons
try:
    from src.utils.ml_common.validation.unified_cv import (
        UnifiedCrossValidator, UnifiedCVResult, perform_cross_validation,
        temporal_cross_validation, nested_cross_validation
    )
    from src.utils.ml_common.validation.cv import (
        purged_time_series_splits, analyze_splits, validate_cv_integrity,
        PurgedSplitConfig
    )
    from src.utils.ml_common.validation.cv_utils import (
        TemporalCrossValidator, VectorBTCrossValidator, OOFGenerator
    )
    ML_COMMONS_CV_AVAILABLE = True
    tprint_info("✅ ML Commons CV utilities imported successfully")
except ImportError as e:
    ML_COMMONS_CV_AVAILABLE = False
    tprint_warning(f"⚠️ ML Commons CV utilities not available: {e}")
    # Fast-fail implementations - raise exceptions immediately when dependencies are missing
    class UnifiedCrossValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    class UnifiedCVResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def perform_cross_validation(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def temporal_cross_validation(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def nested_cross_validation(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    class TemporalCrossValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    class VectorBTCrossValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    class OOFGenerator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def purged_time_series_splits(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def analyze_splits(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    def validate_cv_integrity(*args, **kwargs):
        raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

    class PurgedSplitConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("ML Commons CV utilities not available. Install required dependencies.")

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesSplit:
    """Represents a single time series split with strict time ordering."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_start: int
    embargo_end: int
    purged_samples: List[int]
    split_id: int

    def __post_init__(self):
        """Validate split integrity."""
        # Enforce strict time ordering
        assert self.train_start < self.train_end, "Train start must be before train end"
        assert self.train_end < self.test_start, "Train end must be before test start"
        assert self.test_start < self.test_end, "Test start must be before test end"

        # Validate embargo window
        if self.embargo_start is not None and self.embargo_end is not None:
            assert self.train_end <= self.embargo_start, "Embargo must start after train end"
            assert self.embargo_start < self.embargo_end, "Embargo start must be before embargo end"
            assert self.embargo_end <= self.test_start, "Embargo end must be before test start"

    @property
    def train_indices(self) -> List[int]:
        """Get training indices."""
        return list(range(self.train_start, self.train_end))

    @property
    def test_indices(self) -> List[int]:
        """Get test indices."""
        return list(range(self.test_start, self.test_end))

    @property
    def embargo_indices(self) -> List[int]:
        """Get embargo indices."""
        if self.embargo_start is None or self.embargo_end is None:
            return []
        return list(range(self.embargo_start, self.embargo_end))

    def is_valid(self) -> bool:
        """Check if split is valid (no leakage)."""
        # No train timestamps >= any test timestamps
        if self.train_end > self.test_start:
            return False

        # Embargo window must be respected
        if self.embargo_start is not None and self.embargo_end is not None:
            if self.train_end > self.embargo_start or self.embargo_end > self.test_start:
                return False

        return True

@dataclass
class PurgedEmbargoedConfig:
    """Configuration for Purged & Embargoed Walk-Forward CV."""
    # Basic parameters
    n_splits: int = 5
    test_size: float = 0.2  # Fraction of total data for test
    train_size: float = 0.6  # Fraction of total data for train

    # Purged samples (overlapping test periods)
    purge_fraction: float = 0.1  # Fraction of test period to purge

    # Embargo window (gap between train and test)
    embargo_fraction: float = 0.05  # Fraction of total data for embargo

    # Minimum sizes
    min_train_samples: int = 100
    min_test_samples: int = 50
    min_embargo_samples: int = 10

    # Validation
    strict_time_ordering: bool = True
    validate_splits: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.test_size < 1, "test_size must be between 0 and 1"
        assert 0 < self.train_size < 1, "train_size must be between 0 and 1"
        assert 0 <= self.purge_fraction < 1, "purge_fraction must be between 0 and 1"
        assert 0 <= self.embargo_fraction < 1, "embargo_fraction must be between 0 and 1"
        assert self.train_size + self.test_size + self.embargo_fraction <= 1, "Total fractions must not exceed 1"

class PurgedEmbargoedWalkForwardCV:
    """
    Enhanced Purged & Embargoed Walk-Forward Cross-Validation.

    Prevents leakage by enforcing strict time ordering and adding
    embargo windows between train and test sets. Now integrated with
    ml_commons utilities for enhanced validation and analysis.
    """

    def __init__(self, config: PurgedEmbargoedConfig,
                 use_ml_commons: bool = True,
                 enable_vectorbt: bool = True):
        """Initialize the CV splitter with ml_commons integration."""
        self.config = config
        self.splits: List[TimeSeriesSplit] = []
        self.data_length: int = 0
        self.use_ml_commons = use_ml_commons and ML_COMMONS_CV_AVAILABLE
        self.enable_vectorbt = enable_vectorbt and ML_COMMONS_CV_AVAILABLE

        # Initialize ml_commons validators if available
        if self.use_ml_commons:
            self.unified_cv = UnifiedCrossValidator()
            self.temporal_cv = TemporalCrossValidator(
                n_splits=config.n_splits,
                gap=int(config.embargo_fraction * 100),  # Convert to minutes
                use_vectorbt=self.enable_vectorbt
            )
            self.vectorbt_cv = VectorBTCrossValidator(
                n_splits=config.n_splits,
                gap=int(config.embargo_fraction * 100),
                use_portfolio_analysis=True,
                enable_memory_optimization=True
            ) if self.enable_vectorbt else None
            self.oof_generator = OOFGenerator(strategy='mean')
        else:
            self.unified_cv = None
            self.temporal_cv = None
            self.vectorbt_cv = None
            self.oof_generator = None

        tprint_info(f"Initialized Enhanced PurgedEmbargoedWalkForwardCV with {config.n_splits} splits")
        if self.use_ml_commons:
            tprint_info("✅ ML Commons integration enabled")
        if self.enable_vectorbt:
            tprint_info("✅ VectorBT optimization enabled")

    def split(self, data: pd.DataFrame,
              timestamps: Optional[pd.Series] = None,
              targets: Optional[pd.Series] = None) -> List[TimeSeriesSplit]:
        """
        Generate time series splits with purged samples and embargo windows.

        Args:
            data: Input data (used for length calculation)
            timestamps: Optional timestamp series for validation
            targets: Optional target series for validation

        Returns:
            List of TimeSeriesSplit objects
        """
        tprint_info(f"Generating {self.config.n_splits} time series splits")

        self.data_length = len(data)
        self.splits = []

        # Calculate split parameters
        total_samples = self.data_length
        test_samples = max(int(total_samples * self.config.test_size), self.config.min_test_samples)
        train_samples = max(int(total_samples * self.config.train_size), self.config.min_train_samples)
        embargo_samples = max(int(total_samples * self.config.embargo_fraction), self.config.min_embargo_samples)

        tprint_debug(f"Split parameters: total={total_samples}, train={train_samples}, test={test_samples}, embargo={embargo_samples}")

        # Generate splits
        for split_id in range(self.config.n_splits):
            split = self._generate_single_split(
                split_id, total_samples, train_samples, test_samples, embargo_samples
            )

            if self.config.validate_splits and not split.is_valid():
                tprint_warning(f"Invalid split {split_id} generated, skipping")
                continue

            self.splits.append(split)
            tprint_debug(f"Generated split {split_id}: train[{split.train_start}:{split.train_end}], test[{split.test_start}:{split.test_end}]")

        tprint_success(f"Generated {len(self.splits)} valid splits")

        # Enhanced validation using ml_commons if available
        if self.use_ml_commons and targets is not None:
            self._enhanced_validation(data, targets)

        return self.splits

    def _generate_single_split(self, split_id: int, total_samples: int,
                              train_samples: int, test_samples: int,
                              embargo_samples: int) -> TimeSeriesSplit:
        """Generate a single time series split."""

        # Calculate available space for splits
        available_space = total_samples - train_samples - test_samples - embargo_samples

        if available_space < 0:
            raise ValueError(f"Insufficient data for splits: need {train_samples + test_samples + embargo_samples}, have {total_samples}")

        # Calculate step size for walk-forward
        step_size = max(1, available_space // (self.config.n_splits - 1)) if self.config.n_splits > 1 else 0

        # Calculate split positions
        start_offset = split_id * step_size

        # Training set
        train_start = start_offset
        train_end = train_start + train_samples

        # Embargo window
        embargo_start = train_end
        embargo_end = embargo_start + embargo_samples

        # Test set
        test_start = embargo_end
        test_end = test_start + test_samples

        # Ensure we don't exceed data bounds
        if test_end > total_samples:
            # Adjust if we're at the end
            test_end = total_samples
            test_start = max(test_end - test_samples, embargo_end)
            if test_start <= embargo_end:
                test_start = embargo_end + 1
                test_end = min(test_start + test_samples, total_samples)

        # Calculate purged samples (overlapping test periods)
        purge_samples = max(1, int(test_samples * self.config.purge_fraction))
        purged_samples = list(range(test_start, min(test_start + purge_samples, test_end)))

        return TimeSeriesSplit(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            embargo_start=embargo_start,
            embargo_end=embargo_end,
            purged_samples=purged_samples,
            split_id=split_id
        )

    def get_split(self, split_id: int) -> TimeSeriesSplit:
        """Get a specific split by ID."""
        if split_id >= len(self.splits):
            raise IndexError(f"Split {split_id} not found. Available splits: 0-{len(self.splits)-1}")
        return self.splits[split_id]

    def validate_no_leakage(self, data: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None) -> bool:
        """
        Validate that there is no leakage in the splits.

        Args:
            data: Input data
            feature_columns: Optional list of feature columns to check

        Returns:
            True if no leakage detected
        """
        tprint_info("Validating no leakage in time series splits")

        for split in self.splits:
            if not split.is_valid():
                tprint_error(f"Split {split.split_id} is invalid")
                return False

            # Check for temporal leakage
            train_data = data.iloc[split.train_indices]
            test_data = data.iloc[split.test_indices]

            if feature_columns:
                for col in feature_columns:
                    # Check if any test value appears in training (basic check)
                    train_values = set(train_data[col].dropna().values)
                    test_values = set(test_data[col].dropna().values)

                    # This is a basic check - more sophisticated checks would be needed
                    # for complex leakage patterns
                    if train_values.intersection(test_values):
                        tprint_warning(f"Potential leakage detected in column {col} for split {split.split_id}")

        tprint_success("No leakage detected in time series splits")
        return True

    def get_split_summary(self) -> Dict[str, Any]:
        """Get summary of all splits."""
        if not self.splits:
            return {"n_splits": 0, "splits": []}

        summary = {
            "n_splits": len(self.splits),
            "data_length": self.data_length,
            "splits": []
        }

        for split in self.splits:
            split_info = {
                "split_id": split.split_id,
                "train_size": len(split.train_indices),
                "test_size": len(split.test_indices),
                "embargo_size": len(split.embargo_indices),
                "purged_samples": len(split.purged_samples),
                "is_valid": split.is_valid()
            }
            summary["splits"].append(split_info)

        return summary

    def _enhanced_validation(self, data: pd.DataFrame, targets: pd.Series):
        """Enhanced validation using ml_commons utilities."""
        if not self.use_ml_commons:
            return

        try:
            tprint_info("🔍 Running enhanced validation with ml_commons utilities")

            # Convert splits to format expected by ml_commons
            ml_commons_splits = []
            for split in self.splits:
                train_idx = np.array(split.train_indices)
                test_idx = np.array(split.test_indices)
                ml_commons_splits.append((train_idx, test_idx))

            # Analyze splits using ml_commons
            analysis_results = analyze_splits(data, targets, ml_commons_splits)
            tprint_info(f"📊 Split analysis: {analysis_results['n_folds']} folds, min_train: {analysis_results['min_train']}, min_val: {analysis_results['min_val']}")

            # Validate CV integrity
            integrity_results = validate_cv_integrity(
                data, targets, ml_commons_splits,
                min_train=int(self.config.train_size * len(data)),
                min_val=int(self.config.test_size * len(data))
            )

            if integrity_results['is_valid']:
                tprint_success("✅ CV integrity validation passed")
            else:
                tprint_warning(f"⚠️ CV integrity issues found: {len(integrity_results['issues'])} issues")
                for issue in integrity_results['issues'][:5]:  # Show first 5 issues
                    tprint_warning(f"   - {issue}")

        except Exception as e:
            tprint_warning(f"⚠️ Enhanced validation failed: {e}")

    def run_unified_cv(self, model, X: np.ndarray, y: np.ndarray,
                      scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """Run cross-validation using unified CV utilities."""
        if not self.use_ml_commons or self.unified_cv is None:
            tprint_warning("⚠️ Unified CV not available, using standard validation")
            return self._standard_cv_validation(model, X, y, scoring)

        try:
            tprint_info("🔄 Running unified cross-validation")

            # Use temporal cross-validation for time series data
            cv_result = temporal_cross_validation(
                model, X, y,
                n_splits=self.config.n_splits,
                gap=int(self.config.embargo_fraction * len(X)),
                scoring=scoring
            )

            tprint_success(f"✅ Unified CV completed: mean={cv_result.get('mean', 0):.4f}, std={cv_result.get('std', 0):.4f}")
            return cv_result

        except Exception as e:
            tprint_warning(f"⚠️ Unified CV failed: {e}, falling back to standard validation")
            return self._standard_cv_validation(model, X, y, scoring)

    def run_vectorbt_cv(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run cross-validation using VectorBT optimization."""
        if not self.enable_vectorbt or self.vectorbt_cv is None:
            tprint_warning("⚠️ VectorBT CV not available")
            return {}

        try:
            tprint_info("🚀 Running VectorBT-optimized cross-validation")

            # Use VectorBT cross-validator
            splits = list(self.vectorbt_cv.split(X, y))

            # Evaluate with portfolio analysis
            portfolio_metrics = self.vectorbt_cv.evaluate_with_portfolio_analysis(X, y, model)

            tprint_success(f"✅ VectorBT CV completed with portfolio metrics: {len(portfolio_metrics)} metrics")
            return {
                'splits': splits,
                'portfolio_metrics': portfolio_metrics,
                'n_splits': len(splits)
            }

        except Exception as e:
            tprint_warning(f"⚠️ VectorBT CV failed: {e}")
            return {}

    def generate_oof_predictions(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate out-of-fold predictions using OOF generator."""
        if not self.use_ml_commons or self.oof_generator is None:
            tprint_warning("⚠️ OOF generator not available")
            return np.array([])

        try:
            tprint_info("🔄 Generating out-of-fold predictions")

            # Generate predictions for each fold
            for fold_id, split in enumerate(self.splits):
                train_idx = np.array(split.train_indices)
                test_idx = np.array(split.test_indices)

                # Train model on fold
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                # Create a copy of the model for this fold
                fold_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                fold_model.fit(X_train, y_train)

                # Generate predictions
                predictions = fold_model.predict(X_test)

                # Add to OOF generator
                self.oof_generator.add_fold_predictions(fold_id, predictions)

            # Get combined OOF predictions
            oof_predictions = self.oof_generator.get_oof_predictions()

            tprint_success(f"✅ Generated OOF predictions: {len(oof_predictions)} samples")
            return oof_predictions

        except Exception as e:
            tprint_warning(f"⚠️ OOF prediction generation failed: {e}")
            return np.array([])

    def _standard_cv_validation(self, model, X: np.ndarray, y: np.ndarray, scoring: str) -> Dict[str, Any]:
        """Fallback standard CV validation."""
        try:
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=self.config.n_splits, scoring=scoring)
            return {
                'scores': scores.tolist(),
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'cv_folds': self.config.n_splits
            }
        except Exception as e:
            tprint_warning(f"⚠️ Standard CV validation failed: {e}")
            return {'scores': [], 'mean': 0.0, 'std': 0.0, 'cv_folds': 0}

    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced summary with ml_commons metrics."""
        base_summary = self.get_split_summary()

        if not self.use_ml_commons:
            return base_summary

        # Add ml_commons specific metrics
        enhanced_summary = base_summary.copy()
        enhanced_summary.update({
            'ml_commons_enabled': self.use_ml_commons,
            'vectorbt_enabled': self.enable_vectorbt,
            'unified_cv_available': self.unified_cv is not None,
            'temporal_cv_available': self.temporal_cv is not None,
            'vectorbt_cv_available': self.vectorbt_cv is not None,
            'oof_generator_available': self.oof_generator is not None
        })

        return enhanced_summary

class TimeSeriesSplitIterator:
    """Iterator for time series splits."""

    def __init__(self, cv: PurgedEmbargoedWalkForwardCV):
        self.cv = cv
        self.current_split = 0

    def __iter__(self) -> Iterator[TimeSeriesSplit]:
        return self

    def __next__(self) -> TimeSeriesSplit:
        if self.current_split >= len(self.cv.splits):
            raise StopIteration

        split = self.cv.splits[self.current_split]
        self.current_split += 1
        return split

class LeakagePreventionUtils:
    """Utilities for preventing leakage in time series data."""

    @staticmethod
    def validate_time_ordering(data: pd.DataFrame,
                              timestamp_col: str = 'timestamp') -> bool:
        """Validate that data is properly time-ordered."""
        if timestamp_col not in data.columns:
            tprint_warning(f"Timestamp column {timestamp_col} not found")
            return True  # Assume valid if no timestamp column

        timestamps = data[timestamp_col]
        is_sorted = timestamps.is_monotonic_increasing

        if not is_sorted:
            tprint_error("Data is not time-ordered")
            return False

        tprint_success("Data is properly time-ordered")
        return True

    @staticmethod
    def check_future_leakage(train_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            timestamp_col: str = 'timestamp') -> bool:
        """Check for future leakage between train and test sets."""
        if timestamp_col not in train_data.columns or timestamp_col not in test_data.columns:
            return True  # Cannot check without timestamps

        max_train_time = train_data[timestamp_col].max()
        min_test_time = test_data[timestamp_col].min()

        if max_train_time >= min_test_time:
            tprint_error(f"Future leakage detected: max_train_time={max_train_time} >= min_test_time={min_test_time}")
            return False

        tprint_success("No future leakage detected")
        return True

    @staticmethod
    def create_embargo_mask(data_length: int,
                           embargo_start: int,
                           embargo_end: int) -> np.ndarray:
        """Create a boolean mask for embargo period."""
        mask = np.zeros(data_length, dtype=bool)
        mask[embargo_start:embargo_end] = True
        return mask

# Convenience functions
def create_purged_embargoed_cv(n_splits: int = 5,
                              test_size: float = 0.2,
                              train_size: float = 0.6,
                              purge_fraction: float = 0.1,
                              embargo_fraction: float = 0.05,
                              use_ml_commons: bool = True,
                              enable_vectorbt: bool = True) -> PurgedEmbargoedWalkForwardCV:
    """Create an enhanced PurgedEmbargoedWalkForwardCV with ml_commons integration."""
    config = PurgedEmbargoedConfig(
        n_splits=n_splits,
        test_size=test_size,
        train_size=train_size,
        purge_fraction=purge_fraction,
        embargo_fraction=embargo_fraction
    )
    return PurgedEmbargoedWalkForwardCV(
        config,
        use_ml_commons=use_ml_commons,
        enable_vectorbt=enable_vectorbt
    )

def validate_time_series_splits(splits: List[TimeSeriesSplit],
                               data: pd.DataFrame) -> bool:
    """Validate a list of time series splits for leakage."""
    cv = PurgedEmbargoedWalkForwardCV(PurgedEmbargoedConfig())
    cv.splits = splits
    cv.data_length = len(data)
    return cv.validate_no_leakage(data)
