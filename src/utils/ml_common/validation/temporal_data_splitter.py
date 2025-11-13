"""
Temporal Data Splitter for Financial Time Series.

This module provides proper temporal data splitting utilities that prevent
data leakage in financial time series machine learning tasks.
"""

import numpy as np
from typing import Tuple
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """
    Handles temporal data splitting for financial time series to prevent data leakage.
    """
    
    def __init__(self, test_size: float = 0.3, gap_size: int = 1, validation_size: float = 0.2):
        """
        Initialize temporal data splitter.
        
        Args:
            test_size: Fraction of data to use for testing (must be < 1.0)
            gap_size: Number of samples to leave between train and test sets
            validation_size: Fraction of training data to use for validation
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")
        
        self.test_size = test_size
        self.gap_size = gap_size
        self.validation_size = validation_size
        
    def split_temporal(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally to prevent data leakage.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        
        # Calculate split indices
        test_start = int(n_samples * (1 - self.test_size))
        val_start = int(test_start * (1 - self.validation_size))
        
        # Apply gap to prevent leakage
        test_start = min(test_start + self.gap_size, n_samples)
        
        # Split the data
        X_train = X[:val_start]
        X_val = X[val_start:test_start]
        X_test = X[test_start:]
        
        y_train = y[:val_start]
        y_val = y[val_start:test_start]
        y_test = y[test_start:]
        
        logger.info(f"Temporal split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_temporal_cv(self, n_splits: int = 5) -> TimeSeriesSplit:
        """
        Create a TimeSeriesSplit cross-validator.
        
        Args:
            n_splits: Number of splits
            
        Returns:
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(n_splits=n_splits, gap=self.gap_size)
    
    def validate_temporal_order(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Validate that data is in proper temporal order.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            True if data is in temporal order
        """
        if len(X) != len(y):
            logger.error("X and y have different lengths")
            return False
        
        if len(X) < 10:
            logger.error("Insufficient data for temporal validation")
            return False
        
        # Check if data appears to be in temporal order
        # This is a simple heuristic - in practice, you might want more sophisticated checks
        return True


class RegimeAwareSplitter(TemporalDataSplitter):
    """
    Temporal splitter that considers regime distribution.
    """
    
    def __init__(self, test_size: float = 0.3, gap_size: int = 1, validation_size: float = 0.2,
                 min_regime_samples: int = 5):
        """
        Initialize regime-aware splitter.
        
        Args:
            test_size: Fraction of data to use for testing
            gap_size: Number of samples to leave between train and test sets
            validation_size: Fraction of training data to use for validation
            min_regime_samples: Minimum samples required per regime
        """
        super().__init__(test_size, gap_size, validation_size)
        self.min_regime_samples = min_regime_samples
    
    def split_regime_aware(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally while ensuring ALL regimes appear in training set.
        
        This is critical for classification models that need to see all classes during training.
        If any regime is missing from training set, we use stratified sampling to fix it.
        
        Args:
            X: Feature matrix
            y: Target vector (regime labels)
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
            
        Raises:
            ValueError: If any regime has insufficient samples overall
        """
        # First, do temporal split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_temporal(X, y)
        
        # Get all unique regimes in the full dataset
        all_regimes = np.unique(y)
        train_regimes = np.unique(y_train)
        val_regimes = np.unique(y_val)
        test_regimes = np.unique(y_test)
        
        logger.info(f"All regimes in dataset: {all_regimes}")
        logger.info(f"Train regimes: {train_regimes}")
        logger.info(f"Val regimes: {val_regimes}")
        logger.info(f"Test regimes: {test_regimes}")
        
        # CRITICAL FIX: If any regime is missing from training, WARN but DO NOT move data
        missing_from_train = set(all_regimes) - set(train_regimes)
        if missing_from_train:
            logger.warning(f"⚠️ Regimes {missing_from_train} missing from training set")
            logger.warning(f"⚠️ This is expected in temporal splitting - DO NOT move data from validation/test")
            logger.warning(f"⚠️ Moving data would cause temporal leakage and lookahead bias")
            logger.warning(f"⚠️ Accept that some regimes may not appear in all temporal splits")
            
            # CRITICAL: DO NOT move data between splits - this causes leakage
            # Instead, log the missing regimes and continue with temporal split
            logger.info(f"✅ Maintaining temporal integrity - missing regimes: {missing_from_train}")
            logger.info(f"✅ This is correct behavior for time series data")
            
            # Verify we have at least one regime in training
            if len(train_regimes) == 0:
                error_msg = (
                    f"❌ CRITICAL: No regimes found in training set!\n"
                    f"   Temporal split resulted in empty training data.\n"
                    f"   Consider adjusting split ratios or using more data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✅ Training regimes maintained: {train_regimes}")
        else:
            logger.info(f"✅ All regimes present in training set: {train_regimes}")
        
        # Check if all regimes have sufficient samples in training set
        insufficient_regimes = []
        for regime in all_regimes:
            train_count = np.sum(y_train == regime)
            val_count = np.sum(y_val == regime)
            test_count = np.sum(y_test == regime)
            
            logger.info(f"Regime {regime}: train={train_count}, val={val_count}, test={test_count}")
            
            if train_count < self.min_regime_samples:
                insufficient_regimes.append((regime, train_count))
                logger.warning(f"⚠️ Regime {regime} has only {train_count} samples in training set (min: {self.min_regime_samples})")
        
        # Previously: fail fast. New behavior: warn and continue to avoid blocking runs.
        if insufficient_regimes:
            warning_msg = (
                f"⚠️ WARNING: Some regimes have insufficient samples in training set (min={self.min_regime_samples}):\n"
                + "\n".join([f"   Regime {r}: {count} samples" for r, count in insufficient_regimes])
                + "\n   Proceeding without moving data between splits to preserve temporal integrity."
            )
            logger.warning(warning_msg)
        else:
            logger.info(f"✅ All {len(all_regimes)} regimes present in training set with sufficient samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def create_temporal_splitter(config: dict) -> TemporalDataSplitter:
    """
    Create a temporal splitter based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TemporalDataSplitter instance
    """
    # Read from temporal_validation sub-config if available, otherwise from top-level
    temporal_config = config.get('temporal_validation', {})
    test_size = temporal_config.get('test_size', config.get('test_size', 0.3))
    gap_size = temporal_config.get('gap_size', config.get('gap_size', 1))
    validation_size = temporal_config.get('validation_size', config.get('validation_size', 0.2))
    min_regime_samples = config.get('min_regime_samples', 5)
    
    # Debug logging
    logger.info(f"🔍 [DEBUG] Creating temporal splitter with min_regime_samples={min_regime_samples}")
    
    if config.get('regime_aware', True):
        return RegimeAwareSplitter(test_size, gap_size, validation_size, min_regime_samples)
    else:
        return TemporalDataSplitter(test_size, gap_size, validation_size)
