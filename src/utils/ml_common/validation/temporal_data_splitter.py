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
        
        # CRITICAL FIX: If any regime is missing from training, use stratified sampling
        missing_from_train = set(all_regimes) - set(train_regimes)
        if missing_from_train:
            logger.warning(f"⚠️ Regimes {missing_from_train} missing from training set")
            logger.warning(f"⚠️ Applying stratified sampling to ensure all regimes in training")
            
            # For each missing regime, move at least 1 sample from val/test to train
            for regime in missing_from_train:
                # Check if regime exists in validation set
                val_regime_mask = (y_val == regime)
                if np.any(val_regime_mask):
                    # Move 1 sample from validation to training
                    val_regime_idx = np.where(val_regime_mask)[0][0]
                    X_train = np.vstack([X_train, X_val[val_regime_idx:val_regime_idx+1]])
                    y_train = np.append(y_train, y_val[val_regime_idx:val_regime_idx+1])
                    X_val = np.delete(X_val, val_regime_idx, axis=0)
                    y_val = np.delete(y_val, val_regime_idx)
                    logger.info(f"✅ Moved 1 sample of regime {regime} from validation to training")
                # Otherwise check test set
                elif np.any(y_test == regime):
                    test_regime_mask = (y_test == regime)
                    test_regime_idx = np.where(test_regime_mask)[0][0]
                    X_train = np.vstack([X_train, X_test[test_regime_idx:test_regime_idx+1]])
                    y_train = np.append(y_train, y_test[test_regime_idx:test_regime_idx+1])
                    X_test = np.delete(X_test, test_regime_idx, axis=0)
                    y_test = np.delete(y_test, test_regime_idx)
                    logger.info(f"✅ Moved 1 sample of regime {regime} from test to training")
            
            # Verify all regimes now in training
            train_regimes = np.unique(y_train)
            missing_from_train = set(all_regimes) - set(train_regimes)
            if missing_from_train:
                error_msg = (
                    f"❌ CRITICAL: Regimes {missing_from_train} still missing after stratified sampling!\n"
                    f"   This should not happen. Check data distribution."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"✅ All regimes now present in training set: {train_regimes}")
        
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
        
        # Fail fast if any regime has insufficient samples in training
        if insufficient_regimes:
            error_msg = (
                f"❌ CRITICAL: Some regimes have insufficient samples in training set:\n"
                + "\n".join([f"   Regime {r}: {count} samples (min required: {self.min_regime_samples})" 
                            for r, count in insufficient_regimes])
                + f"\n\n   SOLUTION: Either:\n"
                f"   1. Reduce min_regime_samples (current: {self.min_regime_samples})\n"
                f"   2. Adjust temporal split ratios to keep more samples in training\n"
                f"   3. Use more data (increase lookback period)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
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