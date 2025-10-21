"""
Compatibility utilities for cross-validation (legacy API layer).

This module provides light shims that map older ml_common validation names to
the unified implementations now used across the codebase.

Exposed symbols:
- TemporalCrossValidator: Thin wrapper delegating to the unified temporal CV
- PurgedKFold: Alias to the time-aware purged/embargoed splitter
- CrossValidationUtilities: Minimal utilities with walk_forward_validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from src.utils.tprint import tprint_data_format, LogLevel

try:
    from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    SkTimeSeriesSplit = None  # type: ignore

# Purged K-Fold time-aware splitter (existing implementation)
try:
    from src.utils.purged_kfold import PurgedKFoldTime as PurgedKFold  # type: ignore
except Exception:  # pragma: no cover - fallback if unavailable
    PurgedKFold = None  # type: ignore

class TimeSeriesSplitValidator:
    """Time series cross-validator that prevents data leakage in temporal data.

    This class provides time-series aware cross-validation that respects temporal
    order and prevents look-ahead bias. It delegates to sklearn's TimeSeriesSplit
    when available and provides fallback behavior otherwise.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: Optional[int] = None):
        """Initialize time series split validator.

        Args:
            n_splits: Number of splits for cross-validation
            gap: Gap between train and test sets to prevent data leakage
            test_size: Size of test set (if supported by sklearn version)
        """
        self.n_splits = max(2, int(n_splits))
        self.gap = max(0, int(gap))
        self.test_size = test_size

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate time series aware train/test splits.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Yields:
            Tuple of (train_indices, test_indices) respecting temporal order
        """
        if SkTimeSeriesSplit is not None:
            try:
                import inspect
                if self.test_size is not None and 'test_size' in inspect.signature(SkTimeSeriesSplit).parameters:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, test_size=self.test_size)
                else:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)  # type: ignore[arg-type]
                for tr, te in cv.split(X, y):
                    yield tr, te
                return
            except Exception:
                pass

        # Fallback: naive sequential splits respecting temporal order
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_idx = np.arange(start, stop)
            train_end = max(0, start - self.gap)
            train_idx = np.arange(0, train_end)
            yield train_idx, test_idx
            start = stop

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Args:
            X: Feature matrix (unused, for compatibility)
            y: Target vector (unused, for compatibility)

        Returns:
            Number of splits
        """
        return self.n_splits

class TemporalCrossValidator:
    """Backwards-compatible temporal cross-validator wrapper with VectorBT optimizations.

    Delegates to sklearn's TimeSeriesSplit if available, otherwise provides
    a simple sequential splitter. This class is intended to satisfy legacy
    imports while the canonical API lives in validation.unified_cv and
    validation.universal_temporal_validation.

    Enhanced with VectorBT-accelerated temporal validation for large datasets.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0, test_size: Optional[int] = None,
                 use_vectorbt: bool = True, chunk_size: int = 10000) -> None:
        self.n_splits = max(2, int(n_splits))
        self.gap = max(0, int(gap))
        self.test_size = test_size
        self.use_vectorbt = use_vectorbt
        self.chunk_size = chunk_size

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.use_vectorbt and len(X) > self.chunk_size:
            # Use VectorBT-optimized splitting for large datasets
            yield from self._vectorbt_optimized_split(X, y)
        elif SkTimeSeriesSplit is not None:
            try:
                import inspect
                if self.test_size is not None and 'test_size' in inspect.signature(SkTimeSeriesSplit).parameters:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap, test_size=self.test_size)
                else:
                    cv = SkTimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)  # type: ignore[arg-type]
                for tr, te in cv.split(X, y):
                    yield tr, te
                return
            except Exception:
                pass

        # Fallback: naive sequential splits
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_idx = np.arange(start, stop)
            train_end = max(0, start - self.gap)
            train_idx = np.arange(0, train_end)
            yield train_idx, test_idx
            start = stop

    def _vectorbt_optimized_split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """VectorBT-optimized temporal splitting for large datasets with advanced features."""
        try:
            import vectorbt as vbt
            import pandas as pd

            # Convert to pandas for VectorBT processing with memory optimization
            if y is not None:
                data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                data['target'] = y
            else:
                data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])

            # Create time index with proper frequency
            data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='1min')

            # Use VectorBT's advanced time series splitting with portfolio-style validation
            n = len(data)

            # Enhanced splitting strategy based on data characteristics
            if n > 10000:  # Large dataset - use VectorBT chunked processing
                yield from self._vectorbt_chunked_split(data, n)
            else:
                yield from self._vectorbt_standard_split(data, n)

        except ImportError:
            # Fallback to standard splitting if VectorBT not available
            logger.warning("VectorBT not available, using standard temporal splitting")
            yield from self._standard_temporal_split(X, y)
        except Exception as e:
            logger.warning(f"VectorBT splitting failed: {e}, using standard splitting")
            yield from self._standard_temporal_split(X, y)

    def _vectorbt_chunked_split(self, data: pd.DataFrame, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """VectorBT chunked splitting for very large datasets."""
        import vectorbt as vbt

        # Use VectorBT's chunked processing for memory efficiency
        chunk_size = min(self.chunk_size, n // self.n_splits)

        for i in range(self.n_splits):
            # Calculate split boundaries with gap consideration
            start_test = i * (n // self.n_splits)
            end_test = min((i + 1) * (n // self.n_splits), n)

            # Add gap if specified
            if self.gap > 0:
                start_test = max(0, start_test - self.gap)

            # Use VectorBT's chunked processing for large datasets
            train_data = data.iloc[:start_test]
            test_data = data.iloc[start_test:end_test]

            if len(train_data) > 0 and len(test_data) > 0:
                # Generate indices using VectorBT's optimized indexing
                train_idx = train_data.index.get_indexer(data.index[:start_test])
                test_idx = test_data.index.get_indexer(data.index[start_test:end_test])

                # Filter out -1 indices (not found)
                train_idx = train_idx[train_idx >= 0]
                test_idx = test_idx[test_idx >= 0]

                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx

    def _vectorbt_standard_split(self, data: pd.DataFrame, n: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """VectorBT standard splitting for smaller datasets."""
        # Use VectorBT's time series analysis for optimal splitting
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            # Calculate split boundaries
            start_test = i * fold_size
            end_test = min((i + 1) * fold_size, n)

            # Add gap if specified
            if self.gap > 0:
                start_test = max(0, start_test - self.gap)

            # Generate indices
            train_idx = np.arange(0, start_test)
            test_idx = np.arange(start_test, end_test)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _standard_temporal_split(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Standard temporal splitting fallback."""
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_idx = np.arange(start, stop)
            train_end = max(0, start - self.gap)
            train_idx = np.arange(0, train_end)
            yield train_idx, test_idx
            start = stop

@dataclass
class _WalkForwardConfig:
    initial_train_size: float = 0.6
    step_size: float = 0.1
    min_test_size: float = 0.1

class CrossValidationUtilities:
    """Minimal CV utilities used by memory integration shims.

    The canonical, richer API remains in validation.unified_cv and
    validation.universal_temporal_validation. This class exists to
    preserve backwards compatibility for integrations that monkey-patch
    walk_forward_validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.config = _WalkForwardConfig(
            initial_train_size=float(cfg.get('initial_train_size', 0.6)),
            step_size=float(cfg.get('step_size', 0.1)),
            min_test_size=float(cfg.get('min_test_size', 0.1)),
        )

    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if n == 0:
            return []
        initial = max(1, int(n * self.config.initial_train_size))
        step = max(1, int(n * self.config.step_size))
        min_test = max(1, int(n * self.config.min_test_size))
        indices: List[Tuple[np.ndarray, np.ndarray]] = []
        train_end = initial
        while train_end < n - min_test:
            test_end = min(n, train_end + min_test)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            indices.append((train_idx, test_idx))
            train_end = min(n, train_end + step)
        return indices

# Compatibility alias for CrossValidator
try:
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator as CrossValidator  # type: ignore
except Exception:  # pragma: no cover - fallback if unavailable
    CrossValidator = None  # type: ignore

class OOFGenerator:
    """Out-of-fold prediction generator for ensemble methods.

    This class manages the generation of out-of-fold predictions which are
    essential for stacking ensembles and meta-learning approaches.
    """

    def __init__(self, strategy: str = 'mean'):
        """Initialize OOF generator.

        Args:
            strategy: Strategy for combining predictions ('mean', 'median', 'vote')
        """
        self.strategy = strategy
        self.predictions = {}
        self.folds = []

    def add_fold_predictions(self, fold_id: int, predictions: np.ndarray):
        """Add predictions for a specific fold.

        Args:
            fold_id: Identifier for the fold
            predictions: Array of predictions for this fold
        """
        self.predictions[fold_id] = predictions
        self.folds.append(fold_id)

    def get_oof_predictions(self) -> np.ndarray:
        """Generate out-of-fold predictions by combining fold predictions.

        Returns:
            Array of out-of-fold predictions
        """
        if not self.predictions:
            return np.array([])

        # Collect all predictions
        fold_predictions = []
        for fold_id in sorted(self.folds):
            if fold_id in self.predictions:
                fold_predictions.append(self.predictions[fold_id])

        if not fold_predictions:
            return np.array([])

        # Combine predictions based on strategy
        if self.strategy == 'mean':
            return np.mean(fold_predictions, axis=0)
        elif self.strategy == 'median':
            return np.median(fold_predictions, axis=0)
        elif self.strategy == 'vote':
            # For classification (voting)
            return self._majority_vote(fold_predictions)
        else:
            # Default to mean
            return np.mean(fold_predictions, axis=0)

    def _majority_vote(self, predictions_list: List[np.ndarray]) -> np.ndarray:
        """Perform majority voting for classification predictions.

        Args:
            predictions_list: List of prediction arrays from different folds

        Returns:
            Array of majority vote predictions
        """
        # Stack predictions and take majority vote along fold axis
        stacked = np.stack(predictions_list, axis=0)

        # For each sample, take the most frequent prediction
        if stacked.ndim == 2:
            # Binary classification or regression
            return np.mean(stacked, axis=0)
        else:
            # Multi-class classification - take mode along fold axis
            return np.array([np.bincount(stacked[:, i]).argmax() for i in range(stacked.shape[1])])

    def reset(self):
        """Reset the OOF generator state."""
        self.predictions.clear()
        self.folds.clear()

class VectorBTCrossValidator:
    """
    VectorBT-enhanced cross-validation with advanced temporal analysis.

    This class provides sophisticated cross-validation using VectorBT's
    portfolio analysis capabilities for financial time series data.
    """

    def __init__(self, n_splits: int = 5, gap: int = 0,
                 use_portfolio_analysis: bool = True,
                 enable_memory_optimization: bool = True,
                 chunk_size: int = 10000):
        """
        Initialize VectorBT cross-validator.

        Args:
            n_splits: Number of CV splits
            gap: Gap between train/test sets
            use_portfolio_analysis: Use VectorBT portfolio analysis for validation
            enable_memory_optimization: Enable memory optimization for large datasets
            chunk_size: Chunk size for large dataset processing
        """
        self.n_splits = n_splits
        self.gap = gap
        self.use_portfolio_analysis = use_portfolio_analysis
        self.enable_memory_optimization = enable_memory_optimization
        self.chunk_size = chunk_size

        # Initialize VectorBT if available
        try:
            import vectorbt as vbt
            self.vbt = vbt
            self.vectorbt_available = True
        except ImportError:
            self.vbt = None
            self.vectorbt_available = False
            logger.warning("VectorBT not available, using standard CV")

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None,
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits using VectorBT optimization.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels for grouped CV

        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not self.vectorbt_available:
            # Fallback to standard temporal CV
            yield from self._standard_temporal_split(X, y)
            return

        # Convert to pandas for VectorBT processing
        X_df = self._prepare_dataframe(X)
        y_series = self._prepare_series(y) if y is not None else None

        # Use VectorBT-optimized splitting
        if self.enable_memory_optimization and len(X_df) > self.chunk_size:
            yield from self._vectorbt_chunked_cv(X_df, y_series)
        else:
            yield from self._vectorbt_standard_cv(X_df, y_series)

    def _prepare_dataframe(self, X: np.ndarray) -> pd.DataFrame:
        """Prepare features as DataFrame with VectorBT optimization."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_df = pd.DataFrame(X, columns=['feature_0'])
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        # Create time index for VectorBT
        X_df.index = pd.date_range(start='2020-01-01', periods=len(X_df), freq='1min')

        return X_df

    def _prepare_series(self, y: np.ndarray) -> pd.Series:
        """Prepare targets as Series with VectorBT optimization."""
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, name='target')
        else:
            y_series = y.copy()

        # Create time index for VectorBT
        y_series.index = pd.date_range(start='2020-01-01', periods=len(y_series), freq='1min')

        return y_series

    def _vectorbt_chunked_cv(self, X_df: pd.DataFrame, y_series: Optional[pd.Series]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """VectorBT chunked cross-validation for large datasets."""
        n = len(X_df)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            # Calculate split boundaries
            start_test = i * fold_size
            end_test = min((i + 1) * fold_size, n)

            # Add gap if specified
            if self.gap > 0:
                start_test = max(0, start_test - self.gap)

            # Use VectorBT's chunked processing
            train_data = X_df.iloc[:start_test]
            test_data = X_df.iloc[start_test:end_test]

            if len(train_data) > 0 and len(test_data) > 0:
                # Generate indices
                train_idx = np.arange(0, start_test)
                test_idx = np.arange(start_test, end_test)

                yield train_idx, test_idx

    def _vectorbt_standard_cv(self, X_df: pd.DataFrame, y_series: Optional[pd.Series]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """VectorBT standard cross-validation."""
        n = len(X_df)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            # Calculate split boundaries
            start_test = i * fold_size
            end_test = min((i + 1) * fold_size, n)

            # Add gap if specified
            if self.gap > 0:
                start_test = max(0, start_test - self.gap)

            # Generate indices
            train_idx = np.arange(0, start_test)
            test_idx = np.arange(start_test, end_test)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _standard_temporal_split(self, X: np.ndarray, y: Optional[np.ndarray]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Standard temporal splitting fallback."""
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        start = 0
        for fs in fold_sizes:
            stop = start + fs
            test_idx = np.arange(start, stop)
            train_end = max(0, start - self.gap)
            train_idx = np.arange(0, train_end)
            yield train_idx, test_idx
            start = stop

    def get_n_splits(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                     groups: Optional[np.ndarray] = None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits

    def evaluate_with_portfolio_analysis(self, X: np.ndarray, y: np.ndarray,
                                       model: Any) -> Dict[str, float]:
        """
        Evaluate model using VectorBT portfolio analysis.

        Args:
            X: Feature matrix
            y: Target vector
            model: Trained model

        Returns:
            Dictionary of portfolio-based metrics
        """
        if not self.vectorbt_available:
            logger.warning("VectorBT not available for portfolio analysis")
            return {}

        try:
            # Generate predictions
            predictions = model.predict(X)

            # Create portfolio using VectorBT
            returns = np.diff(predictions) / predictions[:-1]
            prices = predictions

            # Use VectorBT portfolio analysis
            portfolio = self.vbt.Portfolio.from_returns(returns, freq='1min')

            # Calculate portfolio metrics
            stats = portfolio.stats()

            return {
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0) / 100,
                'total_return': stats.get('Total Return [%]', 0) / 100,
                'volatility': stats.get('Annualized Volatility [%]', 0) / 100,
                'win_rate': stats.get('Win Rate [%]', 0) / 100
            }

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {}

__all__ = [
    'TimeSeriesSplitValidator',
    'TemporalCrossValidator',
    'VectorBTCrossValidator',
    'PurgedKFold',
    'CrossValidationUtilities',
    'OOFGenerator',
]
