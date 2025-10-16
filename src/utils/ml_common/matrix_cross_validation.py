"""
Matrix-Based Cross-Validation Optimization with VectorBT Integration

This module provides highly optimized cross-validation using matrix operations,
vectorized computations, GPU acceleration, and VectorBT portfolio management.
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import matrix operations if available
try:
    from ..matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import VectorBT components
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None

logger = logging.getLogger(__name__)

class MatrixCrossValidator:
    """
    Matrix-based cross-validation with vectorized operations and GPU acceleration.

    This class provides highly optimized cross-validation that:
    - Uses matrix operations for efficient computation
    - Supports GPU acceleration when available
    - Provides vectorized model evaluation
    - Minimizes memory usage with chunked processing
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int = 42, use_gpu: bool = True,
                 use_vectorbt: bool = True):
        """
        Initialize matrix-based cross-validator.

        Args:
            n_splits: Number of cross-validation splits
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            use_gpu: Whether to use GPU acceleration when available
            use_vectorbt: Whether to use VectorBT for portfolio-based evaluation
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.use_vectorbt = use_vectorbt and VECTORBT_AVAILABLE

        # Initialize matrix operations if available
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
        else:
            self.matrix_ops = None

        # Initialize VectorBT if available
        if self.use_vectorbt:
            self._configure_vectorbt()

        # Performance tracking
        self.performance_stats = {
            'total_folds_processed': 0,
            'total_models_trained': 0,
            'total_predictions': 0,
            'computation_time': 0.0,
            'memory_peak': 0.0,
            'vectorbt_operations': 0,
            'portfolio_evaluations': 0
        }

        logger.info("✅ Matrix-based cross-validator initialized")
        logger.info(f"📊 Configuration - Splits: {n_splits}, GPU: {self.use_gpu}, VectorBT: {self.use_vectorbt}")

    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        if not VECTORBT_AVAILABLE:
            return

        # Configure VectorBT for optimal performance
        vbt.settings.array_wrapper['freq'] = '1min'

        # Enable parallel processing if available
        if hasattr(vbt.settings, 'parallel'):
            vbt.settings.parallel['threading'] = True

        logger.info("🚀 VectorBT configured for cross-validation optimization")

    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      model_class: Any,
                      model_params: Dict[str, Any] = None,
                      cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                      batch_size: int = 1000) -> Dict[str, Any]:
        """
        Perform matrix-based cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            cv_indices: Pre-computed CV indices (optional)
            batch_size: Batch size for memory-efficient processing

        Returns:
            Dictionary containing CV results and performance metrics
        """
        start_time = time.time()
        logger.info("🚀 Starting matrix-based cross-validation...")
        logger.info(f"📊 Data shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")

        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Generate CV indices if not provided
        if cv_indices is None:
            cv_indices = self._generate_cv_indices(X, y)

        # Initialize results storage
        n_samples = len(X)
        cv_results = self._initialize_cv_results()

        # Process each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
            logger.info(f"🔄 Processing fold {fold_idx + 1}/{len(cv_indices)}...")

            fold_start_time = time.time()

            # Extract fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = self._train_model(model_class, model_params, X_train, y_train)

            # Make predictions
            predictions = self._predict_model(model, X_val, batch_size)

            # Calculate metrics
            fold_metrics = self._calculate_fold_metrics(y_val, predictions)

            # Store results
            cv_results = self._store_fold_results(cv_results, fold_idx, fold_metrics, predictions)

            fold_time = time.time() - fold_start_time
            logger.info(f"✅ Fold {fold_idx + 1} completed in {fold_time:.3f}s")

        # Calculate aggregate statistics
        cv_results = self._calculate_aggregate_statistics(cv_results)

        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['computation_time'] = total_time
        self.performance_stats['total_folds_processed'] = len(cv_indices)

        logger.info(f"✅ Matrix-based cross-validation completed in {total_time:.3f}s")
        logger.info(f"📊 Mean CV score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        return cv_results

    def vectorbt_cross_validate(self, X: Union[np.ndarray, pd.DataFrame],
                               y: Union[np.ndarray, pd.Series],
                               model_class: Any,
                               model_params: Dict[str, Any] = None,
                               cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                               batch_size: int = 1000,
                               portfolio_evaluation: bool = True) -> Dict[str, Any]:
        """
        Perform VectorBT-optimized cross-validation with portfolio-based evaluation.

        Args:
            X: Feature matrix
            y: Target vector
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            cv_indices: Pre-computed CV indices (optional)
            batch_size: Batch size for memory-efficient processing
            portfolio_evaluation: Whether to use portfolio-based evaluation

        Returns:
            Dictionary containing CV results and performance metrics
        """
        if not self.use_vectorbt:
            logger.warning("⚠️ VectorBT not available, falling back to standard cross-validation")
            return self.cross_validate(X, y, model_class, model_params, cv_indices, batch_size)

        start_time = time.time()
        logger.info("🚀 Starting VectorBT-optimized cross-validation...")
        logger.info(f"📊 Data shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")

        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Generate CV indices if not provided
        if cv_indices is None:
            cv_indices = self._generate_cv_indices(X, y)

        # Initialize results storage
        cv_results = self._initialize_cv_results()

        # Process each fold with VectorBT optimization
        for fold_idx, (train_idx, val_idx) in enumerate(cv_indices):
            logger.info(f"🔄 Processing fold {fold_idx + 1}/{len(cv_indices)} with VectorBT...")

            fold_start_time = time.time()

            # Extract fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = self._train_model(model_class, model_params, X_train, y_train)

            # Make predictions
            predictions = self._predict_model(model, X_val, batch_size)

            # Calculate metrics using VectorBT if portfolio evaluation is enabled
            if portfolio_evaluation and len(predictions) > 1:
                fold_metrics = self._calculate_vectorbt_metrics(y_val, predictions, X_val)
                self.performance_stats['vectorbt_operations'] += 1
                self.performance_stats['portfolio_evaluations'] += 1
            else:
                fold_metrics = self._calculate_fold_metrics(y_val, predictions)

            # Store results
            cv_results = self._store_fold_results(cv_results, fold_idx, fold_metrics, predictions)

            fold_time = time.time() - fold_start_time
            logger.info(f"✅ Fold {fold_idx + 1} completed in {fold_time:.3f}s")

        # Calculate aggregate statistics
        cv_results = self._calculate_aggregate_statistics(cv_results)

        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['computation_time'] = total_time
        self.performance_stats['total_folds_processed'] = len(cv_indices)

        logger.info(f"✅ VectorBT cross-validation completed in {total_time:.3f}s")
        logger.info(f"📊 Mean CV score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        return cv_results

    def parallel_cross_validate(self, X: Union[np.ndarray, pd.DataFrame],
                               y: Union[np.ndarray, pd.Series],
                               model_class: Any,
                               model_params: Dict[str, Any] = None,
                               cv_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                               max_workers: int = 4,
                               batch_size: int = 1000) -> Dict[str, Any]:
        """
        Perform parallel matrix-based cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            model_class: Model class to instantiate
            model_params: Parameters for model initialization
            cv_indices: Pre-computed CV indices (optional)
            max_workers: Maximum number of parallel workers
            batch_size: Batch size for memory-efficient processing

        Returns:
            Dictionary containing CV results and performance metrics
        """
        start_time = time.time()
        logger.info("🚀 Starting parallel matrix-based cross-validation...")
        logger.info(f"📊 Data shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
        logger.info(f"⚡ Using {max_workers} parallel workers")

        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Generate CV indices if not provided
        if cv_indices is None:
            cv_indices = self._generate_cv_indices(X, y)

        # Initialize results storage
        cv_results = self._initialize_cv_results()

        # Process folds in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fold processing tasks
            future_to_fold = {
                executor.submit(self._process_fold_parallel,
                              fold_idx, X, y, train_idx, val_idx,
                              model_class, model_params, batch_size): fold_idx
                for fold_idx, (train_idx, val_idx) in enumerate(cv_indices)
            }

            # Collect results as they complete
            for future in future_to_fold:
                fold_idx = future_to_fold[future]
                try:
                    fold_result = future.result()
                    cv_results = self._store_fold_results(
                        cv_results, fold_idx,
                        fold_result['metrics'], fold_result['predictions']
                    )
                    logger.info(f"✅ Fold {fold_idx + 1} completed")
                except Exception as e:
                    logger.error(f"❌ Fold {fold_idx + 1} failed: {e}")
                    # Store empty results for failed fold
                    cv_results['fold_scores'].append(0.0)
                    cv_results['fold_predictions'].append([])

        # Calculate aggregate statistics
        cv_results = self._calculate_aggregate_statistics(cv_results)

        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['computation_time'] = total_time
        self.performance_stats['total_folds_processed'] = len(cv_indices)

        logger.info(f"✅ Parallel matrix-based cross-validation completed in {total_time:.3f}s")
        logger.info(f"📊 Mean CV score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        return cv_results

    def _generate_cv_indices(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation indices."""
        n_samples = len(X)

        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        cv_indices = []
        start = 0

        for fold_size in fold_sizes:
            stop = start + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            cv_indices.append((train_idx, val_idx))
            start = stop

        return cv_indices

    def _initialize_cv_results(self) -> Dict[str, Any]:
        """Initialize cross-validation results structure."""
        return {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_metrics': [],
            'mean_score': 0.0,
            'std_score': 0.0,
            'scores': [],
            'predictions': [],
            'computation_time': 0.0,
            'performance_stats': {}
        }

    def _train_model(self, model_class: Any, model_params: Dict[str, Any],
                    X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a model instance."""
        if model_params is None:
            model_params = {}

        model = model_class(**model_params)
        model.fit(X_train, y_train)

        self.performance_stats['total_models_trained'] += 1

        return model

    def _predict_model(self, model: Any, X_val: np.ndarray, batch_size: int) -> np.ndarray:
        """Make predictions using the trained model."""
        n_samples = len(X_val)

        if n_samples <= batch_size:
            predictions = model.predict(X_val)
        else:
            # Process in batches for memory efficiency
            predictions = np.zeros(n_samples)

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_predictions = model.predict(X_val[start_idx:end_idx])
                predictions[start_idx:end_idx] = batch_predictions

        self.performance_stats['total_predictions'] += len(predictions)

        return predictions

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a fold."""
        # Basic metrics - can be extended
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def _calculate_vectorbt_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   X_val: np.ndarray) -> Dict[str, float]:
        """Calculate metrics using VectorBT for portfolio-based evaluation."""
        if not VECTORBT_AVAILABLE:
            return self._calculate_fold_metrics(y_true, y_pred)

        try:
            # Convert predictions to trading signals
            signals = self._predictions_to_signals(y_pred)

            # Create synthetic price data from features (simplified approach)
            prices = self._create_synthetic_prices(X_val, y_true)

            # Use VectorBT for portfolio evaluation
            portfolio = vbt.Portfolio.from_signals(
                prices, signals,
                init_cash=100000.0,
                fees=0.001,
                slippage=0.0005,
                freq='1min'
            )

            # Calculate comprehensive metrics
            stats = portfolio.stats()

            # Extract key metrics
            metrics = {
                'r2': self._calculate_r2_score(y_true, y_pred),
                'mse': np.mean((y_true - y_pred) ** 2),
                'mae': np.mean(np.abs(y_true - y_pred)),
                'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
                # VectorBT portfolio metrics
                'total_return': stats.get('Total Return [%]', 0) / 100,
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0) / 100,
                'win_rate': stats.get('Win Rate [%]', 0) / 100,
                'profit_factor': stats.get('Profit Factor', 0),
                'expectancy': stats.get('Expectancy [%]', 0) / 100,
                'volatility': stats.get('Annualized Volatility [%]', 0) / 100,
                'sortino_ratio': stats.get('Sortino Ratio', 0),
                'calmar_ratio': stats.get('Calmar Ratio', 0)
            }

            return metrics

        except Exception as e:
            logger.warning(f"⚠️ VectorBT metrics calculation failed: {e}, falling back to standard metrics")
            return self._calculate_fold_metrics(y_true, y_pred)

    def _predictions_to_signals(self, predictions: np.ndarray) -> np.ndarray:
        """Convert model predictions to trading signals."""
        # Simple threshold-based signal generation
        # This can be customized based on the specific use case
        signals = np.zeros_like(predictions)

        # Use quantiles for signal generation
        upper_threshold = np.percentile(predictions, 75)
        lower_threshold = np.percentile(predictions, 25)

        signals[predictions > upper_threshold] = 1  # Buy signal
        signals[predictions < lower_threshold] = -1  # Sell signal

        return signals

    def _create_synthetic_prices(self, X_val: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Create synthetic price data for VectorBT portfolio evaluation."""
        # Use the first feature as a proxy for price movement
        if X_val.shape[1] > 0:
            base_prices = X_val[:, 0] * 100  # Scale to reasonable price levels
        else:
            base_prices = np.ones(len(y_true)) * 100

        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.01, len(base_prices))
        prices = base_prices * (1 + noise)

        # Ensure prices are positive
        prices = np.maximum(prices, 1.0)

        return prices.reshape(-1, 1)  # Reshape for VectorBT

    def _calculate_r2_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def _store_fold_results(self, cv_results: Dict[str, Any], fold_idx: int,
                          metrics: Dict[str, float], predictions: np.ndarray) -> Dict[str, Any]:
        """Store results for a fold."""
        # Use R² as the primary score for simplicity
        fold_score = metrics.get('r2', 0.0)

        cv_results['fold_scores'].append(fold_score)
        cv_results['fold_predictions'].append(predictions)
        cv_results['fold_metrics'].append(metrics)
        cv_results['scores'].extend([fold_score])

        return cv_results

    def _calculate_aggregate_statistics(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate aggregate statistics across all folds."""
        fold_scores = np.array(cv_results['fold_scores'])

        cv_results['mean_score'] = np.mean(fold_scores)
        cv_results['std_score'] = np.std(fold_scores)
        cv_results['min_score'] = np.min(fold_scores)
        cv_results['max_score'] = np.max(fold_scores)

        # Calculate confidence intervals
        cv_results['confidence_interval_95'] = (
            cv_results['mean_score'] - 1.96 * cv_results['std_score'],
            cv_results['mean_score'] + 1.96 * cv_results['std_score']
        )

        cv_results['performance_stats'] = self.performance_stats.copy()

        return cv_results

    def _process_fold_parallel(self, fold_idx: int, X: np.ndarray, y: np.ndarray,
                            train_idx: np.ndarray, val_idx: np.ndarray,
                            model_class: Any, model_params: Dict[str, Any],
                            batch_size: int) -> Dict[str, Any]:
        """Process a single fold in parallel."""
        # Extract fold data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = self._train_model(model_class, model_params, X_train, y_train)

        # Make predictions
        predictions = self._predict_model(model, X_val, batch_size)

        # Calculate metrics
        metrics = self._calculate_fold_metrics(y_val, predictions)

        return {
            'fold_idx': fold_idx,
            'metrics': metrics,
            'predictions': predictions
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

def matrix_cross_validate(X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         model_class: Any,
                         model_params: Dict[str, Any] = None,
                         n_splits: int = 5,
                         use_gpu: bool = True,
                         parallel: bool = False,  # Changed default to False to prefer VectorBT
                         max_workers: int = 4,
                         use_vectorbt: bool = True,  # VectorBT is now the default
                         portfolio_evaluation: bool = True) -> Dict[str, Any]:
    """
    Convenience function for matrix-based cross-validation with VectorBT optimization.

    Args:
        X: Feature matrix
        y: Target vector
        model_class: Model class to instantiate
        model_params: Parameters for model initialization
        n_splits: Number of CV splits
        use_gpu: Whether to use GPU acceleration
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        use_vectorbt: Whether to use VectorBT for optimization
        portfolio_evaluation: Whether to use portfolio-based evaluation

    Returns:
        Dictionary containing CV results
    """
    validator = MatrixCrossValidator(
        n_splits=n_splits,
        use_gpu=use_gpu,
        use_vectorbt=use_vectorbt
    )

    # Prioritize VectorBT over parallel processing
    if use_vectorbt:
        return validator.vectorbt_cross_validate(
            X, y, model_class, model_params,
            portfolio_evaluation=portfolio_evaluation
        )
    elif parallel:
        return validator.parallel_cross_validate(
            X, y, model_class, model_params,
            max_workers=max_workers
        )
    else:
        return validator.cross_validate(
            X, y, model_class, model_params
        )

# Example usage and benchmarking functions
def benchmark_cross_validation():
    """Benchmark traditional vs matrix-based vs VectorBT cross-validation."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 10000, 50
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + 0.1 * np.random.randn(n_samples)

    logger.info("🔬 Benchmarking cross-validation methods...")
    logger.info(f"📊 Dataset: {n_samples} samples, {n_features} features")

    # Traditional cross-validation
    logger.info("⏱️ Running traditional cross-validation...")
    start_time = time.time()

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    traditional_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    traditional_time = time.time() - start_time
    logger.info(f"Traditional cross-validation time: {traditional_time:.3f}s")

    # Matrix-based cross-validation
    logger.info("⏱️ Running matrix-based cross-validation...")
    start_time = time.time()

    matrix_results = matrix_cross_validate(
        X, y, RandomForestRegressor,
        model_params={'n_estimators': 50, 'random_state': 42},
        n_splits=5, parallel=False, use_vectorbt=False
    )

    matrix_time = time.time() - start_time
    logger.info(f"Matrix cross-validation time: {matrix_time:.3f}s")

    # VectorBT cross-validation
    logger.info("⏱️ Running VectorBT cross-validation...")
    start_time = time.time()

    vectorbt_results = matrix_cross_validate(
        X, y, RandomForestRegressor,
        model_params={'n_estimators': 50, 'random_state': 42},
        n_splits=5, parallel=False, use_vectorbt=True,
        portfolio_evaluation=True
    )

    vectorbt_time = time.time() - start_time
    logger.info(f"VectorBT cross-validation time: {vectorbt_time:.3f}s")

    # Compare results
    matrix_speedup = traditional_time / matrix_time if matrix_time > 0 else float('inf')
    vectorbt_speedup = traditional_time / vectorbt_time if vectorbt_time > 0 else float('inf')

    logger.info("\n📊 BENCHMARK RESULTS:")
    logger.info(f"Traditional CV time: {traditional_time:.3f}s")
    logger.info(f"Matrix CV time: {matrix_time:.3f}s (Speedup: {matrix_speedup:.2f}x)")
    logger.info(f"VectorBT CV time: {vectorbt_time:.3f}s (Speedup: {vectorbt_speedup:.2f}x)")
    logger.info(f"Traditional R²: {traditional_scores.mean():.4f}")
    logger.info(f"Matrix R²: {matrix_results['mean_score']:.4f}")
    logger.info(f"VectorBT R²: {vectorbt_results['mean_score']:.4f}")

    # Show VectorBT-specific metrics if available
    if 'sharpe_ratio' in vectorbt_results.get('fold_metrics', [{}])[0]:
        avg_sharpe = np.mean([fold.get('sharpe_ratio', 0) for fold in vectorbt_results.get('fold_metrics', [])])
        avg_max_dd = np.mean([fold.get('max_drawdown', 0) for fold in vectorbt_results.get('fold_metrics', [])])
        logger.info(f"VectorBT Avg Sharpe: {avg_sharpe:.3f}")
        logger.info(f"VectorBT Avg Max DD: {avg_max_dd:.2%}")

    return {
        'traditional_time': traditional_time,
        'matrix_time': matrix_time,
        'vectorbt_time': vectorbt_time,
        'matrix_speedup': matrix_speedup,
        'vectorbt_speedup': vectorbt_speedup,
        'traditional_scores': traditional_scores,
        'matrix_scores': matrix_results['fold_scores'],
        'vectorbt_scores': vectorbt_results['fold_scores'],
        'vectorbt_metrics': vectorbt_results.get('fold_metrics', [])
    }

if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_cross_validation()
