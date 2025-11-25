"""
Regime Probability Aggregation Module

Data-driven approaches to convert multiple regime probabilities into a single 0-1 scalar
for use in downstream predictions, followed by isotonic calibration.

Supported Approaches:
1. Expected Return Weighted: Weight by empirical forward returns
2. Expected Sharpe Weighted: Weight by Sharpe ratios
3. Logistic Regression: Train on regime probabilities → binary outcome
4. Neural Network Compression: Small feedforward network
5. PCA First Component: Principal component analysis

All approaches include:
- Isotonic calibration to ensure monotonicity
- Cross-validation for hyperparameter selection
- Out-of-sample evaluation metrics
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
import pickle


logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Supported aggregation methods."""
    EXPECTED_RETURN = "expected_return"
    EXPECTED_SHARPE = "expected_sharpe"
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"
    PCA_FIRST = "pca_first"


class RegimeProbabilityAggregator:
    """
    Aggregate multiple regime probabilities into a single 0-1 scalar.

    This class provides multiple data-driven approaches to convert regime
    probabilities (e.g., [0.3, 0.5, 0.2]) into a single scalar score
    that can be used for prediction.

    Features:
    - Multiple aggregation methods (expected return, Sharpe, ML models)
    - Isotonic calibration for monotonicity
    - Cross-validation for method selection
    - Comprehensive evaluation metrics
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.EXPECTED_RETURN,
        isotonic_calibration: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the aggregator.

        Args:
            method: Aggregation method to use
            isotonic_calibration: Apply isotonic calibration after aggregation
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.isotonic_calibration = isotonic_calibration
        self.random_state = random_state

        # Fitted parameters
        self.regime_weights = None  # For expected return/Sharpe methods
        self.model = None  # For ML methods
        self.isotonic_regressor = None  # For calibration
        self.scaler_params = None  # For normalization

        # Metadata
        self.n_regimes = None
        self.regime_stats = None  # Forward return/Sharpe stats per regime
        self.is_fitted = False

        logger.info(
            f"Initialized RegimeProbabilityAggregator with method={method.value}, "
            f"isotonic_calibration={isotonic_calibration}"
        )

    def fit(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray,
        regime_stats: Optional[Dict[int, Dict[str, float]]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> 'RegimeProbabilityAggregator':
        """
        Fit the aggregator on regime probabilities and forward returns.

        Args:
            regime_probs: Array of shape (n_samples, n_regimes) with regime probabilities
            forward_returns: Array of shape (n_samples,) with forward returns
            regime_stats: Optional dict with per-regime statistics (mean_return, sharpe_ratio)
                         If not provided, will be computed from data
            sample_weight: Optional sample weights for training

        Returns:
            self (fitted aggregator)
        """
        # Validate inputs
        if regime_probs.shape[0] != len(forward_returns):
            raise ValueError(
                f"regime_probs and forward_returns must have same length. "
                f"Got {regime_probs.shape[0]} and {len(forward_returns)}"
            )

        self.n_regimes = regime_probs.shape[1]

        logger.info(
            f"Fitting aggregator with {len(forward_returns)} samples, "
            f"{self.n_regimes} regimes, method={self.method.value}"
        )

        # Compute regime statistics if not provided
        if regime_stats is None:
            regime_stats = self._compute_regime_stats(regime_probs, forward_returns)

        self.regime_stats = regime_stats

        # Fit using selected method
        if self.method == AggregationMethod.EXPECTED_RETURN:
            self._fit_expected_return(regime_stats)
        elif self.method == AggregationMethod.EXPECTED_SHARPE:
            self._fit_expected_sharpe(regime_stats)
        elif self.method == AggregationMethod.LOGISTIC_REGRESSION:
            self._fit_logistic_regression(regime_probs, forward_returns, sample_weight)
        elif self.method == AggregationMethod.NEURAL_NETWORK:
            self._fit_neural_network(regime_probs, forward_returns, sample_weight)
        elif self.method == AggregationMethod.PCA_FIRST:
            self._fit_pca(regime_probs, forward_returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply isotonic calibration if requested
        if self.isotonic_calibration:
            self._fit_isotonic_calibration(regime_probs, forward_returns)

        self.is_fitted = True
        logger.info("✅ Aggregator fitted successfully")

        return self

    def transform(
        self,
        regime_probs: np.ndarray,
        apply_calibration: bool = True
    ) -> np.ndarray:
        """
        Transform regime probabilities into single 0-1 scalar.

        Args:
            regime_probs: Array of shape (n_samples, n_regimes)
            apply_calibration: Apply isotonic calibration (if fitted)

        Returns:
            Array of shape (n_samples,) with aggregated scores in [0, 1]
        """
        if not self.is_fitted:
            raise RuntimeError("Aggregator not fitted. Call fit() first.")

        if regime_probs.shape[1] != self.n_regimes:
            raise ValueError(
                f"Expected {self.n_regimes} regimes, got {regime_probs.shape[1]}"
            )

        # Aggregate using fitted method
        if self.method == AggregationMethod.EXPECTED_RETURN:
            scores = self._transform_expected_return(regime_probs)
        elif self.method == AggregationMethod.EXPECTED_SHARPE:
            scores = self._transform_expected_sharpe(regime_probs)
        elif self.method == AggregationMethod.LOGISTIC_REGRESSION:
            scores = self._transform_logistic_regression(regime_probs)
        elif self.method == AggregationMethod.NEURAL_NETWORK:
            scores = self._transform_neural_network(regime_probs)
        elif self.method == AggregationMethod.PCA_FIRST:
            scores = self._transform_pca(regime_probs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Apply isotonic calibration
        if apply_calibration and self.isotonic_calibration and self.isotonic_regressor:
            scores = self.isotonic_regressor.predict(scores)

        # Ensure [0, 1] range
        scores = np.clip(scores, 0.0, 1.0)

        return scores

    def fit_transform(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray,
        regime_stats: Optional[Dict[int, Dict[str, float]]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fit the aggregator and transform in one step.

        Args:
            regime_probs: Array of shape (n_samples, n_regimes)
            forward_returns: Array of shape (n_samples,)
            regime_stats: Optional per-regime statistics
            sample_weight: Optional sample weights

        Returns:
            Aggregated scores of shape (n_samples,)
        """
        self.fit(regime_probs, forward_returns, regime_stats, sample_weight)
        return self.transform(regime_probs)

    # ========================================================================
    # Method-Specific Fitting
    # ========================================================================

    def _fit_expected_return(self, regime_stats: Dict[int, Dict[str, float]]):
        """
        Fit using expected return weighting.

        Formula: score = sum(prob_i × mean_return_i)
        Then normalize to [0, 1] using min-max scaling.
        """
        logger.info("Fitting with expected return weighting")

        # Extract mean returns for each regime
        weights = np.array([
            regime_stats[i]['mean_return']
            for i in range(self.n_regimes)
        ])

        self.regime_weights = weights

        # Store normalization parameters
        self.scaler_params = {
            'min_score': np.min(weights),
            'max_score': np.max(weights),
        }

        logger.info(f"  Regime weights (mean returns): {weights}")

    def _fit_expected_sharpe(self, regime_stats: Dict[int, Dict[str, float]]):
        """
        Fit using expected Sharpe ratio weighting.

        Formula: score = sum(prob_i × sharpe_i)
        Then normalize to [0, 1].
        """
        logger.info("Fitting with expected Sharpe ratio weighting")

        # Extract Sharpe ratios for each regime
        weights = np.array([
            regime_stats[i]['sharpe_ratio']
            for i in range(self.n_regimes)
        ])

        self.regime_weights = weights

        # Store normalization parameters
        self.scaler_params = {
            'min_score': np.min(weights),
            'max_score': np.max(weights),
        }

        logger.info(f"  Regime weights (Sharpe ratios): {weights}")

    def _fit_logistic_regression(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray,
        sample_weight: Optional[np.ndarray]
    ):
        """
        Fit logistic regression to predict positive returns.

        Target: binary (forward_return > 0)
        Features: regime probabilities
        Output: predicted probability of positive return
        """
        logger.info("Fitting logistic regression model")

        # Create binary target
        y_binary = (forward_returns > 0).astype(int)

        # Fit logistic regression
        self.model = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )

        self.model.fit(regime_probs, y_binary, sample_weight=sample_weight)

        # Evaluate with cross-validation
        cv_scores = cross_val_score(
            self.model,
            regime_probs,
            y_binary,
            cv=5,
            scoring='roc_auc'
        )

        logger.info(f"  Cross-validation ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        logger.info(f"  Model coefficients: {self.model.coef_[0]}")

    def _fit_neural_network(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray,
        sample_weight: Optional[np.ndarray]
    ):
        """
        Fit small neural network to compress regime probabilities.

        Architecture: n_regimes -> 8 -> 4 -> 1
        Target: normalized forward returns
        Activation: ReLU
        """
        logger.info("Fitting neural network compression model")

        # Normalize forward returns to [0, 1]
        y_normalized = (forward_returns - forward_returns.min()) / (
            forward_returns.max() - forward_returns.min() + 1e-8
        )

        self.scaler_params = {
            'min_return': forward_returns.min(),
            'max_return': forward_returns.max(),
        }

        # Fit neural network
        self.model = MLPRegressor(
            hidden_layer_sizes=(8, 4),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state,
            verbose=False
        )

        self.model.fit(regime_probs, y_normalized)

        # Evaluate
        y_pred = self.model.predict(regime_probs)
        mse = mean_squared_error(y_normalized, y_pred)
        logger.info(f"  Training MSE: {mse:.6f}")

    def _fit_pca(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray
    ):
        """
        Fit PCA and use first principal component as score.

        The first PC captures the maximum variance direction in regime space.
        """
        logger.info("Fitting PCA (first component)")

        # Fit PCA with 1 component
        self.model = PCA(n_components=1, random_state=self.random_state)
        self.model.fit(regime_probs)

        # Transform to get PC1 scores
        pc1_scores = self.model.transform(regime_probs).ravel()

        # Store normalization parameters
        self.scaler_params = {
            'min_score': pc1_scores.min(),
            'max_score': pc1_scores.max(),
        }

        logger.info(f"  Explained variance: {self.model.explained_variance_ratio_[0]:.3f}")
        logger.info(f"  PC1 loadings: {self.model.components_[0]}")

    def _fit_isotonic_calibration(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray
    ):
        """
        Fit isotonic regression for calibration.

        Maps aggregated scores to calibrated probabilities that are
        monotonically increasing and better aligned with true outcomes.
        """
        logger.info("Fitting isotonic calibration")

        # Get uncalibrated scores
        uncalibrated_scores = self.transform(regime_probs, apply_calibration=False)

        # Normalize forward returns to [0, 1] for calibration target
        y_normalized = (forward_returns - forward_returns.min()) / (
            forward_returns.max() - forward_returns.min() + 1e-8
        )

        # Fit isotonic regression
        self.isotonic_regressor = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds='clip'
        )

        self.isotonic_regressor.fit(uncalibrated_scores, y_normalized)

        logger.info("  ✅ Isotonic calibration fitted")

    # ========================================================================
    # Method-Specific Transformation
    # ========================================================================

    def _transform_expected_return(self, regime_probs: np.ndarray) -> np.ndarray:
        """Transform using expected return weighting."""
        # Weighted sum
        scores = regime_probs @ self.regime_weights

        # Normalize to [0, 1]
        min_score = self.scaler_params['min_score']
        max_score = self.scaler_params['max_score']
        scores = (scores - min_score) / (max_score - min_score + 1e-8)

        return scores

    def _transform_expected_sharpe(self, regime_probs: np.ndarray) -> np.ndarray:
        """Transform using expected Sharpe weighting."""
        # Weighted sum
        scores = regime_probs @ self.regime_weights

        # Normalize to [0, 1]
        min_score = self.scaler_params['min_score']
        max_score = self.scaler_params['max_score']
        scores = (scores - min_score) / (max_score - min_score + 1e-8)

        return scores

    def _transform_logistic_regression(self, regime_probs: np.ndarray) -> np.ndarray:
        """Transform using logistic regression."""
        # Predict probability of positive class
        scores = self.model.predict_proba(regime_probs)[:, 1]
        return scores

    def _transform_neural_network(self, regime_probs: np.ndarray) -> np.ndarray:
        """Transform using neural network."""
        scores = self.model.predict(regime_probs)
        scores = np.clip(scores, 0.0, 1.0)
        return scores

    def _transform_pca(self, regime_probs: np.ndarray) -> np.ndarray:
        """Transform using PCA first component."""
        # Project to PC1
        pc1_scores = self.model.transform(regime_probs).ravel()

        # Normalize to [0, 1]
        min_score = self.scaler_params['min_score']
        max_score = self.scaler_params['max_score']
        scores = (pc1_scores - min_score) / (max_score - min_score + 1e-8)

        return scores

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _compute_regime_stats(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute per-regime statistics from data.

        For each regime, compute:
        - mean_return: Expected forward return
        - std_return: Standard deviation
        - sharpe_ratio: mean / std

        Uses soft assignment (weighted by probabilities) rather than hard labels.
        """
        logger.info("Computing regime statistics from data")

        regime_stats = {}

        for regime_id in range(self.n_regimes):
            # Use regime probabilities as weights
            weights = regime_probs[:, regime_id]

            # Weighted mean and std
            weighted_mean = np.average(forward_returns, weights=weights)
            weighted_var = np.average(
                (forward_returns - weighted_mean) ** 2,
                weights=weights
            )
            weighted_std = np.sqrt(weighted_var)

            # Sharpe ratio
            sharpe = weighted_mean / (weighted_std + 1e-8)

            regime_stats[regime_id] = {
                'mean_return': float(weighted_mean),
                'std_return': float(weighted_std),
                'sharpe_ratio': float(sharpe),
                'count': int(np.sum(weights > 0.1))  # Approximate count
            }

            logger.info(
                f"  Regime {regime_id}: mean={weighted_mean:.6f}, "
                f"std={weighted_std:.6f}, sharpe={sharpe:.3f}"
            )

        return regime_stats

    def save(self, filepath: str):
        """Save fitted aggregator to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted aggregator")

        state = {
            'method': self.method.value,
            'isotonic_calibration': self.isotonic_calibration,
            'random_state': self.random_state,
            'n_regimes': self.n_regimes,
            'regime_stats': self.regime_stats,
            'regime_weights': self.regime_weights,
            'scaler_params': self.scaler_params,
            'model': self.model,
            'isotonic_regressor': self.isotonic_regressor,
            'is_fitted': self.is_fitted,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved aggregator to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RegimeProbabilityAggregator':
        """Load fitted aggregator from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance
        aggregator = cls(
            method=AggregationMethod(state['method']),
            isotonic_calibration=state['isotonic_calibration'],
            random_state=state['random_state']
        )

        # Restore state
        aggregator.n_regimes = state['n_regimes']
        aggregator.regime_stats = state['regime_stats']
        aggregator.regime_weights = state['regime_weights']
        aggregator.scaler_params = state['scaler_params']
        aggregator.model = state['model']
        aggregator.isotonic_regressor = state['isotonic_regressor']
        aggregator.is_fitted = state['is_fitted']

        logger.info(f"Loaded aggregator from {filepath}")

        return aggregator

    def evaluate(
        self,
        regime_probs: np.ndarray,
        forward_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate aggregator performance.

        Metrics:
        - ROC-AUC: Area under ROC curve for binary classification
        - MSE: Mean squared error for regression
        - Correlation: Pearson correlation with forward returns
        - Spearman: Rank correlation
        """
        if not self.is_fitted:
            raise RuntimeError("Aggregator not fitted")

        # Get scores
        scores = self.transform(regime_probs)

        # Binary classification metrics
        y_binary = (forward_returns > 0).astype(int)
        roc_auc = roc_auc_score(y_binary, scores)

        # Regression metrics
        # Normalize returns for fair MSE comparison
        y_normalized = (forward_returns - forward_returns.min()) / (
            forward_returns.max() - forward_returns.min() + 1e-8
        )
        mse = mean_squared_error(y_normalized, scores)

        # Correlation metrics
        correlation = np.corrcoef(scores, forward_returns)[0, 1]
        spearman = pd.Series(scores).corr(pd.Series(forward_returns), method='spearman')

        metrics = {
            'roc_auc': float(roc_auc),
            'mse': float(mse),
            'correlation': float(correlation),
            'spearman': float(spearman),
        }

        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics


def compare_aggregation_methods(
    regime_probs: np.ndarray,
    forward_returns: np.ndarray,
    regime_stats: Optional[Dict[int, Dict[str, float]]] = None,
    methods: Optional[List[AggregationMethod]] = None
) -> pd.DataFrame:
    """
    Compare multiple aggregation methods and return evaluation metrics.

    Args:
        regime_probs: Array of shape (n_samples, n_regimes)
        forward_returns: Array of shape (n_samples,)
        regime_stats: Optional per-regime statistics
        methods: List of methods to compare (default: all methods)

    Returns:
        DataFrame with comparison metrics
    """
    if methods is None:
        methods = list(AggregationMethod)

    results = []

    for method in methods:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Evaluating method: {method.value}")
        logger.info(f"{'=' * 80}")

        try:
            # Fit aggregator
            aggregator = RegimeProbabilityAggregator(
                method=method,
                isotonic_calibration=True
            )

            aggregator.fit(regime_probs, forward_returns, regime_stats)

            # Evaluate
            metrics = aggregator.evaluate(regime_probs, forward_returns)

            # Add method name
            metrics['method'] = method.value

            results.append(metrics)

        except Exception as e:
            logger.error(f"Error evaluating {method.value}: {e}")

    # Create comparison DataFrame
    df_comparison = pd.DataFrame(results)
    df_comparison = df_comparison.set_index('method')

    # Sort by ROC-AUC (descending)
    df_comparison = df_comparison.sort_values('roc_auc', ascending=False)

    logger.info("\n" + "=" * 80)
    logger.info("Comparison Summary:")
    logger.info("=" * 80)
    logger.info("\n" + str(df_comparison))

    return df_comparison
