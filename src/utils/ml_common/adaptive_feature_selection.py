"""
Advanced Feature Selection for XGBoost Models.

Implements intelligent feature selection and constraint optimization:
1. Monotonic constraints based on Spearman correlation
2. Zero gain pruning (remove features with 0 importance)
3. Null importance test (target shuffling validation)
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging
from pathlib import Path
import json
import pickle

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MonotonicConstraintCalculator:
    """
    Calculate monotonic constraints for XGBoost based on feature-target correlation.

    Monotonic constraints ensure that features with strong directional relationships
    with the target maintain that relationship in the model, improving interpretability
    and preventing overfitting.
    """

    def __init__(
        self,
        positive_threshold: float = 0.06,
        negative_threshold: float = -0.06
    ):
        """
        Initialize monotonic constraint calculator.

        Args:
            positive_threshold: Correlation threshold for enforcing increasing constraint
            negative_threshold: Correlation threshold for enforcing decreasing constraint
        """
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.constraints = {}
        self.correlations = {}

    def calculate_constraints(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'spearman'
    ) -> Dict[str, int]:
        """
        Calculate monotonic constraints based on feature-target correlation.

        Args:
            X: Feature DataFrame
            y: Target series
            method: Correlation method ('spearman' or 'pearson')

        Returns:
            Dictionary mapping feature names to constraint values:
            - 1: Monotonically increasing
            - -1: Monotonically decreasing
            - 0: No constraint
        """
        constraints = {}
        correlations = {}

        for feature in X.columns:
            # Calculate correlation
            if method == 'spearman':
                corr, _ = spearmanr(X[feature], y, nan_policy='omit')
            else:
                corr = np.corrcoef(X[feature], y)[0, 1]

            correlations[feature] = corr

            # Determine constraint
            if corr > self.positive_threshold:
                constraints[feature] = 1  # Force increasing
            elif corr < self.negative_threshold:
                constraints[feature] = -1  # Force decreasing
            else:
                constraints[feature] = 0  # No constraint

        self.constraints = constraints
        self.correlations = correlations

        # Log statistics
        n_increasing = sum(1 for c in constraints.values() if c == 1)
        n_decreasing = sum(1 for c in constraints.values() if c == -1)
        n_unconstrained = sum(1 for c in constraints.values() if c == 0)

        logger.info(
            f"Monotonic constraints calculated: "
            f"{n_increasing} increasing, "
            f"{n_decreasing} decreasing, "
            f"{n_unconstrained} unconstrained"
        )

        return constraints

    def get_constraint_tuple(self, feature_names: List[str]) -> Tuple[int, ...]:
        """
        Get monotonic constraint tuple for XGBoost.

        Args:
            feature_names: Ordered list of feature names

        Returns:
            Tuple of constraint values matching feature order
        """
        if not self.constraints:
            return tuple(0 for _ in feature_names)

        return tuple(self.constraints.get(f, 0) for f in feature_names)

    def get_strong_features(
        self,
        min_abs_correlation: float = 0.05
    ) -> List[str]:
        """
        Get features with strong correlation to target.

        Args:
            min_abs_correlation: Minimum absolute correlation threshold

        Returns:
            List of feature names with strong correlation
        """
        return [
            f for f, corr in self.correlations.items()
            if abs(corr) >= min_abs_correlation
        ]


class ZeroGainPruner:
    """
    Prune features with zero importance from XGBoost models.

    After training, any feature with zero gain (importance) is removed
    as it contributes nothing to the model's predictive power.
    """

    def __init__(self):
        """Initialize zero gain pruner."""
        self.pruned_features = set()
        self.kept_features = []

    def identify_zero_gain_features(
        self,
        model: 'xgb.XGBModel',
        feature_names: List[str],
        importance_type: str = 'gain'
    ) -> Set[str]:
        """
        Identify features with zero importance.

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            Set of feature names with zero importance
        """
        # Get feature importance
        importance_dict = model.get_booster().get_score(importance_type=importance_type)

        # Find features with zero importance (not in importance dict)
        zero_gain_features = set()
        for feature in feature_names:
            if feature not in importance_dict or importance_dict[feature] == 0:
                zero_gain_features.add(feature)

        self.pruned_features = zero_gain_features
        self.kept_features = [f for f in feature_names if f not in zero_gain_features]

        logger.info(
            f"Zero gain pruning: {len(zero_gain_features)} features removed, "
            f"{len(self.kept_features)} kept"
        )

        return zero_gain_features

    def prune_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove zero-gain features from DataFrame.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with zero-gain features removed
        """
        return X[[f for f in X.columns if f in self.kept_features]]


class NullImportanceTest:
    """
    Test feature importance significance using target shuffling.

    Trains models with shuffled targets to establish a null distribution
    of feature importance, then compares real importance against this baseline.
    """

    def __init__(
        self,
        n_shuffles: int = 10,
        significance_threshold: float = 0.95
    ):
        """
        Initialize null importance test.

        Args:
            n_shuffles: Number of shuffled target runs
            significance_threshold: Percentile threshold for significance
        """
        self.n_shuffles = n_shuffles
        self.significance_threshold = significance_threshold
        self.null_importances = {}
        self.real_importances = {}
        self.significant_features = []

    def run_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict[str, Any],
        real_importances: Optional[Dict[str, float]] = None
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Run null importance test using target shuffling.

        Args:
            X: Feature DataFrame
            y: Target series
            model_params: XGBoost model parameters
            real_importances: Optional pre-computed real importances

        Returns:
            Tuple of (significant_features, significance_scores)
            - significant_features: Features passing significance test
            - significance_scores: P-values for each feature
        """
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available, skipping null importance test")
            return list(X.columns), {}

        feature_names = X.columns.tolist()

        # Train model with real target if not provided
        if real_importances is None:
            logger.info("Training model with real target...")
            model = xgb.XGBRegressor(**model_params)
            model.fit(X, y)
            self.real_importances = model.get_booster().get_score(importance_type='gain')
        else:
            self.real_importances = real_importances

        # Initialize null importance storage
        null_importances = {f: [] for f in feature_names}

        # Run shuffled target experiments
        logger.info(f"Running null importance test with {self.n_shuffles} shuffles...")
        for i in range(self.n_shuffles):
            # Shuffle target
            y_shuffled = y.sample(frac=1.0, random_state=i).reset_index(drop=True)

            # Train model
            model_shuffled = xgb.XGBRegressor(**model_params)
            model_shuffled.fit(X, y_shuffled)

            # Get importances
            shuffled_importance = model_shuffled.get_booster().get_score(importance_type='gain')

            # Store null importances
            for feature in feature_names:
                null_importances[feature].append(shuffled_importance.get(feature, 0.0))

        self.null_importances = null_importances

        # Calculate significance scores
        significance_scores = {}
        significant_features = []

        for feature in feature_names:
            real_imp = self.real_importances.get(feature, 0.0)
            null_dist = np.array(null_importances[feature])

            # Calculate percentile of real importance in null distribution
            if len(null_dist) > 0:
                percentile = (null_dist < real_imp).mean()
                significance_scores[feature] = percentile

                # Feature is significant if real importance exceeds threshold of null distribution
                if percentile >= self.significance_threshold:
                    significant_features.append(feature)
            else:
                significance_scores[feature] = 0.0

        self.significant_features = significant_features

        logger.info(
            f"Null importance test completed: "
            f"{len(significant_features)}/{len(feature_names)} features significant "
            f"(threshold: {self.significance_threshold})"
        )

        return significant_features, significance_scores

    def filter_dataframe(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove non-significant features from DataFrame.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with only significant features
        """
        return X[self.significant_features]


class AdaptiveFeatureSelector:
    """
    Orchestrates adaptive feature selection with monthly retraining updates.

    Combines:
    - Monotonic constraints based on correlation
    - Zero gain pruning after each retraining
    - Null importance test for full monthly retraining
    """

    def __init__(
        self,
        cache_dir: Path = Path("cache/feature_selection"),
        correlation_threshold: float = 0.06,
        n_null_shuffles: int = 10
    ):
        """
        Initialize adaptive feature selector.

        Args:
            cache_dir: Directory to cache feature selection state
            correlation_threshold: Threshold for monotonic constraints
            n_null_shuffles: Number of shuffles for null importance test
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.constraint_calculator = MonotonicConstraintCalculator(
            positive_threshold=correlation_threshold,
            negative_threshold=-correlation_threshold
        )
        self.zero_gain_pruner = ZeroGainPruner()
        self.null_importance_test = NullImportanceTest(n_shuffles=n_null_shuffles)

        self.selected_features = []
        self.monotonic_constraints = {}
        self.retraining_count = 0

    def _get_cache_path(self, model_id: str) -> Path:
        """Get path to feature selection cache."""
        return self.cache_dir / f"{model_id}_feature_selection.pkl"

    def load_state(self, model_id: str) -> bool:
        """
        Load feature selection state from cache.

        Args:
            model_id: Unique identifier for the model

        Returns:
            True if state was loaded successfully
        """
        cache_path = self._get_cache_path(model_id)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    state = pickle.load(f)

                self.selected_features = state['selected_features']
                self.monotonic_constraints = state['monotonic_constraints']
                self.retraining_count = state['retraining_count']

                logger.info(
                    f"Loaded feature selection state: "
                    f"{len(self.selected_features)} features, "
                    f"{self.retraining_count} retrainings"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load feature selection state: {e}")

        return False

    def save_state(self, model_id: str):
        """
        Save feature selection state to cache.

        Args:
            model_id: Unique identifier for the model
        """
        cache_path = self._get_cache_path(model_id)

        state = {
            'selected_features': self.selected_features,
            'monotonic_constraints': self.monotonic_constraints,
            'retraining_count': self.retraining_count
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved feature selection state for {model_id}")

    def full_monthly_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Perform full feature selection for monthly retraining.

        Includes:
        1. Calculate monotonic constraints from correlation
        2. Run null importance test
        3. Apply zero gain pruning

        Args:
            X: Feature DataFrame
            y: Target series
            model_params: XGBoost model parameters

        Returns:
            Tuple of (selected_X, monotonic_constraints)
        """
        logger.info("=" * 80)
        logger.info("FULL MONTHLY FEATURE SELECTION")
        logger.info("=" * 80)

        # Step 1: Calculate monotonic constraints
        logger.info("Step 1: Calculating monotonic constraints...")
        self.monotonic_constraints = self.constraint_calculator.calculate_constraints(X, y)

        # Step 2: Run null importance test
        logger.info("Step 2: Running null importance test...")
        significant_features, _ = self.null_importance_test.run_test(
            X, y, model_params
        )

        # Step 3: Filter to significant features
        X_filtered = X[significant_features]

        # Update monotonic constraints to match filtered features
        self.monotonic_constraints = {
            f: c for f, c in self.monotonic_constraints.items()
            if f in significant_features
        }

        self.selected_features = significant_features
        self.retraining_count = 0

        logger.info(f"Full selection complete: {len(significant_features)} features selected")
        logger.info("=" * 80)

        return X_filtered, self.monotonic_constraints

    def quick_retraining_selection(
        self,
        X: pd.DataFrame,
        trained_model: 'xgb.XGBModel'
    ) -> pd.DataFrame:
        """
        Perform quick feature selection for regular retraining (5-day cycle).

        Only applies zero gain pruning, using existing monotonic constraints.

        Args:
            X: Feature DataFrame
            trained_model: Recently trained XGBoost model

        Returns:
            DataFrame with zero-gain features removed
        """
        logger.info("Quick retraining selection: applying zero gain pruning...")

        # Identify and remove zero-gain features
        zero_gain = self.zero_gain_pruner.identify_zero_gain_features(
            trained_model,
            X.columns.tolist()
        )

        # Update selected features
        self.selected_features = [f for f in self.selected_features if f not in zero_gain]

        # Update constraints to match
        self.monotonic_constraints = {
            f: c for f, c in self.monotonic_constraints.items()
            if f in self.selected_features
        }

        self.retraining_count += 1

        # Return filtered DataFrame
        return X[self.selected_features]

    def should_do_full_selection(self, retrainings_per_month: int = 6) -> bool:
        """
        Determine if a full selection should be performed.

        Args:
            retrainings_per_month: Number of retraining cycles per month

        Returns:
            True if full selection should be performed
        """
        return self.retraining_count >= retrainings_per_month

    def get_xgboost_params(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Get XGBoost parameters including monotonic constraints.

        Args:
            feature_names: Ordered list of feature names

        Returns:
            Dictionary of XGBoost parameters
        """
        constraint_tuple = self.constraint_calculator.get_constraint_tuple(feature_names)

        return {
            'monotone_constraints': constraint_tuple,
            'tree_method': 'hist',  # Use histogram method
            'max_bin': 256,
        }
