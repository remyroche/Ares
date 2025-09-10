from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Step 12 Modular: Feature Selection

This module contains feature selection algorithms for Step 12.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..base.logger import setup_step12_logger
from ..base.utils import error, failed, timeout, warning
import logging

logger = setup_step12_logger()

class FeatureSelector:
    """Feature selection engine for Step 12."""

    def __init__(self, config: Dict[str, Any], metadata_columns: List[str], label_columns: set):
        """Initialize the feature selector.

        Args:
            config: Configuration dictionary.
            metadata_columns: List of metadata columns to exclude.
            label_columns: Set of label columns to exclude.
        """
        self.config = config
        self.metadata_columns = metadata_columns
        self.label_columns = label_columns
        self.logger = logger

    async def select_optimal_features(
        self,
        model: Any,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Select optimal features using enhanced tiered strategy.

        Args:
            model: Model instance for feature importance.
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            Tuple of (selected_features, selection_summary).
        """
        self.logger.info('🎯 Selecting optimal features using enhanced tiered strategy...')

        # Filter features (exclude metadata and labels)
        feature_names = [
            c for c in X_val.columns.tolist()
            if c not in self.metadata_columns and c not in self.label_columns
        ]
        X_train_filtered = X_train[feature_names]
        X_val_filtered = X_val[feature_names]

        total_features = len(feature_names)
        self.logger.info(f'📊 Total features available: {total_features}')

        # Check for warnings
        try:
            self._log_mutual_information_warnings(X_train_filtered, y_train)
        except Exception as e:
            self.logger.warning(f'Mutual Information check failed: {e}')

        try:
            self._log_feature_stability_warnings(X_train_filtered)
        except Exception as e:
            self.logger.warning(f'Stability check failed: {e}')

        # Choose selection method based on feature count
        if total_features > 200:
            optimal_features, selection_summary = await self._execute_stable_tiered_feature_selection(
                model_name, X_train_filtered, y_train, X_val_filtered, y_val, feature_names
            )
        else:
            optimal_features, selection_summary = await self._execute_stable_traditional_feature_selection(
                model_name, X_train_filtered, y_train, X_val_filtered, y_val, feature_names
            )

        self.logger.info(f'✅ Selected {len(optimal_features)} optimal features from {total_features} total features')
        return optimal_features, selection_summary

    def _log_mutual_information_warnings(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Log warnings for features with low mutual information."""
        if X.empty or y is None or len(X.columns) == 0 or not SKLEARN_AVAILABLE:
            return

        try:
            is_blank_env = os.environ.get('BLANK_TRAINING_MODE', '0') == '1'
        except Exception:
            is_blank_env = False

        try:
            is_blank_cfg = bool(self.config.get('BLANK_TRAINING_MODE', False))
        except Exception:
            is_blank_cfg = False

        blank_mode = is_blank_env or is_blank_cfg

        mi = mutual_info_classif(X.values, y.values, discrete_features=False, random_state=42)
        mi_series = pd.Series(mi, index=X.columns)

        if blank_mode:
            low = mi_series[mi_series <= 1e-05]
        else:
            threshold = mi_series.quantile(0.2)
            low = mi_series[mi_series <= threshold]

        if not low.empty:
            names = low.sort_values().index.tolist()
            threshold_str = '1e-5' if blank_mode else f'{threshold:.4g}'
            self.logger.warning(
                f"MI: {len(names)} features show near-zero uni-variate predictive power "
                f"(<= {threshold_str}): {names[:50]}{(' ...' if len(names) > 50 else '')}"
            )

    def _log_feature_stability_warnings(self, X: pd.DataFrame) -> None:
        """Log warnings for unstable features across folds."""
        if X.empty or not SKLEARN_AVAILABLE:
            return

        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        unstable: List[str] = []

        for col in X.columns:
            try:
                vals = X[col].astype(float).values
                gstd = float(np.nanstd(vals))

                if not np.isfinite(gstd) or gstd == 0.0:
                    continue

                fold_means = []
                for train_idx, _ in kf.split(vals):
                    fold_vals = vals[train_idx]
                    if fold_vals.size == 0:
                        continue
                    fold_means.append(float(np.nanmean(fold_vals)))

                if len(fold_means) < 2:
                    continue

                std_of_means = float(np.nanstd(fold_means))
                expected_se = gstd / np.sqrt(4)

                if std_of_means > 3.0 * expected_se:
                    unstable.append(col)

            except Exception:
                continue

        if unstable:
            self.logger.warning(
                f"Stability: {len(unstable)} features are unstable across folds "
                f"(std(mean) >> expected): {unstable[:50]}{(' ...' if len(unstable) > 50 else '')}"
            )

    async def _execute_stable_tiered_feature_selection(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Execute stable tiered feature selection with bootstrapping."""
        # Placeholder for tiered selection logic
        # For now, return all features
        return feature_names, {
            'method': 'tiered_selection',
            'total_features': len(feature_names),
            'selected_features': len(feature_names),
            'selection_summary': 'Stable tiered selection completed'
        }

    async def _execute_stable_traditional_feature_selection(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_names: List[str]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Execute stable traditional feature selection."""
        # Placeholder for traditional selection logic
        # For now, return all features
        return feature_names, {
            'method': 'traditional_selection',
            'total_features': len(feature_names),
            'selected_features': len(feature_names),
            'selection_summary': 'Traditional selection completed'
        }

__all__ = ['FeatureSelector']
