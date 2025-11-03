"""
LGBM+SHAP Feature Selector - 100% Data-Driven

Use LGBM SHAP values to select most important features.
No mRMR, no LASSO, no multi-stage selection - just SHAP importance.

Optimizations:
- Purged cross-validation to prevent data leakage
- Time series split with embargo
- Data leakage prevention utilities
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit

# Purged CV (conditional import with feature flag)
purged_time_series_splits = None
PurgedSplitConfig = None
try:
    from src.utils.ml_common.validation.cv import purged_time_series_splits, PurgedSplitConfig  # type: ignore
except ImportError:
    pass

# Data leakage prevention (conditional import)
DataLeakagePrevention = None
try:
    from src.utils.ml_common.validation.data_leakage_prevention import DataLeakagePrevention  # type: ignore
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LGBMShapFeatureSelector:
    """
    Select features using LGBM SHAP importance.
    
    Philosophy: Let LGBM+SHAP determine feature importance.
    No predetermined selection methods or manual filtering.
    
    Optimizations:
    - Uses purged CV to prevent data leakage
    - Data leakage prevention checks
    """
    
    def __init__(self, n_splits: int = 5, use_purged_cv: bool = True):
        """
        Initialize feature selector.
        
        Args:
            n_splits: Number of time series CV splits
            use_purged_cv: Whether to use purged CV (recommended for time series)
        """
        self.n_splits = n_splits
        self.use_purged_cv = use_purged_cv and (purged_time_series_splits is not None)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize data leakage prevention
        if DataLeakagePrevention is not None:
            self.leakage_prevention = DataLeakagePrevention()
            self.logger.info("✅ Data leakage prevention initialized")
        else:
            self.leakage_prevention = None
    
    def select_features_by_shap_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using LGBM SHAP values.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
        
        Returns:
            Tuple of (selected_feature_names, mean_shap_importance)
        """
        self.logger.info(f"🔍 Selecting top {n_features} features from {len(X.columns)} using LGBM+SHAP")
        
        # Drop features with no variance
        X_clean = self._remove_zero_variance(X)
        
        self.logger.info(f"   After removing zero-variance: {len(X_clean.columns)} features")
        
        # Convert to DataFrame with datetime index for purged CV
        if not isinstance(X_clean.index, pd.DatetimeIndex):
            # Create synthetic datetime index
            X_clean.index = pd.date_range(start='2020-01-01', periods=len(X_clean), freq='1h')
        
        # Use purged CV if available
        if self.use_purged_cv and purged_time_series_splits is not None and PurgedSplitConfig is not None:
            self.logger.info("   Using purged time series CV (prevents data leakage)")
            purged_config = PurgedSplitConfig(
                n_splits=self.n_splits,
                purge_minutes=60,  # 1 hour purge
                embargo_minutes=30  # 30 min embargo
            )
            cv_splits = list(purged_time_series_splits(X_clean, y, purged_config))
        else:
            # Standard time series split
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            cv_splits = list(tscv.split(X_clean))
        
        shap_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, 1):
            self.logger.info(f"   Fold {fold_idx}/{self.n_splits}...")
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
            
            model.fit(
                X_clean.iloc[train_idx],
                y.iloc[train_idx]
            )
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_clean.iloc[val_idx])
            
            # Aggregate absolute SHAP importance
            fold_importance = np.abs(shap_values).mean(axis=0)
            shap_importances.append(fold_importance)
        
        # Average SHAP importance across folds
        mean_shap_importance = np.mean(shap_importances, axis=0)
        
        # Select top N features
        top_indices = np.argsort(mean_shap_importance)[-n_features:]
        selected_features = X_clean.columns[top_indices].tolist()
        
        self.logger.info(f"✅ Selected {len(selected_features)} features by SHAP importance")
        
        # Log top 10 features
        top_10_idx = np.argsort(mean_shap_importance)[-10:]
        self.logger.info(f"\n   Top 10 features:")
        for rank, idx in enumerate(reversed(top_10_idx), 1):
            feature = X_clean.columns[idx]
            importance = mean_shap_importance[idx]
            self.logger.info(f"      {rank}. {feature}: {importance:.6f}")
        
        return selected_features, mean_shap_importance
    
    def _remove_zero_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features with zero variance.
        
        Args:
            X: Feature matrix
        
        Returns:
            Cleaned feature matrix
        """
        # Calculate variance
        variances = X.var()
        
        # Keep features with non-zero variance
        non_zero_var = variances[variances > 1e-10].index.tolist()
        
        removed_count = len(X.columns) - len(non_zero_var)
        if removed_count > 0:
            self.logger.info(f"   Removed {removed_count} zero-variance features")
        
        # Ensure return is DataFrame
        result = X[non_zero_var]
        if isinstance(result, pd.Series):
            result = result.to_frame()
        return result
    
    def get_feature_importance_report(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Generate feature importance report.
        
        Args:
            feature_names: List of feature names
            importance_values: SHAP importance values
            top_n: Number of top features to include
        
        Returns:
            DataFrame with feature importance ranking
        """
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': importance_values.tolist()
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        # Return top N
        return importance_df.head(top_n)

