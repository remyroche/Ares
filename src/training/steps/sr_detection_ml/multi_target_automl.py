"""
Multi-Target AutoML - 100% Data-Driven

Train models for ALL possible targets and select best by validation performance.
No predetermined target, let cross-validation decide.

Optimizations:
- Purged time series cross-validation
- Data leakage prevention
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Purged CV (conditional import)
purged_time_series_splits = None
PurgedSplitConfig = None
try:
    from src.utils.ml_common.validation.cv import purged_time_series_splits, PurgedSplitConfig  # type: ignore
except ImportError:
    pass

logger = logging.getLogger(__name__)


class MultiTargetAutoML:
    """
    AutoML target selection via validation performance.
    
    Philosophy: Train model for EACH target, select best by out-of-sample R².
    No assumptions about which target is 'best' - let data decide.
    
    Uses purged CV to prevent data leakage.
    """
    
    def __init__(self, n_splits: int = 5, min_target_coverage: float = 0.8, use_purged_cv: bool = True):
        """
        Initialize AutoML target selector.
        
        Args:
            n_splits: Number of time series CV splits
            min_target_coverage: Minimum fraction of non-NaN values required
            use_purged_cv: Whether to use purged CV (recommended)
        """
        self.n_splits = n_splits
        self.min_target_coverage = min_target_coverage
        self.use_purged_cv = use_purged_cv and (purged_time_series_splits is not None)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def find_best_target(
        self,
        X: pd.DataFrame,
        all_targets_df: pd.DataFrame
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Find best target by training model for each and comparing performance.
        
        Args:
            X: Feature matrix
            all_targets_df: DataFrame with all possible targets
        
        Returns:
            Tuple of (best_target_name, results_dict)
        """
        self.logger.info(f"🎯 Finding best target from {len(all_targets_df.columns)} candidates")
        
        # Prepare data with datetime index for purged CV
        if not isinstance(X.index, pd.DatetimeIndex):
            X.index = pd.date_range(start='2020-01-01', periods=len(X), freq='1h')
            all_targets_df.index = X.index
        
        results = {}
        
        for target_col in all_targets_df.columns:
            try:
                y = all_targets_df[target_col]
                
                # Check coverage (skip if too many NaNs)
                coverage = 1 - (y.isna().sum() / len(y))
                if coverage < self.min_target_coverage:
                    self.logger.debug(f"   Skipping {target_col}: coverage {coverage:.1%} < {self.min_target_coverage:.1%}")
                    continue
                
                # Drop NaN values
                valid_mask = ~y.isna()
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                if len(y_valid) < 100:
                    self.logger.debug(f"   Skipping {target_col}: only {len(y_valid)} valid samples")
                    continue
                
                # Get CV splits (purged or standard)
                if self.use_purged_cv and purged_time_series_splits is not None and PurgedSplitConfig is not None:
                    purged_config = PurgedSplitConfig(
                        n_splits=self.n_splits,
                        purge_minutes=60,
                        embargo_minutes=30
                    )
                    cv_splits = list(purged_time_series_splits(X_valid, y_valid, purged_config))
                else:
                    tscv = TimeSeriesSplit(n_splits=self.n_splits)
                    cv_splits = list(tscv.split(X_valid))
                
                # Train with cross-validation
                cv_scores = []
                cv_rmse = []
                cv_mae = []
                
                for train_idx, val_idx in cv_splits:
                    # Train model
                    model = lgb.LGBMRegressor(
                        n_estimators=200,
                        random_state=42,
                        verbose=-1,
                        force_col_wise=True
                    )
                    
                    model.fit(
                        X_valid.iloc[train_idx],
                        y_valid.iloc[train_idx]
                    )
                    
                    # Predict
                    preds = model.predict(X_valid.iloc[val_idx])
                    
                    # Metrics
                    r2 = r2_score(y_valid.iloc[val_idx], preds)
                    rmse = np.sqrt(mean_squared_error(y_valid.iloc[val_idx], preds))
                    mae = mean_absolute_error(y_valid.iloc[val_idx], preds)
                    
                    cv_scores.append(r2)
                    cv_rmse.append(rmse)
                    cv_mae.append(mae)
                
                # Store results
                results[target_col] = {
                    'mean_r2': np.mean(cv_scores),
                    'std_r2': np.std(cv_scores),
                    'mean_rmse': np.mean(cv_rmse),
                    'mean_mae': np.mean(cv_mae),
                    'coverage': coverage,
                    'n_samples': len(y_valid)
                }
                
                self.logger.debug(
                    f"   {target_col}: R²={np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
                )
                
            except Exception as e:
                self.logger.warning(f"   Failed to evaluate {target_col}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid targets found! All targets have insufficient coverage or failed.")
        
        # Select target with best mean R²
        best_target = max(results.items(), key=lambda x: x[1]['mean_r2'])[0]
        
        self.logger.info(f"\n✅ Best target selected: {best_target}")
        self.logger.info(f"   R²: {results[best_target]['mean_r2']:.4f} ± {results[best_target]['std_r2']:.4f}")
        self.logger.info(f"   RMSE: {results[best_target]['mean_rmse']:.6f}")
        self.logger.info(f"   MAE: {results[best_target]['mean_mae']:.6f}")
        self.logger.info(f"   Coverage: {results[best_target]['coverage']:.1%}")
        self.logger.info(f"   Samples: {results[best_target]['n_samples']:,}")
        
        # Log top 10 targets
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
        self.logger.info(f"\n   Top 10 targets by R²:")
        for rank, (target, metrics) in enumerate(sorted_results[:10], 1):
            self.logger.info(
                f"      {rank}. {target}: R²={metrics['mean_r2']:.4f} ± {metrics['std_r2']:.4f}"
            )
        
        return best_target, results
    
    def get_target_analysis(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate target analysis report.
        
        Args:
            results: Results dictionary from find_best_target
        
        Returns:
            DataFrame with target performance metrics
        """
        # Convert to DataFrame
        df = pd.DataFrame(results).T
        
        # Sort by R²
        df = df.sort_values('mean_r2', ascending=False)
        
        # Add rank
        df['rank'] = range(1, len(df) + 1)
        
        return df

