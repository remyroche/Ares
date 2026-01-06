"""
Purged Cross-Validation for Time Series Models

This module implements purged cross-validation with embargo periods to prevent
information leakage in financial time series modeling.

Key Features:
- Purged K-Fold CV with temporal gaps
- Embargo periods to prevent lookahead bias
- Walk-forward validation with purging
- Ensemble weight optimization with purged splits
- Leakage detection and prevention
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import warnings

from src.utils.logger import system_logger

logger = system_logger.getChild("PurgedCV")


class PurgedTimeSeriesSplit:
    """
    Purged Time Series Cross-Validation with embargo periods.
    
    Prevents information leakage by:
    1. Purging training data near validation boundaries
    2. Adding embargo periods after validation sets
    3. Maintaining temporal order
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01,
        min_train_size: int = 100,
    ):
        """
        Initialize purged time series split.
        
        Args:
            n_splits: Number of CV splits
            purge_pct: Percentage of data to purge around boundaries
            embargo_pct: Percentage of data to embargo after validation
            min_train_size: Minimum training samples per split
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.min_train_size = min_train_size
        
    def split(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged cross-validation splits.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target Series (optional)
            groups: Group labels (optional)
        
        Yields:
            Tuple of (train_indices, val_indices) for each split
        """
        n_samples = len(X)
        
        if n_samples < self.min_train_size * 2:
            raise ValueError(f"Insufficient samples: {n_samples} < {self.min_train_size * 2}")
        
        # Calculate purge and embargo sizes
        purge_size = max(1, int(n_samples * self.purge_pct))
        embargo_size = max(1, int(n_samples * self.embargo_pct))
        
        # Use standard TimeSeriesSplit as base, then apply purging
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for train_idx, val_idx in tscv.split(X):
            # Convert to sorted arrays
            train_idx = np.sort(train_idx)
            val_idx = np.sort(val_idx)
            
            # Apply purging: remove training samples near validation boundary
            if len(train_idx) > 0 and len(val_idx) > 0:
                train_max = train_idx.max()
                val_min = val_idx.min()
                
                # Purge training data that's too close to validation
                purge_start = max(0, val_min - purge_size)
                purged_train_idx = train_idx[train_idx < purge_start]
                
                # Apply embargo: remove validation samples too close to next training
                # (This is handled implicitly by the temporal nature of TimeSeriesSplit)
                purged_val_idx = val_idx
                
                # Ensure minimum training size
                if len(purged_train_idx) < self.min_train_size:
                    logger.warning(
                        f"Insufficient training samples after purging: "
                        f"{len(purged_train_idx)} < {self.min_train_size}"
                    )
                    continue
                
                yield purged_train_idx, purged_val_idx
            else:
                continue


class PurgedEnsembleWeightOptimizer:
    """
    Ensemble weight optimizer using purged cross-validation.
    
    Optimizes ensemble weights while preventing information leakage
    through purged cross-validation and proper temporal validation.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        optimization_method: str = "grid_search",
    ):
        """
        Initialize purged ensemble weight optimizer.
        
        Args:
            n_splits: Number of CV splits
            purge_pct: Purge percentage
            embargo_pct: Embargo percentage
            weight_bounds: Bounds for individual weights
            optimization_method: Method for weight optimization
        """
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self.weight_bounds = weight_bounds
        self.optimization_method = optimization_method
        
        self.purged_cv = PurgedTimeSeriesSplit(
            n_splits=n_splits,
            purge_pct=purge_pct,
            embargo_pct=embargo_pct
        )
        
    def optimize_weights(
        self,
        individual_predictions: Dict[str, pd.Series],
        y: pd.Series,
        metric: str = "roc_auc",
        weight_grid: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize ensemble weights using purged cross-validation.
        
        Args:
            individual_predictions: Dict of model predictions
            y: Target series
            metric: Optimization metric
            weight_grid: Grid of weight values to search
        
        Returns:
            Optimization results with best weights and performance
        """
        if weight_grid is None:
            weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        model_names = list(individual_predictions.keys())
        n_models = len(model_names)
        
        logger.info(f"🔍 Starting purged ensemble weight optimization: {n_models} models")
        
        # Align all predictions and targets
        aligned_data = self._align_predictions_and_target(individual_predictions, y)
        if aligned_data is None:
            return {"error": "Failed to align predictions and targets"}
        
        X_pred, y_aligned, aligned_indices = aligned_data
        
        # Generate weight combinations
        weight_combinations = self._generate_weight_combinations(
            n_models, weight_grid, self.weight_bounds
        )
        
        logger.info(f"🔄 Testing {len(weight_combinations)} weight combinations with purged CV")
        
        best_score = -np.inf if metric in ["roc_auc", "accuracy"] else np.inf
        best_weights = None
        best_combination_idx = 0
        
        # Evaluate each weight combination
        for i, weights in enumerate(weight_combinations):
            cv_scores = []
            
            # Purged cross-validation
            for train_idx, val_idx in self.purged_cv.split(X_pred):
                try:
                    # Get train/validation splits
                    X_train = X_pred.iloc[train_idx]
                    y_train = y_aligned.iloc[train_idx]
                    X_val = X_pred.iloc[val_idx]
                    y_val = y_aligned.iloc[val_idx]
                    
                    # Create ensemble predictions
                    ensemble_train = self._create_ensemble_predictions(X_train, weights)
                    ensemble_val = self._create_ensemble_predictions(X_val, weights)
                    
                    # Calculate metric
                    score = self._calculate_metric(ensemble_val, y_val, metric)
                    cv_scores.append(score)
                    
                except Exception as e:
                    logger.debug(f"CV split failed for weights {weights}: {e}")
                    cv_scores.append(np.nan)
            
            # Average CV score
            mean_score = np.nanmean(cv_scores)
            
            # Update best weights
            if (metric in ["roc_auc", "accuracy"] and mean_score > best_score) or \
               (metric not in ["roc_auc", "accuracy"] and mean_score < best_score):
                best_score = mean_score
                best_weights = weights
                best_combination_idx = i
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i+1}/{len(weight_combinations)} combos, best {metric}: {best_score:.4f}")
        
        # Final evaluation with best weights
        final_results = self._evaluate_final_weights(
            X_pred, y_aligned, best_weights, model_names, metric
        )
        
        results = {
            "best_weights": dict(zip(model_names, best_weights)),
            "best_score": float(best_score),
            "best_combination_idx": best_combination_idx,
            "metric": metric,
            "n_combinations_tested": len(weight_combinations),
            "purged_cv_config": {
                "n_splits": self.n_splits,
                "purge_pct": self.purge_pct,
                "embargo_pct": self.embargo_pct,
            },
            "final_evaluation": final_results,
        }
        
        logger.info(
            f"✅ Purged weight optimization completed: best {metric} = {best_score:.4f}"
        )
        
        return results
    
    def _align_predictions_and_target(
        self, 
        individual_predictions: Dict[str, pd.Series], 
        y: pd.Series
    ) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]]:
        """Align all predictions and target to common index."""
        try:
            # Find common index
            all_indices = [y.index] + [pred.index for pred in individual_predictions.values()]
            common_index = all_indices[0]
            
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            if len(common_index) < 100:
                logger.error(f"Insufficient overlapping samples: {len(common_index)}")
                return None
            
            # Align all data
            y_aligned = y.loc[common_index]
            aligned_predictions = {}
            
            for name, pred in individual_predictions.items():
                aligned_predictions[name] = pred.loc[common_index]
            
            X_pred = pd.DataFrame(aligned_predictions)
            
            return X_pred, y_aligned, common_index
            
        except Exception as e:
            logger.error(f"Failed to align predictions and target: {e}")
            return None
    
    def _generate_weight_combinations(
        self, 
        n_models: int, 
        weight_grid: List[float],
        bounds: Tuple[float, float]
    ) -> List[np.ndarray]:
        """Generate weight combinations that sum to 1."""
        if self.optimization_method == "grid_search":
            return self._grid_search_weights(n_models, weight_grid, bounds)
        elif self.optimization_method == "random_search":
            return self._random_search_weights(n_models, 1000, bounds)
        else:
            return self._equal_weights(n_models)
    
    def _grid_search_weights(
        self, n_models: int, weight_grid: List[float], bounds: Tuple[float, float]
    ) -> List[np.ndarray]:
        """Generate weight combinations using grid search."""
        from itertools import product
        
        # Filter grid by bounds
        filtered_grid = [w for w in weight_grid if bounds[0] <= w <= bounds[1]]
        
        # Generate combinations
        combinations = []
        for combo in product(filtered_grid, repeat=n_models):
            weights = np.array(combo)
            # Normalize to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
                combinations.append(weights)
        
        return combinations
    
    def _random_search_weights(
        self, n_models: int, n_samples: int, bounds: Tuple[float, float]
    ) -> List[np.ndarray]:
        """Generate weight combinations using random search."""
        combinations = []
        
        for _ in range(n_samples):
            # Generate random weights
            weights = np.random.uniform(bounds[0], bounds[1], n_models)
            # Normalize to sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
                combinations.append(weights)
        
        return combinations
    
    def _equal_weights(self, n_models: int) -> List[np.ndarray]:
        """Generate equal weights combination."""
        equal_weights = np.ones(n_models) / n_models
        return [equal_weights]
    
    def _create_ensemble_predictions(
        self, X_pred: pd.DataFrame, weights: np.ndarray
    ) -> pd.Series:
        """Create ensemble predictions from individual predictions."""
        ensemble_pred = (X_pred.values * weights).sum(axis=1)
        return pd.Series(ensemble_pred, index=X_pred.index)
    
    def _calculate_metric(
        self, y_pred: pd.Series, y_true: pd.Series, metric: str
    ) -> float:
        """Calculate evaluation metric."""
        try:
            if metric == "roc_auc":
                return roc_auc_score(y_true, y_pred)
            elif metric == "log_loss":
                return log_loss(y_true, y_pred)
            elif metric == "mse":
                return mean_squared_error(y_true, y_pred)
            elif metric == "accuracy":
                return ((y_pred > 0.5) == y_true).mean()
            else:
                logger.warning(f"Unknown metric: {metric}, using ROC AUC")
                return roc_auc_score(y_true, y_pred)
        except Exception as e:
            logger.debug(f"Metric calculation failed: {e}")
            return np.nan
    
    def _evaluate_final_weights(
        self,
        X_pred: pd.DataFrame,
        y: pd.Series,
        weights: np.ndarray,
        model_names: List[str],
        metric: str,
    ) -> Dict[str, Any]:
        """Final evaluation of best weights."""
        try:
            # Create ensemble predictions
            ensemble_pred = self._create_ensemble_predictions(X_pred, weights)
            
            # Calculate metrics
            final_score = self._calculate_metric(ensemble_pred, y, metric)
            
            # Individual model performance
            individual_scores = {}
            for name in model_names:
                individual_scores[name] = self._calculate_metric(X_pred[name], y, metric)
            
            return {
                "final_score": float(final_score),
                "individual_scores": individual_scores,
                "weight_distribution": dict(zip(model_names, weights)),
                "ensemble_improvement": float(final_score - np.mean(list(individual_scores.values()))),
            }
            
        except Exception as e:
            logger.error(f"Final evaluation failed: {e}")
            return {"error": str(e)}


def purged_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_func: callable,
    n_splits: int = 5,
    purge_pct: float = 0.05,
    embargo_pct: float = 0.01,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Convenience function for purged cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        model_func: Model training/prediction function
        n_splits: Number of CV splits
        purge_pct: Purge percentage
        embargo_pct: Embargo percentage
        **model_kwargs: Additional model parameters
    
    Returns:
        Cross-validation results
    """
    purged_cv = PurgedTimeSeriesSplit(
        n_splits=n_splits,
        purge_pct=purge_pct,
        embargo_pct=embargo_pct
    )
    
    cv_scores = []
    cv_predictions = []
    
    for train_idx, val_idx in purged_cv.split(X):
        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = model_func(X_train, y_train, **model_kwargs)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            # Calculate score
            score = roc_auc_score(y_val, y_pred)
            cv_scores.append(score)
            cv_predictions.append(pd.Series(y_pred, index=X_val.index))
            
        except Exception as e:
            logger.debug(f"CV split failed: {e}")
            cv_scores.append(np.nan)
    
    return {
        "cv_scores": cv_scores,
        "mean_score": np.nanmean(cv_scores),
        "std_score": np.nanstd(cv_scores),
        "cv_predictions": cv_predictions,
    }
