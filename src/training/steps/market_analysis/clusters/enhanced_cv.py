"""
Enhanced Cross-Validation for Regime Clustering.

This module provides proper time series cross-validation with purging
and comprehensive CV metrics for regime clustering validation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)


class EnhancedCrossValidation:
    """Enhanced cross-validation for regime clustering with purging."""
    
    def __init__(self, cv_folds: int = 5, purged_pct: float = 0.1, min_train_size: int = 100):
        """
        Initialize enhanced cross-validation.
        
        Args:
            cv_folds: Number of cross-validation folds
            purged_pct: Percentage of data to purge around regime transitions
            min_train_size: Minimum training set size
        """
        self.cv_folds = cv_folds
        self.purged_pct = purged_pct
        self.min_train_size = min_train_size
        
    def temporal_cv_split(self, n_samples: int, test_size: float = 0.2) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits with purging.
        
        Args:
            n_samples: Total number of samples
            test_size: Fraction of data to use for testing
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        try:
            splits = []
            tscv = TimeSeriesSplit(n_splits=self.cv_folds, test_size=int(n_samples * test_size))
            
            for train_idx, test_idx in tscv.split(np.arange(n_samples)):
                # Apply purging around regime transitions
                purged_train, purged_test = self._apply_purging(train_idx, test_idx, n_samples)
                
                # Ensure minimum training size
                if len(purged_train) >= self.min_train_size:
                    splits.append((purged_train, purged_test))
                else:
                    tprint_warning(f"Skipping fold with insufficient training data: {len(purged_train)} < {self.min_train_size}")
            
            tprint_info(f"Created {len(splits)} CV splits with purging")
            return splits
            
        except Exception as e:
            tprint_error(f"Temporal CV split failed: {e}")
            return []
    
    def _apply_purging(self, train_idx: np.ndarray, test_idx: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply purging around regime transitions."""
        try:
            # Calculate purge size
            purge_size = int(len(test_idx) * self.purged_pct)
            
            # Purge from end of training set
            if len(train_idx) > purge_size:
                purged_train = train_idx[:-purge_size]
            else:
                purged_train = train_idx
            
            # Purge from start of test set
            if len(test_idx) > purge_size:
                purged_test = test_idx[purge_size:]
            else:
                purged_test = test_idx
            
            return purged_train, purged_test
            
        except Exception as e:
            tprint_warning(f"Purging failed: {e}")
            return train_idx, test_idx
    
    def compute_cv_metrics(self, features: np.ndarray, labels: np.ndarray, 
                          market_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute comprehensive cross-validation metrics.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            market_data: Optional market data for economic validation
            
        Returns:
            Dictionary of CV metrics
        """
        try:
            tprint_info("Computing enhanced cross-validation metrics...")
            
            # Basic CV splits
            cv_splits = self.temporal_cv_split(len(features))
            if not cv_splits:
                return self._get_fallback_metrics()
            
            # Initialize metrics storage
            cv_scores = {
                'silhouette_scores': [],
                'davies_bouldin_scores': [],
                'calinski_harabasz_scores': [],
                'cv_ratios': [],
                'within_cluster_cvs': [],
                'between_cluster_cvs': []
            }
            
            # Compute metrics for each fold
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                try:
                    # Get fold data
                    X_train, X_test = features[train_idx], features[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]
                    
                    # Skip if insufficient data
                    if len(np.unique(y_test)) < 2 or len(X_test) < 10:
                        continue
                    
                    # Compute clustering metrics
                    fold_metrics = self._compute_fold_metrics(X_test, y_test)
                    
                    # Store metrics
                    for key, value in fold_metrics.items():
                        if key in cv_scores and value is not None:
                            cv_scores[key].append(value)
                    
                except Exception as e:
                    tprint_warning(f"Fold {fold_idx} failed: {e}")
                    continue
            
            # Calculate final metrics
            final_metrics = self._aggregate_cv_metrics(cv_scores)
            
            # Add economic validation if market data available
            if market_data is not None:
                economic_metrics = self._compute_economic_cv_metrics(features, labels, market_data, cv_splits)
                final_metrics.update(economic_metrics)
            
            tprint_success(f"CV metrics computed: {len(cv_scores['silhouette_scores'])} folds")
            return final_metrics
            
        except Exception as e:
            tprint_error(f"CV metrics computation failed: {e}")
            return self._get_fallback_metrics()
    
    def _compute_fold_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute metrics for a single fold."""
        try:
            metrics = {}
            
            # Basic clustering metrics
            if len(np.unique(y)) > 1:
                metrics['silhouette_score'] = silhouette_score(X, y)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, y)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, y)
            else:
                metrics['silhouette_score'] = 0.0
                metrics['davies_bouldin_score'] = float('inf')
                metrics['calinski_harabasz_score'] = 0.0
            
            # CV ratio (between-cluster / within-cluster variance)
            cv_ratio = self._compute_cv_ratio(X, y)
            metrics['cv_ratio'] = cv_ratio
            
            # Within-cluster CV
            within_cv = self._compute_within_cluster_cv(X, y)
            metrics['within_cluster_cv'] = within_cv
            
            # Between-cluster CV
            between_cv = self._compute_between_cluster_cv(X, y)
            metrics['between_cluster_cv'] = between_cv
            
            return metrics
            
        except Exception as e:
            tprint_warning(f"Fold metrics computation failed: {e}")
            return {}
    
    def _compute_cv_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute CV ratio (between-cluster / within-cluster variance)."""
        try:
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                return 0.0
            
            # Calculate within-cluster variance
            within_var = 0.0
            total_samples = 0
            
            for label in unique_labels:
                cluster_mask = y == label
                cluster_data = X[cluster_mask]
                if len(cluster_data) > 1:
                    cluster_var = np.var(cluster_data, axis=0).sum()
                    within_var += cluster_var * len(cluster_data)
                    total_samples += len(cluster_data)
            
            if total_samples == 0:
                return 0.0
            
            within_var /= total_samples
            
            # Calculate between-cluster variance
            overall_mean = np.mean(X, axis=0)
            between_var = 0.0
            
            for label in unique_labels:
                cluster_mask = y == label
                cluster_data = X[cluster_mask]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data, axis=0)
                    between_var += len(cluster_data) * np.sum((cluster_mean - overall_mean) ** 2)
            
            between_var /= total_samples
            
            # CV ratio
            if within_var > 0:
                return between_var / within_var
            else:
                return 0.0
                
        except Exception as e:
            tprint_warning(f"CV ratio computation failed: {e}")
            return 0.0
    
    def _compute_within_cluster_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute within-cluster coefficient of variation."""
        try:
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                return 0.0
            
            within_cvs = []
            
            for label in unique_labels:
                cluster_mask = y == label
                cluster_data = X[cluster_mask]
                if len(cluster_data) > 1:
                    cluster_std = np.std(cluster_data, axis=0)
                    cluster_mean = np.mean(cluster_data, axis=0)
                    # Avoid division by zero
                    cluster_mean = np.where(cluster_mean == 0, 1e-8, cluster_mean)
                    cluster_cv = np.mean(cluster_std / cluster_mean)
                    within_cvs.append(cluster_cv)
            
            return np.mean(within_cvs) if within_cvs else 0.0
            
        except Exception as e:
            tprint_warning(f"Within-cluster CV computation failed: {e}")
            return 0.0
    
    def _compute_between_cluster_cv(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute between-cluster coefficient of variation."""
        try:
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                return 0.0
            
            cluster_means = []
            for label in unique_labels:
                cluster_mask = y == label
                cluster_data = X[cluster_mask]
                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data, axis=0)
                    cluster_means.append(cluster_mean)
            
            if len(cluster_means) < 2:
                return 0.0
            
            cluster_means = np.array(cluster_means)
            between_std = np.std(cluster_means, axis=0)
            between_mean = np.mean(cluster_means, axis=0)
            
            # Avoid division by zero
            between_mean = np.where(between_mean == 0, 1e-8, between_mean)
            between_cv = np.mean(between_std / between_mean)
            
            return between_cv
            
        except Exception as e:
            tprint_warning(f"Between-cluster CV computation failed: {e}")
            return 0.0
    
    def _compute_economic_cv_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                   market_data: pd.DataFrame, cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Compute economic cross-validation metrics."""
        try:
            if 'close' not in market_data.columns:
                return {}
            
            economic_scores = []
            
            for train_idx, test_idx in cv_splits:
                try:
                    # Get test period data
                    test_data = market_data.iloc[test_idx]
                    test_labels = labels[test_idx]
                    
                    # Calculate economic metrics for this fold
                    fold_economic_score = self._calculate_economic_separation(test_data, test_labels)
                    economic_scores.append(fold_economic_score)
                    
                except Exception as e:
                    tprint_warning(f"Economic CV fold failed: {e}")
                    continue
            
            if economic_scores:
                return {
                    'economic_cv_score': np.mean(economic_scores),
                    'economic_cv_std': np.std(economic_scores),
                    'economic_cv_stability': 1.0 - np.std(economic_scores) / (np.mean(economic_scores) + 1e-8)
                }
            else:
                return {}
                
        except Exception as e:
            tprint_warning(f"Economic CV metrics failed: {e}")
            return {}
    
    def _calculate_economic_separation(self, market_data: pd.DataFrame, labels: np.ndarray) -> float:
        """Calculate economic separation between regimes."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            regime_volatilities = []
            regime_returns = []
            
            for label in unique_labels:
                regime_mask = labels == label
                regime_data = market_data[regime_mask]
                
                if len(regime_data) > 1 and 'close' in regime_data.columns:
                    # Calculate volatility
                    returns = regime_data['close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std()
                        regime_volatilities.append(volatility)
                        
                        # Calculate average return
                        avg_return = returns.mean()
                        regime_returns.append(avg_return)
            
            if len(regime_volatilities) < 2:
                return 0.0
            
            # Calculate separation based on volatility differences
            vol_separation = np.std(regime_volatilities) / (np.mean(regime_volatilities) + 1e-8)
            
            # Calculate separation based on return differences
            if len(regime_returns) >= 2:
                return_separation = np.std(regime_returns) / (np.std(regime_returns) + 1e-8)
            else:
                return_separation = 0.0
            
            # Combined economic separation score
            economic_separation = (vol_separation + return_separation) / 2
            
            return economic_separation
            
        except Exception as e:
            tprint_warning(f"Economic separation calculation failed: {e}")
            return 0.0
    
    def _aggregate_cv_metrics(self, cv_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """Aggregate CV scores across folds."""
        try:
            final_metrics = {}
            
            for metric_name, scores in cv_scores.items():
                if scores:
                    final_metrics[f'{metric_name}_mean'] = np.mean(scores)
                    final_metrics[f'{metric_name}_std'] = np.std(scores)
                    final_metrics[f'{metric_name}_min'] = np.min(scores)
                    final_metrics[f'{metric_name}_max'] = np.max(scores)
                else:
                    final_metrics[f'{metric_name}_mean'] = 0.0
                    final_metrics[f'{metric_name}_std'] = 0.0
                    final_metrics[f'{metric_name}_min'] = 0.0
                    final_metrics[f'{metric_name}_max'] = 0.0
            
            # Calculate overall CV quality score
            if 'cv_ratio_mean' in final_metrics and 'within_cluster_cv_mean' in final_metrics:
                cv_quality = final_metrics['cv_ratio_mean'] / (final_metrics['within_cluster_cv_mean'] + 1e-8)
                final_metrics['overall_cv_quality'] = cv_quality
            
            return final_metrics
            
        except Exception as e:
            tprint_error(f"CV metrics aggregation failed: {e}")
            return {}
    
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Get fallback metrics when CV fails."""
        return {
            'cv_ratio_mean': 0.0,
            'cv_ratio_std': 0.0,
            'within_cluster_cv_mean': 0.0,
            'between_cluster_cv_mean': 0.0,
            'silhouette_score_mean': 0.0,
            'davies_bouldin_score_mean': float('inf'),
            'calinski_harabasz_score_mean': 0.0,
            'overall_cv_quality': 0.0,
            'economic_cv_score': 0.0,
            'economic_cv_stability': 0.0
        }


def create_enhanced_cv(cv_folds: int = 5, purged_pct: float = 0.1) -> EnhancedCrossValidation:
    """Create enhanced cross-validation instance."""
    return EnhancedCrossValidation(cv_folds=cv_folds, purged_pct=purged_pct)