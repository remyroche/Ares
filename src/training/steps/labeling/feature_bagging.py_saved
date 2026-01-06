"""
Feature Bagging and Group Subsampling Module

This module provides feature bagging and group subsampling techniques to reduce
importance concentration and improve feature selection robustness.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
import warnings

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
except ImportError:
    # Fallback implementation if tprint not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class FeatureBagger:
    """
    Feature bagging ensemble to reduce importance concentration.
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        sample_fraction: float = 0.7,
        feature_fraction: float = 0.8,
        random_state: int = 42,
        importance_threshold: float = 0.01,
        min_selection_frequency: float = 0.3,
        max_features_per_bag: Optional[int] = None
    ):
        """
        Initialize feature bagger.
        
        Args:
            n_estimators: Number of bagging estimators
            sample_fraction: Fraction of samples to use per estimator
            feature_fraction: Fraction of features to use per estimator
            random_state: Random state for reproducibility
            importance_threshold: Minimum importance threshold for selection
            min_selection_frequency: Minimum frequency for feature to be selected
            max_features_per_bag: Maximum features per bag (None for auto)
        """
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.feature_fraction = feature_fraction
        self.random_state = random_state
        self.importance_threshold = importance_threshold
        self.min_selection_frequency = min_selection_frequency
        self.max_features_per_bag = max_features_per_bag
        
        self.feature_importances_ = {}
        self.selection_frequency_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Fit feature bagger ensemble.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            Dictionary with selection results
        """
        tprint_info(f"Fitting feature bagger with {self.n_estimators} estimators")
        
        # Initialize feature importance tracking
        feature_importances = {feature: [] for feature in X.columns}
        
        # Determine max features per bag
        n_features = len(X.columns)
        if self.max_features_per_bag is None:
            max_features_per_bag = max(10, int(n_features * self.feature_fraction))
        else:
            max_features_per_bag = min(self.max_features_per_bag, n_features)
        
        # Train bagged estimators
        for estimator_idx in range(self.n_estimators):
            # Subsample data
            rng = np.random.RandomState(self.random_state + estimator_idx)
            n_samples = len(X)
            sample_indices = rng.choice(
                n_samples, 
                size=int(n_samples * self.sample_fraction), 
                replace=False
            )
            
            # Subsample features
            feature_indices = rng.choice(
                n_features,
                size=max_features_per_bag,
                replace=False
            )
            
            X_bag = X.iloc[sample_indices].iloc[:, feature_indices]
            y_bag = y.iloc[sample_indices]
            
            # Skip if insufficient class variety
            if len(np.unique(y_bag)) < 2:
                continue
            
            try:
                # Train Random Forest
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state + estimator_idx,
                    n_jobs=-1
                )
                rf.fit(X_bag, y_bag)
                
                # Store feature importances
                for i, feature in enumerate(X_bag.columns):
                    feature_importances[feature].append(rf.feature_importances_[i])
                    
            except Exception as e:
                tprint_warning(f"Estimator {estimator_idx} failed: {e}")
                continue
        
        # Calculate selection frequency and mean importance
        selection_freq = {}
        mean_importance = {}
        
        for feature in X.columns:
            importances = feature_importances[feature]
            if importances:
                mean_imp = np.mean(importances)
                # Count selections above threshold
                selections = sum(1 for imp in importances if imp >= self.importance_threshold)
                freq = selections / len(importances)
                
                mean_importance[feature] = mean_imp
                selection_freq[feature] = freq
            else:
                mean_importance[feature] = 0.0
                selection_freq[feature] = 0.0
        
        # Select features based on frequency threshold
        selected_features = [
            feature for feature, freq in selection_freq.items()
            if freq >= self.min_selection_frequency
        ]
        
        # Store results
        self.feature_importances_ = mean_importance
        self.selection_frequency_ = selection_freq
        
        # Calculate concentration metrics
        concentration_metrics = self._calculate_concentration_metrics(mean_importance)
        
        results = {
            "selected_features": selected_features,
            "feature_importances": mean_importance,
            "selection_frequency": selection_freq,
            "concentration_metrics": concentration_metrics,
            "n_estimators": self.n_estimators,
            "sample_fraction": self.sample_fraction,
            "feature_fraction": self.feature_fraction,
            "importance_threshold": self.importance_threshold,
            "min_selection_frequency": self.min_selection_frequency
        }
        
        tprint_success(f"Feature bagging selected {len(selected_features)} features")
        return results
    
    def _calculate_concentration_metrics(self, feature_importances: Dict[str, float]) -> Dict[str, float]:
        """Calculate importance concentration metrics."""
        if not feature_importances:
            return {}
        
        importances = np.array(list(feature_importances.values()))
        
        # Normalize to sum to 1
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        # Calculate concentration metrics
        sorted_importances = np.sort(importances)[::-1]
        
        # Top-k concentration
        top_5 = sorted_importances[:5].sum() if len(sorted_importances) >= 5 else sorted_importances.sum()
        top_10 = sorted_importances[:10].sum() if len(sorted_importances) >= 10 else sorted_importances.sum()
        top_20 = sorted_importances[:20].sum() if len(sorted_importances) >= 20 else sorted_importances.sum()
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = (importances ** 2).sum()
        
        # Entropy
        entropy = -(importances * np.log(importances + 1e-8)).sum()
        
        # Gini coefficient
        n = len(importances)
        if n > 1:
            sorted_imp = np.sort(importances)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n
        else:
            gini = 0
        
        return {
            'importance_concentration': float(hhi),
            'top_5_concentration': float(top_5),
            'top_10_concentration': float(top_10),
            'top_20_concentration': float(top_20),
            'hhi': float(hhi),
            'entropy': float(entropy),
            'gini': float(gini),
            'n_features': len(importances)
        }


def feature_bagging_selection(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Apply feature bagging to reduce importance concentration.
    
    Args:
        X: Feature DataFrame
        y: Target series
        config: Configuration dictionary
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (selected_features, selection_results)
    """
    if config is None:
        config = {}
    
    # Initialize feature bagger
    bagger = FeatureBagger(
        n_estimators=config.get('bag_n_estimators', 50),
        sample_fraction=config.get('bag_sample_fraction', 0.7),
        feature_fraction=config.get('bag_feature_fraction', 0.8),
        importance_threshold=config.get('bag_importance_threshold', 0.01),
        min_selection_frequency=config.get('bag_min_selection_frequency', 0.3),
        max_features_per_bag=config.get('bag_max_features_per_bag', None)
    )
    
    # Fit bagger and get selected features
    selection_results = bagger.fit(X, y)
    selected_features = selection_results['selected_features']
    
    if verbose:
        tprint_success(f"Feature bagging selected {len(selected_features)} features")
        tprint_info(f"  Importance concentration: {selection_results['concentration_metrics']['importance_concentration']:.3f}")
        tprint_info(f"  Top-10 concentration: {selection_results['concentration_metrics']['top_10_concentration']:.3f}")
    
    return selected_features, selection_results


def group_subsampling_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_groups: int = 5,
    group_size: int = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Perform group subsampling feature selection.
    
    Splits features into groups and evaluates each group separately,
    then selects top features from each group.
    
    Args:
        X: Feature DataFrame
        y: Target series
        n_groups: Number of feature groups
        group_size: Size of each group (None for auto)
        cv_folds: Number of CV folds
        random_state: Random state
        
    Returns:
        Tuple of (selected_features, selection_results)
    """
    feature_names = list(X.columns)
    n_features = len(feature_names)
    
    # Determine group size
    if group_size is None:
        group_size = max(1, n_features // n_groups)
    
    # Create feature groups
    np.random.seed(random_state)
    shuffled_features = feature_names.copy()
    np.random.shuffle(shuffled_features)
    
    feature_groups = []
    for i in range(0, n_features, group_size):
        group = shuffled_features[i:i + group_size]
        feature_groups.append(group)
    
    # Evaluate each group
    group_scores = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for group_idx, group_features in enumerate(feature_groups):
        X_group = X[group_features]
        
        # Train Random Forest on this group
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        
        # Cross-validate
        cv_scores = []
        for train_idx, val_idx in cv.split(X_group, y):
            X_train, X_val = X_group.iloc[train_idx], X_group.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf.fit(X_train, y_train)
            score = rf.score(X_val, y_val)
            cv_scores.append(score)
        
        group_scores[group_idx] = np.mean(cv_scores)
    
    # Select top groups
    top_group_indices = sorted(group_scores.keys(), key=lambda k: group_scores[k], reverse=True)[:n_groups//2 + 1]
    
    # Select all features from top groups
    selected_features = []
    for group_idx in top_group_indices:
        selected_features.extend(feature_groups[group_idx])
    
    selection_results = {
        'group_scores': group_scores,
        'selected_groups': top_group_indices,
        'feature_groups': feature_groups,
        'n_groups': n_groups,
        'group_size': group_size
    }
    
    return selected_features, selection_results


def reduce_importance_concentration(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "bagging",
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Reduce feature importance concentration using various methods.
    
    Args:
        X: Feature DataFrame
        y: Target series
        method: Method to use ('bagging', 'group_subsampling', 'combined')
        config: Configuration dictionary
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (selected_features, selection_results)
    """
    if verbose:
        tprint_info(f"Reducing importance concentration using {method} method...")
    
    if method == "bagging":
        return feature_bagging_selection(X, y, config, verbose)
    
    elif method == "group_subsampling":
        return group_subsampling_feature_selection(X, y, **config or {})
    
    elif method == "combined":
        # Apply bagging first, then group subsampling
        bag_features, bag_results = feature_bagging_selection(X, y, config, verbose=False)
        
        # Apply group subsampling on bagged features
        X_bagged = X[bag_features]
        group_features, group_results = group_subsampling_feature_selection(
            X_bagged, y, **config.get('group_config', {})
        )
        
        combined_results = {
            'bagging_results': bag_results,
            'group_results': group_results,
            'method': 'combined'
        }
        
        if verbose:
            tprint_success(f"Combined method selected {len(group_features)} features")
        
        return group_features, combined_results
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_importance_concentration(
    feature_importances: Dict[str, float]
) -> Dict[str, float]:
    """
    Analyze importance concentration metrics.
    
    Args:
        feature_importances: Dictionary of feature importances
        
    Returns:
        Dictionary with concentration metrics
    """
    if not feature_importances:
        return {}
    
    importances = np.array(list(feature_importances.values()))
    
    # Normalize to sum to 1
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    # Calculate concentration metrics
    sorted_importances = np.sort(importances)[::-1]
    
    # Top-k concentration
    top_5 = sorted_importances[:5].sum()
    top_10 = sorted_importances[:10].sum()
    top_20 = sorted_importances[:20].sum()
    
    # Herfindahl-Hirschman Index (HHI)
    hhi = (importances ** 2).sum()
    
    # Entropy
    entropy = -(importances * np.log(importances + 1e-8)).sum()
    
    # Gini coefficient
    n = len(importances)
    if n > 1:
        sorted_imp = np.sort(importances)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n
    else:
        gini = 0
    
    return {
        'top_5_concentration': float(top_5),
        'top_10_concentration': float(top_10),
        'top_20_concentration': float(top_20),
        'hhi': float(hhi),
        'entropy': float(entropy),
        'gini': float(gini),
        'n_features': len(importances)
    }
