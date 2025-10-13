"""
Feature selection algorithms and methods.

This module contains various feature selection algorithms including
MRMR, LASSO, correlation filtering, RFE, and variance filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from sklearn.feature_selection import (
    mutual_info_regression, 
    mutual_info_classif,
    SelectKBest,
    f_regression,
    f_classif,
    RFE,
    RFECV
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
import warnings

from src.utils.tprint import tprint_debug, tprint_warning


class FeatureSelector:
    """Core feature selection algorithms and methods."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def mrmr_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> List[str]:
        """
        Maximum Relevance Minimum Redundancy (MRMR) feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 MRMR selection: selecting {k} features from {len(X.columns)}")
        
        try:
            # Calculate mutual information with target
            if self._is_classification(y):
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            
            # Calculate pairwise mutual information between features
            n_features = len(X.columns)
            redundancy_matrix = np.zeros((n_features, n_features))
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if self._is_classification(y):
                        mi_ij = mutual_info_classif(
                            X.iloc[:, [i, j]], 
                            y, 
                            random_state=self.random_state
                        )[0]
                    else:
                        mi_ij = mutual_info_regression(
                            X.iloc[:, [i, j]], 
                            y, 
                            random_state=self.random_state
                        )[0]
                    
                    redundancy_matrix[i, j] = mi_ij
                    redundancy_matrix[j, i] = mi_ij
            
            # MRMR greedy selection
            selected_features = []
            remaining_features = list(range(n_features))
            
            # Start with the feature with highest MI
            first_feature = np.argmax(mi_scores)
            selected_features.append(first_feature)
            remaining_features.remove(first_feature)
            
            # Greedily add features that maximize relevance and minimize redundancy
            for _ in range(min(k - 1, len(remaining_features))):
                best_score = -np.inf
                best_feature = None
                
                for feature_idx in remaining_features:
                    # Calculate relevance (MI with target)
                    relevance = mi_scores[feature_idx]
                    
                    # Calculate redundancy (average MI with already selected features)
                    if selected_features:
                        redundancy = np.mean([
                            redundancy_matrix[feature_idx, sel_idx] 
                            for sel_idx in selected_features
                        ])
                    else:
                        redundancy = 0
                    
                    # MRMR score: relevance - redundancy
                    mrmr_score = relevance - redundancy
                    
                    if mrmr_score > best_score:
                        best_score = mrmr_score
                        best_feature = feature_idx
                
                if best_feature is not None:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
            
            selected_feature_names = [X.columns[i] for i in selected_features]
            tprint_debug(f"✅ MRMR selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ MRMR selection failed: {e}")
            # Fallback to simple MI selection
            return self._fallback_mi_selection(X, y, k)
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series, 
                       alpha: Optional[float] = None, 
                       max_features: int = 50) -> List[str]:
        """
        LASSO-based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            alpha: Regularization parameter (if None, use cross-validation)
            max_features: Maximum number of features to select
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 LASSO selection: max {max_features} features from {len(X.columns)}")
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            if alpha is None:
                # Use cross-validation to find optimal alpha
                lasso = LassoCV(cv=5, random_state=self.random_state, max_iter=1000)
            else:
                lasso = Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
            
            lasso.fit(X_scaled, y)
            
            # Get non-zero coefficients
            non_zero_mask = lasso.coef_ != 0
            selected_indices = np.where(non_zero_mask)[0]
            
            # If too many features selected, take the ones with highest absolute coefficients
            if len(selected_indices) > max_features:
                coef_abs = np.abs(lasso.coef_[selected_indices])
                top_indices = np.argsort(coef_abs)[-max_features:]
                selected_indices = selected_indices[top_indices]
            
            selected_feature_names = [X.columns[i] for i in selected_indices]
            tprint_debug(f"✅ LASSO selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ LASSO selection failed: {e}")
            return []
    
    def correlation_filtering(self, X: pd.DataFrame, y: pd.Series, 
                            threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Correlation threshold for removal
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 Correlation filtering: threshold {threshold}")
        
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        high_corr_pairs.append((i, j, corr_matrix.iloc[i, j]))
            
            # Remove features with highest average correlation
            features_to_remove = set()
            for i, j, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
                if i not in features_to_remove and j not in features_to_remove:
                    # Calculate average correlation for each feature
                    avg_corr_i = corr_matrix.iloc[i, :].mean()
                    avg_corr_j = corr_matrix.iloc[j, :].mean()
                    
                    # Remove the feature with higher average correlation
                    if avg_corr_i > avg_corr_j:
                        features_to_remove.add(i)
                    else:
                        features_to_remove.add(j)
            
            # Select remaining features
            remaining_indices = [i for i in range(len(X.columns)) if i not in features_to_remove]
            selected_feature_names = [X.columns[i] for i in remaining_indices]
            
            tprint_debug(f"✅ Correlation filtering: removed {len(features_to_remove)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Correlation filtering failed: {e}")
            return X.columns.tolist()
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 10, 
                                    step: float = 0.1) -> List[str]:
        """
        Recursive Feature Elimination (RFE).
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            step: Step size for elimination
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 RFE selection: {n_features} features from {len(X.columns)}")
        
        try:
            # Choose estimator based on problem type
            if self._is_classification(y):
                estimator = RandomForestClassifier(
                    n_estimators=50, 
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                estimator = RandomForestRegressor(
                    n_estimators=50, 
                    random_state=self.random_state,
                    n_jobs=-1
                )
            
            # Apply RFE
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=step
            )
            
            rfe.fit(X, y)
            
            # Get selected features
            selected_mask = rfe.support_
            selected_feature_names = X.columns[selected_mask].tolist()
            
            tprint_debug(f"✅ RFE selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ RFE selection failed: {e}")
            return []
    
    def variance_filtering(self, X: pd.DataFrame, y: pd.Series, 
                          threshold: float = 0.01) -> List[str]:
        """
        Remove low-variance features.
        
        Args:
            X: Feature matrix
            y: Target variable
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 Variance filtering: threshold {threshold}")
        
        try:
            # Calculate variances
            variances = X.var()
            
            # Select features above threshold
            high_variance_mask = variances > threshold
            selected_feature_names = X.columns[high_variance_mask].tolist()
            
            tprint_debug(f"✅ Variance filtering: selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Variance filtering failed: {e}")
            return X.columns.tolist()
    
    def mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, 
                            k: int = 10) -> List[str]:
        """
        Mutual information-based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        tprint_debug(f"🔍 Mutual information selection: {k} features from {len(X.columns)}")
        
        try:
            if self._is_classification(y):
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            
            # Select top k features
            top_k_indices = np.argsort(mi_scores)[-k:]
            selected_feature_names = [X.columns[i] for i in top_k_indices]
            
            tprint_debug(f"✅ Mutual information selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Mutual information selection failed: {e}")
            return self._fallback_mi_selection(X, y, k)
    
    def ensemble_selection(self, X: pd.DataFrame, y: pd.Series, 
                          methods: List[str] = None,
                          k: int = 10) -> List[str]:
        """
        Ensemble feature selection using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            methods: List of selection methods to use
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        if methods is None:
            methods = ['mrmr', 'lasso', 'mutual_info', 'rfe']
        
        tprint_debug(f"🔍 Ensemble selection: {k} features using {methods}")
        
        try:
            # Get selections from each method
            method_selections = {}
            
            if 'mrmr' in methods:
                method_selections['mrmr'] = self.mrmr_selection(X, y, k)
            
            if 'lasso' in methods:
                method_selections['lasso'] = self.lasso_selection(X, y, max_features=k)
            
            if 'mutual_info' in methods:
                method_selections['mutual_info'] = self.mutual_info_selection(X, y, k)
            
            if 'rfe' in methods:
                method_selections['rfe'] = self.recursive_feature_elimination(X, y, k)
            
            # Count votes for each feature
            feature_votes = {}
            for method, features in method_selections.items():
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            # Select features with most votes
            sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
            selected_feature_names = [feature for feature, votes in sorted_features[:k]]
            
            tprint_debug(f"✅ Ensemble selected {len(selected_feature_names)} features")
            return selected_feature_names
            
        except Exception as e:
            tprint_warning(f"⚠️ Ensemble selection failed: {e}")
            return []
    
    def _is_classification(self, y: pd.Series) -> bool:
        """Check if the target variable is for classification."""
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        # Check if values are integers and limited range
        unique_values = y.unique()
        if len(unique_values) <= 10 and all(isinstance(val, (int, np.integer)) for val in unique_values):
            return True
        
        return False
    
    def _fallback_mi_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """Fallback mutual information selection."""
        try:
            # Simple correlation-based selection as fallback
            correlations = X.corrwith(y).abs()
            top_k_indices = correlations.nlargest(k).index
            return top_k_indices.tolist()
        except Exception:
            # Ultimate fallback: random selection
            return X.columns[:k].tolist()