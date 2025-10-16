"""
Final Feature Selection Component

This module provides final feature selection functionality for the pre-training pipeline.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from src.utils.logger import system_logger


class FinalFeatureSelectionConfig:
    """
    Configuration for final feature selection.
    """
    
    def __init__(
        self,
        max_features: int = 100,
        min_features: int = 10,
        selection_method: str = "mutual_info",
        scoring_threshold: float = 0.01,
        use_tree_based: bool = True
    ):
        """
        Initialize final feature selection configuration.
        
        Args:
            max_features: Maximum number of features to select
            min_features: Minimum number of features to select
            selection_method: Method for feature selection
            scoring_threshold: Minimum score threshold for features
            use_tree_based: Whether to use tree-based feature importance
        """
        self.max_features = max_features
        self.min_features = min_features
        self.selection_method = selection_method
        self.scoring_threshold = scoring_threshold
        self.use_tree_based = use_tree_based


class FinalFeatureSelectionComponent:
    """
    Final feature selection component.
    """
    
    def __init__(self, config: FinalFeatureSelectionConfig):
        """
        Initialize the final feature selection component.
        
        Args:
            config: Configuration for feature selection
        """
        self.config = config
        self.logger = system_logger.getChild("FinalFeatureSelectionComponent")
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select final features based on the configuration.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Optional list of feature names
            
        Returns:
            List of selected feature names
        """
        try:
            if feature_names is None:
                feature_names = list(X.columns)
            
            # Ensure we don't select more features than available
            max_features = min(self.config.max_features, len(feature_names))
            min_features = min(self.config.min_features, max_features)
            
            if max_features <= 0:
                self.logger.warning("No features to select")
                return []
            
            # Select features based on method
            if self.config.selection_method == "mutual_info":
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=max_features
                )
            elif self.config.selection_method == "f_regression":
                selector = SelectKBest(
                    score_func=f_regression,
                    k=max_features
                )
            else:
                # Default to mutual info
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=max_features
                )
            
            # Fit selector
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Store feature scores
            if hasattr(selector, 'scores_'):
                self.feature_scores = {
                    feature_names[i]: selector.scores_[i]
                    for i in selected_indices
                }
            
            # Apply tree-based selection if enabled
            if self.config.use_tree_based and len(selected_features) > min_features:
                selected_features = self._apply_tree_based_selection(
                    X[selected_features], y, selected_features
                )
            
            # Filter by scoring threshold
            if self.config.scoring_threshold > 0:
                selected_features = [
                    feat for feat in selected_features
                    if self.feature_scores.get(feat, 0) >= self.config.scoring_threshold
                ]
            
            # Ensure minimum features
            if len(selected_features) < min_features:
                # Take top features by score
                scored_features = sorted(
                    self.feature_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                selected_features = [
                    feat for feat, _ in scored_features[:min_features]
                ]
            
            self.selected_features = selected_features
            self.logger.info(f"Selected {len(selected_features)} features")
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {e}")
            return []
    
    def _apply_tree_based_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str]
    ) -> List[str]:
        """
        Apply tree-based feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            
        Returns:
            List of selected features
        """
        try:
            # Use Random Forest for feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance and select top features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top features up to max_features
            max_features = min(self.config.max_features, len(sorted_features))
            selected_features = [feat for feat, _ in sorted_features[:max_features]]
            
            return selected_features
            
        except Exception as e:
            self.logger.error(f"Error in tree-based selection: {e}")
            return feature_names
    
    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get feature scores from the last selection.
        
        Returns:
            Dictionary of feature scores
        """
        return self.feature_scores.copy()
    
    def get_selected_features(self) -> List[str]:
        """
        Get the last selected features.
        
        Returns:
            List of selected feature names
        """
        return self.selected_features.copy()
