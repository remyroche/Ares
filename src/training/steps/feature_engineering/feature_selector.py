"""
Tactician Feature Selector

This module provides feature selection utilities for tactician models.
"""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.utils.tprint import tprint

logger = logging.getLogger(__name__)

class TacticianFeatureSelector:
    """
    Feature selector for tactician models with multiple selection strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.selected_features = []
        self.feature_scores = {}
        self.selection_method = self.config.get('selection_method', 'mutual_info')
        self.n_features = self.config.get('n_features', 50)
        
    async def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        signal_type: str = 'long'
    ) -> Dict[str, Any]:
        """
        Select features for the given signal type.
        
        Args:
            X: Feature matrix
            y: Target labels
            signal_type: Type of signal ('long' or 'short')
            
        Returns:
            Dictionary with selection results
        """
        try:
            tprint(f"🔍 [FEATURE_SELECTOR] Selecting features for {signal_type} signals...", color="blue")
            
            # Handle missing values
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)
            
            # Select features based on method
            if self.selection_method == 'mutual_info':
                selector = SelectKBest(
                    score_func=mutual_info_classif, 
                    k=min(self.n_features, X_clean.shape[1])
                )
            elif self.selection_method == 'f_classif':
                selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(self.n_features, X_clean.shape[1])
                )
            else:
                # Default to mutual information
                selector = SelectKBest(
                    score_func=mutual_info_classif, 
                    k=min(self.n_features, X_clean.shape[1])
                )
            
            # Fit selector
            X_selected = selector.fit_transform(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            # Store results
            self.selected_features = selected_features
            self.feature_scores = dict(zip(selected_features, selector.scores_[selector.get_support()]))
            
            tprint(f"✅ [FEATURE_SELECTOR] Selected {len(selected_features)} features for {signal_type}", color="green")
            
            return {
                'success': True,
                'selected_features': selected_features,
                'feature_scores': self.feature_scores,
                'n_features': len(selected_features),
                'signal_type': signal_type
            }
            
        except Exception as e:
            tprint(f"❌ [FEATURE_SELECTOR] Error selecting features: {e}", color="red")
            return {
                'success': False,
                'error': str(e),
                'signal_type': signal_type
            }
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Get feature importance using Random Forest.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X.fillna(X.median()), y.fillna(0))
            
            importance_scores = dict(zip(X.columns, rf.feature_importances_))
            return importance_scores
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
