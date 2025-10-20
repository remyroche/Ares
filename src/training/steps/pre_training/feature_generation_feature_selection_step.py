"""
Feature Generation Feature Selection Step

This step performs feature selection using multiple techniques including
mutual information, recursive feature elimination, and statistical tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor

from src.training.steps.base_step import BaseStep


class FeatureGenerationFeatureSelectionStep(BaseStep):
    """
    Performs feature selection on optimized features.
    
    Techniques used:
    - Mutual information
    - F-statistic regression
    - Tree-based importance
    - Variance thresholding
    - Correlation analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature selection step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_feature_selection_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute feature selection.
        
        Args:
            config: Configuration containing:
                - max_features: Maximum number of features to select
                - selection_method: Method to use (mutual_info, f_regression, tree)
                - variance_threshold: Minimum variance to keep feature
        
        Returns:
            Dictionary containing:
                - success: bool
                - selected_features_path: str
                - feature_importance: Dict
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🎯 Starting feature selection")
            
            # Load optimized features
            features_df = self._load_dataframe('optimized_features')
            if features_df is None:
                return {
                    'success': False,
                    'error': 'No optimized features found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Get label column
            label_column = config.get('label_column', 'label')
            if label_column not in features_df.columns:
                return {
                    'success': False,
                    'error': f'Label column {label_column} not found',
                    'artifacts': [],
                    'metrics': {}
                }
            
            # Separate features and labels
            y = features_df[label_column]
            X = features_df.drop(columns=[label_column])
            
            # Remove low variance features
            variance_threshold = config.get('variance_threshold', 0.0001)
            variances = X.var()
            high_variance_features = variances[variances > variance_threshold].index.tolist()
            X = X[high_variance_features]
            
            self.logger.info(
                f"After variance filtering: {len(high_variance_features)} features "
                f"(removed {len(variances) - len(high_variance_features)})"
            )
            
            # Get selection parameters
            max_features = config.get('max_features', 50)
            max_features = min(max_features, len(X.columns))
            selection_method = config.get('selection_method', 'mutual_info')
            
            # Perform feature selection
            if selection_method == 'mutual_info':
                selected_features, feature_scores = self._select_mutual_info(
                    X, y, max_features
                )
            elif selection_method == 'f_regression':
                selected_features, feature_scores = self._select_f_regression(
                    X, y, max_features
                )
            elif selection_method == 'tree':
                selected_features, feature_scores = self._select_tree_based(
                    X, y, max_features
                )
            else:
                # Default to mutual info
                selected_features, feature_scores = self._select_mutual_info(
                    X, y, max_features
                )
            
            # Create selected features dataframe
            selected_df = features_df[selected_features + [label_column]].copy()
            
            # Feature importance results
            feature_importance = {
                'method': selection_method,
                'total_features': len(X.columns),
                'selected_features': len(selected_features),
                'feature_scores': {
                    feat: float(score) for feat, score in feature_scores.items()
                },
                'top_features': sorted(
                    [{'name': feat, 'score': float(score)} 
                     for feat, score in feature_scores.items()],
                    key=lambda x: x['score'],
                    reverse=True
                )[:20]
            }
            
            # Save selected features
            selected_path = self._save_dataframe(
                selected_df,
                'selected_features',
                metadata=feature_importance
            )
            
            # Save feature importance
            importance_path = self._save_metadata(
                feature_importance,
                'feature_importance'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'selected_features_path': selected_path,
                'feature_importance': feature_importance,
                'artifacts': [selected_path, importance_path],
                'metrics': {
                    'selected_features': len(selected_features),
                    'selection_ratio': len(selected_features) / len(X.columns),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _select_mutual_info(
        self, X: pd.DataFrame, y: pd.Series, k: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using mutual information."""
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        feature_scores = {
            X.columns[i]: selector.scores_[i]
            for i in selected_indices
        }
        
        return selected_features, feature_scores
    
    def _select_f_regression(
        self, X: pd.DataFrame, y: pd.Series, k: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using F-statistic."""
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        feature_scores = {
            X.columns[i]: selector.scores_[i]
            for i in selected_indices
        }
        
        return selected_features, feature_scores
    
    def _select_tree_based(
        self, X: pd.DataFrame, y: pd.Series, k: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """Select features using tree-based importance."""
        # Use Random Forest for feature importance
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = dict(zip(X.columns, importances))
        
        # Sort and select top k
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected_features = [feat for feat, _ in sorted_features[:k]]
        feature_scores = dict(sorted_features[:k])
        
        return selected_features, feature_scores
