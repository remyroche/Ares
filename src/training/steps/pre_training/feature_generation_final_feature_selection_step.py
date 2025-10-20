"""
Feature Generation Final Feature Selection Step

This step performs the final feature selection after interaction generation,
ensuring the best features are selected for model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

from src.training.steps.base_step import BaseStep


class FeatureGenerationFinalFeatureSelectionStep(BaseStep):
    """
    Performs final feature selection on all generated features.
    
    This step:
    - Combines base features and interactions
    - Removes highly correlated features
    - Selects top features using ensemble methods
    - Validates feature quality
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the final feature selection step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_final_feature_selection_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final feature selection.
        
        Args:
            config: Configuration containing:
                - max_features: Maximum final features
                - correlation_threshold: Max correlation between features
                - selection_methods: Methods to use (ensemble)
                - model: Model type (Analyst/Tactician)
        
        Returns:
            Dictionary containing:
                - success: bool
                - final_features_path: str
                - selected_feature_names: List[str]
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🎯 Starting final feature selection")
            
            # Determine which interaction features to load based on model
            model_type = config.get('model', 'Analyst')
            
            if model_type == 'Analyst':
                features_df = self._load_dataframe('analyst_interaction_features')
            elif model_type == 'Tactician':
                features_df = self._load_dataframe('tactician_interaction_features')
            else:
                # Try to load either
                features_df = self._load_dataframe('analyst_interaction_features')
                if features_df is None:
                    features_df = self._load_dataframe('tactician_interaction_features')
            
            if features_df is None:
                return {
                    'success': False,
                    'error': 'No interaction features found',
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
            
            self.logger.info(f"Starting with {len(X.columns)} features")
            
            # 1. Remove highly correlated features
            correlation_threshold = config.get('correlation_threshold', 0.95)
            X = self._remove_correlated_features(X, correlation_threshold)
            self.logger.info(
                f"After correlation filtering: {len(X.columns)} features "
                f"(threshold={correlation_threshold})"
            )
            
            # 2. Get target number of features
            max_features = config.get('max_features', 80)
            max_features = min(max_features, len(X.columns))
            
            # 3. Ensemble feature selection
            selection_methods = config.get(
                'selection_methods',
                ['mutual_info', 'tree', 'correlation']
            )
            
            feature_scores = {}
            
            # Mutual information scores
            if 'mutual_info' in selection_methods:
                mi_scores = self._get_mutual_info_scores(X, y)
                for feat, score in mi_scores.items():
                    feature_scores[feat] = feature_scores.get(feat, 0) + score
            
            # Tree-based importance
            if 'tree' in selection_methods:
                tree_scores = self._get_tree_importance(X, y)
                for feat, score in tree_scores.items():
                    feature_scores[feat] = feature_scores.get(feat, 0) + score
            
            # Correlation with target
            if 'correlation' in selection_methods:
                corr_scores = self._get_correlation_scores(X, y)
                for feat, score in corr_scores.items():
                    feature_scores[feat] = feature_scores.get(feat, 0) + score
            
            # Normalize scores by number of methods used
            for feat in feature_scores:
                feature_scores[feat] /= len(selection_methods)
            
            # 4. Select top features
            sorted_features = sorted(
                feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_feature_names = [feat for feat, _ in sorted_features[:max_features]]
            
            # Create final features dataframe
            final_df = features_df[selected_feature_names + [label_column]].copy()
            
            # Selection results
            selection_results = {
                'initial_features': len(features_df.columns) - 1,
                'after_correlation_filter': len(X.columns),
                'final_selected': len(selected_feature_names),
                'selection_methods': selection_methods,
                'model_type': model_type,
                'top_20_features': [
                    {'name': feat, 'score': float(score)}
                    for feat, score in sorted_features[:20]
                ]
            }
            
            # Save final features
            final_path = self._save_dataframe(
                final_df,
                'final_selected_features',
                metadata=selection_results
            )
            
            # Save feature names
            names_path = self._save_metadata(
                {'selected_features': selected_feature_names},
                'final_feature_names'
            )
            
            # Save selection results
            results_path = self._save_metadata(
                selection_results,
                'final_selection_results'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'final_features_path': final_path,
                'selected_feature_names': selected_feature_names,
                'artifacts': [final_path, names_path, results_path],
                'metrics': {
                    'final_features': len(selected_feature_names),
                    'reduction_ratio': len(selected_feature_names) / (len(features_df.columns) - 1),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Final feature selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _remove_correlated_features(
        self, X: pd.DataFrame, threshold: float = 0.95
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > threshold)
        ]
        
        # Drop features
        X_reduced = X.drop(columns=to_drop)
        
        self.logger.info(f"Removed {len(to_drop)} correlated features")
        
        return X_reduced
    
    def _get_mutual_info_scores(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Get mutual information scores."""
        scores = mutual_info_regression(X, y, random_state=42)
        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1.0
        normalized_scores = scores / max_score
        return dict(zip(X.columns, normalized_scores))
    
    def _get_tree_importance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Get tree-based feature importance."""
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        # Normalize scores
        max_importance = max(importances) if max(importances) > 0 else 1.0
        normalized_importance = importances / max_importance
        
        return dict(zip(X.columns, normalized_importance))
    
    def _get_correlation_scores(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Get correlation-based scores."""
        scores = {}
        for col in X.columns:
            try:
                corr, _ = spearmanr(X[col], y)
                scores[col] = abs(corr)
            except:
                scores[col] = 0.0
        
        # Normalize scores
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
