"""
Feature Generation Period Lookback Optimization Step

This step optimizes the lookback periods for features based on their
predictive power and correlation with the target variable.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from scipy.stats import spearmanr

from src.training.steps.base_step import BaseStep


class FeatureGenerationPeriodLookbackOptimizationStep(BaseStep):
    """
    Optimizes feature lookback periods.
    
    This step:
    - Tests multiple lookback periods for each feature
    - Evaluates correlation with target variable
    - Selects optimal periods
    - Removes redundant features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the period lookback optimization step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(
            step_name="feature_generation_period_lookback_optimization_step",
            config=config
        )
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute lookback period optimization.
        
        Args:
            config: Configuration containing:
                - test_periods: List of periods to test
                - correlation_threshold: Minimum correlation to keep feature
                - max_features_per_type: Maximum features per type
        
        Returns:
            Dictionary containing:
                - success: bool
                - optimized_features_path: str
                - optimization_results: Dict
                - artifacts: list
                - metrics: dict
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🎯 Starting lookback period optimization")
            
            # Load generated features
            features_df = self._load_dataframe('generated_features')
            if features_df is None:
                return {
                    'success': False,
                    'error': 'No generated features found',
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
            
            # Calculate feature correlations with target
            correlations = {}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    # Use Spearman correlation (robust to outliers)
                    corr, p_value = spearmanr(X[col].dropna(), y.loc[X[col].dropna().index])
                    correlations[col] = {
                        'correlation': abs(corr),
                        'p_value': p_value,
                        'direction': 'positive' if corr > 0 else 'negative'
                    }
            
            # Filter features by correlation threshold
            correlation_threshold = config.get('correlation_threshold', 0.01)
            significant_features = {
                col: info for col, info in correlations.items()
                if info['correlation'] >= correlation_threshold
            }
            
            self.logger.info(
                f"Found {len(significant_features)}/{len(correlations)} "
                f"features with correlation >= {correlation_threshold}"
            )
            
            # Group features by type (based on naming convention)
            feature_groups = self._group_features_by_type(significant_features.keys())
            
            # Select best features per group
            max_per_type = config.get('max_features_per_type', 10)
            selected_features = []
            
            for feature_type, features in feature_groups.items():
                # Sort by correlation
                sorted_features = sorted(
                    features,
                    key=lambda f: correlations[f]['correlation'],
                    reverse=True
                )
                # Take top N
                selected = sorted_features[:max_per_type]
                selected_features.extend(selected)
                self.logger.info(
                    f"Selected {len(selected)} {feature_type} features "
                    f"(from {len(features)} candidates)"
                )
            
            # Create optimized dataframe
            optimized_df = features_df[selected_features + [label_column]].copy()
            
            # Optimization results
            optimization_results = {
                'total_features_generated': len(X.columns),
                'significant_features': len(significant_features),
                'selected_features': len(selected_features),
                'feature_groups': {
                    group: len(features) for group, features in feature_groups.items()
                },
                'top_features': sorted(
                    [
                        {
                            'name': col,
                            'correlation': correlations[col]['correlation'],
                            'p_value': correlations[col]['p_value']
                        }
                        for col in selected_features
                    ],
                    key=lambda x: x['correlation'],
                    reverse=True
                )[:20]  # Top 20
            }
            
            # Save optimized features
            optimized_path = self._save_dataframe(
                optimized_df,
                'optimized_features',
                metadata=optimization_results
            )
            
            # Save optimization results
            results_path = self._save_metadata(
                optimization_results,
                'lookback_optimization_results'
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'optimized_features_path': optimized_path,
                'optimization_results': optimization_results,
                'artifacts': [optimized_path, results_path],
                'metrics': {
                    'selected_features': len(selected_features),
                    'reduction_ratio': len(selected_features) / len(X.columns),
                    'execution_time': execution_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Lookback optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }
    
    def _group_features_by_type(self, feature_names: List[str]) -> Dict[str, List[str]]:
        """Group features by their type based on naming convention."""
        groups = {}
        
        for feature in feature_names:
            # Extract feature type from name
            # Common patterns: returns_10, momentum_20, rsi_14, etc.
            parts = feature.split('_')
            if len(parts) >= 2:
                feature_type = '_'.join(parts[:-1])  # Everything except the period
            else:
                feature_type = 'other'
            
            if feature_type not in groups:
                groups[feature_type] = []
            groups[feature_type].append(feature)
        
        return groups
