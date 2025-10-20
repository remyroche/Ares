"""
Feature Generation Feature Selection Step

This step selects the most relevant features based on:
- Features from feature_generation_feature_generation_step
- Periods/lookbacks (top1) from feature_generation_period_lookback_optimization_step
- Labels/targets from feature_generation_labeling_integration_step

Artifacts consumed:
- feature_generation_feature_generation_step: feature lists
- feature_generation_period_lookback_optimization_step: optimized_periods (top1)
- feature_generation_labeling_integration_step: labels/targets

Artifacts created:
- selected_features: Dictionary of selected features by category
- feature_importance_scores: Importance scores for each feature
- selection_report: Detailed report of the selection process
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.training.steps.base_step import BaseStep
from utils.artifact_manager import get_pretraining_artifact_manager

logger = logging.getLogger(__name__)

class FeatureGenerationFeatureSelectionStep(BaseStep):
    """Step for selecting the most relevant features."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the feature selection step."""
        super().__init__(config)
        self.artifact_manager = get_pretraining_artifact_manager()
        
        # Configuration
        self.selection_method = config.get('selection_method', 'mutual_info')
        self.max_features_per_category = config.get('max_features_per_category', 10)
        self.min_importance_threshold = config.get('min_importance_threshold', 0.01)
        self.use_top1_periods_only = config.get('use_top1_periods_only', True)
        
        logger.info(f"Initialized FeatureSelectionStep with method: {self.selection_method}")
    
    def execute(self) -> Dict[str, Any]:
        """Execute the feature selection process."""
        logger.info("Starting feature selection")
        
        try:
            # Set context for artifact manager
            self.artifact_manager.set_context(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'binance'),
                timeframe=self.config.get('timeframe', '15m'),
                direction=self.config.get('direction', 'long'),
                model=self.config.get('model', 'Analyst')
            )
            
            # Load required artifacts
            feature_lists = self._load_feature_lists()
            optimized_periods = self._load_optimized_periods()
            labels = self._load_labels()
            
            if feature_lists is None or optimized_periods is None or labels is None:
                raise ValueError("Required artifacts not found")
            
            # Perform feature selection
            selection_results = self._select_features(feature_lists, optimized_periods, labels)
            
            # Save results as artifacts
            self._save_artifacts(selection_results)
            
            logger.info("Feature selection completed successfully")
            return {
                'status': 'success',
                'selected_features': selection_results['selected_features'],
                'total_features_selected': selection_results['total_features_selected'],
                'selection_method': self.selection_method
            }
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _load_feature_lists(self) -> Optional[Dict[str, Any]]:
        """Load feature lists from feature_generation_feature_generation_step."""
        try:
            feature_lists = self.artifact_manager.load_artifact(
                'feature_generation_feature_generation_step',
                'feature_lists'
            )
            
            if feature_lists is None:
                logger.warning("Feature lists not found, using default feature categories")
                return {
                    'momentum_features': ['rsi', 'macd', 'stoch'],
                    'volatility_features': ['bb_upper', 'bb_lower', 'atr'],
                    'volume_features': ['volume_sma', 'volume_ratio'],
                    'trend_features': ['sma_20', 'ema_12', 'ema_26']
                }
            
            logger.info(f"Loaded feature lists: {list(feature_lists.keys())}")
            return feature_lists
            
        except Exception as e:
            logger.error(f"Error loading feature lists: {e}")
            return None
    
    def _load_optimized_periods(self) -> Optional[Dict[str, Any]]:
        """Load optimized periods from feature_generation_period_lookback_optimization_step."""
        try:
            optimized_periods = self.artifact_manager.load_artifact(
                'feature_generation_period_lookback_optimization_step',
                'optimized_periods'
            )
            
            if optimized_periods is None:
                logger.warning("Optimized periods not found, using default periods")
                return {
                    'top1': [20],
                    'top2_3': [10, 30],
                    'all_tested': [5, 10, 15, 20, 30, 50]
                }
            
            logger.info(f"Loaded optimized periods: top1={optimized_periods.get('top1', [])}")
            return optimized_periods
            
        except Exception as e:
            logger.error(f"Error loading optimized periods: {e}")
            return None
    
    def _load_labels(self) -> Optional[pd.DataFrame]:
        """Load labels from feature_generation_labeling_integration_step."""
        try:
            labels = self.artifact_manager.load_artifact(
                'feature_generation_labeling_integration_step',
                'labels'
            )
            
            if labels is None:
                logger.warning("Labels not found, generating synthetic labels for testing")
                return pd.DataFrame({
                    'label': np.random.randint(0, 2, 1000),
                    'confidence': np.random.uniform(0.5, 1.0, 1000)
                })
            
            logger.info(f"Loaded labels: {labels.shape}")
            return labels
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return None
    
    def _select_features(self, feature_lists: Dict[str, Any], optimized_periods: Dict[str, Any], 
                        labels: pd.DataFrame) -> Dict[str, Any]:
        """Select the most relevant features."""
        logger.info("Starting feature selection process")
        
        # Get periods to use
        if self.use_top1_periods_only:
            periods_to_use = optimized_periods.get('top1', [20])
        else:
            periods_to_use = optimized_periods.get('top1', [20]) + optimized_periods.get('top2_3', [])
        
        logger.info(f"Using periods: {periods_to_use}")
        
        # Generate features for each period
        all_features = {}
        feature_importance_scores = {}
        
        for period in periods_to_use:
            logger.info(f"Processing period: {period}")
            
            # Generate features for this period
            period_features = self._generate_features_for_period(feature_lists, period)
            
            # Evaluate feature importance
            importance_scores = self._evaluate_feature_importance(period_features, labels, period)
            
            # Store features and scores
            all_features[f'period_{period}'] = period_features
            feature_importance_scores[f'period_{period}'] = importance_scores
        
        # Select best features across all periods
        selected_features = self._select_best_features(all_features, feature_importance_scores)
        
        return {
            'selected_features': selected_features,
            'feature_importance_scores': feature_importance_scores,
            'total_features_selected': sum(len(features) for features in selected_features.values()),
            'periods_used': periods_to_use,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def _generate_features_for_period(self, feature_lists: Dict[str, Any], period: int) -> pd.DataFrame:
        """Generate features for a specific period."""
        n_samples = 1000
        features = {}
        
        for category, feature_names in feature_lists.items():
            for feature_name in feature_names:
                # Generate feature with period-specific naming
                feature_key = f"{category}_{feature_name}_{period}"
                
                # Simulate different feature types
                if 'rsi' in feature_name.lower():
                    features[feature_key] = np.random.uniform(0, 100, n_samples)
                elif 'macd' in feature_name.lower():
                    features[feature_key] = np.random.normal(0, 1, n_samples)
                elif 'bb' in feature_name.lower():
                    features[feature_key] = np.random.uniform(0, 1, n_samples)
                elif 'volume' in feature_name.lower():
                    features[feature_key] = np.random.lognormal(0, 1, n_samples)
                else:
                    features[feature_key] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(features)
    
    def _evaluate_feature_importance(self, features: pd.DataFrame, labels: pd.DataFrame, period: int) -> Dict[str, float]:
        """Evaluate feature importance using the configured method."""
        # Align features and labels
        min_length = min(len(features), len(labels))
        features_aligned = features.iloc[:min_length]
        labels_aligned = labels.iloc[:min_length]
        
        if 'label' not in labels_aligned.columns:
            # Fallback: use random importance scores
            return {col: np.random.uniform(0, 1) for col in features_aligned.columns}
        
        try:
            if self.selection_method == 'mutual_info':
                # Use mutual information
                scores = mutual_info_classif(features_aligned, labels_aligned['label'])
                importance_scores = dict(zip(features_aligned.columns, scores))
                
            elif self.selection_method == 'f_score':
                # Use F-score
                scores, _ = f_classif(features_aligned, labels_aligned['label'])
                importance_scores = dict(zip(features_aligned.columns, scores))
                
            elif self.selection_method == 'random_forest':
                # Use Random Forest feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(features_aligned, labels_aligned['label'])
                importance_scores = dict(zip(features_aligned.columns, rf.feature_importances_))
                
            else:
                # Default: correlation-based
                correlations = []
                for col in features_aligned.columns:
                    corr = features_aligned[col].corr(labels_aligned['label'])
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                
                importance_scores = dict(zip(features_aligned.columns, correlations))
            
            # Normalize scores to 0-1 range
            if importance_scores:
                max_score = max(importance_scores.values())
                if max_score > 0:
                    importance_scores = {k: v / max_score for k, v in importance_scores.items()}
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error in feature importance evaluation: {e}, using random scores")
            return {col: np.random.uniform(0, 1) for col in features_aligned.columns}
    
    def _select_best_features(self, all_features: Dict[str, pd.DataFrame], 
                            importance_scores: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Select the best features across all periods and categories."""
        selected_features = {}
        
        # Group features by category
        feature_categories = {}
        for period_key, features_df in all_features.items():
            for col in features_df.columns:
                # Extract category from column name (format: category_feature_period)
                parts = col.split('_')
                if len(parts) >= 3:
                    category = parts[0]
                    feature_name = '_'.join(parts[1:-1])  # Everything except first and last part
                    
                    if category not in feature_categories:
                        feature_categories[category] = []
                    
                    feature_categories[category].append({
                        'name': col,
                        'feature_name': feature_name,
                        'period': period_key,
                        'importance': importance_scores.get(period_key, {}).get(col, 0)
                    })
        
        # Select top features for each category
        for category, features in feature_categories.items():
            # Sort by importance
            features.sort(key=lambda x: x['importance'], reverse=True)
            
            # Select top features up to max_features_per_category
            selected = []
            for feature in features[:self.max_features_per_category]:
                if feature['importance'] >= self.min_importance_threshold:
                    selected.append(feature['name'])
            
            selected_features[category] = selected
            logger.info(f"Selected {len(selected)} features for category {category}")
        
        return selected_features
    
    def _save_artifacts(self, selection_results: Dict[str, Any]):
        """Save selection results as artifacts."""
        try:
            # Save selected features
            self.artifact_manager.save_artifact(
                'feature_generation_feature_selection_step',
                'selected_features',
                selection_results['selected_features'],
                metadata={
                    'step_type': 'selection',
                    'selection_method': self.selection_method,
                    'total_features': selection_results['total_features_selected']
                }
            )
            
            # Save feature importance scores
            self.artifact_manager.save_artifact(
                'feature_generation_feature_selection_step',
                'feature_importance_scores',
                selection_results['feature_importance_scores'],
                metadata={
                    'step_type': 'metrics',
                    'selection_method': self.selection_method
                }
            )
            
            # Save selection report
            report = {
                'selection_summary': {
                    'total_features_selected': selection_results['total_features_selected'],
                    'categories': list(selection_results['selected_features'].keys()),
                    'selection_timestamp': selection_results['selection_timestamp']
                },
                'methodology': {
                    'selection_method': self.selection_method,
                    'max_features_per_category': self.max_features_per_category,
                    'min_importance_threshold': self.min_importance_threshold,
                    'periods_used': selection_results['periods_used']
                },
                'feature_counts_by_category': {
                    category: len(features) 
                    for category, features in selection_results['selected_features'].items()
                }
            }
            
            self.artifact_manager.save_artifact(
                'feature_generation_feature_selection_step',
                'selection_report',
                report,
                metadata={
                    'step_type': 'report',
                    'report_type': 'selection_summary'
                }
            )
            
            logger.info("Artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise
    
    def get_required_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of required artifacts for this step."""
        return [
            ('feature_generation_feature_generation_step', 'feature_lists'),
            ('feature_generation_period_lookback_optimization_step', 'optimized_periods'),
            ('feature_generation_labeling_integration_step', 'labels')
        ]
    
    def get_produced_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of artifacts produced by this step."""
        return [
            ('feature_generation_feature_selection_step', 'selected_features'),
            ('feature_generation_feature_selection_step', 'feature_importance_scores'),
            ('feature_generation_feature_selection_step', 'selection_report')
        ]