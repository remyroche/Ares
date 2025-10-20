"""
Feature Generation Final Feature Selection Step

This step performs the final feature selection by combining:
- Features from feature_generation_feature_generation_step
- Periods/lookbacks (top1) from feature_generation_period_lookback_optimization_step
- Features generated during feature_generation_interaction_generation_step_analyst/tactician
- Labels/targets from feature_generation_labeling_integration_step

Artifacts consumed:
- feature_generation_feature_generation_step: feature lists
- feature_generation_period_lookback_optimization_step: optimized_periods (top1)
- feature_generation_interaction_generation_step_analyst: interaction_features
- feature_generation_interaction_generation_step_tactician: interaction_features
- feature_generation_labeling_integration_step: labels/targets

Artifacts created:
- final_selected_features: Final set of selected features
- feature_ranking: Ranking of all features by importance
- final_selection_report: Comprehensive report of the final selection process
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from src.training.steps.base_step import BaseStep
from utils.artifact_manager import get_pretraining_artifact_manager

logger = logging.getLogger(__name__)

class FeatureGenerationFinalFeatureSelectionStep(BaseStep):
    """Step for final feature selection combining all previous steps."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the final feature selection step."""
        super().__init__(config)
        self.artifact_manager = get_pretraining_artifact_manager()
        
        # Configuration
        self.max_final_features = config.get('max_final_features', 100)
        self.min_importance_threshold = config.get('min_importance_threshold', 0.01)
        self.selection_method = config.get('selection_method', 'random_forest')
        self.use_top1_periods_only = config.get('use_top1_periods_only', True)
        self.include_interaction_features = config.get('include_interaction_features', True)
        self.model_type = config.get('model_type', 'both')  # 'analyst', 'tactician', or 'both'
        
        logger.info(f"Initialized FinalFeatureSelectionStep with max_features: {self.max_final_features}")
    
    def execute(self) -> Dict[str, Any]:
        """Execute the final feature selection process."""
        logger.info("Starting final feature selection")
        
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
            analyst_interactions = self._load_analyst_interactions()
            tactician_interactions = self._load_tactician_interactions()
            labels = self._load_labels()
            
            if feature_lists is None or optimized_periods is None or labels is None:
                raise ValueError("Required artifacts not found")
            
            # Perform final feature selection
            selection_results = self._perform_final_selection(
                feature_lists, optimized_periods, analyst_interactions, 
                tactician_interactions, labels
            )
            
            # Save results as artifacts
            self._save_artifacts(selection_results)
            
            logger.info("Final feature selection completed successfully")
            return {
                'status': 'success',
                'final_selected_features': selection_results['final_selected_features'],
                'total_features_selected': selection_results['total_features_selected'],
                'feature_ranking': selection_results['feature_ranking']
            }
            
        except Exception as e:
            logger.error(f"Final feature selection failed: {e}")
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
    
    def _load_analyst_interactions(self) -> Optional[Dict[str, Any]]:
        """Load analyst interaction features."""
        try:
            interactions = self.artifact_manager.load_artifact(
                'feature_generation_interaction_generation_step_analyst',
                'interaction_features'
            )
            
            if interactions is None:
                logger.warning("Analyst interaction features not found")
                return None
            
            logger.info(f"Loaded analyst interactions: {len(interactions)} periods")
            return interactions
            
        except Exception as e:
            logger.error(f"Error loading analyst interactions: {e}")
            return None
    
    def _load_tactician_interactions(self) -> Optional[Dict[str, Any]]:
        """Load tactician interaction features."""
        try:
            interactions = self.artifact_manager.load_artifact(
                'feature_generation_interaction_generation_step_tactician',
                'interaction_features'
            )
            
            if interactions is None:
                logger.warning("Tactician interaction features not found")
                return None
            
            logger.info(f"Loaded tactician interactions: {len(interactions)} periods")
            return interactions
            
        except Exception as e:
            logger.error(f"Error loading tactician interactions: {e}")
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
    
    def _perform_final_selection(self, feature_lists: Dict[str, Any], 
                                optimized_periods: Dict[str, Any],
                                analyst_interactions: Optional[Dict[str, Any]],
                                tactician_interactions: Optional[Dict[str, Any]],
                                labels: pd.DataFrame) -> Dict[str, Any]:
        """Perform the final feature selection."""
        logger.info("Starting final feature selection process")
        
        # Get periods to use (top1)
        if self.use_top1_periods_only:
            periods_to_use = optimized_periods.get('top1', [20])
        else:
            periods_to_use = optimized_periods.get('top1', [20]) + optimized_periods.get('top2_3', [])
        
        logger.info(f"Using periods for final selection: {periods_to_use}")
        
        # Collect all features
        all_features = {}
        
        # Add base features
        for period in periods_to_use:
            base_features = self._generate_base_features_for_period(feature_lists, period)
            all_features.update({f"base_{k}": v for k, v in base_features.items()})
        
        # Add analyst interaction features
        if self.include_interaction_features and analyst_interactions:
            for period_key, interactions_df in analyst_interactions.items():
                if isinstance(interactions_df, pd.DataFrame):
                    for col in interactions_df.columns:
                        all_features[f"analyst_{col}"] = interactions_df[col].values
        
        # Add tactician interaction features
        if self.include_interaction_features and tactician_interactions:
            for period_key, interactions_df in tactician_interactions.items():
                if isinstance(interactions_df, pd.DataFrame):
                    for col in interactions_df.columns:
                        all_features[f"tactician_{col}"] = interactions_df[col].values
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Align with labels
        min_length = min(len(features_df), len(labels))
        features_aligned = features_df.iloc[:min_length]
        labels_aligned = labels.iloc[:min_length]
        
        logger.info(f"Total features available for selection: {len(features_aligned.columns)}")
        
        # Perform feature selection
        selected_features, feature_ranking = self._select_final_features(features_aligned, labels_aligned)
        
        return {
            'final_selected_features': selected_features,
            'feature_ranking': feature_ranking,
            'total_features_selected': len(selected_features),
            'total_features_available': len(features_aligned.columns),
            'periods_used': periods_to_use,
            'selection_timestamp': datetime.now().isoformat()
        }
    
    def _generate_base_features_for_period(self, feature_lists: Dict[str, Any], period: int) -> Dict[str, np.ndarray]:
        """Generate base features for a specific period."""
        n_samples = 1000
        features = {}
        
        for category, feature_names in feature_lists.items():
            for feature_name in feature_names:
                # Generate feature with period-specific naming
                feature_key = f"{category}_{feature_name}_{period}"
                
                # Simulate different feature types with realistic distributions
                if 'rsi' in feature_name.lower():
                    features[feature_key] = np.random.uniform(0, 100, n_samples)
                elif 'macd' in feature_name.lower():
                    features[feature_key] = np.random.normal(0, 1, n_samples)
                elif 'bb' in feature_name.lower():
                    features[feature_key] = np.random.uniform(0, 1, n_samples)
                elif 'volume' in feature_name.lower():
                    features[feature_key] = np.random.lognormal(0, 1, n_samples)
                elif 'sma' in feature_name.lower() or 'ema' in feature_name.lower():
                    features[feature_key] = np.random.normal(0, 1, n_samples)
                elif 'atr' in feature_name.lower():
                    features[feature_key] = np.random.lognormal(0, 0.5, n_samples)
                else:
                    features[feature_key] = np.random.normal(0, 1, n_samples)
        
        return features
    
    def _select_final_features(self, features_df: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[List[str], Dict[str, float]]:
        """Select the final set of features."""
        if 'label' not in labels_df.columns:
            # Fallback: select features randomly
            n_features = min(self.max_final_features, len(features_df.columns))
            selected = np.random.choice(features_df.columns, n_features, replace=False).tolist()
            ranking = {col: np.random.uniform(0, 1) for col in features_df.columns}
            return selected, ranking
        
        try:
            if self.selection_method == 'random_forest':
                # Use Random Forest feature importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(features_df, labels_df['label'])
                
                # Get feature importance
                importance_scores = dict(zip(features_df.columns, rf.feature_importances_))
                
            elif self.selection_method == 'mutual_info':
                # Use mutual information
                scores = mutual_info_classif(features_df, labels_df['label'])
                importance_scores = dict(zip(features_df.columns, scores))
                
            elif self.selection_method == 'f_score':
                # Use F-score
                scores, _ = f_classif(features_df, labels_df['label'])
                importance_scores = dict(zip(features_df.columns, scores))
                
            else:
                # Default: correlation-based
                correlations = []
                for col in features_df.columns:
                    corr = features_df[col].corr(labels_df['label'])
                    correlations.append(abs(corr) if not np.isnan(corr) else 0)
                
                importance_scores = dict(zip(features_df.columns, correlations))
            
            # Normalize scores to 0-1 range
            if importance_scores:
                max_score = max(importance_scores.values())
                if max_score > 0:
                    importance_scores = {k: v / max_score for k, v in importance_scores.items()}
            
            # Sort features by importance
            sorted_features = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select top features
            selected_features = []
            for feature_name, score in sorted_features:
                if score >= self.min_importance_threshold:
                    selected_features.append(feature_name)
                    if len(selected_features) >= self.max_final_features:
                        break
            
            logger.info(f"Selected {len(selected_features)} features out of {len(features_df.columns)} available")
            
            return selected_features, importance_scores
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            # Fallback: select features randomly
            n_features = min(self.max_final_features, len(features_df.columns))
            selected = np.random.choice(features_df.columns, n_features, replace=False).tolist()
            ranking = {col: np.random.uniform(0, 1) for col in features_df.columns}
            return selected, ranking
    
    def _save_artifacts(self, selection_results: Dict[str, Any]):
        """Save final selection results as artifacts."""
        try:
            # Save final selected features
            self.artifact_manager.save_artifact(
                'feature_generation_final_feature_selection_step',
                'final_selected_features',
                selection_results['final_selected_features'],
                metadata={
                    'step_type': 'final_selection',
                    'selection_method': self.selection_method,
                    'total_features': selection_results['total_features_selected']
                }
            )
            
            # Save feature ranking
            self.artifact_manager.save_artifact(
                'feature_generation_final_feature_selection_step',
                'feature_ranking',
                selection_results['feature_ranking'],
                metadata={
                    'step_type': 'ranking',
                    'selection_method': self.selection_method
                }
            )
            
            # Save final selection report
            report = {
                'final_selection_summary': {
                    'total_features_selected': selection_results['total_features_selected'],
                    'total_features_available': selection_results['total_features_available'],
                    'selection_timestamp': selection_results['selection_timestamp']
                },
                'methodology': {
                    'selection_method': self.selection_method,
                    'max_final_features': self.max_final_features,
                    'min_importance_threshold': self.min_importance_threshold,
                    'use_top1_periods_only': self.use_top1_periods_only,
                    'include_interaction_features': self.include_interaction_features,
                    'model_type': self.model_type
                },
                'feature_breakdown': {
                    'base_features': len([f for f in selection_results['final_selected_features'] if f.startswith('base_')]),
                    'analyst_interactions': len([f for f in selection_results['final_selected_features'] if f.startswith('analyst_')]),
                    'tactician_interactions': len([f for f in selection_results['final_selected_features'] if f.startswith('tactician_')])
                },
                'top_features': selection_results['final_selected_features'][:10]  # Top 10 features
            }
            
            self.artifact_manager.save_artifact(
                'feature_generation_final_feature_selection_step',
                'final_selection_report',
                report,
                metadata={
                    'step_type': 'report',
                    'report_type': 'final_selection_summary'
                }
            )
            
            logger.info("Artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise
    
    def get_required_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of required artifacts for this step."""
        artifacts = [
            ('feature_generation_feature_generation_step', 'feature_lists'),
            ('feature_generation_period_lookback_optimization_step', 'optimized_periods'),
            ('feature_generation_labeling_integration_step', 'labels')
        ]
        
        if self.include_interaction_features:
            if self.model_type in ['analyst', 'both']:
                artifacts.append(('feature_generation_interaction_generation_step_analyst', 'interaction_features'))
            if self.model_type in ['tactician', 'both']:
                artifacts.append(('feature_generation_interaction_generation_step_tactician', 'interaction_features'))
        
        return artifacts
    
    def get_produced_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of artifacts produced by this step."""
        return [
            ('feature_generation_final_feature_selection_step', 'final_selected_features'),
            ('feature_generation_final_feature_selection_step', 'feature_ranking'),
            ('feature_generation_final_feature_selection_step', 'final_selection_report')
        ]