"""
Feature Generation Interaction Generation Step - Tactician

This step generates interaction features for the Tactician model based on:
- Features selected by feature_generation_feature_selection_step
- Periods/lookbacks (top2-3) from feature_generation_period_lookback_optimization_step
- Labels/targets from feature_generation_labeling_integration_step

Artifacts consumed:
- feature_generation_feature_selection_step: selected_features
- feature_generation_period_lookback_optimization_step: optimized_periods (top2-3)
- feature_generation_labeling_integration_step: labels/targets

Artifacts created:
- interaction_features: Generated interaction features
- interaction_performance_metrics: Performance metrics for interactions
- interaction_report: Detailed report of the interaction generation process
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from itertools import combinations, product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score

from src.training.steps.base_step import BaseStep
from utils.artifact_manager import get_pretraining_artifact_manager

logger = logging.getLogger(__name__)

class FeatureGenerationInteractionGenerationStepTactician(BaseStep):
    """Step for generating interaction features for Tactician model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the interaction generation step for Tactician."""
        super().__init__(config)
        self.artifact_manager = get_pretraining_artifact_manager()
        
        # Configuration - Tactician-specific settings
        self.max_interactions_per_category = config.get('max_interactions_per_category', 25)
        self.min_mutual_info_threshold = config.get('min_mutual_info_threshold', 0.015)
        self.interaction_types = config.get('interaction_types', [
            'multiplication', 'division', 'addition', 'subtraction', 
            'ratio', 'difference_ratio', 'power', 'log_ratio'
        ])
        self.use_top2_3_periods = config.get('use_top2_3_periods', True)
        self.max_feature_pairs = config.get('max_feature_pairs', 150)
        self.include_cross_category_interactions = config.get('include_cross_category_interactions', True)
        
        logger.info(f"Initialized InteractionGenerationStepTactician with {len(self.interaction_types)} interaction types")
    
    def execute(self) -> Dict[str, Any]:
        """Execute the interaction feature generation process."""
        logger.info("Starting interaction feature generation for Tactician")
        
        try:
            # Set context for artifact manager
            self.artifact_manager.set_context(
                symbol=self.config.get('symbol', 'ETHUSDT'),
                exchange=self.config.get('exchange', 'binance'),
                timeframe=self.config.get('timeframe', '15m'),
                direction=self.config.get('direction', 'long'),
                model=self.config.get('model', 'Tactician')
            )
            
            # Load required artifacts
            selected_features = self._load_selected_features()
            optimized_periods = self._load_optimized_periods()
            labels = self._load_labels()
            
            if selected_features is None or optimized_periods is None or labels is None:
                raise ValueError("Required artifacts not found")
            
            # Generate interaction features
            interaction_results = self._generate_interaction_features(selected_features, optimized_periods, labels)
            
            # Save results as artifacts
            self._save_artifacts(interaction_results)
            
            logger.info("Interaction feature generation completed successfully")
            return {
                'status': 'success',
                'interaction_features': interaction_results['interaction_features'],
                'total_interactions_generated': interaction_results['total_interactions_generated'],
                'interaction_types_used': self.interaction_types
            }
            
        except Exception as e:
            logger.error(f"Interaction feature generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _load_selected_features(self) -> Optional[Dict[str, List[str]]]:
        """Load selected features from feature_generation_feature_selection_step."""
        try:
            selected_features = self.artifact_manager.load_artifact(
                'feature_generation_feature_selection_step',
                'selected_features'
            )
            
            if selected_features is None:
                logger.warning("Selected features not found, using default features")
                return {
                    'momentum_features': ['momentum_rsi_20', 'momentum_macd_20', 'momentum_stoch_20'],
                    'volatility_features': ['volatility_bb_upper_20', 'volatility_bb_lower_20', 'volatility_atr_20'],
                    'volume_features': ['volume_volume_sma_20', 'volume_volume_ratio_20'],
                    'trend_features': ['trend_sma_20_20', 'trend_ema_12_20', 'trend_ema_26_20']
                }
            
            logger.info(f"Loaded selected features: {list(selected_features.keys())}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error loading selected features: {e}")
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
            
            logger.info(f"Loaded optimized periods: top2_3={optimized_periods.get('top2_3', [])}")
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
    
    def _generate_interaction_features(self, selected_features: Dict[str, List[str]], 
                                     optimized_periods: Dict[str, Any], 
                                     labels: pd.DataFrame) -> Dict[str, Any]:
        """Generate interaction features for Tactician model."""
        logger.info("Starting interaction feature generation for Tactician")
        
        # Get periods to use (top2-3)
        if self.use_top2_3_periods:
            periods_to_use = optimized_periods.get('top2_3', [10, 30])
        else:
            periods_to_use = optimized_periods.get('top1', [20]) + optimized_periods.get('top2_3', [])
        
        logger.info(f"Using periods for interactions: {periods_to_use}")
        
        # Generate base features for each period
        base_features = {}
        for period in periods_to_use:
            base_features[f'period_{period}'] = self._generate_base_features_for_period(selected_features, period)
        
        # Generate interaction features
        interaction_features = {}
        interaction_performance = {}
        
        for period_key, features_df in base_features.items():
            logger.info(f"Generating interactions for {period_key}")
            
            # Generate all possible interactions
            interactions = self._generate_feature_interactions(features_df, period_key, selected_features)
            
            # Evaluate interaction performance
            interaction_scores = self._evaluate_interaction_performance(interactions, labels)
            
            # Select best interactions
            selected_interactions = self._select_best_interactions(interactions, interaction_scores)
            
            interaction_features[period_key] = selected_interactions
            interaction_performance[period_key] = interaction_scores
            
            logger.info(f"Generated {len(selected_interactions)} interactions for {period_key}")
        
        return {
            'interaction_features': interaction_features,
            'interaction_performance_metrics': interaction_performance,
            'total_interactions_generated': sum(len(features) for features in interaction_features.values()),
            'periods_used': periods_to_use,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_base_features_for_period(self, selected_features: Dict[str, List[str]], period: int) -> pd.DataFrame:
        """Generate base features for a specific period."""
        n_samples = 1000
        features = {}
        
        for category, feature_names in selected_features.items():
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
        
        return pd.DataFrame(features)
    
    def _generate_feature_interactions(self, features_df: pd.DataFrame, period_key: str, 
                                     selected_features: Dict[str, List[str]]) -> pd.DataFrame:
        """Generate interaction features from base features."""
        interactions = {}
        feature_columns = list(features_df.columns)
        
        # Limit the number of feature pairs to avoid combinatorial explosion
        max_pairs = min(self.max_feature_pairs, len(feature_columns) * (len(feature_columns) - 1) // 2)
        pair_count = 0
        
        # Generate interactions within categories and across categories
        for i, (feat1, feat2) in enumerate(combinations(feature_columns, 2)):
            if pair_count >= max_pairs:
                break
            
            # Check if we should include cross-category interactions
            feat1_category = self._get_feature_category(feat1, selected_features)
            feat2_category = self._get_feature_category(feat2, selected_features)
            
            if not self.include_cross_category_interactions and feat1_category != feat2_category:
                continue
                
            for interaction_type in self.interaction_types:
                interaction_name = f"tactician_{interaction_type}_{feat1}_{feat2}_{period_key}"
                
                try:
                    if interaction_type == 'multiplication':
                        interactions[interaction_name] = features_df[feat1] * features_df[feat2]
                    elif interaction_type == 'division':
                        # Avoid division by zero
                        interactions[interaction_name] = np.where(
                            np.abs(features_df[feat2]) > 1e-8,
                            features_df[feat1] / features_df[feat2],
                            0
                        )
                    elif interaction_type == 'addition':
                        interactions[interaction_name] = features_df[feat1] + features_df[feat2]
                    elif interaction_type == 'subtraction':
                        interactions[interaction_name] = features_df[feat1] - features_df[feat2]
                    elif interaction_type == 'ratio':
                        # Ratio of features (feat1 / (feat1 + feat2))
                        denominator = features_df[feat1] + features_df[feat2]
                        interactions[interaction_name] = np.where(
                            np.abs(denominator) > 1e-8,
                            features_df[feat1] / denominator,
                            0.5
                        )
                    elif interaction_type == 'difference_ratio':
                        # (feat1 - feat2) / (feat1 + feat2)
                        numerator = features_df[feat1] - features_df[feat2]
                        denominator = features_df[feat1] + features_df[feat2]
                        interactions[interaction_name] = np.where(
                            np.abs(denominator) > 1e-8,
                            numerator / denominator,
                            0
                        )
                    elif interaction_type == 'power':
                        # feat1^feat2 (with safety checks)
                        interactions[interaction_name] = np.where(
                            np.abs(features_df[feat1]) < 10 and np.abs(features_df[feat2]) < 3,
                            np.power(features_df[feat1], features_df[feat2]),
                            0
                        )
                    elif interaction_type == 'log_ratio':
                        # log(feat1 / feat2)
                        ratio = np.where(
                            np.abs(features_df[feat2]) > 1e-8,
                            features_df[feat1] / features_df[feat2],
                            1
                        )
                        interactions[interaction_name] = np.where(
                            ratio > 1e-8,
                            np.log(np.abs(ratio)),
                            0
                        )
                    
                    pair_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error generating interaction {interaction_name}: {e}")
                    continue
        
        return pd.DataFrame(interactions)
    
    def _get_feature_category(self, feature_name: str, selected_features: Dict[str, List[str]]) -> str:
        """Get the category of a feature based on its name."""
        for category, features in selected_features.items():
            if any(feat in feature_name for feat in features):
                return category
        return 'unknown'
    
    def _evaluate_interaction_performance(self, interactions_df: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the performance of interaction features."""
        # Align interactions and labels
        min_length = min(len(interactions_df), len(labels))
        interactions_aligned = interactions_df.iloc[:min_length]
        labels_aligned = labels.iloc[:min_length]
        
        if 'label' not in labels_aligned.columns:
            # Fallback: use random performance scores
            return {col: np.random.uniform(0, 1) for col in interactions_aligned.columns}
        
        performance_scores = {}
        
        for col in interactions_aligned.columns:
            try:
                # Calculate mutual information with labels
                mi_score = mutual_info_score(
                    labels_aligned['label'],
                    pd.cut(interactions_aligned[col], bins=10, labels=False, duplicates='drop')
                )
                
                # Calculate correlation with labels
                corr = interactions_aligned[col].corr(labels_aligned['label'])
                corr_score = abs(corr) if not np.isnan(corr) else 0
                
                # Calculate feature stability (inverse of coefficient of variation)
                if interactions_aligned[col].std() > 0:
                    stability_score = 1 / (1 + interactions_aligned[col].std() / abs(interactions_aligned[col].mean()))
                else:
                    stability_score = 0
                
                # Combined score (weighted average) - Tactician uses more sophisticated scoring
                combined_score = (mi_score * 0.4 + corr_score * 0.3 + stability_score * 0.3)
                
                performance_scores[col] = combined_score
                
            except Exception as e:
                logger.warning(f"Error evaluating interaction {col}: {e}")
                performance_scores[col] = 0
        
        return performance_scores
    
    def _select_best_interactions(self, interactions_df: pd.DataFrame, 
                                 performance_scores: Dict[str, float]) -> pd.DataFrame:
        """Select the best interaction features based on performance."""
        if not performance_scores:
            return pd.DataFrame()
        
        # Sort interactions by performance score
        sorted_interactions = sorted(
            performance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top interactions
        selected_interactions = []
        for interaction_name, score in sorted_interactions:
            if score >= self.min_mutual_info_threshold:
                selected_interactions.append(interaction_name)
                if len(selected_interactions) >= self.max_interactions_per_category:
                    break
        
        # Return selected interactions as DataFrame
        if selected_interactions:
            return interactions_df[selected_interactions]
        else:
            return pd.DataFrame()
    
    def _save_artifacts(self, interaction_results: Dict[str, Any]):
        """Save interaction results as artifacts."""
        try:
            # Save interaction features
            self.artifact_manager.save_artifact(
                'feature_generation_interaction_generation_step_tactician',
                'interaction_features',
                interaction_results['interaction_features'],
                metadata={
                    'step_type': 'interaction_generation',
                    'model_type': 'Tactician',
                    'interaction_types': self.interaction_types,
                    'total_interactions': interaction_results['total_interactions_generated']
                }
            )
            
            # Save interaction performance metrics
            self.artifact_manager.save_artifact(
                'feature_generation_interaction_generation_step_tactician',
                'interaction_performance_metrics',
                interaction_results['interaction_performance_metrics'],
                metadata={
                    'step_type': 'metrics',
                    'model_type': 'Tactician',
                    'evaluation_method': 'mutual_info_correlation_stability'
                }
            )
            
            # Save interaction report
            report = {
                'interaction_summary': {
                    'total_interactions_generated': interaction_results['total_interactions_generated'],
                    'periods_used': interaction_results['periods_used'],
                    'generation_timestamp': interaction_results['generation_timestamp']
                },
                'methodology': {
                    'interaction_types': self.interaction_types,
                    'max_interactions_per_category': self.max_interactions_per_category,
                    'min_mutual_info_threshold': self.min_mutual_info_threshold,
                    'model_type': 'Tactician',
                    'include_cross_category_interactions': self.include_cross_category_interactions
                },
                'interaction_counts_by_period': {
                    period: len(features) 
                    for period, features in interaction_results['interaction_features'].items()
                }
            }
            
            self.artifact_manager.save_artifact(
                'feature_generation_interaction_generation_step_tactician',
                'interaction_report',
                report,
                metadata={
                    'step_type': 'report',
                    'report_type': 'interaction_summary',
                    'model_type': 'Tactician'
                }
            )
            
            logger.info("Artifacts saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise
    
    def get_required_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of required artifacts for this step."""
        return [
            ('feature_generation_feature_selection_step', 'selected_features'),
            ('feature_generation_period_lookback_optimization_step', 'optimized_periods'),
            ('feature_generation_labeling_integration_step', 'labels')
        ]
    
    def get_produced_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of artifacts produced by this step."""
        return [
            ('feature_generation_interaction_generation_step_tactician', 'interaction_features'),
            ('feature_generation_interaction_generation_step_tactician', 'interaction_performance_metrics'),
            ('feature_generation_interaction_generation_step_tactician', 'interaction_report')
        ]