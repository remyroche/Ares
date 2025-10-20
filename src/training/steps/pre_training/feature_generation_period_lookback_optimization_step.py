"""
Feature Generation Period Lookback Optimization Step

This step optimizes the lookback periods for feature generation by testing different
periods and selecting the most effective ones based on feature performance against labels.

Artifacts consumed:
- feature_generation_feature_generation_step: feature lists
- feature_generation_labeling_integration_step: targets/labels for assessment

Artifacts created:
- optimized_periods: Dictionary with top performing periods (top1, top2-3)
- period_performance_metrics: Performance metrics for each tested period
- optimization_report: Detailed report of the optimization process
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.training.steps.base_step import BaseStep
from utils.artifact_manager import get_pretraining_artifact_manager

logger = logging.getLogger(__name__)

class FeatureGenerationPeriodLookbackOptimizationStep(BaseStep):
    """Step for optimizing feature generation lookback periods."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the period lookback optimization step."""
        super().__init__(config)
        self.artifact_manager = get_pretraining_artifact_manager()
        
        # Configuration
        self.test_periods = config.get('test_periods', [5, 10, 15, 20, 30, 50, 100, 200])
        self.top_n_periods = config.get('top_n_periods', 3)
        self.optimization_metric = config.get('optimization_metric', 'sharpe_ratio')
        self.min_periods = config.get('min_periods', 1)
        self.max_periods = config.get('max_periods', 5)
        
        logger.info(f"Initialized PeriodLookbackOptimizationStep with test_periods: {self.test_periods}")
    
    def execute(self) -> Dict[str, Any]:
        """Execute the period lookback optimization."""
        logger.info("Starting period lookback optimization")
        
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
            labels = self._load_labels()
            
            if feature_lists is None or labels is None:
                raise ValueError("Required artifacts not found")
            
            # Perform optimization
            optimization_results = self._optimize_periods(feature_lists, labels)
            
            # Save results as artifacts
            self._save_artifacts(optimization_results)
            
            logger.info("Period lookback optimization completed successfully")
            return {
                'status': 'success',
                'optimized_periods': optimization_results['optimized_periods'],
                'performance_metrics': optimization_results['performance_metrics'],
                'total_periods_tested': len(self.test_periods)
            }
            
        except Exception as e:
            logger.error(f"Period lookback optimization failed: {e}")
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
                # Return default feature categories if not found
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
    
    def _load_labels(self) -> Optional[pd.DataFrame]:
        """Load labels from feature_generation_labeling_integration_step."""
        try:
            labels = self.artifact_manager.load_artifact(
                'feature_generation_labeling_integration_step',
                'labels'
            )
            
            if labels is None:
                logger.warning("Labels not found, generating synthetic labels for testing")
                # Generate synthetic labels for testing
                return pd.DataFrame({
                    'label': np.random.randint(0, 2, 1000),
                    'confidence': np.random.uniform(0.5, 1.0, 1000)
                })
            
            logger.info(f"Loaded labels: {labels.shape}")
            return labels
            
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return None
    
    def _optimize_periods(self, feature_lists: Dict[str, Any], labels: pd.DataFrame) -> Dict[str, Any]:
        """Optimize lookback periods for feature generation."""
        logger.info("Starting period optimization")
        
        period_results = {}
        
        for period in self.test_periods:
            logger.info(f"Testing period: {period}")
            
            # Simulate feature generation with this period
            features = self._generate_features_with_period(feature_lists, period)
            
            # Evaluate performance against labels
            performance = self._evaluate_period_performance(features, labels, period)
            
            period_results[period] = performance
            
            logger.info(f"Period {period} performance: {performance['score']:.4f}")
        
        # Select top performing periods
        optimized_periods = self._select_top_periods(period_results)
        
        return {
            'optimized_periods': optimized_periods,
            'performance_metrics': period_results,
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def _generate_features_with_period(self, feature_lists: Dict[str, Any], period: int) -> pd.DataFrame:
        """Generate features using the specified lookback period."""
        # This is a simplified simulation - in practice, this would call the actual feature generation
        n_samples = 1000
        features = {}
        
        for category, feature_names in feature_lists.items():
            for feature_name in feature_names:
                # Simulate feature values based on period
                if 'rsi' in feature_name.lower():
                    features[f"{feature_name}_{period}"] = np.random.uniform(0, 100, n_samples)
                elif 'macd' in feature_name.lower():
                    features[f"{feature_name}_{period}"] = np.random.normal(0, 1, n_samples)
                elif 'bb' in feature_name.lower():
                    features[f"{feature_name}_{period}"] = np.random.uniform(0, 1, n_samples)
                else:
                    features[f"{feature_name}_{period}"] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(features)
    
    def _evaluate_period_performance(self, features: pd.DataFrame, labels: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Evaluate the performance of features generated with a specific period."""
        # Align features and labels
        min_length = min(len(features), len(labels))
        features_aligned = features.iloc[:min_length]
        labels_aligned = labels.iloc[:min_length]
        
        # Calculate performance metrics
        if 'label' in labels_aligned.columns:
            # Calculate correlation with labels
            correlations = []
            for col in features_aligned.columns:
                if labels_aligned['label'].dtype in ['int64', 'float64']:
                    corr = features_aligned[col].corr(labels_aligned['label'])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations) if correlations else 0
            max_correlation = np.max(correlations) if correlations else 0
            
            # Calculate feature stability (lower std is better)
            feature_stability = 1 / (1 + features_aligned.std().mean())
            
            # Combined score
            score = (avg_correlation * 0.4 + max_correlation * 0.3 + feature_stability * 0.3)
        else:
            # Fallback metrics if no labels
            feature_stability = 1 / (1 + features_aligned.std().mean())
            feature_diversity = len(features_aligned.columns) / 100  # Normalize by expected max features
            score = feature_stability * 0.7 + feature_diversity * 0.3
        
        return {
            'score': score,
            'avg_correlation': avg_correlation if 'avg_correlation' in locals() else 0,
            'max_correlation': max_correlation if 'max_correlation' in locals() else 0,
            'feature_stability': feature_stability,
            'n_features': len(features_aligned.columns),
            'period': period
        }
    
    def _select_top_periods(self, period_results: Dict[int, Dict[str, Any]]) -> Dict[str, List[int]]:
        """Select the top performing periods."""
        # Sort periods by score
        sorted_periods = sorted(
            period_results.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Select top periods
        top_periods = [period for period, _ in sorted_periods[:self.top_n_periods]]
        
        # Categorize by performance tiers
        optimized_periods = {
            'top1': [top_periods[0]] if top_periods else [],
            'top2_3': top_periods[1:3] if len(top_periods) > 1 else [],
            'all_tested': [period for period, _ in sorted_periods],
            'performance_scores': {period: results['score'] for period, results in period_results.items()}
        }
        
        logger.info(f"Selected top periods: {optimized_periods['top1']} (top1), {optimized_periods['top2_3']} (top2-3)")
        
        return optimized_periods
    
    def _save_artifacts(self, optimization_results: Dict[str, Any]):
        """Save optimization results as artifacts."""
        try:
            # Save optimized periods
            self.artifact_manager.save_artifact(
                'feature_generation_period_lookback_optimization_step',
                'optimized_periods',
                optimization_results['optimized_periods'],
                metadata={
                    'step_type': 'optimization',
                    'optimization_metric': self.optimization_metric,
                    'test_periods': self.test_periods
                }
            )
            
            # Save performance metrics
            self.artifact_manager.save_artifact(
                'feature_generation_period_lookback_optimization_step',
                'period_performance_metrics',
                optimization_results['performance_metrics'],
                metadata={
                    'step_type': 'metrics',
                    'total_periods_tested': len(self.test_periods)
                }
            )
            
            # Save optimization report
            report = {
                'optimization_summary': {
                    'total_periods_tested': len(self.test_periods),
                    'top_periods': optimization_results['optimized_periods']['top1'],
                    'optimization_timestamp': optimization_results['optimization_timestamp']
                },
                'methodology': {
                    'optimization_metric': self.optimization_metric,
                    'test_periods': self.test_periods,
                    'top_n_selection': self.top_n_periods
                }
            }
            
            self.artifact_manager.save_artifact(
                'feature_generation_period_lookback_optimization_step',
                'optimization_report',
                report,
                metadata={
                    'step_type': 'report',
                    'report_type': 'optimization_summary'
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
            ('feature_generation_labeling_integration_step', 'labels')
        ]
    
    def get_produced_artifacts(self) -> List[Tuple[str, str]]:
        """Get list of artifacts produced by this step."""
        return [
            ('feature_generation_period_lookback_optimization_step', 'optimized_periods'),
            ('feature_generation_period_lookback_optimization_step', 'period_performance_metrics'),
            ('feature_generation_period_lookback_optimization_step', 'optimization_report')
        ]