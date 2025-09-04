#!/usr/bin/env python3
"""MLflow integration for HMM clustering."""

from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

from src.utils.logger import system_logger

# Safe imports for MLflow utilities
try:
    from src.utils.enhanced_mlflow_integration import (
        log_step_dataframe_with_standardized_name,
        log_step_report,
        log_step_metrics,
        log_model_with_standardized_name
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger = system_logger.getChild("HMMMLflowIntegration")
    logger.warning("⚠️ MLflow integration not available - logging disabled")

logger = system_logger.getChild("HMMMLflowIntegration")


class MLflowIntegrationMixin:
    """Mixin class for MLflow integration in HMM clustering."""
    
    def log_to_mlflow(self,
                     results: Dict[str, Any],
                     training_input: Dict[str, Any]) -> None:
        """Log HMM clustering results to MLflow.
        
        Args:
            results: Results from HMM clustering
            training_input: Training input configuration
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("⚠️ MLflow not available - skipping logging")
            return
            
        try:
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            
            # Log composite clusters DataFrame
            if 'composite_df' in results:
                composite_df = results['composite_df']
                artifact_name = log_step_dataframe_with_standardized_name(
                    config=self.config,
                    step_name='step03_hmm_regime_discovery',
                    df=composite_df,
                    artifact_type='composite_clusters',
                    additional_metadata={
                        'artifact_type': 'composite_clusters',
                        'dataframe_shape': list(composite_df.shape),
                        'regime_count': results.get('n_regimes', 0),
                        'timeframe': timeframe,
                        'enhanced_version': True
                    }
                )
                logger.info(f'✅ Logged composite clusters to MLflow: {artifact_name}')
            
            # Log metrics
            metrics = {}
            
            # Basic metrics
            metrics['step3_n_regimes'] = float(results.get('n_regimes', 0))
            metrics['step3_total_periods'] = float(len(results.get('regime_states', [])))
            metrics['step3_execution_time'] = float(results.get('execution_time', 0))
            metrics['step3_overall_quality_score'] = float(results.get('overall_quality_score', 0))
            
            # Regime distribution metrics
            regime_dist = results.get('regime_distribution', {})
            for regime, count in regime_dist.items():
                metrics[f'step3_regime_{regime}_count'] = float(count)
            
            # Transition metrics
            transitions = results.get('regime_transitions', {})
            if transitions:
                metrics['step3_total_transitions'] = float(transitions.get('total_transitions', 0))
                metrics['step3_transition_rate'] = float(transitions.get('transition_rate', 0))
            
            # Ensemble quality metrics
            ensemble_quality = results.get('ensemble_quality', {})
            if ensemble_quality:
                for method, score in ensemble_quality.items():
                    if isinstance(score, (int, float)):
                        metrics[f'step3_ensemble_{method}_quality'] = float(score)
            
            # Economic validation metrics
            metrics['step3_economic_significance'] = float(results.get('economic_significance', False))
            
            # ML transition detection metrics
            if results.get('enhanced_ml_transition_detection', False):
                transition_models = results.get('transition_models', {})
                if transition_models:
                    metrics['step3_ml_best_performance'] = float(
                        transition_models.get('best_performance', 0)
                    )
                    
                    final_perf = transition_models.get('final_performance', {})
                    if isinstance(final_perf, dict):
                        for metric_name, value in final_perf.items():
                            if isinstance(value, (int, float)):
                                metrics[f'step3_ml_{metric_name}'] = float(value)
            
            # Log all metrics
            if metrics:
                log_step_metrics(
                    config=self.config,
                    step_name='step03_hmm_regime_discovery',
                    metrics=metrics,
                    additional_metadata={
                        'metrics_type': 'enhanced_regime_discovery',
                        'hmm_states': results.get('n_regimes', 0),
                        'enhanced_features': True
                    }
                )
                logger.info(f'✅ Logged {len(metrics)} metrics to MLflow')
            
            # Log comprehensive report
            report_data = {
                'execution_summary': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'n_regimes': results.get('n_regimes', 0),
                    'execution_time': results.get('execution_time', 0),
                    'overall_quality_score': results.get('overall_quality_score', 0)
                },
                'regime_analysis': {
                    'distribution': results.get('regime_distribution', {}),
                    'transitions': results.get('regime_transitions', {}),
                    'economic_significance': results.get('economic_significance', False)
                },
                'ensemble_results': {
                    'quality_scores': results.get('ensemble_quality', {}),
                    'weights': results.get('ensemble_weights', {})
                },
                'ml_transition_detection': {
                    'enabled': results.get('enhanced_ml_transition_detection', False),
                    'performance': results.get('transition_models', {})
                },
                'optimization': {
                    'bayesian_optimization_used': 'optimized_params' in results,
                    'parameters': results.get('optimized_params', {})
                }
            }
            
            report_name = log_step_report(
                config=self.config,
                step_name='step03_hmm_regime_discovery',
                report_data=report_data,
                report_type='enhanced_regime_discovery_report',
                additional_metadata={
                    'enhanced_version': True,
                    'n_regimes': results.get('n_regimes', 0),
                    'features_used': len(results.get('feature_names', []))
                }
            )
            logger.info(f'✅ Logged comprehensive report to MLflow: {report_name}')
            
            # Log model artifacts if available
            if 'transition_models' in results and results['transition_models']:
                # Note: Actual model logging would require the model objects
                # This logs model metadata
                model_metadata = {
                    'model_type': 'enhanced_ml_transition_detector',
                    'features': results['transition_models'].get('selected_features', []),
                    'performance': results['transition_models'].get('final_performance', {}),
                    'training_completed': results['transition_models'].get('lgb_training_completed', False)
                }
                
                # Log as a report since we don't have actual model objects
                model_report_name = log_step_report(
                    config=self.config,
                    step_name='step03_hmm_regime_discovery',
                    report_data=model_metadata,
                    report_type='transition_model_metadata',
                    additional_metadata={
                        'model_type': 'enhanced_ml_transition_detector'
                    }
                )
                logger.info(f'✅ Logged model metadata to MLflow: {model_report_name}')
            
            logger.info('✅ Successfully logged all artifacts to MLflow')
            
        except Exception as e:
            logger.error(f'❌ Error logging to MLflow: {e}')
            # Don't raise - MLflow logging should not break the pipeline