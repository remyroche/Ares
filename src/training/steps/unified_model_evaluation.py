"""
Unified Model Evaluation Infrastructure

This module provides unified model evaluation across all training steps using
ModelEvaluationUtilities from ml_common, replacing custom evaluation logic.

Key Features:
- Unified model evaluation using ModelEvaluationUtilities
- Standardized evaluation metrics and reporting
- Automatic model comparison and benchmarking
- Comprehensive evaluation reports
- Integration with ML Common utilities
- Support for multiple evaluation types
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np

# Import new simplified infrastructure
from .simplified_pipeline_infrastructure import (
    create_simple_step_function,
    create_data_processing_step_function
)

# Import standardized validation
from .standardized_config_validation import (
    validate_config,
    validate_and_fix_config
)

# Import unified data quality
from .unified_data_quality import (
    validate_data_quality,
    clean_data,
    generate_quality_report
)

# Import ML Common utilities
from src.utils.ml_common import (
    ModelEvaluationUtilities,
    DataQualityUtilities,
    MLTrainingSafeguards,
    FeatureSelectionFramework
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class UnifiedModelEvaluationManager:
    """
    Unified model evaluation manager for all training steps.
    
    This replaces custom model evaluation logic with a unified
    approach using ModelEvaluationUtilities from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize unified model evaluation manager."""
        self.config = validate_and_fix_config(config, 'model_evaluation')
        self.logger = logger.getChild('UnifiedModelEvaluationManager')
        
        # Initialize ML Common utilities
        self.model_evaluator = ModelEvaluationUtilities(self.config.get('evaluation_config', {}))
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Model evaluation configuration
        self.evaluation_config = self.config.get('evaluation_config', {})
        
        # Standard model evaluation settings
        self.standard_settings = {
            'evaluation_type': 'comprehensive',
            'enable_cross_validation': True,
            'enable_time_series_validation': True,
            'enable_confidence_intervals': True,
            'enable_model_comparison': True,
            'enable_feature_importance_analysis': True,
            'enable_prediction_analysis': True,
            'cv_folds': 5,
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'confidence_level': 0.95,
            'enable_statistical_tests': True,
            'enable_visualization': True,
            'save_evaluation_reports': True,
            'evaluation_report_path': 'evaluation_reports'
        }
        
        # Update with user configuration
        self.standard_settings.update(self.evaluation_config)
        
        self.logger.info("🚀 Unified Model Evaluation Manager initialized")
    
    async def evaluate_model(self, model: Any, features: pd.DataFrame, targets: pd.Series, 
                           evaluation_type: str = 'comprehensive', model_name: str = 'default_model') -> Dict[str, Any]:
        """
        Evaluate model using unified approach.
        
        Args:
            model: Trained model to evaluate
            features: Evaluation features
            targets: Evaluation targets
            evaluation_type: Type of evaluation ('basic', 'standard', 'comprehensive')
            model_name: Name of the model
            
        Returns:
            Model evaluation result
        """
        try:
            self.logger.info(f"📊 Starting {evaluation_type} model evaluation for '{model_name}'...")
            
            # Validate input data
            features_validation = validate_data_quality(features, 'features', 'comprehensive')
            targets_validation = validate_data_quality(targets, 'targets', 'standard')
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation['errors']}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation['errors']}")
            
            # Prepare data for evaluation
            X_eval, y_eval = await self._prepare_evaluation_data(features, targets)
            
            # Evaluate model based on type
            if evaluation_type == 'basic':
                evaluation_result = await self._evaluate_basic_model(model, X_eval, y_eval, model_name)
            elif evaluation_type == 'standard':
                evaluation_result = await self._evaluate_standard_model(model, X_eval, y_eval, model_name)
            elif evaluation_type == 'comprehensive':
                evaluation_result = await self._evaluate_comprehensive_model(model, X_eval, y_eval, model_name)
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
            
            # Generate evaluation metadata
            evaluation_metadata = self._generate_evaluation_metadata(
                model, features, targets, evaluation_result, evaluation_type, model_name
            )
            
            # Generate evaluation report
            evaluation_report = self._generate_evaluation_report(evaluation_result, evaluation_metadata)
            
            return {
                'evaluation_metrics': evaluation_result.get('evaluation_metrics', {}),
                'model_performance': evaluation_result.get('model_performance', {}),
                'feature_importance': evaluation_result.get('feature_importance', {}),
                'prediction_analysis': evaluation_result.get('prediction_analysis', {}),
                'evaluation_metadata': evaluation_metadata,
                'evaluation_report': evaluation_report,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'evaluation_type': evaluation_type,
                'model_name': model_name
            }
            
        except Exception as e:
            self.logger.exception(f"Error evaluating model: {e}")
            raise
    
    async def _prepare_evaluation_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for evaluation."""
        try:
            # Convert to numpy arrays
            X = features.values if hasattr(features, 'values') else features
            y = targets.values if hasattr(targets, 'values') else targets
            
            self.logger.info(f"Evaluation data prepared: Features {X.shape}, Targets {y.shape}")
            
            return X, y
            
        except Exception as e:
            self.logger.exception(f"Error preparing evaluation data: {e}")
            raise
    
    async def _evaluate_basic_model(self, model: Any, X_eval: np.ndarray, y_eval: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate model with basic metrics."""
        try:
            self.logger.info("Evaluating model with basic metrics...")
            
            # Make predictions
            y_pred = model.predict(X_eval)
            
            # Calculate basic metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_eval, y_pred)
            precision = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
            
            return {
                'evaluation_metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'model_performance': {
                    'predictions': y_pred,
                    'actual': y_eval
                }
            }
            
        except Exception as e:
            self.logger.exception(f"Error in basic model evaluation: {e}")
            raise
    
    async def _evaluate_standard_model(self, model: Any, X_eval: np.ndarray, y_eval: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate model with standard metrics and cross-validation."""
        try:
            self.logger.info("Evaluating model with standard metrics...")
            
            # Use ModelEvaluationUtilities for standard evaluation
            evaluation_result = self.model_evaluator.evaluate_model(
                model=model,
                X=X_eval,
                y=y_eval,
                model_name=model_name,
                enable_cross_validation=self.standard_settings.get('enable_cross_validation', True),
                cv_folds=self.standard_settings.get('cv_folds', 5)
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.exception(f"Error in standard model evaluation: {e}")
            raise
    
    async def _evaluate_comprehensive_model(self, model: Any, X_eval: np.ndarray, y_eval: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Evaluate model with comprehensive metrics and analysis."""
        try:
            self.logger.info("Evaluating model with comprehensive metrics...")
            
            # Use ModelEvaluationUtilities for comprehensive evaluation
            evaluation_result = self.model_evaluator.evaluate_model(
                model=model,
                X=X_eval,
                y=y_eval,
                model_name=model_name,
                enable_cross_validation=self.standard_settings.get('enable_cross_validation', True),
                enable_time_series_validation=self.standard_settings.get('enable_time_series_validation', True),
                enable_confidence_intervals=self.standard_settings.get('enable_confidence_intervals', True),
                enable_feature_importance_analysis=self.standard_settings.get('enable_feature_importance_analysis', True),
                enable_prediction_analysis=self.standard_settings.get('enable_prediction_analysis', True),
                cv_folds=self.standard_settings.get('cv_folds', 5),
                confidence_level=self.standard_settings.get('confidence_level', 0.95)
            )
            
            return evaluation_result
            
        except Exception as e:
            self.logger.exception(f"Error in comprehensive model evaluation: {e}")
            raise
    
    def _generate_evaluation_metadata(self, model: Any, features: pd.DataFrame, targets: pd.Series, 
                                    evaluation_result: Dict[str, Any], evaluation_type: str, model_name: str) -> Dict[str, Any]:
        """Generate metadata about model evaluation."""
        try:
            metadata = {
                'evaluation_type': evaluation_type,
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'features_shape': features.shape,
                    'targets_shape': targets.shape,
                    'feature_names': list(features.columns) if hasattr(features, 'columns') else None
                },
                'evaluation_settings': self.standard_settings,
                'model_info': {
                    'model_type': type(model).__name__,
                    'model_params': getattr(model, 'get_params', lambda: {})()
                }
            }
            
            # Add evaluation metrics summary
            if 'evaluation_metrics' in evaluation_result:
                eval_metrics = evaluation_result['evaluation_metrics']
                metadata['evaluation_summary'] = {
                    'primary_metric': eval_metrics.get('accuracy', eval_metrics.get('test_accuracy', 0)),
                    'metrics_count': len(eval_metrics),
                    'available_metrics': list(eval_metrics.keys())
                }
            
            # Add model performance summary
            if 'model_performance' in evaluation_result:
                performance = evaluation_result['model_performance']
                metadata['performance_summary'] = {
                    'performance_indicators': list(performance.keys()),
                    'has_predictions': 'predictions' in performance,
                    'has_actual': 'actual' in performance
                }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating evaluation metadata: {e}")
            return {'error': str(e)}
    
    def _generate_evaluation_report(self, evaluation_result: Dict[str, Any], evaluation_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        try:
            report = {
                'report_timestamp': datetime.now().isoformat(),
                'model_name': evaluation_metadata.get('model_name', 'unknown'),
                'evaluation_type': evaluation_metadata.get('evaluation_type', 'unknown'),
                'executive_summary': self._generate_executive_summary(evaluation_result),
                'detailed_metrics': evaluation_result.get('evaluation_metrics', {}),
                'model_performance': evaluation_result.get('model_performance', {}),
                'feature_analysis': evaluation_result.get('feature_importance', {}),
                'recommendations': self._generate_recommendations(evaluation_result),
                'metadata': evaluation_metadata
            }
            
            return report
            
        except Exception as e:
            self.logger.warning(f"Error generating evaluation report: {e}")
            return {'error': str(e)}
    
    def _generate_executive_summary(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of evaluation results."""
        try:
            eval_metrics = evaluation_result.get('evaluation_metrics', {})
            
            # Extract key metrics
            accuracy = eval_metrics.get('accuracy', eval_metrics.get('test_accuracy', 0))
            precision = eval_metrics.get('precision', eval_metrics.get('test_precision', 0))
            recall = eval_metrics.get('recall', eval_metrics.get('test_recall', 0))
            f1 = eval_metrics.get('f1_score', eval_metrics.get('test_f1', 0))
            
            # Determine performance level
            if accuracy >= 0.9:
                performance_level = "Excellent"
            elif accuracy >= 0.8:
                performance_level = "Good"
            elif accuracy >= 0.7:
                performance_level = "Fair"
            else:
                performance_level = "Poor"
            
            return {
                'performance_level': performance_level,
                'key_metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'overall_assessment': f"Model shows {performance_level.lower()} performance with {accuracy:.3f} accuracy"
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating executive summary: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, evaluation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        try:
            recommendations = []
            
            eval_metrics = evaluation_result.get('evaluation_metrics', {})
            accuracy = eval_metrics.get('accuracy', eval_metrics.get('test_accuracy', 0))
            
            # Performance-based recommendations
            if accuracy < 0.7:
                recommendations.append("❌ Model performance is below acceptable threshold - consider retraining with more data or different features")
            elif accuracy < 0.8:
                recommendations.append("⚠️ Model performance is acceptable but could be improved - consider feature engineering or hyperparameter tuning")
            else:
                recommendations.append("✅ Model performance is good - consider deploying for production use")
            
            # Feature importance recommendations
            if 'feature_importance' in evaluation_result and evaluation_result['feature_importance']:
                importance = evaluation_result['feature_importance']
                if isinstance(importance, dict):
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    recommendations.append(f"🔍 Top performing features: {[f[0] for f in top_features]}")
            
            # Cross-validation recommendations
            if 'cross_validation_scores' in eval_metrics:
                cv_scores = eval_metrics['cross_validation_scores']
                if isinstance(cv_scores, (list, np.ndarray)):
                    cv_std = np.std(cv_scores)
                    if cv_std > 0.05:
                        recommendations.append("⚠️ High variance in cross-validation scores - model may be unstable")
                    else:
                        recommendations.append("✅ Low variance in cross-validation scores - model is stable")
            
            if not recommendations:
                recommendations.append("✅ No specific recommendations - model evaluation completed successfully")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
            return [f"Error generating recommendations: {e}"]
    
    def get_model_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of model evaluation capabilities."""
        return {
            'config': self.config,
            'standard_settings': self.standard_settings,
            'model_evaluator_info': {
                'evaluator_type': 'ModelEvaluationUtilities',
                'available_features': [
                    'cross_validation',
                    'time_series_validation',
                    'confidence_intervals',
                    'model_comparison',
                    'feature_importance_analysis',
                    'prediction_analysis',
                    'statistical_tests',
                    'visualization'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# Simplified model evaluation step functions
async def unified_model_evaluation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified model evaluation logic using ModelEvaluationUtilities.
    
    Args:
        config: Configuration dictionary
        pipeline_state: Current pipeline state
        
    Returns:
        Model evaluation result
    """
    logger.info("📊 Starting unified model evaluation...")
    
    try:
        # Get model, features, and targets from pipeline state
        model = pipeline_state.get('model')
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if model is None:
            raise ValueError("No model found in pipeline state for evaluation")
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model evaluation")
        
        # Initialize unified model evaluation manager
        evaluation_manager = UnifiedModelEvaluationManager(config)
        
        # Determine evaluation type and model name from configuration
        evaluation_type = config.get('evaluation_type', 'comprehensive')
        model_name = config.get('model_name', 'unified_model')
        
        # Evaluate model
        result = await evaluation_manager.evaluate_model(model, features, targets, evaluation_type, model_name)
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in unified model evaluation: {e}")
        raise


async def basic_model_evaluation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Basic model evaluation logic (simple metrics only)."""
    logger.info("📊 Starting basic model evaluation...")
    
    try:
        # Get model, features, and targets from pipeline state
        model = pipeline_state.get('model')
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if model is None:
            raise ValueError("No model found in pipeline state for evaluation")
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model evaluation")
        
        # Initialize unified model evaluation manager
        evaluation_manager = UnifiedModelEvaluationManager(config)
        
        # Evaluate model
        result = await evaluation_manager.evaluate_model(model, features, targets, 'basic', 'basic_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in basic model evaluation: {e}")
        raise


async def standard_model_evaluation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Standard model evaluation logic (cross-validation and standard metrics)."""
    logger.info("📊 Starting standard model evaluation...")
    
    try:
        # Get model, features, and targets from pipeline state
        model = pipeline_state.get('model')
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if model is None:
            raise ValueError("No model found in pipeline state for evaluation")
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model evaluation")
        
        # Initialize unified model evaluation manager
        evaluation_manager = UnifiedModelEvaluationManager(config)
        
        # Evaluate model
        result = await evaluation_manager.evaluate_model(model, features, targets, 'standard', 'standard_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in standard model evaluation: {e}")
        raise


async def comprehensive_model_evaluation_logic(config: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive model evaluation logic (all evaluation features)."""
    logger.info("📊 Starting comprehensive model evaluation...")
    
    try:
        # Get model, features, and targets from pipeline state
        model = pipeline_state.get('model')
        features = pipeline_state.get('features') or pipeline_state.get('selected_features')
        targets = pipeline_state.get('labels') or pipeline_state.get('targets')
        
        if model is None:
            raise ValueError("No model found in pipeline state for evaluation")
        
        if features is None or targets is None:
            raise ValueError("Missing features or targets in pipeline state for model evaluation")
        
        # Initialize unified model evaluation manager
        evaluation_manager = UnifiedModelEvaluationManager(config)
        
        # Evaluate model
        result = await evaluation_manager.evaluate_model(model, features, targets, 'comprehensive', 'comprehensive_model')
        
        return result
        
    except Exception as e:
        logger.exception(f"Error in comprehensive model evaluation: {e}")
        raise


# Create step functions
unified_model_evaluation = create_simple_step_function("unified_model_evaluation", unified_model_evaluation_logic)
basic_model_evaluation = create_simple_step_function("basic_model_evaluation", basic_model_evaluation_logic)
standard_model_evaluation = create_simple_step_function("standard_model_evaluation", standard_model_evaluation_logic)
comprehensive_model_evaluation = create_simple_step_function("comprehensive_model_evaluation", comprehensive_model_evaluation_logic)


class SimplifiedModelEvaluation:
    """
    Simplified model evaluation using unified infrastructure.
    
    This replaces custom model evaluation logic with a unified
    approach using ModelEvaluationUtilities from ml_common.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simplified model evaluation."""
        self.config = validate_and_fix_config(config, 'model_evaluation')
        self.logger = logger.getChild('SimplifiedModelEvaluation')
        
        # Initialize unified model evaluation manager
        self.evaluation_manager = UnifiedModelEvaluationManager(self.config)
        
        self.logger.info("🚀 Simplified Model Evaluation initialized")
    
    async def evaluate_model(self, model: Any, features: pd.DataFrame, targets: pd.Series, 
                           evaluation_type: str = 'comprehensive', model_name: str = 'default_model') -> Dict[str, Any]:
        """
        Evaluate model using unified approach.
        
        Args:
            model: Trained model to evaluate
            features: Evaluation features
            targets: Evaluation targets
            evaluation_type: Type of evaluation
            model_name: Name of the model
            
        Returns:
            Model evaluation result
        """
        try:
            self.logger.info(f"🚀 Evaluating {evaluation_type} model '{model_name}'...")
            
            # Evaluate model
            result = await self.evaluation_manager.evaluate_model(model, features, targets, evaluation_type, model_name)
            
            self.logger.info(f"✅ Model evaluation completed: {result['evaluation_metadata']['model_name']}")
            
            return result
            
        except Exception as e:
            self.logger.exception(f"Model evaluation error: {e}")
            raise
    
    def get_model_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of model evaluation capabilities."""
        return self.evaluation_manager.get_model_evaluation_summary()


# Example usage and testing
async def example_model_evaluation():
    """Example of using the unified model evaluation."""
    
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 50
    
    # Create features with some signal
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Add some signal to first 10 features
    for i in range(10):
        features[f'feature_{i}'] += np.random.randn(n_samples) * 0.5
    
    # Create targets based on first 10 features
    targets = pd.Series(
        (features.iloc[:, :10].sum(axis=1) > 0).astype(int),
        name='target'
    )
    
    # Train a simple model for evaluation
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, targets)
    
    # Configuration for different evaluation types
    configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'evaluation_type': 'basic',
            'model_name': 'basic_model',
            'evaluation_config': {
                'enable_cross_validation': False,
                'enable_confidence_intervals': False,
                'enable_feature_importance_analysis': False
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'evaluation_type': 'standard',
            'model_name': 'standard_model',
            'evaluation_config': {
                'enable_cross_validation': True,
                'enable_confidence_intervals': True,
                'enable_feature_importance_analysis': True,
                'cv_folds': 5
            }
        },
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m',
            'evaluation_type': 'comprehensive',
            'model_name': 'comprehensive_model',
            'evaluation_config': {
                'enable_cross_validation': True,
                'enable_time_series_validation': True,
                'enable_confidence_intervals': True,
                'enable_model_comparison': True,
                'enable_feature_importance_analysis': True,
                'enable_prediction_analysis': True,
                'cv_folds': 5,
                'confidence_level': 0.95
            }
        }
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Testing Model Evaluation Type {i+1}: {config['evaluation_type']} ===")
        
        # Create simplified model evaluation
        model_evaluator = SimplifiedModelEvaluation(config)
        
        # Evaluate model
        result = await model_evaluator.evaluate_model(model, features, targets, config['evaluation_type'], config['model_name'])
        
        # Get summary
        summary = model_evaluator.get_model_evaluation_summary()
        
        print(f"Evaluation type: {result['evaluation_type']}")
        print(f"Model name: {result['model_name']}")
        print(f"Evaluation timestamp: {result['evaluation_metadata']['evaluation_timestamp']}")
        
        # Show evaluation metrics
        if 'evaluation_metrics' in result:
            eval_metrics = result['evaluation_metrics']
            print(f"Evaluation metrics: {list(eval_metrics.keys())}")
            if 'accuracy' in eval_metrics:
                print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
        
        # Show executive summary
        if 'evaluation_report' in result and 'executive_summary' in result['evaluation_report']:
            exec_summary = result['evaluation_report']['executive_summary']
            print(f"Performance level: {exec_summary.get('performance_level', 'unknown')}")
            print(f"Overall assessment: {exec_summary.get('overall_assessment', 'unknown')}")
        
        # Show recommendations
        if 'evaluation_report' in result and 'recommendations' in result['evaluation_report']:
            recommendations = result['evaluation_report']['recommendations']
            print(f"Recommendations: {recommendations[:2]}")  # Show first 2 recommendations
        
        results.append((result, summary))
    
    return results


# Main execution
async def main():
    """Main execution function."""
    try:
        results = await example_model_evaluation()
        print("✅ Model evaluation example completed successfully")
        return results
    except Exception as e:
        logger.exception(f"Model evaluation example failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())