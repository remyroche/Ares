"""
Model Evaluation Utilities

This module provides comprehensive model evaluation utilities extracted from training steps
to eliminate code duplication and provide consistent model evaluation across all steps.

Key Features:
- Evaluation metric calculation and aggregation
- Cross-validation and time series validation utilities
- Confidence metrics and calibration assessment
- Model explanation generation
- Integration with ML Common utilities
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import numpy as np

# Import ML Common utilities
from src.utils.ml_common import (
    ModelEvaluationUtilities,
    DataQualityUtilities,
    MLTrainingSafeguards
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class ModelEvaluationUtilities:
    """
    Model evaluation utilities for all training steps.
    
    This provides common model evaluation patterns and utilities
    extracted from multiple training step implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model evaluation utilities."""
        self.config = config or {}
        self.logger = logger.getChild('ModelEvaluationUtilities')
        
        # Initialize ML Common utilities
        self.model_evaluator = ModelEvaluationUtilities(self.config.get('evaluation_config', {}))
        self.data_quality = DataQualityUtilities()
        self.safeguards = MLTrainingSafeguards()
        
        # Model evaluation configuration
        self.evaluation_config = self.config.get('evaluation_config', {})
        
        # Standard model evaluation settings
        self.standard_settings = {
            'enable_cross_validation': True,
            'enable_time_series_validation': True,
            'enable_confidence_intervals': True,
            'enable_model_comparison': True,
            'enable_feature_importance_analysis': True,
            'enable_prediction_analysis': True,
            'enable_calibration_assessment': True,
            'enable_confidence_metrics': True,
            'cv_folds': 5,
            'confidence_level': 0.95,
            'test_size': 0.2,
            'validation_size': 0.2,
            'random_state': 42,
            'enable_early_stopping': True,
            'early_stopping_patience': 10
        }
        
        # Update with user configuration
        self.standard_settings.update(self.evaluation_config)
        
        self.logger.info("🚀 Model Evaluation Utilities initialized")
    
    def evaluate_basic_model(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model using basic approach (accuracy only).
        
        Args:
            model: Trained model to evaluate
            features: Test features
            targets: Test targets
            
        Returns:
            Evaluation result dictionary
        """
        try:
            self.logger.info("Evaluating model using basic approach...")
            
            # Make predictions
            predictions = model.predict(features)
            
            # Calculate basic metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(targets, predictions)
            precision = precision_score(targets, predictions, average='weighted', zero_division=0)
            recall = recall_score(targets, predictions, average='weighted', zero_division=0)
            f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            return {
                'evaluation_metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'predictions': predictions
            }
            
        except Exception as e:
            self.logger.exception(f"Error in basic model evaluation: {e}")
            raise
    
    def evaluate_standard_model(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model using standard approach (basic metrics + cross-validation).
        
        Args:
            model: Trained model to evaluate
            features: Test features
            targets: Test targets
            
        Returns:
            Evaluation result dictionary
        """
        try:
            self.logger.info("Evaluating model using standard approach...")
            
            # Start with basic evaluation
            basic_result = self.evaluate_basic_model(model, features, targets)
            
            # Add cross-validation if enabled
            if self.standard_settings.get('enable_cross_validation', True):
                cv_metrics = self._perform_cross_validation(model, features, targets)
                basic_result['evaluation_metrics'].update(cv_metrics)
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features.columns, model.feature_importances_))
                basic_result['feature_importance'] = feature_importance
            
            return basic_result
            
        except Exception as e:
            self.logger.exception(f"Error in standard model evaluation: {e}")
            raise
    
    def evaluate_comprehensive_model(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model using comprehensive approach (all features enabled).
        
        Args:
            model: Trained model to evaluate
            features: Test features
            targets: Test targets
            
        Returns:
            Evaluation result dictionary
        """
        try:
            self.logger.info("Evaluating model using comprehensive approach...")
            
            # Start with standard evaluation
            standard_result = self.evaluate_standard_model(model, features, targets)
            
            # Add confidence metrics if enabled
            if self.standard_settings.get('enable_confidence_metrics', True):
                confidence_metrics = self._calculate_confidence_metrics(model, features, targets)
                standard_result['confidence_metrics'] = confidence_metrics
            
            # Add calibration assessment if enabled
            if self.standard_settings.get('enable_calibration_assessment', True):
                calibration_metrics = self._assess_calibration(model, features, targets)
                standard_result['calibration_metrics'] = calibration_metrics
            
            # Add model explanations if enabled
            if self.standard_settings.get('enable_model_explanations', True):
                model_explanations = self._generate_model_explanations(model, features, targets)
                standard_result['model_explanations'] = model_explanations
            
            # Add time series validation if enabled
            if self.standard_settings.get('enable_time_series_validation', True):
                ts_metrics = self._perform_time_series_validation(model, features, targets)
                standard_result['time_series_metrics'] = ts_metrics
            
            return standard_result
            
        except Exception as e:
            self.logger.exception(f"Error in comprehensive model evaluation: {e}")
            raise
    
    def _perform_cross_validation(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import make_scorer, accuracy_score, f1_score
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, features, targets, 
                cv=self.standard_settings.get('cv_folds', 5),
                scoring='accuracy'
            )
            
            f1_scores = cross_val_score(
                model, features, targets,
                cv=self.standard_settings.get('cv_folds', 5),
                scoring='f1_weighted'
            )
            
            return {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'cv_f1_mean': f1_scores.mean(),
                'cv_f1_std': f1_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-validation error: {e}")
            return {}
    
    def _calculate_confidence_metrics(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Calculate confidence metrics."""
        try:
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
                max_probabilities = np.max(probabilities, axis=1)
                
                return {
                    'mean_confidence': np.mean(max_probabilities),
                    'std_confidence': np.std(max_probabilities),
                    'min_confidence': np.min(max_probabilities),
                    'max_confidence': np.max(max_probabilities),
                    'confidence_distribution': {
                        'high_confidence': np.sum(max_probabilities > 0.8) / len(max_probabilities),
                        'medium_confidence': np.sum((max_probabilities > 0.6) & (max_probabilities <= 0.8)) / len(max_probabilities),
                        'low_confidence': np.sum(max_probabilities <= 0.6) / len(max_probabilities)
                    }
                }
            else:
                return {'message': 'Model does not support probability predictions'}
                
        except Exception as e:
            self.logger.warning(f"Confidence metrics calculation error: {e}")
            return {'error': str(e)}
    
    def _assess_calibration(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Assess model calibration."""
        try:
            if hasattr(model, 'predict_proba'):
                from sklearn.calibration import calibration_curve
                
                probabilities = model.predict_proba(features)[:, 1]  # Assuming binary classification
                
                # Calculate calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    targets, probabilities, n_bins=10
                )
                
                # Calculate calibration error
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                
                return {
                    'calibration_error': calibration_error,
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist(),
                    'is_well_calibrated': calibration_error < 0.1
                }
            else:
                return {'message': 'Model does not support probability predictions for calibration assessment'}
                
        except Exception as e:
            self.logger.warning(f"Calibration assessment error: {e}")
            return {'error': str(e)}
    
    def _generate_model_explanations(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Generate model explanations."""
        try:
            explanations = {}
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features.columns, model.feature_importances_))
                explanations['feature_importance'] = feature_importance
                
                # Top features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                explanations['top_features'] = sorted_features[:10]
            
            # Model complexity
            if hasattr(model, 'n_estimators'):
                explanations['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                explanations['max_depth'] = model.max_depth
            
            return explanations
            
        except Exception as e:
            self.logger.warning(f"Model explanations generation error: {e}")
            return {'error': str(e)}
    
    def _perform_time_series_validation(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Perform time series validation."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            
            # Use time series split
            tscv = TimeSeriesSplit(n_splits=self.standard_settings.get('cv_folds', 5))
            
            scores = []
            for train_idx, test_idx in tscv.split(features):
                X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                y_train, y_test = targets.iloc[train_idx], targets.iloc[test_idx]
                
                # Train model on training set
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train, y_train)
                
                # Evaluate on test set
                score = model_copy.score(X_test, y_test)
                scores.append(score)
            
            return {
                'ts_cv_scores': scores,
                'ts_cv_mean': np.mean(scores),
                'ts_cv_std': np.std(scores)
            }
            
        except Exception as e:
            self.logger.warning(f"Time series validation error: {e}")
            return {'error': str(e)}
    
    def generate_evaluation_metadata(self, model: Any, features: pd.DataFrame, targets: pd.Series, 
                                   evaluation_result: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
        """
        Generate metadata about model evaluation.
        
        Args:
            model: Trained model
            features: Test features
            targets: Test targets
            evaluation_result: Evaluation result dictionary
            evaluation_type: Type of evaluation
            
        Returns:
            Evaluation metadata dictionary
        """
        try:
            metadata = {
                'evaluation_type': evaluation_type,
                'model_type': type(model).__name__,
                'evaluation_timestamp': datetime.now().isoformat(),
                'data_shape': {
                    'features_shape': features.shape,
                    'targets_shape': targets.shape,
                    'feature_names': list(features.columns) if hasattr(features, 'columns') else None
                },
                'evaluation_settings': self.standard_settings,
                'model_params': getattr(model, 'get_params', lambda: {})()
            }
            
            # Add evaluation metrics summary
            if 'evaluation_metrics' in evaluation_result:
                eval_metrics = evaluation_result['evaluation_metrics']
                metadata['evaluation_summary'] = {
                    'primary_metric': eval_metrics.get('accuracy', eval_metrics.get('cv_accuracy_mean', 0)),
                    'metrics_count': len(eval_metrics),
                    'available_metrics': list(eval_metrics.keys())
                }
            
            # Add feature importance summary
            if 'feature_importance' in evaluation_result and evaluation_result['feature_importance']:
                importance_dict = evaluation_result['feature_importance']
                if isinstance(importance_dict, dict):
                    metadata['feature_importance_summary'] = {
                        'top_features': sorted(
                            importance_dict.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10],
                        'total_features': len(importance_dict),
                        'max_importance': max(importance_dict.values()) if importance_dict else 0,
                        'min_importance': min(importance_dict.values()) if importance_dict else 0
                    }
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Error generating evaluation metadata: {e}")
            return {'error': str(e)}
    
    def evaluate_model(self, model: Any, features: pd.DataFrame, targets: pd.Series, 
                      evaluation_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Evaluate model using unified approach.
        
        Args:
            model: Trained model to evaluate
            features: Test features
            targets: Test targets
            evaluation_type: Type of evaluation ('basic', 'standard', 'comprehensive')
            
        Returns:
            Model evaluation result
        """
        try:
            self.logger.info(f"📊 Evaluating model using {evaluation_type} approach...")
            
            # Validate input data
            features_validation = self.data_quality.analyze_data_quality(features)
            targets_validation = self.data_quality.analyze_data_quality(targets)
            
            if not features_validation['passed']:
                self.logger.warning(f"Features validation issues: {features_validation.get('warnings', [])}")
            
            if not targets_validation['passed']:
                self.logger.warning(f"Targets validation issues: {targets_validation.get('warnings', [])}")
            
            # Evaluate model based on type
            if evaluation_type == 'basic':
                evaluation_result = self.evaluate_basic_model(model, features, targets)
            elif evaluation_type == 'standard':
                evaluation_result = self.evaluate_standard_model(model, features, targets)
            elif evaluation_type == 'comprehensive':
                evaluation_result = self.evaluate_comprehensive_model(model, features, targets)
            else:
                raise ValueError(f"Unknown evaluation type: {evaluation_type}")
            
            # Generate evaluation metadata
            evaluation_metadata = self.generate_evaluation_metadata(
                model, features, targets, evaluation_result, evaluation_type
            )
            
            return {
                'evaluation_metrics': evaluation_result.get('evaluation_metrics', {}),
                'confidence_metrics': evaluation_result.get('confidence_metrics', {}),
                'feature_importance': evaluation_result.get('feature_importance', {}),
                'model_explanations': evaluation_result.get('model_explanations', {}),
                'evaluation_metadata': evaluation_metadata,
                'features_validation': features_validation,
                'targets_validation': targets_validation,
                'evaluation_type': evaluation_type
            }
            
        except Exception as e:
            self.logger.exception(f"Error evaluating model: {e}")
            raise
    
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
                    'calibration_assessment',
                    'confidence_metrics'
                ]
            },
            'timestamp': datetime.now().isoformat()
        }


# Global instance for easy access
_global_evaluation_utilities = None

def get_model_evaluation_utilities(config: Optional[Dict[str, Any]] = None) -> ModelEvaluationUtilities:
    """Get model evaluation utilities instance."""
    return ModelEvaluationUtilities(config)


# Convenience functions
def evaluate_model(model: Any, features: pd.DataFrame, targets: pd.Series, 
                  evaluation_type: str = 'comprehensive') -> Dict[str, Any]:
    """Evaluate model using utilities."""
    utils = get_model_evaluation_utilities()
    return utils.evaluate_model(model, features, targets, evaluation_type)


def generate_evaluation_metadata(model: Any, features: pd.DataFrame, targets: pd.Series, 
                               evaluation_result: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
    """Generate evaluation metadata using utilities."""
    utils = get_model_evaluation_utilities()
    return utils.generate_evaluation_metadata(model, features, targets, evaluation_result, evaluation_type)


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
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
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, targets)
    
    # Test model evaluation utilities
    utils = ModelEvaluationUtilities()
    
    print("=== Basic Model Evaluation ===")
    basic_result = utils.evaluate_model(model, features, targets, 'basic')
    print(f"Basic evaluation metrics: {list(basic_result['evaluation_metrics'].keys())}")
    print(f"Accuracy: {basic_result['evaluation_metrics']['accuracy']:.3f}")
    
    print("\n=== Standard Model Evaluation ===")
    standard_result = utils.evaluate_model(model, features, targets, 'standard')
    print(f"Standard evaluation metrics: {list(standard_result['evaluation_metrics'].keys())}")
    print(f"CV Accuracy: {standard_result['evaluation_metrics'].get('cv_accuracy_mean', 0):.3f} ± {standard_result['evaluation_metrics'].get('cv_accuracy_std', 0):.3f}")
    
    print("\n=== Comprehensive Model Evaluation ===")
    comprehensive_result = utils.evaluate_model(model, features, targets, 'comprehensive')
    print(f"Comprehensive evaluation metrics: {list(comprehensive_result['evaluation_metrics'].keys())}")
    print(f"Confidence metrics: {list(comprehensive_result.get('confidence_metrics', {}).keys())}")
    print(f"Model explanations: {list(comprehensive_result.get('model_explanations', {}).keys())}")
    
    print("\n=== Evaluation Metadata ===")
    metadata = utils.generate_evaluation_metadata(model, features, targets, comprehensive_result, 'comprehensive')
    print(f"Evaluation type: {metadata['evaluation_type']}")
    print(f"Model type: {metadata['model_type']}")
    print(f"Evaluation timestamp: {metadata['evaluation_timestamp']}")
    print(f"Data shape: {metadata['data_shape']['features_shape']}")
    
    print("\n=== Evaluation Summary ===")
    summary = utils.get_model_evaluation_summary()
    print(f"Available features: {summary['model_evaluator_info']['available_features']}")
    print(f"Standard settings: {list(summary['standard_settings'].keys())}")