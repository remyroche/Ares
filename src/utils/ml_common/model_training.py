"""
Comprehensive Model Training Utilities with Automatic Confidence Metrics

This module provides enhanced model training utilities that automatically include
comprehensive evaluation metrics, confidence analysis, and calibration assessment.

Key Features:
- Automatic confidence metrics calculation
- Comprehensive model evaluation
- Calibration assessment
- Feature importance analysis
- Model comparison utilities
- Training progress monitoring

Built on existing utilities:
- Uses model_evaluation.py for comprehensive evaluation
- Integrates with confidence_metrics.py for prediction confidence
- Leverages common_operations.py for robust error handling
- Builds on existing training patterns
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import logging
from datetime import datetime

from .model_evaluation import ModelEvaluator
from .confidence_metrics import calculate_confidence_metrics, log_confidence_metrics
from .model_explanations import explain_model_with_shap_lime
from .hpo_utils import HPOptimizer
from ..common_operations import create_fallback_logger

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ModelTraining")
    print("✅ Custom logger available for MLCommon.ModelTraining")
except Exception as e:
    print(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ModelTraining")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        matthews_corrcoef, cohen_kappa_score, log_loss
    )
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited training functionality")


class EnhancedModelTrainer:
    """Enhanced model trainer with automatic confidence metrics and comprehensive evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced model trainer.
        
        Args:
            config: Configuration dictionary for training parameters
        """
        self.config = config or {}
        self.logger = create_fallback_logger(__name__)
        
        _LOGGER.info("🚀 Initializing EnhancedModelTrainer...")
        
        # Initialize model evaluator
        self.evaluator = ModelEvaluator(self.config.get('evaluation', {}))
        
        # Training configuration
        self.enable_confidence_metrics = self.config.get('enable_confidence_metrics', True)
        self.enable_calibration_assessment = self.config.get('enable_calibration_assessment', True)
        self.enable_feature_importance = self.config.get('enable_feature_importance', True)
        self.enable_cross_validation = self.config.get('enable_cross_validation', True)
        self.enable_model_explanations = self.config.get('enable_model_explanations', True)
        self.enable_post_training_hpo = self.config.get('enable_post_training_hpo', True)
        self.cv_folds = self.config.get('cv_folds', 5)
        
        _LOGGER.info(f"⚙️ Configuration - Confidence metrics: {self.enable_confidence_metrics}")
        _LOGGER.info(f"⚙️ Configuration - Calibration assessment: {self.enable_calibration_assessment}")
        _LOGGER.info(f"⚙️ Configuration - Feature importance: {self.enable_feature_importance}")
        _LOGGER.info(f"⚙️ Configuration - Cross validation: {self.enable_cross_validation}")
        _LOGGER.info(f"⚙️ Configuration - Model explanations: {self.enable_model_explanations}")
        _LOGGER.info(f"⚙️ Configuration - Post-training HPO: {self.enable_post_training_hpo}")
        _LOGGER.info(f"⚙️ Configuration - CV folds: {self.cv_folds}")
        
        # Initialize HPO optimizer for post-training optimization
        if self.enable_post_training_hpo:
            _LOGGER.debug("🔧 Initializing HPO optimizer for post-training optimization...")
            self.hpo_optimizer = HPOptimizer(self.config.get('hpo', {}))
        
        _LOGGER.info("✅ EnhancedModelTrainer initialized successfully")
        
    def train_and_evaluate_model(self, model: Any, model_name: str,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                enable_class_weights: bool = False,
                                class_weight_config: str = 'balanced') -> Dict[str, Any]:
        """
        Train and comprehensively evaluate a model with automatic confidence metrics.
        
        Args:
            model: The model to train
            model_name: Name of the model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            enable_class_weights: Whether to use class weights
            class_weight_config: Class weight configuration
            
        Returns:
            Comprehensive training and evaluation results
        """
        start_time = time.time()
        _LOGGER.info(f'🏃 Starting training for {model_name}...')
        _LOGGER.info(f'📊 Data shapes - Train: {X_train.shape}, Test: {X_test.shape}')
        _LOGGER.info(f'📊 Target shapes - Train: {y_train.shape}, Test: {y_test.shape}')
        _LOGGER.info(f'📊 Features: {len(feature_names) if feature_names else "Unknown"}')
        
        try:
            # Apply class weights if enabled
            sample_weight_train = None
            if enable_class_weights and hasattr(model, 'fit'):
                _LOGGER.debug(f'⚖️ Computing class weights using {class_weight_config} strategy...')
                try:
                    from sklearn.utils.class_weight import compute_sample_weight
                    sample_weight_train = compute_sample_weight(class_weight_config, y_train)
                    _LOGGER.info(f'✅ Class weights computed for {len(sample_weight_train)} samples')
                except Exception as e:
                    _LOGGER.warning(f'⚠️ Class weight computation failed: {e}')
            
            # Train the model
            _LOGGER.info(f'🔄 Training {model_name}...')
            training_start = time.time()
            
            if sample_weight_train is not None:
                model.fit(X_train, y_train, sample_weight=sample_weight_train)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - training_start
            _LOGGER.info(f'✅ Model training completed in {training_time:.3f}s')
            
            # Make predictions
            _LOGGER.info(f'🔮 Making predictions with {model_name}...')
            prediction_start = time.time()
            
            y_pred = model.predict(X_test)
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                _LOGGER.debug(f'📊 Prediction probabilities shape: {y_pred_proba.shape}')
            
            prediction_time = time.time() - prediction_start
            _LOGGER.info(f'✅ Predictions completed in {prediction_time:.3f}s')
            
            # Calculate basic metrics
            _LOGGER.debug('📊 Calculating basic metrics...')
            basic_metrics = self._calculate_basic_metrics(y_test, y_pred, y_pred_proba)
            
            # Calculate confidence metrics if enabled
            confidence_metrics = {}
            if self.enable_confidence_metrics and y_pred_proba is not None:
                _LOGGER.debug('🎯 Calculating confidence metrics...')
                confidence_metrics = calculate_confidence_metrics(y_test, y_pred_proba)
            elif self.enable_confidence_metrics:
                _LOGGER.warning('⚠️ Confidence metrics requested but model does not support predict_proba')
            
            # Calculate feature importance if enabled
            feature_importance = None
            if self.enable_feature_importance:
                _LOGGER.debug('🔍 Extracting feature importance...')
                feature_importance = self._extract_feature_importance(model, feature_names)
            
            # Generate model explanations if enabled
            model_explanations = {}
            if self.enable_model_explanations:
                _LOGGER.debug('🧠 Generating model explanations...')
                try:
                    # Use smaller samples for explanations to avoid memory issues
                    sample_size = min(50, len(X_test))
                    test_indices = np.random.choice(len(X_test), sample_size, replace=False)
                    X_test_sample = X_test[test_indices]
                    
                    _LOGGER.debug(f'📊 Using sample size {sample_size} for explanations')
                    
                    model_explanations = explain_model_with_shap_lime(
                        model=model,
                        X_train=X_train[:100],  # Use small sample for background
                        X_test=X_test_sample,
                        feature_names=feature_names,
                        model_name=model_name,
                        config={
                            'enable_shap': True,
                            'enable_lime': True,
                            'shap_sample_size': 20,
                            'lime_sample_size': 5
                        }
                    )
                    _LOGGER.info('✅ Model explanations generated successfully')
                except Exception as e:
                    _LOGGER.warning(f'⚠️ Model explanations failed for {model_name}: {e}')
                    model_explanations = {'error': str(e)}
            
            # Perform cross-validation if enabled
            cv_results = {}
            if self.enable_cross_validation and SKLEARN_AVAILABLE:
                _LOGGER.debug(f'🔄 Performing cross-validation with {self.cv_folds} folds...')
                cv_results = self._perform_cross_validation(model, X_train, y_train)
            elif self.enable_cross_validation and not SKLEARN_AVAILABLE:
                _LOGGER.warning('⚠️ Cross-validation requested but scikit-learn not available')
            
            # Comprehensive evaluation using ModelEvaluator
            _LOGGER.debug('📊 Performing comprehensive evaluation...')
            evaluation_results = self.evaluator.comprehensive_evaluation(
                y_test, y_pred, y_pred_proba, task_type='classification'
            )
            
            # Post-training HPO if enabled and model performance is good
            post_training_hpo_results = {}
            if (self.enable_post_training_hpo and 
                hasattr(self, 'hpo_optimizer') and 
                basic_metrics.get('accuracy', 0) > 0.6):  # Only optimize if model is decent
                try:
                    _LOGGER.info(f'🔧 Running post-training HPO for {model_name}...')
                    post_training_hpo_results = self._perform_post_training_hpo(
                        model, model_name, X_train, y_train, X_test, y_test, feature_names
                    )
                except Exception as e:
                    _LOGGER.warning(f'⚠️ Post-training HPO failed for {model_name}: {e}')
                    post_training_hpo_results = {'error': str(e)}
            
            # Compile comprehensive results
            results = {
                'model_name': model_name,
                'model': model,
                'training_time': training_time,
                'basic_metrics': basic_metrics,
                'confidence_metrics': confidence_metrics,
                'model_explanations': model_explanations,
                'feature_importance': feature_importance,
                'cross_validation': cv_results,
                'comprehensive_evaluation': evaluation_results,
                'post_training_hpo': post_training_hpo_results,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'success': True
            }
            
            # Log comprehensive results
            self._log_training_results(results, model_name)
            
            total_time = time.time() - start_time
            _LOGGER.info(f'✅ Complete training and evaluation for {model_name} finished in {total_time:.3f}s')
            
            return results
            
        except Exception as e:
            total_time = time.time() - start_time
            _LOGGER.error(f'❌ Model training failed for {model_name} after {total_time:.3f}s: {e}')
            return {
                'model_name': model_name,
                'error': str(e),
                'success': False
            }
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate basic classification metrics."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn not available'}
            
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            matthews_corr = matthews_corrcoef(y_true, y_pred)
            cohen_kappa = cohen_kappa_score(y_true, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            # ROC-AUC and log loss (if probabilities available)
            roc_auc = None
            log_loss_score = None
            if y_pred_proba is not None and len(np.unique(y_true)) > 1:
                try:
                    if y_pred_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                        log_loss_score = log_loss(y_true, y_pred_proba)
                    else:
                        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                        log_loss_score = log_loss(y_true, y_pred_proba)
                except Exception:
                    pass
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'matthews_corrcoef': float(matthews_corr),
                'cohen_kappa': float(cohen_kappa),
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'log_loss': float(log_loss_score) if log_loss_score is not None else None,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }
            
        except Exception as e:
            self.logger.warning(f'Basic metrics calculation failed: {e}')
            return {'error': str(e)}
    
    def _extract_feature_importance(self, model: Any, feature_names: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        """Extract feature importance from the model."""
        try:
            if feature_names is None:
                return None
            
            # Try different methods to get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                return None
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, (feature, importance) in enumerate(zip(feature_names, importance_scores)):
                feature_importance[feature] = float(importance)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_scores': feature_importance,
                'sorted_features': sorted_features,
                'top_features': sorted_features[:10],  # Top 10 features
                'total_importance': float(np.sum(importance_scores)),
                'mean_importance': float(np.mean(importance_scores)),
                'std_importance': float(np.std(importance_scores))
            }
            
        except Exception as e:
            self.logger.warning(f'Feature importance extraction failed: {e}')
            return None
    
    def _perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation on the model."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn not available'}
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='accuracy')
            
            return {
                'scores': cv_scores.tolist(),
                'mean_score': float(np.mean(cv_scores)),
                'std_score': float(np.std(cv_scores)),
                'min_score': float(np.min(cv_scores)),
                'max_score': float(np.max(cv_scores)),
                'cv_folds': self.cv_folds
            }
            
        except Exception as e:
            self.logger.warning(f'Cross-validation failed: {e}')
            return {'error': str(e)}
    
    def _perform_post_training_hpo(self, model: Any, model_name: str,
                                  X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Perform post-training hyperparameter optimization."""
        try:
            if not hasattr(self, 'hpo_optimizer'):
                return {'error': 'HPO optimizer not available'}
            
            # Get model class and current parameters
            model_class = type(model)
            current_params = model.get_params() if hasattr(model, 'get_params') else {}
            
            # Define search space for post-training optimization
            search_space = self._get_post_training_search_space(model_name, current_params)
            
            if not search_space:
                return {'error': 'No search space defined for post-training HPO'}
            
            # Perform optimization
            hpo_results = self.hpo_optimizer.optimize(
                model_class=model_class,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv_folds=3,  # Use fewer folds for post-training
                n_trials=20,  # Use fewer trials for post-training
                scoring='accuracy'
            )
            
            if 'error' in hpo_results:
                return hpo_results
            
            # Evaluate the best model on test set
            best_model = hpo_results.get('best_model')
            if best_model:
                best_model.fit(X_train, y_train)
                y_pred_best = best_model.predict(X_test)
                best_accuracy = accuracy_score(y_test, y_pred_best)
                
                # Calculate improvement
                original_accuracy = accuracy_score(y_test, model.predict(X_test))
                improvement = best_accuracy - original_accuracy
                
                return {
                    'best_score': float(best_accuracy),
                    'original_score': float(original_accuracy),
                    'improvement': float(improvement),
                    'best_params': hpo_results.get('best_params', {}),
                    'optimization_time': hpo_results.get('optimization_time', 0),
                    'n_trials': hpo_results.get('n_trials', 0),
                    'best_model': best_model
                }
            else:
                return {'error': 'No best model found in HPO results'}
                
        except Exception as e:
            self.logger.warning(f'Post-training HPO failed: {e}')
            return {'error': str(e)}
    
    def _get_post_training_search_space(self, model_name: str, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get search space for post-training HPO based on model type."""
        try:
            # Define focused search spaces for post-training optimization
            if 'RandomForest' in model_name:
                return {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif 'LogisticRegression' in model_name:
                return {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            elif 'HistGradientBoosting' in model_name:
                return {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_iter': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'min_samples_leaf': [10, 20, 30]
                }
            else:
                # Generic search space
                return {
                    'random_state': [42]
                }
                
        except Exception as e:
            self.logger.warning(f'Failed to create search space for {model_name}: {e}')
            return {}
    
    def _log_training_results(self, results: Dict[str, Any], model_name: str) -> None:
        """Log comprehensive training results."""
        try:
            basic_metrics = results.get('basic_metrics', {})
            
            # Log basic performance
            self.logger.info(f'✅ {model_name} - '
                           f'Accuracy: {basic_metrics.get("accuracy", 0):.4f}, '
                           f'Precision: {basic_metrics.get("precision", 0):.4f}, '
                           f'Recall: {basic_metrics.get("recall", 0):.4f}, '
                           f'F1: {basic_metrics.get("f1_score", 0):.4f}, '
                           f'Time: {results.get("training_time", 0):.2f}s')
            
            # Log confidence metrics if available
            confidence_metrics = results.get('confidence_metrics', {})
            if confidence_metrics and 'error' not in confidence_metrics:
                log_confidence_metrics(confidence_metrics, model_name, self.logger)
            
            # Log cross-validation results if available
            cv_results = results.get('cross_validation', {})
            if cv_results and 'error' not in cv_results:
                self.logger.info(f'🔄 {model_name} CV: '
                               f'Mean={cv_results.get("mean_score", 0):.4f} ± '
                               f'{cv_results.get("std_score", 0):.4f}')
            
            # Log feature importance if available
            feature_importance = results.get('feature_importance')
            if feature_importance and 'top_features' in feature_importance:
                top_features = feature_importance['top_features'][:5]  # Top 5
                self.logger.info(f'🔍 {model_name} Top Features: '
                               f'{", ".join([f"{feat}({imp:.3f})" for feat, imp in top_features])}')
            
            # Log model explanations if available
            model_explanations = results.get('model_explanations', {})
            if model_explanations and 'error' not in model_explanations:
                if 'shap_explanations' in model_explanations and 'top_features' in model_explanations['shap_explanations']:
                    shap_top = model_explanations['shap_explanations']['top_features'][:3]
                    self.logger.info(f'🎯 {model_name} SHAP Top: '
                                   f'{", ".join([f"{feat}({imp:.3f})" for feat, imp in shap_top])}')
                
                if 'lime_explanations' in model_explanations and 'top_features' in model_explanations['lime_explanations']:
                    lime_top = model_explanations['lime_explanations']['top_features'][:3]
                    self.logger.info(f'🍋 {model_name} LIME Top: '
                                   f'{", ".join([f"{feat}({imp:.3f})" for feat, imp in lime_top])}')
            
            # Log post-training HPO results if available
            post_hpo = results.get('post_training_hpo', {})
            if post_hpo and 'error' not in post_hpo and 'best_score' in post_hpo:
                self.logger.info(f'🚀 {model_name} Post-HPO: '
                               f'Best Score: {post_hpo["best_score"]:.4f}, '
                               f'Improvement: {post_hpo.get("improvement", 0):.4f}')
                
        except Exception as e:
            self.logger.warning(f'Failed to log training results for {model_name}: {e}')
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple trained models and rank them.
        
        Args:
            model_results: List of model training results
            
        Returns:
            Model comparison results
        """
        try:
            if not model_results:
                return {'error': 'No model results to compare'}
            
            # Extract performance metrics for comparison
            comparison_data = []
            for result in model_results:
                if result.get('success', False) and 'basic_metrics' in result:
                    basic_metrics = result['basic_metrics']
                    comparison_data.append({
                        'model_name': result['model_name'],
                        'accuracy': basic_metrics.get('accuracy', 0),
                        'precision': basic_metrics.get('precision', 0),
                        'recall': basic_metrics.get('recall', 0),
                        'f1_score': basic_metrics.get('f1_score', 0),
                        'roc_auc': basic_metrics.get('roc_auc', 0),
                        'matthews_corrcoef': basic_metrics.get('matthews_corrcoef', 0),
                        'training_time': result.get('training_time', 0)
                    })
            
            if not comparison_data:
                return {'error': 'No valid model results for comparison'}
            
            # Sort by accuracy (primary metric)
            comparison_data.sort(key=lambda x: x['accuracy'], reverse=True)
            
            # Add rankings
            for i, model_data in enumerate(comparison_data, 1):
                model_data['rank'] = i
            
            # Calculate comparison statistics
            best_model = comparison_data[0]
            accuracy_scores = [m['accuracy'] for m in comparison_data]
            
            comparison_results = {
                'rankings': comparison_data,
                'best_model': best_model['model_name'],
                'best_accuracy': best_model['accuracy'],
                'accuracy_range': {
                    'min': float(np.min(accuracy_scores)),
                    'max': float(np.max(accuracy_scores)),
                    'mean': float(np.mean(accuracy_scores)),
                    'std': float(np.std(accuracy_scores))
                },
                'model_count': len(comparison_data)
            }
            
            # Log comparison results
            self.logger.info('📊 === MODEL COMPARISON RESULTS ===')
            for i, model_data in enumerate(comparison_data[:5], 1):  # Top 5
                status = "🏆" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                self.logger.info(f'{status} #{i} {model_data["model_name"]}: '
                               f'Acc={model_data["accuracy"]:.4f}, '
                               f'F1={model_data["f1_score"]:.4f}, '
                               f'Time={model_data["training_time"]:.2f}s')
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f'Model comparison failed: {e}')
            return {'error': str(e)}


def train_model_with_confidence_metrics(model: Any, model_name: str,
                                      X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray,
                                      feature_names: Optional[List[str]] = None,
                                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to train a model with automatic confidence metrics.
    
    Args:
        model: The model to train
        model_name: Name of the model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        config: Configuration dictionary
        
    Returns:
        Comprehensive training and evaluation results
    """
    trainer = EnhancedModelTrainer(config)
    return trainer.train_and_evaluate_model(
        model, model_name, X_train, y_train, X_test, y_test, feature_names
    )
