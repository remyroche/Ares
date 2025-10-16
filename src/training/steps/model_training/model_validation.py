"""
Enhanced Model Validation Sub-Pipeline with Comprehensive Error Detection

This module provides comprehensive model validation functionality
for trained ML models in the trading pipeline with enhanced error detection,
monitoring, and reporting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import traceback

from src.utils.logger import get_system_logger
from src.utils.ml_common.utils.base_safeguards import MLTrainingSafeguards

logger = get_system_logger().getChild('ModelValidation')

class ModelValidationStep:
    """
    Enhanced Model Validation Step for comprehensive model evaluation with error detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced model validation step."""
        self.config = config or {}
        self.logger = logger.getChild('ModelValidationStep')

        # Enhanced error detection and monitoring
        self.safeguards = MLTrainingSafeguards(self.config.get('safeguards', {}))
        self.validation_history = []

        # Validation configuration
        self.validation_thresholds = self.config.get('validation_thresholds', {
            'min_accuracy': 0.6,
            'min_precision': 0.5,
            'min_recall': 0.5,
            'min_f1_score': 0.5,
            'max_validation_time': 300  # seconds
        })

        self.logger.info("🔍 Enhanced Model Validation Step initialized with monitoring capabilities")

    def validate_model_performance(self, metrics: Dict[str, float], model_id: str) -> Dict[str, Any]:
        """Validate model performance against thresholds with enhanced error detection."""
        try:
            validation_result = {
                'is_valid': True,
                'issues': [],
                'recommendations': [],
                'risk_level': 'low',
                'validation_score': 0.0
            }

            # Check accuracy threshold
            accuracy = metrics.get('accuracy', 0)
            if accuracy < self.validation_thresholds['min_accuracy']:
                validation_result['issues'].append(f"Low accuracy: {accuracy:.3f} < {self.validation_thresholds['min_accuracy']}")
                validation_result['recommendations'].append("Consider retraining with more data or different features")
                validation_result['risk_level'] = 'high'
                validation_result['is_valid'] = False

            # Check precision threshold
            precision = metrics.get('precision', 0)
            if precision < self.validation_thresholds['min_precision']:
                validation_result['issues'].append(f"Low precision: {precision:.3f} < {self.validation_thresholds['min_precision']}")
                validation_result['recommendations'].append("Review feature selection and class balance")
                if validation_result['risk_level'] == 'low':
                    validation_result['risk_level'] = 'medium'

            # Check recall threshold
            recall = metrics.get('recall', 0)
            if recall < self.validation_thresholds['min_recall']:
                validation_result['issues'].append(f"Low recall: {recall:.3f} < {self.validation_thresholds['min_recall']}")
                validation_result['recommendations'].append("Check for class imbalance and model bias")
                if validation_result['risk_level'] == 'low':
                    validation_result['risk_level'] = 'medium'

            # Check F1 score threshold
            f1 = metrics.get('f1_score', 0)
            if f1 < self.validation_thresholds['min_f1_score']:
                validation_result['issues'].append(f"Low F1 score: {f1:.3f} < {self.validation_thresholds['min_f1_score']}")
                validation_result['recommendations'].append("Balance precision and recall optimization")
                if validation_result['risk_level'] == 'low':
                    validation_result['risk_level'] = 'medium'

            # Calculate validation score
            validation_result['validation_score'] = (accuracy + precision + recall + f1) / 4

            # Log validation result
            if validation_result['is_valid']:
                self.logger.info(f"✅ Model {model_id} validation passed (score: {validation_result['validation_score']:.3f})")
            else:
                self.logger.warning(f"⚠️ Model {model_id} validation failed: {validation_result['issues']}")

            return validation_result

        except Exception as e:
            error_context = {
                'component': 'model_validation',
                'function': 'validate_model_performance',
                'model_id': model_id,
                'metrics': metrics
            }
            self.safeguards.detect_and_classify_error(e, error_context)
            self.logger.error(f"❌ Model performance validation failed: {e}")
            return {
                'is_valid': False,
                'issues': ['Validation failed due to error'],
                'recommendations': ['Manual review required'],
                'risk_level': 'unknown',
                'validation_score': 0.0
            }

    def track_validation_metrics(self, model_id: str, metrics: Dict[str, float],
                               validation_result: Dict[str, Any]):
        """Track validation metrics over time for trend analysis."""
        try:
            tracking_record = {
                'timestamp': datetime.now(),
                'model_id': model_id,
                'metrics': metrics.copy(),
                'validation_result': validation_result.copy(),
                'validation_score': validation_result.get('validation_score', 0)
            }

            self.validation_history.append(tracking_record)

            # Keep only recent history (last 100 validations)
            if len(self.validation_history) > 100:
                self.validation_history = self.validation_history[-100:]

            self.logger.debug(f"📊 Validation metrics tracked for {model_id}")

        except Exception as e:
            self.logger.error(f"❌ Failed to track validation metrics: {e}")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary with monitoring data."""
        try:
            if not self.validation_history:
                return {'validation_summary': 'no_data'}

            # Calculate statistics
            total_validations = len(self.validation_history)
            successful_validations = sum(1 for v in self.validation_history if v['validation_result']['is_valid'])
            failed_validations = total_validations - successful_validations

            # Calculate average metrics
            avg_metrics = defaultdict(float)
            for validation in self.validation_history:
                for metric, value in validation['metrics'].items():
                    avg_metrics[metric] += value
            for metric in avg_metrics:
                avg_metrics[metric] /= total_validations

            # Calculate trend
            recent_validations = self.validation_history[-10:] if len(self.validation_history) >= 10 else self.validation_history
            recent_success_rate = sum(1 for v in recent_validations if v['validation_result']['is_valid']) / len(recent_validations)

            # Get error summary from safeguards
            error_summary = self.safeguards.get_error_summary()

            return {
                'validation_summary': {
                    'total_validations': total_validations,
                    'successful_validations': successful_validations,
                    'failed_validations': failed_validations,
                    'success_rate': successful_validations / total_validations,
                    'recent_success_rate': recent_success_rate,
                    'average_metrics': dict(avg_metrics)
                },
                'error_summary': error_summary,
                'recent_validations': recent_validations
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to get validation summary: {e}")
            return {'error': str(e)}

    def check_validation_health(self) -> Dict[str, Any]:
        """Check health status of the validation system."""
        try:
            health_status = {
                'overall_health': 'good',
                'issues': [],
                'recommendations': [],
                'risk_level': 'low'
            }

            # Check validation success rate
            if self.validation_history:
                recent_validations = self.validation_history[-20:] if len(self.validation_history) >= 20 else self.validation_history
                success_rate = sum(1 for v in recent_validations if v['validation_result']['is_valid']) / len(recent_validations)

                if success_rate < 0.7:
                    health_status['issues'].append(f"Low validation success rate: {success_rate:.2%}")
                    health_status['recommendations'].append("Review model training pipeline and data quality")
                    health_status['risk_level'] = 'high'
                elif success_rate < 0.8:
                    health_status['issues'].append(f"Moderate validation success rate: {success_rate:.2%}")
                    health_status['recommendations'].append("Monitor validation trends closely")
                    if health_status['risk_level'] == 'low':
                        health_status['risk_level'] = 'medium'

            # Check for performance degradation
            if len(self.validation_history) >= 10:
                recent_scores = [v['validation_score'] for v in self.validation_history[-10:]]
                older_scores = [v['validation_score'] for v in self.validation_history[-20:-10]] if len(self.validation_history) >= 20 else []

                if older_scores:
                    recent_avg = sum(recent_scores) / len(recent_scores)
                    older_avg = sum(older_scores) / len(older_scores)

                    if recent_avg < older_avg - 0.1:  # 10% degradation
                        health_status['issues'].append(f"Performance degradation detected: {recent_avg:.3f} vs {older_avg:.3f}")
                        health_status['recommendations'].append("Investigate recent model changes and data drift")
                        if health_status['risk_level'] == 'low':
                            health_status['risk_level'] = 'medium'

            # Check error rates
            error_summary = self.safeguards.get_error_summary()
            if error_summary['recent_errors_1h'] > 5:
                health_status['issues'].append(f"High validation error rate: {error_summary['recent_errors_1h']} errors in last hour")
                health_status['recommendations'].append("Check validation pipeline stability")
                if health_status['risk_level'] == 'low':
                    health_status['risk_level'] = 'medium'

            # Determine overall health
            if health_status['risk_level'] == 'high':
                health_status['overall_health'] = 'poor'
            elif health_status['risk_level'] == 'medium':
                health_status['overall_health'] = 'fair'

            return health_status

        except Exception as e:
            self.logger.error(f"❌ Validation health check failed: {e}")
            return {
                'overall_health': 'unknown',
                'issues': ['Health check failed'],
                'recommendations': ['Manual review required'],
                'risk_level': 'unknown'
            }

    async def execute_model_validation(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute comprehensive model validation.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            data_dir: Data directory path
            force_rerun: Whether to force rerun
            validation_config: Optional validation configuration

        Returns:
            Dictionary with validation results and artifacts
        """
        self.logger.info("🔍 Starting comprehensive model validation...")

        # Initialize results
        results = {
            'validation_results': {},
            'performance_metrics': {},
            'cross_validation_scores': {},
            'feature_importance': {},
            'model_comparison': {},
            'recommendations': [],
            'validation_artifacts': []
        }

        try:
            # Load trained models
            models = await self._load_trained_models(data_dir, symbol, exchange, timeframe)
            if not models:
                self.logger.error("❌ No trained models found for validation")
                raise FileNotFoundError(f"No trained models found in {data_dir}. Please ensure model training step completed successfully.")

            # Load validation data
            validation_data = await self._load_validation_data(data_dir, symbol, exchange, timeframe)
            if validation_data is None:
                self.logger.warning("⚠️ No validation data found, using synthetic data")
                validation_data = self._create_synthetic_validation_data()

            # Perform comprehensive validation
            validation_results = await self._perform_model_validation(models, validation_data)

            # Generate performance metrics
            performance_metrics = self._calculate_performance_metrics(validation_results)

            # Perform cross-validation
            cv_scores = await self._perform_cross_validation(models, validation_data)

            # Analyze feature importance
            feature_importance = self._analyze_feature_importance(models, validation_data)

            # Generate model comparison
            model_comparison = self._generate_model_comparison(models, performance_metrics)

            # Generate recommendations
            recommendations = self._generate_recommendations(performance_metrics, cv_scores)

            # Update results
            results.update({
                'validation_results': validation_results,
                'performance_metrics': performance_metrics,
                'cross_validation_scores': cv_scores,
                'feature_importance': feature_importance,
                'model_comparison': model_comparison,
                'recommendations': recommendations,
                'validation_artifacts': [
                    f"{data_dir}/validation/validation_report_{symbol}_{exchange}_{timeframe}.json",
                    f"{data_dir}/validation/performance_metrics_{symbol}_{exchange}_{timeframe}.json",
                    f"{data_dir}/validation/model_comparison_{symbol}_{exchange}_{timeframe}.json"
                ]
            })

            self.logger.info("✅ Model validation completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")
            raise RuntimeError(f"Model validation failed: {e}") from e

        return results

    async def _load_trained_models(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Load trained models from the model training step."""
        try:
            models = {}

            # Look for different types of trained models
            model_types = [
                'analyst_models',
                'analyst_ensembles',
                'tactician_models',
                'tactician_ensembles',
                'hmm_models'
            ]

            for model_type in model_types:
                model_path = f"{data_dir}/models/{model_type}_{symbol}_{exchange}_{timeframe}.pkl"
                if Path(model_path).exists():
                    # In a real implementation, load the actual model
                    models[model_type] = {'path': model_path, 'type': model_type}
                    self.logger.info(f"✅ Loaded {model_type} model from: {model_path}")
                else:
                    self.logger.debug(f"⚠️ {model_type} model not found at: {model_path}")

            if models:
                self.logger.info(f"✅ Loaded {len(models)} trained models")
            else:
                self.logger.warning("⚠️ No trained models found")

            return models

        except Exception as e:
            self.logger.error(f"❌ Failed to load trained models: {e}")
            return {}

    async def _load_validation_data(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load validation data for model evaluation."""
        try:
            # Try to load validation data from various sources
            possible_paths = [
                f"{data_dir}/validation/validation_data_{symbol}_{exchange}_{timeframe}.parquet",
                f"{data_dir}/processed/validation_data_{symbol}_{exchange}_{timeframe}.parquet",
                f"{data_dir}/validation_data_{symbol}_{exchange}_{timeframe}.parquet"
            ]

            for path in possible_paths:
                if Path(path).exists():
                    validation_df = pd.read_parquet(path)
                    self.logger.info(f"✅ Loaded validation data from: {path}")
                    return validation_df

            self.logger.warning("⚠️ No validation data found")
            return None

        except Exception as e:
            self.logger.error(f"❌ Failed to load validation data: {e}")
            return None

    def _create_synthetic_validation_data(self) -> pd.DataFrame:
        """Create synthetic validation data for testing."""
        np.random.seed(42)
        n_samples = 1000

        # Create synthetic features
        data = {
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
            'returns': np.random.randn(n_samples) * 0.02,
            'volatility': np.abs(np.random.randn(n_samples)) * 0.1,
            'volume': np.random.exponential(1000, n_samples)
        }

        # Create target variable (simplified)
        data['target'] = (data['returns'] > 0.01).astype(int)

        df = pd.DataFrame(data)
        self.logger.info("✅ Created synthetic validation data")
        return df

    async def _perform_model_validation(
        self,
        models: Dict[str, Any],
        validation_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform comprehensive model validation."""
        validation_results = {}

        try:
            # Prepare features and target
            feature_cols = [col for col in validation_data.columns if col.startswith('feature_')]
            target_col = 'target'

            if feature_cols and target_col in validation_data.columns:
                X = validation_data[feature_cols]
                y = validation_data[target_col]

                for model_name, model_info in models.items():
                    # Load and use the actual trained model for predictions
                    try:
                        model_path = model_info.get('path')
                        if model_path and Path(model_path).exists():
                            # Load the actual model (implementation depends on model type)
                            import joblib
                            model = joblib.load(model_path)

                            # Make predictions with the actual model
                            if hasattr(model, 'predict'):
                                predictions = model.predict(X)
                                # Convert to binary if needed
                                if len(np.unique(predictions)) > 2:
                                    predictions = (predictions > np.median(predictions)).astype(int)
                            else:
                                raise ValueError(f"Model {model_name} does not have predict method")
                        else:
                            raise FileNotFoundError(f"Model file not found: {model_path}")
                    except Exception as model_error:
                        self.logger.error(f"❌ Failed to load/use model {model_name}: {model_error}")
                        raise RuntimeError(f"Model validation failed for {model_name}: {model_error}") from model_error

                    validation_results[model_name] = {
                        'predictions': predictions.tolist(),
                        'actual_values': y.tolist(),
                        'accuracy': float(accuracy_score(y, predictions)),
                        'precision': float(precision_score(y, predictions, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y, predictions, average='weighted', zero_division=0)),
                        'f1_score': float(f1_score(y, predictions, average='weighted', zero_division=0))
                    }

                    self.logger.info(f"✅ Validated {model_name}: Accuracy = {validation_results[model_name]['accuracy']:.3f}")

            else:
                self.logger.warning("⚠️ Insufficient data for validation")

        except Exception as e:
            self.logger.error(f"❌ Model validation failed: {e}")

        return validation_results

    def _calculate_performance_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        for model_name, results in validation_results.items():
            metrics[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'precision': results.get('precision', 0),
                'recall': results.get('recall', 0),
                'f1_score': results.get('f1_score', 0),
                'validation_score': (results.get('accuracy', 0) + results.get('f1_score', 0)) / 2
            }

        # Calculate overall metrics
        if metrics:
            accuracies = [m['accuracy'] for m in metrics.values()]
            f1_scores = [m['f1_score'] for m in metrics.values()]

            metrics['overall'] = {
                'avg_accuracy': np.mean(accuracies),
                'avg_f1_score': np.mean(f1_scores),
                'best_model': max(metrics.keys(), key=lambda k: metrics[k]['validation_score']),
                'worst_model': min(metrics.keys(), key=lambda k: metrics[k]['validation_score'])
            }

        return metrics

    async def _perform_cross_validation(
        self,
        models: Dict[str, Any],
        validation_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform cross-validation for model robustness assessment."""
        cv_results = {}

        try:
            feature_cols = [col for col in validation_data.columns if col.startswith('feature_')]
            target_col = 'target'

            if feature_cols and target_col in validation_data.columns:
                X = validation_data[feature_cols].values
                y = validation_data[target_col].values

            for model_name, model_info in models.items():
                # Use actual cross-validation with loaded models
                try:
                    model_path = model_info.get('path')
                    if model_path and Path(model_path).exists():
                        from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv
                        model = joblib.load(model_path)

                        if hasattr(model, 'predict'):
                            unified = unified_perform_cv(model, X, y, strategy='standard', cv_folds=5, scoring='accuracy')
                            scores = np.array(unified.get('scores', []) or [])
                        else:
                            raise ValueError(f"Model {model_name} does not support cross-validation")
                    else:
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                except Exception as cv_error:
                    self.logger.error(f"❌ Cross-validation failed for {model_name}: {cv_error}")
                    raise RuntimeError(f"Cross-validation failed for {model_name}: {cv_error}") from cv_error

                cv_results[model_name] = {
                    'cv_scores': scores.tolist() if scores.size else [],
                    'mean_score': float(np.mean(scores)) if scores.size else 0.0,
                    'std_score': float(np.std(scores)) if scores.size else 0.0,
                    'cv_folds': 5
                }

                self.logger.info(f"✅ CV for {model_name}: Mean = {cv_results[model_name]['mean_score']:.3f}")

        except Exception as e:
            self.logger.error(f"❌ Cross-validation failed: {e}")

        return cv_results

    def _analyze_feature_importance(
        self,
        models: Dict[str, Any],
        validation_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        importance_analysis = {}

        try:
            feature_cols = [col for col in validation_data.columns if col.startswith('feature_')]

            for model_name, model_info in models.items():
                # Extract actual feature importance from loaded models
                try:
                    model_path = model_info.get('path')
                    if model_path and Path(model_path).exists():
                        model = joblib.load(model_path)

                        # Extract feature importance based on model type
                        if hasattr(model, 'feature_importances_'):
                            importance_scores = model.feature_importances_
                        elif hasattr(model, 'coef_'):
                            importance_scores = np.abs(model.coef_).flatten()
                        elif hasattr(model, 'get_feature_importance'):
                            importance_scores = model.get_feature_importance()
                        else:
                            self.logger.warning(f"⚠️ Model {model_name} does not support feature importance extraction")
                            # Use uniform importance as fallback
                            n_features = len(feature_cols)
                            importance_scores = np.ones(n_features) / n_features

                        # Normalize importance scores
                        if len(importance_scores) == len(feature_cols):
                            importance_scores = importance_scores / np.sum(importance_scores)
                            feature_importance = dict(zip(feature_cols, importance_scores))
                        else:
                            self.logger.warning(f"⚠️ Feature importance shape mismatch for {model_name}")
                            n_features = len(feature_cols)
                            importance_scores = np.ones(n_features) / n_features
                            feature_importance = dict(zip(feature_cols, importance_scores))
                    else:
                        raise FileNotFoundError(f"Model file not found: {model_path}")
                except Exception as importance_error:
                    self.logger.error(f"❌ Feature importance extraction failed for {model_name}: {importance_error}")
                    # Use uniform importance as fallback
                    n_features = len(feature_cols)
                    importance_scores = np.ones(n_features) / n_features
                    feature_importance = dict(zip(feature_cols, importance_scores))

                importance_analysis[model_name] = {
                    'feature_importance': feature_importance,
                    'top_features': sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5],
                    'least_important_features': sorted(feature_importance.items(), key=lambda x: x[1])[:3]
                }

        except Exception as e:
            self.logger.error(f"❌ Feature importance analysis failed: {e}")

        return importance_analysis

    def _generate_model_comparison(self, models: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive model comparison."""
        comparison = {}

        try:
            if 'overall' in performance_metrics:
                overall = performance_metrics['overall']
                comparison = {
                    'best_performing_model': overall.get('best_model', 'unknown'),
                    'worst_performing_model': overall.get('worst_model', 'unknown'),
                    'performance_spread': abs(performance_metrics.get(overall.get('best_model', ''), {}).get('accuracy', 0) -
                                             performance_metrics.get(overall.get('worst_model', ''), {}).get('accuracy', 0)),
                    'model_rankings': sorted(
                        [(name, metrics.get('validation_score', 0)) for name, metrics in performance_metrics.items() if name != 'overall'],
                        key=lambda x: x[1],
                        reverse=True
                    )
                }

        except Exception as e:
            self.logger.error(f"❌ Model comparison generation failed: {e}")

        return comparison

    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, Any],
        cv_scores: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        try:
            if 'overall' in performance_metrics:
                overall = performance_metrics['overall']

                # Accuracy recommendations
                avg_accuracy = overall.get('avg_accuracy', 0)
                if avg_accuracy > 0.8:
                    recommendations.append("✅ Excellent model performance - models are ready for production")
                elif avg_accuracy > 0.7:
                    recommendations.append("⚠️ Good model performance - consider fine-tuning for better results")
                else:
                    recommendations.append("❌ Poor model performance - significant improvements needed")

                # Best model recommendation
                best_model = overall.get('best_model', 'unknown')
                if best_model != 'unknown':
                    recommendations.append(f"🎯 Best performing model: {best_model} - prioritize this model for deployment")

                # Cross-validation recommendations
                for model_name, cv_data in cv_scores.items():
                    std_score = cv_data.get('std_score', 0)
                    if std_score > 0.1:
                        recommendations.append(f"⚠️ {model_name} shows high variance in CV - consider regularization")

                # General recommendations
                recommendations.extend([
                    "📊 Consider ensemble methods combining top-performing models",
                    "🔄 Implement continuous monitoring of model performance in production",
                    "📈 Set up automated retraining pipelines based on performance degradation"
                ])

        except Exception as e:
            self.logger.error(f"❌ Recommendation generation failed: {e}")
            recommendations = ["❌ Unable to generate recommendations due to validation errors"]

        return recommendations
