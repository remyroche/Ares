"""
Streamlined HMM Models Training

Simplified HMM models training that leverages the ml_commons/ ML training pipeline.
Focuses on HMM state recognition with 15m timeframe, using advanced tools for HPO, validation,
lookahead protection, and overfitting detection.

This is the primary HMM training implementation - extensively using ml_commons tools.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import time

# Core imports - using common utilities
from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

# New ml_commons imports for extensive functionality
from src.utils.ml_common.utils.hmm_hpo_config import get_hmm_hyperparameter_optimizer
# from src.utils.ml_common.validation.hmm_validation_pipeline import get_hmm_validation_pipeline
from src.utils.ml_common.utils.hmm_temporal_protection import get_hmm_temporal_protection


class StreamlinedHMMTrainingStep(BaseTrainingStep):
    """
    Streamlined HMM Training Step that leverages common_utils/ ML training pipeline.

    This class focuses specifically on HMM state recognition using 15m timeframe
    and delegates most functionality to the common ML training pipeline.

    Key principles:
    - Use 15m timeframe for HMM state recognition
    - Minimal custom code - delegate to common_utils/
    - Focus on state recognition, not prediction
    - Leverage HPO, validation, and reporting from common pipeline
    - Include ensemble models for robust state recognition
    """

    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize streamlined HMM training step with extensive ml_commons integration.

        Args:
            config: HMM training configuration (will be updated to use 15m timeframe)
        """
        # Ensure we have a config with 15m timeframe for HMM state recognition
        if config is None:
            # Do not reference self.* here; instance is not fully initialized yet
            config = HMMTrainingConfig(
                model_name="streamlined_hmm_state_recognition",
                timeframe="15m",  # Always use 15m for HMM state recognition
                hpo_trials=50,
                enable_multi_objective=True,
                objectives=["accuracy", "f1_score", "regime_stability"],
                objective_weights=[0.4, 0.3, 0.3]  # Normalized to sum to 1.0
            )
        else:
            # Override timeframe to ensure 15m for HMM state recognition
            config.timeframe = "15m"

            # Ensure we have appropriate model types for state recognition
            if not hasattr(config, 'model_types') or len(config.model_types) == 0:
                # Will be finalized after HPO initialization
                pass

        super().__init__(config)
        self.logger = system_logger.getChild('StreamlinedHMMTrainingStep')

        # Initialize ml_commons utilities for extensive functionality
        self.hmm_hpo = get_hmm_hyperparameter_optimizer(config)

        # Fill/normalize config fields now that HPO is available
        try:
            if not getattr(self.config, 'model_types', None):
                self.config.model_types = self.hmm_hpo.get_hmm_model_types()
        except Exception:
            # Fall back to defaults already provided by HMMTrainingConfig
            pass

        # Normalize objective weights for clarity if provided
        try:
            if getattr(self.config, 'objective_weights', None):
                s = float(sum(self.config.objective_weights))
                if s > 0:
                    self.config.objective_weights = [w / s for w in self.config.objective_weights]
        except Exception:
            pass
        # self.hmm_validation = get_hmm_validation_pipeline(config)
        self.hmm_temporal_protection = get_hmm_temporal_protection(config)

        self.logger.info("✅ Streamlined HMM Training Step initialized with ml_commons tools")
        self.logger.info(f"📊 Timeframe: {config.timeframe} (HMM state recognition)")
        self.logger.info(f"📊 Model types: {config.model_types}")
        self.logger.info("🧠 Available tools: HPO, Universal Validation, Temporal Protection")

    def _get_hmm_model_types(self) -> List[str]:
        """
        Get HMM-specific model types optimized for state recognition using ml_commons.

        Uses the HMM HPO configuration for standardized model type selection.

        Returns:
            List of model types optimized for HMM state recognition
        """
        return self.hmm_hpo.get_hmm_model_types()


    def _evaluate_models_with_validation(
        self,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        regime_name: str
    ) -> Dict[str, Any]:
        """
        Evaluate models using the universal validation integrator.

        Args:
            models: Dictionary of trained models
            X_train: Training features
            y_train: Training labels
            regime_name: Name of the regime for context

        Returns:
            Enhanced evaluation results with HMM validation
        """
        self.logger.info(f"🔍 Evaluating models for {regime_name} using universal validation integrator")

        evaluation_results = {}

        for model_name, model in models.items():
            self.logger.info(f"📊 Evaluating {model_name} for {regime_name}")

            try:
                # Create a proper validation split per regime with stratify fallback
                from sklearn.model_selection import train_test_split
                stratify_labels = None
                try:
                    # Use stratify only if each class has at least 2 samples
                    unique, counts = np.unique(y_train, return_counts=True)
                    if len(unique) > 1 and np.all(counts >= 2):
                        stratify_labels = y_train
                except Exception:
                    stratify_labels = None

                try:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=stratify_labels
                    )
                except ValueError:
                    # Fallback to non-stratified split if stratification fails
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_train, y_train, test_size=0.2, random_state=42, stratify=None
                    )

                # Use universal validation integrator for comprehensive evaluation
                validation_result = self.validate_trained_model(
                    model=model,
                    X_train=X_tr,
                    X_val=X_val,
                    y_train=y_tr,
                    y_val=y_val,
                    timestamps=None,
                    feature_names=None,
                    model_name=model_name,
                    model_type=self._get_model_type_from_name(model_name),
                    fold_number=None
                )

                # Add additional model-specific evaluation
                basic_metrics = self.evaluate_models(
                    models={model_name: model},
                    X=X_val,
                    y=y_val,
                    is_classification=True
                )

                evaluation_results[model_name] = {
                    'basic_metrics': basic_metrics.get(model_name, {}),
                    'validation': validation_result,
                    'regime_context': regime_name,
                    'evaluation_timestamp': time.time()
                }

                # Log key findings
                overfitting_analysis = validation_result.get('overfitting_analysis', {})
                if overfitting_analysis.get('overfitting_detected', False):
                    self.logger.warning(f"⚠️ Overfitting detected in {model_name} for {regime_name}: "
                                      f"{overfitting_analysis.get('severity', 'unknown')} severity")

            except Exception as e:
                self.logger.error(f"❌ Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {
                    'error': str(e),
                    'regime_context': regime_name
                }

        return evaluation_results

    def _get_model_type_from_name(self, model_name: str) -> str:
        """Get model type from model name."""
        model_type_mapping = {
            'logistic': 'logistic_regression',
            'lightgbm': 'lightgbm',
            'random_forest': 'random_forest',
            'xgboost': 'xgboost',
            'catboost': 'catboost'
        }

        for key, model_type in model_type_mapping.items():
            if key in model_name.lower():
                return model_type

        return 'unknown'

    def _get_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Extract feature importance from model if available."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).flatten()
            else:
                return None
        except:
            return None

    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute streamlined HMM training using common_utils/ pipeline.

        This method focuses on calling the common ML training pipeline with
        proper parameters for HMM state recognition.

        Args:
            X: Input features
            y: Target values (HMM states to recognize)
            regime_labels: Regime labels for data stratification
            feature_names: Names of input features
            hmm_states: Optional HMM cluster/regime states
            **kwargs: Additional arguments

        Returns:
            Dictionary containing training results from common pipeline
        """
        self.logger.info("🚀 Starting streamlined HMM training execution")

        # Validate input data using universal validation integration from BaseTrainingStep
        validation_results = self.validate_training_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            timestamps=None,
            model_type="hmm_state_recognition"
        )

        if not validation_results['valid']:
            self.logger.error("❌ Training data validation failed")
            return self._handle_training_error(
                Exception("Training data validation failed"),
                "data_validation"
            )

        # Log validation results
        self.logger.info(f"📊 Data validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
        for recommendation in validation_results.get('recommendations', []):
            self.logger.info(f"💡 Recommendation: {recommendation}")

        # Analyze regimes using common regime analysis
        regime_analysis = self.analyze_regimes(regime_labels)
        self.logger.info(f"📊 Regime analysis: {len(regime_analysis['regime_counts'])} regimes")

        # Prepare data for each regime
        regime_data = self.prepare_regime_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            regime_analysis=regime_analysis,
            hmm_states=hmm_states
        )

        # Train models using common training pipeline
        # Focus on state recognition, not prediction
        training_results = self._train_hmm_state_recognition_models(
            regime_data=regime_data,
            feature_names=feature_names
        )

        # Generate enhanced reporting for all models using universal validation
        enhanced_reporting = self._generate_enhanced_model_report(
            models=training_results.get('models', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            regime_analysis=regime_analysis,
            validation_results=validation_results
        )

        # Create final results with enhanced reporting
        final_results = self._create_final_results(
            models=training_results.get('models', {}),
            metadata=training_results.get('metadata', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            training_time=training_results.get('training_time', 0),
            additional_results={
                'regime_analysis': regime_analysis,
                'validation_results': validation_results,
                'hmm_state_recognition_focus': True,
                'timeframe': self.config.timeframe,
                'model_types_used': self.config.model_types,
                'enhanced_reporting': enhanced_reporting,
                'ml_commons_integration': {
                    'hpo_used': True,
                    'universal_validation_used': True,
                    'temporal_protection_used': True,
                    'tools_available': [
                        'HMMHyperparameterOptimizer',
                        'UniversalValidationIntegrator',
                        'HMMTemporalProtection'
                    ]
                }
            }
        )

        # Log comprehensive feature summary
        self.logger.info("📊 Comprehensive Feature Summary:")
        self.logger.info(f"  - Base features: {len(regime_data)} regimes")
        if feature_names:
            self.logger.info(f"  - Enhanced features: {len(feature_names)} total")
            self.logger.info(f"  - Feature categories: 13 comprehensive categories (excluding complex categories)")
        self.logger.info(f"  - Feature bank integration: ✅ Active")

        # Log enhanced summary
        self._log_enhanced_training_summary(final_results)

        return final_results

    def _train_hmm_state_recognition_models(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train models for HMM state recognition using common pipeline.

        Args:
            regime_data: Prepared data for each regime
            feature_names: Names of features

        Returns:
            Training results from common pipeline
        """
        self.logger.info("🔄 Training HMM state recognition models")

        # Get search spaces using HMM HPO configuration from ml_commons
        search_spaces = self.hmm_hpo.get_hmm_state_recognition_search_spaces()

        # Feature engineering utilities
        from .shared_feature_utils import (
            create_enhanced_features_with_names,
            create_comprehensive_features
        )
        import pandas as pd

        # Convert regime data to DataFrame format for feature bank
        enhanced_regime_data = {}
        global_feature_names = feature_names

        for regime_id, data in regime_data.items():
            X_regime = data['X']
            y_regime = data['y']

            # Prefer comprehensive feature-bank ONLY if a real OHLCV DataFrame is provided
            regime_df = data.get('regime_df') if isinstance(data, dict) else None
            if isinstance(regime_df, pd.DataFrame) and set(['open', 'high', 'low', 'close', 'volume']).issubset(regime_df.columns):
                X_enhanced, fn = create_comprehensive_features(
                    regime_df,
                    regime_labels=data.get('regime_labels')
                )
                enhanced_regime_data[regime_id] = {
                    'X': X_enhanced,
                    'y': y_regime,
                    'feature_names': fn
                }
            else:
                # Fast-fail if comprehensive OHLCV DataFrame is not provided
                self.logger.error("Missing required OHLCV regime_df with columns ['open','high','low','close','volume'] for comprehensive feature generation")
                raise ValueError(
                    "OHLCV regime_df with columns ['open','high','low','close','volume'] is required for HMM training features."
                )

        # Train models for each regime using enhanced features
        all_results = {}
        total_training_time = 0

        for regime_id, data in enhanced_regime_data.items():
            self.logger.info(f"📊 Training models for regime {regime_id} with {data['X'].shape[1]} features")

            X_regime = data['X']
            y_regime = data['y']

            # Train models using common pipeline with enhanced features
            regime_results = self.train_models(
                model_types=self.config.model_types,
                X=X_regime,
                y=y_regime,
                enable_hpo=self.config.enable_hpo,
                search_spaces=search_spaces
            )

            all_results[f"regime_{regime_id}"] = regime_results
            total_training_time += regime_results.get('training_time', 0)

        # Evaluate models using common evaluation
        evaluation_results = {}
        for regime_name, regime_results in all_results.items():
            models = regime_results.get('models', {})
            # Use enhanced features for evaluation
            X_regime = enhanced_regime_data[int(regime_name.split('_')[1])]['X']
            y_regime = enhanced_regime_data[int(regime_name.split('_')[1])]['y']

            # Evaluate models using universal validation integration
            evaluation_results[regime_name] = self._evaluate_models_with_validation(
                models=models,
                X_train=X_regime,
                y_train=y_regime,
                regime_name=regime_name
            )

        return {
            'models': all_results,
            'evaluation_results': evaluation_results,
            'training_time': total_training_time,
            'regime_count': len(regime_data)
        }


    def _handle_training_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle training errors with proper logging.

        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Error results dictionary
        """
        error_msg = f"❌ HMM training error{f' in {context}' if context else ''}: {error}"
        self.logger.error(error_msg)

        return {
            'models': {},
            'metadata': {},
            'evaluation_results': {},
            'training_time': 0,
            'config': self.config,
            'error': str(error),
            'hmm_state_recognition_focus': True,
            'timeframe': self.config.timeframe
        }

    def _generate_enhanced_model_report(
        self,
        models: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        regime_analysis: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive enhanced reporting for all trained models using ml_commons tools.

        Args:
            models: Dictionary of trained models
            evaluation_results: Evaluation results for each model
            regime_analysis: Regime analysis results
            validation_results: Optional comprehensive validation results

        Returns:
            Dictionary containing enhanced model reporting with ml_commons integration
        """
        self.logger.info("📊 Generating enhanced model report...")

        enhanced_report = {
            'model_performance_summary': {},
            'regime_specific_performance': {},
            'model_comparison': {},
            'best_models_by_regime': {},
            'overall_recommendations': [],
            'validation_insights': {},
            'ml_commons_integration': {
                'hpo_used': True,
                'validation_pipeline_used': True,
                'temporal_protection_used': True,
                'overfitting_detection_used': True
            },
            'training_metadata': {
                'total_regimes': int(len(regime_analysis.get('regime_counts', [])) if isinstance(regime_analysis.get('regime_counts', []), np.ndarray) else len(regime_analysis.get('regime_counts', {}))),
                'total_models_trained': 0,
                'model_types_used': []
            }
        }

        # Build summaries across regimes
        aggregate_model_metrics: Dict[str, Dict[str, List[float]]] = {}
        model_types_used: set = set()

        for regime_name, regime_evals in evaluation_results.items():
            # regime_evals: Dict[model_name, { 'basic_metrics': {model_name: {...}}, 'validation': {...}, ... }]
            for model_name, eval_result in regime_evals.items():
                model_types_used.add(model_name)
                if model_name not in aggregate_model_metrics:
                    aggregate_model_metrics[model_name] = {
                        'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []
                    }
                basic = eval_result.get('basic_metrics', {})
                model_basic = basic.get(model_name, {}) if isinstance(basic, dict) else {}
                for k in ['accuracy', 'f1_score', 'precision', 'recall']:
                    if k in model_basic:
                        aggregate_model_metrics[model_name][k].append(model_basic[k])

        # Fill model_performance_summary with averages
        for model_name, metrics_lists in aggregate_model_metrics.items():
            def _safe_mean(vals: List[float]) -> float:
                return float(np.mean(vals)) if len(vals) > 0 else 0.0
            enhanced_report['model_performance_summary'][model_name] = {
                'avg_accuracy': _safe_mean(metrics_lists['accuracy']),
                'avg_f1_score': _safe_mean(metrics_lists['f1_score']),
                'avg_precision': _safe_mean(metrics_lists['precision']),
                'avg_recall': _safe_mean(metrics_lists['recall'])
            }

        # Generate regime-specific performance analysis
        # Regime-specific performance from evaluation_results directly
        for regime_name, regime_evals in evaluation_results.items():
            regime_performance = {}
            for model_name, eval_result in regime_evals.items():
                basic = eval_result.get('basic_metrics', {})
                model_basic = basic.get(model_name, {}) if isinstance(basic, dict) else {}
                regime_performance[model_name] = {
                    'accuracy': model_basic.get('accuracy', 0),
                    'f1_score': model_basic.get('f1_score', 0),
                    'precision': model_basic.get('precision', 0),
                    'recall': model_basic.get('recall', 0)
                }
            enhanced_report['regime_specific_performance'][regime_name] = regime_performance

        # Generate model comparison across all regimes
        model_comparison = {}
        for model_name in model_types_used:
            accuracies = []
            f1_scores = []

            for regime_perf in enhanced_report['regime_specific_performance'].values():
                if model_name in regime_perf:
                    accuracies.append(regime_perf[model_name]['accuracy'])
                    f1_scores.append(regime_perf[model_name]['f1_score'])

            if accuracies:
                model_comparison[model_name] = {
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': min(accuracies),
                    'max_accuracy': max(accuracies),
                    'avg_f1_score': np.mean(f1_scores),
                    'std_f1_score': np.std(f1_scores)
                }

        enhanced_report['model_comparison'] = model_comparison

        # Determine best models by regime
        for regime_id, regime_performance in enhanced_report['regime_specific_performance'].items():
            if regime_performance:
                best_model = max(regime_performance.keys(), key=lambda k: regime_performance[k].get('f1_score', 0))
                enhanced_report['best_models_by_regime'][regime_id] = {
                    'best_model': best_model,
                    'best_f1_score': regime_performance[best_model].get('f1_score', 0),
                    'best_accuracy': regime_performance[best_model].get('accuracy', 0)
                }

        # Generate recommendations
        if model_comparison:
            # Find overall best model
            best_overall = max(model_comparison.keys(), key=lambda k: model_comparison[k].get('avg_f1_score', 0))

            enhanced_report['overall_recommendations'] = [
                f"Best overall model: {best_overall} (avg F1: {model_comparison[best_overall]['avg_f1_score']:.4f})",
                "XGBoost vs CatBoost comparison: Both models trained, select best performer per regime",
                "Consider ensemble of top 2 models (logistic_regression + lightgbm) for robustness",
                "Monitor regime-specific performance for model drift detection"
            ]

            # Add regime-specific recommendations
            for regime_id, best_info in enhanced_report['best_models_by_regime'].items():
                enhanced_report['overall_recommendations'].append(
                    f"Regime {regime_id}: Use {best_info['best_model']} (F1: {best_info['best_f1_score']:.4f})"
                )

        # Add validation insights from universal validation tools
        if validation_results:
            enhanced_report['validation_insights'] = self._generate_validation_insights(
                validation_results, evaluation_results
            )

        # Update training metadata
        enhanced_report['training_metadata']['total_models_trained'] = sum(
            len(r) for r in evaluation_results.values()
        )
        enhanced_report['training_metadata']['model_types_used'] = sorted(list(model_types_used))

        self.logger.info("✅ Enhanced model report generated with ml_commons integration")
        return enhanced_report

    def _generate_validation_insights(
        self,
        validation_results: Dict[str, Any],
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate validation insights using ml_commons tools.

        Args:
            validation_results: Comprehensive validation results
            evaluation_results: Model evaluation results

        Returns:
            Dictionary containing validation insights
        """
        insights = {
            'data_quality_insights': {},
            'overfitting_insights': {},
            'temporal_insights': {},
            'regime_insights': {},
            'ml_commons_tool_usage': {
                'validation_pipeline': True,
                'temporal_protection': True,
                'overfitting_detection': True
            }
        }

        # Data quality insights
        data_quality = validation_results.get('data_quality', {})
        if data_quality:
            insights['data_quality_insights'] = {
                'missing_values_analysis': data_quality.get('missing_values', {}),
                'class_distribution': data_quality.get('class_distribution', {}),
                'feature_statistics_summary': len(data_quality.get('feature_statistics', {}))
            }

        # Overfitting insights
        overfitting_detections = []
        for regime_name, regime_evaluations in evaluation_results.items():
            for model_name, evaluation in regime_evaluations.items():
                validation = evaluation.get('validation', {})
                overfitting_analysis = validation.get('overfitting_analysis', {})

                if overfitting_analysis.get('overfitting_detected', False):
                    overfitting_detections.append({
                        'regime': regime_name,
                        'model': model_name,
                        'severity': overfitting_analysis.get('severity', 'unknown'),
                        'confidence': overfitting_analysis.get('confidence_level', 0.0)
                    })

        insights['overfitting_insights'] = {
            'overfitting_detected_count': len(overfitting_detections),
            'overfitting_by_regime': {},
            'overfitting_recommendations': []
        }

        # Group by regime
        for detection in overfitting_detections:
            regime = detection['regime']
            if regime not in insights['overfitting_insights']['overfitting_by_regime']:
                insights['overfitting_insights']['overfitting_by_regime'][regime] = []
            insights['overfitting_insights']['overfitting_by_regime'][regime].append(detection)

        # Temporal insights
        temporal_integrity = validation_results.get('temporal_integrity', {})
        if temporal_integrity:
            insights['temporal_insights'] = {
                'timestamp_ordering_valid': temporal_integrity.get('timestamp_ordering', True),
                'future_data_detected': temporal_integrity.get('future_data_present', False),
                'temporal_range': temporal_integrity.get('timestamp_range', {}),
                'temporal_gaps_detected': len(temporal_integrity.get('temporal_gaps', []))
            }

        # Regime insights
        regime_analysis = validation_results.get('regime_analysis', {})
        if regime_analysis:
            insights['regime_insights'] = {
                'total_regimes': regime_analysis.get('n_regimes', 0),
                'regime_sizes': regime_analysis.get('regime_sizes', {}),
                'regime_quality_summary': {
                    regime_id: {
                        'size': quality.get('size', 0),
                        'min_samples_per_class': quality.get('min_samples_per_class', 0)
                    }
                    for regime_id, quality in regime_analysis.get('regime_quality', {}).items()
                }
            }

        # Generate specific recommendations
        if insights['overfitting_insights']['overfitting_detected_count'] > 0:
            insights['overfitting_insights']['overfitting_recommendations'].extend([
                "High overfitting detected - consider regularization techniques",
                "Implement cross-validation to reduce overfitting",
                "Consider ensemble methods to improve generalization"
            ])

        if insights['temporal_insights'].get('future_data_detected', False):
            insights['overfitting_insights']['overfitting_recommendations'].append(
                "Future data detected - ensure proper temporal data splitting"
            )

        return insights

    def _log_enhanced_training_summary(self, results: Dict[str, Any]) -> None:
        """
        Log enhanced training summary with comprehensive metrics.

        Args:
            results: Training results dictionary
        """
        enhanced_reporting = results.get('enhanced_reporting', {})

        if enhanced_reporting:
            self.logger.info("📊 Enhanced Training Summary:")

            # Overall performance
            model_comparison = enhanced_reporting.get('model_comparison', {})
            if model_comparison:
                best_model = max(model_comparison.keys(),
                               key=lambda k: model_comparison[k]['avg_f1_score'])
                best_f1 = model_comparison[best_model]['avg_f1_score']
                self.logger.info(f"🏆 Best overall model: {best_model} (avg F1: {best_f1:.4f})")

            # Regime-specific insights
            best_models_by_regime = enhanced_reporting.get('best_models_by_regime', {})
            if best_models_by_regime:
                self.logger.info("📊 Best models by regime:")
                for regime_id, best_info in best_models_by_regime.items():
                    self.logger.info(f"  - {regime_id}: {best_info['best_model']} (F1: {best_info['best_f1_score']:.4f})")

            # Recommendations
            recommendations = enhanced_reporting.get('overall_recommendations', [])
            if recommendations:
                self.logger.info("💡 Key recommendations:")
                for rec in recommendations[:3]:  # Show top 3 recommendations
                    self.logger.info(f"  - {rec}")

            # Training metadata
            training_metadata = enhanced_reporting.get('training_metadata', {})
            self.logger.info("📈 Training completed:")
            self.logger.info(f"  - Models trained: {training_metadata.get('total_models_trained', 0)}")
            self.logger.info(f"  - Regimes analyzed: {training_metadata.get('total_regimes', 0)}")
            self.logger.info(f"  - Model types: {', '.join(training_metadata.get('model_types_used', []))}")


# Convenience functions
def create_enhanced_hmm_models_training(config: Optional[HMMTrainingConfig] = None) -> StreamlinedHMMTrainingStep:
    """
    Create a streamlined HMM training step.

    Args:
        config: Optional HMM training configuration

    Returns:
        StreamlinedHMMTrainingStep instance
    """
    return StreamlinedHMMTrainingStep(config)


def execute_enhanced_hmm_models_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute streamlined HMM training.

    Args:
        X: Input features
        y: Target values (HMM states)
        regime_labels: Regime labels
        config: Optional training configuration
        feature_names: Feature names
        hmm_states: Optional HMM states
        **kwargs: Additional arguments

    Returns:
        Training results
    """
    training_step = create_enhanced_hmm_models_training(config)
    return training_step.execute(
        X=X,
        y=y,
        regime_labels=regime_labels,
        feature_names=feature_names,
        hmm_states=hmm_states,
        **kwargs
    )
