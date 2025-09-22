"""
Streamlined HMM Models Training

Simplified HMM models training that leverages the common_utils/ ML training pipeline.
Focuses on HMM state recognition with 15m timeframe, minimal custom code.

This is the primary HMM training implementation - complete migration to common_utils pipeline.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time

# Core imports - using common utilities
from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.config.universal_timeframe_config import get_primary_timeframe


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
        Initialize streamlined HMM training step.

        Args:
            config: HMM training configuration (will be updated to use 15m timeframe)
        """
        # Ensure we have a config with 15m timeframe for HMM state recognition
        if config is None:
            config = HMMTrainingConfig(
                model_name="streamlined_hmm_state_recognition",
                timeframe="15m",  # Always use 15m for HMM state recognition
                model_types=self._get_hmm_model_types(),
                hpo_trials=50,
                enable_multi_objective=True,
                objectives=["accuracy", "f1_score", "regime_stability"],
                objective_weights=[0.4, 0.3, 0.3]
            )
        else:
            # Override timeframe to ensure 15m for HMM state recognition
            config.timeframe = "15m"

            # Ensure we have appropriate model types for state recognition
            if not hasattr(config, 'model_types') or len(config.model_types) == 0:
                config.model_types = self._get_hmm_model_types()

        super().__init__(config)
        self.logger = system_logger.getChild('StreamlinedHMMTrainingStep')

        self.logger.info("✅ Streamlined HMM Training Step initialized")
        self.logger.info(f"📊 Timeframe: {config.timeframe} (HMM state recognition)")
        self.logger.info(f"📊 Model types: {config.model_types}")

    def _get_hmm_model_types(self) -> List[str]:
        """
        Get HMM-specific model types optimized for state recognition.

        Base models: logistic_regression, lightgbm, random_forest (top 2) + xgboost, catboost (compare both)
        No ensemble models or deep learning models for HMM state recognition.

        Returns:
            List of model types optimized for HMM state recognition
        """
        return [
            # Base models for state recognition (top 2 + gradient boosters to compare)
            "logistic_regression",  # Interpretable linear model
            "lightgbm",             # Fast, efficient gradient boosting
            "random_forest",        # Robust ensemble tree model
            "xgboost",              # XGBoost gradient boosting
            "catboost"              # CatBoost gradient boosting (compare with XGBoost)
        ]

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

        # Validate input data using common validation
        validation_results = self.validate_training_data(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            model_type="hmm_state_recognition"
        )

        if not validation_results['valid']:
            self.logger.error("❌ Training data validation failed")
            return self._handle_training_error(
                Exception("Training data validation failed"),
                "data_validation"
            )

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

        # Generate enhanced reporting for all models
        enhanced_reporting = self._generate_enhanced_model_report(
            models=training_results.get('models', {}),
            evaluation_results=training_results.get('evaluation_results', {}),
            regime_analysis=regime_analysis
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
                'enhanced_reporting': enhanced_reporting
            }
        )

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

        # Get search spaces for HMM state recognition
        search_spaces = self._get_hmm_state_recognition_search_spaces()

        # Train models for each regime using common training utilities
        all_results = {}
        total_training_time = 0

        for regime_id, data in regime_data.items():
            self.logger.info(f"📊 Training models for regime {regime_id}")

            X_regime = data['X']
            y_regime = data['y']

            # Train models using common pipeline
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
            X_regime = regime_data[int(regime_name.split('_')[1])]['X']
            y_regime = regime_data[int(regime_name.split('_')[1])]['y']

            evaluation_results[regime_name] = self.evaluate_models(
                models=models,
                X=X_regime,
                y=y_regime,
                is_classification=True  # HMM state recognition is classification
            )

        return {
            'models': all_results,
            'evaluation_results': evaluation_results,
            'training_time': total_training_time,
            'regime_count': len(regime_data)
        }

    def _get_hmm_state_recognition_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """
        Get HPO search spaces optimized for HMM state recognition.

        Returns:
            Dictionary of search spaces for each model type
        """
        return {
            # Base models for HMM state recognition
            'logistic_regression': {
                'C': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
                'penalty': {'type': 'categorical', 'choices': ['l1', 'l2', 'elasticnet']},
                'solver': {'type': 'categorical', 'choices': ['liblinear', 'saga']},
                'max_iter': {'type': 'int', 'low': 500, 'high': 2000}
            },
            'lightgbm': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0}
            },
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 5, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },
            'catboost': {
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'depth': {'type': 'int', 'low': 4, 'high': 10},
                'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
            }
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
        regime_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive enhanced reporting for all trained models.

        Args:
            models: Dictionary of trained models
            evaluation_results: Evaluation results for each model
            regime_analysis: Regime analysis results

        Returns:
            Dictionary containing enhanced model reporting
        """
        self.logger.info("📊 Generating enhanced model report...")

        enhanced_report = {
            'model_performance_summary': {},
            'regime_specific_performance': {},
            'model_comparison': {},
            'best_models_by_regime': {},
            'overall_recommendations': [],
            'training_metadata': {
                'total_regimes': len(regime_analysis.get('regime_counts', {})),
                'total_models_trained': len(models),
                'model_types_used': list(models.keys())
            }
        }

        # Generate performance summary for each model
        for model_name, model_result in models.items():
            if model_name in evaluation_results:
                eval_result = evaluation_results[model_name]

                enhanced_report['model_performance_summary'][model_name] = {
                    'accuracy': eval_result.get('accuracy', 0),
                    'f1_score': eval_result.get('f1_score', 0),
                    'precision': eval_result.get('precision', 0),
                    'recall': eval_result.get('recall', 0),
                    'training_time': eval_result.get('training_time', 0),
                    'regime_specific_metrics': eval_result.get('regime_metrics', {}),
                    'feature_importance_available': eval_result.get('feature_importance_available', False)
                }

        # Generate regime-specific performance analysis
        for regime_id, regime_data in regime_analysis.get('regime_data', {}).items():
            regime_performance = {}

            for model_name, model_result in models.items():
                if model_name in evaluation_results:
                    eval_result = evaluation_results[model_name]
                    regime_metrics = eval_result.get('regime_metrics', {}).get(f'regime_{regime_id}', {})

                    regime_performance[model_name] = {
                        'accuracy': regime_metrics.get('accuracy', 0),
                        'f1_score': regime_metrics.get('f1_score', 0),
                        'precision': regime_metrics.get('precision', 0),
                        'recall': regime_metrics.get('recall', 0),
                        'samples': regime_data.get('n_samples', 0)
                    }

            enhanced_report['regime_specific_performance'][f'regime_{regime_id}'] = regime_performance

        # Generate model comparison across all regimes
        model_comparison = {}
        for model_name in models.keys():
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
            best_model = max(regime_performance.keys(),
                           key=lambda k: regime_performance[k]['f1_score'])
            enhanced_report['best_models_by_regime'][regime_id] = {
                'best_model': best_model,
                'best_f1_score': regime_performance[best_model]['f1_score'],
                'best_accuracy': regime_performance[best_model]['accuracy']
            }

        # Generate recommendations
        if model_comparison:
            # Find overall best model
            best_overall = max(model_comparison.keys(),
                             key=lambda k: model_comparison[k]['avg_f1_score'])

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

        self.logger.info("✅ Enhanced model report generated")
        return enhanced_report

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
