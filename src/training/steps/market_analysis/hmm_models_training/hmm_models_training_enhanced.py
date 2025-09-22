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

        Base models: logistic_regression, lightgbm, random_forest (top 2)
        Ensemble models: voting, stacking, bagging, ada boost, extra trees, xgboost (best of XGBoost/CatBoost)

        Returns:
            List of model types optimized for HMM state recognition
        """
        return [
            # Base models for state recognition (top 2)
            "logistic_regression",
            "lightgbm",
            "random_forest",

            # Ensemble models for robust state recognition
            "voting_classifier",        # Voting ensemble
            "stacking_classifier",      # Stacking ensemble
            "bagging_classifier",       # Bagging ensemble
            "ada_boost_classifier",     # AdaBoost ensemble
            "extra_trees_classifier",   # Extra Trees ensemble
            "xgboost",                  # XGBoost as ensemble model (best of XGBoost/CatBoost)

            # Deep learning models (if available)
            "tabnet_classifier",        # TabNet
            "neural_network_classifier"  # Simple NN
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

        # Create final results
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
                'model_types_used': self.config.model_types
            }
        )

        self._log_training_summary(final_results, "HMM State Recognition", len(training_results.get('models', {})))

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
            # Base models (top 2)
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

            # Ensemble models
            'voting_classifier': {
                'voting': {'type': 'categorical', 'choices': ['hard', 'soft']},
                'weights': {'type': 'categorical', 'choices': [None, 'balanced']},
                'flatten_transform': {'type': 'categorical', 'choices': [True, False]}
            },
            'stacking_classifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 8},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            'bagging_classifier': {
                'n_estimators': {'type': 'int', 'low': 10, 'high': 100},
                'max_samples': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'max_features': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'ada_boost_classifier': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True},
                'algorithm': {'type': 'categorical', 'choices': ['SAMME', 'SAMME.R']}
            },
            'extra_trees_classifier': {
                'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 5, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            'xgboost': {  # XGBoost as ensemble model (best of XGBoost/CatBoost)
                'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                'max_depth': {'type': 'int', 'low': 4, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
                'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
            },

            # Deep learning models
            'tabnet_classifier': {
                'n_d': {'type': 'int', 'low': 32, 'high': 128},
                'n_a': {'type': 'int', 'low': 32, 'high': 128},
                'n_steps': {'type': 'int', 'low': 3, 'high': 10},
                'gamma': {'type': 'float', 'low': 1.0, 'high': 2.0},
                'lambda_sparse': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True}
            },
            'neural_network_classifier': {
                'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50)]},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'logistic']},
                'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
                'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True}
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
