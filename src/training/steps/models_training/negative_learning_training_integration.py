"""
Negative Learning Training Integration

This module provides integration patches for existing Analyst/Tactician training pipelines
to seamlessly incorporate negative learning features.

Key Features:
- Drop-in integration with existing training functions
- Automatic feature enhancement before training
- Model constraint application
- Sample weight integration
- Backward compatibility
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
import warnings

from src.utils.logger import system_logger
from src.feature_generation.categories.negative_learning_pipeline_integration import (
    NegativeLearningPipelineIntegrator,
    create_negative_learning_integrator
)

# Import enhanced validation utilities
try:
    from src.training.steps.pre_training.utils.validation_utils import (
        PreTrainingValidator, ValidationConfig, ValidationContext,
        validate_negative_learning_inputs, validate_training_data,
        ValidationResult
    )
    VALIDATION_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ WARNING: Enhanced validation utilities not available: {e}")
    VALIDATION_UTILS_AVAILABLE = False

class NegativeLearningTrainingIntegration:
    """
    Integration wrapper for existing training pipelines.
    Automatically enhances features and applies constraints.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = system_logger.getChild('NegativeLearningTrainingIntegration')

        # Initialize negative learning integrator
        self.integrator = create_negative_learning_integrator(
            self.config.get('negative_learning', {})
        )

        # State
        self.is_initialized = False
        self.last_retrain_timestamp = None
        self.enhanced_features_cache = {}

    def initialize_for_training(
        self,
        analyst_features: pd.DataFrame,
        analyst_target: pd.Series,
        tactician_features: pd.DataFrame,
        tactician_target: pd.Series,
        analyst_outputs: Optional[pd.DataFrame] = None,
        retrain_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Initialize negative learning for training.
        Call this before any training step.

        Args:
            analyst_features: 1h Analyst features
            analyst_target: 1h Analyst target
            tactician_features: 15m Tactician features
            tactician_target: 15m Tactician target
            analyst_outputs: Analyst ensemble outputs
            retrain_timestamp: Retrain timestamp

        Returns:
            Initialization results
        """
        self.logger.info("🎯 Initializing negative learning for training...")

        try:
            # Initialize the integrator
            init_results = self.integrator.initialize_negative_learning(
                analyst_features=analyst_features,
                analyst_target=analyst_target,
                tactician_features=tactician_features,
                tactician_target=tactician_target,
                analyst_outputs=analyst_outputs,
                retrain_timestamp=retrain_timestamp
            )

            self.is_initialized = True
            self.last_retrain_timestamp = retrain_timestamp or datetime.now()

            # Clear cache
            self.enhanced_features_cache = {}

            self.logger.info("✅ Negative learning initialized for training")
            return init_results

        except Exception as e:
            self.logger.error(f"Failed to initialize negative learning for training: {e}")
            return {'status': 'failed', 'error': str(e)}

    def enhance_training_features(
        self,
        features_df: pd.DataFrame,
        pipeline_type: str = 'analyst',
        analyst_outputs: Optional[pd.DataFrame] = None,
        inference_timestamp: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Enhance features for training with negative learning.

        Args:
            features_df: Original feature matrix
            pipeline_type: 'analyst' or 'tactician'
            analyst_outputs: Analyst outputs (for tactician)
            inference_timestamp: Inference timestamp

        Returns:
            Enhanced feature matrix
        """
        if not self.is_initialized:
            self.logger.warning("Negative learning not initialized, returning original features")
            return features_df

        # Check cache first
        cache_key = f"{pipeline_type}_{id(features_df)}_{inference_timestamp}"
        if cache_key in self.enhanced_features_cache:
            return self.enhanced_features_cache[cache_key]

        try:
            if pipeline_type == 'analyst':
                enhanced_features = self.integrator.analyst_integration.enhance_analyst_features(
                    features_df, inference_timestamp
                )
            elif pipeline_type == 'tactician':
                enhanced_features = self.integrator.tactician_integration.enhance_tactician_features(
                    features_df, analyst_outputs, inference_timestamp
                )
            else:
                self.logger.warning(f"Unknown pipeline type: {pipeline_type}")
                return features_df

            # Cache the result
            self.enhanced_features_cache[cache_key] = enhanced_features

            return enhanced_features

        except Exception as e:
            self.logger.error(f"Failed to enhance {pipeline_type} features: {e}")
            return features_df

    def get_training_constraints(
        self,
        pipeline_type: str = 'analyst',
        model_type: str = 'lightgbm'
    ) -> Dict[str, Any]:
        """
        Get training constraints for the specified pipeline and model type.

        Args:
            pipeline_type: 'analyst' or 'tactician'
            model_type: 'lightgbm', 'xgboost', 'catboost'

        Returns:
            Training constraints dictionary
        """
        if not self.is_initialized:
            return {}

        try:
            if pipeline_type == 'analyst':
                constraints = self.integrator.analyst_integration.model_constraints
            elif pipeline_type == 'tactician':
                constraints = self.integrator.tactician_integration.model_constraints
            else:
                return {}

            if not constraints:
                return {}

            # Convert to model-specific format
            result = {}

            # Monotone constraints
            if constraints.monotone_constraints:
                if model_type in ['lightgbm', 'catboost']:
                    result['monotone_constraints'] = constraints.monotone_constraints

            # Feature caps
            if constraints.feature_caps:
                result['feature_caps'] = constraints.feature_caps

            # Sample weights (handled separately)
            if constraints.sample_weights is not None:
                result['has_sample_weights'] = True

            return result

        except Exception as e:
            self.logger.error(f"Failed to get training constraints: {e}")
            return {}

    def get_training_sample_weights(
        self,
        features_df: pd.DataFrame,
        pipeline_type: str = 'analyst',
        base_weights: Optional[pd.Series] = None
    ) -> Optional[pd.Series]:
        """
        Get sample weights for training.

        Args:
            features_df: Feature matrix
            pipeline_type: 'analyst' or 'tactician'
            base_weights: Base sample weights

        Returns:
            Enhanced sample weights or None
        """
        if not self.is_initialized:
            return base_weights

        try:
            if pipeline_type == 'analyst':
                return self.integrator.analyst_integration.get_analyst_sample_weights(
                    features_df, base_weights
                )
            elif pipeline_type == 'tactician':
                return self.integrator.tactician_integration.get_tactician_sample_weights(
                    features_df, base_weights
                )
            else:
                return base_weights

        except Exception as e:
            self.logger.error(f"Failed to get sample weights: {e}")
            return base_weights

    def get_negative_learning_features(
        self,
        pipeline_type: str = 'analyst'
    ) -> List[str]:
        """
        Get list of negative learning features for the specified pipeline.

        Args:
            pipeline_type: 'analyst' or 'tactician'

        Returns:
            List of negative learning feature names
        """
        if not self.is_initialized:
            return []

        try:
            if pipeline_type == 'analyst':
                return self.integrator.analyst_integration.negative_features
            elif pipeline_type == 'tactician':
                return self.integrator.tactician_integration.negative_features
            else:
                return []

        except Exception as e:
            self.logger.error(f"Failed to get negative learning features: {e}")
            return []

    def validate_training_performance(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        pipeline_type: str = 'analyst',
        analyst_outputs: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Validate training performance with negative learning.

        Args:
            features_df: Feature matrix
            target: Target variable
            pipeline_type: 'analyst' or 'tactician'
            analyst_outputs: Analyst outputs (for tactician)

        Returns:
            Validation results
        """
        if not self.is_initialized:
            return {'status': 'not_initialized'}

        try:
            if pipeline_type == 'analyst':
                return self.integrator.analyst_integration.validate_analyst_performance(
                    features_df, target
                )
            elif pipeline_type == 'tactician':
                return self.integrator.tactician_integration.validate_tactician_performance(
                    features_df, target, analyst_outputs
                )
            else:
                return {'status': 'unknown_pipeline_type'}

        except Exception as e:
            self.logger.error(f"Failed to validate training performance: {e}")
            return {'status': 'failed', 'error': str(e)}

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'is_initialized': self.is_initialized,
            'last_retrain': self.last_retrain_timestamp,
            'integrator_status': self.integrator.get_integration_status(),
            'cached_features': len(self.enhanced_features_cache)
        }

# Global integration instance
_negative_learning_integration = None

def get_negative_learning_integration() -> Optional[NegativeLearningTrainingIntegration]:
    """Get the global negative learning integration instance"""
    return _negative_learning_integration

def initialize_negative_learning_integration(config: Optional[Dict[str, Any]] = None) -> NegativeLearningTrainingIntegration:
    """Initialize the global negative learning integration instance"""
    global _negative_learning_integration
    _negative_learning_integration = NegativeLearningTrainingIntegration(config)
    return _negative_learning_integration

def enhance_features_for_training(
    features_df: pd.DataFrame,
    pipeline_type: str = 'analyst',
    analyst_outputs: Optional[pd.DataFrame] = None,
    inference_timestamp: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Convenience function to enhance features for training.

    Args:
        features_df: Original feature matrix
        pipeline_type: 'analyst' or 'tactician'
        analyst_outputs: Analyst outputs (for tactician)
        inference_timestamp: Inference timestamp

    Returns:
        Enhanced feature matrix
    """
    integration = get_negative_learning_integration()
    if integration is None:
        return features_df

    return integration.enhance_training_features(
        features_df, pipeline_type, analyst_outputs, inference_timestamp
    )

def get_training_constraints(
    pipeline_type: str = 'analyst',
    model_type: str = 'lightgbm'
) -> Dict[str, Any]:
    """
    Convenience function to get training constraints.

    Args:
        pipeline_type: 'analyst' or 'tactician'
        model_type: 'lightgbm', 'xgboost', 'catboost'

    Returns:
        Training constraints dictionary
    """
    integration = get_negative_learning_integration()
    if integration is None:
        return {}

    return integration.get_training_constraints(pipeline_type, model_type)

def get_training_sample_weights(
    features_df: pd.DataFrame,
    pipeline_type: str = 'analyst',
    base_weights: Optional[pd.Series] = None
) -> Optional[pd.Series]:
    """
    Convenience function to get training sample weights.

    Args:
        features_df: Feature matrix
        pipeline_type: 'analyst' or 'tactician'
        base_weights: Base sample weights

    Returns:
        Enhanced sample weights or None
    """
    integration = get_negative_learning_integration()
    if integration is None:
        return base_weights

    return integration.get_training_sample_weights(features_df, pipeline_type, base_weights)
