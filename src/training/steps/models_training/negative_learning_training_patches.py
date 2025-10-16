"""
Negative Learning Training Patches

This module provides patches for existing training functions to automatically
integrate negative learning features without modifying the original code.

Key Features:
- Automatic feature enhancement before training
- Model constraint application
- Sample weight integration
- Backward compatibility
- Drop-in patches for existing functions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
from datetime import datetime
import functools
import warnings

from src.utils.logger import system_logger
from src.training.steps.models_training.negative_learning_training_integration import (
    get_negative_learning_integration,
    enhance_features_for_training,
    get_training_constraints,
    get_training_sample_weights
)

def patch_analyst_training_function(original_function):
    """
    Decorator to patch Analyst training functions with negative learning.

    Args:
        original_function: Original training function to patch

    Returns:
        Patched function with negative learning integration
    """
    @functools.wraps(original_function)
    async def patched_function(
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Patched Analyst training function with negative learning integration.
        """
        logger = system_logger.getChild('PatchedAnalystTraining')

        try:
            # Get negative learning integration
            integration = get_negative_learning_integration()

            if integration is None or not integration.is_initialized:
                logger.warning("Negative learning not initialized, using original function")
                return await original_function(
                    training_data, feature_columns, target_columns, sample_weight, **kwargs
                )

            # Enhance features with negative learning
            logger.info("🔄 Enhancing Analyst features with negative learning...")
            enhanced_training_data = enhance_features_for_training(
                training_data, pipeline_type='analyst'
            )

            # Get enhanced feature columns
            enhanced_feature_columns = [col for col in enhanced_training_data.columns
                                     if col in feature_columns or col.startswith(('momentum_', 'volatility_', 'trend_', 'volume_'))]

            # Get training constraints
            constraints = get_training_constraints(pipeline_type='analyst', model_type='lightgbm')

            # Get enhanced sample weights
            enhanced_sample_weights = get_training_sample_weights(
                enhanced_training_data, pipeline_type='analyst',
                base_weights=pd.Series(sample_weight) if sample_weight is not None else None
            )

            # Convert sample weights back to numpy array
            if enhanced_sample_weights is not None:
                enhanced_sample_weights = enhanced_sample_weights.values

            # Add constraints to kwargs
            if constraints:
                kwargs['monotone_constraints'] = constraints.get('monotone_constraints')
                kwargs['feature_caps'] = constraints.get('feature_caps')

            logger.info(f"✅ Enhanced Analyst features: {len(feature_columns)} -> {len(enhanced_feature_columns)}")

            # Call original function with enhanced data
            result = await original_function(
                enhanced_training_data,
                enhanced_feature_columns,
                target_columns,
                enhanced_sample_weights,
                **kwargs
            )

            # Add negative learning metadata to result
            if isinstance(result, dict):
                result['negative_learning'] = {
                    'enabled': True,
                    'enhanced_features': len(enhanced_feature_columns) - len(feature_columns),
                    'constraints_applied': bool(constraints),
                    'sample_weights_enhanced': enhanced_sample_weights is not None
                }

            return result

        except Exception as e:
            logger.error(f"Negative learning integration failed, falling back to original: {e}")
            return await original_function(
                training_data, feature_columns, target_columns, sample_weight, **kwargs
            )

    return patched_function

def patch_tactician_training_function(original_function):
    """
    Decorator to patch Tactician training functions with negative learning.

    Args:
        original_function: Original training function to patch

    Returns:
        Patched function with negative learning integration
    """
    @functools.wraps(original_function)
    async def patched_function(
        training_data: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        sample_weight: Optional[np.ndarray] = None,
        analyst_outputs: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        """
        Patched Tactician training function with negative learning integration.
        """
        logger = system_logger.getChild('PatchedTacticianTraining')

        try:
            # Get negative learning integration
            integration = get_negative_learning_integration()

            if integration is None or not integration.is_initialized:
                logger.warning("Negative learning not initialized, using original function")
                return await original_function(
                    training_data, feature_columns, target_columns, sample_weight,
                    analyst_outputs=analyst_outputs, **kwargs
                )

            # Enhance features with negative learning
            logger.info("🔄 Enhancing Tactician features with negative learning...")
            enhanced_training_data = enhance_features_for_training(
                training_data, pipeline_type='tactician', analyst_outputs=analyst_outputs
            )

            # Get enhanced feature columns
            enhanced_feature_columns = [col for col in enhanced_training_data.columns
                                     if col in feature_columns or col.startswith(('momentum_', 'rsi_', 'vwap_', 'volume_'))]

            # Get training constraints
            constraints = get_training_constraints(pipeline_type='tactician', model_type='lightgbm')

            # Get enhanced sample weights
            enhanced_sample_weights = get_training_sample_weights(
                enhanced_training_data, pipeline_type='tactician',
                base_weights=pd.Series(sample_weight) if sample_weight is not None else None
            )

            # Convert sample weights back to numpy array
            if enhanced_sample_weights is not None:
                enhanced_sample_weights = enhanced_sample_weights.values

            # Add constraints to kwargs
            if constraints:
                kwargs['monotone_constraints'] = constraints.get('monotone_constraints')
                kwargs['feature_caps'] = constraints.get('feature_caps')

            logger.info(f"✅ Enhanced Tactician features: {len(feature_columns)} -> {len(enhanced_feature_columns)}")

            # Call original function with enhanced data
            result = await original_function(
                enhanced_training_data,
                enhanced_feature_columns,
                target_columns,
                enhanced_sample_weights,
                analyst_outputs=analyst_outputs,
                **kwargs
            )

            # Add negative learning metadata to result
            if isinstance(result, dict):
                result['negative_learning'] = {
                    'enabled': True,
                    'enhanced_features': len(enhanced_feature_columns) - len(feature_columns),
                    'constraints_applied': bool(constraints),
                    'sample_weights_enhanced': enhanced_sample_weights is not None
                }

            return result

        except Exception as e:
            logger.error(f"Negative learning integration failed, falling back to original: {e}")
            return await original_function(
                training_data, feature_columns, target_columns, sample_weight,
                analyst_outputs=analyst_outputs, **kwargs
            )

    return patched_function

def apply_negative_learning_patches():
    """
    Apply negative learning patches to existing training functions.
    This function should be called during module initialization.
    """
    logger = system_logger.getChild('NegativeLearningPatches')
    logger.info("🔧 Applying negative learning patches to training functions...")

    try:
        # Import the training modules
        from . import analyst_models_training
        from . import tactician_models_training

        # Patch Analyst training functions
        if hasattr(analyst_models_training, 'execute_analyst_models_training'):
            analyst_models_training.execute_analyst_models_training = patch_analyst_training_function(
                analyst_models_training.execute_analyst_models_training
            )
            logger.info("✅ Patched execute_analyst_models_training")

        if hasattr(analyst_models_training, 'AnalystModelsTrainingStep'):
            if hasattr(analyst_models_training.AnalystModelsTrainingStep, 'train_analyst_models'):
                analyst_models_training.AnalystModelsTrainingStep.train_analyst_models = patch_analyst_training_function(
                    analyst_models_training.AnalystModelsTrainingStep.train_analyst_models
                )
                logger.info("✅ Patched AnalystModelsTrainingStep.train_analyst_models")

        # Patch Tactician training functions
        if hasattr(tactician_models_training, 'execute_tactician_models_training'):
            tactician_models_training.execute_tactician_models_training = patch_tactician_training_function(
                tactician_models_training.execute_tactician_models_training
            )
            logger.info("✅ Patched execute_tactician_models_training")

        if hasattr(tactician_models_training, 'TacticianModelsTrainingStep'):
            if hasattr(tactician_models_training.TacticianModelsTrainingStep, 'train_tactician_models'):
                tactician_models_training.TacticianModelsTrainingStep.train_tactician_models = patch_tactician_training_function(
                    tactician_models_training.TacticianModelsTrainingStep.train_tactician_models
                )
                logger.info("✅ Patched TacticianModelsTrainingStep.train_tactician_models")

        logger.info("✅ All negative learning patches applied successfully")

    except Exception as e:
        logger.error(f"Failed to apply negative learning patches: {e}")
        logger.warning("Training functions will work without negative learning integration")

def create_enhanced_training_wrapper(
    original_function,
    pipeline_type: str = 'analyst'
):
    """
    Create an enhanced training wrapper with negative learning integration.

    Args:
        original_function: Original training function
        pipeline_type: 'analyst' or 'tactician'

    Returns:
        Enhanced training function
    """
    if pipeline_type == 'analyst':
        return patch_analyst_training_function(original_function)
    elif pipeline_type == 'tactician':
        return patch_tactician_training_function(original_function)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

# Auto-apply patches when module is imported
try:
    apply_negative_learning_patches()
except Exception as e:
    logger = system_logger.getChild('NegativeLearningPatches')
    logger.warning(f"Failed to auto-apply patches: {e}")

# Convenience functions for manual integration
def enhance_analyst_training_data(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """
    Enhance Analyst training data with negative learning.

    Args:
        training_data: Original training data
        feature_columns: Original feature columns
        target_columns: Target columns
        sample_weight: Original sample weights

    Returns:
        Tuple of (enhanced_data, enhanced_columns, enhanced_weights, constraints)
    """
    # Enhance features
    enhanced_data = enhance_features_for_training(training_data, pipeline_type='analyst')

    # Get enhanced feature columns
    enhanced_columns = [col for col in enhanced_data.columns
                       if col in feature_columns or col.startswith(('momentum_', 'volatility_', 'trend_', 'volume_'))]

    # Get constraints
    constraints = get_training_constraints(pipeline_type='analyst', model_type='lightgbm')

    # Get enhanced sample weights
    enhanced_weights = get_training_sample_weights(
        enhanced_data, pipeline_type='analyst',
        base_weights=pd.Series(sample_weight) if sample_weight is not None else None
    )

    if enhanced_weights is not None:
        enhanced_weights = enhanced_weights.values

    return enhanced_data, enhanced_columns, enhanced_weights, constraints

def enhance_tactician_training_data(
    training_data: pd.DataFrame,
    feature_columns: List[str],
    target_columns: List[str],
    sample_weight: Optional[np.ndarray] = None,
    analyst_outputs: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """
    Enhance Tactician training data with negative learning.

    Args:
        training_data: Original training data
        feature_columns: Original feature columns
        target_columns: Target columns
        sample_weight: Original sample weights
        analyst_outputs: Analyst outputs

    Returns:
        Tuple of (enhanced_data, enhanced_columns, enhanced_weights, constraints)
    """
    # Enhance features
    enhanced_data = enhance_features_for_training(
        training_data, pipeline_type='tactician', analyst_outputs=analyst_outputs
    )

    # Get enhanced feature columns
    enhanced_columns = [col for col in enhanced_data.columns
                       if col in feature_columns or col.startswith(('momentum_', 'rsi_', 'vwap_', 'volume_'))]

    # Get constraints
    constraints = get_training_constraints(pipeline_type='tactician', model_type='lightgbm')

    # Get enhanced sample weights
    enhanced_weights = get_training_sample_weights(
        enhanced_data, pipeline_type='tactician',
        base_weights=pd.Series(sample_weight) if sample_weight is not None else None
    )

    if enhanced_weights is not None:
        enhanced_weights = enhanced_weights.values

    return enhanced_data, enhanced_columns, enhanced_weights, constraints
