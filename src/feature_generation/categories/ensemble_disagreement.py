"""
Ensemble Disagreement Features

This module provides a centralized implementation of disagreement features
used by ensemble models (analyst_ensemble, tactician_ensemble, regime_ensemble).

Disagreement features capture the level of agreement/disagreement among base models,
which is useful for the meta-learner to assess prediction confidence and uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_ensemble_disagreement_features(
    predictions: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
    feature_prefix: str = "disagreement",
    return_dataframe: bool = True,
    index: Optional[pd.Index] = None
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Calculate standardized disagreement features from ensemble model predictions.

    This function provides a unified implementation of disagreement features used
    across all ensemble training steps (analyst, tactician, regime).

    Features calculated:
    1. variance - Variance of predictions across models
    2. std - Standard deviation of predictions across models
    3. range - Range (max - min) of predictions across models
    4. mad - Mean Absolute Deviation from mean
    5. cv - Coefficient of Variation (normalized std)
    6. iqr - Interquartile Range (robust to outliers)

    Args:
        predictions: Predictions from base models. Can be:
            - pd.DataFrame: columns are model predictions
            - np.ndarray: shape (n_samples, n_models) or (n_models, n_samples, n_classes)
            - List[np.ndarray]: list of prediction arrays from each model
        feature_prefix: Prefix for feature names (default: "disagreement")
        return_dataframe: If True, return pd.DataFrame; else return dict of arrays
        index: Optional pandas Index to use for DataFrame (only if return_dataframe=True)

    Returns:
        pd.DataFrame or dict containing disagreement features:
        - {prefix}_variance: Variance across models
        - {prefix}_std: Standard deviation across models
        - {prefix}_range: Max - min across models
        - {prefix}_mad: Mean absolute deviation
        - {prefix}_cv: Coefficient of variation
        - {prefix}_iqr: Interquartile range

    Examples:
        >>> # From DataFrame (analyst/tactician ensemble)
        >>> base_preds = pd.DataFrame({'model1': [0.1, 0.2], 'model2': [0.15, 0.25]})
        >>> disagreement_feats = calculate_ensemble_disagreement_features(base_preds)

        >>> # From numpy array
        >>> base_preds = np.array([[0.1, 0.15], [0.2, 0.25]])  # (n_samples, n_models)
        >>> disagreement_feats = calculate_ensemble_disagreement_features(base_preds)

        >>> # From list of arrays (for classification with probabilities)
        >>> model_probs = [np.array([[0.7, 0.3], [0.6, 0.4]]),
        ...                np.array([[0.8, 0.2], [0.7, 0.3]])]
        >>> disagreement_feats = calculate_ensemble_disagreement_features(model_probs)
    """
    try:
        # Convert input to numpy array format: (n_samples, n_models)
        predictions_array = _convert_predictions_to_array(predictions)

        if predictions_array.size == 0:
            logger.warning("Empty predictions array provided")
            return _get_empty_disagreement_features(
                n_samples=0,
                feature_prefix=feature_prefix,
                return_dataframe=return_dataframe,
                index=index
            )

        n_samples = predictions_array.shape[0]

        # Calculate disagreement features
        features = {}

        # 1. Variance across models
        features[f'{feature_prefix}_variance'] = np.var(predictions_array, axis=1)

        # 2. Standard deviation across models
        features[f'{feature_prefix}_std'] = np.std(predictions_array, axis=1)

        # 3. Range (max - min) across models
        features[f'{feature_prefix}_range'] = (
            np.max(predictions_array, axis=1) - np.min(predictions_array, axis=1)
        )

        # 4. Mean Absolute Deviation (MAD) from mean
        mean_pred = np.mean(predictions_array, axis=1, keepdims=True)
        features[f'{feature_prefix}_mad'] = np.mean(
            np.abs(predictions_array - mean_pred), axis=1
        )

        # 5. Coefficient of Variation (normalized disagreement)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = features[f'{feature_prefix}_std'] / mean_pred.flatten()
            features[f'{feature_prefix}_cv'] = np.where(np.isfinite(cv), cv, 0.0)

        # 6. Interquartile Range (IQR) - robust to outliers
        q75 = np.percentile(predictions_array, 75, axis=1)
        q25 = np.percentile(predictions_array, 25, axis=1)
        features[f'{feature_prefix}_iqr'] = q75 - q25

        # Return as DataFrame or dict
        if return_dataframe:
            if index is None:
                index = pd.RangeIndex(n_samples)
            return pd.DataFrame(features, index=index)
        else:
            return features

    except Exception as e:
        logger.error(f"Failed to calculate disagreement features: {e}", exc_info=True)
        n_samples = _get_sample_count(predictions)
        return _get_empty_disagreement_features(
            n_samples=n_samples,
            feature_prefix=feature_prefix,
            return_dataframe=return_dataframe,
            index=index
        )


def _convert_predictions_to_array(
    predictions: Union[pd.DataFrame, np.ndarray, List[np.ndarray]]
) -> np.ndarray:
    """
    Convert various prediction formats to a standard numpy array.

    Args:
        predictions: Predictions in various formats

    Returns:
        np.ndarray with shape (n_samples, n_models)
    """
    # Case 1: pandas DataFrame - columns are model predictions
    if isinstance(predictions, pd.DataFrame):
        return predictions.values

    # Case 2: numpy array
    elif isinstance(predictions, np.ndarray):
        # If 3D (n_models, n_samples, n_classes), take max probability per class
        if predictions.ndim == 3:
            # For classification: use the probability of the positive class (last column)
            # or take max probability across classes
            if predictions.shape[2] == 2:
                # Binary classification: take positive class probability
                predictions = predictions[:, :, 1]  # (n_models, n_samples)
            else:
                # Multi-class: take max probability
                predictions = np.max(predictions, axis=2)  # (n_models, n_samples)

        # Ensure shape is (n_samples, n_models)
        if predictions.ndim == 2:
            # Check if it's (n_models, n_samples) and transpose if needed
            # Heuristic: usually n_samples > n_models
            if predictions.shape[0] < predictions.shape[1]:
                return predictions.T
            return predictions
        elif predictions.ndim == 1:
            # Single model - expand to (n_samples, 1)
            return predictions.reshape(-1, 1)
        else:
            raise ValueError(f"Unexpected array shape: {predictions.shape}")

    # Case 3: List of arrays (e.g., from multiple models)
    elif isinstance(predictions, list):
        if len(predictions) == 0:
            return np.array([]).reshape(0, 0)

        # Convert each element to array
        arrays = []
        for pred in predictions:
            arr = np.asarray(pred)

            # Handle 2D predictions (classification with probabilities)
            if arr.ndim == 2:
                if arr.shape[1] == 2:
                    # Binary classification: take positive class
                    arr = arr[:, 1]
                else:
                    # Multi-class: take max probability
                    arr = np.max(arr, axis=1)

            arrays.append(arr.flatten())

        # Stack arrays: (n_models, n_samples) then transpose
        return np.column_stack(arrays)

    else:
        raise TypeError(
            f"Unsupported prediction type: {type(predictions)}. "
            f"Expected pd.DataFrame, np.ndarray, or List[np.ndarray]"
        )


def _get_sample_count(
    predictions: Union[pd.DataFrame, np.ndarray, List[np.ndarray]]
) -> int:
    """Get the number of samples from predictions."""
    if isinstance(predictions, pd.DataFrame):
        return len(predictions)
    elif isinstance(predictions, np.ndarray):
        return predictions.shape[0] if predictions.ndim > 0 else 0
    elif isinstance(predictions, list) and len(predictions) > 0:
        first = predictions[0]
        if isinstance(first, np.ndarray):
            return first.shape[0] if first.ndim > 0 else 0
    return 0


def _get_empty_disagreement_features(
    n_samples: int,
    feature_prefix: str = "disagreement",
    return_dataframe: bool = True,
    index: Optional[pd.Index] = None
) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
    """Return empty/zero disagreement features."""
    if index is None:
        index = pd.RangeIndex(n_samples)

    features = {
        f'{feature_prefix}_variance': np.zeros(n_samples),
        f'{feature_prefix}_std': np.zeros(n_samples),
        f'{feature_prefix}_range': np.zeros(n_samples),
        f'{feature_prefix}_mad': np.zeros(n_samples),
        f'{feature_prefix}_cv': np.zeros(n_samples),
        f'{feature_prefix}_iqr': np.zeros(n_samples),
    }

    if return_dataframe:
        return pd.DataFrame(features, index=index)
    else:
        return features


# Backward compatibility: provide function aliases
generate_disagreement_features = calculate_ensemble_disagreement_features
calculate_disagreement_features = calculate_ensemble_disagreement_features
