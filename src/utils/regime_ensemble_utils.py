"""
Regime Ensemble Utilities

Shared utilities for regime ensemble operations to avoid code duplication
between regime models training and regime ensemble training components.
"""

import numpy as np
from typing import Dict, Any, List
from src.utils.tprint import tprint

def generate_ensemble_probabilities(
    models: Dict[str, Any],
    X_scaled: np.ndarray,
    feature_names: List[str],
    component_name: str = "REGIME_ENSEMBLE"
) -> Dict[str, np.ndarray]:
    """
    Generate probability predictions from all available models in the ensemble.

    Args:
        models: Dictionary of trained models
        X_scaled: Scaled feature matrix
        feature_names: List of feature names
        component_name: Name of the component for logging (e.g., "REGIME_MODELS", "REGIME_ENSEMBLE")

    Returns:
        Dictionary mapping model names to their probability predictions
    """
    try:
        tprint(f"🔮 [{component_name}] Generating ensemble probabilities from all models", color="cyan")

        ensemble_probabilities = {}

        # Get all available models (excluding metadata)
        available_models = {name: model for name, model in models.items()
                          if model is not None and hasattr(model, 'predict')
                          and name not in ['stacker_lgbm_calibrated_feature_indices']}

        for model_name, model in available_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    # Get probability predictions
                    proba = model.predict_proba(X_scaled)
                    ensemble_probabilities[model_name] = proba
                    tprint(f"✅ [{component_name}] {model_name}: Generated {proba.shape[1]} regime probabilities", color="green")
                else:
                    # Convert class predictions to one-hot probabilities
                    predictions = model.predict(X_scaled)
                    unique_classes = np.unique(predictions)
                    n_classes = len(unique_classes)

                    # Create one-hot encoded probabilities
                    proba = np.zeros((len(predictions), n_classes))
                    for i, class_val in enumerate(unique_classes):
                        proba[predictions == class_val, i] = 1.0

                    ensemble_probabilities[model_name] = proba
                    tprint(f"✅ [{component_name}] {model_name}: Converted class predictions to {n_classes} regime probabilities", color="green")

            except Exception as e:
                tprint(f"⚠️ [{component_name}] Failed to get probabilities from {model_name}: {e}", color="yellow")
                continue

        tprint(f"📊 [{component_name}] Generated ensemble probabilities from {len(ensemble_probabilities)} models", color="blue")
        return ensemble_probabilities

    except Exception as e:
        tprint(f"❌ [{component_name}] Failed to generate ensemble probabilities: {e}", color="red")
        return {}
