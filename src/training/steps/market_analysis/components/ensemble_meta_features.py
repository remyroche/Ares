"""
Ensemble Meta-Features Generator

Generates uncertainty, confidence, and disagreement features from base model predictions.
These meta-features help the ensemble understand prediction quality and model agreement.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy.stats import entropy
from src.utils.tprint import tprint


class EnsembleMetaFeaturesGenerator:
    """
    Generate meta-features from base model predictions for ensemble training.
    
    Meta-features include:
    1. Base model predictions (probabilities or classes)
    2. Uncertainty measures (entropy, variance)
    3. Confidence scores (max probability, margin)
    4. Disagreement measures (variance across models, prediction diversity)
    """
    
    def __init__(self, component_name: str = "META_FEATURES"):
        """Initialize the meta-features generator."""
        self.component_name = component_name
        tprint(f"🔧 [{self.component_name}] Meta-features generator initialized", color="cyan")
    
    def generate_meta_features(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        include_uncertainty: bool = True,
        include_confidence: bool = True,
        include_disagreement: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate comprehensive meta-features from base model predictions.
        
        Args:
            base_models: Dictionary of trained base models
            X: Feature matrix for base models
            y: Optional target labels for computing additional meta-features
            include_uncertainty: Include uncertainty features
            include_confidence: Include confidence features
            include_disagreement: Include disagreement features
            
        Returns:
            Tuple of (meta_features, feature_names)
        """
        tprint(
            f"🔧 [{self.component_name}] Generating meta-features from {len(base_models)} base models",
            color="cyan",
            bold=True
        )
        
        all_features = []
        feature_names = []
        
        # 1. Extract base model predictions
        base_predictions_dict = self._extract_base_predictions(base_models, X)
        
        if not base_predictions_dict:
            raise ValueError("No valid base model predictions generated")
        
        # 2. Stack base predictions
        base_predictions_list = []
        base_predictions_names = []
        
        for model_name, predictions in base_predictions_dict.items():
            if predictions.ndim == 1:
                # Class predictions - convert to one-hot
                predictions = self._convert_to_onehot(predictions, y)
            
            base_predictions_list.append(predictions)
            n_classes = predictions.shape[1]
            for class_idx in range(n_classes):
                base_predictions_names.append(f"{model_name}_class_{class_idx}_prob")
        
        # Stack all base predictions
        base_predictions_stacked = np.column_stack(base_predictions_list)
        all_features.append(base_predictions_stacked)
        feature_names.extend(base_predictions_names)
        
        tprint(
            f"📊 [{self.component_name}] Base predictions: {base_predictions_stacked.shape}",
            color="blue"
        )
        
        # 3. Uncertainty features
        if include_uncertainty:
            uncertainty_features, uncertainty_names = self._generate_uncertainty_features(
                base_predictions_list
            )
            all_features.append(uncertainty_features)
            feature_names.extend(uncertainty_names)
            
            tprint(
                f"📊 [{self.component_name}] Uncertainty features: {uncertainty_features.shape}",
                color="blue"
            )
        
        # 4. Confidence features
        if include_confidence:
            confidence_features, confidence_names = self._generate_confidence_features(
                base_predictions_list
            )
            all_features.append(confidence_features)
            feature_names.extend(confidence_names)
            
            tprint(
                f"📊 [{self.component_name}] Confidence features: {confidence_features.shape}",
                color="blue"
            )
        
        # 5. Disagreement features
        if include_disagreement:
            disagreement_features, disagreement_names = self._generate_disagreement_features(
                base_predictions_list
            )
            all_features.append(disagreement_features)
            feature_names.extend(disagreement_names)
            
            tprint(
                f"📊 [{self.component_name}] Disagreement features: {disagreement_features.shape}",
                color="blue"
            )
        
        # Combine all features
        meta_features = np.column_stack(all_features)
        
        tprint(
            f"✅ [{self.component_name}] Generated meta-features: {meta_features.shape} with {len(feature_names)} features",
            color="green",
            bold=True
        )
        
        # Log feature breakdown
        tprint(f"📋 [{self.component_name}] Feature breakdown:", color="cyan")
        tprint(f"   - Base predictions: {base_predictions_stacked.shape[1]}", color="blue")
        if include_uncertainty:
            tprint(f"   - Uncertainty: {len(uncertainty_names)}", color="blue")
        if include_confidence:
            tprint(f"   - Confidence: {len(confidence_names)}", color="blue")
        if include_disagreement:
            tprint(f"   - Disagreement: {len(disagreement_names)}", color="blue")
        
        return meta_features, feature_names
    
    def _extract_base_predictions(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract predictions from all base models."""
        tprint(f"🔧 [{self.component_name}] Extracting base model predictions", color="cyan")
        
        predictions_dict = {}
        
        for model_name, model in base_models.items():
            try:
                if model is None:
                    tprint(f"⚠️ [{self.component_name}] Skipping None model: {model_name}", color="yellow")
                    continue
                
                if not hasattr(model, 'predict'):
                    tprint(f"⚠️ [{self.component_name}] Skipping model without predict: {model_name}", color="yellow")
                    continue
                
                # Get probabilities if available, otherwise class predictions
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    pred_type = "probabilities"
                else:
                    pred = model.predict(X)
                    pred_type = "classes"
                
                predictions_dict[model_name] = pred
                tprint(
                    f"✅ [{self.component_name}] {model_name}: {pred_type} shape {pred.shape}",
                    color="green"
                )
                
            except Exception as e:
                tprint(
                    f"⚠️ [{self.component_name}] Failed to get predictions from {model_name}: {e}",
                    color="yellow"
                )
                continue
        
        tprint(
            f"📊 [{self.component_name}] Extracted predictions from {len(predictions_dict)}/{len(base_models)} models",
            color="blue"
        )
        
        return predictions_dict
    
    def _convert_to_onehot(self, predictions: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert class predictions to one-hot encoded probabilities."""
        if y is not None:
            unique_classes = np.unique(y)
        else:
            unique_classes = np.unique(predictions)
        
        n_classes = len(unique_classes)
        n_samples = len(predictions)
        
        onehot = np.zeros((n_samples, n_classes))
        for i, class_val in enumerate(unique_classes):
            onehot[predictions == class_val, i] = 1.0
        
        return onehot
    
    def _generate_uncertainty_features(
        self,
        predictions_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate uncertainty features from predictions.
        
        Uncertainty measures:
        - Entropy (per model): measure of prediction uncertainty
        - Mean entropy: average uncertainty across models
        - Max entropy: highest uncertainty among models
        - Variance of probabilities: spread in probability distributions
        """
        features = []
        feature_names = []
        
        n_samples = predictions_list[0].shape[0]
        
        # Per-model entropy
        entropies_per_model = []
        for idx, pred in enumerate(predictions_list):
            # Calculate entropy for each sample
            model_entropy = np.array([entropy(pred[i] + 1e-10) for i in range(n_samples)])
            entropies_per_model.append(model_entropy)
            features.append(model_entropy.reshape(-1, 1))
            feature_names.append(f"uncertainty_entropy_model_{idx}")
        
        entropies_per_model = np.column_stack(entropies_per_model)
        
        # Mean entropy across models
        mean_entropy = np.mean(entropies_per_model, axis=1).reshape(-1, 1)
        features.append(mean_entropy)
        feature_names.append("uncertainty_mean_entropy")
        
        # Max entropy across models
        max_entropy = np.max(entropies_per_model, axis=1).reshape(-1, 1)
        features.append(max_entropy)
        feature_names.append("uncertainty_max_entropy")
        
        # Variance of probabilities across models (per class)
        # Stack all predictions: (n_samples, n_models * n_classes)
        all_probs = np.column_stack(predictions_list)
        n_classes = predictions_list[0].shape[1]
        
        # Reshape to (n_samples, n_models, n_classes)
        probs_reshaped = all_probs.reshape(n_samples, len(predictions_list), n_classes)
        
        # Variance across models for each class
        variance_per_class = np.var(probs_reshaped, axis=1)  # Shape: (n_samples, n_classes)
        features.append(variance_per_class)
        for class_idx in range(n_classes):
            feature_names.append(f"uncertainty_variance_class_{class_idx}")
        
        # Mean variance across classes
        mean_variance = np.mean(variance_per_class, axis=1).reshape(-1, 1)
        features.append(mean_variance)
        feature_names.append("uncertainty_mean_variance")
        
        return np.column_stack(features), feature_names
    
    def _generate_confidence_features(
        self,
        predictions_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate confidence features from predictions.
        
        Confidence measures:
        - Max probability (per model): highest class probability
        - Mean max probability: average confidence across models
        - Min max probability: lowest confidence among models
        - Margin (per model): difference between top 2 probabilities
        - Mean margin: average margin across models
        """
        features = []
        feature_names = []
        
        n_samples = predictions_list[0].shape[0]
        
        # Per-model max probability
        max_probs_per_model = []
        for idx, pred in enumerate(predictions_list):
            max_prob = np.max(pred, axis=1).reshape(-1, 1)
            max_probs_per_model.append(max_prob)
            features.append(max_prob)
            feature_names.append(f"confidence_max_prob_model_{idx}")
        
        max_probs_per_model = np.column_stack(max_probs_per_model)
        
        # Mean max probability across models
        mean_max_prob = np.mean(max_probs_per_model, axis=1).reshape(-1, 1)
        features.append(mean_max_prob)
        feature_names.append("confidence_mean_max_prob")
        
        # Min max probability across models
        min_max_prob = np.min(max_probs_per_model, axis=1).reshape(-1, 1)
        features.append(min_max_prob)
        feature_names.append("confidence_min_max_prob")
        
        # Per-model margin (difference between top 2 probabilities)
        margins_per_model = []
        for idx, pred in enumerate(predictions_list):
            sorted_probs = np.sort(pred, axis=1)
            if sorted_probs.shape[1] > 1:
                margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)
            else:
                margin = sorted_probs[:, -1].reshape(-1, 1)
            
            margins_per_model.append(margin)
            features.append(margin)
            feature_names.append(f"confidence_margin_model_{idx}")
        
        margins_per_model = np.column_stack(margins_per_model)
        
        # Mean margin across models
        mean_margin = np.mean(margins_per_model, axis=1).reshape(-1, 1)
        features.append(mean_margin)
        feature_names.append("confidence_mean_margin")
        
        return np.column_stack(features), feature_names
    
    def _generate_disagreement_features(
        self,
        predictions_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate disagreement features from predictions.

        DEPRECATED: Use centralized disagreement features from
        src.feature_generation.categories.ensemble_disagreement instead.

        This method is kept for backwards compatibility but returns empty features.
        All ensemble models should use the centralized disagreement calculator.
        """
        n_samples = predictions_list[0].shape[0]

        # Return empty arrays - disagreement features now come from centralized calculator
        features = np.zeros((n_samples, 0))
        feature_names = []

        return features, feature_names


# Convenience function
def generate_ensemble_meta_features(
    base_models: Dict[str, Any],
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    component_name: str = "META_FEATURES"
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function to generate ensemble meta-features.
    
    Args:
        base_models: Dictionary of trained base models
        X: Feature matrix for base models
        y: Optional target labels
        component_name: Name for logging
        
    Returns:
        Tuple of (meta_features, feature_names)
    """
    generator = EnsembleMetaFeaturesGenerator(component_name)
    return generator.generate_meta_features(
        base_models, X, y,
        include_uncertainty=True,
        include_confidence=True,
        include_disagreement=True
    )

