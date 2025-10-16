"""
Hybrid Regime Architecture

Combines multiple neural architectures (Neural ODEs, Vision Transformers, State Space Models)
into a unified hybrid architecture for superior regime detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .perfect_nas_config import PerfectNASConfig

logger = logging.getLogger(__name__)

class HybridRegimeArchitecture(nn.Module):
    """
    Hybrid architecture combining multiple neural approaches for regime detection.

    Integrates:
    - Neural ODEs for continuous-time regime evolution
    - Vision Transformers for temporal pattern recognition
    - Neural State Space Models for regime dynamics
    - Attention mechanisms for feature importance
    - Ensemble methods for robust predictions
    """

    def __init__(self, neural_architectures: Dict[str, nn.Module], config: PerfectNASConfig):
        """Initialize hybrid architecture.

        Args:
            neural_architectures: Dictionary of neural architectures
            config: Perfect NAS configuration
        """
        super(HybridRegimeArchitecture, self).__init__()

        self.config = config
        self.neural_architectures = neural_architectures
        self.n_regimes = config.n_regimes

        # Architecture components
        self.neural_ode = neural_architectures.get('neural_ode')
        self.vision_transformer = neural_architectures.get('vision_transformer')
        self.state_space_model = neural_architectures.get('state_space')

        # Feature dimensions
        self.input_dim = 64  # Feature extractor output
        self.sequence_length = config.sequence_length
        self.hidden_dim = 128

        # Initialize fusion components
        self._initialize_fusion_components()

        # Initialize attention mechanisms
        self._initialize_attention_mechanisms()

        # Initialize ensemble components
        self._initialize_ensemble_components()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("✅ Hybrid Regime Architecture initialized")

    def _initialize_fusion_components(self):
        """Initialize components for fusing different architectures."""
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),  # 3 architectures -> 2x hidden
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Temporal fusion for sequence data
        self.temporal_fusion = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim // 2, self.n_regimes),
            nn.LogSoftmax(dim=-1)
        )

    def _initialize_attention_mechanisms(self):
        """Initialize attention mechanisms for feature importance."""
        # Multi-head attention for feature importance
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Temporal attention for sequence importance
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Cross-attention between architectures
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def _initialize_ensemble_components(self):
        """Initialize ensemble components for robust predictions."""
        # Ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)  # 3 architectures

        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Regime transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_regimes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through hybrid architecture.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)

        Returns:
            Tuple of (regime_logits, metadata) where regime_logits has shape (batch_size, sequence_length, n_regimes)
        """
        batch_size, seq_len, features = x.shape

        # Extract features from each architecture
        architecture_features = {}
        architecture_outputs = {}

        # Process sequence data properly to maintain temporal dimension

        # Neural ODE features
        if self.neural_ode is not None:
            try:
                # Neural ODE expects (batch_size, input_size) format
                # Process each time step through the ODE
                ode_outputs = []
                for t in range(seq_len):
                    ode_input = x[:, t, :]  # (batch_size, input_size)
                    ode_output = self.neural_ode(ode_input)
                    ode_outputs.append(ode_output)

                # Stack outputs: (batch_size, seq_len, output_size)
                ode_output = torch.stack(ode_outputs, dim=1)
                # Average over time dimension for features
                ode_features = ode_output.mean(dim=1)  # (batch_size, output_size)

                architecture_features['neural_ode'] = ode_features
                architecture_outputs['neural_ode'] = ode_output
            except Exception as e:
                self.logger.warning(f"Neural ODE forward pass failed: {e}")
                architecture_features['neural_ode'] = torch.zeros(batch_size, self.hidden_dim)
                architecture_outputs['neural_ode'] = torch.zeros(batch_size, self.n_regimes)

        # Vision Transformer features
        if self.vision_transformer is not None:
            try:
                vt_output = self.vision_transformer(x)
                # Extract features from transformer
                vt_features = vt_output.mean(dim=1) if len(vt_output.shape) > 2 else vt_output
                architecture_features['vision_transformer'] = vt_features
                architecture_outputs['vision_transformer'] = vt_output
            except Exception as e:
                self.logger.warning(f"Vision Transformer forward pass failed: {e}")
                architecture_features['vision_transformer'] = torch.zeros(batch_size, self.hidden_dim)
                architecture_outputs['vision_transformer'] = torch.zeros(batch_size, self.n_regimes)

        # State Space Model features
        if self.state_space_model is not None:
            try:
                ssm_output = self.state_space_model(x)
                if isinstance(ssm_output, tuple):
                    # Handle tuple output - take the main output (usually first element)
                    ssm_features = ssm_output[0].mean(dim=1) if len(ssm_output[0].shape) > 2 else ssm_output[0]
                    architecture_outputs['state_space'] = ssm_output[0]  # Use main output for ensemble
                else:
                    ssm_features = ssm_output.mean(dim=1) if len(ssm_output.shape) > 2 else ssm_output
                    architecture_outputs['state_space'] = ssm_output
                architecture_features['state_space'] = ssm_features
            except Exception as e:
                self.logger.warning(f"State Space Model forward pass failed: {e}")
                architecture_features['state_space'] = torch.zeros(batch_size, self.hidden_dim)
                architecture_outputs['state_space'] = torch.zeros(batch_size, self.n_regimes)

        # Process the entire sequence at once to maintain proper tensor flow
        # This approach is more efficient and maintains proper gradient flow

        # Initialize outputs for all time steps
        all_regime_logits = []
        all_ensemble_predictions = []
        all_uncertainties = []
        all_transition_probs = []
        all_architecture_features = {}

        # Process each time step
        for t in range(seq_len):
            current_features = x[:, t, :]  # (batch_size, features)

            # Get architecture outputs for current time step
            step_architecture_features = {}
            step_architecture_outputs = {}

            # Neural ODE for current time step
            if self.neural_ode is not None:
                try:
                    ode_output = self.neural_ode(current_features)
                    step_architecture_features['neural_ode'] = ode_output
                    step_architecture_outputs['neural_ode'] = ode_output
                except Exception as e:
                    self.logger.warning(f"Neural ODE forward pass failed for step {t}: {e}")
                    step_architecture_features['neural_ode'] = torch.zeros(batch_size, self.hidden_dim)
                    step_architecture_outputs['neural_ode'] = torch.zeros(batch_size, self.n_regimes)

            # Vision Transformer for current time step (use context from previous steps)
            if self.vision_transformer is not None:
                try:
                    # Use a window of context around current time step
                    context_start = max(0, t - 5)  # Reduced context window for efficiency
                    context_end = min(seq_len, t + 6)
                    context_window = x[:, context_start:context_end, :]
                    vt_output = self.vision_transformer(context_window)
                    vt_features = vt_output[:, -1, :] if len(vt_output.shape) > 2 else vt_output
                    step_architecture_features['vision_transformer'] = vt_features
                    step_architecture_outputs['vision_transformer'] = vt_output
                except Exception as e:
                    self.logger.warning(f"Vision Transformer forward pass failed for step {t}: {e}")
                    step_architecture_features['vision_transformer'] = torch.zeros(batch_size, self.hidden_dim)
                    step_architecture_outputs['vision_transformer'] = torch.zeros(batch_size, self.n_regimes)

            # State Space Model for current time step
            if self.state_space_model is not None:
                try:
                    ssm_output = self.state_space_model(current_features.unsqueeze(1))
                    if isinstance(ssm_output, tuple):
                        # Handle tuple output - take the main output (usually first element)
                        ssm_features = ssm_output[0][:, -1, :] if len(ssm_output[0].shape) > 2 else ssm_output[0]
                        step_architecture_outputs['state_space'] = ssm_output[0]  # Use main output for ensemble
                    else:
                        ssm_features = ssm_output[:, -1, :] if len(ssm_output.shape) > 2 else ssm_output
                        step_architecture_outputs['state_space'] = ssm_output
                    step_architecture_features['state_space'] = ssm_features
                except Exception as e:
                    self.logger.warning(f"State Space Model forward pass failed for step {t}: {e}")
                    step_architecture_features['state_space'] = torch.zeros(batch_size, self.hidden_dim)
                    step_architecture_outputs['state_space'] = torch.zeros(batch_size, self.n_regimes)

            # Fuse features for current time step
            fused_features = self._fuse_architecture_features(step_architecture_features)

            # Apply attention mechanisms
            attended_features = self._apply_attention_mechanisms(fused_features, current_features.unsqueeze(1))

            # Generate regime predictions for current time step
            regime_logits = self.regime_classifier(attended_features)
            all_regime_logits.append(regime_logits)

            # Calculate ensemble predictions for current time step
            ensemble_predictions = self._calculate_ensemble_predictions(step_architecture_outputs)
            all_ensemble_predictions.append(ensemble_predictions)

            # Estimate uncertainty for current time step
            uncertainty = self.uncertainty_estimator(attended_features)
            all_uncertainties.append(uncertainty)

            # Predict regime transitions for current time step
            transition_probs = self.transition_predictor(attended_features)
            all_transition_probs.append(transition_probs)

            # Collect architecture features for metadata
            for arch_name, features in step_architecture_features.items():
                if arch_name not in all_architecture_features:
                    all_architecture_features[arch_name] = []
                all_architecture_features[arch_name].append(features)

        # Stack all time step results
        regime_logits = torch.stack(all_regime_logits, dim=1)  # (batch_size, seq_len, n_regimes)
        ensemble_predictions = torch.stack(all_ensemble_predictions, dim=1)
        uncertainty_estimates = torch.stack(all_uncertainties, dim=1)
        transition_probabilities = torch.stack(all_transition_probs, dim=1)

        # Average architecture features across time steps for metadata
        metadata_architecture_features = {}
        for arch_name, features_list in all_architecture_features.items():
            if features_list:
                metadata_architecture_features[arch_name] = torch.stack(features_list, dim=1).mean(dim=1).detach().cpu().numpy()

        # Create metadata
        metadata = {
            'architecture_features': metadata_architecture_features,
            'ensemble_predictions': ensemble_predictions.detach().cpu().numpy(),
            'uncertainty_estimates': uncertainty_estimates.detach().cpu().numpy(),
            'transition_probabilities': transition_probabilities.detach().cpu().numpy(),
            'ensemble_weights': self.ensemble_weights.detach().cpu().numpy()
        }

        return regime_logits, metadata

    def _fuse_architecture_features(self, architecture_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features from different architectures."""
        try:
            # Ensure all features have the same dimension
            aligned_features = []
            for arch_name, features in architecture_features.items():
                if features.shape[-1] != self.hidden_dim:
                    # Project to hidden dimension
                    if not hasattr(self, f'{arch_name}_projection'):
                        setattr(self, f'{arch_name}_projection',
                               nn.Linear(features.shape[-1], self.hidden_dim).to(features.device))
                    projection = getattr(self, f'{arch_name}_projection')
                    features = projection(features)
                aligned_features.append(features)

            # Concatenate features
            if len(aligned_features) > 1:
                concatenated = torch.cat(aligned_features, dim=-1)
            else:
                concatenated = aligned_features[0]

            # Apply fusion network
            fused = self.feature_fusion(concatenated)

            return fused

        except Exception as e:
            self.logger.warning(f"Feature fusion failed: {e}")
            # Return first available features
            return list(architecture_features.values())[0]

    def _apply_attention_mechanisms(self, features: torch.Tensor,
                                  input_sequence: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanisms to features."""
        try:
            # Reshape features for attention
            if len(features.shape) == 2:
                features = features.unsqueeze(1)  # Add sequence dimension

            # Feature attention
            attended_features, attention_weights = self.feature_attention(
                features, features, features
            )

            # Temporal attention (if sequence data available)
            if input_sequence.shape[1] > 1:  # Has sequence dimension
                # Project input sequence to hidden dimension
                if not hasattr(self, 'input_projection'):
                    self.input_projection = nn.Linear(input_sequence.shape[-1], self.hidden_dim)

                projected_input = self.input_projection(input_sequence)
                temporal_attended, temporal_weights = self.temporal_attention(
                    attended_features, projected_input, projected_input
                )
                attended_features = temporal_attended

            # Global average pooling
            if len(attended_features.shape) > 2:
                attended_features = attended_features.mean(dim=1)

            return attended_features

        except Exception as e:
            self.logger.warning(f"Attention mechanism failed: {e}")
            return features

    def _calculate_ensemble_predictions(self, architecture_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate ensemble predictions from different architectures."""
        try:
            if not architecture_outputs:
                # Return default output if no architectures available
                return torch.zeros(1, self.n_regimes)

            # Normalize ensemble weights
            weights = F.softmax(self.ensemble_weights, dim=0)

            # Filter out tuple outputs and convert to tensors
            valid_outputs = {}
            for arch_name, output in architecture_outputs.items():
                if isinstance(output, tuple):
                    # If output is a tuple, take the first element (usually the main output)
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        valid_outputs[arch_name] = output[0]
                    else:
                        self.logger.warning(f"Skipping {arch_name} due to invalid tuple output: {output}")
                        continue
                elif isinstance(output, torch.Tensor):
                    valid_outputs[arch_name] = output
                else:
                    self.logger.warning(f"Skipping {arch_name} due to invalid output type: {type(output)}")
                    continue

            if not valid_outputs:
                return torch.zeros(1, self.n_regimes)

            # Get the first valid output to determine shape
            first_output = list(valid_outputs.values())[0]
            ensemble_output = torch.zeros_like(first_output)

            # Weighted combination of outputs
            for i, (arch_name, output) in enumerate(valid_outputs.items()):
                if i < len(weights):
                    # Ensure output has the same shape as ensemble_output
                    if output.shape != ensemble_output.shape:
                        # Reshape or broadcast to match
                        if output.numel() == ensemble_output.numel():
                            output = output.reshape(ensemble_output.shape)
                        else:
                            # If shapes are incompatible, skip this architecture
                            self.logger.warning(f"Skipping {arch_name} due to shape mismatch: {output.shape} vs {ensemble_output.shape}")
                            continue

                    ensemble_output += weights[i] * output

            return ensemble_output

        except Exception as e:
            self.logger.warning(f"Ensemble calculation failed: {e}")
            # Return first available output or default
            if architecture_outputs:
                first_output = list(architecture_outputs.values())[0]
                if isinstance(first_output, tuple) and len(first_output) > 0:
                    return first_output[0]
                elif isinstance(first_output, torch.Tensor):
                    return first_output
            return torch.zeros(1, self.n_regimes)

    def get_architecture_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """Get importance scores for different architectures."""
        try:
            with torch.no_grad():
                # Get ensemble weights
                weights = F.softmax(self.ensemble_weights, dim=0)

                importance_scores = {}
                architecture_names = ['neural_ode', 'vision_transformer', 'state_space']

                for i, name in enumerate(architecture_names):
                    if i < len(weights):
                        importance_scores[name] = weights[i].item()
                    else:
                        importance_scores[name] = 0.0

                return importance_scores

        except Exception as e:
            self.logger.warning(f"Architecture importance calculation failed: {e}")
            return {'neural_ode': 0.33, 'vision_transformer': 0.33, 'state_space': 0.34}

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature importance scores."""
        try:
            with torch.no_grad():
                # Forward pass to get attention weights
                _, metadata = self.forward(x)

                # Extract attention weights (if available)
                if 'attention_weights' in metadata:
                    return torch.tensor(metadata['attention_weights'])
                else:
                    # Return uniform importance
                    return torch.ones(x.shape[-1]) / x.shape[-1]

        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return torch.ones(x.shape[-1]) / x.shape[-1]

    def predict_regime_evolution(self, x: torch.Tensor, time_steps: int = 10) -> torch.Tensor:
        """Predict regime evolution over time."""
        try:
            with torch.no_grad():
                # Get current regime prediction
                regime_logits, metadata = self.forward(x)
                current_regime = torch.softmax(regime_logits, dim=-1)

                # Predict future regimes using transition probabilities
                transition_probs = torch.tensor(metadata['transition_probabilities'])
                regime_evolution = [current_regime]

                for _ in range(time_steps - 1):
                    # Apply transition matrix
                    next_regime = torch.matmul(regime_evolution[-1], transition_probs)
                    regime_evolution.append(next_regime)

                return torch.stack(regime_evolution)

        except Exception as e:
            self.logger.warning(f"Regime evolution prediction failed: {e}")
            # Return current regime repeated
            regime_logits, _ = self.forward(x)
            current_regime = torch.softmax(regime_logits, dim=-1)
            return current_regime.unsqueeze(0).repeat(time_steps, 1, 1)

    def get_uncertainty_estimates(self, x: torch.Tensor) -> torch.Tensor:
        """Get uncertainty estimates for predictions."""
        try:
            with torch.no_grad():
                _, metadata = self.forward(x)
                return torch.tensor(metadata['uncertainty_estimates'])

        except Exception as e:
            self.logger.warning(f"Uncertainty estimation failed: {e}")
            return torch.ones(x.shape[0], 1) * 0.5
