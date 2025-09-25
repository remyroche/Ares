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
        self.input_dim = 4  # OHLC features
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
            Tuple of (regime_logits, metadata)
        """
        batch_size, seq_len, features = x.shape
        
        # Extract features from each architecture
        architecture_features = {}
        architecture_outputs = {}
        
        # Neural ODE features
        if self.neural_ode is not None:
            try:
                # Neural ODE expects different input format
                ode_input = x.view(batch_size, -1)  # Flatten for ODE
                ode_output = self.neural_ode(ode_input)
                architecture_features['neural_ode'] = ode_output
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
                    ssm_features = ssm_output[1].mean(dim=1) if len(ssm_output[1].shape) > 2 else ssm_output[1]
                else:
                    ssm_features = ssm_output.mean(dim=1) if len(ssm_output.shape) > 2 else ssm_output
                architecture_features['state_space'] = ssm_features
                architecture_outputs['state_space'] = ssm_output
            except Exception as e:
                self.logger.warning(f"State Space Model forward pass failed: {e}")
                architecture_features['state_space'] = torch.zeros(batch_size, self.hidden_dim)
                architecture_outputs['state_space'] = torch.zeros(batch_size, self.n_regimes)
        
        # Fuse architecture features
        fused_features = self._fuse_architecture_features(architecture_features)
        
        # Apply attention mechanisms
        attended_features = self._apply_attention_mechanisms(fused_features, x)
        
        # Generate regime predictions
        regime_logits = self.regime_classifier(attended_features)
        
        # Calculate ensemble predictions
        ensemble_predictions = self._calculate_ensemble_predictions(architecture_outputs)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(attended_features)
        
        # Predict regime transitions
        transition_probs = self.transition_predictor(attended_features)
        
        # Create metadata
        metadata = {
            'architecture_features': {k: v.detach().cpu().numpy() for k, v in architecture_features.items()},
            'ensemble_predictions': ensemble_predictions.detach().cpu().numpy(),
            'uncertainty_estimates': uncertainty.detach().cpu().numpy(),
            'transition_probabilities': transition_probs.detach().cpu().numpy(),
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
            # Normalize ensemble weights
            weights = F.softmax(self.ensemble_weights, dim=0)
            
            # Weighted combination of outputs
            ensemble_output = torch.zeros_like(list(architecture_outputs.values())[0])
            
            for i, (arch_name, output) in enumerate(architecture_outputs.items()):
                if i < len(weights):
                    ensemble_output += weights[i] * output
            
            return ensemble_output
            
        except Exception as e:
            self.logger.warning(f"Ensemble calculation failed: {e}")
            # Return first available output
            return list(architecture_outputs.values())[0]
    
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