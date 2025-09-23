"""
RegimeNAS Framework: Neural Architecture Search for Regime-Aware Models

This module implements the RegimeNAS framework for hierarchical regime detection
and regime-specific model architecture optimization.

Key features:
- Hierarchical regime detection
- Neural Architecture Search (NAS) for regime-specific models
- Regime prediction for next 1-2 periods (15-30mn)
- Meta-learning for regime transitions
- Dynamic architecture adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from enum import Enum

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('RegimeNASFramework')

class RegimeLevel(Enum):
    """Hierarchical regime levels."""
    MICRO = "micro"      # 5-minute level
    SHORT = "short"      # 15-minute level  
    MEDIUM = "medium"     # 1-hour level
    LONG = "long"        # 4-hour level

@dataclass
class RegimeNASConfig:
    """Configuration for RegimeNAS framework."""
    
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60  # 5 hours of 5-minute data
    
    # Hierarchical regime configuration
    regime_levels: List[RegimeLevel] = field(default_factory=lambda: [
        RegimeLevel.MICRO, RegimeLevel.SHORT, RegimeLevel.MEDIUM
    ])
    regime_horizons: Dict[RegimeLevel, int] = field(default_factory=lambda: {
        RegimeLevel.MICRO: 1,    # 1 period (5min)
        RegimeLevel.SHORT: 3,     # 3 periods (15min)
        RegimeLevel.MEDIUM: 12,   # 12 periods (1h)
        RegimeLevel.LONG: 48      # 48 periods (4h)
    })
    
    # NAS configuration
    search_space_size: int = 100
    max_architectures: int = 10
    architecture_epochs: int = 50
    nas_epochs: int = 200
    
    # Regime detection
    num_regimes_per_level: Dict[RegimeLevel, int] = field(default_factory=lambda: {
        RegimeLevel.MICRO: 3,     # Low, Medium, High volatility
        RegimeLevel.SHORT: 4,     # Trending, Ranging, Volatile, Stable
        RegimeLevel.MEDIUM: 5,    # Bull, Bear, Sideways, Volatile, Stable
        RegimeLevel.LONG: 3       # Bull, Bear, Sideways
    })
    
    # Architecture search
    layer_options: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    activation_options: List[str] = field(default_factory=lambda: ['relu', 'gelu', 'swish', 'mish'])
    dropout_options: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    
    # Meta-learning
    meta_learning_rate: float = 0.001
    meta_batch_size: int = 32
    meta_epochs: int = 100
    
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0


class HierarchicalRegimeDetector(nn.Module):
    """Hierarchical regime detection network."""
    
    def __init__(self, config: RegimeNASConfig):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        self.shared_extractor = nn.Sequential(
            nn.Linear(config.input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Level-specific regime detectors
        self.regime_detectors = nn.ModuleDict()
        for level in config.regime_levels:
            num_regimes = config.num_regimes_per_level[level]
            self.regime_detectors[level.value] = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, num_regimes)
            )
        
        # Regime transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, sum(config.num_regimes_per_level.values()))
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical regime detector."""
        # Shared feature extraction
        shared_features = self.shared_extractor(x)
        
        # Level-specific regime predictions
        regime_predictions = {}
        for level in self.config.regime_levels:
            level_key = level.value
            regime_predictions[level_key] = self.regime_detectors[level_key](shared_features)
        
        # Regime transition prediction
        transition_prediction = self.transition_predictor(shared_features)
        
        return {
            'regime_predictions': regime_predictions,
            'transition_prediction': transition_prediction,
            'shared_features': shared_features
        }


class ArchitectureSearchSpace:
    """Neural Architecture Search space definition."""
    
    def __init__(self, config: RegimeNASConfig):
        self.config = config
        self.search_space = self._build_search_space()
        
    def _build_search_space(self) -> Dict[str, List[Any]]:
        """Build the architecture search space."""
        return {
            'num_layers': [2, 3, 4, 5],
            'layer_sizes': self.config.layer_options,
            'activations': self.config.activation_options,
            'dropout_rates': self.config.dropout_options,
            'use_batch_norm': [True, False],
            'use_residual': [True, False]
        }
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture from the search space."""
        architecture = {}
        for key, options in self.search_space.items():
            architecture[key] = np.random.choice(options)
        return architecture
    
    def mutate_architecture(self, architecture: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate an existing architecture."""
        mutated = architecture.copy()
        for key, options in self.search_space.items():
            if np.random.random() < mutation_rate:
                mutated[key] = np.random.choice(options)
        return mutated


class RegimeSpecificArchitecture(nn.Module):
    """Regime-specific neural architecture."""
    
    def __init__(self, architecture_config: Dict[str, Any], input_dim: int, output_dim: int):
        super().__init__()
        self.architecture_config = architecture_config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build architecture
        self.layers = self._build_layers()
        
    def _build_layers(self) -> nn.Module:
        """Build the neural network layers based on architecture config."""
        layers = []
        in_dim = self.input_dim
        
        # Hidden layers
        for i in range(self.architecture_config['num_layers']):
            out_dim = self.architecture_config['layer_sizes']
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim))
            
            # Batch normalization
            if self.architecture_config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(out_dim))
            
            # Activation
            activation = self.architecture_config['activations']
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            elif activation == 'mish':
                layers.append(nn.Mish())
            
            # Dropout
            layers.append(nn.Dropout(self.architecture_config['dropout_rates']))
            
            in_dim = out_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the architecture."""
        return self.layers(x)


class RegimeNASController(nn.Module):
    """Controller network for Neural Architecture Search."""
    
    def __init__(self, config: RegimeNASConfig):
        super().__init__()
        self.config = config
        
        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(config.input_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, config.search_space_size)
        )
        
        # Architecture embedding
        self.architecture_embedding = nn.Embedding(config.search_space_size, 64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate architecture selection probabilities."""
        return F.softmax(self.controller(x), dim=-1)
    
    def sample_architecture(self, x: torch.Tensor) -> torch.Tensor:
        """Sample architecture based on controller output."""
        probs = self.forward(x)
        return torch.multinomial(probs, 1).squeeze(-1)


class RegimeNASFramework(nn.Module):
    """Complete RegimeNAS framework implementation."""
    
    def __init__(self, config: RegimeNASConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.regime_detector = HierarchicalRegimeDetector(config)
        self.nas_controller = RegimeNASController(config)
        self.architecture_search_space = ArchitectureSearchSpace(config)
        
        # Regime-specific architectures
        self.regime_architectures = nn.ModuleDict()
        self._initialize_regime_architectures()
        
        # Meta-learning components
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config.meta_learning_rate)
        
    def _initialize_regime_architectures(self):
        """Initialize regime-specific architectures."""
        for level in self.config.regime_levels:
            level_key = level.value
            num_regimes = self.config.num_regimes_per_level[level]
            
            # Create architecture for each regime
            for regime_id in range(num_regimes):
                regime_key = f"{level_key}_regime_{regime_id}"
                architecture_config = self.architecture_search_space.sample_architecture()
                
                self.regime_architectures[regime_key] = RegimeSpecificArchitecture(
                    architecture_config,
                    input_dim=128,  # From shared features
                    output_dim=self.config.regime_horizons[level]
                )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through RegimeNAS framework."""
        # Hierarchical regime detection
        regime_outputs = self.regime_detector(x)
        
        # Architecture selection
        architecture_probs = self.nas_controller(x)
        selected_architecture = self.nas_controller.sample_architecture(x)
        
        # Regime-specific predictions
        regime_predictions = {}
        for level in self.config.regime_levels:
            level_key = level.value
            level_regimes = regime_outputs['regime_predictions'][level_key]
            
            # Get predicted regime
            predicted_regime = torch.argmax(level_regimes, dim=-1)
            
            # Get regime-specific prediction
            regime_key = f"{level_key}_regime_{predicted_regime.item()}"
            if regime_key in self.regime_architectures:
                regime_prediction = self.regime_architectures[regime_key](regime_outputs['shared_features'])
                regime_predictions[level_key] = regime_prediction
        
        return {
            'regime_predictions': regime_outputs['regime_predictions'],
            'transition_prediction': regime_outputs['transition_prediction'],
            'architecture_selection': selected_architecture,
            'architecture_probs': architecture_probs,
            'regime_specific_predictions': regime_predictions
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss for RegimeNAS."""
        losses = {}
        
        # Regime prediction losses
        for level_key, regime_pred in outputs['regime_predictions'].items():
            if f'{level_key}_regime' in targets:
                regime_loss = F.cross_entropy(regime_pred, targets[f'{level_key}_regime'])
                losses[f'{level_key}_regime_loss'] = regime_loss
        
        # Transition prediction loss
        if 'transition' in targets:
            transition_loss = F.mse_loss(outputs['transition_prediction'], targets['transition'])
            losses['transition_loss'] = transition_loss
        
        # Regime-specific prediction losses
        for level_key, prediction in outputs['regime_specific_predictions'].items():
            if f'{level_key}_prediction' in targets:
                pred_loss = F.mse_loss(prediction, targets[f'{level_key}_prediction'])
                losses[f'{level_key}_prediction_loss'] = pred_loss
        
        # Architecture selection loss (encourage diversity)
        arch_probs = outputs['architecture_probs']
        entropy_loss = -torch.sum(arch_probs * torch.log(arch_probs + 1e-8), dim=-1).mean()
        losses['entropy_loss'] = entropy_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses
    
    def meta_update(self, support_data: List[Tuple[torch.Tensor, torch.Tensor]], 
                   query_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Perform meta-learning update."""
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Inner loop: adapt to support data
        for _ in range(self.config.meta_epochs):
            support_loss = 0
            for data, targets in support_data:
                outputs = self.forward(data)
                losses = self.compute_loss(outputs, targets)
                support_loss += losses['total_loss']
            
            # Update parameters
            self.meta_optimizer.zero_grad()
            support_loss.backward()
            self.meta_optimizer.step()
        
        # Outer loop: evaluate on query data
        query_loss = 0
        for data, targets in query_data:
            outputs = self.forward(data)
            losses = self.compute_loss(outputs, targets)
            query_loss += losses['total_loss']
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_params[name].data
        
        return {'support_loss': support_loss.item(), 'query_loss': query_loss.item()}


class RegimeNASTrainer:
    """Trainer for RegimeNAS framework."""
    
    def __init__(self, model: RegimeNASFramework, config: RegimeNASConfig):
        self.model = model
        self.config = config
        self.logger = get_logger('RegimeNASTrainer')
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True
        )
        
    @traced(span_name='train_epoch')
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0}
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            losses = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    @traced(span_name='validate_epoch')
    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total_loss': 0}
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = self.model(data)
                losses = self.model.compute_loss(outputs, targets)
                
                for key, value in losses.items():
                    epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training loop."""
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Record history
            history['train_loss'].append(train_losses['total_loss'])
            history['val_loss'].append(val_losses['total_loss'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            if epoch % 20 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss: {train_losses["total_loss"]:.4f}, '
                               f'Val Loss: {val_losses["total_loss"]:.4f}, '
                               f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return history


# Factory functions
def create_regime_nas_framework(config: Optional[RegimeNASConfig] = None) -> RegimeNASFramework:
    """Create RegimeNAS framework with default configuration."""
    if config is None:
        config = RegimeNASConfig()
    
    return RegimeNASFramework(config)


def create_regime_nas_trainer(model: RegimeNASFramework, config: Optional[RegimeNASConfig] = None) -> RegimeNASTrainer:
    """Create RegimeNAS trainer."""
    if config is None:
        config = RegimeNASConfig()
    
    return RegimeNASTrainer(model, config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing RegimeNAS Framework')
    
    # Test configuration
    config = RegimeNASConfig(
        input_features=50,
        sequence_length=60
    )
    
    tprint(f'📊 RegimeNAS Configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Sequence length: {config.sequence_length}')
    tprint(f'   → Regime levels: {[level.value for level in config.regime_levels]}')
    tprint(f'   → Regime horizons: {config.regime_horizons}')
    tprint(f'   → Search space size: {config.search_space_size}')
    
    # Test model creation
    try:
        model = create_regime_nas_framework(config)
        trainer = create_regime_nas_trainer(model, config)
        
        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, config.sequence_length, config.input_features)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        tprint('✅ RegimeNAS framework created successfully')
        tprint(f'   → Output keys: {list(outputs.keys())}')
        tprint(f'   → Regime predictions: {list(outputs["regime_predictions"].keys())}')
        tprint(f'   → Architecture selection shape: {outputs["architecture_selection"].shape}')
        tprint(f'   → Architecture probs shape: {outputs["architecture_probs"].shape}')
        
        # Test loss computation
        test_targets = {
            'micro_regime': torch.randint(0, 3, (batch_size,)),
            'short_regime': torch.randint(0, 4, (batch_size,)),
            'medium_regime': torch.randint(0, 5, (batch_size,)),
            'transition': torch.randn(batch_size, sum(config.num_regimes_per_level.values())),
            'micro_prediction': torch.randn(batch_size, config.regime_horizons[RegimeLevel.MICRO]),
            'short_prediction': torch.randn(batch_size, config.regime_horizons[RegimeLevel.SHORT]),
            'medium_prediction': torch.randn(batch_size, config.regime_horizons[RegimeLevel.MEDIUM])
        }
        
        losses = model.compute_loss(outputs, test_targets)
        tprint(f'   → Loss components: {list(losses.keys())}')
        tprint(f'   → Total loss: {losses["total_loss"].item():.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating RegimeNAS framework: {e}')
    
    tprint('✅ RegimeNAS Framework test completed!')