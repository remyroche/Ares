"""
Integrated ML Architecture: CLVSA + MultiScaleNBEATS + RegimeNAS + Meta-Labels

This module integrates all the advanced ML architectures into a unified system:
1. CLVSA Architecture (Convolutional-LSTM-Variational-Spatial-Attention)
2. MultiScaleNBEATS for improved forecasting
3. RegimeNAS framework for hierarchical regime detection
4. Meta-labels and patterns for enhanced training
5. Regime-specific hyperparameter optimization

The integrated system provides a comprehensive solution for advanced financial ML.
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

# Import all architecture components
from .clvsa_architecture import CLVSAArchitecture, CLVSAConfig, CLVSATrainer
from .multiscale_nbeats import MultiScaleNBEATS, MultiScaleNBEATSConfig, MultiScaleNBEATSTrainer
from .regime_nas_framework import RegimeNASFramework, RegimeNASConfig, RegimeNASTrainer
from .meta_labels_patterns import MetaLabelsPatternsSystem, MetaLabelsConfig, MetaLabelsPatternsTrainer
from .regime_specific_hpo import RegimeSpecificHPO, RegimeHPOConfig

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('IntegratedMLArchitecture')

@dataclass
class IntegratedMLConfig:
    """Configuration for integrated ML architecture."""
    
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60
    forecast_horizon: int = 12
    
    # Component configurations
    clvsa_config: Optional[CLVSAConfig] = None
    multiscale_nbeats_config: Optional[MultiScaleNBEATSConfig] = None
    regime_nas_config: Optional[RegimeNASConfig] = None
    meta_labels_config: Optional[MetaLabelsConfig] = None
    hpo_config: Optional[RegimeHPOConfig] = None
    
    # Integration configuration
    use_clvsa: bool = True
    use_multiscale_nbeats: bool = True
    use_regime_nas: bool = True
    use_meta_labels: bool = True
    use_hpo: bool = True
    
    # Ensemble configuration
    ensemble_method: str = 'weighted_average'  # 'weighted_average', 'stacking', 'voting'
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'clvsa': 0.3,
        'multiscale_nbeats': 0.3,
        'regime_nas': 0.2,
        'meta_labels': 0.2
    })
    
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    epochs: int = 100
    
    def __post_init__(self):
        """Initialize component configurations if not provided."""
        if self.clvsa_config is None:
            self.clvsa_config = CLVSAConfig(
                input_features=self.input_features,
                sequence_length=self.sequence_length
            )
        
        if self.multiscale_nbeats_config is None:
            self.multiscale_nbeats_config = MultiScaleNBEATSConfig(
                input_features=self.input_features,
                sequence_length=self.sequence_length,
                forecast_horizon=self.forecast_horizon
            )
        
        if self.regime_nas_config is None:
            self.regime_nas_config = RegimeNASConfig(
                input_features=self.input_features,
                sequence_length=self.sequence_length
            )
        
        if self.meta_labels_config is None:
            self.meta_labels_config = MetaLabelsConfig(
                input_features=self.input_features,
                sequence_length=self.sequence_length
            )
        
        if self.hpo_config is None:
            self.hpo_config = RegimeHPOConfig()


class IntegratedMLArchitecture(nn.Module):
    """Integrated ML architecture combining all components."""
    
    def __init__(self, config: IntegratedMLConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.components = nn.ModuleDict()
        self.component_outputs = {}
        
        # CLVSA Architecture
        if config.use_clvsa:
            self.components['clvsa'] = CLVSAArchitecture(config.clvsa_config)
        
        # MultiScaleNBEATS
        if config.use_multiscale_nbeats:
            self.components['multiscale_nbeats'] = MultiScaleNBEATS(config.multiscale_nbeats_config)
        
        # RegimeNAS Framework
        if config.use_regime_nas:
            self.components['regime_nas'] = RegimeNASFramework(config.regime_nas_config)
        
        # Meta-labels and Patterns
        if config.use_meta_labels:
            self.components['meta_labels'] = MetaLabelsPatternsSystem(config.meta_labels_config)
        
        # Regime-specific HPO
        if config.use_hpo:
            self.hpo_system = RegimeSpecificHPO(config.hpo_config)
        
        # Ensemble layer
        self.ensemble_layer = self._create_ensemble_layer()
        
        # Output layers
        self.output_layers = nn.ModuleDict({
            'final_prediction': nn.Linear(self._get_ensemble_input_dim(), self.forecast_horizon),
            'uncertainty': nn.Linear(self._get_ensemble_input_dim(), 1),
            'regime_prediction': nn.Linear(self._get_ensemble_input_dim(), 3)
        })
        
    def _get_ensemble_input_dim(self) -> int:
        """Calculate ensemble input dimension."""
        total_dim = 0
        
        if self.config.use_clvsa:
            total_dim += self.config.clvsa_config.attention_input_dim
        
        if self.config.use_multiscale_nbeats:
            total_dim += self.config.multiscale_nbeats_config.forecast_horizon
        
        if self.config.use_regime_nas:
            total_dim += sum(self.config.regime_nas_config.regime_horizons.values())
        
        if self.config.use_meta_labels:
            total_dim += self.config.meta_labels_config.pattern_embedding_dim
        
        return total_dim
    
    def _create_ensemble_layer(self) -> nn.Module:
        """Create ensemble layer for combining component outputs."""
        input_dim = self._get_ensemble_input_dim()
        
        if self.config.ensemble_method == 'weighted_average':
            return nn.Linear(input_dim, input_dim)
        elif self.config.ensemble_method == 'stacking':
            return nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(input_dim // 2, input_dim // 4)
            )
        else:  # voting
            return nn.Linear(input_dim, input_dim)
    
    def forward(self, x: torch.Tensor, regime_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through integrated architecture."""
        component_outputs = {}
        
        # CLVSA Architecture
        if self.config.use_clvsa and 'clvsa' in self.components:
            clvsa_outputs = self.components['clvsa'](x)
            component_outputs['clvsa'] = clvsa_outputs
        
        # MultiScaleNBEATS
        if self.config.use_multiscale_nbeats and 'multiscale_nbeats' in self.components:
            nbeats_outputs = self.components['multiscale_nbeats'](x, regime_ids)
            component_outputs['multiscale_nbeats'] = nbeats_outputs
        
        # RegimeNAS Framework
        if self.config.use_regime_nas and 'regime_nas' in self.components:
            regime_nas_outputs = self.components['regime_nas'](x)
            component_outputs['regime_nas'] = regime_nas_outputs
        
        # Meta-labels and Patterns
        if self.config.use_meta_labels and 'meta_labels' in self.components:
            meta_labels_outputs = self.components['meta_labels'](x)
            component_outputs['meta_labels'] = meta_labels_outputs
        
        # Ensemble combination
        ensemble_features = self._combine_component_outputs(component_outputs)
        
        # Final predictions
        final_outputs = {
            'prediction': self.output_layers['final_prediction'](ensemble_features),
            'uncertainty': self.output_layers['uncertainty'](ensemble_features),
            'regime_prediction': self.output_layers['regime_prediction'](ensemble_features),
            'component_outputs': component_outputs,
            'ensemble_features': ensemble_features
        }
        
        return final_outputs
    
    def _combine_component_outputs(self, component_outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Combine outputs from all components."""
        combined_features = []
        
        # CLVSA features
        if 'clvsa' in component_outputs:
            clvsa_features = component_outputs['clvsa'].get('global_features', 
                                                          torch.zeros(component_outputs['clvsa']['regime_prediction'].size(0), 
                                                                    self.config.clvsa_config.attention_input_dim))
            combined_features.append(clvsa_features)
        
        # MultiScaleNBEATS features
        if 'multiscale_nbeats' in component_outputs:
            nbeats_features = component_outputs['multiscale_nbeats']['forecast']
            combined_features.append(nbeats_features)
        
        # RegimeNAS features
        if 'regime_nas' in component_outputs:
            regime_nas_features = torch.cat([
                pred for pred in component_outputs['regime_nas']['regime_specific_predictions'].values()
            ], dim=-1)
            combined_features.append(regime_nas_features)
        
        # Meta-labels features
        if 'meta_labels' in component_outputs:
            meta_features = component_outputs['meta_labels']['pattern_embedding']
            combined_features.append(meta_features)
        
        # Combine all features
        if combined_features:
            ensemble_features = torch.cat(combined_features, dim=-1)
        else:
            # Fallback to zero features
            ensemble_features = torch.zeros(x.size(0), self._get_ensemble_input_dim())
        
        # Apply ensemble layer
        ensemble_output = self.ensemble_layer(ensemble_features)
        
        return ensemble_output
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss for integrated architecture."""
        losses = {}
        
        # Final prediction loss
        if 'prediction' in targets:
            prediction_loss = F.mse_loss(outputs['prediction'], targets['prediction'])
            losses['prediction_loss'] = prediction_loss
        
        # Uncertainty loss
        if 'uncertainty' in targets:
            uncertainty_loss = F.mse_loss(outputs['uncertainty'], targets['uncertainty'])
            losses['uncertainty_loss'] = uncertainty_loss
        
        # Regime prediction loss
        if 'regime' in targets:
            regime_loss = F.cross_entropy(outputs['regime_prediction'], targets['regime'])
            losses['regime_loss'] = regime_loss
        
        # Component-specific losses
        component_outputs = outputs.get('component_outputs', {})
        
        # CLVSA losses
        if 'clvsa' in component_outputs:
            clvsa_losses = self.components['clvsa'].compute_loss(component_outputs['clvsa'], targets)
            for key, value in clvsa_losses.items():
                losses[f'clvsa_{key}'] = value
        
        # MultiScaleNBEATS losses
        if 'multiscale_nbeats' in component_outputs:
            nbeats_losses = self.components['multiscale_nbeats'].compute_loss(component_outputs['multiscale_nbeats'], targets)
            for key, value in nbeats_losses.items():
                losses[f'nbeats_{key}'] = value
        
        # RegimeNAS losses
        if 'regime_nas' in component_outputs:
            regime_nas_losses = self.components['regime_nas'].compute_loss(component_outputs['regime_nas'], targets)
            for key, value in regime_nas_losses.items():
                losses[f'regime_nas_{key}'] = value
        
        # Meta-labels losses
        if 'meta_labels' in component_outputs:
            meta_labels_losses = self.components['meta_labels'].compute_loss(component_outputs['meta_labels'], targets)
            for key, value in meta_labels_losses.items():
                losses[f'meta_labels_{key}'] = value
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class IntegratedMLTrainer:
    """Trainer for integrated ML architecture."""
    
    def __init__(self, model: IntegratedMLArchitecture, config: IntegratedMLConfig):
        self.model = model
        self.config = config
        self.logger = get_logger('IntegratedMLTrainer')
        
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
        
        # Component trainers
        self.component_trainers = {}
        self._initialize_component_trainers()
        
    def _initialize_component_trainers(self):
        """Initialize trainers for individual components."""
        if self.config.use_clvsa:
            self.component_trainers['clvsa'] = CLVSATrainer(
                self.model.components['clvsa'], 
                self.config.clvsa_config
            )
        
        if self.config.use_multiscale_nbeats:
            self.component_trainers['multiscale_nbeats'] = MultiScaleNBEATSTrainer(
                self.model.components['multiscale_nbeats'],
                self.config.multiscale_nbeats_config
            )
        
        if self.config.use_regime_nas:
            self.component_trainers['regime_nas'] = RegimeNASTrainer(
                self.model.components['regime_nas'],
                self.config.regime_nas_config
            )
        
        if self.config.use_meta_labels:
            self.component_trainers['meta_labels'] = MetaLabelsPatternsTrainer(
                self.model.components['meta_labels'],
                self.config.meta_labels_config
            )
    
    @traced(span_name='train_epoch')
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0}
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data, targets.get('regime_ids'))
            
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
                outputs = self.model(data, targets.get('regime_ids'))
                losses = self.model.compute_loss(outputs, targets)
                
                for key, value in losses.items():
                    epoch_losses[key] = epoch_losses.get(key, 0) + value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              epochs: int = None) -> Dict[str, List[float]]:
        """Complete training loop."""
        if epochs is None:
            epochs = self.config.epochs
        
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
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using regime-specific HPO."""
        if not self.config.use_hpo:
            self.logger.warning("HPO not enabled in configuration")
            return {}
        
        self.logger.info("🔍 Starting hyperparameter optimization")
        
        # Run regime-specific optimization
        hpo_results = self.model.hpo_system.optimize_all_regimes(X, y, regime_labels)
        
        # Create optimized models
        optimized_models = self.model.hpo_system.create_regime_models(X, y, regime_labels)
        
        self.logger.info("✅ Hyperparameter optimization completed")
        
        return {
            'optimization_results': hpo_results,
            'optimized_models': optimized_models
        }


# Factory functions
def create_integrated_ml_architecture(config: Optional[IntegratedMLConfig] = None) -> IntegratedMLArchitecture:
    """Create integrated ML architecture with default configuration."""
    if config is None:
        config = IntegratedMLConfig()
    
    return IntegratedMLArchitecture(config)


def create_integrated_ml_trainer(model: IntegratedMLArchitecture, config: Optional[IntegratedMLConfig] = None) -> IntegratedMLTrainer:
    """Create integrated ML trainer."""
    if config is None:
        config = IntegratedMLConfig()
    
    return IntegratedMLTrainer(model, config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Integrated ML Architecture')
    
    # Test configuration
    config = IntegratedMLConfig(
        input_features=50,
        sequence_length=60,
        forecast_horizon=12,
        use_hpo=False  # Disable HPO for testing
    )
    
    tprint(f'📊 Integrated ML Configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Sequence length: {config.sequence_length}')
    tprint(f'   → Forecast horizon: {config.forecast_horizon}')
    tprint(f'   → Use CLVSA: {config.use_clvsa}')
    tprint(f'   → Use MultiScaleNBEATS: {config.use_multiscale_nbeats}')
    tprint(f'   → Use RegimeNAS: {config.use_regime_nas}')
    tprint(f'   → Use Meta-labels: {config.use_meta_labels}')
    tprint(f'   → Use HPO: {config.use_hpo}')
    tprint(f'   → Ensemble method: {config.ensemble_method}')
    
    # Test model creation
    try:
        model = create_integrated_ml_architecture(config)
        trainer = create_integrated_ml_trainer(model, config)
        
        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, config.sequence_length, config.input_features)
        test_regime_ids = torch.randint(0, 3, (batch_size,))
        
        with torch.no_grad():
            outputs = model(test_input, test_regime_ids)
        
        tprint('✅ Integrated ML architecture created successfully')
        tprint(f'   → Output keys: {list(outputs.keys())}')
        tprint(f'   → Prediction shape: {outputs["prediction"].shape}')
        tprint(f'   → Uncertainty shape: {outputs["uncertainty"].shape}')
        tprint(f'   → Regime prediction shape: {outputs["regime_prediction"].shape}')
        tprint(f'   → Component outputs: {list(outputs["component_outputs"].keys())}')
        
        # Test loss computation
        test_targets = {
            'prediction': torch.randn(batch_size, config.forecast_horizon),
            'uncertainty': torch.randn(batch_size, 1),
            'regime': test_regime_ids,
            'regime_ids': test_regime_ids
        }
        
        losses = model.compute_loss(outputs, test_targets)
        tprint(f'   → Loss components: {list(losses.keys())}')
        tprint(f'   → Total loss: {losses["total_loss"].item():.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating integrated ML architecture: {e}')
    
    tprint('✅ Integrated ML Architecture test completed!')