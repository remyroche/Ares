"""
Multi-Horizon Model Architecture - Updated for Multi-Horizon Profit Labeling

This module provides model architectures optimized for multi-horizon profit probability
prediction, replacing binary classification with multi-output regression.

Key features:
- Multi-output regression (20+ probability targets)
- Specialized loss functions for probability prediction
- Architecture optimized for profit probability learning
- Support for both individual and ensemble models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging

# Optimized imports using common utilities
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context,
    integrate_with_m1_optimizers, create_m1_optimized_array
)
from src.utils.math_validation import (
    validate_finite, validate_positive, validate_range,
    safe_matrix_inverse, validate_correlation_matrix
)
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager, create_m1_optimized_array
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer

# ML Framework imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class MultiHorizonModelConfig:
    """Configuration for multi-horizon model architecture."""
    
    # Input/Output dimensions
    input_features: int = 50  # Number of input features from 5m data
    
    # Target configuration (matches MultiHorizonConfig) - SHORT-TERM FOCUSED
    profit_targets: List[str] = field(default_factory=lambda: [
        'micro', 'small', 'medium', 'good'
    ])
    time_horizons: List[str] = field(default_factory=lambda: [
        'immediate', 'short'
    ])
    
    # Composite targets (SHORT-TERM FOCUSED)
    composite_targets: List[str] = field(default_factory=lambda: [
        'immediate_opportunity', 'short_term_opportunity',
        'overall_opportunity', 'leverage_adjusted_score',
        'reversal_capture_score', 'reassessment_frequency'
    ])
    
    # Model architecture
    model_type: str = 'neural_network'  # 'neural_network', 'ensemble', 'xgboost'
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2
    activation: str = 'relu'
    output_activation: str = 'sigmoid'  # For probabilities [0,1]
    
    # Training configuration
    loss_function: str = 'mse'  # 'mse', 'mae', 'huber', 'custom_probability'
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    validation_split: float = 0.2
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    early_stopping_patience: int = 15
    
    # Multi-output specific
    output_weights: Optional[Dict[str, float]] = None  # Custom weights for different outputs
    shared_layers: int = 2  # Number of shared layers before branching
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Total number of probability outputs
        self.num_probability_outputs = len(self.profit_targets) * len(self.time_horizons)
        
        # Total number of outputs (probabilities + composites)
        self.total_outputs = self.num_probability_outputs + len(self.composite_targets)
        
        # Default output weights (equal weighting)
        if self.output_weights is None:
            self.output_weights = {
                'probabilities': 0.7,  # Higher weight on individual probabilities
                'composites': 0.3      # Lower weight on composite scores
            }

class MultiHorizonModelBuilder:
    """
    Builder for multi-horizon profit probability prediction models.
    """
    
    def __init__(self, config: MultiHorizonModelConfig):
        """Initialize the model builder with hardware optimizations."""
        self.config = config
        self.logger = get_logger('MultiHorizonModelBuilder')
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Optimize CPU for mathematical operations
        if self.cpu_optimizer:
            self.cpu_optimizer.optimize_numpy_operations()
        
        # Validate framework availability
        self._check_framework_availability()
        
        self.logger.info(f'🏗️ Multi-Horizon Model Builder initialized with M1 optimizations')
        self.logger.info(f'   → Input features: {self.config.input_features}')
        self.logger.info(f'   → Probability outputs: {self.config.num_probability_outputs}')
        self.logger.info(f'   → Total outputs: {self.config.total_outputs}')
        self.logger.info(f'   → Model type: {self.config.model_type}')
    
    def _check_framework_availability(self):
        """Check which ML frameworks are available."""
        available_frameworks = []
        if TENSORFLOW_AVAILABLE:
            available_frameworks.append('tensorflow')
        if PYTORCH_AVAILABLE:
            available_frameworks.append('pytorch')
        if SKLEARN_AVAILABLE:
            available_frameworks.append('sklearn')
        
        if not available_frameworks:
            raise ImportError("No ML frameworks available. Install tensorflow, pytorch, or sklearn.")
        
        self.logger.info(f'📦 Available frameworks: {available_frameworks}')
    
    @timed_operation
    @traced(span_name='build_model')
    @validates()
    @handles_errors(exceptions=(Exception,), default_return=None)
    def build_model(self, framework: str = 'auto') -> Any:
        """
        Build multi-horizon model based on specified framework.
        
        Args:
            framework: 'tensorflow', 'pytorch', 'sklearn', or 'auto'
            
        Returns:
            Compiled model ready for training
        """
        if framework == 'auto':
            framework = self._select_best_framework()
        
        self.logger.info(f'🔨 Building {self.config.model_type} model with {framework}')
        
        # Use memory checkpoint for model building
        with memory_checkpoint('model_building'):
            if framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
                with gpu_context('tensorflow_model') if self.gpu_manager else memory_checkpoint('tensorflow_model'):
                    return self._build_tensorflow_model()
            elif framework == 'pytorch' and PYTORCH_AVAILABLE:
                with gpu_context('pytorch_model') if self.gpu_manager else memory_checkpoint('pytorch_model'):
                    return self._build_pytorch_model()
            elif framework == 'sklearn' and SKLEARN_AVAILABLE:
                with memory_checkpoint('sklearn_model'):
                    return self._build_sklearn_model()
            else:
                raise ValueError(f"Framework {framework} not available or supported")
    
    def _select_best_framework(self) -> str:
        """Select the best available framework for the model type."""
        if self.config.model_type == 'neural_network':
            if TENSORFLOW_AVAILABLE:
                return 'tensorflow'
            elif PYTORCH_AVAILABLE:
                return 'pytorch'
            elif SKLEARN_AVAILABLE:
                return 'sklearn'
        elif self.config.model_type in ['ensemble', 'xgboost']:
            if SKLEARN_AVAILABLE:
                return 'sklearn'
        
        # Fallback to first available
        if TENSORFLOW_AVAILABLE:
            return 'tensorflow'
        elif SKLEARN_AVAILABLE:
            return 'sklearn'
        elif PYTORCH_AVAILABLE:
            return 'pytorch'
        
        raise RuntimeError("No suitable framework available")
    
    def _build_tensorflow_model(self) -> Model:
        """Build TensorFlow/Keras model."""
        # Input layer
        inputs = keras.Input(shape=(self.config.input_features,), name='features')
        
        # Shared layers
        x = inputs
        for i, units in enumerate(self.config.hidden_layers[:self.config.shared_layers]):
            x = layers.Dense(
                units, 
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_reg, 
                    l2=self.config.l2_reg
                ),
                name=f'shared_dense_{i}'
            )(x)
            x = layers.Dropout(self.config.dropout_rate, name=f'shared_dropout_{i}')(x)
        
        # Branch for probability outputs
        prob_branch = x
        for i, units in enumerate(self.config.hidden_layers[self.config.shared_layers:]):
            prob_branch = layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_reg,
                    l2=self.config.l2_reg
                ),
                name=f'prob_dense_{i}'
            )(prob_branch)
            prob_branch = layers.Dropout(self.config.dropout_rate, name=f'prob_dropout_{i}')(prob_branch)
        
        # Individual probability outputs
        probability_outputs = []
        for target in self.config.profit_targets:
            for horizon in self.config.time_horizons:
                output_name = f'{target}_{horizon}_prob'
                prob_output = layers.Dense(
                    1, 
                    activation=self.config.output_activation,
                    name=output_name
                )(prob_branch)
                probability_outputs.append(prob_output)
        
        # Branch for composite outputs
        composite_branch = x
        for i, units in enumerate(self.config.hidden_layers[self.config.shared_layers:]):
            composite_branch = layers.Dense(
                units,
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_reg,
                    l2=self.config.l2_reg
                ),
                name=f'composite_dense_{i}'
            )(composite_branch)
            composite_branch = layers.Dropout(self.config.dropout_rate, name=f'composite_dropout_{i}')(composite_branch)
        
        # Composite outputs
        composite_outputs = []
        for composite_name in self.config.composite_targets:
            composite_output = layers.Dense(
                1,
                activation=self.config.output_activation,
                name=composite_name
            )(composite_branch)
            composite_outputs.append(composite_output)
        
        # Combine all outputs
        all_outputs = probability_outputs + composite_outputs
        
        # Create model
        model = Model(inputs=inputs, outputs=all_outputs, name='multi_horizon_profit_model')
        
        # Compile model
        self._compile_tensorflow_model(model)
        
        return model
    
    def _compile_tensorflow_model(self, model: Model):
        """Compile TensorFlow model with appropriate loss and metrics."""
        # Loss function
        if self.config.loss_function == 'mse':
            loss = 'mse'
        elif self.config.loss_function == 'mae':
            loss = 'mae'
        elif self.config.loss_function == 'huber':
            loss = keras.losses.Huber()
        elif self.config.loss_function == 'custom_probability':
            loss = self._custom_probability_loss
        else:
            loss = 'mse'
        
        # Optimizer
        if self.config.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        else:
            optimizer = 'adam'
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['mae', 'mse']
        )
        
        self.logger.info(f'✅ TensorFlow model compiled with {loss} loss and {optimizer} optimizer')
    
    def _custom_probability_loss(self, y_true, y_pred):
        """Custom loss function optimized for probability prediction."""
        # Ensure predictions are in [0,1] range
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Binary cross-entropy component for probability-like behavior
        bce = -y_true * tf.math.log(y_pred_clipped) - (1 - y_true) * tf.math.log(1 - y_pred_clipped)
        
        # MSE component for regression-like behavior
        mse = tf.square(y_true - y_pred)
        
        # Combine both (weighted)
        return 0.7 * bce + 0.3 * mse
    
    def _build_sklearn_model(self) -> Any:
        """Build scikit-learn model."""
        if self.config.model_type == 'neural_network':
            base_model = MLPRegressor(
                hidden_layer_sizes=tuple(self.config.hidden_layers),
                activation=self.config.activation.replace('relu', 'relu'),
                alpha=self.config.l2_reg,
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.epochs,
                early_stopping=True,
                validation_fraction=self.config.validation_split,
                n_iter_no_change=self.config.early_stopping_patience,
                random_state=42
            )
        elif self.config.model_type == 'ensemble':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Default to random forest
            base_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        # Wrap in MultiOutputRegressor for multiple targets
        model = MultiOutputRegressor(base_model, n_jobs=-1)
        
        self.logger.info(f'✅ Sklearn model created: {type(base_model).__name__} with MultiOutputRegressor')
        
        return model
    
    def _build_pytorch_model(self) -> nn.Module:
        """Build PyTorch model."""
        
        class MultiHorizonNet(nn.Module):
            def __init__(self, config: MultiHorizonModelConfig):
                super().__init__()
                self.config = config
                
                # Shared layers
                shared_layers = []
                input_dim = config.input_features
                
                for units in config.hidden_layers[:config.shared_layers]:
                    shared_layers.extend([
                        nn.Linear(input_dim, units),
                        nn.ReLU() if config.activation == 'relu' else nn.Tanh(),
                        nn.Dropout(config.dropout_rate)
                    ])
                    input_dim = units
                
                self.shared = nn.Sequential(*shared_layers)
                
                # Probability branch
                prob_layers = []
                prob_input_dim = input_dim
                
                for units in config.hidden_layers[config.shared_layers:]:
                    prob_layers.extend([
                        nn.Linear(prob_input_dim, units),
                        nn.ReLU() if config.activation == 'relu' else nn.Tanh(),
                        nn.Dropout(config.dropout_rate)
                    ])
                    prob_input_dim = units
                
                self.prob_branch = nn.Sequential(*prob_layers)
                self.prob_outputs = nn.Linear(prob_input_dim, config.num_probability_outputs)
                
                # Composite branch
                composite_layers = []
                comp_input_dim = input_dim
                
                for units in config.hidden_layers[config.shared_layers:]:
                    composite_layers.extend([
                        nn.Linear(comp_input_dim, units),
                        nn.ReLU() if config.activation == 'relu' else nn.Tanh(),
                        nn.Dropout(config.dropout_rate)
                    ])
                    comp_input_dim = units
                
                self.composite_branch = nn.Sequential(*composite_layers)
                self.composite_outputs = nn.Linear(comp_input_dim, len(config.composite_targets))
            
            def forward(self, x):
                # Shared processing
                shared_features = self.shared(x)
                
                # Probability outputs
                prob_features = self.prob_branch(shared_features)
                prob_outputs = torch.sigmoid(self.prob_outputs(prob_features))
                
                # Composite outputs  
                comp_features = self.composite_branch(shared_features)
                comp_outputs = torch.sigmoid(self.composite_outputs(comp_features))
                
                # Concatenate all outputs
                return torch.cat([prob_outputs, comp_outputs], dim=1)
        
        model = MultiHorizonNet(self.config)
        
        self.logger.info(f'✅ PyTorch model created with {sum(p.numel() for p in model.parameters())} parameters')
        
        return model
    
    def get_target_names(self) -> List[str]:
        """Get ordered list of target names for the model outputs."""
        target_names = []
        
        # Individual probability targets
        for target in self.config.profit_targets:
            for horizon in self.config.time_horizons:
                target_names.append(f'{target}_{horizon}_prob')
        
        # Composite targets
        target_names.extend(self.config.composite_targets)
        
        return target_names
    
    def create_training_callbacks(self, framework: str = 'tensorflow') -> List[Any]:
        """Create training callbacks for the specified framework."""
        callbacks = []
        
        if framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Learning rate reduction
            lr_reducer = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config.early_stopping_patience // 2,
                min_lr=self.config.learning_rate * 0.01,
                verbose=1
            )
            callbacks.append(lr_reducer)
        
        return callbacks

# Convenience functions
def create_multi_horizon_model(config: Optional[MultiHorizonModelConfig] = None,
                             framework: str = 'auto') -> Any:
    """Create multi-horizon profit probability model."""
    if config is None:
        config = MultiHorizonModelConfig()
    
    builder = MultiHorizonModelBuilder(config)
    return builder.build_model(framework)

def get_model_target_names(config: Optional[MultiHorizonModelConfig] = None) -> List[str]:
    """Get target names for multi-horizon model."""
    if config is None:
        config = MultiHorizonModelConfig()
    
    builder = MultiHorizonModelBuilder(config)
    return builder.get_target_names()

# Test function
if __name__ == '__main__':
    tprint('🧪 Testing Multi-Horizon Model Architecture')
    
    # Test configuration
    config = MultiHorizonModelConfig(
        input_features=50,
        hidden_layers=[128, 64, 32],
        model_type='neural_network'
    )
    
    tprint(f'📊 Model configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Probability outputs: {config.num_probability_outputs}')
    tprint(f'   → Total outputs: {config.total_outputs}')
    
    # Test model creation
    try:
        builder = MultiHorizonModelBuilder(config)
        model = builder.build_model('auto')
        
        if model is not None:
            tprint('✅ Model created successfully')
            
            # Get target names
            target_names = builder.get_target_names()
            tprint(f'📝 Target names ({len(target_names)}):')
            for i, name in enumerate(target_names[:5]):  # Show first 5
                tprint(f'   → {i+1}: {name}')
            if len(target_names) > 5:
                tprint(f'   → ... and {len(target_names)-5} more')
        else:
            tprint('❌ Model creation failed')
            
    except Exception as e:
        tprint(f'❌ Error creating model: {e}')
    
    tprint('✅ Multi-Horizon Model Architecture test completed!')