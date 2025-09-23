"""
Regime-Specific Neural Network Architectures

This module implements specialized neural network architectures optimized for different
types of regime detection in financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
import time

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

# Import matrix operations for optimized computations
from src.utils.matrix_operations import UnifiedMatrixOperations

# Import hardware optimization
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)

from .temporal_layers import (
    TemporalConvolutionLayer, RegimeLSTMLayer, RegimeGRULayer,
    MultiScaleTemporalLayer, TemporalAttentionLayer
)
from .attention_mechanisms import (
    RegimeAttention, MultiHeadRegimeAttention, TemporalAttention
)

logger = logging.getLogger(__name__)


class BaseRegimeNetwork(ABC):
    """Abstract base class for regime detection networks."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize base regime network."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.matrix_ops = matrix_ops
        self.hardware_manager = hardware_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance tracking
        self.training_history = []
        self.inference_times = []
        
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the network architecture."""
        pass
    
    def train_network(self, data: np.ndarray, labels: np.ndarray, 
                     epochs: int = 100, batch_size: int = 32,
                     learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train the network on regime detection data."""
        try:
            if not TORCH_AVAILABLE:
                self.logger.warning("PyTorch not available, using simplified training")
                return self._simplified_training(data, labels)
            
            start_time = time.time()
            
            # Start hardware optimization
            if self.hardware_manager:
                self.hardware_manager.start_optimization(
                    workload_type=WorkloadType.ML_TRAINING,
                    optimization_level=OptimizationLevel.BALANCED
                )
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(data).to(self.device)
            y_tensor = torch.LongTensor(labels).to(self.device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training setup
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            training_losses = []
            training_accuracies = []
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                num_batches = 0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.forward(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    epoch_accuracy += (predicted == batch_y).float().mean().item()
                    num_batches += 1
                
                # Average statistics
                avg_loss = epoch_loss / num_batches
                avg_accuracy = epoch_accuracy / num_batches
                
                training_losses.append(avg_loss)
                training_accuracies.append(avg_accuracy)
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
            
            training_time = time.time() - start_time
            
            # Record training history
            training_result = {
                'final_loss': training_losses[-1],
                'final_accuracy': training_accuracies[-1],
                'best_accuracy': max(training_accuracies),
                'training_time': training_time,
                'epochs': epochs,
                'training_losses': training_losses,
                'training_accuracies': training_accuracies
            }
            
            self.training_history.append(training_result)
            
            self.logger.info(f"✅ Training completed in {training_time:.2f}s")
            self.logger.info(f"   Final accuracy: {training_result['final_accuracy']:.4f}")
            self.logger.info(f"   Best accuracy: {training_result['best_accuracy']:.4f}")
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Network training failed: {e}")
            return {'error': str(e), 'final_accuracy': 0.0}
        
        finally:
            # Stop hardware optimization
            if self.hardware_manager:
                self.hardware_manager.stop_optimization()
    
    def _simplified_training(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Simplified training without PyTorch."""
        try:
            start_time = time.time()
            
            # Use matrix operations for simplified training
            if self.matrix_ops:
                # Simple linear regression approximation
                X = self.matrix_ops.matrix_normalize(data)
                y = labels
                
                # Compute pseudo-inverse for weights
                weights = self.matrix_ops.matrix_solve(X, y)
                
                # Compute predictions
                predictions = self.matrix_ops.matrix_multiply(X, weights)
                
                # Compute accuracy (simplified)
                predicted_labels = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
                accuracy = np.mean(predicted_labels == y)
                
                training_time = time.time() - start_time
                
                return {
                    'final_accuracy': accuracy,
                    'best_accuracy': accuracy,
                    'training_time': training_time,
                    'epochs': 1,
                    'simplified_training': True
                }
            else:
                # Fallback to random predictions
                accuracy = 1.0 / len(np.unique(labels))  # Random accuracy
                training_time = time.time() - start_time
                
                return {
                    'final_accuracy': accuracy,
                    'best_accuracy': accuracy,
                    'training_time': training_time,
                    'epochs': 1,
                    'simplified_training': True,
                    'fallback_training': True
                }
                
        except Exception as e:
            self.logger.warning(f"Simplified training failed: {e}")
            return {
                'final_accuracy': 0.0,
                'best_accuracy': 0.0,
                'training_time': 0.0,
                'epochs': 0,
                'error': str(e)
            }
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        try:
            start_time = time.time()
            
            if not TORCH_AVAILABLE:
                return self._simplified_predict(data)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(data).to(self.device)
            
            # Set to evaluation mode
            self.eval()
            
            with torch.no_grad():
                outputs = self.forward(X_tensor)
                predictions = torch.argmax(outputs, dim=1)
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            return predictions.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return random predictions as fallback
            return np.random.randint(0, self.output_dim, size=len(data))
    
    def _simplified_predict(self, data: np.ndarray) -> np.ndarray:
        """Simplified prediction without PyTorch."""
        try:
            # Return random predictions
            return np.random.randint(0, self.output_dim, size=len(data))
        except Exception as e:
            self.logger.warning(f"Simplified prediction failed: {e}")
            return np.zeros(len(data), dtype=int)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        try:
            stats = {
                'training_history_length': len(self.training_history),
                'inference_times_count': len(self.inference_times)
            }
            
            if self.training_history:
                latest_training = self.training_history[-1]
                stats.update({
                    'latest_training_accuracy': latest_training.get('final_accuracy', 0.0),
                    'best_training_accuracy': latest_training.get('best_accuracy', 0.0),
                    'latest_training_time': latest_training.get('training_time', 0.0)
                })
            
            if self.inference_times:
                stats.update({
                    'average_inference_time': np.mean(self.inference_times),
                    'median_inference_time': np.median(self.inference_times),
                    'max_inference_time': np.max(self.inference_times),
                    'min_inference_time': np.min(self.inference_times)
                })
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Performance stats calculation failed: {e}")
            return {}


class VolatilityRegimeNetwork(BaseRegimeNetwork):
    """Neural network specialized for volatility regime detection."""
    
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize volatility regime network."""
        super().__init__(input_dim, output_dim, matrix_ops, hardware_manager)
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        if TORCH_AVAILABLE:
            self._build_torch_network()
        else:
            self._build_simplified_network()
    
    def _build_torch_network(self):
        """Build PyTorch network architecture."""
        layers = []
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers with LSTM for temporal patterns
        for i, hidden_dim in enumerate(self.hidden_dims):
            if i == 0:
                # First layer: LSTM for temporal volatility patterns
                layers.append(RegimeLSTMLayer(prev_dim, hidden_dim, dropout=self.dropout_rate))
            else:
                # Subsequent layers: Dense with attention
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.network = self.network.to(self.device)
    
    def _build_simplified_network(self):
        """Build simplified network without PyTorch."""
        self.network = None
        self.logger.info("Built simplified volatility network (no PyTorch)")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through volatility network."""
        try:
            if TORCH_AVAILABLE and self.network is not None:
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                return self.network(x)
            else:
                return self._simplified_forward(x)
        except Exception as e:
            self.logger.error(f"Volatility network forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def _simplified_forward(self, x: np.ndarray) -> np.ndarray:
        """Simplified forward pass without PyTorch."""
        try:
            # Simple linear transformation
            if self.matrix_ops:
                # Use matrix operations for computation
                normalized_x = self.matrix_ops.matrix_normalize(x)
                # Simple projection to output dimension
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                output = self.matrix_ops.matrix_multiply(normalized_x, weights)
                return output
            else:
                # Fallback to numpy
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                return np.dot(x, weights)
        except Exception as e:
            self.logger.warning(f"Simplified forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get volatility network architecture information."""
        return {
            'network_type': 'volatility_regime',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'total_layers': len(self.hidden_dims) + 2,
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(self.device) if self.device else None
        }


class TrendRegimeNetwork(BaseRegimeNetwork):
    """Neural network specialized for trend regime detection."""
    
    def __init__(self, input_dim: int, output_dim: int,
                 conv_filters: List[int] = [64, 128, 256],
                 dense_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.3,
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize trend regime network."""
        super().__init__(input_dim, output_dim, matrix_ops, hardware_manager)
        
        self.conv_filters = conv_filters
        self.dense_dims = dense_dims
        self.dropout_rate = dropout_rate
        
        if TORCH_AVAILABLE:
            self._build_torch_network()
        else:
            self._build_simplified_network()
    
    def _build_torch_network(self):
        """Build PyTorch network architecture."""
        layers = []
        
        # Convolutional layers for trend patterns
        prev_channels = 1  # Assuming single channel input
        
        for i, filters in enumerate(self.conv_filters):
            layers.append(nn.Conv1d(prev_channels, filters, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(self.dropout_rate))
            prev_channels = filters
        
        # Global pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        # Dense layers
        prev_dim = prev_channels
        for dense_dim in self.dense_dims:
            layers.append(nn.Linear(prev_dim, dense_dim))
            layers.append(nn.BatchNorm1d(dense_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = dense_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.network = self.network.to(self.device)
    
    def _build_simplified_network(self):
        """Build simplified network without PyTorch."""
        self.network = None
        self.logger.info("Built simplified trend network (no PyTorch)")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through trend network."""
        try:
            if TORCH_AVAILABLE and self.network is not None:
                if isinstance(x, np.ndarray):
                    # Reshape for conv1d: (batch, channels, sequence)
                    if x.ndim == 2:
                        x = x.unsqueeze(1) if hasattr(x, 'unsqueeze') else np.expand_dims(x, 1)
                        x = torch.FloatTensor(x).to(self.device)
                    return self.network(x)
                else:
                    return self.network(x)
            else:
                return self._simplified_forward(x)
        except Exception as e:
            self.logger.error(f"Trend network forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def _simplified_forward(self, x: np.ndarray) -> np.ndarray:
        """Simplified forward pass without PyTorch."""
        try:
            if self.matrix_ops:
                normalized_x = self.matrix_ops.matrix_normalize(x)
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                output = self.matrix_ops.matrix_multiply(normalized_x, weights)
                return output
            else:
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                return np.dot(x, weights)
        except Exception as e:
            self.logger.warning(f"Simplified forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get trend network architecture information."""
        return {
            'network_type': 'trend_regime',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'conv_filters': self.conv_filters,
            'dense_dims': self.dense_dims,
            'dropout_rate': self.dropout_rate,
            'total_layers': len(self.conv_filters) + len(self.dense_dims) + 3,
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(self.device) if self.device else None
        }


class VolumeRegimeNetwork(BaseRegimeNetwork):
    """Neural network specialized for volume regime detection."""
    
    def __init__(self, input_dim: int, output_dim: int,
                 attention_heads: int = 8,
                 attention_dim: int = 64,
                 dense_dims: List[int] = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize volume regime network."""
        super().__init__(input_dim, output_dim, matrix_ops, hardware_manager)
        
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.dense_dims = dense_dims
        self.dropout_rate = dropout_rate
        
        if TORCH_AVAILABLE:
            self._build_torch_network()
        else:
            self._build_simplified_network()
    
    def _build_torch_network(self):
        """Build PyTorch network architecture."""
        layers = []
        
        # Input projection
        layers.append(nn.Linear(self.input_dim, self.attention_dim))
        layers.append(nn.LayerNorm(self.attention_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        
        # Multi-head attention for volume patterns
        layers.append(MultiHeadRegimeAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            dropout=self.dropout_rate
        ))
        
        # Dense layers
        prev_dim = self.attention_dim
        for dense_dim in self.dense_dims:
            layers.append(nn.Linear(prev_dim, dense_dim))
            layers.append(nn.LayerNorm(dense_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = dense_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        self.network = self.network.to(self.device)
    
    def _build_simplified_network(self):
        """Build simplified network without PyTorch."""
        self.network = None
        self.logger.info("Built simplified volume network (no PyTorch)")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through volume network."""
        try:
            if TORCH_AVAILABLE and self.network is not None:
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                return self.network(x)
            else:
                return self._simplified_forward(x)
        except Exception as e:
            self.logger.error(f"Volume network forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def _simplified_forward(self, x: np.ndarray) -> np.ndarray:
        """Simplified forward pass without PyTorch."""
        try:
            if self.matrix_ops:
                normalized_x = self.matrix_ops.matrix_normalize(x)
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                output = self.matrix_ops.matrix_multiply(normalized_x, weights)
                return output
            else:
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                return np.dot(x, weights)
        except Exception as e:
            self.logger.warning(f"Simplified forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get volume network architecture information."""
        return {
            'network_type': 'volume_regime',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'attention_heads': self.attention_heads,
            'attention_dim': self.attention_dim,
            'dense_dims': self.dense_dims,
            'dropout_rate': self.dropout_rate,
            'total_layers': len(self.dense_dims) + 4,
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(self.device) if self.device else None
        }


class HybridRegimeNetwork(BaseRegimeNetwork):
    """Hybrid neural network combining multiple regime detection approaches."""
    
    def __init__(self, input_dim: int, output_dim: int,
                 volatility_branch_dims: List[int] = [128, 64],
                 trend_branch_dims: List[int] = [128, 64],
                 volume_branch_dims: List[int] = [128, 64],
                 fusion_dim: int = 64,
                 dropout_rate: float = 0.2,
                 matrix_ops: Optional[UnifiedMatrixOperations] = None,
                 hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize hybrid regime network."""
        super().__init__(input_dim, output_dim, matrix_ops, hardware_manager)
        
        self.volatility_branch_dims = volatility_branch_dims
        self.trend_branch_dims = trend_branch_dims
        self.volume_branch_dims = volume_branch_dims
        self.fusion_dim = fusion_dim
        self.dropout_rate = dropout_rate
        
        if TORCH_AVAILABLE:
            self._build_torch_network()
        else:
            self._build_simplified_network()
    
    def _build_torch_network(self):
        """Build PyTorch hybrid network architecture."""
        # Volatility branch (LSTM-based)
        volatility_layers = []
        prev_dim = self.input_dim
        for dim in self.volatility_branch_dims:
            volatility_layers.append(RegimeLSTMLayer(prev_dim, dim, dropout=self.dropout_rate))
            prev_dim = dim
        
        self.volatility_branch = nn.Sequential(*volatility_layers)
        
        # Trend branch (CNN-based)
        trend_layers = []
        trend_layers.append(nn.Conv1d(1, 64, kernel_size=3, padding=1))
        trend_layers.append(nn.ReLU())
        trend_layers.append(nn.MaxPool1d(2))
        trend_layers.append(nn.Dropout(self.dropout_rate))
        
        prev_dim = 64
        for dim in self.trend_branch_dims:
            trend_layers.append(nn.Linear(prev_dim, dim))
            trend_layers.append(nn.ReLU())
            trend_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = dim
        
        self.trend_branch = nn.Sequential(*trend_layers)
        
        # Volume branch (Attention-based)
        volume_layers = []
        volume_layers.append(MultiHeadRegimeAttention(
            embed_dim=self.input_dim,
            num_heads=8,
            dropout=self.dropout_rate
        ))
        
        prev_dim = self.input_dim
        for dim in self.volume_branch_dims:
            volume_layers.append(nn.Linear(prev_dim, dim))
            volume_layers.append(nn.ReLU())
            volume_layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = dim
        
        self.volume_branch = nn.Sequential(*volume_layers)
        
        # Fusion layer
        total_branch_dim = (self.volatility_branch_dims[-1] + 
                           self.trend_branch_dims[-1] + 
                           self.volume_branch_dims[-1])
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_branch_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fusion_dim, self.output_dim)
        )
        
        # Move to device
        self.volatility_branch = self.volatility_branch.to(self.device)
        self.trend_branch = self.trend_branch.to(self.device)
        self.volume_branch = self.volume_branch.to(self.device)
        self.fusion_layer = self.fusion_layer.to(self.device)
    
    def _build_simplified_network(self):
        """Build simplified network without PyTorch."""
        self.volatility_branch = None
        self.trend_branch = None
        self.volume_branch = None
        self.fusion_layer = None
        self.logger.info("Built simplified hybrid network (no PyTorch)")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through hybrid network."""
        try:
            if TORCH_AVAILABLE and self.volatility_branch is not None:
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x).to(self.device)
                
                # Forward through branches
                volatility_out = self.volatility_branch(x)
                trend_out = self.trend_branch(x.unsqueeze(1))  # Add channel dimension
                volume_out = self.volume_branch(x)
                
                # Concatenate branch outputs
                fused = torch.cat([volatility_out, trend_out, volume_out], dim=1)
                
                # Fusion layer
                output = self.fusion_layer(fused)
                
                return output
            else:
                return self._simplified_forward(x)
        except Exception as e:
            self.logger.error(f"Hybrid network forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def _simplified_forward(self, x: np.ndarray) -> np.ndarray:
        """Simplified forward pass without PyTorch."""
        try:
            if self.matrix_ops:
                normalized_x = self.matrix_ops.matrix_normalize(x)
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                output = self.matrix_ops.matrix_multiply(normalized_x, weights)
                return output
            else:
                weights = np.random.randn(self.input_dim, self.output_dim) * 0.1
                return np.dot(x, weights)
        except Exception as e:
            self.logger.warning(f"Simplified forward pass failed: {e}")
            return np.zeros((len(x), self.output_dim))
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get hybrid network architecture information."""
        return {
            'network_type': 'hybrid_regime',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'volatility_branch_dims': self.volatility_branch_dims,
            'trend_branch_dims': self.trend_branch_dims,
            'volume_branch_dims': self.volume_branch_dims,
            'fusion_dim': self.fusion_dim,
            'dropout_rate': self.dropout_rate,
            'total_layers': (len(self.volatility_branch_dims) + 
                           len(self.trend_branch_dims) + 
                           len(self.volume_branch_dims) + 3),
            'pytorch_available': TORCH_AVAILABLE,
            'device': str(self.device) if self.device else None
        }


class RegimeNetworkFactory:
    """Factory for creating regime-specific neural networks."""
    
    @staticmethod
    def create_network(network_type: str, input_dim: int, output_dim: int,
                      matrix_ops: Optional[UnifiedMatrixOperations] = None,
                      hardware_manager: Optional[UnifiedHardwareManager] = None,
                      **kwargs) -> BaseRegimeNetwork:
        """Create a regime-specific neural network."""
        try:
            if network_type.lower() == 'volatility':
                return VolatilityRegimeNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    matrix_ops=matrix_ops,
                    hardware_manager=hardware_manager,
                    **kwargs
                )
            elif network_type.lower() == 'trend':
                return TrendRegimeNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    matrix_ops=matrix_ops,
                    hardware_manager=hardware_manager,
                    **kwargs
                )
            elif network_type.lower() == 'volume':
                return VolumeRegimeNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    matrix_ops=matrix_ops,
                    hardware_manager=hardware_manager,
                    **kwargs
                )
            elif network_type.lower() == 'hybrid':
                return HybridRegimeNetwork(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    matrix_ops=matrix_ops,
                    hardware_manager=hardware_manager,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown network type: {network_type}")
                
        except Exception as e:
            logger.error(f"Network creation failed: {e}")
            # Return a default hybrid network
            return HybridRegimeNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                matrix_ops=matrix_ops,
                hardware_manager=hardware_manager
            )
    
    @staticmethod
    def get_available_network_types() -> List[str]:
        """Get list of available network types."""
        return ['volatility', 'trend', 'volume', 'hybrid']
    
    @staticmethod
    def get_network_recommendations(input_dim: int, output_dim: int, 
                                  data_characteristics: Dict[str, Any]) -> List[str]:
        """Get network type recommendations based on data characteristics."""
        recommendations = []
        
        try:
            # Check for volatility patterns
            if data_characteristics.get('high_volatility', False):
                recommendations.append('volatility')
            
            # Check for trend patterns
            if data_characteristics.get('strong_trends', False):
                recommendations.append('trend')
            
            # Check for volume patterns
            if data_characteristics.get('volume_important', False):
                recommendations.append('volume')
            
            # Always include hybrid as a fallback
            recommendations.append('hybrid')
            
            # If no specific characteristics, recommend all types
            if not recommendations:
                recommendations = ['volatility', 'trend', 'volume', 'hybrid']
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Network recommendations failed: {e}")
            return ['hybrid']  # Default fallback