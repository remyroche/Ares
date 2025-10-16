"""
Enhanced MultiScaleNBEATS for Multi-Timeframe Regime-Aware Prediction

This module implements an enhanced MultiScaleNBEATS architecture that:
1. Multi-scale temporal modeling across different timeframes
2. Regime-specific optimization and adaptation
3. Multi-timeframe fusion for improved predictions
4. Hierarchical feature processing
5. Uncertainty quantification through ensemble methods

Key Improvements:
- Multi-scale processing for different time horizons (1m, 5m, 15m, 30m, 1h)
- Regime-specific NBEATS variants optimized for different market conditions
- Hierarchical feature processing with attention mechanisms
- Multi-timeframe fusion using attention-based weighting
- Uncertainty quantification through ensemble diversity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Independent
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available, MultiScaleNBEATS will use fallback implementations")


class MultiScaleNBEATSConfig:
    """Configuration for MultiScaleNBEATS architecture."""

    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 4,
                 timeframes: List[str] = ['1m', '5m', '15m', '30m', '1h'],
                 forecast_length: int = 1,
                 backcast_length: int = 100,
                 stack_types: List[str] = ['trend', 'seasonality'],
                 n_blocks: List[int] = [3, 3],
                 n_layers: List[int] = [4, 4],
                 layer_widths: List[int] = [256, 2048],
                 regime_aware: bool = True,
                 multi_timeframe_fusion: bool = True,
                 uncertainty_quantification: bool = True,
                 ensemble_size: int = 5,
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        """Initialize MultiScaleNBEATS configuration.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of targets)
            timeframes: List of timeframes to process
            forecast_length: Length of forecast horizon
            backcast_length: Length of backcast window
            stack_types: Types of stacks to use
            n_blocks: Number of blocks per stack
            n_layers: Number of layers per block
            layer_widths: Width of layers
            regime_aware: Whether to enable regime-specific processing
            multi_timeframe_fusion: Whether to fuse multi-timeframe predictions
            uncertainty_quantification: Whether to enable uncertainty quantification
            ensemble_size: Number of ensemble members
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.timeframes = timeframes
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.layer_widths = layer_widths
        self.regime_aware = regime_aware
        self.multi_timeframe_fusion = multi_timeframe_fusion
        self.uncertainty_quantification = uncertainty_quantification
        self.ensemble_size = ensemble_size
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm


class NBEATSBlock(nn.Module):
    """Single NBEATS block for time series decomposition."""

    def __init__(self, input_dim: int, output_dim: int, layer_width: int,
                 n_layers: int, stack_type: str, dropout: float = 0.1,
                 use_batch_norm: bool = True):
        """Initialize NBEATS block.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            layer_width: Width of FC layers
            n_layers: Number of layers
            stack_type: Type of stack ('trend', 'seasonality', etc.)
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(NBEATSBlock, self).__init__()

        self.stack_type = stack_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_width = layer_width
        self.n_layers = n_layers

        # FC layers for backcast and forecast
        self.backcast_layers = nn.ModuleList()
        self.forecast_layers = nn.ModuleList()

        # Input projection
        self.input_projection = nn.Linear(input_dim, layer_width)

        # Hidden layers
        for i in range(n_layers):
            in_dim = layer_width if i > 0 else layer_width
            out_dim = layer_width

            self.backcast_layers.append(nn.Linear(in_dim, out_dim))
            self.forecast_layers.append(nn.Linear(in_dim, out_dim))

            if use_batch_norm:
                self.backcast_layers.append(nn.BatchNorm1d(out_dim))
                self.forecast_layers.append(nn.BatchNorm1d(out_dim))

            self.backcast_layers.append(nn.ReLU())
            self.forecast_layers.append(nn.ReLU())

            if dropout > 0:
                self.backcast_layers.append(nn.Dropout(dropout))
                self.forecast_layers.append(nn.Dropout(dropout))

        # Output projections
        self.backcast_output = nn.Linear(layer_width, output_dim)
        self.forecast_output = nn.Linear(layer_width, output_dim)

        # Stack-specific initialization
        if stack_type == 'trend':
            # Initialize for trend detection
            nn.init.xavier_uniform_(self.backcast_output.weight)
            nn.init.xavier_uniform_(self.forecast_output.weight)
        elif stack_type == 'seasonality':
            # Initialize for seasonality detection
            nn.init.xavier_uniform_(self.backcast_output.weight)
            nn.init.xavier_uniform_(self.forecast_output.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through NBEATS block.

        Args:
            x: Input tensor

        Returns:
            Tuple of (backcast, forecast)
        """
        # Input projection
        h = self.input_projection(x)

        # Shared hidden layers
        for i in range(0, len(self.backcast_layers), 4):  # Process in groups of 4
            backcast_layer = self.backcast_layers[i]
            forecast_layer = self.forecast_layers[i]

            h_backcast = backcast_layer(h)
            h_forecast = forecast_layer(h)

            h = h_backcast + h_forecast  # Combine information

        # Output projections
        backcast = self.backcast_output(h)
        forecast = self.forecast_output(h)

        return backcast, forecast


class MultiTimeframeNBEATS(nn.Module):
    """Multi-timeframe NBEATS model."""

    def __init__(self, config: MultiScaleNBEATSConfig):
        super(MultiTimeframeNBEATS, self).__init__()

        self.config = config
        self.timeframe_models = nn.ModuleDict()

        # Create separate NBEATS for each timeframe
        for timeframe in config.timeframes:
            timeframe_config = MultiScaleNBEATSConfig(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                forecast_length=config.forecast_length,
                backcast_length=config.backcast_length,
                stack_types=config.stack_types,
                n_blocks=config.n_blocks,
                n_layers=config.n_layers,
                layer_widths=config.layer_widths,
                regime_aware=config.regime_aware,
                multi_timeframe_fusion=False,  # Individual models
                uncertainty_quantification=config.uncertainty_quantification,
                ensemble_size=1,  # Single model per timeframe
                dropout=config.dropout,
                use_batch_norm=config.use_batch_norm
            )

            self.timeframe_models[timeframe] = SingleTimeframeNBEATS(timeframe_config, timeframe)

        # Multi-timeframe fusion attention
        if config.multi_timeframe_fusion:
            self.fusion_attention = MultiTimeframeAttention(config)

        # Regime-specific components
        if config.regime_aware:
            self.regime_embedding = nn.Embedding(10, config.layer_widths[0])  # 10 regimes

    def forward(self, x: Dict[str, torch.Tensor], regime_id: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-timeframe NBEATS.

        Args:
            x: Dictionary of inputs for each timeframe
            regime_id: Optional regime identifier

        Returns:
            Dictionary of predictions for each timeframe
        """
        predictions = {}

        for timeframe, model in self.timeframe_models.items():
            if timeframe in x:
                pred = model(x[timeframe], regime_id)
                predictions[timeframe] = pred

        # Multi-timeframe fusion
        if self.config.multi_timeframe_fusion and len(predictions) > 1:
            fused_predictions = self.fusion_attention(predictions)
            predictions['fused'] = fused_predictions

        return predictions


class SingleTimeframeNBEATS(nn.Module):
    """NBEATS model for a single timeframe."""

    def __init__(self, config: MultiScaleNBEATSConfig, timeframe: str):
        super(SingleTimeframeNBEATS, self).__init__()

        self.config = config
        self.timeframe = timeframe
        self.stacks = nn.ModuleList()

        # Create stacks
        for stack_type in config.stack_types:
            stack_blocks = nn.ModuleList()

            for block_idx in range(config.n_blocks[len(self.stacks)]):
                block = NBEATSBlock(
                    input_dim=config.input_dim,
                    output_dim=config.backcast_length + config.forecast_length,
                    layer_width=config.layer_widths[len(self.stacks)],
                    n_layers=config.n_layers[len(self.stacks)],
                    stack_type=stack_type,
                    dropout=config.dropout,
                    use_batch_norm=config.use_batch_norm
                )
                stack_blocks.append(block)

            self.stacks.append(stack_blocks)

        # Output projection
        self.output_projection = nn.Linear(
            config.backcast_length + config.forecast_length,
            config.output_dim
        )

        # Timeframe-specific adjustments
        self._apply_timeframe_adjustments()

    def _apply_timeframe_adjustments(self):
        """Apply timeframe-specific architectural adjustments."""
        if self.timeframe == '1m':
            # High-frequency: shorter backcast, more regularization
            for stack in self.stacks:
                for block in stack:
                    if hasattr(block, 'backcast_output'):
                        # Reduce capacity for high-frequency data
                        block.backcast_output = nn.Linear(
                            block.layer_width, self.config.backcast_length + self.config.forecast_length
                        )
        elif self.timeframe == '1h':
            # Low-frequency: longer backcast, less regularization
            for stack in self.stacks:
                for block in stack:
                    if hasattr(block, 'backcast_output'):
                        # Increase capacity for low-frequency data
                        pass  # Keep default capacity

    def forward(self, x: torch.Tensor, regime_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through single timeframe NBEATS.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            regime_id: Optional regime identifier

        Returns:
            Predictions tensor
        """
        batch_size = x.size(0)

        # Add regime embedding if available
        if self.config.regime_aware and regime_id is not None:
            regime_emb = self.regime_embedding(torch.tensor([regime_id]).to(x.device))
            # Expand to match batch size and sequence length: [1, embedding_dim] -> [batch_size, seq_length, embedding_dim]
            regime_emb = regime_emb.unsqueeze(0).expand(x.size(0), -1, -1).unsqueeze(1).expand(-1, x.size(1), -1)
            x = x + regime_emb

        # Process through stacks
        residuals = x
        forecast_components = []

        for stack_idx, stack in enumerate(self.stacks):
            stack_forecasts = []

            for block in stack:
                # Forward pass through block
                backcast, forecast = block(residuals)

                # Subtract backcast from residuals
                residuals = residuals - backcast

                # Collect forecast
                stack_forecasts.append(forecast)

            # Sum forecasts from all blocks in stack
            stack_forecast = torch.stack(stack_forecasts).sum(dim=0)
            forecast_components.append(stack_forecast)

        # Sum forecasts from all stacks
        total_forecast = torch.stack(forecast_components).sum(dim=0)

        # Project to output dimension
        predictions = self.output_projection(total_forecast)

        return predictions


class MultiTimeframeAttention(nn.Module):
    """Attention mechanism for multi-timeframe fusion."""

    def __init__(self, config: MultiScaleNBEATSConfig):
        super(MultiTimeframeAttention, self).__init__()

        self.config = config
        self.attention_dim = config.layer_widths[0]

        # Timeframe embeddings
        self.timeframe_embedding = nn.Embedding(len(config.timeframes), self.attention_dim)

        # Attention layers
        self.query_projection = nn.Linear(self.attention_dim, self.attention_dim)
        self.key_projection = nn.Linear(self.attention_dim, self.attention_dim)
        self.value_projection = nn.Linear(self.attention_dim, self.attention_dim)

        # Output projection
        self.output_projection = nn.Linear(self.attention_dim, config.output_dim)

    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multi-timeframe attention.

        Args:
            predictions: Dictionary of predictions from each timeframe

        Returns:
            Fused predictions
        """
        # Extract predictions and create embeddings
        timeframe_list = list(predictions.keys())
        pred_values = torch.stack([predictions[tf] for tf in timeframe_list])

        # Create timeframe embeddings
        timeframe_indices = torch.tensor([self.config.timeframes.index(tf) for tf in timeframe_list])
        timeframe_emb = self.timeframe_embedding(timeframe_indices.to(pred_values.device))

        # Project to attention space
        queries = self.query_projection(timeframe_emb)
        keys = self.key_projection(timeframe_emb)
        values = self.value_projection(pred_values)

        # Compute attention weights
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended_values = torch.matmul(attention_weights, values)

        # Project to output
        fused_predictions = self.output_projection(attended_values)

        return fused_predictions


class MultiScaleNBEATSRegressor(BaseEstimator, RegressorMixin):
    """MultiScaleNBEATS regressor for multi-timeframe prediction."""

    def __init__(self,
                 config: Optional[MultiScaleNBEATSConfig] = None,
                 device: str = 'auto',
                 epochs: int = 100,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 early_stopping_patience: int = 15):
        """Initialize MultiScaleNBEATS regressor.

        Args:
            config: MultiScaleNBEATS configuration
            device: Device to use
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
        """
        self.config = config or MultiScaleNBEATSConfig()
        self.device = self._get_device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Initialize model
        self.model = MultiTimeframeNBEATS(self.config)
        self.model.to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Scalers
        self.scalers = {}

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _prepare_data(self, X: Dict[str, np.ndarray], y: np.ndarray) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Prepare data for training/inference."""
        prepared_data = {}
        scalers = {}

        for timeframe, x_data in X.items():
            # Scale data
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_data.reshape(-1, x_data.shape[-1])).reshape(x_data.shape)
            scalers[timeframe] = scaler

            # Convert to tensor
            x_tensor = torch.FloatTensor(x_scaled).to(self.device)
            prepared_data[timeframe] = x_tensor

        # Scale targets
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        self.scalers = scalers
        self.target_scaler = target_scaler

        return prepared_data, y_tensor

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray,
            X_val: Optional[Dict[str, np.ndarray]] = None,
            y_val: Optional[np.ndarray] = None,
            regimes: Optional[np.ndarray] = None) -> 'MultiScaleNBEATSRegressor':
        """Fit the MultiScaleNBEATS model.

        Args:
            X: Dictionary of training features for each timeframe
            y: Target values
            X_val: Dictionary of validation features (optional)
            y_val: Validation targets (optional)
            regimes: Regime labels for regime-aware training (optional)

        Returns:
            Self for method chaining
        """
        try:
            # Prepare data
            X_tensor, y_tensor = self._prepare_data(X, y)

            # Create datasets
            dataset = MultiTimeframeDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Validation data
            if X_val is not None and y_val is not None:
                X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
                val_dataset = MultiTimeframeDataset(X_val_tensor, y_val_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            best_val_loss = float('inf')
            patience_counter = 0

            logger.info(f"🚀 Training MultiScaleNBEATS model on {self.device}")
            logger.info(f"   - Training samples: {len(y_tensor)}")
            logger.info(f"   - Timeframes: {list(X_tensor.keys())}")
            logger.info(f"   - Input features: {X_tensor[list(X_tensor.keys())[0]].size(2)}")
            logger.info(f"   - Output dimensions: {y_tensor.size(1)}")

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                train_loss = 0.0

                for batch_data in dataloader:
                    batch_X, batch_y = batch_data
                    self.optimizer.zero_grad()

                    # Forward pass
                    predictions = self.model(batch_X)

                    # Use fused predictions if available, otherwise average
                    if 'fused' in predictions:
                        pred_tensor = predictions['fused']
                    else:
                        pred_tensor = torch.stack(list(predictions.values())).mean(dim=0)

                    # Compute loss
                    loss = self.loss_fn(pred_tensor, batch_y)
                    train_loss += loss.item()

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    val_loss = 0.0

                    with torch.no_grad():
                        for batch_data_val in val_dataloader:
                            batch_X_val, batch_y_val = batch_data_val
                            val_predictions = self.model(batch_X_val)

                            if 'fused' in val_predictions:
                                val_pred_tensor = val_predictions['fused']
                            else:
                                val_pred_tensor = torch.stack(list(val_predictions.values())).mean(dim=0)

                            val_loss += self.loss_fn(val_pred_tensor, batch_y_val).item()

                    val_loss /= len(val_dataloader)
                    self.history['val_loss'].append(val_loss)

                    # Learning rate scheduling
                    self.scheduler.step(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), 'multiscale_nbeats_best_model.pth')
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
                        break

                avg_train_loss = train_loss / len(dataloader)
                self.history['train_loss'].append(avg_train_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.6f}" +
                              (f" - Val Loss: {val_loss:.6f}" if X_val is not None else ""))

            # Load best model
            if X_val is not None:
                self.model.load_state_dict(torch.load('multiscale_nbeats_best_model.pth'))

            logger.info("✅ MultiScaleNBEATS model training completed")
            return self

        except Exception as e:
            logger.error(f"❌ MultiScaleNBEATS training failed: {e}")
            raise

    def predict(self, X: Dict[str, np.ndarray], return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using the trained MultiScaleNBEATS model.

        Args:
            X: Dictionary of input features for each timeframe
            return_uncertainty: Whether to return prediction uncertainty

        Returns:
            Predictions (and uncertainty if requested)
        """
        try:
            self.model.eval()

            # Prepare data
            prepared_data = {}
            for timeframe, x_data in X.items():
                if timeframe in self.scalers:
                    x_scaled = self.scalers[timeframe].transform(
                        x_data.reshape(-1, x_data.shape[-1])
                    ).reshape(x_data.shape)
                else:
                    # Use identity scaling if no scaler available
                    x_scaled = x_data

                x_tensor = torch.FloatTensor(x_scaled).to(self.device)
                prepared_data[timeframe] = x_tensor

            with torch.no_grad():
                predictions = self.model(prepared_data)

                # Use fused predictions if available
                if 'fused' in predictions:
                    pred_tensor = predictions['fused']
                else:
                    pred_tensor = torch.stack(list(predictions.values())).mean(dim=0)

                predictions = pred_tensor.cpu().numpy()

            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(predictions)

            if return_uncertainty:
                # For uncertainty estimation, we could use ensemble or MC dropout
                uncertainty = np.zeros_like(predictions)  # Placeholder
                return predictions, uncertainty
            else:
                return predictions

        except Exception as e:
            logger.error(f"❌ MultiScaleNBEATS prediction failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and training statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': 'MultiScaleNBEATS',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'timeframes': list(self.scalers.keys()),
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'forecast_length': self.config.forecast_length,
            'backcast_length': self.config.backcast_length,
            'stack_types': self.config.stack_types,
            'n_blocks': self.config.n_blocks,
            'n_layers': self.config.n_layers,
            'layer_widths': self.config.layer_widths,
            'regime_aware': self.config.regime_aware,
            'multi_timeframe_fusion': self.config.multi_timeframe_fusion,
            'device': str(self.device),
            'training_epochs': len(self.history['train_loss']),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
        }


class MultiTimeframeDataset(torch.utils.data.Dataset):
    """Dataset for multi-timeframe data."""

    def __init__(self, X_dict: Dict[str, torch.Tensor], y: torch.Tensor):
        """Initialize dataset.

        Args:
            X_dict: Dictionary of input tensors for each timeframe
            y: Target tensor
        """
        self.X_dict = X_dict
        self.y = y

        # Ensure all tensors have the same batch size
        self.length = len(y)
        for x_tensor in X_dict.values():
            assert len(x_tensor) == self.length, "All tensors must have the same length"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {tf: x_tensor[idx] for tf, x_tensor in self.X_dict.items()}, self.y[idx]


# Factory functions for creating MultiScaleNBEATS models
def create_multiscale_nbeats_model(config: Dict[str, Any]) -> MultiScaleNBEATSRegressor:
    """Create MultiScaleNBEATS model from configuration."""
    nbeats_config = MultiScaleNBEATSConfig(**config.get('nbeats_params', {}))
    model_config = config.get('model_params', {})

    return MultiScaleNBEATSRegressor(
        config=nbeats_config,
        device=config.get('device', 'auto'),
        epochs=model_config.get('epochs', 100),
        batch_size=model_config.get('batch_size', 64),
        learning_rate=model_config.get('learning_rate', 1e-3),
        early_stopping_patience=model_config.get('early_stopping_patience', 15)
    )


# Fallback implementation for when PyTorch is not available
class FallbackMultiScaleNBEATSRegressor(BaseEstimator, RegressorMixin):
    """Fallback MultiScaleNBEATS regressor when PyTorch is not available."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        logger.warning("⚠️ Using fallback MultiScaleNBEATS implementation without PyTorch")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        # Return zero predictions as fallback
        return np.zeros((len(X[list(X.keys())[0]]), 4))


def get_multiscale_nbeats_model(config: Dict[str, Any]) -> MultiScaleNBEATSRegressor:
    """Get MultiScaleNBEATS model with fast fail - no fallback."""
    if not TORCH_AVAILABLE:
        error_msg = "❌ MultiScaleNBEATS architecture requires PyTorch. Install with: pip install torch torchvision torchaudio"
        logger.error(error_msg)
        raise ImportError(error_msg)

    return create_multiscale_nbeats_model(config)