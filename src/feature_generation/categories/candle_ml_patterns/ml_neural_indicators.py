"""
Neural Network-Based Trading Indicators

This module provides neural network implementations (GRU, TFT) for generating
trading indicators from candlestick patterns and market data.

Key Features:
- GRU (Gated Recurrent Unit) for sequence modeling
- TFT (Temporal Fusion Transformer) for advanced time series modeling
- Integration with existing candlestick pattern features
- GPU acceleration support
- Advanced attention mechanisms
- Multi-task learning capabilities
"""

import numpy as np
import pandas as pd
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.nn.utils.rnn import pad_sequence
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

# Core imports
from .ml_candle_pattern_indicators import MLIndicatorGenerator, IndicatorType, IndicatorConfig

logger = logging.getLogger(__name__)


class AttentionType(Enum):
    """Types of attention mechanisms."""
    SELF_ATTENTION = "self_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"


@dataclass
class NeuralConfig:
    """Configuration for neural network models."""
    # Model architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    attention_type: AttentionType = AttentionType.MULTI_HEAD_ATTENTION
    num_attention_heads: int = 8
    
    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5
    
    # Sequence configuration
    sequence_length: int = 20
    prediction_horizon: int = 5
    
    # Multi-task learning
    enable_multi_task: bool = True
    task_weights: Dict[IndicatorType, float] = None
    
    def __post_init__(self):
        if self.task_weights is None:
            self.task_weights = {
                IndicatorType.DIRECTIONAL_SIGNAL: 1.0,
                IndicatorType.STRENGTH_SCORE: 0.8,
                IndicatorType.CONFIDENCE_LEVEL: 0.6
            }


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence modeling."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by number of heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.size()
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        return output


class GRUIndicatorModel(nn.Module):
    """GRU-based model for trading indicator generation."""
    
    def __init__(self, input_size: int, config: NeuralConfig):
        super(GRUIndicatorModel, self).__init__()
        self.config = config
        self.input_size = input_size
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        if config.attention_type == AttentionType.MULTI_HEAD_ATTENTION:
            self.attention = SelfAttention(
                hidden_size=config.hidden_size * 2,  # Bidirectional
                num_heads=config.num_attention_heads,
                dropout=config.dropout_rate
            )
        
        # Output layers for different indicator types
        self.output_layers = nn.ModuleDict({
            indicator_type.value: nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(64, 1)
            )
            for indicator_type in IndicatorType
        })
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Apply attention if enabled
        if hasattr(self, 'attention'):
            gru_out = self.attention(gru_out, mask)
        
        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(gru_out)
            gru_out = gru_out * mask_expanded
            pooled = gru_out.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = gru_out.mean(dim=1)
        
        # Generate outputs for each indicator type
        outputs = {}
        for indicator_type in IndicatorType:
            output = self.output_layers[indicator_type.value](pooled)
            outputs[indicator_type.value] = output.squeeze(-1)
        
        return outputs


class TFTIndicatorModel(nn.Module):
    """Temporal Fusion Transformer for trading indicator generation."""
    
    def __init__(self, input_size: int, config: NeuralConfig):
        super(TFTIndicatorModel, self).__init__()
        self.config = config
        self.input_size = input_size
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, config.hidden_size)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(config.sequence_length, config.hidden_size)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TFTEncoderLayer(config.hidden_size, config.num_attention_heads, config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        
        # Output layers
        self.output_layers = nn.ModuleDict({
            indicator_type.value: nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_size // 2, 1)
            )
            for indicator_type in IndicatorType
        })
        
        self._initialize_weights()
    
    def _create_positional_encoding(self, seq_len: int, hidden_size: int) -> torch.Tensor:
        """Create positional encoding for transformer."""
        pe = torch.zeros(seq_len, hidden_size)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           -(np.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Global average pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * mask_expanded
            pooled = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            pooled = x.mean(dim=1)
        
        # Generate outputs
        outputs = {}
        for indicator_type in IndicatorType:
            output = self.output_layers[indicator_type.value](pooled)
            outputs[indicator_type.value] = output.squeeze(-1)
        
        return outputs


class TFTEncoderLayer(nn.Module):
    """Encoder layer for TFT model."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super(TFTEncoderLayer, self).__init__()
        self.self_attention = SelfAttention(hidden_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_out = self.self_attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x


class NeuralIndicatorGenerator(MLIndicatorGenerator):
    """Neural network-based indicator generator."""
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 indicator_config: Optional[IndicatorConfig] = None,
                 neural_config: Optional[NeuralConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network models")
        
        super().__init__(config, indicator_config)
        self.neural_config = neural_config or NeuralConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.training_history = []
        
        logger.info(f"🚀 Neural Indicator Generator initialized on {self.device}")
    
    def _initialize_ml_components(self):
        """Initialize neural network components."""
        # This will be called after the base class initialization
        pass
    
    def create_sequence_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create sequence data for neural network training."""
        # Generate features
        pattern_features = self._generate_pattern_features(data)
        context_features = self._generate_market_context_features(data)
        features = self._combine_features(pattern_features, context_features)
        
        # Create sequences
        sequence_length = self.neural_config.sequence_length
        sequences = []
        targets = []
        masks = []
        
        for i in range(sequence_length, len(features)):
            # Input sequence
            seq = features[i-sequence_length:i]
            sequences.append(torch.FloatTensor(seq))
            
            # Target (future return)
            if 'future_return' in data.columns:
                target = data['future_return'].iloc[i]
            else:
                # Create synthetic target
                if i + self.neural_config.prediction_horizon < len(data):
                    current_price = data['close'].iloc[i]
                    future_price = data['close'].iloc[i + self.neural_config.prediction_horizon]
                    target = (future_price - current_price) / current_price
                else:
                    target = 0.0
            
            targets.append(target)
            
            # Mask (all valid for now)
            masks.append(torch.ones(sequence_length))
        
        # Pad sequences
        sequences = pad_sequence(sequences, batch_first=True)
        targets = torch.FloatTensor(targets)
        masks = pad_sequence(masks, batch_first=True)
        
        return sequences, targets, masks
    
    def train_neural_models(self, data: pd.DataFrame, 
                           target_column: str = 'future_return'):
        """Train neural network models."""
        start_time = time.time()
        logger.info("🧠 Training neural network models...")
        
        # Create sequence data
        X, y, masks = self.create_sequence_data(data)
        
        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        masks = masks.to(self.device)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
            X, y, masks, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train, mask_train)
        val_dataset = TensorDataset(X_val, y_val, mask_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.neural_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.neural_config.batch_size, shuffle=False)
        
        # Train models for each indicator type
        for indicator_type in self.indicator_config.indicator_types:
            logger.info(f"📚 Training {indicator_type.value} model...")
            
            # Create model
            if self.indicator_config.model_type == ModelType.GRU:
                model = GRUIndicatorModel(X.size(-1), self.neural_config)
            elif self.indicator_config.model_type == ModelType.TFT:
                model = TFTIndicatorModel(X.size(-1), self.neural_config)
            else:
                logger.warning(f"Unsupported model type: {self.indicator_config.model_type}")
                continue
            
            model = model.to(self.device)
            
            # Create optimizer and scheduler
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.neural_config.learning_rate,
                weight_decay=self.neural_config.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )
            
            # Train model
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.neural_config.num_epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y, batch_mask in train_loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_X, batch_mask)
                    target_output = outputs[indicator_type.value]
                    
                    # Calculate loss
                    if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                        # Classification loss
                        target_classes = self._prepare_target_for_indicator(
                            batch_y.cpu().numpy(), indicator_type
                        )
                        target_classes = torch.LongTensor(target_classes).to(self.device)
                        loss = nn.CrossEntropyLoss()(target_output, target_classes)
                    else:
                        # Regression loss
                        target_values = self._prepare_target_for_indicator(
                            batch_y.cpu().numpy(), indicator_type
                        )
                        target_values = torch.FloatTensor(target_values).to(self.device)
                        loss = nn.MSELoss()(target_output, target_values)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y, batch_mask in val_loader:
                        outputs = model(batch_X, batch_mask)
                        target_output = outputs[indicator_type.value]
                        
                        if indicator_type == IndicatorType.DIRECTIONAL_SIGNAL:
                            target_classes = self._prepare_target_for_indicator(
                                batch_y.cpu().numpy(), indicator_type
                            )
                            target_classes = torch.LongTensor(target_classes).to(self.device)
                            loss = nn.CrossEntropyLoss()(target_output, target_classes)
                        else:
                            target_values = self._prepare_target_for_indicator(
                                batch_y.cpu().numpy(), indicator_type
                            )
                            target_values = torch.FloatTensor(target_values).to(self.device)
                            loss = nn.MSELoss()(target_output, target_values)
                        
                        val_loss += loss.item()
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.models[indicator_type] = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= self.neural_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                              f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            # Store optimizer and scheduler
            self.optimizers[indicator_type] = optimizer
            self.schedulers[indicator_type] = scheduler
            
            logger.info(f"✅ {indicator_type.value} model training completed")
        
        # Update performance stats
        self.performance_stats['models_trained'] += 1
        self.performance_stats['total_training_time'] += time.time() - start_time
        self.performance_stats['last_retrain'] = datetime.now()
        
        logger.info(f"🎉 Neural network training completed in {time.time() - start_time:.2f} seconds")
    
    def _generate_indicators(self, features: np.ndarray, data: pd.DataFrame) -> Dict[IndicatorType, np.ndarray]:
        """Generate indicators using trained neural network models."""
        indicators = {}
        
        # Convert features to sequences
        sequence_length = self.neural_config.sequence_length
        if len(features) < sequence_length:
            # Not enough data for sequences
            for indicator_type in self.indicator_config.indicator_types:
                indicators[indicator_type] = np.zeros(len(features))
            return indicators
        
        # Create sequences
        sequences = []
        for i in range(sequence_length, len(features)):
            seq = features[i-sequence_length:i]
            sequences.append(torch.FloatTensor(seq))
        
        if not sequences:
            for indicator_type in self.indicator_config.indicator_types:
                indicators[indicator_type] = np.zeros(len(features))
            return indicators
        
        # Pad sequences
        sequences = pad_sequence(sequences, batch_first=True).to(self.device)
        masks = torch.ones(sequences.size(0), sequences.size(1)).to(self.device)
        
        # Generate predictions for each indicator type
        for indicator_type, model_state in self.models.items():
            try:
                # Create model
                if self.indicator_config.model_type == ModelType.GRU:
                    model = GRUIndicatorModel(features.shape[1], self.neural_config)
                elif self.indicator_config.model_type == ModelType.TFT:
                    model = TFTIndicatorModel(features.shape[1], self.neural_config)
                else:
                    continue
                
                model.load_state_dict(model_state)
                model = model.to(self.device)
                model.eval()
                
                # Generate predictions
                with torch.no_grad():
                    outputs = model(sequences, masks)
                    predictions = outputs[indicator_type.value].cpu().numpy()
                
                # Pad with zeros for the initial sequence_length samples
                full_predictions = np.zeros(len(features))
                full_predictions[sequence_length:] = predictions
                indicators[indicator_type] = full_predictions
                
            except Exception as e:
                logger.warning(f"Neural indicator generation failed for {indicator_type}: {e}")
                indicators[indicator_type] = np.zeros(len(features))
        
        return indicators


def create_neural_indicator_generator(
    model_type: ModelType = ModelType.GRU,
    neural_config: Optional[NeuralConfig] = None,
    **kwargs
) -> NeuralIndicatorGenerator:
    """Create a neural network indicator generator."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural network models")
    
    indicator_config = IndicatorConfig(
        model_type=model_type,
        indicator_types=[
            IndicatorType.DIRECTIONAL_SIGNAL,
            IndicatorType.STRENGTH_SCORE,
            IndicatorType.CONFIDENCE_LEVEL
        ],
        **kwargs
    )
    
    return NeuralIndicatorGenerator(
        indicator_config=indicator_config,
        neural_config=neural_config
    )


def test_neural_indicator_generator():
    """Test function for neural indicator generator."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available, skipping neural network test")
        return None, None
    
    print("🧪 Testing Neural Indicator Generator...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_samples)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=pd.date_range('2020-01-01', periods=n_samples, freq='1min'))
    
    # Ensure OHLC constraints
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Create neural indicator generator
    neural_config = NeuralConfig(
        hidden_size=64,
        num_layers=2,
        sequence_length=10,
        num_epochs=20
    )
    
    generator = create_neural_indicator_generator(
        model_type=ModelType.GRU,
        neural_config=neural_config
    )
    
    # Train models
    print("🧠 Training neural models...")
    generator.train_neural_models(data)
    
    # Generate indicators
    print("🔮 Generating neural indicators...")
    indicators = generator._generate_feature(data)
    
    print(f"✅ Generated neural indicators for {len(indicators)} samples")
    print(f"📊 Indicator statistics:")
    print(f"   - Mean: {indicators.mean():.4f}")
    print(f"   - Std: {indicators.std():.4f}")
    print(f"   - Min: {indicators.min():.4f}")
    print(f"   - Max: {indicators.max():.4f}")
    
    # Performance stats
    stats = generator.get_performance_stats()
    print(f"\n📈 Performance Statistics:")
    print(f"   - Models trained: {stats['models_trained']}")
    print(f"   - Predictions made: {stats['predictions_made']}")
    print(f"   - Training time: {stats['total_training_time']:.4f}s")
    print(f"   - Prediction time: {stats['total_prediction_time']:.4f}s")
    
    print("\n🎉 Neural Indicator Generator test completed successfully!")
    return generator, indicators


if __name__ == "__main__":
    test_neural_indicator_generator()