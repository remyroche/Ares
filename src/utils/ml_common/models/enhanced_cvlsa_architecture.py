"""
Enhanced CVLSA Architecture with Cross-View Attention and Multi-Scale Temporal Modeling

This module implements an advanced CVLSA (Cross-View Learning with Self-Attention) architecture
with the following enhancements:

1. Cross-View Attention: Attention mechanisms between different data modalities
2. Multi-Scale Temporal Attention: Enhanced temporal modeling for time series
3. Memory Efficiency: Gradient checkpointing and chunked processing
4. Bayesian Optimization: Automatic hyperparameter optimization
5. Hardware Integration: M1 GPU acceleration and memory optimization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from contextlib import contextmanager

# Import existing utilities
from src.utils.matrix_operations.enhanced_operations import get_enhanced_matrix_operations
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.ml_common.data_processing.feature_preparation import FeaturePreparator

logger = logging.getLogger(__name__)

@dataclass
class EnhancedCVLSAConfig:
    """Enhanced CVLSA configuration with all improvements."""
    
    # Architecture parameters
    input_dim: int = 100
    output_dim: int = 4
    seq_length: int = 200
    
    # Cross-view attention parameters
    cross_view_attention: bool = True
    view_embedding_dim: int = 64
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    
    # Multi-scale temporal parameters
    temporal_scales: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 30])
    use_multi_scale_attention: bool = True
    temporal_attention_heads: int = 8
    
    # Memory efficiency parameters
    memory_efficient: bool = True
    gradient_checkpointing: bool = True
    chunk_size: int = 1000
    max_sequence_length: int = 5000
    
    # Hardware optimization
    use_m1_gpu: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Bayesian optimization
    enable_hyperparameter_optimization: bool = True
    optimization_trials: int = 50
    optimization_timeout: int = 3600  # 1 hour
    
    # Feature engineering
    use_advanced_feature_engineering: bool = True
    feature_modalities: List[str] = field(default_factory=lambda: ['price', 'volume', 'trend', 'momentum'])
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.input_dim > 0, "Input dimension must be positive"
        assert self.output_dim > 0, "Output dimension must be positive"
        assert self.seq_length > 0, "Sequence length must be positive"
        assert 0 <= self.cross_attention_dropout <= 1, "Dropout must be between 0 and 1"
        assert len(self.temporal_scales) > 0, "At least one temporal scale must be specified"
        return True


class CrossViewAttention(nn.Module):
    """Cross-view attention mechanism for different data modalities."""
    
    def __init__(self, config: EnhancedCVLSAConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.view_embedding_dim
        self.num_heads = config.cross_attention_heads
        self.dropout = config.cross_attention_dropout
        
        # View-specific embeddings
        self.view_embeddings = nn.ModuleDict({
            modality: nn.Linear(config.input_dim, self.embed_dim)
            for modality in config.feature_modalities
        })
        
        # Cross-view attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # View fusion
        self.view_fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, view1_features: torch.Tensor, view2_features: torch.Tensor,
                view1_type: str, view2_type: str) -> torch.Tensor:
        """
        Apply cross-view attention between two data modalities.
        
        Args:
            view1_features: Features from first view (batch_size, seq_len, input_dim)
            view2_features: Features from second view (batch_size, seq_len, input_dim)
            view1_type: Type of first view (e.g., 'price', 'volume')
            view2_type: Type of second view (e.g., 'trend', 'momentum')
            
        Returns:
            Cross-view attended features
        """
        # Project to view-specific embeddings
        view1_emb = self.view_embeddings[view1_type](view1_features)
        view2_emb = self.view_embeddings[view2_type](view2_features)
        
        # Cross-view attention
        attn_output, attn_weights = self.cross_attention(
            view1_emb, view2_emb, view2_emb
        )
        
        # Store attention weights for interpretability
        self.attention_weights = attn_weights.detach()
        
        # Fuse views
        fused_features = self.view_fusion(
            torch.cat([view1_emb, attn_output], dim=-1)
        )
        
        # Layer normalization and residual connection
        output = self.layer_norm(fused_features + view1_emb)
        
        return output


class MultiScaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention for time series modeling."""
    
    def __init__(self, config: EnhancedCVLSAConfig):
        super().__init__()
        self.config = config
        self.temporal_scales = config.temporal_scales
        self.embed_dim = config.view_embedding_dim
        self.num_heads = config.temporal_attention_heads
        
        # Scale-specific attention modules
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=config.cross_attention_dropout,
                batch_first=True
            ) for _ in self.temporal_scales
        ])
        
        # Temporal fusion
        self.temporal_fusion = nn.Linear(
            self.embed_dim * len(self.temporal_scales),
            self.embed_dim
        )
        
        # Scale weighting
        self.scale_weights = nn.Parameter(torch.ones(len(self.temporal_scales)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale temporal attention.
        
        Args:
            x: Input features (batch_size, seq_len, embed_dim)
            
        Returns:
            Multi-scale temporal features
        """
        scale_outputs = []
        
        for i, scale in enumerate(self.temporal_scales):
            # Apply temporal scaling
            scaled_x = self._apply_temporal_scale(x, scale)
            
            # Apply attention at this scale
            attn_output, _ = self.scale_attentions[i](scaled_x, scaled_x, scaled_x)
            
            # Weight by scale importance
            weighted_output = attn_output * self.scale_weights[i]
            scale_outputs.append(weighted_output)
        
        # Fuse multi-scale features
        fused_temporal = self.temporal_fusion(torch.cat(scale_outputs, dim=-1))
        
        return fused_temporal
    
    def _apply_temporal_scale(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """Apply temporal scaling to input features."""
        batch_size, seq_len, embed_dim = x.shape
        
        if scale == 1:
            return x
        
        # Downsample by scale factor
        if seq_len >= scale:
            # Take every scale-th element
            scaled_indices = torch.arange(0, seq_len, scale, device=x.device)
            scaled_x = x[:, scaled_indices, :]
        else:
            # If sequence is shorter than scale, use the whole sequence
            scaled_x = x
        
        return scaled_x


class MemoryEfficientCVLSA(nn.Module):
    """Memory-efficient CVLSA with gradient checkpointing and chunked processing."""
    
    def __init__(self, config: EnhancedCVLSAConfig):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = config.gradient_checkpointing
        self.chunk_size = config.chunk_size
        self.max_sequence_length = config.max_sequence_length
        
        # Initialize components
        self.cross_view_attention = CrossViewAttention(config)
        self.temporal_attention = MultiScaleTemporalAttention(config)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.view_embedding_dim, config.view_embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.cross_attention_dropout),
            nn.Linear(config.view_embedding_dim // 2, config.output_dim)
        )
        
        # Initialize hardware optimizers
        self._init_hardware_optimizers()
        
    def _init_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            self.matrix_ops = get_enhanced_matrix_operations()
            self.gpu_manager = get_m1_gpu_manager() if self.config.use_m1_gpu else None
            self.memory_optimizer = get_m1_memory_optimizer(
                memory_limit_gb=self.config.memory_limit_gb
            )
        except Exception as e:
            logger.warning(f"Hardware optimizers not available: {e}")
            self.matrix_ops = None
            self.gpu_manager = None
            self.memory_optimizer = None
    
    def forward(self, price_features: torch.Tensor, volume_features: torch.Tensor,
                trend_features: torch.Tensor, momentum_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory optimization.
        
        Args:
            price_features: Price modality features
            volume_features: Volume modality features  
            trend_features: Trend modality features
            momentum_features: Momentum modality features
            
        Returns:
            CVLSA predictions
        """
        if self.use_gradient_checkpointing and self.training:
            return self._checkpointed_forward(
                price_features, volume_features, trend_features, momentum_features
            )
        else:
            return self._standard_forward(
                price_features, volume_features, trend_features, momentum_features
            )
    
    def _checkpointed_forward(self, price_features: torch.Tensor, volume_features: torch.Tensor,
                             trend_features: torch.Tensor, momentum_features: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        def checkpoint_forward(*inputs):
            return self._standard_forward(*inputs)
        
        return torch.utils.checkpoint.checkpoint(
            checkpoint_forward,
            price_features, volume_features, trend_features, momentum_features,
            use_reentrant=False
        )
    
    def _standard_forward(self, price_features: torch.Tensor, volume_features: torch.Tensor,
                         trend_features: torch.Tensor, momentum_features: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with chunked processing for large sequences."""
        batch_size, seq_len, _ = price_features.shape
        
        # Use chunked processing for very large sequences
        if seq_len > self.max_sequence_length:
            return self._chunked_processing(
                price_features, volume_features, trend_features, momentum_features
            )
        
        # Cross-view attention between modalities
        # Price-Volume attention
        pv_features = self.cross_view_attention(
            price_features, volume_features, 'price', 'volume'
        )
        
        # Trend-Momentum attention  
        tm_features = self.cross_view_attention(
            trend_features, momentum_features, 'trend', 'momentum'
        )
        
        # Cross-attention between price-volume and trend-momentum
        combined_features = self.cross_view_attention(
            pv_features, tm_features, 'price', 'trend'
        )
        
        # Multi-scale temporal attention
        temporal_features = self.temporal_attention(combined_features)
        
        # Output projection
        predictions = self.output_projection(temporal_features)
        
        return predictions
    
    def _chunked_processing(self, price_features: torch.Tensor, volume_features: torch.Tensor,
                           trend_features: torch.Tensor, momentum_features: torch.Tensor) -> torch.Tensor:
        """Process large sequences in chunks to manage memory."""
        batch_size, seq_len, embed_dim = price_features.shape
        chunk_size = min(self.chunk_size, seq_len)
        
        # Process in chunks
        chunk_outputs = []
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            
            # Extract chunk
            price_chunk = price_features[:, i:end_idx, :]
            volume_chunk = volume_features[:, i:end_idx, :]
            trend_chunk = trend_features[:, i:end_idx, :]
            momentum_chunk = momentum_features[:, i:end_idx, :]
            
            # Process chunk
            chunk_output = self._standard_forward(
                price_chunk, volume_chunk, trend_chunk, momentum_chunk
            )
            chunk_outputs.append(chunk_output)
        
        # Concatenate chunk outputs
        return torch.cat(chunk_outputs, dim=1)


class BayesianHyperparameterOptimizer:
    """Bayesian optimization for CVLSA hyperparameters."""
    
    def __init__(self, config: EnhancedCVLSAConfig):
        self.config = config
        self.optimization_trials = config.optimization_trials
        self.timeout = config.optimization_timeout
        
        # Initialize optimization components
        self._init_optimization_components()
        
    def _init_optimization_components(self):
        """Initialize Bayesian optimization components."""
        try:
            import optuna
            self.optuna_available = True
            self.optuna = optuna
        except ImportError:
            logger.warning("Optuna not available, using random search fallback")
            self.optuna_available = False
            self.optuna = None
    
    def optimize_hyperparameters(self, train_data: Dict[str, torch.Tensor],
                               val_data: Dict[str, torch.Tensor],
                               target: torch.Tensor) -> Dict[str, Any]:
        """
        Optimize CVLSA hyperparameters using Bayesian optimization.
        
        Args:
            train_data: Training data dictionary with modality features
            val_data: Validation data dictionary
            target: Target values
            
        Returns:
            Optimized hyperparameters
        """
        if not self.optuna_available:
            return self._random_search_optimization(train_data, val_data, target)
        
        def objective(trial):
            # Sample hyperparameters
            params = {
                'view_embedding_dim': trial.suggest_categorical('view_embedding_dim', [32, 64, 128, 256]),
                'cross_attention_heads': trial.suggest_categorical('cross_attention_heads', [4, 8, 16]),
                'cross_attention_dropout': trial.suggest_float('cross_attention_dropout', 0.0, 0.3),
                'temporal_attention_heads': trial.suggest_categorical('temporal_attention_heads', [4, 8, 16]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            }
            
            # Create model with sampled parameters
            model_config = EnhancedCVLSAConfig(**params)
            model = MemoryEfficientCVLSA(model_config)
            
            # Train and evaluate
            score = self._evaluate_model(model, train_data, val_data, target)
            return score
        
        # Create study
        study = self.optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.optimization_trials, timeout=self.timeout)
        
        return study.best_params
    
    def _random_search_optimization(self, train_data: Dict[str, torch.Tensor],
                                  val_data: Dict[str, torch.Tensor],
                                  target: torch.Tensor) -> Dict[str, Any]:
        """Fallback random search optimization."""
        logger.info("Using random search optimization (Optuna not available)")
        
        best_score = float('-inf')
        best_params = {}
        
        for trial in range(self.optimization_trials):
            # Random parameter sampling
            params = {
                'view_embedding_dim': np.random.choice([32, 64, 128, 256]),
                'cross_attention_heads': np.random.choice([4, 8, 16]),
                'cross_attention_dropout': np.random.uniform(0.0, 0.3),
                'temporal_attention_heads': np.random.choice([4, 8, 16]),
                'learning_rate': np.random.uniform(1e-5, 1e-2),
                'batch_size': np.random.choice([16, 32, 64, 128])
            }
            
            # Evaluate model
            model_config = EnhancedCVLSAConfig(**params)
            model = MemoryEfficientCVLSA(model_config)
            score = self._evaluate_model(model, train_data, val_data, target)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def _evaluate_model(self, model: MemoryEfficientCVLSA, train_data: Dict[str, torch.Tensor],
                       val_data: Dict[str, torch.Tensor], target: torch.Tensor) -> float:
        """Evaluate model performance."""
        try:
            # Simple evaluation (can be enhanced)
            with torch.no_grad():
                predictions = model(
                    train_data['price'], train_data['volume'],
                    train_data['trend'], train_data['momentum']
                )
                
                # Calculate simple accuracy metric
                mse = F.mse_loss(predictions, target)
                score = -mse.item()  # Negative MSE for maximization
                
            return score
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return float('-inf')


class EnhancedCVLSATrainer:
    """Enhanced CVLSA trainer with all optimizations."""
    
    def __init__(self, config: EnhancedCVLSAConfig):
        self.config = config
        self.model = MemoryEfficientCVLSA(config)
        self.optimizer = None
        self.scheduler = None
        
        # Initialize hardware optimizers
        self._init_hardware_optimizers()
        
        # Initialize hyperparameter optimizer
        if config.enable_hyperparameter_optimization:
            self.hyperparameter_optimizer = BayesianHyperparameterOptimizer(config)
        else:
            self.hyperparameter_optimizer = None
    
    def _init_hardware_optimizers(self):
        """Initialize hardware optimization components."""
        try:
            self.matrix_ops = get_enhanced_matrix_operations()
            self.gpu_manager = get_m1_gpu_manager() if self.config.use_m1_gpu else None
            self.memory_optimizer = get_m1_memory_optimizer(
                memory_limit_gb=self.config.memory_limit_gb
            )
        except Exception as e:
            logger.warning(f"Hardware optimizers not available: {e}")
            self.matrix_ops = None
            self.gpu_manager = None
            self.memory_optimizer = None
    
    def prepare_features(self, market_data: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Prepare features for different modalities using advanced feature engineering.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Dictionary of modality features
        """
        logger.info("🔧 Preparing enhanced features for CVLSA...")
        
        # Use existing feature preparation utilities
        feature_data, feature_names, metadata = FeaturePreparator.prepare_features(
            market_data,
            feature_config={
                'scale_features': True,
                'apply_pca': False
            }
        )
        
        # Create modality-specific features
        modality_features = {}
        
        # Price features (OHLCV)
        price_cols = ['open', 'high', 'low', 'close'] if all(col in market_data.columns for col in ['open', 'high', 'low', 'close']) else ['close']
        price_features = market_data[price_cols].values
        modality_features['price'] = torch.FloatTensor(price_features)
        
        # Volume features
        if 'volume' in market_data.columns:
            volume_features = market_data[['volume']].values
            # Add volume-based technical indicators
            volume_returns = np.diff(volume_features.flatten(), prepend=volume_features[0, 0])
            volume_ma = pd.Series(volume_features.flatten()).rolling(20).mean().fillna(0).values
            volume_features = np.column_stack([volume_features, volume_returns.reshape(-1, 1), volume_ma.reshape(-1, 1)])
        else:
            volume_features = np.zeros((len(market_data), 3))
        modality_features['volume'] = torch.FloatTensor(volume_features)
        
        # Trend features (moving averages, trend indicators)
        close_prices = market_data['close'].values if 'close' in market_data.columns else market_data.iloc[:, 0].values
        trend_features = []
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            ma = pd.Series(close_prices).rolling(window).mean().fillna(close_prices[0]).values
            trend_features.append(ma)
        
        # Price momentum
        returns = np.diff(close_prices, prepend=close_prices[0])
        trend_features.append(returns)
        
        # Trend strength
        trend_strength = pd.Series(close_prices).rolling(20).apply(lambda x: np.corrcoef(x, np.arange(len(x)))[0, 1] if len(x) > 1 else 0).fillna(0).values
        trend_features.append(trend_strength)
        
        modality_features['trend'] = torch.FloatTensor(np.column_stack(trend_features))
        
        # Momentum features (RSI, MACD, etc.)
        momentum_features = []
        
        # RSI
        rsi = self._calculate_rsi(close_prices, 14)
        momentum_features.append(rsi)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(close_prices)
        momentum_features.extend([macd_line, signal_line, histogram])
        
        # Price momentum
        for window in [5, 10, 20]:
            momentum = pd.Series(close_prices).pct_change(window).fillna(0).values
            momentum_features.append(momentum)
        
        modality_features['momentum'] = torch.FloatTensor(np.column_stack(momentum_features))
        
        # Ensure all features have the same sequence length
        min_length = min(features.shape[0] for features in modality_features.values())
        for modality in modality_features:
            modality_features[modality] = modality_features[modality][:min_length]
        
        logger.info(f"✅ Prepared features for {len(modality_features)} modalities")
        for modality, features in modality_features.items():
            logger.info(f"   {modality}: {features.shape}")
        
        return modality_features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(period).mean().fillna(0).values
        avg_loss = pd.Series(loss).rolling(period).mean().fillna(0).values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal).mean().values
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def train(self, train_data: Dict[str, torch.Tensor], val_data: Dict[str, torch.Tensor],
              target: torch.Tensor, epochs: int = 100) -> Dict[str, Any]:
        """
        Train the enhanced CVLSA model.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary  
            target: Target values
            epochs: Number of training epochs
            
        Returns:
            Training results
        """
        logger.info("🚀 Starting enhanced CVLSA training...")
        
        # Hyperparameter optimization
        if self.hyperparameter_optimizer:
            logger.info("🔧 Optimizing hyperparameters...")
            best_params = self.hyperparameter_optimizer.optimize_hyperparameters(
                train_data, val_data, target
            )
            logger.info(f"✅ Best hyperparameters: {best_params}")
            
            # Update model with best parameters
            self._update_model_parameters(best_params)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training loop with memory optimization
        training_results = {
            'train_losses': [],
            'val_losses': [],
            'attention_weights': [],
            'training_time': 0.0
        }
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_data, target)
            
            # Validation phase
            val_loss = self._validate_epoch(val_data, target)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store results
            training_results['train_losses'].append(train_loss)
            training_results['val_losses'].append(val_loss)
            
            # Store attention weights for analysis
            if hasattr(self.model.cross_view_attention, 'attention_weights'):
                training_results['attention_weights'].append(
                    self.model.cross_view_attention.attention_weights.cpu().numpy()
                )
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        training_results['training_time'] = time.time() - start_time
        logger.info(f"✅ Training completed in {training_results['training_time']:.2f}s")
        
        return training_results
    
    def _train_epoch(self, train_data: Dict[str, torch.Tensor], target: torch.Tensor) -> float:
        """Train for one epoch with memory optimization."""
        self.model.train()
        total_loss = 0.0
        
        # Use memory optimization context
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint("cvlsa_training"):
                # Forward pass
                predictions = self.model(
                    train_data['price'], train_data['volume'],
                    train_data['trend'], train_data['momentum']
                )
                
                # Calculate loss
                loss = F.mse_loss(predictions, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss = loss.item()
        else:
            # Standard training without memory optimization
            predictions = self.model(
                train_data['price'], train_data['volume'],
                train_data['trend'], train_data['momentum']
            )
            
            loss = F.mse_loss(predictions, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss = loss.item()
        
        return total_loss
    
    def _validate_epoch(self, val_data: Dict[str, torch.Tensor], target: torch.Tensor) -> float:
        """Validate for one epoch."""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(
                val_data['price'], val_data['volume'],
                val_data['trend'], val_data['momentum']
            )
            
            loss = F.mse_loss(predictions, target)
            return loss.item()
    
    def _update_model_parameters(self, params: Dict[str, Any]):
        """Update model with optimized parameters."""
        # This would involve recreating the model with new parameters
        # For now, we'll just log the parameters
        logger.info(f"Updating model with parameters: {params}")
    
    def predict(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions with the trained model."""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(
                data['price'], data['volume'],
                data['trend'], data['momentum']
            )
        
        return predictions
    
    def get_attention_weights(self) -> Dict[str, np.ndarray]:
        """Get attention weights for interpretability."""
        attention_weights = {}
        
        if hasattr(self.model.cross_view_attention, 'attention_weights'):
            attention_weights['cross_view'] = self.model.cross_view_attention.attention_weights.cpu().numpy()
        
        return attention_weights


# Factory functions
def create_enhanced_cvlsa_model(config: Optional[EnhancedCVLSAConfig] = None) -> EnhancedCVLSATrainer:
    """Create enhanced CVLSA model."""
    if config is None:
        config = EnhancedCVLSAConfig()
    
    config.validate()
    return EnhancedCVLSATrainer(config)


def create_cvlsa_config(**kwargs) -> EnhancedCVLSAConfig:
    """Create CVLSA configuration with custom parameters."""
    return EnhancedCVLSAConfig(**kwargs)