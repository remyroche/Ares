"""
Enhanced PatchTST Model for Analyst with VectorBT Integration

This module implements an enhanced PatchTST model for the analyst with VectorBT integration:
- Lookback: 8-24h (configurable)
- d_model: 64-128 (configurable)
- heads: 2-4 (configurable)
- layers: 2
- Export: 8-12 dims + ŷ, conf (OOF)
- VectorBT backtesting and financial metrics integration
- VectorBT feature generation and optimization
- Memory management and performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.records.base import Records
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    Records = None

# VectorBT utils imports
try:
    from src.utils.ml_common.vectorbt_backtesting_engine import VectorBTBacktestingEngine, VectorBTBacktestConfig, BacktestMode
    from src.utils.ml_common.vectorbt_financial_metrics import VectorBTFinancialMetrics, FinancialMetricsConfig
    from src.feature_generation.core.vectorbt_feature_generator import VectorBTFeatureGenerator, VectorBTVolatilityGenerator, VectorBTMomentumGenerator, VectorBTTrendGenerator
    from src.utils.ml_common.vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
    from src.utils.ml_common.vectorbt_performance_monitor import get_performance_monitor, monitor_operation
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTBacktestingEngine = None
    VectorBTFinancialMetrics = None
    VectorBTFeatureGenerator = None

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPatchTSTConfig:
    """Configuration for Enhanced PatchTST model with VectorBT integration."""
    # PatchTST parameters
    lookback_hours: int = 16  # 8-24h range, using 16h as middle
    d_model: int = 96  # 64-128 range, using 96 as middle
    heads: int = 3  # 2-4 range, using 3 as middle
    layers: int = 2
    export_dims: int = 10  # 8-12 range, using 10 as middle
    
    # Patch parameters
    patch_len: int = 16
    stride: int = 8
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    random_state: int = 42
    
    # OOF parameters
    cv_folds: int = 5
    include_confidence: bool = True
    include_oof_predictions: bool = True
    
    # VectorBT integration parameters
    enable_vectorbt: bool = True
    enable_vectorbt_backtesting: bool = True
    enable_vectorbt_metrics: bool = True
    enable_vectorbt_features: bool = True
    enable_memory_optimization: bool = True
    enable_performance_monitoring: bool = True
    
    # VectorBT backtesting configuration
    vectorbt_backtest_config: Optional[VectorBTBacktestConfig] = None
    vectorbt_metrics_config: Optional[FinancialMetricsConfig] = None
    
    # Performance settings
    memory_limit_gb: float = 8.0
    enable_gpu: bool = False
    enable_parallel: bool = True
    chunk_size: int = 1000


class PatchEmbedding(nn.Module):
    """Patch embedding layer for PatchTST."""
    
    def __init__(self, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        
        # Linear projection for patches
        self.patch_projection = nn.Linear(patch_len, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through patch embedding."""
        # x shape: (batch_size, seq_len, patch_len)
        batch_size, seq_len, _ = x.shape
        
        # Project patches to d_model
        x = self.patch_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0)
        x = x + pos_enc
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        
        assert d_model % heads == 0, "d_model must be divisible by heads"
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention."""
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.out_linear(attended)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, d_model: int, heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class EnhancedPatchTST(nn.Module):
    """Enhanced PatchTST model for analyst."""
    
    def __init__(self, input_size: int, patch_len: int = 16, d_model: int = 96,
                 heads: int = 3, layers: int = 2, export_dims: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.patch_len = patch_len
        self.d_model = d_model
        self.export_dims = export_dims
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, d_model, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, dropout=dropout)
            for _ in range(layers)
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projections
        self.prediction_head = nn.Linear(d_model, 1)
        self.confidence_head = nn.Linear(d_model, 1)
        self.embedding_head = nn.Linear(d_model, export_dims)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through PatchTST."""
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        
        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.patch_len // 2):
            patch = x[:, i:i + self.patch_len, :]
            patches.append(patch)
        
        if not patches:
            # If no patches can be created, use the whole sequence
            patches = [x]
        
        # Stack patches
        patches = torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_len, input_size)
        
        # Flatten patches
        patches = patches.view(batch_size, patches.shape[1], -1)  # (batch_size, num_patches, patch_len * input_size)
        
        # Patch embedding
        embedded = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            embedded = block(embedded)
        
        # Global pooling
        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)
        
        # Outputs
        prediction = self.prediction_head(pooled)  # (batch_size, 1)
        confidence = torch.sigmoid(self.confidence_head(pooled))  # (batch_size, 1)
        embedding = self.embedding_head(pooled)  # (batch_size, export_dims)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'embedding': embedding
        }


class EnhancedPatchTSTModel(BaseEstimator, RegressorMixin):
    """
    Enhanced PatchTST Model for Analyst with VectorBT Integration.
    
    This model uses patch-based time series transformation with transformer
    architecture to generate features for the analyst models, enhanced with
    VectorBT backtesting, financial metrics, and feature generation capabilities.
    """
    
    def __init__(self, config: Optional[EnhancedPatchTSTConfig] = None):
        """Initialize the Enhanced PatchTST model with VectorBT integration."""
        self.config = config or EnhancedPatchTSTConfig()
        
        # Components
        self.patchtst_model = None
        self.scaler = None
        
        # State
        self.fitted = False
        self.feature_names = None
        self.oof_predictions = None
        self.oof_confidence = None
        self.oof_embeddings = None
        
        # VectorBT components
        self.vectorbt_backtesting_engine = None
        self.vectorbt_metrics_calculator = None
        self.vectorbt_feature_generators = []
        self.memory_manager = None
        self.performance_monitor = None
        
        # Initialize VectorBT components if available
        if self.config.enable_vectorbt and VECTORBT_UTILS_AVAILABLE:
            self._initialize_vectorbt_components()
        
        # Performance tracking
        self.vectorbt_stats = {
            'backtests_run': 0,
            'metrics_calculated': 0,
            'features_generated': 0,
            'memory_optimizations': 0,
            'performance_operations': 0
        }
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT components for enhanced functionality."""
        try:
            # Initialize memory manager
            if self.config.enable_memory_optimization:
                self.memory_manager = get_memory_manager()
                logger.info("✅ VectorBT memory manager initialized")
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                self.performance_monitor = get_performance_monitor()
                logger.info("✅ VectorBT performance monitor initialized")
            
            # Initialize backtesting engine
            if self.config.enable_vectorbt_backtesting and VectorBTBacktestingEngine:
                backtest_config = self.config.vectorbt_backtest_config
                if backtest_config is None:
                    backtest_config = VectorBTBacktestConfig(
                        initial_capital=100000.0,
                        commission_rate=0.001,
                        slippage_rate=0.0005,
                        use_gpu=self.config.enable_gpu,
                        enable_parallel=self.config.enable_parallel,
                        memory_limit_gb=self.config.memory_limit_gb
                    )
                
                self.vectorbt_backtesting_engine = VectorBTBacktestingEngine(backtest_config)
                logger.info("✅ VectorBT backtesting engine initialized")
            
            # Initialize metrics calculator
            if self.config.enable_vectorbt_metrics and VectorBTFinancialMetrics:
                metrics_config = self.config.vectorbt_metrics_config
                if metrics_config is None:
                    metrics_config = FinancialMetricsConfig(
                        risk_free_rate=0.02,
                        annualization_factor=252,
                        enable_regime_analysis=True,
                        enable_parallel=self.config.enable_parallel
                    )
                
                self.vectorbt_metrics_calculator = VectorBTFinancialMetrics(metrics_config)
                logger.info("✅ VectorBT financial metrics calculator initialized")
            
            # Initialize feature generators
            if self.config.enable_vectorbt_features and VectorBTFeatureGenerator:
                self.vectorbt_feature_generators = [
                    VectorBTVolatilityGenerator(period=20),
                    VectorBTMomentumGenerator(period=14),
                    VectorBTTrendGenerator(period=20)
                ]
                logger.info(f"✅ VectorBT feature generators initialized: {len(self.vectorbt_feature_generators)} generators")
            
            logger.info("🚀 VectorBT components initialization completed")
            
        except Exception as e:
            logger.warning(f"⚠️ VectorBT components initialization failed: {e}")
            self.config.enable_vectorbt = False
    
    def generate_vectorbt_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using VectorBT feature generators."""
        if not self.config.enable_vectorbt_features or not self.vectorbt_feature_generators:
            logger.warning("⚠️ VectorBT feature generation not enabled or generators not available")
            return pd.DataFrame(index=data.index)
        
        try:
            with monitor_operation(
                f"vectorbt_feature_generation_{len(self.vectorbt_feature_generators)}",
                metadata={'n_generators': len(self.vectorbt_feature_generators), 'data_shape': data.shape}
            ):
                features = []
                
                for generator in self.vectorbt_feature_generators:
                    try:
                        feature = generator.generate(data)
                        if isinstance(feature, pd.Series):
                            features.append(feature)
                        elif isinstance(feature, pd.DataFrame):
                            features.extend([feature[col] for col in feature.columns])
                    except Exception as e:
                        logger.warning(f"⚠️ Feature generator {generator.__class__.__name__} failed: {e}")
                        continue
                
                if features:
                    result_df = pd.DataFrame(features).T
                    result_df.index = data.index
                    self.vectorbt_stats['features_generated'] += len(features)
                    logger.info(f"✅ Generated {len(features)} VectorBT features")
                    return result_df
                else:
                    logger.warning("⚠️ No VectorBT features generated")
                    return pd.DataFrame(index=data.index)
        
        except Exception as e:
            logger.error(f"❌ VectorBT feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def run_vectorbt_backtest(self, signals: Union[np.ndarray, pd.DataFrame], 
                            prices: Union[np.ndarray, pd.DataFrame],
                            timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                            mode: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Run VectorBT backtest on model predictions."""
        if not self.config.enable_vectorbt_backtesting or not self.vectorbt_backtesting_engine:
            logger.warning("⚠️ VectorBT backtesting not enabled or engine not available")
            return None
        
        try:
            # Convert mode string to BacktestMode enum
            if mode == 'gpu':
                backtest_mode = BacktestMode.VECTORBT_GPU
            elif mode == 'parallel':
                backtest_mode = BacktestMode.VECTORBT_PARALLEL
            elif mode == 'hybrid':
                backtest_mode = BacktestMode.HYBRID
            else:
                backtest_mode = BacktestMode.VECTORBT_CPU
            
            with monitor_operation(
                f"vectorbt_backtest_{mode}",
                metadata={'signals_shape': signals.shape if hasattr(signals, 'shape') else len(signals),
                         'prices_shape': prices.shape if hasattr(prices, 'shape') else len(prices)}
            ):
                results = self.vectorbt_backtesting_engine.run_backtest(
                    signals=signals,
                    prices=prices,
                    timestamps=timestamps,
                    mode=backtest_mode
                )
                
                self.vectorbt_stats['backtests_run'] += 1
                logger.info(f"✅ VectorBT backtest completed with mode: {mode}")
                return {
                    'results': results,
                    'performance_metrics': results.performance_metrics,
                    'risk_metrics': results.risk_metrics,
                    'drawdown_analysis': results.drawdown_analysis,
                    'computation_time': results.computation_time,
                    'memory_usage': results.memory_usage
                }
        
        except Exception as e:
            logger.error(f"❌ VectorBT backtest failed: {e}")
            return None
    
    def calculate_vectorbt_metrics(self, portfolio_values: Union[np.ndarray, pd.Series],
                                 returns: Optional[Union[np.ndarray, pd.Series]] = None,
                                 benchmark_values: Optional[Union[np.ndarray, pd.Series]] = None,
                                 timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive financial metrics using VectorBT."""
        if not self.config.enable_vectorbt_metrics or not self.vectorbt_metrics_calculator:
            logger.warning("⚠️ VectorBT metrics calculation not enabled or calculator not available")
            return None
        
        try:
            with monitor_operation(
                "vectorbt_metrics_calculation",
                metadata={'portfolio_shape': portfolio_values.shape if hasattr(portfolio_values, 'shape') else len(portfolio_values)}
            ):
                metrics = self.vectorbt_metrics_calculator.calculate_comprehensive_metrics(
                    portfolio_values=portfolio_values,
                    returns=returns,
                    benchmark_values=benchmark_values,
                    timestamps=timestamps
                )
                
                self.vectorbt_stats['metrics_calculated'] += 1
                logger.info(f"✅ Calculated {len(metrics)} VectorBT financial metrics")
                return metrics
        
        except Exception as e:
            logger.error(f"❌ VectorBT metrics calculation failed: {e}")
            return None
    
    def get_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        stats = self.vectorbt_stats.copy()
        
        # Add memory manager stats if available
        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            stats.update({
                'memory_usage_gb': memory_stats.get('current_usage_gb', 0),
                'memory_peak_gb': memory_stats.get('peak_usage_gb', 0),
                'memory_available_gb': memory_stats.get('available_memory_gb', 0),
                'memory_utilization': memory_stats.get('usage_percentage', 0)
            })
        
        # Add performance monitor stats if available
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_performance_summary()
            stats.update({
                'total_operations_monitored': perf_stats.get('total_operations', 0),
                'average_operation_duration': perf_stats.get('average_duration', 0),
                'gpu_utilization_rate': perf_stats.get('gpu_utilization_rate', 0),
                'cache_hit_rate': perf_stats.get('cache_hit_rate', 0),
                'error_rate': perf_stats.get('error_rate', 0)
            })
        
        return stats
    
    def reset_vectorbt_stats(self):
        """Reset VectorBT performance statistics."""
        self.vectorbt_stats = {
            'backtests_run': 0,
            'metrics_calculated': 0,
            'features_generated': 0,
            'memory_optimizations': 0,
            'performance_operations': 0
        }
        
    def _prepare_sequences(self, X: np.ndarray, lookback_bars: int) -> np.ndarray:
        """Prepare sequences for PatchTST input."""
        try:
            sequences = []
            for i in range(lookback_bars, len(X)):
                sequence = X[i-lookback_bars:i]
                sequences.append(sequence)
            
            if not sequences:
                # If no sequences can be created, pad the data
                padded_X = np.zeros((lookback_bars, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, lookback_bars, -1)
            
            return np.array(sequences)
        except Exception as e:
            logger.warning(f"⚠️ Sequence preparation failed: {e}")
            # Fallback: create single sequence with padding
            if len(X) < lookback_bars:
                padded_X = np.zeros((lookback_bars, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, lookback_bars, -1)
            return X[-lookback_bars:].reshape(1, lookback_bars, -1)
    
    def _get_oof_predictions(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get out-of-fold predictions for training features."""
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            
            oof_predictions = []
            oof_confidence = []
            oof_embeddings = []
            
            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Prepare sequences
                lookback_bars = self.config.lookback_hours * 12  # Assuming 5m bars
                X_train_seq = self._prepare_sequences(X_train_scaled, lookback_bars)
                X_val_seq = self._prepare_sequences(X_val_scaled, lookback_bars)
                
                # Convert to tensors
                X_train_tensor = torch.FloatTensor(X_train_seq)
                X_val_tensor = torch.FloatTensor(X_val_seq)
                y_train_tensor = torch.FloatTensor(y_train)
                
                # Create model
                model = EnhancedPatchTST(
                    input_size=X.shape[1],
                    patch_len=self.config.patch_len,
                    d_model=self.config.d_model,
                    heads=self.config.heads,
                    layers=self.config.layers,
                    export_dims=self.config.export_dims
                )
                
                # Training setup
                optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
                criterion = torch.nn.MSELoss()
                
                # Data loader
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
                
                # Training
                model.train()
                for epoch in range(self.config.epochs):
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        
                        outputs = model(batch_X)
                        loss = criterion(outputs['prediction'].squeeze(), batch_y)
                        
                        loss.backward()
                        optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    
                    oof_predictions.extend(val_outputs['prediction'].squeeze().numpy())
                    oof_confidence.extend(val_outputs['confidence'].squeeze().numpy())
                    oof_embeddings.extend(val_outputs['embedding'].numpy())
            
            return np.array(oof_predictions), np.array(oof_confidence), np.array(oof_embeddings)
            
        except Exception as e:
            logger.warning(f"⚠️ OOF prediction generation failed: {e}")
            # Return zeros as fallback
            n_samples = len(X)
            return (np.zeros(n_samples), np.zeros(n_samples), 
                   np.zeros((n_samples, self.config.export_dims)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'EnhancedPatchTSTModel':
        """Fit the Enhanced PatchTST model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
                X = X.values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Get OOF predictions if requested
            if self.config.include_oof_predictions:
                self.oof_predictions, self.oof_confidence, self.oof_embeddings = self._get_oof_predictions(X_scaled, y)
            
            # Prepare sequences for final training
            lookback_bars = self.config.lookback_hours * 12  # Assuming 5m bars
            X_seq = self._prepare_sequences(X_scaled, lookback_bars)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y)
            
            # Create PatchTST model
            self.patchtst_model = EnhancedPatchTST(
                input_size=X.shape[1],
                patch_len=self.config.patch_len,
                d_model=self.config.d_model,
                heads=self.config.heads,
                layers=self.config.layers,
                export_dims=self.config.export_dims
            )
            
            # Training setup
            optimizer = optim.Adam(self.patchtst_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            # Data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            # Training loop
            self.patchtst_model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.patchtst_model(batch_X)
                    loss = criterion(outputs['prediction'].squeeze(), batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            self.fitted = True
            logger.info(f"✅ Enhanced PatchTST model fitted with {X.shape[1]} features")
            
            return self
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using fallback linear model")
            return self._fit_fallback(X, y, sample_weight)
        except Exception as e:
            logger.error(f"❌ Enhanced PatchTST model fitting failed: {e}")
            return self._fit_fallback(X, y, sample_weight)
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'EnhancedPatchTSTModel':
        """Fallback to simple linear model."""
        try:
            from sklearn.linear_model import LinearRegression
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Simple linear model as fallback
            self.patchtst_model = LinearRegression()
            self.patchtst_model.fit(X_scaled, y, sample_weight)
            
            # Create dummy OOF predictions
            if self.config.include_oof_predictions:
                self.oof_predictions = self.patchtst_model.predict(X_scaled)
                self.oof_confidence = np.ones(len(y)) * 0.5
                self.oof_embeddings = np.random.randn(len(y), self.config.export_dims) * 0.1
            
            self.fitted = True
            logger.info("✅ Fallback linear model fitted")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Fallback model fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Check if model is PyTorch model
            if hasattr(self.patchtst_model, 'forward'):
                import torch
                
                # Prepare sequences
                lookback_bars = self.config.lookback_hours * 12
                X_seq = self._prepare_sequences(X_scaled, lookback_bars)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_seq)
                
                # Predict
                self.patchtst_model.eval()
                with torch.no_grad():
                    outputs = self.patchtst_model(X_tensor)
                    predictions = outputs['prediction'].squeeze().numpy()
                
                # Ensure we have the right number of predictions
                if len(predictions) < X.shape[0]:
                    # Pad with the last prediction
                    padding = np.full(X.shape[0] - len(predictions), predictions[-1])
                    predictions = np.concatenate([predictions, padding])
                elif len(predictions) > X.shape[0]:
                    # Truncate to match
                    predictions = predictions[:X.shape[0]]
                
                return predictions
            else:
                # Fallback model
                return self.patchtst_model.predict(X_scaled)
                
        except Exception as e:
            logger.error(f"❌ Enhanced PatchTST model prediction failed: {e}")
            raise
    
    def get_features(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get PatchTST features (predictions, confidence, embeddings)."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting features")
        
        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Check if model is PyTorch model
            if hasattr(self.patchtst_model, 'forward'):
                import torch
                
                # Prepare sequences
                lookback_bars = self.config.lookback_hours * 12
                X_seq = self._prepare_sequences(X_scaled, lookback_bars)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_seq)
                
                # Get features
                self.patchtst_model.eval()
                with torch.no_grad():
                    outputs = self.patchtst_model(X_tensor)
                    
                    predictions = outputs['prediction'].squeeze().numpy()
                    confidence = outputs['confidence'].squeeze().numpy()
                    embeddings = outputs['embedding'].numpy()
                
                # Ensure we have the right number of features
                if len(predictions) < X.shape[0]:
                    # Pad with the last values
                    pred_padding = np.full(X.shape[0] - len(predictions), predictions[-1])
                    conf_padding = np.full(X.shape[0] - len(confidence), confidence[-1])
                    emb_padding = np.tile(embeddings[-1:], (X.shape[0] - len(embeddings), 1))
                    
                    predictions = np.concatenate([predictions, pred_padding])
                    confidence = np.concatenate([confidence, conf_padding])
                    embeddings = np.vstack([embeddings, emb_padding])
                elif len(predictions) > X.shape[0]:
                    # Truncate to match
                    predictions = predictions[:X.shape[0]]
                    confidence = confidence[:X.shape[0]]
                    embeddings = embeddings[:X.shape[0]]
                
                return {
                    'predictions': predictions,
                    'confidence': confidence,
                    'embeddings': embeddings
                }
            else:
                # Fallback model
                predictions = self.patchtst_model.predict(X_scaled)
                confidence = np.ones(len(predictions)) * 0.5
                embeddings = np.random.randn(len(predictions), self.config.export_dims) * 0.1
                
                return {
                    'predictions': predictions,
                    'confidence': confidence,
                    'embeddings': embeddings
                }
                
        except Exception as e:
            logger.error(f"❌ PatchTST feature extraction failed: {e}")
            # Return zeros as fallback
            return {
                'predictions': np.zeros(X.shape[0]),
                'confidence': np.zeros(X.shape[0]),
                'embeddings': np.zeros((X.shape[0], self.config.export_dims))
            }
    
    def get_oof_features(self) -> Dict[str, np.ndarray]:
        """Get out-of-fold features if available."""
        if not self.fitted or not self.config.include_oof_predictions:
            return {}
        
        return {
            'oof_predictions': self.oof_predictions,
            'oof_confidence': self.oof_confidence,
            'oof_embeddings': self.oof_embeddings
        }


# Factory function
def create_enhanced_patchtst(config: Optional[EnhancedPatchTSTConfig] = None) -> EnhancedPatchTSTModel:
    """Create Enhanced PatchTST model."""
    return EnhancedPatchTSTModel(config)