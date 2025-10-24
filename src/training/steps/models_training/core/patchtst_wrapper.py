"""
PatchTST Wrapper for Time Series Forecasting

This module provides a wrapper for PatchTST models with weighted loss support
for enhanced time series feature learning in trading strategies.

Key Features:
1. PatchTST-based temporal feature extraction
2. LightGBM for final prediction
3. Configurable PatchTST architecture
4. Sklearn-compatible API
5. Weighted loss support for negative learning approximation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
import logging

# Import tprint utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success, 
    tprint_debug, tprint_data_format, tprint_data_preview, LogLevel
)

from .error_handling import (
    handle_errors, validate_data, safe_import,
    MLModelTrainerError, DataValidationError, ModelTrainingError, PredictionError, ResourceError
)
from .weighted_loss_framework import (
    WeightedLossManager, WeightedLossConfig, WeightingStrategy
)

# Use safe imports with fallbacks
torch = safe_import('torch', 'torch')
nn = safe_import('torch.nn', 'torch')
optim = safe_import('torch.optim', 'torch')
DataLoader = safe_import('torch.utils.data.DataLoader', 'torch')
TensorDataset = safe_import('torch.utils.data.TensorDataset', 'torch')

TORCH_AVAILABLE = torch is not None

lgb = safe_import('lightgbm', 'lightgbm')
LIGHTGBM_AVAILABLE = lgb is not None

logger = logging.getLogger(__name__)

class PatchTSTEmbeddingLayer(nn.Module):
    """PatchTST-based embedding layer for temporal feature extraction."""
    
    def __init__(self, 
                 input_size: int,
                 patch_length: int = 16,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 embedding_dim: int = 32):
        super(PatchTSTEmbeddingLayer, self).__init__()
        
        self.input_size = input_size
        self.patch_length = patch_length
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_length, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Create patches
        patches = x.unfold(1, self.patch_length, self.patch_length // 2)
        patches = patches.contiguous().view(batch_size, -1, self.patch_length)
        
        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :patch_embeddings.size(1), :]
        patch_embeddings = patch_embeddings + pos_encoding
        
        # Transformer encoding
        transformer_output = self.transformer(patch_embeddings)
        
        # Global average pooling
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Project to embedding dimension
        embedding = self.output_projection(pooled_output)
        embedding = self.dropout(embedding)
        
        return embedding

class PatchTSTWrapper(BaseEstimator):
    """PatchTST with LightGBM wrapper."""
    
    def __init__(self, 
                 # PatchTST parameters
                 patch_length: int = 16,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 embedding_dim: int = 32,
                 sequence_length: int = 100,
                 patchtst_learning_rate: float = 0.001,
                 patchtst_epochs: int = 50,
                 patchtst_batch_size: int = 32,
                 
                 # LightGBM parameters
                 objective: str = "binary",
                 metric: str = "binary_logloss",
                 boosting_type: str = "gbdt",
                 num_leaves: int = 31,
                 learning_rate: float = 0.08,
                 feature_fraction: float = 0.8,
                 bagging_fraction: float = 0.8,
                 bagging_freq: int = 5,
                 verbose: int = -1,
                 random_state: int = 42,
                 n_estimators: int = 600,
                 early_stopping_rounds: int = 30,
                 min_data_in_leaf: int = 20,
                 min_gain_to_split: float = 0.0,
                 lambda_l1: float = 0.1,
                 lambda_l2: float = 0.1,
                 
                 # Training parameters
                 validation_split: float = 0.2,
                 device: str = "auto",
                 
                 # Weighted loss parameters
                 enable_weighted_loss: bool = True,
                 weighted_loss_config: Optional[Dict[str, Any]] = None):
        """Initialize PatchTST wrapper."""
        
        if not TORCH_AVAILABLE:
            tprint_error("PyTorch is required for PatchTSTWrapper. Install with: pip install torch")
            raise ImportError("PyTorch is required for PatchTSTWrapper. Install with: pip install torch")
        
        if not LIGHTGBM_AVAILABLE:
            tprint_error("LightGBM is required for PatchTSTWrapper. Install with: pip install lightgbm")
            raise ImportError("LightGBM is required for PatchTSTWrapper. Install with: pip install lightgbm")
        
        tprint_info(f"Initializing PatchTST wrapper with patch length: {patch_length}, embedding dim: {embedding_dim}")
        
        # PatchTST parameters
        self.patch_length = patch_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.patchtst_learning_rate = patchtst_learning_rate
        self.patchtst_epochs = patchtst_epochs
        self.patchtst_batch_size = patchtst_batch_size
        
        # LightGBM parameters
        self.objective = objective
        self.metric = metric
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.verbose = verbose
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.min_data_in_leaf = min_data_in_leaf
        self.min_gain_to_split = min_gain_to_split
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
        # Training parameters
        self.validation_split = validation_split
        self.device = device
        
        # Weighted loss parameters
        self.enable_weighted_loss = enable_weighted_loss
        self.weighted_loss_config = weighted_loss_config or {}
        self.weighted_loss_manager = None
        
        # Model components
        self.patchtst_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.n_features_in_ = None
        self.classes_ = None
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize weighted loss manager if enabled
        if self.enable_weighted_loss:
            config = WeightedLossConfig(**self.weighted_loss_config)
            self.weighted_loss_manager = WeightedLossManager(config)
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from time series data."""
        tprint_debug(f"Creating sequences with length {self.sequence_length} from data shape {X.shape}")
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        result = np.array(sequences)
        tprint_data_format(result, f"Created sequences", LogLevel.DEBUG)
        return result
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for PatchTST training."""
        tprint_debug(f"Preparing data for PatchTST training - X shape: {X.shape}, y shape: {y.shape if y is not None else None}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Prepare targets
        y_seq = None
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
        
        tprint_data_format(X_seq, "PatchTST input sequences", LogLevel.DEBUG)
        if y_seq is not None:
            tprint_data_format(y_seq, "PatchTST targets", LogLevel.DEBUG)
        
        return X_seq, y_seq
    
    def _build_patchtst_model(self, input_size: int) -> PatchTSTEmbeddingLayer:
        """Build PatchTST model for embeddings."""
        tprint_info(f"Building PatchTST model - input_size: {input_size}, d_model: {self.d_model}, embedding_dim: {self.embedding_dim}")
        model = PatchTSTEmbeddingLayer(
            input_size=input_size,
            patch_length=self.patch_length,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            embedding_dim=self.embedding_dim
        ).to(self.device)
        tprint_success(f"PatchTST model built and moved to device: {self.device}")
        return model
    
    def _train_patchtst(self, X_seq: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train PatchTST model and extract embeddings."""
        tprint_info(f"Starting PatchTST training - epochs: {self.patchtst_epochs}, batch_size: {self.patchtst_batch_size}, learning_rate: {self.patchtst_learning_rate}")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        tprint_data_format(X_tensor.cpu().numpy(), "Input tensor for PatchTST", LogLevel.DEBUG)
        tprint_data_format(y_tensor.cpu().numpy(), "Target tensor for PatchTST", LogLevel.DEBUG)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.patchtst_batch_size, shuffle=True)
        
        # Setup training
        optimizer = optim.Adam(self.patchtst_model.parameters(), lr=self.patchtst_learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.patchtst_model.train()
        for epoch in range(self.patchtst_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                embeddings = self.patchtst_model(batch_X)
                
                # Simple prediction (mean of embeddings)
                predictions = torch.mean(embeddings, dim=1)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                tprint_info(f"PatchTST Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
                logger.info(f"PatchTST Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Extract embeddings
        tprint_debug("Extracting PatchTST embeddings")
        self.patchtst_model.eval()
        with torch.no_grad():
            embeddings = self.patchtst_model(X_tensor).cpu().numpy()
        
        tprint_data_format(embeddings, "PatchTST embeddings extracted", LogLevel.DEBUG)
        tprint_success(f"PatchTST training completed - embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _build_lgb_model(self, is_classification: bool):
        """Build LightGBM model."""
        tprint_info(f"Building LightGBM model - classification: {is_classification}")
        
        if is_classification:
            model = lgb.LGBMClassifier(
                objective=self.objective,
                metric=self.metric,
                boosting_type=self.boosting_type,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                feature_fraction=self.feature_fraction,
                bagging_fraction=self.bagging_fraction,
                bagging_freq=self.bagging_freq,
                verbose=self.verbose,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                min_gain_to_split=self.min_gain_to_split,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2
            )
        else:
            model = lgb.LGBMRegressor(
                objective=self.objective,
                metric=self.metric,
                boosting_type=self.boosting_type,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                feature_fraction=self.feature_fraction,
                bagging_fraction=self.bagging_fraction,
                bagging_freq=self.bagging_freq,
                verbose=self.verbose,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                min_data_in_leaf=self.min_data_in_leaf,
                min_gain_to_split=self.min_gain_to_split,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2
            )
        
        tprint_success(f"LightGBM model built - type: {'Classifier' if is_classification else 'Regressor'}")
        return model
    
    @handle_errors(error_type=ModelTrainingError, reraise=True)
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'PatchTSTWrapper':
        """Fit the PatchTST-LightGBM model."""
        # Validate inputs
        validate_data(X, y, min_samples=self.sequence_length, min_features=1)
        if len(X) < self.sequence_length:
            raise DataValidationError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        tprint_info(f"Training PatchTST-LightGBM model - features: {self.n_features_in_}, samples: {len(X)}")
        
        # Determine if classification
        is_classification = len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer)
        tprint_info(f"Task type: {'Classification' if is_classification else 'Regression'}")
        
        if is_classification:
            self.classes_ = np.unique(y)
            tprint_data_format(self.classes_, "Classes for classification", LogLevel.DEBUG)
        
        # Prepare data
        X_seq, y_seq = self._prepare_data(X, y)
        
        if y_seq is None:
            tprint_error("Target data is required for training")
            raise DataValidationError("Target data is required for training")
        
        # Train PatchTST and get embeddings
        tprint_info("Training PatchTST for embeddings...")
        patchtst_embeddings = self._train_patchtst(X_seq, y_seq)
        
        # Combine original features with PatchTST embeddings
        # Use the last sequence_length features for each sample
        X_combined = np.hstack([
            X[self.sequence_length - 1:],  # Original features
            patchtst_embeddings  # PatchTST embeddings
        ])
        tprint_data_format(X_combined, "Combined features (original + PatchTST embeddings)", LogLevel.DEBUG)
        
        # Initialize weighted loss manager if enabled
        if self.enable_weighted_loss and self.weighted_loss_manager is not None:
            tprint_info("Initializing weighted loss manager...")
            self.weighted_loss_manager.fit(X_combined, y_seq)
        
        # Train LightGBM
        tprint_info("Training LightGBM...")
        self.lgb_model = self._build_lgb_model(is_classification)
        
        # Split for validation if needed
        if self.validation_split > 0:
            split_idx = int(len(X_combined) * (1 - self.validation_split))
            X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            tprint_info(f"Training with validation split - train: {len(X_train)}, val: {len(X_val)}")
            
            # Get sample weights if weighted loss is enabled
            sample_weight = None
            if self.enable_weighted_loss and self.weighted_loss_manager is not None:
                tprint_info("Calculating sample weights for training...")
                sample_weight = self.weighted_loss_manager.get_sample_weights(X_train, y_train)
                tprint_debug(f"Sample weight statistics - Mean: {np.mean(sample_weight):.3f}, Std: {np.std(sample_weight):.3f}")
            
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weight,
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            tprint_info("Training without validation split")
            
            # Get sample weights if weighted loss is enabled
            sample_weight = None
            if self.enable_weighted_loss and self.weighted_loss_manager is not None:
                tprint_info("Calculating sample weights for training...")
                sample_weight = self.weighted_loss_manager.get_sample_weights(X_combined, y_seq)
                tprint_debug(f"Sample weight statistics - Mean: {np.mean(sample_weight):.3f}, Std: {np.std(sample_weight):.3f}")
            
            self.lgb_model.fit(X_combined, y_seq, sample_weight=sample_weight)
        
        self.is_fitted = True
        tprint_success("PatchTST-LightGBM model training completed")
        logger.info("PatchTST-LightGBM model training completed")
        
        return self
    
    @handle_errors(error_type=PredictionError, reraise=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise PredictionError("Model must be fitted before prediction")
        
        validate_data(X, min_samples=1, min_features=self.n_features_in_)
        
        # Prepare data
        X_seq, _ = self._prepare_data(X)
        
        # Get PatchTST embeddings
        self.patchtst_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            patchtst_embeddings = self.patchtst_model(X_tensor).cpu().numpy()
        
        # Combine features
        X_combined = np.hstack([
            X[self.sequence_length - 1:],
            patchtst_embeddings
        ])
        
        # Make predictions
        predictions = self.lgb_model.predict(X_combined)
        tprint_data_format(predictions, "PatchTST-LightGBM predictions", LogLevel.DEBUG)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise PredictionError("Model must be fitted before prediction")
        
        if not hasattr(self.lgb_model, 'predict_proba'):
            raise PredictionError("Model does not support probability prediction")
        
        # Prepare data
        X_seq, _ = self._prepare_data(X)
        
        # Get PatchTST embeddings
        self.patchtst_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            patchtst_embeddings = self.patchtst_model(X_tensor).cpu().numpy()
        
        # Combine features
        X_combined = np.hstack([
            X[self.sequence_length - 1:],
            patchtst_embeddings
        ])
        
        # Make probability predictions
        probabilities = self.lgb_model.predict_proba(X_combined)
        tprint_data_format(probabilities, "PatchTST-LightGBM probabilities", LogLevel.DEBUG)
        
        return probabilities
    
    def get_patchtst_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get PatchTST embeddings for analysis."""
        if not self.is_fitted:
            raise PredictionError("Model must be fitted before getting embeddings")
        
        X_seq, _ = self._prepare_data(X)
        
        self.patchtst_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            embeddings = self.patchtst_model(X_tensor).cpu().numpy()
        
        return embeddings

# Factory function
def create_patchtst_wrapper(**kwargs) -> PatchTSTWrapper:
    """Create PatchTST wrapper with default parameters."""
    return PatchTSTWrapper(**kwargs)