"""
LightGBM with GRU Embeddings Wrapper

This module provides a hybrid model that combines LightGBM with GRU-based embeddings
for enhanced time series feature learning in trading strategies.

Key Features:
1. GRU-based temporal feature extraction
2. LightGBM for final prediction
3. Configurable GRU architecture
4. Sklearn-compatible API
5. Memory-efficient training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
import logging

from .error_handling import (
    handle_errors, validate_data, safe_import,
    MLModelTrainerError, DataValidationError, ModelTrainingError, PredictionError, ResourceError
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

class GRUEmbeddingLayer(nn.Module):
    """GRU-based embedding layer for temporal feature extraction."""
    
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2, 
                 dropout: float = 0.2, bidirectional: bool = False, embedding_dim: int = 16):
        super(GRUEmbeddingLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_dim = embedding_dim
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection to embedding dimension
        gru_output_size = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(gru_output_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)
        
        # Use the last output from GRU
        last_output = gru_out[:, -1, :]  # (batch_size, gru_output_size)
        
        # Project to embedding dimension
        embedding = self.projection(last_output)
        embedding = self.dropout(embedding)
        
        return embedding

class LGBMGRUWrapper(BaseEstimator):
    """LightGBM with GRU embeddings wrapper."""
    
    def __init__(self, 
                 # GRU parameters
                 gru_hidden_size: int = 32,
                 gru_num_layers: int = 2,
                 gru_dropout: float = 0.2,
                 gru_bidirectional: bool = False,
                 embedding_dim: int = 16,
                 sequence_length: int = 20,
                 gru_learning_rate: float = 0.002,
                 gru_epochs: int = 30,
                 gru_batch_size: int = 64,
                 
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
                 device: str = "auto"):
        """Initialize LGBM-GRU wrapper."""
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LGBMGRUWrapper. Install with: pip install torch")
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required for LGBMGRUWrapper. Install with: pip install lightgbm")
        
        # GRU parameters
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gru_dropout = gru_dropout
        self.gru_bidirectional = gru_bidirectional
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.gru_learning_rate = gru_learning_rate
        self.gru_epochs = gru_epochs
        self.gru_batch_size = gru_batch_size
        
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
        
        # Model components
        self.gru_model = None
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
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from time series data."""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare data for GRU training."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if not self.is_fitted else self.scaler.transform(X)
        
        # Create sequences
        X_seq = self._create_sequences(X_scaled)
        
        # Align targets with sequences
        y_seq = None
        if y is not None:
            y_seq = y[self.sequence_length - 1:]
        
        return X_seq, y_seq
    
    def _build_gru_model(self, input_size: int) -> GRUEmbeddingLayer:
        """Build GRU model for embeddings."""
        return GRUEmbeddingLayer(
            input_size=input_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=self.gru_dropout,
            bidirectional=self.gru_bidirectional,
            embedding_dim=self.embedding_dim
        ).to(self.device)
    
    def _train_gru(self, X_seq: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Train GRU model and extract embeddings."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.gru_batch_size, shuffle=True)
        
        # Build model
        self.gru_model = self._build_gru_model(X_seq.shape[2])
        
        # Setup training
        optimizer = optim.Adam(self.gru_model.parameters(), lr=self.gru_learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.gru_model.train()
        for epoch in range(self.gru_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Get embeddings
                embeddings = self.gru_model(batch_X)
                
                # Simple regression head for training
                if embeddings.shape[1] == 1:
                    predictions = embeddings.squeeze()
                else:
                    # Use mean of embeddings as prediction
                    predictions = torch.mean(embeddings, dim=1)
                
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"GRU Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Extract embeddings
        self.gru_model.eval()
        with torch.no_grad():
            embeddings = self.gru_model(X_tensor).cpu().numpy()
        
        return embeddings
    
    def _build_lgb_model(self, is_classification: bool = True):
        """Build LightGBM model."""
        if is_classification:
            return lgb.LGBMClassifier(
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
            return lgb.LGBMRegressor(
                objective="regression",
                metric="rmse",
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
    
    @handle_errors(error_type=ModelTrainingError, reraise=True)
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'LGBMGRUWrapper':
        """Fit the LGBM-GRU model."""
        # Validate inputs
        validate_data(X, y, min_samples=self.sequence_length, min_features=1)
        if len(X) < self.sequence_length:
            raise DataValidationError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        
        # Determine if classification
        is_classification = len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer)
        
        if is_classification:
            self.classes_ = np.unique(y)
        
        # Prepare data
        X_seq, y_seq = self._prepare_data(X, y)
        
        if y_seq is None:
            raise DataValidationError("Target data is required for training")
        
        # Train GRU and get embeddings
        logger.info("Training GRU for embeddings...")
        gru_embeddings = self._train_gru(X_seq, y_seq)
        
        # Combine original features with GRU embeddings
        # Use the last sequence_length features for each sample
        X_combined = np.hstack([
            X[self.sequence_length - 1:],  # Original features
            gru_embeddings  # GRU embeddings
        ])
        
        # Train LightGBM
        logger.info("Training LightGBM...")
        self.lgb_model = self._build_lgb_model(is_classification)
        
        # Split for validation if needed
        if self.validation_split > 0:
            split_idx = int(len(X_combined) * (1 - self.validation_split))
            X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.lgb_model.fit(X_combined, y_seq)
        
        self.is_fitted = True
        logger.info("LGBM-GRU model training completed")
        
        return self
    
    @handle_errors(error_type=PredictionError, reraise=True)
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise PredictionError("Model must be fitted before prediction")
        
        validate_data(X, min_samples=1, min_features=self.n_features_in_)
        
        # Prepare data
        X_seq, _ = self._prepare_data(X)
        
        # Get GRU embeddings
        self.gru_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            gru_embeddings = self.gru_model(X_tensor).cpu().numpy()
        
        # Combine features
        X_combined = np.hstack([
            X[self.sequence_length - 1:],
            gru_embeddings
        ])
        
        # Make predictions
        return self.lgb_model.predict(X_combined)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not hasattr(self.lgb_model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        # Prepare data
        X_seq, _ = self._prepare_data(X)
        
        # Get GRU embeddings
        self.gru_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            gru_embeddings = self.gru_model(X_tensor).cpu().numpy()
        
        # Combine features
        X_combined = np.hstack([
            X[self.sequence_length - 1:],
            gru_embeddings
        ])
        
        # Make probability predictions
        return self.lgb_model.predict_proba(X_combined)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from LightGBM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.lgb_model.feature_importances_
    
    def get_gru_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get GRU embeddings for analysis."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting embeddings")
        
        X_seq, _ = self._prepare_data(X)
        
        self.gru_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            embeddings = self.gru_model(X_tensor).cpu().numpy()
        
        return embeddings

# Factory function
def create_lgbm_gru_wrapper(**kwargs) -> LGBMGRUWrapper:
    """Create LGBM-GRU wrapper with default parameters."""
    return LGBMGRUWrapper(**kwargs)