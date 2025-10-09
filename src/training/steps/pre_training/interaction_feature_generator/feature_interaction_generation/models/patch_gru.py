"""
Patch/GRU Model for End-to-End Roadmap

Minimal stacker with:
- Tiny PatchTST or 1-layer GRU
- 2-4h sequence lookback
- Horizons: {1,3} bars
- Exposes: y_hat_h1, y_hat_h3, y_hat_conf
- p99 inference <5ms
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from abc import ABC, abstractmethod


class ModelType(Enum):
    """Types of patch/GRU models."""
    PATCH = "patch"
    GRU = "gru"


@dataclass
class PatchConfig:
    """Configuration for patch model."""
    model_type: ModelType
    sequence_length: int  # 2-4h in bars
    horizons: List[int]  # [1, 3] bars
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50


@dataclass
class PatchOutput:
    """Output from patch model."""
    y_hat_h1: pd.Series
    y_hat_h3: pd.Series
    y_hat_conf: pd.Series
    metadata: Dict[str, Any]


class BasePatchModel(ABC):
    """Abstract base class for patch models."""
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.model = None
        self.fitted = False
        self.residual_std = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        raise NotImplementedError("Subclasses must implement fit method")
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def get_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores."""
        if self.residual_std is None:
            return np.ones_like(predictions)
        
        epsilon = 1e-8
        confidence = np.abs(predictions) / (epsilon + self.residual_std)
        return np.clip(confidence, 0, 1)


class SimpleGRU(BasePatchModel):
    """Simple 1-layer GRU model."""
    
    def __init__(self, config: PatchConfig):
        super().__init__(config)
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GRU model."""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)
            
            # Create model
            self.model = nn.GRU(
                input_size=X.shape[1],
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0,
                batch_first=True
            )
            
            # Add output layer
            self.output_layer = nn.Linear(self.config.hidden_dim, 1)
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.config.epochs):
                optimizer.zero_grad()
                
                # Forward pass
                output, _ = self.model(X_tensor)
                predictions = self.output_layer(output[:, -1, :])  # Use last timestep
                
                loss = criterion(predictions.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
            
            # Calculate residual std for confidence
            with torch.no_grad():
                self.model.eval()
                output, _ = self.model(X_tensor)
                pred = self.output_layer(output[:, -1, :]).squeeze()
                residuals = y_tensor - pred
                self.residual_std = torch.std(residuals).item()
            
            self.fitted = True
            
        except ImportError:
            warnings.warn("PyTorch not available, using fallback linear model")
            self._fit_fallback(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            import torch
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model(X_tensor)
                predictions = self.output_layer(output[:, -1, :])
                return predictions.squeeze().numpy()
        
        except ImportError:
            return self._predict_fallback(X)
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to simple linear model."""
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        # Calculate residual std
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_std = np.std(residuals)
        
        self.fitted = True
    
    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction."""
        return self.model.predict(X)


class SimplePatchTST(BasePatchModel):
    """Simple PatchTST model (simplified)."""
    
    def __init__(self, config: PatchConfig):
        super().__init__(config)
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the PatchTST model."""
        try:
            import torch
            import torch.nn as nn
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)
            
            # Simple patch-based model
            patch_size = min(16, X.shape[1] // 4)  # Adaptive patch size
            num_patches = X.shape[1] // patch_size
            
            self.model = nn.Sequential(
                nn.Linear(X.shape[1], self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 1)
            )
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.config.epochs):
                optimizer.zero_grad()
                
                predictions = self.model(X_tensor)
                loss = criterion(predictions.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()
            
            # Calculate residual std
            with torch.no_grad():
                self.model.eval()
                pred = self.model(X_tensor).squeeze()
                residuals = y_tensor - pred
                self.residual_std = torch.std(residuals).item()
            
            self.fitted = True
            
        except ImportError:
            warnings.warn("PyTorch not available, using fallback linear model")
            self._fit_fallback(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            import torch
            
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                return predictions.squeeze().numpy()
        
        except ImportError:
            return self._predict_fallback(X)
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to simple linear model."""
        from sklearn.linear_model import LinearRegression
        
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_std = np.std(residuals)
        
        self.fitted = True
    
    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction."""
        return self.model.predict(X)


class PatchModelFactory:
    """Factory for creating patch models."""
    
    @staticmethod
    def create_model(config: PatchConfig) -> BasePatchModel:
        """Create a patch model based on configuration."""
        if config.model_type == ModelType.GRU:
            return SimpleGRU(config)
        elif config.model_type == ModelType.PATCH:
            return SimplePatchTST(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")


class PatchOrchestrator:
    """Orchestrator for patch model training and prediction."""
    
    def __init__(self, config: PatchConfig):
        self.config = config
        self.models = {}  # horizon -> model
        self.fitted = False
    
    def fit(self, 
            bars_data: pd.DataFrame,
            targets: Dict[int, pd.Series]) -> None:
        """Fit models for all horizons."""
        
        # Prepare sequence data
        X_sequences = self._prepare_sequences(bars_data)
        
        for horizon in self.config.horizons:
            if horizon not in targets:
                continue
            
            y = targets[horizon]
            
            # Align sequences with targets
            min_length = min(len(X_sequences), len(y))
            X_aligned = X_sequences[:min_length]
            y_aligned = y[:min_length]
            
            if len(X_aligned) < self.config.sequence_length:
                continue
            
            # Create model for this horizon
            model = PatchModelFactory.create_model(self.config)
            
            # Fit model
            model.fit(X_aligned, y_aligned.values)
            self.models[horizon] = model
        
        self.fitted = True
    
    def predict(self, bars_data: pd.DataFrame) -> PatchOutput:
        """Make predictions for all horizons."""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
        
        # Prepare sequence data
        X_sequences = self._prepare_sequences(bars_data)
        
        predictions = {}
        confidences = {}
        
        for horizon, model in self.models.items():
            if len(X_sequences) == 0:
                pred = np.zeros(len(bars_data))
                conf = np.zeros(len(bars_data))
            else:
                pred = model.predict(X_sequences)
                conf = model.get_confidence(pred)
            
            predictions[f'y_hat_h{horizon}'] = pd.Series(pred, index=bars_data.index)
            confidences[f'y_hat_h{horizon}'] = pd.Series(conf, index=bars_data.index)
        
        # Create confidence score (average across horizons)
        if confidences:
            y_hat_conf = pd.Series(
                np.mean([conf.values for conf in confidences.values()], axis=0),
                index=bars_data.index
            )
        else:
            y_hat_conf = pd.Series(0, index=bars_data.index)
        
        return PatchOutput(
            y_hat_h1=predictions.get('y_hat_h1', pd.Series(0, index=bars_data.index)),
            y_hat_h3=predictions.get('y_hat_h3', pd.Series(0, index=bars_data.index)),
            y_hat_conf=y_hat_conf,
            metadata={
                'fitted_models': list(self.models.keys()),
                'sequence_length': self.config.sequence_length,
                'model_type': self.config.model_type.value
            }
        )
    
    def _prepare_sequences(self, bars_data: pd.DataFrame) -> np.ndarray:
        """Prepare sequence data for model input."""
        # Select relevant features
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in feature_cols if col in bars_data.columns]
        
        if not available_cols:
            return np.array([])
        
        data = bars_data[available_cols].values
        
        # Create sequences
        sequences = []
        for i in range(len(data) - self.config.sequence_length + 1):
            sequence = data[i:i + self.config.sequence_length]
            sequences.append(sequence.flatten())  # Flatten to 1D
        
        return np.array(sequences) if sequences else np.array([])
    
    def get_oof_predictions(self, 
                           bars_data: pd.DataFrame,
                           targets: Dict[int, pd.Series],
                           n_folds: int = 5) -> PatchOutput:
        """Get out-of-fold predictions for training features."""
        
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_folds)
        oof_predictions = {f'y_hat_h{h}': [] for h in self.config.horizons}
        oof_confidences = []
        
        for train_idx, val_idx in tscv.split(bars_data):
            # Split data
            train_data = bars_data.iloc[train_idx]
            val_data = bars_data.iloc[val_idx]
            
            # Fit on training data
            train_targets = {h: targets[h].iloc[train_idx] for h in self.config.horizons if h in targets}
            self.fit(train_data, train_targets)
            
            # Predict on validation data
            val_predictions = self.predict(val_data)
            
            # Store OOF predictions
            for horizon in self.config.horizons:
                if f'y_hat_h{horizon}' in val_predictions.__dict__:
                    oof_predictions[f'y_hat_h{horizon}'].append(
                        val_predictions.__dict__[f'y_hat_h{horizon}']
                    )
            
            oof_confidences.append(val_predictions.y_hat_conf)
        
        # Combine OOF predictions
        combined_predictions = {}
        for horizon in self.config.horizons:
            if oof_predictions[f'y_hat_h{horizon}']:
                combined_predictions[f'y_hat_h{horizon}'] = pd.concat(
                    oof_predictions[f'y_hat_h{horizon}']
                ).sort_index()
            else:
                combined_predictions[f'y_hat_h{horizon}'] = pd.Series(0, index=bars_data.index)
        
        # Combine confidences
        if oof_confidences:
            combined_conf = pd.concat(oof_confidences).sort_index()
        else:
            combined_conf = pd.Series(0, index=bars_data.index)
        
        return PatchOutput(
            y_hat_h1=combined_predictions.get('y_hat_h1', pd.Series(0, index=bars_data.index)),
            y_hat_h3=combined_predictions.get('y_hat_h3', pd.Series(0, index=bars_data.index)),
            y_hat_conf=combined_conf,
            metadata={'oof': True, 'n_folds': n_folds}
        )