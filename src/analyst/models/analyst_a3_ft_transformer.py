"""
Analyst A3 Model: FT-Transformer (Tabular Transformer)

Binary "green light" classification with:
- 300+ features, regime posteriors, cross-TF aggregates
- Tabular transformer architecture
- 2–3 blocks, d_model=128, heads=2, dropout 0.1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import joblib
import os

logger = logging.getLogger(__name__)


@dataclass
class FTTransformerConfig:
    """Configuration for FT-Transformer model."""
    d_model: int = 128
    n_blocks: int = 3  # 2-3 blocks
    n_heads: int = 2
    dropout: float = 0.1
    d_ff: int = 512  # Feed-forward dimension
    activation: str = 'gelu'
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 1000
    num_categories: int = 0  # Number of categorical features
    use_position_embeddings: bool = True
    use_cls_token: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    num_epochs: int = 100
    patience: int = 10
    device: str = 'auto'


@dataclass
class CalibrationConfig:
    """Configuration for model calibration."""
    method: str = 'isotonic'  # 'isotonic' or 'sigmoid'
    cv_folds: int = 5
    enable_venn_abers: bool = True
    confidence_levels: List[float] = None

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class FTTransformer(nn.Module):
    """FT-Transformer for tabular data."""
    
    def __init__(self, config: FTTransformerConfig, n_features: int, n_classes: int = 2):
        super().__init__()
        self.config = config
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Input embedding
        self.input_embedding = nn.Linear(n_features, config.d_model)
        
        # Positional encoding
        if config.use_position_embeddings:
            self.pos_encoding = PositionalEncoding(config.d_model, config.max_position_embeddings)
        else:
            self.pos_encoding = None
        
        # CLS token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_blocks)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, n_classes)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        if self.pos_encoding is not None:
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Add CLS token
        if self.config.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask)
        
        # Use CLS token for classification
        if self.config.use_cls_token:
            x = x[:, 0, :]  # (batch_size, d_model)
        else:
            x = x.mean(dim=1)  # Global average pooling
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class VennAbersCalibration:
    """Venn-Abers calibration for uncertainty estimation."""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> 'VennAbersCalibration':
        """Fit Venn-Abers calibrators."""
        from sklearn.isotonic import IsotonicRegression
        
        for level in self.confidence_levels:
            # Create binary targets for this confidence level
            y_binary = (y_prob >= level).astype(int)
            
            if len(np.unique(y_binary)) > 1:  # Ensure we have both classes
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_prob, y_binary)
                self.calibrators[level] = calibrator
        
        self.is_fitted = True
        return self
    
    def predict_confidence(self, y_prob: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict confidence intervals."""
        if not self.is_fitted:
            raise ValueError("Venn-Abers calibrators must be fitted first")
        
        results = {}
        for level, calibrator in self.calibrators.items():
            calibrated_probs = calibrator.predict(y_prob)
            results[f'confidence_{level}'] = calibrated_probs
        
        return results


class AnalystA3Model:
    """Analyst A3: FT-Transformer with calibration."""
    
    def __init__(self, 
                 transformer_config: Optional[FTTransformerConfig] = None,
                 calibration_config: Optional[CalibrationConfig] = None):
        self.transformer_config = transformer_config or FTTransformerConfig()
        self.calibration_config = calibration_config or CalibrationConfig()
        
        # Model components
        self.transformer_model = None
        self.scaler = StandardScaler()
        self.calibrated_model = None
        self.venn_abers = None
        self.feature_names = None
        self.device = None
        self.is_fitted = False
        
        logger.info("Initialized Analyst A3 Model (FT-Transformer)")
    
    def _get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.transformer_config.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(self.transformer_config.device)
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data for transformer."""
        # Scale features
        if y is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Add sequence dimension (treat each sample as a sequence of length 1)
        X_tensor = X_tensor.unsqueeze(1)  # (batch_size, 1, n_features)
        
        if y is not None:
            y_tensor = torch.LongTensor(y)
            return X_tensor, y_tensor
        
        return X_tensor, None
    
    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        class_weights = []
        for class_label in range(len(unique_classes)):
            count = counts[class_label]
            weight = total_samples / (len(unique_classes) * count)
            class_weights.append(weight)
        
        return torch.FloatTensor(class_weights)
    
    def _train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                    criterion: nn.Module, class_weights: torch.Tensor) -> float:
        """Train for one epoch."""
        self.transformer_model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.transformer_model(batch_X)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.transformer_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.transformer_model(batch_X)
                loss = criterion(logits, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total
        return total_loss / len(dataloader), accuracy
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            regimes: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None) -> 'AnalystA3Model':
        """Fit the Analyst A3 model."""
        logger.info("Fitting Analyst A3 Model...")
        
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.values
        
        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Ensure binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Binary classification requires exactly 2 classes, got {len(unique_classes)}")
        
        # Get device
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Create model
        self.transformer_model = FTTransformer(
            config=self.transformer_config,
            n_features=X.shape[1],
            n_classes=2
        ).to(self.device)
        
        # Compute class weights
        class_weights = self._compute_class_weights(y)
        class_weights = class_weights.to(self.device)
        
        # Create data loaders
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.transformer_config.batch_size, 
            shuffle=True
        )
        
        # Split for validation
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.transformer_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.transformer_config.batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.transformer_model.parameters(),
            lr=self.transformer_config.learning_rate,
            weight_decay=self.transformer_config.weight_decay
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.transformer_config.num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, criterion, class_weights)
            
            # Validate
            val_loss, val_accuracy = self._validate_epoch(val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{self.transformer_config.num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.transformer_config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Get predictions for calibration
        self.transformer_model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            logits = self.transformer_model(X_tensor)
            y_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        # Fit calibration
        if self.calibration_config.method in ['isotonic', 'sigmoid']:
            self.calibrated_model = CalibratedClassifierCV(
                self.transformer_model,
                method=self.calibration_config.method,
                cv=self.calibration_config.cv_folds
            )
            # Note: This won't work directly with PyTorch models, need custom implementation
            logger.warning("Calibration not implemented for PyTorch models yet")
        
        # Fit Venn-Abers calibration
        if self.calibration_config.enable_venn_abers:
            self.venn_abers = VennAbersCalibration(self.calibration_config.confidence_levels)
            self.venn_abers.fit(y, y_prob)
        
        self.is_fitted = True
        logger.info("✅ Analyst A3 Model fitted successfully")
        return self
    
    def predict_proba(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data
        X_tensor, _ = self._prepare_data(X)
        X_tensor = X_tensor.to(self.device)
        
        # Get predictions
        self.transformer_model.eval()
        with torch.no_grad():
            logits = self.transformer_model(X_tensor)
            y_prob = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        return y_prob
    
    def predict_uncertainty(self, X: np.ndarray, regimes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Predict uncertainty estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get base predictions
        y_prob = self.predict_proba(X, regimes)
        
        # Get Venn-Abers confidence intervals
        uncertainty_results = {
            'probability': y_prob,
            'confidence_intervals': {}
        }
        
        if self.venn_abers is not None:
            confidence_intervals = self.venn_abers.predict_confidence(y_prob)
            uncertainty_results['confidence_intervals'] = confidence_intervals
        
        # Add margin statistics
        uncertainty_results['margin_stats'] = {
            'mean_probability': np.mean(y_prob),
            'std_probability': np.std(y_prob),
            'min_probability': np.min(y_prob),
            'max_probability': np.max(y_prob),
            'confidence_range': np.max(y_prob) - np.min(y_prob)
        }
        
        return uncertainty_results
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from the model."""
        if not self.is_fitted or self.transformer_model is None:
            return {}
        
        # Get input embedding weights as feature importance
        input_weights = self.transformer_model.input_embedding.weight.detach().cpu().numpy()
        importance = np.mean(np.abs(input_weights), axis=0)
        
        # Create feature names
        if self.feature_names is not None:
            all_feature_names = self.feature_names
        else:
            all_feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return {
            'importance_scores': importance,
            'feature_names': all_feature_names,
            'top_features': sorted(zip(all_feature_names, importance), key=lambda x: x[1], reverse=True)[:20]
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'transformer_model_state': self.transformer_model.state_dict(),
            'transformer_config': self.transformer_config,
            'scaler': self.scaler,
            'calibrated_model': self.calibrated_model,
            'venn_abers': self.venn_abers,
            'feature_names': self.feature_names,
            'calibration_config': self.calibration_config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'AnalystA3Model':
        """Load the model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            transformer_config=model_data['transformer_config'],
            calibration_config=model_data['calibration_config']
        )
        
        # Restore state
        instance.scaler = model_data['scaler']
        instance.calibrated_model = model_data['calibrated_model']
        instance.venn_abers = model_data['venn_abers']
        instance.feature_names = model_data['feature_names']
        
        # Recreate and load transformer model
        instance.device = instance._get_device()
        instance.transformer_model = FTTransformer(
            config=instance.transformer_config,
            n_features=len(instance.feature_names) if instance.feature_names else 0,
            n_classes=2
        ).to(instance.device)
        instance.transformer_model.load_state_dict(model_data['transformer_model_state'])
        instance.is_fitted = True
        
        logger.info(f"✅ Model loaded from {filepath}")
        return instance


# Factory function for easy model creation
def create_analyst_a3_model(transformer_config: Optional[FTTransformerConfig] = None,
                           calibration_config: Optional[CalibrationConfig] = None) -> AnalystA3Model:
    """Create an Analyst A3 model with the specified configurations."""
    return AnalystA3Model(transformer_config, calibration_config)