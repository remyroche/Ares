from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd

'\nAdvanced Neural Network Models for Multi-Output Training\n\nThis module provides implementations of advanced neural network architectures\nthat can be integrated with the multi-output training framework.\n'
import logging
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch import nn
logger = logging.getLogger(__name__)

class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series classification.

    Based on the paper: "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling" by Bai et al.
    """

    def __init__(self, input_size: int, num_channels: list[int], kernel_size: int=2, dropout: float=0.2, num_classes: int=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        num_levels = len(num_channels)
        layers = []
        in_channels = input_size
        for i in range(num_levels):
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=2 ** i, padding=(kernel_size - 1) * 2 ** i, dropout=dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(num_channels[-1], 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: Any) -> None:
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.classifier(x)

class TemporalBlock(nn.Module):
    """Temporal Block for TCN with residual connections."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float=0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: Any) -> None:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out

class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series classification.
    """

    def __init__(self, input_size: int, num_filters: list[int]=None, kernel_sizes: list[int]=None, dropout: float=0.2, num_classes: int=2) -> None:
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if num_filters is None:
            num_filters = [64, 128, 256]
        super().__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.num_classes = num_classes
        layers = []
        in_channels = input_size
        for _i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes, strict=False)):
            layers.extend([nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2), nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout), nn.MaxPool1d(2)])
            in_channels = filters
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Linear(num_filters[-1], 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: Any) -> None:
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for time series data.
    """

    def __init__(self, input_size: int, d_model: int=128, nhead: int=8, num_layers: int=4, dropout: float=0.1, num_classes: int=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: Any) -> None:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Any) -> None:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for time series data.
    """

    def __init__(self, input_size: int, hidden_size: int=128, num_layers: int=2, dropout: float=0.2, bidirectional: bool=True, num_classes: int=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional, batch_first=True)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(nn.Linear(lstm_output_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: Any) -> None:
        lstm_out, (hidden, cell) = self.lstm(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.classifier(hidden)

class GRUClassifier(nn.Module):
    """
    GRU-based classifier for time series data.
    """

    def __init__(self, input_size: int, hidden_size: int=128, num_layers: int=2, dropout: float=0.2, bidirectional: bool=True, num_classes: int=2) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional, batch_first=True)
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Sequential(nn.Linear(gru_output_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: Any) -> None:
        gru_out, hidden = self.gru(x)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.classifier(hidden)

class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make PyTorch models compatible with scikit-learn interface.
    """

    def __init__(self, model_class: type, model_params: dict[str, Any], device: str='auto', batch_size: int=32, epochs: int=100, learning_rate: float=0.001, early_stopping_patience: int=10) -> None:
        self.model_class = model_class
        self.model_params = model_params
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.classes_ = None

    def _get_device(self) -> None:
        """Get the appropriate device for training."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            return torch.device('cpu')
        return torch.device(self.device)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], sample_weight: Any=None) -> None:
        """Fit the neural network model."""
        X, y = check_X_y(X, y, multi_output=False)
        self.classes_ = unique_labels(y)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        device = self._get_device()
        self.model = self.model_class(**self.model_params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_loss = float('inf')
        patience_counter = 0
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = (batch_X.to(device), batch_y.to(device))
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.early_stopping_patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Predict class labels."""
        check_is_fitted(self, ['model', 'classes_'])
        X = check_array(X)
        X_tensor = torch.FloatTensor(X)
        device = self._get_device()
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, ['model', 'classes_'])
        X = check_array(X)
        X_tensor = torch.FloatTensor(X)
        device = self._get_device()
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

def create_neural_model(model_type: str, input_size: int, num_classes: int=2, **kwargs) -> NeuralNetworkWrapper:
    """
    Factory function to create neural network models.

    Args:
        model_type: Type of neural network ('tcn', 'cnn', 'transformer', 'lstm', 'gru')
        input_size: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        NeuralNetworkWrapper instance
    """
    if model_type.lower() == 'tcn':
        model_params = {'input_size': input_size, 'num_channels': kwargs.get('num_channels', [64, 128, 256]), 'kernel_size': kwargs.get('kernel_size', 2), 'dropout': kwargs.get('dropout', 0.2), 'num_classes': num_classes}
        return NeuralNetworkWrapper(TemporalConvNet, model_params, **kwargs)
    if model_type.lower() == 'cnn':
        model_params = {'input_size': input_size, 'num_filters': kwargs.get('num_filters', [64, 128, 256]), 'kernel_sizes': kwargs.get('kernel_sizes', [3, 3, 3]), 'dropout': kwargs.get('dropout', 0.2), 'num_classes': num_classes}
        return NeuralNetworkWrapper(CNN1D, model_params, **kwargs)
    if model_type.lower() == 'transformer':
        model_params = {'input_size': input_size, 'd_model': kwargs.get('d_model', 128), 'nhead': kwargs.get('nhead', 8), 'num_layers': kwargs.get('num_layers', 4), 'dropout': kwargs.get('dropout', 0.1), 'num_classes': num_classes}
        return NeuralNetworkWrapper(TransformerClassifier, model_params, **kwargs)
    if model_type.lower() == 'lstm':
        model_params = {'input_size': input_size, 'hidden_size': kwargs.get('hidden_size', 128), 'num_layers': kwargs.get('num_layers', 2), 'dropout': kwargs.get('dropout', 0.2), 'bidirectional': kwargs.get('bidirectional', True), 'num_classes': num_classes}
        return NeuralNetworkWrapper(LSTMClassifier, model_params, **kwargs)
    if model_type.lower() == 'gru':
        model_params = {'input_size': input_size, 'hidden_size': kwargs.get('hidden_size', 128), 'num_layers': kwargs.get('num_layers', 2), 'dropout': kwargs.get('dropout', 0.2), 'bidirectional': kwargs.get('bidirectional', True), 'num_classes': num_classes}
        return NeuralNetworkWrapper(GRUClassifier, model_params, **kwargs)
    msg = f'Unsupported model type: {model_type}'
    raise ValueError(msg)
NEURAL_MODEL_CONFIGS = {'tcn': {'num_channels': [64, 128, 256], 'kernel_size': 2, 'dropout': 0.2, 'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001}, 'cnn': {'num_filters': [64, 128, 256], 'kernel_sizes': [3, 3, 3], 'dropout': 0.2, 'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001}, 'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 4, 'dropout': 0.1, 'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001}, 'lstm': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.2, 'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001}, 'gru': {'hidden_size': 128, 'num_layers': 2, 'bidirectional': True, 'dropout': 0.2, 'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001}}