"""
Advanced Neural Network Models for Multi-Output Training

This module provides implementations of advanced neural network architectures
that can be integrated with the multi-output training framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import logging

logger = logging.getLogger(__name__)


class TemporalConvNet(nn.Module):
    """
Temporal Convolutional Network (TCN) for time series classification.

Based on the paper: "An Empirical Evaluation of Generic Convolutional and
Recurrent Networks for Sequence Modeling" by Bai et al.
"""

def __init__(
self,
input_size: int,
num_channels: List[int],
kernel_size: int = 2,
dropout: float = 0.2,
num_classes: int = 2
):
        super(TemporalConvNet, self).__init__()

self.input_size = input_size
self.num_channels = num_channels
self.kernel_size = kernel_size
self.dropout = dropout
self.num_classes = num_classes

# Calculate the number of layers
num_levels = len(num_channels)

# Create temporal convolution layers
layers = []
in_channels = input_size

for i in range(num_levels):
            out_channels = num_channels[i]
layers.append(
TemporalBlock(
in_channels, out_channels, kernel_size,
stride=1, dilation=2**i, padding=(kernel_size-1) * 2**i,
dropout=dropout
)
)
in_channels = out_channels

self.tcn = nn.Sequential(*layers)

# Final classification layer
self.classifier = nn.Sequential(
nn.Linear(num_channels[-1], 128),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(128, num_classes)
)

def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
# TCN expects: (batch_size, input_size, sequence_length)
x = x.transpose(1, 2)

# Apply TCN layers
x = self.tcn(x)

# Global average pooling over time dimension
x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

# Classification
x = self.classifier(x)
return x


class TemporalBlock(nn.Module):
    """Temporal Block for TCN with residual connections."""

def __init__(
self,
in_channels: int,
out_channels: int,
kernel_size: int,
stride: int,
dilation: int,
padding: int,
dropout: float = 0.2
):
        super(TemporalBlock, self).__init__()

self.conv1 = nn.Conv1d(
in_channels, out_channels, kernel_size,
stride=stride, padding=padding, dilation=dilation
)
self.conv2 = nn.Conv1d(
out_channels, out_channels, kernel_size,
stride=stride, padding=padding, dilation=dilation
)

self.relu = nn.ReLU()
self.dropout = nn.Dropout(dropout)

# Residual connection
self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

def forward(self, x):
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

def __init__(
self,
input_size: int,
num_filters: List[int] = [64, 128, 256],
kernel_sizes: List[int] = [3, 3, 3],
dropout: float = 0.2,
num_classes: int = 2
):
        super(CNN1D, self).__init__()

self.input_size = input_size
self.num_filters = num_filters
self.kernel_sizes = kernel_sizes
self.dropout = dropout
self.num_classes = num_classes

# Convolutional layers
layers = []
in_channels = input_size

for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            layers.extend([
nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
nn.BatchNorm1d(filters),
nn.ReLU(),
nn.Dropout(dropout),
nn.MaxPool1d(2)
])
in_channels = filters

self.conv_layers = nn.Sequential(*layers)

# Global average pooling
self.global_pool = nn.AdaptiveAvgPool1d(1)

# Classification layers
self.classifier = nn.Sequential(
nn.Linear(num_filters[-1], 128),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(128, num_classes)
)

def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
# CNN expects: (batch_size, input_size, sequence_length)
x = x.transpose(1, 2)

# Convolutional layers
x = self.conv_layers(x)

# Global pooling
x = self.global_pool(x).squeeze(-1)

# Classification
x = self.classifier(x)
return x


class TransformerClassifier(nn.Module):
    """
Transformer-based classifier for time series data.
"""

def __init__(
self,
input_size: int,
d_model: int = 128,
nhead: int = 8,
num_layers: int = 4,
dropout: float = 0.1,
num_classes: int = 2
):
        super(TransformerClassifier, self).__init__()

self.input_size = input_size
self.d_model = d_model
self.nhead = nhead
self.num_layers = num_layers
self.dropout = dropout
self.num_classes = num_classes

# Input projection
self.input_projection = nn.Linear(input_size, d_model)

# Positional encoding
self.pos_encoder = PositionalEncoding(d_model, dropout)

# Transformer encoder
encoder_layer = nn.TransformerEncoderLayer(
d_model=d_model,
nhead=nhead,
dim_feedforward=d_model * 4,
dropout=dropout,
batch_first=True
)
self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

# Classification head
self.classifier = nn.Sequential(
nn.Linear(d_model, 128),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(128, num_classes)
)

def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

# Project to d_model dimensions
x = self.input_projection(x)

# Add positional encoding
x = self.pos_encoder(x)

# Transformer encoding
x = self.transformer_encoder(x)

# Global average pooling
x = x.mean(dim=1)

# Classification
x = self.classifier(x)
return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
self.dropout = nn.Dropout(p=dropout)

pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)

self.register_buffer('pe', pe)

def forward(self, x):
        x = x + self.pe[:x.size(0), :]
return self.dropout(x)


class LSTMClassifier(nn.Module):
    """
LSTM-based classifier for time series data.
"""

def __init__(
self,
input_size: int,
hidden_size: int = 128,
num_layers: int = 2,
dropout: float = 0.2,
bidirectional: bool = True,
num_classes: int = 2
):
        super(LSTMClassifier, self).__init__()

self.input_size = input_size
self.hidden_size = hidden_size
self.num_layers = num_layers
self.dropout = dropout
self.bidirectional = bidirectional
self.num_classes = num_classes

# LSTM layer
self.lstm = nn.LSTM(
input_size=input_size,
hidden_size=hidden_size,
num_layers=num_layers,
dropout=dropout if num_layers > 1 else 0,
bidirectional=bidirectional,
batch_first=True
)

# Calculate output size
lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

# Classification layers
self.classifier = nn.Sequential(
nn.Linear(lstm_output_size, 128),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(128, num_classes)
)

def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

# LSTM forward pass
lstm_out, (hidden, cell) = self.lstm(x)

# Use the last hidden state for classification
if self.bidirectional:
            # Concatenate forward and backward hidden states
hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
else:
            hidden = hidden[-1]

# Classification
x = self.classifier(hidden)
return x


class GRUClassifier(nn.Module):
    """
GRU-based classifier for time series data.
"""

def __init__(
self,
input_size: int,
hidden_size: int = 128,
num_layers: int = 2,
dropout: float = 0.2,
bidirectional: bool = True,
num_classes: int = 2
):
        super(GRUClassifier, self).__init__()

self.input_size = input_size
self.hidden_size = hidden_size
self.num_layers = num_layers
self.dropout = dropout
self.bidirectional = bidirectional
self.num_classes = num_classes

# GRU layer
self.gru = nn.GRU(
input_size=input_size,
hidden_size=hidden_size,
num_layers=num_layers,
dropout=dropout if num_layers > 1 else 0,
bidirectional=bidirectional,
batch_first=True
)

# Calculate output size
gru_output_size = hidden_size * 2 if bidirectional else hidden_size

# Classification layers
self.classifier = nn.Sequential(
nn.Linear(gru_output_size, 128),
nn.ReLU(),
nn.Dropout(dropout),
nn.Linear(128, num_classes)
)

def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

# GRU forward pass
gru_out, hidden = self.gru(x)

# Use the last hidden state for classification
if self.bidirectional:
            # Concatenate forward and backward hidden states
hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
else:
            hidden = hidden[-1]

# Classification
x = self.classifier(hidden)
return x


class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """
Wrapper class to make PyTorch models compatible with scikit-learn interface.
"""

def __init__(
self,
model_class: type,
model_params: Dict[str, Any],
device: str = 'auto',
batch_size: int = 32,
epochs: int = 100,
learning_rate: float = 0.001,
early_stopping_patience: int = 10
):
        self.model_class = model_class
self.model_params = model_params
self.device = device
self.batch_size = batch_size
self.epochs = epochs
self.learning_rate = learning_rate
self.early_stopping_patience = early_stopping_patience
self.model = None
self.classes_ = None

def _get_device(self):
        """Get the appropriate device for training."""
if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
else:
                return torch.device('cpu')
else:
            return torch.device(self.device)

def fit(self, X, y, sample_weight=None):
        """Fit the neural network model."""
X, y = check_X_y(X, y, multi_output=False)
self.classes_ = unique_labels(y)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Create dataset and dataloader
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(
dataset, batch_size=self.batch_size, shuffle=True
)

# Initialize model
device = self._get_device()
self.model = self.model_class(**self.model_params).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

# Early stopping
best_loss = float('inf')
patience_counter = 0

# Training loop
self.model.train()
for epoch in range(self.epochs):
            epoch_loss = 0.0
for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

optimizer.zero_grad()
outputs = self.model(batch_X)
loss = criterion(outputs, batch_y)
loss.backward()
optimizer.step()

epoch_loss += loss.item()

# Early stopping
avg_loss = epoch_loss / len(dataloader)
if avg_loss < best_loss:
                best_loss = avg_loss
patience_counter = 0
else:
                patience_counter += 1

if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
break

return self

def predict(self, X):
        """Predict class labels."""
check_is_fitted(self, ['model', 'classes_'])
X = check_array(X)

# Convert to PyTorch tensor
X_tensor = torch.FloatTensor(X)

# Predict
device = self._get_device()
self.model.eval()
with torch.no_grad():
            X_tensor = X_tensor.to(device)
outputs = self.model(X_tensor)
_, predicted = torch.max(outputs, 1)

return predicted.cpu().numpy()

def predict_proba(self, X):
        """Predict class probabilities."""
check_is_fitted(self, ['model', 'classes_'])
X = check_array(X)

# Convert to PyTorch tensor
X_tensor = torch.FloatTensor(X)

# Predict probabilities
device = self._get_device()
self.model.eval()
with torch.no_grad():
            X_tensor = X_tensor.to(device)
outputs = self.model(X_tensor)
probabilities = F.softmax(outputs, dim=1)

return probabilities.cpu().numpy()


def create_neural_model(
model_type: str,
input_size: int,
num_classes: int = 2,
**kwargs
) -> NeuralNetworkWrapper:
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
        model_params = {
'input_size': input_size,
'num_channels': kwargs.get('num_channels', [64, 128, 256]),
'kernel_size': kwargs.get('kernel_size', 2),
'dropout': kwargs.get('dropout', 0.2),
'num_classes': num_classes
}
return NeuralNetworkWrapper(TemporalConvNet, model_params, **kwargs)

elif model_type.lower() == 'cnn':
        model_params = {
'input_size': input_size,
'num_filters': kwargs.get('num_filters', [64, 128, 256]),
'kernel_sizes': kwargs.get('kernel_sizes', [3, 3, 3]),
'dropout': kwargs.get('dropout', 0.2),
'num_classes': num_classes
}
return NeuralNetworkWrapper(CNN1D, model_params, **kwargs)

elif model_type.lower() == 'transformer':
        model_params = {
'input_size': input_size,
'd_model': kwargs.get('d_model', 128),
'nhead': kwargs.get('nhead', 8),
'num_layers': kwargs.get('num_layers', 4),
'dropout': kwargs.get('dropout', 0.1),
'num_classes': num_classes
}
return NeuralNetworkWrapper(TransformerClassifier, model_params, **kwargs)

elif model_type.lower() == 'lstm':
        model_params = {
'input_size': input_size,
'hidden_size': kwargs.get('hidden_size', 128),
'num_layers': kwargs.get('num_layers', 2),
'dropout': kwargs.get('dropout', 0.2),
'bidirectional': kwargs.get('bidirectional', True),
'num_classes': num_classes
}
return NeuralNetworkWrapper(LSTMClassifier, model_params, **kwargs)

elif model_type.lower() == 'gru':
        model_params = {
'input_size': input_size,
'hidden_size': kwargs.get('hidden_size', 128),
'num_layers': kwargs.get('num_layers', 2),
'dropout': kwargs.get('dropout', 0.2),
'bidirectional': kwargs.get('bidirectional', True),
'num_classes': num_classes
}
return NeuralNetworkWrapper(GRUClassifier, model_params, **kwargs)

else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Model configuration presets
NEURAL_MODEL_CONFIGS = {
'tcn': {
'num_channels': [64, 128, 256],
'kernel_size': 2,
'dropout': 0.2,
'batch_size': 32,
'epochs': 100,
'learning_rate': 0.001
},
'cnn': {
'num_filters': [64, 128, 256],
'kernel_sizes': [3, 3, 3],
'dropout': 0.2,
'batch_size': 32,
'epochs': 100,
'learning_rate': 0.001
},
'transformer': {
'd_model': 128,
'nhead': 8,
'num_layers': 4,
'dropout': 0.1,
'batch_size': 32,
'epochs': 100,
'learning_rate': 0.001
},
'lstm': {
'hidden_size': 128,
'num_layers': 2,
'bidirectional': True,
'dropout': 0.2,
'batch_size': 32,
'epochs': 100,
'learning_rate': 0.001
},
'gru': {
'hidden_size': 128,
'num_layers': 2,
'bidirectional': True,
'dropout': 0.2,
'batch_size': 32,
'epochs': 100,
'learning_rate': 0.001
}
}