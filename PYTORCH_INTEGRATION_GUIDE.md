# PyTorch Integration Guide for Ares Trading Bot

## Overview

This guide explains how to use PyTorch with the Ares trading bot codebase. PyTorch is already integrated into the project and provides powerful deep learning capabilities for trading predictions.

## Current PyTorch Setup

### ✅ What's Already Available

1. **PyTorch Installation**: PyTorch 2.8.0 is installed in the virtual environment
2. **Existing Models**: The codebase already includes several PyTorch models:
   - **CNN Model** (`CNNModel`): For 1-minute timeframe predictions
   - **TCN Model** (`TCNModel`): Temporal Convolutional Network for 5-minute timeframe
   - **Transformer Model** (`TransformerModel`): For 15-minute timeframe predictions
3. **Training Infrastructure**: Complete training pipelines with PyTorch Lightning
4. **GPU Support**: Automatic GPU detection and usage when available

### 📁 Key PyTorch Files

- `src/training/steps/step6_hmm_based_training.py` - Main PyTorch models
- `src/training/optimization/advanced_surrogate_models.py` - Deep learning surrogate models
- `src/training/multi_output_model_trainer.py` - Multi-output PyTorch training
- `src/transition/seq2seq_trainer.py` - Sequence-to-sequence models with PyTorch Lightning

## Getting Started

### 1. Activate the Virtual Environment

```bash
source venv/bin/activate
```

### 2. Verify PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 3. Run the Integration Test

```bash
python test_pytorch_integration.py
```

## Using Existing PyTorch Models

### CNN Model (1-minute timeframe)

```python
from src.training.steps.step6_hmm_based_training import CNNModel

# Create CNN model
input_channels = 10  # Number of input features
sequence_length = 100  # Time series length
num_classes = 3  # Buy, Hold, Sell

model = CNNModel(input_channels, sequence_length, num_classes)

# Prepare input data (batch_size, channels, sequence_length)
x = torch.randn(32, input_channels, sequence_length)
output = model(x)
```

### TCN Model (5-minute timeframe)

```python
from src.training.steps.step6_hmm_based_training import TCNModel

# Create TCN model
input_size = 10
num_channels = [64, 128, 256]
kernel_size = 3
num_classes = 3

model = TCNModel(input_size, num_channels, kernel_size, num_classes)

# Prepare input data (batch_size, sequence_length, features)
x = torch.randn(32, 100, input_size)
output = model(x)
```

### Transformer Model (15-minute timeframe)

```python
from src.training.steps.step6_hmm_based_training import TransformerModel

# Create Transformer model
input_size = 10
d_model = 128
nhead = 4
num_layers = 2
num_classes = 3

model = TransformerModel(input_size, d_model, nhead, num_layers, num_classes)

# Prepare input data (batch_size, sequence_length, features)
x = torch.randn(32, 100, input_size)
output = model(x)
```

## Creating Custom PyTorch Models

### Simple Feedforward Model

```python
import torch.nn as nn

class SimpleTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleTradingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Usage
model = SimpleTradingModel(input_size=20, hidden_size=64, num_classes=3)
```

### LSTM Model for Time Series

```python
class LSTMTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMTradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout and final classification
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

# Usage
model = LSTMTradingModel(input_size=10, hidden_size=64, num_layers=2, num_classes=3)
```

## Training PyTorch Models

### Basic Training Loop

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prepare data
X = torch.FloatTensor(your_features)
y = torch.LongTensor(your_labels)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Setup training
model = YourModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Using PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class TradingLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Training
trainer = pl.Trainer(max_epochs=100, callbacks=[ModelCheckpoint()])
trainer.fit(model, train_dataloader)
```

## Integration with Existing Pipeline

### Using PyTorch Models in Training Steps

```python
from src.training.steps.step6_hmm_based_training import HMMBasedTrainingStep

# The existing training step already supports PyTorch models
config = {
    "HMM_LM": {
        "specialist_models": {
            "1m": {"architecture": "CNN"},
            "5m": {"architecture": "TCN"},
            "15m": {"architecture": "Transformer"}
        }
    }
}

trainer = HMMBasedTrainingStep(config)
# The trainer will automatically use PyTorch models based on the config
```

### Multi-Output Training

```python
from src.training.multi_output_model_trainer import MultiOutputModelTrainer

# Configure for PyTorch models
config = MultiOutputModelConfig(
    model_type="PyTorch",
    direction_target="direction",
    profit_target="expected_profit",
    use_profit_features=True
)

trainer = MultiOutputModelTrainer(config)
```

## GPU Acceleration

### Automatic GPU Detection

The codebase automatically detects and uses GPU when available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Manual GPU Usage

```python
# Move model to GPU
model = model.cuda()

# Move data to GPU
x = x.cuda()
y = y.cuda()

# Or use device variable
device = torch.device('cuda')
model = model.to(device)
x = x.to(device)
y = y.to(device)
```

## Best Practices

### 1. Model Architecture Selection

- **CNN**: Best for short-term patterns and local features
- **TCN**: Good for medium-term temporal dependencies
- **Transformer**: Excellent for long-term relationships and attention mechanisms
- **LSTM**: Good for sequential data with variable length

### 2. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)
```

### 3. Model Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

### 4. Evaluation

```python
model.eval()
with torch.no_grad():
    outputs = model(test_X)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the virtual environment is activated
2. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
3. **Model Not Learning**: Check learning rate and data preprocessing
4. **Overfitting**: Add dropout, regularization, or early stopping

### Performance Optimization

1. **Use Mixed Precision**: Enable automatic mixed precision for faster training
2. **DataLoader Optimization**: Use `num_workers` for parallel data loading
3. **Model Optimization**: Use model pruning and quantization for inference

## Next Steps

1. **Experiment with Model Architectures**: Try different combinations of layers
2. **Hyperparameter Tuning**: Use Optuna or Ray Tune for optimization
3. **Ensemble Methods**: Combine multiple PyTorch models
4. **Real-time Inference**: Deploy models for live trading

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/)
- [Deep Learning for Trading](https://github.com/Ares-Trading-Bot/docs)

---

This guide provides a foundation for using PyTorch with the Ares trading bot. The existing codebase already includes sophisticated PyTorch implementations, and you can extend them based on your specific trading strategies.