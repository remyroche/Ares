#!/usr/bin/env python3
"""
Practical PyTorch Trading Example

This script demonstrates how to use PyTorch for trading predictions with real data.
It shows data preparation, model training, and prediction workflows.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TradingLSTMModel(nn.Module):
    """LSTM model for trading predictions."""

    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(TradingLSTMModel, self).__init__()
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

class TradingTransformerModel(nn.Module):
    """Transformer model for trading predictions."""

    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super(TradingTransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)

        # Take the last output
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

def generate_synthetic_trading_data(n_samples=10000, sequence_length=50):
    """Generate synthetic trading data for demonstration."""
    print("Generating synthetic trading data...")

    # Generate price data with trends and noise
    np.random.seed(42)

    # Base price trend
    base_price = 100
    trend = np.cumsum(np.random.randn(n_samples + sequence_length) * 0.01)
    prices = base_price + trend

    # Create features
    features = []
    labels = []

    for i in range(sequence_length, n_samples + sequence_length):
        # Price-based features
        price_window = prices[i-sequence_length:i]
        returns = np.diff(price_window) / price_window[:-1]

        # Technical indicators
        sma_5 = np.mean(price_window[-5:])
        sma_20 = np.mean(price_window[-20:])
        rsi = calculate_rsi(returns)
        volatility = np.std(returns)

        # Volume (synthetic)
        volume = np.random.lognormal(10, 1, sequence_length)

        # Combine features
        feature_vector = np.concatenate([
            returns,  # Price returns
            [sma_5 / price_window[-1] - 1],  # SMA ratio
            [sma_20 / price_window[-1] - 1],  # SMA ratio
            [rsi],  # RSI
            [volatility],  # Volatility
            volume / np.mean(volume) - 1,  # Volume ratio
        ])

        features.append(feature_vector)

        # Create labels (0: Hold, 1: Buy, 2: Sell)
        future_return = (prices[i+5] - prices[i]) / prices[i]  # 5-step ahead return

        if future_return > 0.02:  # 2% gain
            label = 1  # Buy
        elif future_return < -0.02:  # 2% loss
            label = 2  # Sell
        else:
            label = 0  # Hold

        labels.append(label)

    return np.array(features), np.array(labels)

def calculate_rsi(returns, period=14):
    """Calculate RSI indicator."""
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data_for_lstm(features, labels, sequence_length=20):
    """Prepare data for LSTM model."""
    print("Preparing data for LSTM...")

    # Reshape features for LSTM (samples, sequence_length, features_per_step)
    n_samples = len(features) - sequence_length + 1
    n_features_per_step = features.shape[1] // sequence_length

    X_lstm = []
    y_lstm = []

    for i in range(n_samples):
        # Take sequence_length consecutive feature vectors
        sequence = features[i:i+sequence_length]

        # Reshape to (sequence_length, features_per_step)
        sequence_reshaped = sequence.reshape(sequence_length, n_features_per_step)
        X_lstm.append(sequence_reshaped)
        y_lstm.append(labels[i+sequence_length-1])

    return np.array(X_lstm), np.array(y_lstm)

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the PyTorch model."""
    print(f"Training {model.__class__.__name__}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss_avg)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    """Plot training results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title(f'{model_name} - Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader):
    """Evaluate the trained model."""
    print("Evaluating model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions,
                              target_names=['Hold', 'Buy', 'Sell']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    return accuracy, all_probabilities, all_predictions, all_labels

def main():
    """Main function to run the trading example."""
    print("PyTorch Trading Example")
    print("=" * 50)

    # Generate data
    features, labels = generate_synthetic_trading_data(n_samples=5000, sequence_length=50)

    # Prepare data for LSTM
    X_lstm, y_lstm = prepare_data_for_lstm(features, labels, sequence_length=20)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_lstm, y_lstm, test_size=0.3, random_state=42, stratify=y_lstm
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train LSTM model
    input_size = X_train.shape[-1]
    hidden_size = 64
    num_layers = 2
    num_classes = 3

    lstm_model = TradingLSTMModel(input_size, hidden_size, num_layers, num_classes)

    lstm_results = train_model(lstm_model, train_loader, val_loader, num_epochs=30)
    plot_training_results(*lstm_results, "LSTM Trading Model")

    # Evaluate LSTM model
    lstm_accuracy, lstm_probs, lstm_preds, lstm_labels = evaluate_model(lstm_model, test_loader)

    # Train Transformer model
    d_model = 128
    nhead = 4
    num_layers = 2

    transformer_model = TradingTransformerModel(
        input_size, d_model, nhead, num_layers, num_classes
    )

    transformer_results = train_model(transformer_model, train_loader, val_loader, num_epochs=30)
    plot_training_results(*transformer_results, "Transformer Trading Model")

    # Evaluate Transformer model
    transformer_accuracy, transformer_probs, transformer_preds, transformer_labels = evaluate_model(transformer_model, test_loader)

    # Compare models
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")
    print(f"Transformer Model Accuracy: {transformer_accuracy:.4f}")

    # Save models
    torch.save(lstm_model.state_dict(), 'lstm_trading_model.pth')
    torch.save(transformer_model.state_dict(), 'transformer_trading_model.pth')
    print("\nModels saved successfully!")

    print("\nExample completed successfully!")
    print("You can now use these models for real trading predictions.")

if __name__ == "__main__":
    main()