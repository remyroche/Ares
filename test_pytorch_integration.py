#!/usr/bin/env python3
"""
PyTorch Integration Test Script

This script demonstrates how to use PyTorch with the existing Ares trading bot codebase.
It shows various ways to integrate PyTorch models and functionality.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add the src directory to the path so we can import the existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_pytorch():
    """Test basic PyTorch functionality."""
    print("=== Basic PyTorch Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Create a simple tensor
    x = torch.randn(3, 4)
    print(f"Random tensor shape: {x.shape}")
    print(f"Tensor:\n{x}")

    # Basic operations
    y = x * 2 + 1
    print(f"After operations:\n{y}")

    return True

def test_existing_models():
    """Test the existing PyTorch models from the codebase."""
    print("\n=== Testing Existing Models ===")

    try:
        # Import the existing models
            CNNModel, TCNModel, TransformerModel, TemporalBlock
        )

        # Test CNN Model
        print("Testing CNN Model...")
        input_channels = 10
        sequence_length = 100
        num_classes = 3

        cnn_model = CNNModel(input_channels, sequence_length, num_classes)
        x = torch.randn(32, input_channels, sequence_length)  # batch_size=32
        output = cnn_model(x)
        print(f"CNN input shape: {x.shape}")
        print(f"CNN output shape: {output.shape}")

        # Test TCN Model
        print("\nTesting TCN Model...")
        input_size = 10
        num_channels = [64, 128, 256]
        kernel_size = 3
        num_classes = 3

        tcn_model = TCNModel(input_size, num_channels, kernel_size, num_classes)
        x = torch.randn(32, sequence_length, input_size)  # batch_size=32
        output = tcn_model(x)
        print(f"TCN input shape: {x.shape}")
        print(f"TCN output shape: {output.shape}")

        # Test Transformer Model
        print("\nTesting Transformer Model...")
        input_size = 10
        d_model = 128
        nhead = 4
        num_layers = 2
        num_classes = 3

        transformer_model = TransformerModel(
            input_size, d_model, nhead, num_layers, num_classes
        )
        x = torch.randn(32, sequence_length, input_size)  # batch_size=32
        output = transformer_model(x)
        print(f"Transformer input shape: {x.shape}")
        print(f"Transformer output shape: {output.shape}")

        return True

    except ImportError as e:
        print(f"Could not import existing models: {e}")
        return False
    except Exception as e:
        print(f"Error testing existing models: {e}")
        return False

def create_simple_trading_model():
    """Create a simple PyTorch model for trading prediction."""
    print("\n=== Creating Simple Trading Model ===")

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

    # Create model
    input_size = 20  # Number of features
    hidden_size = 64
    num_classes = 3  # Buy, Hold, Sell

    model = SimpleTradingModel(input_size, hidden_size, num_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test the model
    x = torch.randn(10, input_size)  # 10 samples, 20 features
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output probabilities: {torch.softmax(output, dim=1)}")

    return model

def train_simple_model():
    """Train a simple PyTorch model with synthetic data."""
    print("\n=== Training Simple Model ===")

    # Generate synthetic trading data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create synthetic features (price movements, indicators, etc.)
    X = np.random.randn(n_samples, n_features)

    # Create synthetic labels (0: Hold, 1: Buy, 2: Sell)
    # Simple rule: if sum of first 5 features > 0, buy; if < -0.5, sell; else hold
    feature_sum = X[:, :5].sum(axis=1)
    y = np.zeros(n_samples, dtype=int)
    y[feature_sum > 0] = 1  # Buy
    y[feature_sum < -0.5] = 2  # Sell

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = create_simple_trading_model()

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")

    return model

def test_model_inference():
    """Test model inference with new data."""
    print("\n=== Testing Model Inference ===")

    # Train a model
    model = train_simple_model()

    # Generate new test data
    test_X = torch.randn(5, 20)  # 5 new samples

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(test_X)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

    print("Test predictions:")
    for i in range(len(test_X)):
        action = ["Hold", "Buy", "Sell"][predictions[i].item()]
        prob = probabilities[i].max().item()
        print(f"Sample {i+1}: {action} (confidence: {prob:.3f})")

    return model

def demonstrate_gpu_usage():
    """Demonstrate GPU usage if available."""
    print("\n=== GPU Usage Demonstration ===")

    if torch.cuda.is_available():
        print("CUDA is available! Using GPU.")
        device = torch.device('cuda')

        # Create tensors on GPU
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)

        # Perform computation on GPU
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        z = torch.mm(x, y)
        end_time.record()

        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)

        print(f"GPU matrix multiplication took {elapsed_time:.2f} ms")
        print(f"Result shape: {z.shape}")

    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')

        # Create tensors on CPU
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)

        # Perform computation on CPU
        import time
        start_time = time.time()
        z = torch.mm(x, y)
        end_time = time.time()

        print(f"CPU matrix multiplication took {(end_time - start_time) * 1000:.2f} ms")
        print(f"Result shape: {z.shape}")

def create_advanced_trading_model():
    """Create a more advanced trading model with LSTM."""
    print("\n=== Creating Advanced LSTM Trading Model ===")

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

    # Create model
    input_size = 10  # Number of features per timestep
    hidden_size = 64
    num_layers = 2
    num_classes = 3
    sequence_length = 20

    model = LSTMTradingModel(input_size, hidden_size, num_layers, num_classes)
    print(f"LSTM Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test the model
    x = torch.randn(32, sequence_length, input_size)  # batch_size=32, sequence_length=20, features=10
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    return model

def main():
    """Main function to run all tests."""
    print("PyTorch Integration Test for Ares Trading Bot")
    print("=" * 50)

    # Run all tests
    tests = [
        test_basic_pytorch,
        test_existing_models,
        create_simple_trading_model,
        train_simple_model,
        test_model_inference,
        demonstrate_gpu_usage,
        create_advanced_trading_model
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, "PASSED" if result else "FAILED"))
        except Exception as e:
            print(f"Error in {test.__name__}: {e}")
            results.append((test.__name__, "ERROR"))

    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    for test_name, status in results:
        print(f"{test_name}: {status}")

    print("\nPyTorch is now ready to use with your Ares trading bot!")
    print("You can:")
    print("1. Use the existing CNN, TCN, and Transformer models")
    print("2. Create custom PyTorch models for trading")
    print("3. Train models with your trading data")
    print("4. Use GPU acceleration if available")

if __name__ == "__main__":
    main()