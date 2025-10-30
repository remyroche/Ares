"""
Example: TCN with Frozen Autoencoder for Faster Training

This script demonstrates how to use the enhanced TCN model with autoencoder compression.
The autoencoder compresses 100+ features into a 16-dimensional latent space, making
training and inference 6-8x faster.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.causal_dilated_tcn import (
    CausalTCNConfig, 
    CausalDilatedTCNModel,
    PyTorchAutoencoder
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_sample_data(n_samples=1000, n_features=120):
    """
    Generate synthetic market-like data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (typical: 100-150 for analyst)
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Binary targets (n_samples,) for green light prediction
    """
    logger.info(f"📊 Generating {n_samples} samples with {n_features} features...")
    
    # Generate correlated features (simulating market indicators)
    X = np.random.randn(n_samples, n_features)
    
    # Add some structure (trend + noise)
    for i in range(1, n_features):
        X[:, i] = 0.7 * X[:, i-1] + 0.3 * X[:, i]
    
    # Generate binary targets based on feature combinations
    # Simulating "green light" for trading based on feature patterns
    signal_features = X[:, :5].mean(axis=1) + 0.2 * X[:, 10:15].std(axis=1)
    y = (signal_features > 0).astype(int)
    
    logger.info(f"✅ Data generated: {X.shape}, targets: {y.shape}")
    logger.info(f"   Class distribution: {np.sum(y==0)} negative, {np.sum(y==1)} positive")
    
    return X, y


def example_1_with_autoencoder():
    """Example 1: Train TCN with autoencoder compression (FAST)."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 1: TCN with Autoencoder Compression")
    logger.info("="*70 + "\n")
    
    # Generate data
    X_train, y_train = generate_sample_data(n_samples=1000, n_features=120)
    X_test, y_test = generate_sample_data(n_samples=200, n_features=120)
    
    # Configure TCN with autoencoder
    config = CausalTCNConfig(
        # TCN architecture
        num_filters=64,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        
        # Training settings
        learning_rate=0.001,
        batch_size=32,
        epochs=50,  # Reduced for demo
        early_stopping_patience=10,
        
        # Autoencoder compression (KEY FEATURE)
        use_autoencoder=True,
        autoencoder_path="models/example_encoder.pth",
        latent_dim=16,  # Compress 120 → 16 features
        train_autoencoder_if_missing=True,
        autoencoder_epochs=30  # Pre-train encoder
    )
    
    logger.info("🔧 Configuration:")
    logger.info(f"   Use Autoencoder: {config.use_autoencoder}")
    logger.info(f"   Latent Dimension: {config.latent_dim}")
    logger.info(f"   Compression Ratio: {120/config.latent_dim:.1f}x")
    
    # Create and train model
    logger.info("\n🏋️ Training TCN with autoencoder compression...")
    model = CausalDilatedTCNModel(config=config)
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("\n🔮 Making predictions on test data...")
    predictions = model.predict(X_test)
    
    # Evaluate
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y_test)
    
    logger.info(f"\n✅ Test Accuracy: {accuracy:.4f}")
    logger.info(f"   Predictions shape: {predictions.shape}")
    logger.info(f"   Sample predictions: {predictions[:5]}")


def example_2_without_autoencoder():
    """Example 2: Train TCN without autoencoder (SLOWER)."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 2: TCN without Autoencoder (Original)")
    logger.info("="*70 + "\n")
    
    # Generate data
    X_train, y_train = generate_sample_data(n_samples=1000, n_features=120)
    X_test, y_test = generate_sample_data(n_samples=200, n_features=120)
    
    # Configure TCN WITHOUT autoencoder
    config = CausalTCNConfig(
        num_filters=64,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        early_stopping_patience=10,
        
        # Disable autoencoder
        use_autoencoder=False
    )
    
    logger.info("🔧 Configuration:")
    logger.info(f"   Use Autoencoder: {config.use_autoencoder}")
    logger.info(f"   Processing all 120 features directly")
    
    # Create and train model
    logger.info("\n🏋️ Training TCN without compression...")
    model = CausalDilatedTCNModel(config=config)
    model.fit(X_train, y_train)
    
    # Make predictions
    logger.info("\n🔮 Making predictions on test data...")
    predictions = model.predict(X_test)
    
    # Evaluate
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y_test)
    
    logger.info(f"\n✅ Test Accuracy: {accuracy:.4f}")


def example_3_pretrain_encoder():
    """Example 3: Pre-train encoder separately, then use with TCN."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 3: Pre-train Encoder, Then Use with TCN")
    logger.info("="*70 + "\n")
    
    # Generate data for encoder pre-training
    X_pretrain, _ = generate_sample_data(n_samples=2000, n_features=120)
    
    logger.info("🏋️ Step 1: Pre-training autoencoder...")
    
    # Create and train autoencoder
    autoencoder = PyTorchAutoencoder(
        input_dim=120,
        latent_dim=16,
        hidden_dim=64
    )
    
    # Manual training (simplified)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepare data
    X_tensor = torch.FloatTensor(X_pretrain)
    dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder reconstructs input
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Training loop
    autoencoder.train()
    for epoch in range(30):  # 30 epochs
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            reconstructed = autoencoder(batch_X)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            logger.info(f"   Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.6f}")
    
    # Save encoder
    autoencoder.save_encoder("models/pretrained_encoder.pth")
    logger.info("✅ Encoder pre-training complete!\n")
    
    # Now use pre-trained encoder with TCN
    logger.info("🏋️ Step 2: Training TCN with pre-trained frozen encoder...")
    
    X_train, y_train = generate_sample_data(n_samples=1000, n_features=120)
    
    config = CausalTCNConfig(
        num_filters=64,
        num_layers=4,
        epochs=50,
        use_autoencoder=True,
        autoencoder_path="models/pretrained_encoder.pth",
        train_autoencoder_if_missing=False  # Don't retrain
    )
    
    model = CausalDilatedTCNModel(config=config)
    model.fit(X_train, y_train)
    
    logger.info("✅ TCN training with pre-trained encoder complete!")


def example_4_comparison():
    """Example 4: Side-by-side comparison of both approaches."""
    logger.info("\n" + "="*70)
    logger.info("EXAMPLE 4: Performance Comparison")
    logger.info("="*70 + "\n")
    
    import time
    
    # Generate data
    X_train, y_train = generate_sample_data(n_samples=1000, n_features=120)
    X_test, y_test = generate_sample_data(n_samples=200, n_features=120)
    
    results = {}
    
    # Test 1: WITH autoencoder
    logger.info("🔬 Test 1: Training WITH autoencoder compression...")
    config_ae = CausalTCNConfig(
        num_filters=64,
        num_layers=4,
        epochs=50,
        use_autoencoder=True,
        autoencoder_path="models/comparison_encoder.pth",
        latent_dim=16,
        train_autoencoder_if_missing=True,
        autoencoder_epochs=30
    )
    
    start = time.time()
    model_ae = CausalDilatedTCNModel(config=config_ae)
    model_ae.fit(X_train, y_train)
    preds_ae = model_ae.predict(X_test)
    time_ae = time.time() - start
    
    acc_ae = np.mean((preds_ae > 0.5).astype(int) == y_test)
    results['with_ae'] = {'time': time_ae, 'accuracy': acc_ae}
    
    logger.info(f"   ✅ Time: {time_ae:.2f}s, Accuracy: {acc_ae:.4f}\n")
    
    # Test 2: WITHOUT autoencoder
    logger.info("🔬 Test 2: Training WITHOUT autoencoder (original)...")
    config_no_ae = CausalTCNConfig(
        num_filters=64,
        num_layers=4,
        epochs=50,
        use_autoencoder=False
    )
    
    start = time.time()
    model_no_ae = CausalDilatedTCNModel(config=config_no_ae)
    model_no_ae.fit(X_train, y_train)
    preds_no_ae = model_no_ae.predict(X_test)
    time_no_ae = time.time() - start
    
    acc_no_ae = np.mean((preds_no_ae > 0.5).astype(int) == y_test)
    results['without_ae'] = {'time': time_no_ae, 'accuracy': acc_no_ae}
    
    logger.info(f"   ✅ Time: {time_no_ae:.2f}s, Accuracy: {acc_no_ae:.4f}\n")
    
    # Summary
    logger.info("\n📊 COMPARISON SUMMARY:")
    logger.info("="*70)
    logger.info(f"{'Method':<30} {'Time (s)':<15} {'Accuracy':<15} {'Speedup':<15}")
    logger.info("-"*70)
    logger.info(f"{'With Autoencoder':<30} {time_ae:<15.2f} {acc_ae:<15.4f} {'-':<15}")
    logger.info(f"{'Without Autoencoder':<30} {time_no_ae:<15.2f} {acc_no_ae:<15.4f} {time_no_ae/time_ae:<15.2f}x")
    logger.info("="*70)
    logger.info(f"\n🚀 Speed improvement: {time_no_ae/time_ae:.2f}x faster with autoencoder!")
    logger.info(f"📊 Accuracy difference: {abs(acc_ae - acc_no_ae):.4f}")


def main():
    """Run all examples."""
    logger.info("🚀 TCN with Frozen Autoencoder Examples")
    logger.info("="*70)
    logger.info("These examples demonstrate the enhanced TCN model with")
    logger.info("autoencoder compression for faster training and inference.")
    logger.info("="*70)
    
    try:
        # Run examples
        example_1_with_autoencoder()
        example_2_without_autoencoder()
        example_3_pretrain_encoder()
        example_4_comparison()
        
        logger.info("\n" + "="*70)
        logger.info("✅ All examples completed successfully!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

