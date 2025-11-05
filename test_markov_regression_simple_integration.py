#!/usr/bin/env python3
"""
Test script for enhanced MarkovRegression adapter with simple data integration.
Uses real ETHUSDT data and basic feature generation.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_markov_regression_with_simple_data():
    """Test MarkovRegression with simple features from real ETHUSDT data."""
    print("🧪 Testing Enhanced MarkovRegression with simple data integration...")
    
    try:
        # Import data downloader from statsmodel_clustering
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            download_clustering_data
        )
        
        # Import enhanced MarkovRegression adapter
        from src.training.steps.market_analysis.statsmodel_clustering.core.markov_regression_adapter import (
            MarkovRegressionAdapter, 
            MarkovRegressionConfig
        )
        
        print("🔄 Downloading real ETHUSDT data...")
        
        # Download real ETHUSDT data
        data = download_clustering_data(
            symbol="ETHUSDT",
            timeframe="1h",
            limit=1000  # Limit for faster testing
        )
        
        if data is None or data.empty:
            print("❌ No data downloaded")
            return False
        
        print(f"✅ Downloaded {len(data)} rows of data")
        
        # Generate simple features
        print("🔄 Generating simple features...")
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        
        # Calculate simple moving averages
        data['sma_5'] = data['close'].rolling(window=5).mean()
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_20'] = data['close'].rolling(window=20).mean()
        
        # Calculate volatility
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_10'] = data['returns'].rolling(window=10).std()
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NaN values
        data = data.dropna()
        
        # Select features for modeling
        feature_columns = ['returns', 'sma_5', 'sma_10', 'sma_20', 'volatility_5', 'volatility_10', 'rsi']
        features = data[feature_columns]
        
        print(f"✅ Generated {len(features)} rows with {len(features.columns)} features")
        
        # Prepare data for MarkovRegression
        # Convert to numpy array
        feature_data = features.values
        
        # Handle any NaN or infinite values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data
        feature_data = (feature_data - np.mean(feature_data, axis=0)) / (np.std(feature_data, axis=0) + 1e-8)
        
        print(f"📊 Prepared data shape: {feature_data.shape}")
        
        # Create configuration for MarkovRegression
        config = MarkovRegressionConfig(
            k_regimes=3,
            trend='c',
            order=0,
            switching_variance=True,
            switching_trend=True,
            maxiter=50,  # Reduced for faster testing
            enable_diagnostics=True,  # Enable diagnostics
            enable_hardware_optimization=False,  # Disabled for simpler testing
            enable_pca=True,  # Enable PCA for high-dimensional data
            pca_components=min(10, feature_data.shape[1]),
            enable_scaling=False  # Already normalized
        )
        
        # Create adapter
        adapter = MarkovRegressionAdapter(config)
        
        # Fit model
        print("🔄 Fitting MarkovRegression model to features...")
        result = adapter.fit(feature_data)
        
        # Check results
        if result.success:
            print("✅ Model fitting successful!")
            print(f"📊 Detected regimes: {result.n_regimes}")
            print(f"📈 Predicted regime distribution: {np.bincount(result.cluster_labels)}")
            print(f"📊 Log likelihood: {result.log_likelihood:.4f}")
            print(f"📊 AIC: {result.aic:.4f}")
            print(f"📊 BIC: {result.bic:.4f}")
            
            # Get transition matrix
            if result.transition_matrix is not None:
                print("📊 Transition Matrix:")
                print(result.transition_matrix)
            
            # Analyze regime characteristics
            if result.diagnostics and 'regime_characteristics' in result.diagnostics:
                print("📊 Regime Characteristics:")
                for regime, characteristics in result.diagnostics['regime_characteristics'].items():
                    print(f"  {regime}:")
                    print(f"    Size: {characteristics.get('size', 'N/A')}")
                    print(f"    Proportion: {characteristics.get('proportion', 0):.2%}")
                    if 'mean' in characteristics and len(characteristics['mean']) > 0:
                        print(f"    Mean return: {characteristics['mean'][0]:.6f}")
                        print(f"    Volatility: {characteristics['std'][0]:.6f}")
            
            # Save results
            output_data = pd.DataFrame({
                'regime_label': result.cluster_labels,
                'timestamp': data.index if hasattr(data, 'index') else range(len(result.cluster_labels))
            })
            
            if result.cluster_probabilities is not None:
                for i in range(result.n_regimes):
                    output_data[f'regime_{i}_probability'] = result.cluster_probabilities[:, i]
            
            # Save to file
            output_path = "markov_regression_simple_results.csv"
            output_data.to_csv(output_path, index=False)
            print(f"📊 Saved results to {output_path}")
            
            # Visualize if matplotlib available
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(12, 8))
                
                # Plot regime labels over time
                plt.subplot(2, 1, 1)
                colors = ['red', 'green', 'blue']
                for i in range(result.n_regimes):
                    mask = result.cluster_labels == i
                    indices = np.where(mask)[0]
                    for j in range(len(indices)):
                        if j == 0:
                            start = 0
                        else:
                            start = indices[j-1] + 1
                        
                        if j < len(indices) - 1:
                            end = indices[j] + 1
                        else:
                            end = len(result.cluster_labels) - 1
                        
                        plt.axvspan(start, end, alpha=0.2, color=colors[i])
                
                plt.title('Regime Classification Over Time')
                plt.ylabel('Regime')
                plt.yticks([0, 1, 2], ['Regime 0', 'Regime 1', 'Regime 2'])
                
                # Plot regime probabilities
                plt.subplot(2, 1, 2)
                if result.cluster_probabilities is not None:
                    for i in range(result.n_regimes):
                        plt.plot(result.cluster_probabilities[:, i], 
                                label=f'Regime {i} Probability', 
                                color=colors[i])
                
                plt.title('Regime Probabilities')
                plt.xlabel('Time')
                plt.ylabel('Probability')
                plt.legend()
                
                plt.tight_layout()
                
                # Save plot
                plot_path = "markov_regression_simple_results.png"
                plt.savefig(plot_path)
                print(f"📊 Saved visualization to {plot_path}")
                
            except ImportError:
                print("⚠️ Matplotlib not available, skipping visualization")
            
            return True
        else:
            print(f"❌ Model fitting failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting MarkovRegression adapter test with simple data integration...")
    
    # Test with simple data
    success = test_markov_regression_with_simple_data()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"📊 Simple data integration test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Test passed! The enhanced MarkovRegression adapter works correctly with real data.")
        sys.exit(0)
    else:
        print("\n❌ Test failed. Please check implementation.")
        sys.exit(1)