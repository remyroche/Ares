#!/usr/bin/env python3
"""
Test script for enhanced MarkovRegression adapter integrated with existing pipeline.
Uses real ETHUSDT data and pipeline-generated features.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_markov_regression_with_pipeline():
    """Test MarkovRegression with pipeline-generated features from real ETHUSDT data."""
    print("🧪 Testing Enhanced MarkovRegression with pipeline integration...")
    
    try:
        # Import base step and artifact manager
        from src.training.steps.base_step import BaseStep
        from src.utils.artifact_manager import ArtifactManager
        
        # Import feature generation step
        from src.training.steps.pre_training.feature_generation_feature_generation_step import (
            FeatureGenerationFeatureGenerationStep
        )
        
        # Import enhanced MarkovRegression adapter
        from src.training.steps.market_analysis.statsmodel_clustering.core.markov_regression_adapter import (
            MarkovRegressionAdapter, 
            MarkovRegressionConfig
        )
        
        # Initialize artifact manager
        artifact_manager = ArtifactManager()
        
        # Create a simple configuration for feature generation
        feature_config = {
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "lookback_periods": [5, 10, 20],
            "feature_categories": ["returns", "momentum", "volatility", "volume"],
            "enable_optimization": False,  # Disable for faster testing
            "max_features": 50  # Limit features for faster testing
        }
        
        # Initialize feature generation step
        feature_step = FeatureGenerationFeatureGenerationStep(
            config=feature_config,
            artifact_manager=artifact_manager
        )
        
        print("🔄 Generating features from real ETHUSDT data...")
        
        # Execute feature generation step
        feature_result = feature_step.execute()
        
        if not feature_result.success:
            print(f"❌ Feature generation failed: {feature_result.error_message}")
            return False
        
        # Get generated features
        features = feature_result.data
        if features is None or features.empty:
            print("❌ No features generated")
            return False
        
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
        print("🔄 Fitting MarkovRegression model to pipeline-generated features...")
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
                'timestamp': features.index if hasattr(features, 'index') else range(len(result.cluster_labels))
            })
            
            if result.cluster_probabilities is not None:
                for i in range(result.n_regimes):
                    output_data[f'regime_{i}_probability'] = result.cluster_probabilities[:, i]
            
            # Save to file
            output_path = "markov_regression_pipeline_results.csv"
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
                plot_path = "markov_regression_pipeline_results.png"
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
    print("🚀 Starting MarkovRegression adapter test with pipeline integration...")
    
    # Test with pipeline-generated features
    success = test_markov_regression_with_pipeline()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"📊 Pipeline integration test: {'✅ PASSED' if success else '❌ FAILED'}")
    
    if success:
        print("\n🎉 Test passed! The enhanced MarkovRegression adapter works correctly with pipeline-generated features.")
        sys.exit(0)
    else:
        print("\n❌ Test failed. Please check implementation.")
        sys.exit(1)