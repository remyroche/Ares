"""
Example: GMM Integration with Layer 2.5 Chaser

This script demonstrates how to use the GMM-enhanced Layer 2.5 Chaser
with State, Shock, and Cluster features.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import the enhanced Layer 2.5 Chaser
from src.training.steps.labeling.layer2_5_chaser import Layer25Chaser

def create_sample_data(n_samples=1000, n_features=20):
    """Create sample data for demonstration."""
    # Generate synthetic features
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame with meaningful column names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add price-related features for Layer 2.5 processing
    np.random.seed(42)
    price_base = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
    X_df['close'] = price_base
    X_df['volume'] = np.random.exponential(1000, n_samples)
    
    # Create returns series
    returns = pd.Series(np.diff(np.log(price_base)), index=X_df.index[1:])
    
    # Align returns with X (shift by 1)
    y_series = pd.Series(y, index=X_df.index)
    
    return X_df, y_series, returns

def main():
    """Main demonstration function."""
    print("🚀 GMM-Enhanced Layer 2.5 Chaser Example")
    print("=" * 50)
    
    # 1. Create sample data
    print("📊 Creating sample data...")
    X, y, returns = create_sample_data(n_samples=1000, n_features=20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Split returns accordingly
    returns_train = returns.loc[X_train.index]
    returns_test = returns.loc[X_test.index]
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    
    # 2. Create GMM-enhanced Layer 2.5 Chaser
    print("\n🧠 Creating GMM-Enhanced Layer 2.5 Chaser...")
    chaser = Layer25Chaser(
        mode="regression",
        regime_split=True,
        enable_gmm_enhancement=True,  # Enable GMM enhancement
        gmm_n_components=6,          # Number of GMM components
        gmm_cache_models=True,      # Cache GMM models
        verbose=True
    )
    
    # 3. Train the chaser with GMM enhancement
    print("\n🎓 Training GMM-Enhanced Chaser...")
    chaser.fit(
        X_train, 
        y_train, 
        returns=returns_train  # Optional returns for GMM anchoring
    )
    
    # 4. Make predictions
    print("\n🔮 Making predictions...")
    predictions = chaser.predict(
        X_test, 
        returns=returns_test  # Optional returns for GMM anchoring
    )
    
    # 5. Evaluate performance
    print("\n📈 Performance Metrics:")
    mse = np.mean((predictions - y_test.values) ** 2)
    mae = np.mean(np.abs(predictions - y_test.values))
    
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    
    # 6. Access GMM processor for analysis
    if chaser._gmm_processor:
        print("\n🔍 GMM Analysis:")
        
        # Get regime probabilities for test set
        regime_probs = chaser._gmm_processor.get_regime_probabilities(X_test)
        if not regime_probs.empty:
            print(f"   Regime probabilities shape: {regime_probs.shape}")
            print(f"   Number of regimes: {regime_probs.shape[1]}")
            
            # Show dominant regimes
            dominant_regimes = regime_probs.idxmax(axis=1)
            regime_counts = dominant_regimes.value_counts()
            print(f"   Regime distribution:")
            for regime, count in regime_counts.items():
                print(f"     {regime}: {count} samples ({count/len(regime_probs)*100:.1f}%)")
        
        # Show GMM feature statistics
        if hasattr(chaser._gmm_processor, '_processed_features_cache'):
            cache_key = "layer25_enhanced"
            if cache_key in chaser._gmm_processor._processed_features_cache:
                _, metadata = chaser._gmm_processor._processed_features_cache[cache_key]
                print(f"\n📊 GMM Feature Statistics:")
                print(f"   Original features: {metadata['original_features']}")
                print(f"   GMM state features: {metadata['gmm_state_features']}")
                print(f"   GMM shock features: {metadata['gmm_shock_features']}")
                print(f"   GMM cluster features: {metadata['gmm_cluster_features']}")
                print(f"   Total enhanced features: {metadata['final_features']}")
                print(f"   Processing time: {metadata['processing_time_seconds']:.2f}s")
    
    print("\n✅ Example completed successfully!")
    
    # 7. Comparison with non-GMM chaser
    print("\n🔄 Comparing with standard Layer 2.5 Chaser...")
    standard_chaser = Layer25Chaser(
        mode="regression",
        regime_split=True,
        enable_gmm_enhancement=False,  # Disable GMM
        verbose=False
    )
    
    standard_chaser.fit(X_train, y_train)
    standard_predictions = standard_chaser.predict(X_test)
    
    standard_mse = np.mean((standard_predictions - y_test.values) ** 2)
    standard_mae = np.mean(np.abs(standard_predictions - y_test.values))
    
    print(f"   Standard Chaser MSE: {standard_mse:.4f}")
    print(f"   GMM-Enhanced MSE: {mse:.4f}")
    print(f"   Improvement: {(standard_mse - mse)/standard_mse*100:.2f}%")

if __name__ == "__main__":
    main()
